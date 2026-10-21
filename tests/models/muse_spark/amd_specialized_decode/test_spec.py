# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark specialized decode's shared contract on CPU: the scratch and peer
layouts (offsets, alignment, non-overlap, width independence, every pair index
inside its region), the K1 / K2 placements (Phase B's GEMV contributors and
epilogue units, Phase C's split units and combine CTAs), the tags, the build
keys, the argument lists and the error-word ABI. Pure Python: no GPU, no
FlyDSL."""

import ast
import dataclasses
import inspect
import itertools
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from vllm.models.muse_spark.amd.specialized_decode.mono import abi, spec
from vllm.models.muse_spark.amd.specialized_decode.mono.spec import K1Build, K2Build

MONO = Path(spec.__file__).resolve().parent

# Section 5.10 of the design: name -> (byte offset, bytes).
SCRATCH = {
    "xrdy": (0, 512),
    "ass": (512, 4096),
    "bss": (4608, 4096),
    "ppart": (8704, 98304),
    "route": (107008, 1024),
    "inter": (108032, 32768),
    "isc": (140800, 8192),
    "pss": (148992, 16384),
    "upart": (165376, 196608),
    "xqd": (361984, 40960),
    "xrow": (402944, 131072),
}
# DESIGN_BC 5.7: Phases B and C, laid out past xrow (every offset above stays).
SCRATCH_BC = {
    "fss": (534016, 4096),
    "nss": (538112, 4096),
    "ardy": (542208, 512),
    "qpart": (542720, 294912),
    "srdy": (837632, 2048),
    "crdy": (839680, 1024),
    "arow": (840704, 131072),
    "qkvd": (971776, 36864),
    "xo": (1008640, 16384),
    "pstat": (1025024, 16384),
    "opart": (1041408, 524288),
}
PEER = {
    "ar": (0, 2097152),
    "rlog": (2097152, 16384),
    "pool": (2113536, 131072),
    "rsx": (2244608, 1048576),
    "sqx": (3293184, 4096),
    "moe": (3297280, 131072),
    "branch": (3428352, 262144),
}


def test_layouts_match_the_design():
    assert spec.scratch_layout() == {**SCRATCH, **SCRATCH_BC}
    assert spec.scratch_bytes() == 1565696  # DESIGN_BC 5.7
    # Phase A's part: every offset of DESIGN 5.10, its end where B / C start
    assert max(o + n for o, n in SCRATCH.values()) == 534016
    pairs = sum(n for name, (_, n) in SCRATCH_BC.items() if name in spec.REGIONS)
    plain = sum(n for name, (_, n) in SCRATCH_BC.items() if name not in spec.REGIONS)
    assert (pairs, plain) == (306688, 724992)
    assert spec.peer_layout() == PEER
    assert spec.peer_half_bytes() == 3690496
    assert spec.peer_bytes() == 7380992
    assert abi.PEER_HEAD + spec.peer_bytes() == 7381248  # what the runner allocates


@pytest.mark.parametrize("layout", ["scratch", "peer"])
def test_regions_are_aligned_ordered_and_disjoint(layout):
    if layout == "scratch":
        got = spec.scratch_layout()
        order = (
            *spec.SCRATCH_REGIONS,
            spec.XQD,
            spec.XROW,
            *spec.SCRATCH_REGIONS_BC,
            *(name for name, _, _ in spec.SCRATCH_PLAIN_BC),
        )
    else:
        got = spec.peer_layout()
        order = spec.PEER_REGIONS
    assert tuple(got) == order
    end = 0
    for name in order:
        off, size = got[name]
        assert off % spec.PAIR_BYTES == 0 and size % spec.PAIR_BYTES == 0, name
        assert size > 0 and off >= end, name
        end = off + size
    assert end < 2**31  # 32-bit buffer offsets
    if layout == "scratch":
        assert got[spec.XROW][0] % spec.XROW_ALIGN == 0
        for name, nbytes, align in spec.SCRATCH_PLAIN_BC:
            assert got[name] == (got[name][0], nbytes) and got[name][0] % align == 0
            assert align == 16  # dwordx4 loads and stores


def test_layouts_do_not_depend_on_the_width():
    # No parameter at all: every width (and both parities) share one layout.
    for fn in (
        spec.scratch_layout,
        spec.scratch_bytes,
        spec.peer_layout,
        spec.peer_half_bytes,
        spec.peer_bytes,
    ):
        assert not inspect.signature(fn).parameters, fn.__name__
    # ... and it holds the widest launch's rows.
    s = spec.MAX_ROWS
    assert spec.scratch_layout()["xrow"][1] >= s * spec.HIDDEN * 2
    assert spec.scratch_layout()["xqd"][1] >= s * (spec.XQ_WORDS + spec.XSC_WORDS) * 4
    lay = spec.scratch_layout()
    assert lay["arow"][1] >= s * spec.HIDDEN * 2
    assert lay["qkvd"][1] >= s * spec.QKV_ROWS * 2
    assert lay["xo"][1] >= s * spec.CORE * 2
    # the split's partials: one record a CTA, whatever the width and kind
    assert lay["pstat"][1] >= spec.BLOCKS * spec.Q_PER_KV * 2 * 4
    assert lay["opart"][1] >= spec.BLOCKS * spec.Q_PER_KV * spec.HEAD_DIM * 4


def test_pair_layout_is_k3s():
    items = [("a", 3), ("b", 1), ("c", 5)]
    assert spec.pair_layout(items) == {"a": (0, 24), "b": (24, 8), "c": (32, 40)}
    assert spec.pair_layout(items, align=16, start=4) == {
        "a": (16, 24),
        "b": (48, 8),
        "c": (64, 40),
    }


def _plan():
    """``mono.common.plan`` (FlyDSL-free in K3), or a visible skip."""
    try:
        from vllm.models.muse_spark.amd.specialized_decode.mono.common import plan
    except ImportError as e:
        pytest.skip(f"mono.common.plan is not importable: {e}")
    return plan


def test_pair_layout_matches_plans():
    plan = _plan()
    items = [("a", 3), ("b", 1), ("c", 5)]
    for kw in ({}, {"align": 16, "start": 4}):
        assert plan.pair_layout(items, **kw) == spec.pair_layout(items, **kw)
    assert plan.pair_layout(spec.PEER_PAIRS) == spec.peer_layout()
    assert plan.PAIR_BYTES == spec.PAIR_BYTES
    for name in ("BLOCKS", "THREADS", "WAVES", "LDS_BYTES"):
        if hasattr(plan, name):
            assert getattr(plan, name) == getattr(spec, name), name
    if hasattr(plan, "HEALTH_WORDS"):
        assert plan.HEALTH_WORDS == abi.HEALTH_WORDS


def _cover(region: str, indexes) -> None:
    """``indexes`` hit every pair of ``region`` exactly once."""
    pairs = dict(spec.SCRATCH_PAIRS + spec.SCRATCH_PAIRS_BC + spec.PEER_PAIRS)[region]
    got = sorted(indexes)
    assert got == list(range(pairs)), region


MT, MP, TP = spec.MAX_TOKENS, spec.MAX_PICKS, spec.TP
P = itertools.product


def test_every_pair_index_formula_tiles_its_region():
    _cover("xrdy", (spec.xrdy_pair(j) for j in range(spec.NA)))
    for r in ("ass", "bss"):
        _cover(r, (spec.ss_pair(j, t) for j, t in P(range(spec.NA), range(MT))))
    _cover(
        "ppart",
        (
            spec.ppart_pair(g, q, t, r)
            for g, q, t, r in P(
                range(spec.P_GROUPS), range(1, spec.P_PARTS), range(MT), range(16)
            )
        ),
    )
    _cover(
        "route",
        [spec.route_id_pair(t, k) for t, k in P(range(MT), range(spec.TOPK))]
        + [spec.route_weight_pair(t, k) for t, k in P(range(MT), range(spec.TOPK))],
    )
    _cover(
        "inter",
        (
            spec.inter_pair(p, g, w)
            for p, g, w in P(range(MP), range(spec.UG_GROUPS), range(4))
        ),
    )
    _cover("isc", (spec.isc_pair(p, g) for p, g in P(range(MP), range(16))))
    _cover("pss", (spec.pss_pair(j, p) for j, p in P(range(spec.RED_TASKS), range(MP))))
    _cover(
        "upart",
        (
            spec.upart_pair(g, q, t, r)
            for g, q, t, r in P(
                range(spec.U_GROUPS), range(1, spec.U_PARTS), range(MT), range(16)
            )
        ),
    )
    even = range(0, spec.HIDDEN, 2)
    _cover("ar", (spec.ar_pair(s, t, c) for s, t, c in P(range(TP), range(MT), even)))
    _cover(
        "rlog",
        (
            spec.rlog_pair(s, t, i)
            for s, t, i in P(range(TP), range(MT), range(spec.EXPERTS_RANK))
        ),
    )
    _cover("pool", (spec.pool_pair(t, c) for t, c in P(range(MT), range(0, 4096, 2))))
    _cover(
        "rsx",
        (
            spec.rsx_pair(s, p, c)
            for s, p, c in P(range(TP), range(MP), range(0, spec.OWN_COLS, 2))
        ),
    )
    _cover("sqx", (spec.sqx_pair(s, p) for s, p in P(range(TP), range(MP))))
    _cover("moe", (spec.moe_pair(t, c) for t, c in P(range(MT), range(0, 4096, 2))))
    _cover("branch", (spec.branch_pair(t, c) for t, c in P(range(MT), even)))
    # Phases B and C
    for r in ("fss", "nss"):
        _cover(r, (spec.ss_pair(j, t) for j, t in P(range(spec.NA), range(MT))))
    _cover("ardy", (spec.ardy_pair(j) for j in range(spec.NA)))
    _cover(
        "qpart",
        (
            spec.qpart_pair(T, c, t, r)
            for T, c, t, r in P(
                range(spec.QKV_TILES), range(spec.K1_MAXC), range(MT), range(16)
            )
        ),
    )
    _cover("srdy", (spec.srdy_pair(u) for u in range(spec.BLOCKS)))
    _cover("crdy", (spec.crdy_pair(t, h) for t, h in P(range(MT), range(spec.HEADS))))


def test_plain_region_indexes():
    off, size = spec.scratch_layout()[spec.XQD]
    words = [
        *(spec.xqd_xq_word(t, w) for t, w in P(range(MT), range(spec.XQ_WORDS))),
        *(spec.xqd_xsc_word(t, k) for t, k in P(range(MT), range(spec.XSC_WORDS))),
    ]
    assert sorted(words) == list(range(MT * spec.XQD_ROW_WORDS))
    assert 4 * len(words) <= size
    off, size = spec.scratch_layout()[spec.XROW]
    elems = sorted(spec.xrow_elem(t, c) for t, c in P(range(MT), range(spec.HIDDEN)))
    assert elems == list(range(size // 2))


def test_plain_bc_region_indexes():
    """Every element / word index formula of the B / C plain regions tiles its
    region exactly (bf16: 2 B an element; fp32: 4 B a word)."""
    lay = spec.scratch_layout()

    def tiles(name, idx, unit):
        assert sorted(idx) == list(range(lay[name][1] // unit)), name

    tiles(
        "arow", (spec.arow_elem(t, c) for t, c in P(range(MT), range(spec.HIDDEN))), 2
    )
    tiles(
        "qkvd", (spec.qkvd_elem(t, r) for t, r in P(range(MT), range(spec.QKV_ROWS))), 2
    )
    heads = P(range(MT), range(spec.HEADS), range(spec.HEAD_DIM))
    tiles("xo", (spec.xo_elem(t, h, d) for t, h, d in heads), 2)
    units = P(range(spec.BLOCKS), range(spec.Q_PER_KV))
    tiles("pstat", (spec.pstat_word(u, j, k) for (u, j), k in P(units, range(2))), 4)
    units = P(range(spec.BLOCKS), range(spec.Q_PER_KV))
    dims = range(spec.HEAD_DIM)
    tiles("opart", (spec.opart_word(u, j, d) for (u, j), d in P(units, dims)), 4)
    # rule 9's regions exist; xrow is Phase A's
    assert (
        set(spec.IN_LAUNCH_PLAIN) <= set(lay) and spec.QKVD not in spec.IN_LAUNCH_PLAIN
    )


def test_route_lds_sizes():
    words = spec.ROUTE_LDS
    assert words == dict(route=128, flag=256, expert=64, kof=512, scan=8, tot=8)
    assert sum(words.values()) * 4 == 3904  # "~3.9 KiB" (5.13)
    red = spec.WAVES * 64 * 4 * 4  # f32[8 * 64 * 4]
    assert red + sum(words.values()) * 4 < spec.LDS_BYTES


def test_placement():
    # Phase 2: seam, router and pooler CTAs are disjoint and inside the grid.
    seam = set(range(spec.NA))
    router = set(range(spec.R0, spec.R0 + spec.R_TASKS))
    pooler = set(range(spec.P0, spec.P0 + spec.P_TASKS))
    assert not (seam & router or seam & pooler or router & pooler)
    assert max(pooler) < spec.BLOCKS
    assert spec.P_TASKS == spec.P_GROUPS * spec.P_PARTS
    assert spec.P_GROUPS * spec.ROWS * TP == spec.MOE_HIDDEN
    assert spec.P_K * spec.P_PARTS == spec.HIDDEN
    # o_proj: two tasks a CTA cover every output row once.
    rows = sorted(16 * t for b in range(spec.BLOCKS) for t in (b, b + spec.BLOCKS))
    assert rows == list(range(0, spec.HIDDEN, 16))
    # Phase 3/4: owner CTAs of every rank tile the grid; unpooler tasks a CTA each.
    owners = [b for r in range(TP) for b in range(32 * r, 32 * r + spec.RED_TASKS)]
    assert owners == list(range(spec.BLOCKS))
    assert spec.DOWN_TASKS * spec.ROWS == spec.MOE_HIDDEN
    assert spec.U_GROUPS * spec.ROWS * TP == spec.HIDDEN
    assert spec.U_K * spec.U_PARTS == spec.MOE_HIDDEN
    assert {(b // spec.U_PARTS, b % spec.U_PARTS) for b in range(spec.U_TASKS)} == set(
        P(range(spec.U_GROUPS), range(spec.U_PARTS))
    )
    assert spec.G_TASKS * spec.COLS == spec.HIDDEN
    # ug: 32 columns a task, gate + up of every slot.
    assert spec.UG_GROUPS * 32 == spec.INTER
    assert spec.W13_STEPS * 128 == spec.W13_K and spec.W2_STEPS * 128 == spec.INTER
    assert spec.MAX_U * spec.UG_GROUPS <= 4 * spec.BLOCKS


def test_k1_placement():
    """DESIGN_BC 5.1: 192 GEMV CTAs x 12 tasks = 144 tiles x 16 K parts; every
    tile has exactly K1_MAXC = 2 contributors, ``k1_contributor0`` and the next
    CTA, holding its K parts as two runs (c = 0 the lower, no wrap) that cover
    each K part once; a GEMV CTA touches at most 2 tiles, its 12 K parts are
    distinct; the boundary CTAs are weight-free and hold the epilogue units."""
    assert spec.QKV_ROWS == 2304 and spec.QKV_TILES == 144
    assert spec.K1_TASKS == 12 * 192 == spec.K1_TPC * spec.K1_GEMV_CTAS
    assert spec.KPART == 512 and spec.K1_ROUNDS == 6 and spec.K1_GEMV0 == spec.NA
    assert (spec.ROW_Q, spec.ROW_K, spec.ROW_V, spec.ROW_G) == (0, 1024, 1152, 1280)
    by_tile: dict[int, dict[int, list[int]]] = {}
    for g in range(spec.K1_GEMV_CTAS):
        taus = [spec.K1_TPC * g + i for i in range(spec.K1_TPC)]
        assert len({tau % spec.KPARTS for tau in taus}) == spec.K1_TPC
        assert len({tau // spec.KPARTS for tau in taus}) <= 2
        for tau in taus:
            by_tile.setdefault(tau // spec.KPARTS, {}).setdefault(g, []).append(
                tau % spec.KPARTS
            )
    assert sorted(by_tile) == list(range(spec.QKV_TILES))
    for T, parts in by_tile.items():
        g0 = spec.k1_contributor0(T)
        assert sorted(parts) == [g0, g0 + 1], T  # NC(T) == K1_MAXC
        lo, hi = parts[g0], parts[g0 + 1]
        assert lo == list(range(len(lo))) and hi == list(range(len(lo), 16)), T
        assert (16 * T - spec.K1_TPC * g0) in (0, 4, 8)
    assert spec.K1_MAXC == 2
    # the K1 grid: weight-free boundary CTAs [0, NA), GEMV CTAs past them
    assert spec.K1_GEMV0 + spec.K1_GEMV_CTAS == spec.BLOCKS
    assert spec.E_UNITS * spec.MAX_ROWS <= spec.K1_GEMV0


@pytest.mark.parametrize("S", range(1, spec.MAX_ROWS + 1))
def test_k1_epilogue_units_cover_every_qkv_row(S):
    """DESIGN_BC 5.6: units [0, 5 S), wave = head slot, lane = dim: every
    (token, qkv row) exactly once; only unit 2's waves 4 .. 7 idle."""
    seen = []
    for u in range(spec.E_UNITS * S):
        for w, d in P(range(spec.WAVES), range(64)):
            row = spec.epilogue_row(u, w, d)
            if row is None:
                assert u % spec.E_UNITS == 2 and w >= 4
                continue
            seen.append((u // spec.E_UNITS, row))
    assert sorted(seen) == list(P(range(S), range(spec.QKV_ROWS)))
    # the kv unit: k heads on waves 0, 1, v heads on waves 2, 3
    assert [spec.epilogue_row(2, w, 0) for w in range(4)] == [1024, 1088, 1152, 1216]


@pytest.mark.parametrize("S", range(1, spec.MAX_ROWS + 1))
def test_split_placement(S):
    """DESIGN_BC 7.1: the units fit the grid, a unit's CTA is its index, the
    combine CTAs are the last 16 S, and a CTA holding both is in range."""
    for window in (False, True):
        n = spec.nsplit(S, window)
        units = spec.n_units(S, window)
        assert units == 2 * S * n <= spec.BLOCKS
        got = [spec.split_unit(u, n) for u in range(units)]
        assert got == list(P(range(S), range(spec.KV_HEADS), range(n)))
    assert spec.nsplit(S, True) == spec.NSPLIT_SW == 16
    assert spec.C0(S) >= spec.BLOCKS // 2 and spec.BLOCKS - spec.C0(S) == 16 * S
    full = [spec.nsplit(s, False) for s in range(1, spec.MAX_ROWS + 1)]
    assert full == [128, 64, 42, 32, 25, 21, 18, 16]


SPLIT_LENS = [
    *range(0, 70),
    *range(240, 270),
    *range(2030, 2100),
    *range(4090, 4120),
    32773,
    131072,
]


@pytest.mark.parametrize("window", [False, True])
def test_split_plan_covers_every_key_once(window):
    """Every live unit's pages are in [b0, b0 + nb), together once each; the
    window's keys [lo, L) are on those pages and no key outside is; units past
    ``live`` are empty; a sliding window has at most 129 pages, so per <= 9 and
    live <= 15 (DESIGN_BC 7.1); the gather never nulls a page the split reads
    (block_table.py: ``max(L - 1 - 2047, 0) // 16`` entries)."""
    for S, L in P((1, 3, 8), SPLIT_LENS):
        n = spec.nsplit(S, window)
        lo, b0, nb, per, live = spec.split_plan(L, window, n)
        assert per >= spec.MIN_PER and live <= n
        pages: list[int] = []
        for p in range(n):
            first, end = b0 + p * per, min(b0 + (p + 1) * per, b0 + nb)
            if p >= live:
                assert first >= end, (L, p)
                continue
            assert first < end, (L, p)
            pages += range(first, end)
        assert pages == list(range(b0, b0 + nb)), (S, L)
        keys = range(lo, L)
        assert {k // spec.KV_BLOCK for k in keys} == set(pages), (S, L)
        assert lo == (max(0, L - spec.WINDOW) if window else 0)
        if window:
            assert nb <= 129 and per <= 9, (L, nb, per, live)
            if nb == 129:  # a full window (DESIGN_BC 7.1: per 9, live 15)
                assert (per, live) == (9, 15)
            assert max(L - 1 - (spec.WINDOW - 1), 0) // spec.KV_BLOCK <= b0
    # the design's examples
    assert spec.split_plan(2048, False, 128)[3:] == (8, 16)
    assert spec.split_plan(32768, False, 128)[3:] == (16, 128)
    assert spec.split_plan(32768, False, 16)[3] == 128
    assert spec.split_plan(0, window, 16) == (0, 0, 0, spec.MIN_PER, 0)


def test_model_constants_match_muse_shapes():
    assert len(spec.MUSE_SHAPE_FIELDS) == len(spec.MUSE_SHAPES)
    s = dict(zip(spec.MUSE_SHAPE_FIELDS, spec.MUSE_SHAPES))
    assert s["hidden_size"] == spec.HIDDEN
    assert s["num_attention_heads"] == spec.HEADS * TP
    assert s["num_key_value_heads"] == spec.KV_HEADS * TP
    assert s["head_dim"] == spec.HEAD_DIM and spec.CORE == 1024
    assert s["num_local_experts"] == spec.EXPERTS
    assert s["num_experts_per_tok"] == spec.TOPK
    assert s["moe_hidden_size"] == spec.MOE_HIDDEN
    assert s["moe_intermediate_size"] == spec.INTER * TP
    assert s["num_shared_experts"] == 0
    assert spec.SERVED_Q == (1, spec.MAX_ROWS)
    # Phases B and C (checkpoint config: qk_scale_factor, rms_norm_eps,
    # post_norm_eps, sliding_window; model.py:408-449 builds impl.scale so)
    assert s["sliding_window"] == spec.WINDOW == 2048
    assert spec.QK_SCALE == 0.125 * 5.4730064863875
    assert spec.QK_SCALE_LOG2E == spec.QK_SCALE * spec.LOG2E
    assert (spec.RMS_EPS, spec.POST_FFN_EPS) == (1e-5, 1e-8)
    assert spec.KV_C == 128 and spec.V_OFF == 64 and spec.Q_PER_KV == 8
    assert spec.KV_BLOCK == 16 and spec.ROPE_THETA == 5e5


def test_tags():
    assert spec.k2_slot(0) == 128 and spec.k2_slot(61) == 189
    assert spec.k1_slot(5) == 5 and spec.k1_slot(127) == 127
    # a step's K1 and K2 tags never alias (DESIGN_BC 5.7)
    k1 = {spec.step_tag(9, spec.k1_slot(i)) for i in range(62)}
    k2 = {spec.step_tag(9, spec.k2_slot(i)) for i in range(62)}
    assert len(k1) == len(k2) == 62 and not k1 & k2
    for bad in (-1, 128):
        with pytest.raises(ValueError):
            spec.k2_slot(bad)
        with pytest.raises(ValueError):
            spec.k1_slot(bad)
    assert spec.step_tag(0, 128) == 129
    assert spec.step_tag(3, 130) == 3 * 256 + 131
    # 22 bits of the epoch: a wrapped (negative) int32 epoch still gives a valid
    # tag, and parity e & 1 keeps alternating across the wrap.
    wrapped = -(2**31)  # int32 2**31 - 1, plus one
    assert spec.step_tag(wrapped, 128) == spec.step_tag(0, 128)
    assert spec.step_tag(2**31 - 1, 128) == spec.EPOCH_MASK * 256 + 129
    assert (wrapped & 1) != ((2**31 - 1) & 1)
    assert spec.step_tag(7, 140) == spec.step_tag(7 + (1 << 22), 140)  # aliasing
    tags = {spec.step_tag(e, s) for e in (-2, -1, 0, 1, 2) for s in range(256)}
    assert len(tags) == 5 * 256
    assert all(0 < t < 2**31 for t in tags)


def test_k2_build_key():
    k = K2Build(tokens=8)
    assert dataclasses.is_dataclass(k) and hash(k) == hash(K2Build(tokens=8))
    with pytest.raises(dataclasses.FrozenInstanceError):
        k.tokens = 1  # type: ignore[misc,union-attr]
    scalars = (int, float, bool, str)
    assert all(isinstance(getattr(k, f.name), scalars) for f in dataclasses.fields(k))
    assert "rank" not in {f.name for f in dataclasses.fields(k)}
    for bad in (
        dict(tokens=0),
        dict(tokens=spec.MAX_ROWS + 1),
        dict(tokens=1, fault_rank=8),
        dict(tokens=1, fault_slot=256),
        dict(tokens=1, fault_ticks=-1),
    ):
        with pytest.raises(ValueError):
            K2Build(**bad)

    builds = [
        *(K2Build(tokens=s) for s in range(1, spec.MAX_ROWS + 1)),
        K2Build(tokens=8, debug_dump=True),
        K2Build(tokens=8, fault_rank=3, fault_epoch=5, fault_slot=129, fault_ticks=9),
        K2Build(tokens=8, fault_rank=3, fault_epoch=5, fault_slot=130, fault_ticks=9),
    ]
    names = [b.kernel_name() for b in builds]
    assert len(set(names)) == len(builds) == len(set(builds))
    assert names[0] == "muse_mono_k2_s1"
    assert names[8] == "muse_mono_k2_s8_dbg"
    assert names[9] == "muse_mono_k2_s8_f3_5_129_9"
    assert not builds[0].faulty and builds[9].faulty


def test_k2_build_rejects_keys_that_would_misbuild():
    for bad in (
        dict(tokens=True),  # a bool is an int: it would name the symbol "sTrue"
        dict(tokens=8.0),
        dict(tokens=1, debug_dump=1),
        dict(tokens=1, fault_rank=3, fault_epoch=5.0, fault_slot=129, fault_ticks=9),
    ):
        with pytest.raises(TypeError):
            K2Build(**bad)
    for bad in (
        # fault fields without the hook: a second key, the same binary and name
        dict(tokens=1, fault_epoch=5),
        dict(tokens=1, fault_slot=129),
        dict(tokens=1, fault_ticks=9),
        # a stall that can never fire
        dict(tokens=1, fault_rank=3, fault_epoch=5, fault_slot=1, fault_ticks=9),
        dict(tokens=1, fault_rank=3, fault_epoch=5, fault_slot=129, fault_ticks=0),
        dict(tokens=1, fault_rank=3, fault_epoch=-1, fault_slot=129, fault_ticks=9),
        dict(tokens=1, fault_rank=3, fault_epoch=2**31, fault_slot=129, fault_ticks=9),
    ):
        with pytest.raises(ValueError):
            K2Build(**bad)
    # The design's fault hooks (6.3 stalled-rank test, 6.6 engine-fault test).
    ticks = abi.WAIT_SECONDS * 100_000_000 * 3 // 2
    for epoch in (2, 256):
        k = K2Build(
            tokens=8, fault_rank=3, fault_epoch=epoch, fault_slot=129, fault_ticks=ticks
        )
        assert k.kernel_name() == f"muse_mono_k2_s8_f3_{epoch}_129_{ticks}"


def test_k2_build_cores():
    """DESIGN_BC 3.2: the core and the window are key fields; a Phase A build
    ("tensor", the default) keeps its Phase A name."""
    assert K2Build(tokens=3) == K2Build(tokens=3, core="tensor", window=False)
    # Phase A's fields first, in Phase A's order: a Phase A key's key_tuple
    # prefix (and every positional construction) is unchanged
    assert [f.name for f in dataclasses.fields(K2Build)] == [
        "tokens",
        "debug_dump",
        "fault_rank",
        "fault_epoch",
        "fault_slot",
        "fault_ticks",
        "core",
        "window",
    ]
    assert spec.K2_CORES == ("tensor", "gated", "split")
    names = {
        K2Build(tokens=3).kernel_name(): "muse_mono_k2_s3",
        K2Build(tokens=3, core="gated").kernel_name(): "muse_mono_k2_s3_g",
        K2Build(tokens=3, core="split").kernel_name(): "muse_mono_k2_s3_full",
        K2Build(tokens=3, core="split", window=True).kernel_name(): (
            "muse_mono_k2_s3_sw"
        ),
        K2Build(
            tokens=8,
            core="gated",
            fault_rank=1,
            fault_epoch=2,
            fault_slot=130,
            fault_ticks=7,
            debug_dump=True,
        ).kernel_name(): "muse_mono_k2_s8_g_f1_2_130_7_dbg",
    }
    assert names == {v: v for v in names.values()}
    for bad in (
        dict(tokens=1, core="dense"),
        dict(tokens=1, window=True),  # tensor: no window
        dict(tokens=1, core="gated", window=True),
    ):
        with pytest.raises(ValueError):
            K2Build(**bad)
    for bad in (dict(tokens=1, core=1), dict(tokens=1, core="split", window=1)):
        with pytest.raises(TypeError):
            K2Build(**bad)


def test_build_counts_and_names_are_distinct():
    """DESIGN_BC 3.2: q = 1 serves K1 8 x {rope, nope} and K2 8 x {sw, full};
    DSpark q = 8 adds K1 x 2 and K2(gated); with Phase A's tensor builds and
    B2's gated widths no two builds share a kernel symbol."""
    widths = range(1, spec.MAX_ROWS + 1)
    k1 = [K1Build(tokens=s, rope=r) for s, r in P(widths, (False, True))]
    k2 = [
        K2Build(tokens=s, core="split", window=w) for s, w in P(widths, (False, True))
    ]
    k2 += [K2Build(tokens=s, core=c) for s, c in P(widths, ("tensor", "gated"))]
    builds = k1 + k2
    names = [b.kernel_name() for b in builds]
    assert len(k1) == 16 and len(k2) == 32
    assert len(set(names)) == len(builds) == len(set(builds))
    assert all(n.startswith("muse_mono_k1_s") for n in names[:16])
    assert K1Build(tokens=8, rope=True).kernel_name() == "muse_mono_k1_s8_rope"
    assert K1Build(tokens=1, rope=False).kernel_name() == "muse_mono_k1_s1_nope"
    plan = _plan()
    assert len({plan.key_tuple(b, "src") for b in builds}) == len(builds)


def test_k1_build_key():
    k = K1Build(tokens=4, rope=True)
    assert hash(k) == hash(K1Build(4, True))  # frozen: hashable, immutable
    with pytest.raises(dataclasses.FrozenInstanceError):
        k.rope = False  # type: ignore[misc]
    scalars = (int, float, bool, str)
    assert all(isinstance(getattr(k, f.name), scalars) for f in dataclasses.fields(k))
    assert "rank" not in {f.name for f in dataclasses.fields(k)}
    assert not k.faulty and not k.debug_dump
    # a K1 fault fires in a K1 launch: slot k1_slot(i) (DESIGN_BC 5.10)
    ticks = abi.WAIT_SECONDS * 100_000_000 * 3 // 2
    f = K1Build(
        tokens=8,
        rope=False,
        fault_rank=3,
        fault_epoch=2,
        fault_slot=1,
        fault_ticks=ticks,
    )
    assert f.faulty and f.kernel_name() == f"muse_mono_k1_s8_nope_f3_2_1_{ticks}"
    assert K1Build(tokens=2, rope=True, debug_dump=True).kernel_name() == (
        "muse_mono_k1_s2_rope_dbg"
    )
    for bad in (
        dict(tokens=0, rope=True),
        dict(tokens=9, rope=True),
        dict(tokens=1, rope=True, fault_rank=8),
        dict(tokens=1, rope=True, fault_slot=1),  # no hook
        # a K2 slot never runs a K1
        dict(
            tokens=1,
            rope=True,
            fault_rank=0,
            fault_epoch=1,
            fault_slot=128,
            fault_ticks=1,
        ),
        dict(
            tokens=1,
            rope=True,
            fault_rank=0,
            fault_epoch=1,
            fault_slot=-1,
            fault_ticks=1,
        ),
        dict(
            tokens=1,
            rope=True,
            fault_rank=0,
            fault_epoch=1,
            fault_slot=5,
            fault_ticks=0,
        ),
    ):
        with pytest.raises(ValueError):
            K1Build(**bad)
    for bad in (
        dict(tokens=True, rope=True),
        dict(tokens=1, rope=1),
        dict(tokens=1, rope=True, debug_dump=0),
    ):
        with pytest.raises(TypeError):
            K1Build(**bad)
    with pytest.raises(TypeError):
        K1Build(tokens=1)  # type: ignore[call-arg]  # the kind is not defaulted
    # and a K2 fault in a K1 slot never fires
    with pytest.raises(ValueError):
        K2Build(tokens=1, fault_rank=0, fault_epoch=1, fault_slot=1, fault_ticks=1)


def test_k2_build_key_tuple():
    plan = _plan()
    builds = [
        *(K2Build(tokens=s) for s in range(1, spec.MAX_ROWS + 1)),
        K2Build(tokens=8, debug_dump=True),
        K2Build(tokens=8, fault_rank=3, fault_epoch=5, fault_slot=129, fault_ticks=9),
    ]
    keys = [plan.key_tuple(b, "src") for b in builds]
    assert len(set(keys)) == len(builds)
    assert keys[0][0] == ("tokens", 1) and keys[0][-1] == ("sources", "src")


LAUNCH_STATE = ("scratch", "peers", "rank", "epoch", "slot", "health", "alarm")
KV_ARGS = ("kv", "kv_sb", "kv_sh", "kv_st", "kv_blocks")


def test_k2_args():
    assert len(spec.K2_ARGS) == len(set(spec.K2_ARGS)) == 26
    assert set(spec.K2_INT32_ARGS) <= set(spec.K2_ARGS)
    # The design's argument order (5.1): inputs, weights, outputs, launch state.
    assert spec.K2_ARGS[:3] == ("core", "stream", "pad")
    assert spec.K2_ARGS[-7:] == LAUNCH_STATE


def test_k2_args_by_core():
    """DESIGN_BC 6.1 / 7.1: "tensor" is Phase A's list; "gated" replaces
    ``core`` with ``attn`` and ``qkvo``; "split" with ``qkvo``, the KV cache
    and the block-table arguments. Everything else stays, in order."""
    by = spec.K2_ARGS_BY_CORE
    assert set(by) == set(spec.K2_CORES) == set(spec.K2_INT32_ARGS_BY_CORE)
    assert by["tensor"] == spec.K2_ARGS
    rest = spec.K2_ARGS[1:]
    assert by["gated"] == ("attn", "qkvo", *rest)
    assert by["split"] == (
        "qkvo",
        *KV_ARGS,
        "block_table",
        "bt_rows",
        "bt_stride",
        "seq_lens",
        *rest,
    )
    for core, args in by.items():
        assert len(args) == len(set(args)) and args[-7:] == LAUNCH_STATE, core
        assert set(spec.K2_INT32_ARGS_BY_CORE[core]) <= set(args), core
    assert set(spec.K2_INT32_ARGS_BY_CORE["split"]) == {
        "kv_blocks",
        "bt_rows",
        "bt_stride",
        "rank",
        "slot",
    }


def test_k1_args():
    """DESIGN_BC 5.2: step tensors, the two layers' weights, the KV cache, the
    outputs, then K2's launch state without ``peers`` (K1 is rank-local, but
    keeps ``rank`` for the fault hook)."""
    args = spec.K1_ARGS
    assert len(args) == len(set(args)) == 25
    assert args[:5] == ("branch", "stream", "pad", "positions", "slots")
    assert args[5:11] == (
        "w_postffn",
        "ffn_alpha",
        "ffn_beta",
        "w_preqkv",
        "w_qkv",
        "cos_sin",
    )
    assert args[11:17] == (*KV_ARGS, "max_pos")
    assert args[17:19] == ("stream_out", "qkvo")
    assert args[19:] == tuple(n for n in LAUNCH_STATE if n != "peers")
    assert spec.K1_INT32_ARGS == ("kv_blocks", "max_pos", "slot", "rank")
    assert set(spec.K1_INT32_ARGS) <= set(args) and "peers" not in args


def test_regions():
    assert spec.REGIONS == spec.SCRATCH_REGIONS_ALL + spec.PEER_REGIONS
    assert spec.SCRATCH_REGIONS_ALL == spec.SCRATCH_REGIONS + spec.SCRATCH_REGIONS_BC
    assert len(set(spec.REGIONS)) == len(spec.REGIONS) == 22
    assert tuple(n for n, _ in spec.SCRATCH_PAIRS) == spec.SCRATCH_REGIONS
    assert tuple(n for n, _ in spec.SCRATCH_PAIRS_BC) == spec.SCRATCH_REGIONS_BC
    assert tuple(n for n, _ in spec.PEER_PAIRS) == spec.PEER_REGIONS
    plain = (spec.XQD, spec.XROW, *(n for n, _, _ in spec.SCRATCH_PLAIN_BC))
    assert not set(plain) & set(spec.REGIONS)
    # who puts and polls what: K2 (every core) Phase A's regions, K1 and the
    # split core the B / C mailboxes, each one exactly one kernel's
    assert spec.K2_REGIONS == spec.SCRATCH_REGIONS + spec.PEER_REGIONS
    assert spec.K1_REGIONS + spec.SPLIT_REGIONS == spec.SCRATCH_REGIONS_BC
    # A wait record names its region by CRC-32 (common/debug.region_id): no clash.
    ids = {zlib.crc32(n.encode()) & 0x7FFFFFFF for n in spec.REGIONS}
    assert len(ids) == len(spec.REGIONS)


def test_abi():
    assert abi.MONO_ABI_VERSION == 1
    assert abi.HEALTH_WORDS == abi.health_words(spec.BLOCKS) == 2056
    assert abi.FAILED == 0 and abi.WAIT_BUDGET == 1 < abi.RECORD_WORDS
    assert abi.STARTUP_WAIT_SECONDS == 120 and abi.WAIT_SECONDS == 10
    # The alarm flags, a word a rank, fit the peer buffer's head.
    assert 4 * spec.TP <= abi.PEER_HEAD
    # _spin's budget in s_memrealtime ticks (100 MHz) fits the int64 it is kept in.
    assert abi.STARTUP_WAIT_SECONDS * 100_000_000 < 2**63


@pytest.mark.parametrize("name", ["spec.py", "abi.py"])
def test_contract_modules_need_no_torch_or_flydsl(name):
    tree = ast.parse((MONO / name).read_text())
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert imported <= {"dataclasses"}, imported
    # Loaded on its own in a clean interpreter, it pulls in neither.
    code = (
        "import importlib.util, sys\n"
        f"s = importlib.util.spec_from_file_location('m', {str(MONO / name)!r})\n"
        "m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
        "assert not {'torch', 'flydsl'} & set(sys.modules), sorted(sys.modules)\n"
    )
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_mono_package_init_imports_nothing():
    tree = ast.parse((MONO / "__init__.py").read_text())
    assert not [
        n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
