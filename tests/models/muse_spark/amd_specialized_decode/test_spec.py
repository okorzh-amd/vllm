# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark specialized decode's shared contract on CPU: the scratch and peer
layouts (offsets, alignment, non-overlap, width independence, every pair index
inside its region), the K2 placement, the tags, the build key and the
error-word ABI. Pure Python: no GPU, no FlyDSL."""

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
from vllm.models.muse_spark.amd.specialized_decode.mono.spec import K2Build

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
    assert spec.scratch_layout() == SCRATCH
    assert spec.scratch_bytes() == 534016
    assert spec.peer_layout() == PEER
    assert spec.peer_half_bytes() == 3690496
    assert spec.peer_bytes() == 7380992
    assert abi.PEER_HEAD + spec.peer_bytes() == 7381248  # what the runner allocates


@pytest.mark.parametrize("layout", ["scratch", "peer"])
def test_regions_are_aligned_ordered_and_disjoint(layout):
    if layout == "scratch":
        got = spec.scratch_layout()
        order = (*spec.SCRATCH_REGIONS, spec.XQD, spec.XROW)
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
    pairs = dict(spec.SCRATCH_PAIRS + spec.PEER_PAIRS)[region]
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


def test_tags():
    assert spec.k2_slot(0) == 128 and spec.k2_slot(61) == 189
    assert spec.k1_slot(5) == 5
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


def test_k2_args():
    assert len(spec.K2_ARGS) == len(set(spec.K2_ARGS)) == 26
    assert set(spec.K2_INT32_ARGS) <= set(spec.K2_ARGS)
    # The design's argument order (5.1): inputs, weights, outputs, launch state.
    assert spec.K2_ARGS[:3] == ("core", "stream", "pad")
    assert spec.K2_ARGS[-7:] == (
        "scratch",
        "peers",
        "rank",
        "epoch",
        "slot",
        "health",
        "alarm",
    )


def test_regions():
    assert spec.REGIONS == spec.SCRATCH_REGIONS + spec.PEER_REGIONS
    assert len(set(spec.REGIONS)) == len(spec.REGIONS)
    assert tuple(n for n, _ in spec.SCRATCH_PAIRS) == spec.SCRATCH_REGIONS
    assert tuple(n for n, _ in spec.PEER_PAIRS) == spec.PEER_REGIONS
    assert spec.XQD not in spec.REGIONS and spec.XROW not in spec.REGIONS
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
