# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1's qkv GEMV and epilogue (``mono/front/qkv.py``, ``front/epilogue.py``,
DESIGN_BC 5.5 / 5.6, WP-B3) in probe kernels.

    front   K1 without the boundary, on the full grid with K1's context
            (``peers=False``, ``route_lds=False``, a debug ``K1Build``): the
            host writes the ``arow`` rows into the probe scratch and
            publishes the 64 ``ardy`` flags; the GEMV CTAs [64, 256) run
            ``qkv_loads`` + ``stage_qkv``, the epilogue CTAs [0, 5 S)
            ``boundary.epilogue_inputs`` + ``stage_epilogue`` (``qkvd``: the
            GEMV rows before the epilogue, ``qkvo``, the cache)
    bulk    the epilogue's math alone (``qk_norm``, ``rope_loads``, ``rope``,
            as ``stage_epilogue`` calls them) on every q / k head of a qkv
            tensor of 1024 tokens, a head a wave, 9 a wave: 1.18 M elements
            a launch, so dims 0 and 7 of every chunk (the two the "obvious"
            RoPE gets wrong, review 0 D1) are exercised ~10^5 times each

[CPU] the GEMV task map (every tile's two partials put once, in K part
order), the epilogue's row map against ``spec.epilogue_row``, the per-lane
RoPE formula against B0's per-dim reference (1.3's table), the LDS sizes,
the stock HPG choice the comparisons rely on; every probe compiles GPU-less
for gfx950 within the budgets (no private segment, no readfirstlane
waterfall), with the GEMV's 24 weight loads before the ``ardy`` poll and
its staging loads before its first ds_write, and the 1.3 arithmetic in the
ISA (one ``v_rsq_f32``, the serial ``row_shr:1`` chain, a ``v_sub_f32`` of
two ``v_mul_f32``, 4 fmas with RoPE and 1 without).
[1GPU] 9.2's ``test_probe_qkv`` acceptance: ``qkvd`` vs fp64 (<= 1 bf16 ulp
past fp32 noise) and ``F.linear``; ``qkvo`` (live q / k / v / og rows) and
the cache bytes ``torch.equal`` the stock fused insert run on ``qkvd``;
the bulk math ``torch.equal`` the stock insert (rope and NoPE); stock vs
stock: 8 rows alone (HPG 1) and the same rows inside a 3072-token call (HPG
4; NoPE 2048 tokens, HPG 2) write the same bytes (E4); slot -1 leaves the
cache byte-identical; slots at n = 15 and n = 0 of a block; LHBNC strides;
a > 4 GiB cache with a block past 2^32 B; positions at and past
``max_pos`` (clamped); a garbage slot (block past ``kv_blocks``); FAILED set
-> no cache write and no hang; two launches equal.
"""

import glob
import os

import pytest
import regex as re
import torch

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

import flydsl.expr as fx  # noqa: E402
from aiter.ops.flydsl.kernels import buffer_ops as bo  # noqa: E402
from flydsl.expr import const_expr, range_constexpr  # noqa: E402
from flydsl.expr.typing import T  # noqa: E402

from tests.models.muse_spark.amd_specialized_decode import (  # noqa: E402
    reference,
)
from vllm.models.muse_spark.amd.specialized_decode.mono import spec  # noqa: E402
from vllm.models.muse_spark.amd.specialized_decode.mono.abi import (  # noqa: E402
    FAILED,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common import (  # noqa: E402
    context,
    plan,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (  # noqa: E402
    ld_i32,
    rsrc,
    traced,
    uniform,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.plan import (  # noqa: E402
    Space,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.front import (  # noqa: E402
    boundary,
    epilogue,
    qkv,
)

bf = torch.bfloat16
QKV, MT, H = spec.QKV_ROWS, spec.MAX_TOKENS, spec.HIDDEN
NQK = spec.HEADS + spec.KV_HEADS  # q and k heads a token (normalized)
SLOT = spec.k1_slot(1)  # the probes' K1 slot (layer 1)
PAD_BYTES = 16  # the pad flags (whole words: ld_pad's)
WIDTHS = (1, 3, 8)
FRONT_ARGS = (
    "w_qkv",
    "positions",
    "slots",
    "pad",
    "cos_sin",
    *spec.K2_KV_ARGS,
    "max_pos",
    "qkvo",
)
BULK_ARGS = ("x", "positions", "cos_sin", "max_pos", "out")
BULK_P = 9  # heads a wave
BULK_TOKENS = spec.BLOCKS * spec.WAVES * BULK_P // NQK  # 1024
MAX_POS = 1 << 20  # the served table's rows (rotary max_position)


def _on_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


def _compiles_for_gfx950() -> bool:
    arch = os.environ.get("FLYDSL_GPU_ARCH") or os.environ.get("ARCH")
    return arch == "gfx950" or _on_gfx950()


needs_gfx950 = pytest.mark.skipif(
    not _on_gfx950(), reason="the Muse mono probes run on gfx950 (MI355X)"
)
compiles = pytest.mark.skipif(
    not _compiles_for_gfx950(),
    reason="GPU-less compile needs FLYDSL_GPU_ARCH=gfx950 (or a gfx950 GPU)",
)


# ---------------------------------------------------------------- the probes


@traced
def _front(c):
    """K1 past the boundary (module docstring): GEMV CTAs, then epilogue
    CTAs (disjoint at S <= 8: 5 S <= 40 < 64)."""
    b = fx.Int32(c["bid"])
    if b >= spec.K1_GEMV0:
        g = b - spec.K1_GEMV0
        ops = qkv.qkv_loads(c, g)
        qkv.stage_qkv(c, g, ops)
    if b < spec.E_UNITS * c["S"]:
        epilogue.stage_epilogue(c, b, boundary.epilogue_inputs(c, b))


def _front_lds(s: int) -> dict:
    return {qkv.STAGE: qkv.lds_struct(s)}


def _front_probe(s: int, rope: bool, debug: bool = True) -> context.Probe:
    kind = "rope" if rope else "nope"
    return context.probe_kernel(
        f"muse_probe_k1_front_s{s}_{kind}" + ("_dbg" if debug else ""),
        s,
        _front,
        _front_lds(s),
        FRONT_ARGS,
        blocks=spec.BLOCKS,
        peers=False,
        route_lds=False,
        build=spec.K1Build(tokens=s, rope=rope, debug_dump=debug),
    )


def _ld_bf16(addr, i):
    return fx.BFloat16(bo.buffer_load(rsrc(addr), i, vec_width=1, dtype=T.bf16)).to(
        fx.Float32
    )


def _bulk_body(rope: bool):
    @traced
    def body(c):
        """Wave w of CTA b: heads (8 b + w) 9 + p, p < 9, of the q / k heads
        of ``x`` (bf16 [1024, 2304]; head hh: token hh // 18, slot hh % 18,
        lane = dim) -> ``out`` at the same element, after ``qk_norm`` (and
        ``rope`` at the token's position: int32 low words, in range)."""
        lane = c["lane"]
        a = c["args"]
        w = uniform(c["wave"])
        b = fx.Int32(c["bid"])
        for p in range_constexpr(BULK_P):
            hh = (b * spec.WAVES + w) * BULK_P + p
            tok = hh // NQK
            e = tok * QKV + hh % NQK * spec.HEAD_DIM + lane
            y = epilogue.qk_norm(_ld_bf16(a["x"], e), lane)
            if const_expr(rope):
                pos = uniform(ld_i32(a["positions"], 2 * tok))
                cs, sn = epilogue.rope_loads(c, pos, lane)
                y = epilogue.rope(y, cs, sn, lane)
            bo.buffer_store(y.to(fx.BFloat16), rsrc(a["out"]), e)

    return body


_BULK = {r: _bulk_body(r) for r in (False, True)}


def _bulk_probe(rope: bool) -> context.Probe:
    return context.probe_kernel(
        f"muse_probe_k1_epi_bulk_{'rope' if rope else 'nope'}",
        1,
        _BULK[rope],
        None,
        BULK_ARGS,
        blocks=spec.BLOCKS,
        peers=False,
        route_lds=False,
    )


# ---------------------------------------------------------------- ISA helpers

# readfirstlane waterfall around a load whose base is not wave-uniform
_WATERFALL = re.compile(r"v_cmp_eq_u64_e\d+\s+\S+,\s*s\[\d+:\d+\],\s*v\[")
_LOAD_NT = re.compile(r"^\s*buffer_load_dwordx4\b.*\bnt\b", re.M)
_LOAD_DEV = re.compile(r"^\s*buffer_load_dwordx4\b[^\n]*\bsc1\b", re.M)
_MFMA = re.compile(r"^\s*v_mfma_f32_16x16x32_bf16\b", re.M)
# a buffer load at the default scope (no sc0 / sc1 bit: not a hand-off)
_PLAIN_LOAD = re.compile(r"^\s*buffer_load_\w+(?![^\n]*\bsc[01]\b)[^\n]*$", re.M)


def _isa(dump: str, name: str) -> tuple[str, dict]:
    for path in glob.glob(f"{dump}/**/*final_isa.s", recursive=True):
        with open(path) as f:
            txt = f.read()
        if re.search(rf"\.amdhsa_kernel {re.escape(name)}\b", txt):
            res = {}
            for k in (
                "next_free_vgpr",
                "group_segment_fixed_size",
                "private_segment_fixed_size",
            ):
                m = re.search(rf"\.amdhsa_{k}\s+(\d+)", txt)
                res[k] = int(m.group(1)) if m else None
            return txt, res
    raise AssertionError(f"no final ISA of {name} under {dump}")


def _compile(probe, tmp_path, monkeypatch) -> tuple[str, dict]:
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    try:
        probe.compile()
    except RuntimeError as e:  # the load, past the compile, needs a device
        if "hipErrorNoDevice" not in str(e) and "no ROCm" not in str(e):
            raise
    txt, res = _isa(str(tmp_path), probe.name)
    print(f"{probe.name}: {res}")
    assert res["private_segment_fixed_size"] == 0, res
    assert res["next_free_vgpr"] <= spec.VGPR_MAX, res
    assert res["group_segment_fixed_size"] <= spec.LDS_BYTES, res
    loops = _WATERFALL.findall(txt)
    assert not loops, f"{len(loops)} waterfall compares, e.g. {loops[0]}"
    return txt, res


def _code(txt: str) -> list[str]:
    """The instruction lines of a final ISA (comments and labels dropped)."""
    out = []
    for line in txt.splitlines():
        s = line.split(";")[0].strip()
        if s and not s.startswith(".") and not s.endswith(":"):
            out.append(s)
    return out


def _fmas(code: list[str]) -> list[str]:
    """The fused multiply-adds (every encoding)."""
    pat = re.compile(r"^(v_(?:pk_)?fma\w*|v_fmac\w*|v_fmamk\w*|v_fmaak\w*|v_mac\w*)")
    return [s for s in code if pat.match(s)]


def _sub_of_two_muls(code: list[str]) -> bool:
    """Some ``v_sub_f32`` whose two sources were last written by
    ``v_mul_f32``s: 1.3's dim 0, RN(RN(x0 c) - RN(x1 s)), not contracted."""
    for i, s in enumerate(code):
        m = re.match(r"v_sub_f32(?:_e\d+)?\s+v\d+,\s*(v\d+),\s*(v\d+)$", s)
        if not m:
            continue
        srcs, found = {m.group(1), m.group(2)}, {}
        for prev in reversed(code[max(0, i - 64) : i]):
            d = re.match(r"(v_\w+)\s+(v\d+),", prev)
            if d and d.group(2) in srcs and d.group(2) not in found:
                found[d.group(2)] = d.group(1)
        if len(found) == 2 and all(o.startswith("v_mul_f32") for o in found.values()):
            return True
    return False


# ---------------------------------------------------------------- [CPU]


def test_stage_regions():
    """The stages poll and put only K1's scratch mailboxes (rank-local)."""
    for mod in (qkv, epilogue):
        assert set(mod.STAGE_REGIONS) <= set(spec.K1_REGIONS), mod.STAGE
    assert qkv.STAGE_REGIONS == ("ardy", "qpart")
    assert epilogue.STAGE_REGIONS == ("qpart",)


def test_task_map_puts_every_partial_once():
    """Every tile's two contributors (``qkv.contributors``) put their partial
    once, c = 0 the lower K parts; together they hold the tile's 16 K parts
    once each, in ascending order within each contributor; a GEMV CTA's 12
    tasks span <= 2 tiles; the put of the last tile is at i == 11."""
    puts: dict = {}
    for g in range(spec.K1_GEMV_CTAS):
        tiles, run = [], []
        for i in range(spec.K1_TPC):
            tile, kp = qkv.task(g, i)
            run.append(kp)
            last = qkv.last_of_tile(g, i)
            assert isinstance(last, bool)
            if last:
                first, n = qkv.contributors(tile)
                assert n == spec.K1_MAXC
                c = g - first
                assert 0 <= c < n, (g, tile, c)
                assert (tile, c) not in puts, (g, tile, c)
                assert run == sorted(run) and len(set(run)) == len(run)
                puts[(tile, c)] = run
                tiles.append(tile)
                run = []
        assert run == [] and 1 <= len(tiles) <= 2, (g, tiles)
    assert len(puts) == spec.QKV_TILES * spec.K1_MAXC
    for tile in range(spec.QKV_TILES):
        lo, hi = puts[(tile, 0)], puts[(tile, 1)]
        assert lo + hi == list(range(spec.KPARTS)), (tile, lo, hi)
        assert lo and hi
        assert qkv.contributors(tile)[0] == spec.k1_contributor0(tile)


def test_epilogue_row_map():
    """``epilogue.unit_row`` (the kernel's formulas on Python ints) equals
    ``spec.epilogue_row``; a token's 5 units cover its 2304 rows once; q / k
    rows are the normalized ones, k / v the inserted ones."""
    for t in range(MT):
        seen: list[int] = []
        for g5 in range(spec.E_UNITS):
            u = spec.E_UNITS * t + g5
            for w in range(spec.WAVES):
                row0, live, qk, kv = epilogue.unit_row(g5, w)
                want = spec.epilogue_row(u, w, 0)
                assert live == (want is not None), (u, w)
                if not live:
                    continue
                assert row0 == want, (u, w, row0, want)
                assert [row0 + d for d in range(64)] == [
                    spec.epilogue_row(u, w, d) for d in range(64)
                ]
                assert qk == (row0 < spec.ROW_V)
                assert kv == (spec.ROW_K <= row0 < spec.ROW_G)
                seen += range(row0, row0 + 64)
        assert sorted(seen) == list(range(QKV))


def _rope_lanes(xn: torch.Tensor, cs: torch.Tensor) -> torch.Tensor:
    """``epilogue.rope``'s per-lane formula on the host (fp32 [N, 64] heads,
    lane = dim; ``cs`` the heads' table rows): partner d ^ 1, a = RN(pr s),
    b = RN(x c); d % 8 == 0: b - a; 7: fma(pr, s, b); even: fma(x, c, -a);
    odd: fma(x, c, a)."""
    d = torch.arange(64)
    c = cs[:, d // 2]
    s = cs[:, 32 + d // 2]
    pr = xn[:, d ^ 1]
    a = pr * s
    b = xn * c
    even = reference.fma32(xn, c, -a)
    odd = reference.fma32(xn, c, a)
    d7 = reference.fma32(pr, s, b)
    mid = torch.where(d % 2 == 0, even, odd)
    k8 = d % 8
    return torch.where(k8 == 0, b - a, torch.where(k8 == 7, d7, mid))


def test_rope_lane_formula_is_the_stock_table():
    """The per-lane select of ``epilogue.rope`` (a head a wave, d = lane)
    reproduces 1.3's per-dim table as B0's ``reference.qk_norm_rope`` writes
    it (dims 0 and 7 included), bit for bit, on 2^16 heads; and differs from
    the "obvious" fp32 RoPE in dims 0 / 7 somewhere (the table bites)."""
    g = torch.Generator().manual_seed(7)
    n = 1 << 16
    heads = (torch.randn(n, 64, generator=g) * 3).to(bf)
    table = reference.rope_table(4096)
    pos = torch.randint(0, 4096, (n,), generator=g)
    cs = table[pos]
    want = reference.qk_norm_rope(heads, cs)
    x = heads.float()
    sq = (x * x).reshape(n, 8, 8)
    ch = sq[..., 0]
    for k in range(1, 8):
        ch = ch + sq[..., k]
    tot = ((ch[:, 7] + ch[:, 3]) + (ch[:, 5] + ch[:, 1])) + (
        (ch[:, 6] + ch[:, 2]) + (ch[:, 4] + ch[:, 0])
    )  # seq8_sum's lane-63 order (commutes with the .cu's)
    r = torch.rsqrt(reference.fma32(tot, torch.tensor(2.0**-6), spec.RMS_EPS))
    xn = x * r[:, None]
    got = _rope_lanes(xn, cs).to(bf)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    c, s = cs[:, torch.arange(64) // 2], cs[:, 32 + torch.arange(64) // 2]
    pr = xn[:, torch.arange(64) ^ 1]
    sign = torch.where(torch.arange(64) % 2 == 0, -1.0, 1.0)
    naive = (xn.double() * c + sign * pr.double() * s).float().to(bf)
    k8 = torch.arange(64) % 8
    for dim in (0, 7):
        cols = k8 == dim
        assert not torch.equal(naive[:, cols], want[:, cols]), dim


def test_lds_sizes():
    """``xa`` = 12 S padded K part rows; with K1's ``red`` (no RouteLds):
    20,672 B at S = 1 and 108,032 B at S = 8 (DESIGN_BC 5.9)."""
    for s in range(1, MT + 1):
        lds = context.lds_total_bytes(_front_lds(s), route_lds=False)
        assert lds == 8192 + spec.K1_TPC * s * qkv.XA_ROW * 2, (s, lds)
        assert lds <= spec.LDS_BYTES
    assert context.lds_total_bytes(_front_lds(1), route_lds=False) == 20672
    assert context.lds_total_bytes(_front_lds(8), route_lds=False) == 108032


def _stock_hpg(tokens: int, rope: bool, sms: int = 256) -> int:
    """The fused insert's HEADS_PER_GROUP for a bf16 KV insert at TP8
    (``launchFusedMuseSpark``, .cu:505-539): widen while the grid covers a
    full wave (256 CUs x 8 blocks), then the TP8 override."""
    slots, lanes, full_wave = NQK + spec.KV_HEADS, 8, sms * 8
    hpg = 1
    for h in (4, 2):
        if slots % h == 0 and (tokens * (slots // h) * lanes + 255) // 256 >= full_wave:
            hpg = h
            break
    qh = tokens * spec.HEADS
    if rope and qh >= 24576:
        hpg = max(hpg, 2)
        if qh >= 49152:
            hpg = 4
    elif not rope and qh >= 32768:
        hpg = max(hpg, 2)
    return hpg


def test_stock_instances_the_tests_compare_with():
    """Decode widths and the bulk test's 1024 tokens run HPG 1 (the instance
    1.3 reads); the stock-vs-stock test's big calls run HPG 4 (rope, 3072
    tokens) and HPG 2 (NoPE, 2048)."""
    for rope in (False, True):
        for s in (*range(1, MT + 1), BULK_TOKENS):
            assert _stock_hpg(s, rope) == 1, (s, rope)
    assert _stock_hpg(3072, True) == 4
    assert _stock_hpg(2048, False) == 2


@compiles
@pytest.mark.parametrize(
    "s, rope, debug",
    [(1, True, True), (1, False, True), (3, True, True), (8, True, True),
     (8, False, True), (8, True, False)],
)  # fmt: skip
def test_front_probe_compiles(s, rope, debug, tmp_path, monkeypatch):
    """The compile gate (5.9 budgets), the GEMV's issue order (rule 8 and
    the one staging pass) and the epilogue's 1.3 arithmetic in the ISA."""
    with plan.recording() as rec:
        txt, res = _compile(_front_probe(s, rope, debug), tmp_path, monkeypatch)
    assert res["group_segment_fixed_size"] == context.lds_total_bytes(
        _front_lds(s), route_lds=False
    )
    one_cta = res["next_free_vgpr"] > 128 or res["group_segment_fixed_size"] > (
        80 << 10
    )
    print(f"one CTA a CU without waves-per-eu: {one_cta} (B4's K1 gate)")
    # every weight before the ardy poll (its _spin waits vmcnt(0): they land
    # first), none after it; the staging loads (device scope) after the poll,
    # all before the first staging ds_write; 6 rounds x 4 MFMAs
    poll = re.search(r"\bs_sleep\b", txt).start()
    mfma = _MFMA.search(txt).start()
    assert len(_LOAD_NT.findall(txt[:poll])) == 4 * spec.K1_ROUNDS
    assert not _LOAD_NT.findall(txt[poll:])
    stage = [m.start() for m in _LOAD_DEV.finditer(txt, poll, mfma)]
    rounds = -(-spec.K1_TPC * s * qkv.X4 // spec.THREADS)
    assert len(stage) == rounds, (len(stage), rounds)
    writes = [m.start() for m in re.finditer(r"\bds_write_b128\b", txt[:mfma])]
    assert writes and stage[-1] < min(w for w in writes if w > poll)
    assert len(_MFMA.findall(txt)) == 4 * spec.K1_ROUNDS
    # the epilogue: 1.3 steps 2-6
    code = _code(txt)
    assert sum(1 for x in code if x.startswith("v_rsq_f32")) == 1
    assert len(re.findall(r"v_add_f32_dpp[^\n]*row_shr:1\b", txt)) == 7
    fmas = _fmas(code)
    assert len(fmas) == (4 if rope else 1), fmas
    assert _sub_of_two_muls(code) == rope
    assert ("quad_perm:[1,0,3,2]" in txt) == rope
    # rule 8 in the epilogue: its poll (the first past the GEMV's MFMAs)
    # holds only its operands; past it no plain load (the qpart reloads and
    # the FAILED word are device / system scope)
    last_mfma = [m.end() for m in _MFMA.finditer(txt)][-1]
    epi_poll = re.compile(r"\bs_sleep\b").search(txt, last_mfma).start()
    late = _PLAIN_LOAD.findall(txt[epi_poll:])
    assert not late, late
    # qkvo, the cache, and qkvd in a debug build: 2-byte stores
    shorts = len(re.findall(r"^\s*buffer_store_short\b", txt, re.M))
    assert shorts == (3 if debug else 2), shorts
    if not rec.accesses:  # traced earlier in this process (FlyDSL's own cache)
        return
    assert not rec.unnamed
    acc = {(a.stage, a.region, a.space, a.kind) for a in rec.accesses}
    assert acc == {
        (qkv.STAGE, "ardy", Space.SCRATCH, "poll"),
        (qkv.STAGE, "qpart", Space.SCRATCH, "put"),
        (epilogue.STAGE, "qpart", Space.SCRATCH, "poll"),
    }, acc
    assert rec.stages == [qkv.STAGE, epilogue.STAGE]


@compiles
@pytest.mark.parametrize("rope", [False, True])
def test_bulk_probe_compiles(rope, tmp_path, monkeypatch):
    txt, _ = _compile(_bulk_probe(rope), tmp_path, monkeypatch)
    code = _code(txt)
    assert len(_fmas(code)) == BULK_P * (4 if rope else 1)
    assert _sub_of_two_muls(code) == rope


# ---------------------------------------------------------------- host side


def _stock(rows, positions, table, slots, kv):
    """The stock fused insert in place on ``rows`` (bf16 [N, 2304]) and
    ``kv`` (the decode instance at N <= 8: HPG 1)."""
    from vllm import _custom_ops as ops

    ops.fused_muse_spark_qknorm_rope_kv_insert(
        rows,
        positions,
        spec.HEADS,
        spec.KV_HEADS,
        spec.HEAD_DIM,
        spec.RMS_EPS,
        table,
        spec.HEAD_DIM if table is not None else 0,
        slots,
        kv,
        spec.KV_BLOCK,
    )


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int16)


def _publish(bufs, arow: torch.Tensor) -> None:
    """What K1's boundary CTAs leave before the GEMV reads: the ``arow``
    rows (plain, in the probe scratch) and every slice's ``ardy`` flag with
    this launch's tag."""
    buf, off = bufs.where(spec.AROW)
    raw = arow.contiguous().view(torch.uint8).reshape(-1)
    buf[off : off + raw.numel()].copy_(raw)
    bufs.write_pairs("ardy", 0, torch.ones(spec.NA, dtype=torch.int32))


def _qkvd(bufs, s: int) -> torch.Tensor:
    buf, off = bufs.where(spec.QKVD)
    return buf[off : off + MT * QKV * 2].view(bf).view(MT, QKV)[:s].clone()


def _kv_args(kv: torch.Tensor) -> dict:
    """The cache's K1 arguments (byte strides, as ``runtime.layers``)."""
    e = kv.element_size()
    return {
        "kv": kv.data_ptr(),
        "kv_sb": kv.stride(0) * e,
        "kv_sh": kv.stride(1) * e,
        "kv_st": kv.stride(2) * e,
        "kv_blocks": kv.shape[0],
    }


def _front_case(s: int, seed: int, blocks: int):
    """(w_qkv bf16 [2304, 8192], arow bf16 [s, 8192], positions int64 [s],
    slots int64 [s], pad bool [16], live) on the CPU. Rows: random, one row
    scaled to the qk-norm's eps (its heads' mean square ~ 1e-5); row 0's slot
    at n = 15 of a block and row 1's at n = 0 of the next (a new block);
    pad rows (a third of them at s >= 3) with slot -1; distinct slots."""
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(QKV, H, generator=g) * H**-0.5).to(bf)
    arow = torch.randn(s, H, generator=g)
    if s > 2:
        arow[2] *= 3e-3
    blk = torch.randperm(blocks - 2, generator=g)[: s + 1] + 1
    n = torch.randint(0, spec.KV_BLOCK, (s,), generator=g)
    slots = blk[:s] * spec.KV_BLOCK + n
    slots[0] = blk[s] * spec.KV_BLOCK + spec.KV_BLOCK - 1
    if s > 1:
        slots[1] = (blk[s] + 1) * spec.KV_BLOCK
    positions = torch.randint(0, MAX_POS, (s,), generator=g)
    positions[0] = MAX_POS - 1
    pad = torch.zeros(PAD_BYTES, dtype=torch.bool)
    for t in range(s):
        pad[t] = s >= 3 and t % 3 == 2
    arow[pad[:s]] = 0  # the boundary's pad rows
    slots[pad[:s]] = reference.PAD_SLOT
    return w, arow.to(bf), positions, slots, pad, ~pad[:s]


def test_front_cases():
    """The GPU cases hold what they claim: distinct live slots, slot -1 on
    every pad row, n = 15 and n = 0 present, a position at max_pos - 1, an
    eps-scale row."""
    for s in WIDTHS:
        w, arow, positions, slots, pad, live = _front_case(s, 100 + s, 4096)
        ls = slots[live].tolist()
        assert len(set(ls)) == len(ls) and all(0 < x < 4096 * 16 for x in ls)
        assert (slots[~live] == -1).all()
        assert (slots % spec.KV_BLOCK == spec.KV_BLOCK - 1).any()
        if s > 1:
            assert (slots % spec.KV_BLOCK == 0).any()
        assert positions.max().item() == MAX_POS - 1
        assert live.any() and (s < 3 or (~live).any())
        if s > 2:
            ms = arow[2].float().pow(2).mean().item()
            assert ms < 1e-4


def _bulk_case(seed: int):
    """(x bf16 [1024, 2304], positions int64 [1024]): q / k heads over 17
    binades, eps-scale heads, all-zero heads, one huge element; positions 0,
    max_pos - 1 and random."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(BULK_TOKENS, QKV, generator=g)
    heads = x[:, : spec.ROW_V].reshape(-1, 64)
    heads *= torch.exp2(torch.randint(-8, 9, (heads.shape[0], 1), generator=g).float())
    heads[0::37] = torch.randn(heads[0::37].shape, generator=g) * 3e-3
    heads[5::101] = 0
    heads[9::113, 17] = 3e4
    x[:, : spec.ROW_V] = heads.reshape(BULK_TOKENS, -1)
    positions = torch.randint(0, MAX_POS, (BULK_TOKENS,), generator=g)
    positions[0], positions[1] = 0, MAX_POS - 1
    return x.to(bf), positions


# ---------------------------------------------------------------- [1GPU]


@needs_gfx950
@pytest.mark.parametrize("rope", [False, True])
def test_epilogue_math_matches_the_fused_insert(rope):
    """The bulk probe's q / k heads (1.18 M elements a launch) ``torch.equal``
    the stock insert's (HPG 1, slots -1) on the same rows; v and og are not
    touched."""
    dev = torch.device("cuda")
    table = reference.rope_table(MAX_POS, device=dev) if rope else None
    probe = _bulk_probe(rope)
    bufs = context.ProbeBuffers(slot=SLOT)
    for seed in (1, 2):
        x, positions = _bulk_case(seed)
        x, positions = x.to(dev), positions.to(dev)
        want = x.clone()
        slots = torch.full((BULK_TOKENS,), -1, dtype=torch.int64, device=dev)
        kv = torch.zeros(4, spec.KV_HEADS, spec.KV_BLOCK, spec.KV_C, dtype=bf)
        _stock(want, positions, table, slots, kv.to(dev))
        out = torch.full_like(x, float("nan"))
        probe(
            bufs,
            x=x,
            positions=positions,
            cos_sin=table if rope else 0,
            max_pos=MAX_POS if rope else 0,
            out=out,
        )
        torch.accelerator.synchronize()
        got, ref = out[:, : spec.ROW_V].cpu(), want[:, : spec.ROW_V].cpu()
        same = (_bits(got) == _bits(ref)).all(1)
        assert same.all(), f"rope={rope}: {(~same).sum().item()} tokens differ"
        assert got.numel() >= 10**6
        host = table.cpu() if table is not None else None
        exp = reference.front(x.cpu(), positions.cpu(), host)
        frac = (_bits(exp[:, : spec.ROW_V]) == _bits(got)).float().mean().item()
        print(f"rope={rope} seed={seed}: B0's reference equal on {frac:.6f}")


@needs_gfx950
@pytest.mark.parametrize("rope", [False, True])
def test_stock_decode_instance_equals_the_prefill_one(rope):
    """Stock vs stock (E4): 8 rows alone (HPG 1) and the same rows inside a
    3072-token call (HPG 4; NoPE 2048 tokens, HPG 2) give equal q / k bytes
    and equal cache bytes at their slots: prefill- and decode-written K
    agree, so K1 matching the decode instance is enough."""
    dev = torch.device("cuda")
    big = 3072 if rope else 2048
    assert _stock_hpg(big, rope) > 1 and _stock_hpg(8, rope) == 1
    g = torch.Generator().manual_seed(21)
    x = (torch.randn(big, QKV, generator=g) * 2).to(bf).to(dev)
    positions = torch.randint(0, MAX_POS, (big,), generator=g).to(dev)
    slots = torch.randperm(big * 2, generator=g)[:big].to(dev)
    table = reference.rope_table(MAX_POS, device=dev) if rope else None
    blocks = -(-big * 2 // spec.KV_BLOCK)
    kv_big = torch.zeros(blocks, spec.KV_HEADS, spec.KV_BLOCK, spec.KV_C, dtype=bf)
    kv_big = kv_big.to(dev)
    kv_small = torch.zeros_like(kv_big)
    all_rows = x.clone()
    _stock(all_rows, positions, table, slots, kv_big)
    rows8 = x[:8].clone()
    _stock(rows8, positions[:8].clone(), table, slots[:8].clone(), kv_small)
    torch.accelerator.synchronize()
    assert torch.equal(_bits(rows8), _bits(all_rows[:8]))
    flat_b = _by_slot(kv_big).index_select(0, slots[:8])
    flat_s = _by_slot(kv_small).index_select(0, slots[:8])
    assert flat_s.abs().sum() > 0
    assert torch.equal(_bits(flat_s), _bits(flat_b))


def _by_slot(kv: torch.Tensor) -> torch.Tensor:
    """A [blocks, heads, 16, 128] cache as [slots, heads, 128]."""
    return kv.permute(0, 2, 1, 3).reshape(-1, spec.KV_HEADS, spec.KV_C)


def _launch_front(probe, bufs, w, arow, positions, slots, pad, table, kv, qkvo):
    _publish(bufs, arow)
    probe(
        bufs,
        w_qkv=w,
        positions=positions,
        slots=slots,
        pad=pad,
        cos_sin=table if table is not None else 0,
        **_kv_args(kv),
        max_pos=table.shape[0] if table is not None else 0,
        qkvo=qkvo,
    )
    torch.accelerator.synchronize()


def _next_step(bufs) -> None:
    """The runner's per-step epoch bump: the next launch's tags are new."""
    bufs.epoch += 1


@needs_gfx950
@pytest.mark.parametrize("rope", [False, True])
@pytest.mark.parametrize("s", WIDTHS)
def test_front_matches_stock(s, rope):
    """9.2: ``qkvd`` (the GEMV rows) vs fp64 and ``F.linear``; ``qkvo`` live
    rows and the whole cache ``torch.equal`` the stock insert run on
    ``qkvd`` (same positions, slots, cache contents); pad rows zero; a
    second launch (next epoch) bit-identical."""
    dev = torch.device("cuda")
    probe = _front_probe(s, rope)
    bufs = context.ProbeBuffers(slot=SLOT)
    table = reference.rope_table(MAX_POS, device=dev) if rope else None
    blocks = 4096
    w, arow, positions, slots, pad, live = _front_case(s, 100 + s, blocks)
    w, arow = w.to(dev), arow.to(dev)
    positions, slots, pad = positions.to(dev), slots.to(dev), pad.to(dev)
    gk = torch.Generator().manual_seed(5)
    kv0 = reference.kv_cache(blocks, null=0.0, generator=gk).to(dev)
    kv = kv0.clone()
    qkvo = torch.full((s, QKV), float("nan"), dtype=bf, device=dev)
    _launch_front(probe, bufs, w, arow, positions, slots, pad, table, kv, qkvo)
    assert not bufs.failed(), bufs.given_up_waits()
    rows = _qkvd(bufs, s)
    # the GEMV: vs fp64, vs F.linear (wvSplitK / hipBLASLt order)
    ref64 = (arow.double() @ w.double().T).to(bf)
    reference.check_gemv(rows, ref64, arow, w, what=f"qkvd s={s}")
    lin = torch.nn.functional.linear(arow, w)
    eq = (_bits(rows) == _bits(lin)).float().mean().item()
    print(f"s={s} rope={rope}: qkvd == F.linear on {eq:.6f}")
    assert eq >= 0.99
    # the epilogue: the stock insert on the kernel's own rows
    stock = rows.clone()
    kv_stock = kv0.clone()
    _stock(stock, positions, table, slots, kv_stock)
    torch.accelerator.synchronize()
    got = qkvo.cpu()
    lv = live.cpu()
    assert torch.equal(_bits(got[lv]), _bits(stock.cpu()[lv]))
    assert (_bits(got[~lv]) == 0).all()
    assert torch.equal(_bits(kv), _bits(kv_stock))
    assert not torch.equal(_bits(kv), _bits(kv0)) and lv.any()
    # determinism: the next step's launch on the same inputs
    _next_step(bufs)
    kv2 = kv0.clone()
    qkvo2 = torch.full_like(qkvo, float("nan"))
    _launch_front(probe, bufs, w, arow, positions, slots, pad, table, kv2, qkvo2)
    assert not bufs.failed(), bufs.given_up_waits()
    assert torch.equal(_bits(qkvo2), _bits(qkvo)) and torch.equal(_bits(kv2), _bits(kv))
    assert torch.equal(_bits(_qkvd(bufs, s)), _bits(rows))


def _edge_cache(layout: str, dev):
    """(kv, its blocks, a block id to test): "LBHNC" / "LHBNC" 4096 blocks
    (B0's ``kv_cache``), "big" a > 4 GiB LBHNC cache whose tested block
    starts past 2^32 B."""
    if layout == "big":
        per = spec.KV_HEADS * spec.KV_BLOCK * spec.KV_C * 2  # 8 KiB a block
        blocks = (1 << 32) // per + 64
        dims = (blocks, spec.KV_HEADS, spec.KV_BLOCK, spec.KV_C)
        kv = torch.zeros(dims, dtype=bf, device=dev)
        assert (blocks - 3) * per > 1 << 32
        return kv, blocks, blocks - 3
    gk = torch.Generator().manual_seed(9)
    kv = reference.kv_cache(4096, layout=layout, null=0.0, generator=gk).to(dev)
    return kv, 4096, 4000


@needs_gfx950
@pytest.mark.parametrize("layout", ["LBHNC", "LHBNC", "big"])
def test_front_cache_edges(layout):
    """S = 8, RoPE, on one cache layout: (1) every slot -1 -> the cache
    byte-identical, qkvo still the stock rows; (2) slots at n = 15 / n = 0 of
    adjacent blocks (one past 2^32 B in "big"), positions at max_pos - 1, at
    max_pos and far past it, negative (clamped to the table, compared with
    the stock insert on the clamped positions), and a garbage slot (block
    past ``kv_blocks``) that writes nothing; (3) FAILED set -> the launch
    returns and the cache is byte-identical."""
    dev = torch.device("cuda")
    s = 8
    probe = _front_probe(s, True)
    bufs = context.ProbeBuffers(slot=SLOT)
    table = reference.rope_table(MAX_POS, device=dev)
    kv, blocks, b = _edge_cache(layout, dev)
    w, arow, positions, slots, pad, live = _front_case(s, 77, 64)
    pad[:] = False
    w, arow, pad = w.to(dev), arow.to(dev), pad.to(dev)
    # (1) slot -1 everywhere
    none = torch.full((s,), -1, dtype=torch.int64, device=dev)
    kv0 = kv.clone()
    qkvo = torch.full((s, QKV), float("nan"), dtype=bf, device=dev)
    _launch_front(probe, bufs, w, arow, positions.to(dev), none, pad, table, kv, qkvo)
    assert not bufs.failed(), bufs.given_up_waits()
    assert torch.equal(_bits(kv), _bits(kv0))
    stock = _qkvd(bufs, s)
    _stock(stock, positions.to(dev), table, none, kv0.clone())
    assert torch.equal(_bits(qkvo), _bits(stock))
    # (2) edge slots and positions
    _next_step(bufs)
    slots = torch.tensor(
        [
            b * 16 + 15,
            (b + 1) * 16,
            (b + 1) * 16 + 7,
            b * 16 + 3,
            (b - 1) * 16 + 9,
            blocks * 16 + 5,  # block past kv_blocks: no write
            1 << 40,  # garbage: no write
            (b - 2) * 16,
        ],
        dtype=torch.int64,
    )
    pos = torch.tensor([MAX_POS - 1, MAX_POS, MAX_POS + 12345, 1 << 40, -5, 7, 0, 2047])
    clamped = pos.clamp(0, MAX_POS - 1)
    slots, pos, clamped = slots.to(dev), pos.to(dev), clamped.to(dev)
    kv0 = kv.clone()
    qkvo = torch.full((s, QKV), float("nan"), dtype=bf, device=dev)
    _launch_front(probe, bufs, w, arow, pos, slots, pad, table, kv, qkvo)
    assert not bufs.failed(), bufs.given_up_waits()
    stock = _qkvd(bufs, s)
    kv_stock = kv0.clone()
    ok = (slots // 16) < blocks
    stock_slots = torch.where(ok, slots, torch.full_like(slots, -1))
    _stock(stock, clamped, table, stock_slots, kv_stock)
    torch.accelerator.synchronize()
    assert torch.equal(_bits(qkvo), _bits(stock))
    assert torch.equal(_bits(kv), _bits(kv_stock))
    assert not torch.equal(_bits(kv), _bits(kv0))
    # (3) a FAILED rank writes no K / V (its waits give up at once)
    _next_step(bufs)
    bufs.health[FAILED] = 1
    kv0 = kv.clone()
    _launch_front(probe, bufs, w, arow, pos, slots, pad, table, kv, qkvo)
    assert torch.equal(_bits(kv), _bits(kv0))
    bufs.health[FAILED] = 0
