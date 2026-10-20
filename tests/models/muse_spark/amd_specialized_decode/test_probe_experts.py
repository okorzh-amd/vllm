# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2 phase 3b (``mono/stages/experts.py``): the union of the step's picks,
the MXFP4 gate/up (``ug``: FP4 x FP4 scaled MFMA, aiter's SiLU, bf16 h, the
RoundUp a2 requant) and the down rows pushed to their column owner, as probe
kernels built with ``context.probe_kernel`` (peers aliased: one GPU).

[CPU] The index contracts between the stages (a2 words written by ug and read
by down, a1 words of ``pool.load_xq``, w13 row groups, rsx elements), the LDS
structs, the torch MXFP4 reference against aiter's, and the GPU-less compile
gate of every probe (no private segment, VGPR / LDS budget).
[1GPU] ``test_union_probe``: the union / kof table for random ids, duplicates
across rows, pads (-1), garbage ids (300, -7), all-pad (nu = 0), on LDS
pre-filled with garbage; clamped weight addresses read in range.
``test_ug_probe``: gate / up vs a torch dequant GEMV, h (bf16) vs the exact
SiLU of the kernel's own gate / up, the a2 bytes / codes bit-exact vs the
torch RoundUp reference on the kernel's own h, dead picks never written; the
same launch's down (32 CTAs, one owner) on the kernel's own a2.
``test_down_probe``: down vs a torch dequant GEMV on host-written a2, pad rows
{5, 6, 7} and a garbage id (their a2 pairs stale: a poll would time out),
all-pad: health stays zero, dead picks' rows are exact zeros.
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
from flydsl.expr import gpu, range_constexpr  # noqa: E402

from vllm.models.muse_spark.amd.specialized_decode.mono import spec  # noqa: E402
from vllm.models.muse_spark.amd.specialized_decode.mono.common import (  # noqa: E402
    context,
    route_lds,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.mx import (  # noqa: E402
    scale_index,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (  # noqa: E402
    ld_i32,
    rsrc,
    traced,
    uniform,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.stages import (  # noqa: E402
    experts,
)

THREADS = spec.THREADS
TOPK = spec.TOPK
INTER = spec.INTER
K1 = spec.MOE_HIDDEN  # w13's K, w2's rows
E_T = 64  # experts the ug / down probes allocate (ids < E_T)
OWNER = 5  # the down probes' owner rank: pooled rows 512 OWNER .. + 512
RANK = 3  # the probes' own rank (rsx rows of src RANK)
PAD_ROWS = (5, 6, 7)
DUMP_N = experts.DUMP_WHICH * spec.MAX_U * spec.MAX_TOKENS * INTER
OUT_WORDS = 4096


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


def _st(out, i, v):
    """out[i] := v (plain store; an fp32 by its bits)."""
    bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
    bo.buffer_store(bits, rsrc(out), i)


# ---------------------------------------------------------------- probe LDS


def _xq_lds(s: int):
    """``pool.xq_lds_struct(s)``: the fields ug reads."""

    @fx.struct
    class XqLds:
        xq: fx.Array[fx.Int32, s * spec.XQ_WORDS, 16]
        xsc: fx.Array[fx.Int32, s * spec.XSC_WORDS, 16]
        xss: fx.Array[fx.Float32, s * spec.XSC_WORDS, 16]

    return XqLds


def _ug_lds(s: int) -> dict:
    """K2's composition: ``xq_ug`` (xq then av, both live through ug) and
    ``down`` (sharing the union's bytes)."""
    return {
        "xq_ug": {"xq": _xq_lds(s), "ug": experts.ug_lds_struct(s)},
        "down": experts.down_lds_struct(s),
    }


# ---------------------------------------------------------------- probe bodies


def _probe_rgd(c):
    """A probe's down row group: CTA b + the int32 at ``rgd0`` (32 CTAs push
    one owner's columns; aliased peers would make the owners collide)."""
    return fx.Int32(c["bid"]) + uniform(ld_i32(c["args"]["rgd0"], 0))


@traced
def _junk_words(ptr, tid32, words, junk):
    """Words 0 .. ``words`` (<= THREADS) of an LDS field := ``junk``."""
    if tid32 < words:
        fx.ptr_store(junk, ptr + tid32)


def _junk_route_lds(rt, tid32):
    """Every ``RouteLds`` word 0xFFFFFFFF: LDS is not zeroed at launch, and a
    stage that reads a word it did not write first reads garbage."""
    for name in route_lds.FIELDS:
        words = spec.ROUTE_LDS[name]
        assert words <= THREADS, (name, words)
        junk = fx.Int32(-1)
        if name == "tot":
            junk = junk.bitcast(fx.Float32)
        _junk_words(rt[name], tid32, words, junk)


@traced
def _union(c):
    """One CTA: the route table and union on LDS full of garbage; the clamped
    expert of slots -2 .. MAX_U + 1; every slot's ug / down weight loads on
    full-size weights (an address out of range faults)."""
    tid, rt = c["tid"], c["rt"]
    out = c["args"]["out"]
    tid32 = fx.Int32(tid)
    _junk_route_lds(rt, tid32)
    gpu.barrier()
    nu = experts.load_route(c)
    if tid == 0:
        _st(out, 0, nu)
    if tid < c["S"] * 16:
        _st(out, 16 + tid32, fx.ptr_load(rt["route"] + tid32))
    if tid < spec.EXPERTS:
        _st(out, 256 + tid32, fx.ptr_load(rt["flag"] + tid32))
    if tid < spec.MAX_U:
        _st(out, 512 + tid32, fx.ptr_load(rt["expert"] + tid32))
    _st(out, 1024 + tid32, fx.ptr_load(rt["kof"] + tid32))
    for sl in range_constexpr(-2, spec.MAX_U + 2):
        e = experts.slot_expert(c, fx.Int32(sl))
        if tid == 0:
            _st(out, 1536 + 2 + sl, e)
    sl = fx.Int32(-2)
    fold = fx.Int32(0)
    while sl < spec.MAX_U + 2:
        # ug tasks are never negative in K2 (task = min(255 - b, last)); a
        # negative one would give a negative row group (``%`` is remsi)
        ops = experts.ug_loads(c, fx.max(sl, 0) * spec.UG_GROUPS + 15)
        dops = experts.down_loads(c, fx.Int32(spec.DOWN_TASKS - 1), nu, sl)
        for i in range_constexpr(experts.HALF_STEPS):
            fold = fold ^ ops[i][0]
        for i in range_constexpr(spec.DOWN_BATCH * spec.W2_STEPS):
            fold = fold ^ dops[i][0]
        sl = sl + 1
    _st(out, 2048 + tid32, fold)


@traced
def _ug(c):
    """The full grid: the a1 rows -> LDS (``load_xq``'s output), the union,
    every ug task; then CTAs 0 .. 31 run down for rows 16 (rgd0 + b) ..."""
    s, tid = c["S"], c["tid"]
    a = c["args"]
    xq = c["lds"]["xq"]
    tid32 = fx.Int32(tid)
    for i in range_constexpr(s * spec.XQ_WORDS // THREADS):
        j = tid32 + THREADS * i
        fx.ptr_store(ld_i32(a["xin"], j), xq["xq"] + j)
    for i in range_constexpr((s * spec.XSC_WORDS + THREADS - 1) // THREADS):
        j = tid32 + THREADS * i
        if j < s * spec.XSC_WORDS:
            fx.ptr_store(ld_i32(a["xscin"], j), xq["xsc"] + j)
    gpu.barrier()
    nu = experts.load_route(c)
    experts.run_ug(c, nu)
    if c["bid"] < spec.RED_TASKS:
        experts.run_down(c, _probe_rgd(c), nu)
    if (c["bid"] == 0) & (tid == 0):
        _st(a["out"], 0, nu)


@traced
def _down(c):
    """32 CTAs: the union, then down for rows 16 (rgd0 + b) .. on a2 pairs the
    host wrote."""
    nu = experts.load_route(c)
    experts.run_down(c, _probe_rgd(c), nu)
    if (c["bid"] == 0) & (c["tid"] == 0):
        _st(c["args"]["out"], 0, nu)


UG_ARGS = (
    "out",
    "w13",
    "w13s",
    "w2",
    "w2s",
    "xin",
    "xscin",
    "rgd0",
    "ug_dump",
    "down_dump",
)
DOWN_ARGS = ("out", "w2", "w2s", "rgd0", "down_dump")
UNION_ARGS = ("out", "w13", "w13s", "w2", "w2s")


def _probe(which: str, S: int):
    if which == "union":
        return context.probe_kernel(
            f"muse_probe_union_s{S}", S, _union, None, UNION_ARGS
        )
    if which == "ug":
        return context.probe_kernel(
            f"muse_probe_ug_s{S}", S, _ug, _ug_lds(S), UG_ARGS, blocks=spec.BLOCKS
        )
    assert which == "down", which
    return context.probe_kernel(
        f"muse_probe_down_s{S}",
        S,
        _down,
        {"down": experts.down_lds_struct(S)},
        DOWN_ARGS,
        blocks=spec.RED_TASKS,
    )


# ---------------------------------------------------------------- torch reference

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)
# RNE on the E2M1 grid: the midpoints, ties to the even code (0.25 -> 0,
# 0.75 -> 1.0, 1.25 -> 1.0, 1.75 -> 2.0, 2.5 -> 2.0, 3.5 -> 4.0, 5 -> 4.0)
_MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
_TIES_UP = (0.75, 1.75, 3.5)


def fp4_values(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., n] of packed E2M1 (element 2i in the low nibble) -> fp64
    [..., 2n]."""
    p = packed.to(torch.int64)
    nib = torch.stack([p & 0xF, p >> 4], -1).flatten(-2)
    mag = _E2M1.to(packed.device)[nib & 7]
    return torch.where((nib & 8) != 0, -mag, mag)


def mx_dequant(packed: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """MXFP4 (uint8 [..., n/2], codes [..., n/32]) -> fp64 [..., n]."""
    x = fp4_values(packed)
    sc = torch.exp2(codes.to(torch.float64) - 127).repeat_interleave(32, -1)
    return x * sc


def code_roundup(amax: torch.Tensor) -> torch.Tensor:
    """The RoundUp E8M0 code of aiter's HIP quant: amax floored at 1e-10,
    times f32(1/6), the biased exponent rounded up (then the kernel's 254
    clamp)."""
    amax = torch.clamp(amax.float(), min=spec.AMAX_FLOOR)
    u = (amax * torch.tensor(1.0 / 6.0, dtype=torch.float32)).view(torch.int32)
    e = (u >> 23) & 0xFF
    return (e + (((u & 0x7FFFFF) != 0) & (e < 0xFF)).int()).clamp(max=254)


def fp4_rne(y: torch.Tensor) -> torch.Tensor:
    """fp32 already divided by its scale -> E2M1 nibbles (RNE, saturating,
    the sign kept)."""
    a = y.abs()
    idx = torch.bucketize(a, _MIDS.to(y.device), right=False)
    for m in _TIES_UP:
        idx = idx + (a == m).to(idx.dtype)
    idx = idx.clamp(max=7)
    return idx | (torch.signbit(y).to(idx.dtype) << 3)


def fold_neg_zero(packed: torch.Tensor) -> torch.Tensor:
    """Packed E2M1 with every -0 nibble (0x8) written as +0 (0x0)."""
    p = packed.to(torch.int32)
    lo, hi = p & 0xF, p >> 4
    lo = torch.where(lo == 8, torch.zeros_like(lo), lo)
    hi = torch.where(hi == 8, torch.zeros_like(hi), hi)
    return (lo | (hi << 4)).to(torch.uint8)


def mx_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 [..., n] -> (packed uint8 [..., n/2], codes int32 [..., n/32]): the
    RoundUp a1 / a2 quant of aiter's HIP ``fused_mx_quant_moe_sort``."""
    g = x.float().unflatten(-1, (-1, 32))
    code = code_roundup(g.abs().amax(-1))
    scale = ((code.to(torch.int32)) << 23).view(torch.float32)
    nib = fp4_rne(g / scale.unsqueeze(-1)).flatten(-2)
    packed = (nib[..., 0::2] | (nib[..., 1::2] << 4)).to(torch.uint8)
    return packed, code.int()


def union_ref(ids: torch.Tensor):
    """The union of the live picks of ids [S, 8]: (nu, flag[256], expert[64],
    kof[64 8]) as ``load_route`` leaves them."""
    ids = ids.cpu()
    live = (ids >= 0) & (ids < spec.EXPERTS)
    u = sorted(set(ids[live].tolist()))
    flag = torch.zeros(spec.EXPERTS, dtype=torch.int32)
    expert = torch.zeros(spec.MAX_U, dtype=torch.int32)
    kof = torch.full((spec.MAX_U * 8,), -1, dtype=torch.int32)
    for slot, e in enumerate(u):
        flag[e] = slot
        expert[slot] = e
    for t in range(ids.shape[0]):
        for k in range(TOPK):
            if live[t, k]:
                kof[u.index(int(ids[t, k])) * 8 + t] = k
    return len(u), flag, expert, kof


def route_words(ids: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The ``route`` pairs' values: ids at t 16 + k, fp32 weight bits at
    t 16 + 8 + k."""
    s = ids.shape[0]
    out = torch.empty(s, 16, dtype=torch.int32)
    out[:, :TOPK] = ids.cpu().to(torch.int32)
    out[:, TOPK:] = w.cpu().float().view(torch.int32)
    return out.reshape(-1)


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.float().to(torch.bfloat16)


def _bf16_ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|a - b| in bf16 ulps (same-sign values; a sign flip counts large)."""
    ia = _bf16(a).view(torch.int16).to(torch.int32)
    ib = _bf16(b).view(torch.int16).to(torch.int32)
    return torch.where((ia ^ ib) < 0, torch.full_like(ia, 1 << 16), (ia - ib).abs())


# ---------------------------------------------------------------- [CPU]


def test_lds_structs():
    """LDS sizes: ug's ``av`` f32[512]; down's interq / intsc / contrib
    (24576 B at S = 8); the probes' LDS fits a CU."""
    for s in (1, 3, 8):
        assert experts.ug_lds_struct(s).__dsl_size_of__() == THREADS * 4
        assert (
            experts.down_lds_struct(s).__dsl_size_of__() == s * 8 * (64 + 16 + 16) * 4
        )
        assert experts.ug_lds_struct(s) is experts.ug_lds_struct(s)
        sizes = context.stage_bytes(_ug_lds(s))
        assert sizes["xq_ug"] == s * (512 + 128 + 128) * 4 + THREADS * 4
        assert context.lds_total_bytes(_ug_lds(s)) <= spec.LDS_BYTES
    assert experts.down_lds_struct(8).__dsl_size_of__() == 24576
    assert context.stage_bytes(_ug_lds(8))["xq_ug"] == 24576 + 2048


def test_pool_xq_struct_matches():
    """``pool.xq_lds_struct`` carries the fields ug reads, sized as the
    probe's stand-in."""
    pool = pytest.importorskip(
        "vllm.models.muse_spark.amd.specialized_decode.mono.stages.pool"
    )
    if not hasattr(pool, "xq_lds_struct"):
        pytest.skip("pool.xq_lds_struct not there yet")
    for s in (1, 8):
        mine = {f.name: f for f in _xq_lds(s).__dsl_field_defs__}
        theirs = {f.name: f for f in pool.xq_lds_struct(s).__dsl_field_defs__}
        for name in ("xq", "xsc"):
            assert name in theirs, name
            assert theirs[name].type_spec.__dsl_size_of__() == (
                mine[name].type_spec.__dsl_size_of__()
            )


def test_index_contracts():
    """The words one stage writes are the words the next reads, in the K order
    of the weight tiles they meet in the MFMA (A lane: row lane % 16, K block
    lane / 16 of 128-K step st, i.e. K 128 st + 32 kb ..)."""
    for st in range(spec.W2_STEPS):
        for kb in range(4):
            for d in range(4):
                # down reads a2 word (st, kb, d): ug wrote it as group g, word w
                word = experts.inter_read_word(5, st, kb, d)
                g, w = divmod(word - 5 * 64, 4)
                assert word == experts.inter_word(5, g, w) == spec.inter_pair(5, g, w)
                assert 32 * g + 8 * w == 128 * st + 32 * kb + 8 * d
                assert experts.isc_index(5, 4 * st + kb) == spec.isc_pair(5, g)
    for st in range(spec.W13_STEPS):
        for kb in range(4):
            for d in range(4):
                # ug reads a1 word (st, kb, d): load_xq stores xq[t 512 + kb' 4 + w]
                # for columns 32 kb' + 8 w ..
                word = experts.xq_read_word(2, st, kb, d) - 2 * spec.XQ_WORDS
                kbp, w = divmod(word, 4)
                assert 32 * kbp + 8 * w == 128 * st + 32 * kb + 8 * d
    rows: dict = {}
    for g in range(spec.UG_GROUPS):
        for wave in range(spec.WAVES):
            rg = experts.ug_row_group(wave, g)
            # waves w and w + 4 share a row group (the two K halves)
            assert rg == experts.ug_row_group(wave % 4, g)
            rows.setdefault(rg, set()).add((g, wave % 4))
            # gate rows of columns 32 g .., up rows 512 + 32 g ..
            assert rg * 16 == (wave % 4 // 2) * INTER + 32 * g + 16 * (wave % 2)
    assert sorted(rows) == list(range(2 * INTER // 16))
    # a 32-row scale block never holds a gate and an up group
    assert all(rg // 2 < 16 for rg in rows if rg < 32)
    for rank in (0, 7):
        for pick in (0, 63):
            for rgd in (0, 37, 255):
                for rp in range(8):
                    cc = (rgd % 32) * 16 + 2 * rp
                    assert experts.rsx_elem(rank, pick, rgd, rp) == 2 * spec.rsx_pair(
                        rank, pick, cc
                    )
    assert experts.W13_EXPERT_DWORDS == 524288 and experts.W2_EXPERT_DWORDS == 262144
    # load_inter polls ISC over ROUTE's base: pair ISC_FROM_ROUTE + i of it is
    # ISC pair i, and a dead pick's ROUTE pair is its id pair
    lay = spec.scratch_layout()
    for i in (0, spec.MAX_PICKS * spec.UG_GROUPS - 1):
        at = lay["route"][0] + 8 * (experts.ISC_FROM_ROUTE + i)
        assert at == lay["isc"][0] + 8 * i
    assert lay["isc"][0] + lay["isc"][1] <= spec.scratch_bytes()


def test_union_reference():
    """The host model the union probe checks against."""
    ids = torch.tensor([[3, 9, 1, -1, 300, 9, -7, 0], [9, 2, 3, 4, 5, 6, 7, 8]])
    nu, flag, expert, kof = union_ref(ids)
    assert nu == 10 and expert[:nu].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert expert[nu:].eq(0).all() and flag[9] == 9 and flag[300 % 256] == 0
    assert kof[9 * 8 + 1] == 0 and kof[3 * 8 + 0] == 0 and kof[2 * 8 + 0] == -1
    assert union_ref(torch.full((8, 8), -1))[0] == 0


def test_mx_reference_matches_aiter():
    """The torch MXFP4 reference (RoundUp codes, RNE E2M1, low nibble first)
    against aiter's ``f32_to_mxfp4`` / ``mxfp4_to_f32`` on 1e5 groups incl.
    every tie, saturation, zeros and denormal-scale groups."""
    from aiter.utility.fp4_utils import f32_to_mxfp4, mxfp4_to_f32

    g = torch.Generator().manual_seed(3)
    x = torch.randn(100_000, 32, generator=g) * torch.exp2(
        torch.randint(-30, 30, (100_000, 1), generator=g).float()
    )
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 0.0, -0.25])
    x[:2] = 0.0
    x[0, : ties.numel()] = ties  # amax 6: scale 1, the ties exact
    x[1, : ties.numel()] = -ties
    x[2] = 0.0
    x[3] = 1e-40  # amax below the floor
    x = x.to(torch.bfloat16).float()  # a2 is bf16 (h), a1 too
    packed, codes = mx_quant(x.reshape(-1, 64)[:, :64])
    scale = torch.exp2(codes.double() - 127).repeat_interleave(32, -1)
    want = f32_to_mxfp4((x.reshape(-1, 64).double() / scale).float())
    want = want.view(torch.uint8)
    # aiter keeps -0 (0x8) as we do; compare bytes exactly
    assert torch.equal(packed, want)
    back = mxfp4_to_f32(packed.clone()).double() * scale
    assert torch.equal(back, mx_dequant(packed, codes))
    assert codes.reshape(-1)[0].item() == 127  # amax 6 -> scale 1
    # an all-zero group and one below the 1e-10 floor: code 92
    assert codes.view(-1)[2].item() == 92 and codes.view(-1)[3].item() == 92


_WATERFALL = re.compile(r"v_cmp_eq_u64_e\d+\s+\S+,\s*s\[\d+:\d+\],\s*v\[")


def _isa(dump: str, name: str) -> tuple[str, dict]:
    """The final ISA of kernel ``name`` under the dump dir and its resources."""
    for path in glob.glob(f"{dump}/**/*final_isa.s", recursive=True):
        with open(path) as f:
            txt = f.read()
        if re.search(rf"\.amdhsa_kernel {re.escape(name)}\b", txt):
            res = {}
            for k in (
                "next_free_vgpr",
                "accum_offset",
                "next_free_sgpr",
                "group_segment_fixed_size",
                "private_segment_fixed_size",
            ):
                m = re.search(rf"\.amdhsa_{k}\s+(\d+)", txt)
                res[k] = int(m.group(1)) if m else None
            return txt, res
    raise AssertionError(f"no final ISA of {name} under {dump}")


@compiles
@pytest.mark.parametrize(
    "which,S", [("union", 8), ("ug", 1), ("ug", 8), ("down", 1), ("down", 8)]
)
def test_probe_compiles(which, S, tmp_path, monkeypatch):
    """The GPU-less compile gate (README, Residency): no private segment (a spilling
    persistent kernel can lose residency and deadlock), VGPR and LDS budget;
    the ug probe is the whole expert phase (union, ug, down) of K2."""
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    probe = _probe(which, S)
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
    if which in ("ug", "down"):
        assert "v_mfma_scale_f32_16x16x128_f8f6f4" in txt
        # no readfirstlane waterfall (a 64-bit compare of a VGPR buffer
        # resource against its first lane's): every load_inter poll keeps a
        # wave-uniform base
        loop = _WATERFALL.search(txt)
        assert loop is None, loop
    if which == "ug":
        assert "v_cvt_scalef32_pk_fp4_f32" in txt
        # aiter's SiLU: hw exp2 and rcp, no IEEE divide
        assert "v_exp_f32" in txt and "v_rcp_f32" in txt
        assert "v_div_fixup_f32" not in txt


# ---------------------------------------------------------------- [1GPU]


def _weights(n_exp: int, seed: int):
    """Random post-load expert weights for ``n_exp`` experts (the per-rank
    shapes): fp4 bytes, ``shuffle_weight(16, 16)``; e8m0 codes 118 .. 122,
    ``e8m0_shuffle``d on the [E N, K / 32] view. -> (w13, w13s, w2, w2s, and
    the unshuffled bytes and codes)."""
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    g = torch.Generator(device="cuda").manual_seed(seed)
    w13 = torch.randint(
        0, 256, (n_exp, 2 * INTER, K1 // 2), dtype=torch.uint8, device="cuda",
        generator=g,
    )  # fmt: skip
    w2 = torch.randint(
        0, 256, (n_exp, K1, INTER // 2), dtype=torch.uint8, device="cuda", generator=g
    )
    c13 = torch.randint(
        118, 123, (n_exp * 2 * INTER, K1 // 32), dtype=torch.uint8, device="cuda",
        generator=g,
    )  # fmt: skip
    c2 = torch.randint(
        118, 123, (n_exp * K1, INTER // 32), dtype=torch.uint8, device="cuda",
        generator=g,
    )  # fmt: skip

    def shuf_scale(c):
        src = c.view(torch.float8_e8m0fnu) if hasattr(torch, "float8_e8m0fnu") else c
        return e8m0_shuffle(src).view(torch.uint8).reshape(-1).contiguous()

    return (
        shuffle_weight(w13, layout=(16, 16)).contiguous(),
        shuf_scale(c13),
        shuffle_weight(w2, layout=(16, 16)).contiguous(),
        shuf_scale(c2),
        (w13, c13.view(n_exp, 2 * INTER, -1), w2, c2.view(n_exp, K1, -1)),
    )


def _route(S: int, n_exp: int, seed: int, pads=(), garbage=()):
    """Top-8 ids (distinct a row, duplicates across rows) of ``n_exp`` experts,
    pad rows -1, ``garbage`` = [(t, k, id)]; weights random."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(n_exp, generator=g)[:TOPK] for _ in range(S)])
    for t in pads:
        if t < S:
            ids[t] = -1
    for t, k, e in garbage:
        if t < S:
            ids[t, k] = e
    w = torch.rand(S, TOPK, generator=g)
    return ids.to(torch.int32), w


def _live_picks(ids):
    """[(pick, t, k, e)] of the live picks."""
    out = []
    for t in range(ids.shape[0]):
        for k in range(TOPK):
            e = int(ids[t, k])
            if 0 <= e < spec.EXPERTS:
                out.append((t * TOPK + k, t, k, e))
    return out


def _pairs(bufs, region: str, index: int, n: int):
    """(values, tags) of ``n`` pairs of ``region``: contiguous host copies."""
    vals, tags = bufs.read_pairs(region, index, n)
    return vals.contiguous().cpu(), tags.contiguous().cpu()


def _stale(bufs, region: str, n: int) -> int:
    """Pre-fill ``n`` pairs of ``region`` with a NaN payload and last step's
    tag; returns that tag (a pair a poll would never accept)."""
    tag = bufs.tag() - spec.TAG_SLOTS
    bufs.write_pairs(
        region, 0, torch.full((n,), 0x7FC00001, dtype=torch.int32), tag=tag
    )
    return tag


def _gemv_check(got, ref, absum):
    """fp32 accumulation of exact products vs the fp64 dot: within 1e-6 of the
    terms' absolute sum (the fp32 rounding scale of the partial sums)."""
    err = (got.double() - ref).abs()
    bound = 1e-6 * absum + 1e-30
    assert (err <= bound).all(), (err / (absum + 1e-30)).max().item()


def _down_check(bufs, ids, a2deq, w2u, c2u, dump, S):
    """The down rows of the owner OWNER's 512 columns: fp32 vs the dequant
    GEMV, the pushed bf16 == bf16 of the kernel's own fp32, vs the reference
    on >= 99.9 %, dead picks exact zeros, every pushed pair tagged."""
    tag = bufs.tag()
    rows = slice(OWNER * 512, OWNER * 512 + 512)
    live = {p: (t, k, e) for p, t, k, e in _live_picks(ids)}
    matched = total = 0
    for pick in range(S * TOPK):
        vals, tags = _pairs(bufs, "rsx", spec.rsx_pair(RANK, pick, 0), 256)
        assert (tags == tag).all(), pick
        pushed = vals.to(torch.int32).view(torch.bfloat16).float()
        if pick not in live:
            assert torch.equal(pushed, torch.zeros(512)), pick
            continue
        e = live[pick][2]
        wdeq = mx_dequant(w2u[e, rows], c2u[e, rows])  # [512, 512]
        terms = wdeq * a2deq[pick].unsqueeze(0)
        ref = terms.sum(-1)
        got = dump[pick, rows]
        _gemv_check(got, ref, terms.abs().sum(-1))
        assert torch.equal(pushed, _bf16(got.cpu()).float()), pick
        matched += (_bf16(ref.cpu()) == _bf16(got.cpu())).sum().item()
        total += 512
    assert total == 0 or matched >= 0.999 * total, (matched, total)


@needs_gfx950
@pytest.mark.parametrize("case", ["mixed", "all_pad", "one_row"])
def test_union_probe(case):
    S = 1 if case == "one_row" else 8
    probe = _probe("union", S)
    bufs = context.ProbeBuffers(budget=2, rank=RANK)
    if case == "all_pad":
        ids = torch.full((S, TOPK), -1, dtype=torch.int32)
        w = torch.zeros(S, TOPK)
    else:
        ids, w = _route(
            S, 40, seed=11, pads=PAD_ROWS, garbage=[(1, 2, 300), (3, 7, -7)]
        )
    bufs.write_pairs("route", 0, route_words(ids, w))
    # full-size weights: a clamped id's tiles are inside them, others fault
    w13 = torch.empty(spec.EXPERTS * experts.W13_EXPERT_DWORDS * 4, dtype=torch.uint8,
                      device="cuda")  # fmt: skip
    w2 = torch.empty(spec.EXPERTS * experts.W2_EXPERT_DWORDS * 4, dtype=torch.uint8,
                     device="cuda")  # fmt: skip
    w13s = torch.empty(spec.EXPERTS * 1024 * 128, dtype=torch.uint8, device="cuda")
    w2s = torch.empty(spec.EXPERTS * K1 * 16, dtype=torch.uint8, device="cuda")
    out = torch.zeros(OUT_WORDS, dtype=torch.int32, device="cuda")
    probe(bufs, out=out, w13=w13, w13s=w13s, w2=w2, w2s=w2s)
    torch.accelerator.synchronize()
    assert not bufs.failed(), bufs.given_up_waits()
    o = out.cpu()
    nu, flag, expert, kof = union_ref(ids)
    assert o[0].item() == nu
    assert torch.equal(o[16 : 16 + S * 16], route_words(ids, w))
    union = expert[:nu]
    got_flag = o[256:512]
    assert torch.equal(got_flag[union.long()], torch.arange(nu, dtype=torch.int32))
    others = torch.ones(spec.EXPERTS, dtype=torch.bool)
    others[union.long()] = False
    assert got_flag[others].eq(0).all()  # no garbage flag survived
    assert torch.equal(o[512 : 512 + spec.MAX_U], expert)  # zeroed past nu
    assert torch.equal(o[1024:1536], kof)
    # the clamped slot -> expert of slots -2 .. 65: in range, U's below nu
    se = o[1536 : 1536 + spec.MAX_U + 4]
    want = torch.cat([expert[:1], expert[:1], expert, expert[-1:], expert[-1:]])
    assert torch.equal(se, want)
    assert ((se >= 0) & (se < spec.EXPERTS)).all()


@needs_gfx950
@pytest.mark.parametrize("S", [1, 3, 8])
def test_ug_probe(S):
    probe = _probe("ug", S)
    bufs = context.ProbeBuffers(budget=2, rank=RANK)
    w13, w13s, w2, w2s, (w13u, c13u, w2u, c2u) = _weights(E_T, seed=5)
    ids, w = _route(S, E_T, seed=S, pads=PAD_ROWS, garbage=[(1, 2, 300)])
    bufs.write_pairs("route", 0, route_words(ids, w))
    stale = _stale(bufs, "inter", spec.MAX_PICKS * 64)
    _stale(bufs, "isc", spec.MAX_PICKS * 16)
    # the a1 rows (load_xq's output): random fp4 and codes, pad rows zero
    g = torch.Generator(device="cuda").manual_seed(100 + S)
    xb = torch.randint(0, 256, (S, K1 // 2), dtype=torch.uint8, device="cuda",
                       generator=g)  # fmt: skip
    xc = torch.randint(120, 128, (S, K1 // 32), dtype=torch.int32, device="cuda",
                       generator=g)  # fmt: skip
    for t in PAD_ROWS:
        if t < S:
            xb[t] = 0
            xc[t] = 92
    rgd0 = torch.tensor([OWNER * 32], dtype=torch.int32, device="cuda")
    ug_dump = torch.zeros(DUMP_N, dtype=torch.float32, device="cuda")
    down_dump = torch.zeros(spec.MAX_PICKS, K1, dtype=torch.float32, device="cuda")
    out = torch.zeros(OUT_WORDS, dtype=torch.int32, device="cuda")
    probe(
        bufs,
        out=out,
        w13=w13,
        w13s=w13s,
        w2=w2,
        w2s=w2s,
        xin=xb.view(torch.int32),
        xscin=xc,
        rgd0=rgd0,
        ug_dump=ug_dump,
        down_dump=down_dump,
    )
    torch.accelerator.synchronize()
    assert not bufs.failed(), bufs.given_up_waits()
    nu, _, expert, _ = union_ref(ids)
    assert out[0].item() == nu
    tag = bufs.tag()
    a1 = mx_dequant(xb, xc)  # [S, 4096]
    dump = ug_dump.view(experts.DUMP_WHICH, spec.MAX_U, spec.MAX_TOKENS, INTER)
    gate, up, h = dump[0], dump[1], dump[2]
    # gate / up of every (slot, row): fp32 MFMA accumulation of exact products
    for slot in range(nu):
        e = int(expert[slot])
        wdeq = mx_dequant(w13u[e], c13u[e])  # [1024, 4096]
        terms_abs = wdeq.abs() @ a1.abs().T  # [1024, S]
        ref = wdeq @ a1.T
        _gemv_check(gate[slot, :S].T, ref[:INTER], terms_abs[:INTER])
        _gemv_check(up[slot, :S].T, ref[INTER:], terms_abs[INTER:])
    # h: bf16((g sigmoid(g)) u) of the kernel's own g, u (hw exp2 / rcp)
    gk = gate[:nu, :S].double()
    hk = h[:nu, :S]
    href = _bf16((gk * torch.sigmoid(gk)) * up[:nu, :S].double())
    ulps = _bf16_ulps(hk, href.float())
    assert ulps.max().item() <= 1
    assert (ulps == 0).float().mean().item() >= 0.999
    assert torch.equal(hk, _bf16(hk).float())  # h is bf16
    # a2: bit-exact RoundUp of the kernel's own h, live picks only
    live = _live_picks(ids)
    slot_of = {int(e): i for i, e in enumerate(expert[:nu])}
    a2deq = torch.zeros(spec.MAX_PICKS, INTER, dtype=torch.float64, device="cuda")
    for pick, t, _k, e in live:
        want_b, want_c = mx_quant(h[slot_of[e], t])
        vals, tags = _pairs(bufs, "inter", pick * 64, 64)
        assert (tags == tag).all(), pick
        got_b = vals.to(torch.int32).view(torch.uint8).cuda()
        # bit-exact; the one open HW fact is whether the cvt keeps
        # the sign of a negative value that rounds to 0 (0x8, as the reference)
        # or writes +0: either is stock's (same instruction), a mix is not
        assert torch.equal(got_b, want_b) or torch.equal(
            got_b, fold_neg_zero(want_b)
        ), pick
        cv, ct = _pairs(bufs, "isc", pick * 16, 16)
        assert (ct == tag).all() and torch.equal(cv.cuda(), want_c), pick
        a2deq[pick] = mx_dequant(got_b, cv.cuda())
    dead = sorted(set(range(S * TOPK)) - {p for p, *_ in live})
    for pick in dead:  # pads, the garbage id: never written
        assert (_pairs(bufs, "inter", pick * 64, 64)[1] == stale).all(), pick
        assert (_pairs(bufs, "isc", pick * 16, 16)[1] == stale).all(), pick
    # the same launch's down (owner OWNER's columns) on the kernel's own a2
    _down_check(bufs, ids, a2deq, w2u, c2u, down_dump, S)


@needs_gfx950
@pytest.mark.parametrize("case", ["pads", "all_live", "all_pad"])
def test_down_probe(case):
    S = 8
    probe = _probe("down", S)
    bufs = context.ProbeBuffers(budget=1, rank=RANK)  # a dead poll fails fast
    _, _, w2, w2s, (_, _, w2u, c2u) = _weights(E_T, seed=9)
    if case == "all_pad":
        ids = torch.full((S, TOPK), -1, dtype=torch.int32)
        w = torch.zeros(S, TOPK)
    else:
        pads = PAD_ROWS if case == "pads" else ()
        garbage = [(2, 4, 300), (0, 1, -7)] if case == "pads" else []
        ids, w = _route(S, E_T, seed=21, pads=pads, garbage=garbage)
    bufs.write_pairs("route", 0, route_words(ids, w))
    _stale(bufs, "inter", spec.MAX_PICKS * 64)
    _stale(bufs, "isc", spec.MAX_PICKS * 16)
    g = torch.Generator(device="cuda").manual_seed(31)
    a2b = torch.randint(0, 256, (spec.MAX_PICKS, INTER // 2), dtype=torch.uint8,
                        device="cuda", generator=g)  # fmt: skip
    a2c = torch.randint(118, 130, (spec.MAX_PICKS, INTER // 32), dtype=torch.int32,
                        device="cuda", generator=g)  # fmt: skip
    a2deq = mx_dequant(a2b, a2c)
    for pick, *_ in _live_picks(ids):  # ug's pushes, live picks only
        bufs.write_pairs("inter", pick * 64, a2b[pick].view(torch.int32))
        bufs.write_pairs("isc", pick * 16, a2c[pick])
    rgd0 = torch.tensor([OWNER * 32], dtype=torch.int32, device="cuda")
    down_dump = torch.zeros(spec.MAX_PICKS, K1, dtype=torch.float32, device="cuda")
    out = torch.zeros(OUT_WORDS, dtype=torch.int32, device="cuda")
    probe(bufs, out=out, w2=w2, w2s=w2s, rgd0=rgd0, down_dump=down_dump)
    torch.accelerator.synchronize()
    # health zero: no dead pick's (stale) pair was polled
    assert not bufs.failed(), bufs.given_up_waits()
    assert bufs.flags().sum().item() == 0
    assert out[0].item() == union_ref(ids)[0]
    _down_check(bufs, ids, a2deq, w2u, c2u, down_dump, S)


def test_scale_and_tile_formulas_on_host():
    """The w2 scale byte and tile dword ``down_loads`` reads for (expert, rgd,
    step, lane), against aiter's shufflers at Muse's w2 shape (2 experts)."""
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    g = torch.Generator().manual_seed(4)
    w2 = torch.randint(0, 256, (2, K1, INTER // 2), dtype=torch.uint8, generator=g)
    c2 = torch.randint(0, 256, (2 * K1, 16), dtype=torch.uint8, generator=g)
    src = c2.view(torch.float8_e8m0fnu) if hasattr(torch, "float8_e8m0fnu") else c2
    sh = e8m0_shuffle(src).view(torch.uint8).reshape(-1)
    ws = shuffle_weight(w2, layout=(16, 16)).contiguous().view(-1).view(torch.int32)
    e, rgd, st, lane = torch.meshgrid(
        torch.arange(2), torch.arange(0, 256, 17), torch.arange(4), torch.arange(64),
        indexing="ij",
    )  # fmt: skip
    row = e * K1 + rgd * 16 + lane % 16
    col = 4 * st + lane // 16
    assert torch.equal(sh[scale_index(row, col, 16)], c2[row, col])
    dw = e * experts.W2_EXPERT_DWORDS + rgd * (2 * INTER) + st * 256 + lane * 4
    got = ws[dw.reshape(-1, 1) + torch.arange(4)].reshape(-1).view(torch.uint8)
    b0 = st * 64 + (lane // 16) * 16
    want = w2[
        e.reshape(-1, 1), (rgd * 16 + lane % 16).reshape(-1, 1),
        b0.reshape(-1, 1) + torch.arange(16),
    ].reshape(-1)  # fmt: skip
    assert torch.equal(got, want)
