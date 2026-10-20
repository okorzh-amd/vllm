# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2's router, routing, pooler and pre-expert norm + MXFP4 a1 (in
``mono/stages/router.py``, ``mono/stages/pool.py``) as probe kernels on one
GPU (``context.probe_kernel``, every peer aliased to the rank's own buffer).

[CPU] The stage LDS structs have the budgeted sizes; every probe compiles
GPU-less for gfx950 with no private segment, within the VGPR and LDS budget;
the routing's sigmoid and renorm are IEEE divides; the test's e2m1 reference
is aiter's (torchao's) RNE.
[1GPU] router: the VALU dot against fp64 and ``rocm_fp32_router_gemm``.
route: ``ieee_sigmoid`` bit-exact against ``tl.sigmoid`` on 2^20 logits; ids
and weights ``torch.equal`` to ``fused_musespark1x_routing`` incl. exact ties,
near ties, saturated sigmoids, pad rows and NaN logits. pooler: the K-split
GEMV against the stock bf16 GEMM. a1: the normed rows against
``fused_rmsnorm_expand`` and the fp4 / e8m0 bytes bit-exact against the
RoundUp reference on the kernel's own normed rows and against aiter's HIP
MXFP4 quant.
"""

import glob
import math
import os

import pytest
import regex as re
import torch

flydsl = pytest.importorskip("flydsl")
pytest.importorskip("aiter")

import flydsl.expr as fx  # noqa: E402
from aiter.ops.flydsl.kernels import buffer_ops as bo  # noqa: E402
from flydsl.expr import range_constexpr  # noqa: E402
from flydsl.expr.typing import T  # noqa: E402

from vllm.models.muse_spark.amd.specialized_decode.mono import spec  # noqa: E402
from vllm.models.muse_spark.amd.specialized_decode.mono.common import (  # noqa: E402
    context,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (  # noqa: E402
    ieee_sigmoid,
    rsrc,
    traced,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.stages import (  # noqa: E402
    pool,
    router,
)

THREADS = spec.THREADS
XQ_CTAS = 2  # the a1 probe's CTAs: every CTA must compute the same bytes
ROW_WORDS = spec.XQ_WORDS + spec.XSC_WORDS  # 640: a row's xq words, then codes
XQ_OUT = spec.MAX_TOKENS * ROW_WORDS + 16  # out words a CTA (+ tot)
SIG_ROUNDS = 8
SIG_N = spec.BLOCKS * THREADS * SIG_ROUNDS  # 2^20 logits
FLOOR_CODE = 92  # the RoundUp code of the 1e-10 amax floor


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
    """out[i] := v (a plain store; an fp32 by its bits)."""
    bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
    bo.buffer_store(bits, rsrc(out), i)


# ---------------------------------------------------------------- probe bodies


def _router_body(c):
    """CTA i: router row ``32 rank + i`` (K2: CTA R0 + i)."""
    i = c["bid"]
    router.stage_router(c, i, router.router_loads(c, i))


def _route_body(c):
    """CTA t: row t's top-8 (K2: seam CTA t)."""
    router.stage_route(c, c["bid"])


def _pool_body(c):
    """CTA task: pooler task ``task`` (K2: CTA P0 + task)."""
    task = c["bid"]
    pool.stage_pool(c, task, pool.pool_loads(c, task))


@traced
def _xq_body(c):
    """``load_xq`` with the debug dump and the normed rows, then every CTA's
    LDS a1 -> ``out`` in the ``xqd`` layout (row t: its 512 xq words, then
    its 128 xsc codes, at ``t 640``), then the rows' sums of squares."""
    s, tid = c["S"], c["tid"]
    a = c["args"]
    pool.load_xq(c, debug_dump=True, xe_out=a["xe"])
    out = a["out"]
    lds = c["lds"]["xq"]
    base = c["bid"] * XQ_OUT
    for i in range_constexpr(-(-s * ROW_WORDS // THREADS)):
        j = tid + THREADS * i
        t = j // ROW_WORDS
        w = j % ROW_WORDS
        if (j < s * ROW_WORDS) & (w < spec.XQ_WORDS):
            _st(out, base + j, fx.ptr_load(lds["xq"] + (t * spec.XQ_WORDS + w)))
        if (j < s * ROW_WORDS) & (w >= spec.XQ_WORDS):
            kb = w - spec.XQ_WORDS
            _st(out, base + j, fx.ptr_load(lds["xsc"] + (t * spec.XSC_WORDS + kb)))
    if tid < s:
        _st(out, base + s * ROW_WORDS + tid, fx.ptr_load(c["rt"]["tot"] + tid))


def _sigmoid_body(c):
    """``ieee_sigmoid`` of 2^20 fp32 logits (every CTA, 8 a thread)."""
    a = c["args"]
    g = c["bid"] * THREADS + c["tid"]
    for i in range(SIG_ROUNDS):
        idx = g + i * (spec.BLOCKS * THREADS)
        x = fx.Float32(bo.buffer_load(rsrc(a["x"]), idx, vec_width=1, dtype=T.f32))
        _st(a["y"], idx, ieee_sigmoid(x))


def _probe(which: str, S: int, key: tuple = ()) -> context.Probe:
    """The probe ``which`` at S rows (``key``: another build of it)."""
    if which == "router":
        return context.probe_kernel(
            f"muse_probe_router_s{S}",
            S,
            _router_body,
            None,
            ("w_router",),
            blocks=spec.R_TASKS,
        )
    if which == "route":
        return context.probe_kernel(
            f"muse_probe_route_s{S}", S, _route_body, None, ("bias", "pad"), blocks=S
        )
    if which == "pool":
        return context.probe_kernel(
            f"muse_probe_pool_s{S}",
            S,
            _pool_body,
            {"pool": pool.lds_struct(S)},
            ("w_pool",),
            blocks=spec.P_TASKS,
        )
    if which == "xq":
        return context.probe_kernel(
            f"muse_probe_xq_s{S}",
            S,
            _xq_body,
            {"xq": pool.xq_lds_struct(S)},
            ("w_preexp", "pad", "xe", "out"),
            blocks=XQ_CTAS,
        )
    if which == "sigmoid":
        return context.probe_kernel(
            "muse_probe_sigmoid",
            1,
            _sigmoid_body,
            None,
            ("x", "y"),
            blocks=spec.BLOCKS,
            key=key,
        )
    raise ValueError(which)


# ---------------------------------------------------------------- references


def code_roundup(amax: torch.Tensor) -> torch.Tensor:
    """The aiter HIP ``fused_mx_quant_moe_sort`` RoundUp: the biased
    e8m0 code of a 32-group's amax, floored at 1e-10."""
    amax = torch.clamp(amax.float(), min=spec.AMAX_FLOOR)
    u = (amax * torch.tensor(1.0 / 6.0, dtype=torch.float32)).view(torch.int32)
    e = (u >> 23) & 0xFF
    return (e + (((u & 0x7FFFFF) != 0) & (e < 0xFF)).int()).clamp(max=254)


_E2M1_MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def e2m1_codes(v: torch.Tensor) -> torch.Tensor:
    """fp32 values (already divided by the scale) -> E2M1 codes 0 .. 15:
    round to nearest, an exact tie to the even code, saturating at 6, the
    sign bit kept (also on values that round to zero)."""
    mag = v.abs().float()
    mids = _E2M1_MIDS.to(v.device)
    code = torch.bucketize(mag, mids)  # a midpoint -> the lower code
    tie = (code < 7) & (mag == mids[code.clamp(max=6)])
    code = torch.where(tie & (code % 2 == 1), code + 1, code)
    return code.int() | (torch.signbit(v).int() << 3)


def mxfp4_reference(xe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [S, 4096] -> (fp4 bytes [S, 2048] as aiter packs them: element
    2i in the low nibble, the e8m0 codes [S, 128])."""
    x = xe.float().view(xe.shape[0], -1, 32)
    code = code_roundup(x.abs().amax(-1))
    scale = (code << 23).view(torch.float32)
    q = e2m1_codes(x / scale.unsqueeze(-1)).view(xe.shape[0], -1)
    return (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8), code


def bf16_ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16 ulps between a and b (+0 and -0 coincide)."""

    def ordered(x):
        i = x.contiguous().view(torch.int16).int()
        return torch.where(i < 0, -(i & 0x7FFF), i)

    return (ordered(a) - ordered(b)).abs()


def _check_gemv(got: torch.Tensor, ref: torch.Tensor, x, w) -> None:
    """bf16 GEMV rows ``got`` vs ``ref`` (both bf16, CPU) of ``x @ w.T``:
    <= 1 bf16 ulp on >= 99.9 %, and past 1 ulp only within fp32 accumulation
    noise (``|got - ref| <= 2^-20 sum_k |x w|``): two fp32 dots summed in
    different orders can land 2 ulps apart on a cancelling (near-zero) sum
    (test_probe_seam's rule; a dropped K block still fails)."""
    u = bf16_ulps(got, ref)
    absdot = (x.double().abs() @ w.double().abs().T).cpu()
    noise = (got.double() - ref.double()).abs() <= 2**-20 * absdot
    assert ((u <= 1) | noise).all(), u.max()
    # >= 99.9 % within 1 ulp, as a count with a floor of 2: at S = 1 a probe
    # has only 512 outputs, where one cancelling dot (bounded by the noise
    # rule above) would otherwise fail the fraction
    assert (u > 1).sum().item() <= max(2, math.ceil(0.001 * u.numel())), u.max()


def _cos(a, b) -> float:
    return torch.nn.functional.cosine_similarity(
        a.double().reshape(1, -1), b.double().reshape(1, -1)
    ).item()


# ---------------------------------------------------------------- [CPU]


def test_lds_structs():
    """LDS sizes at S = 8: the pooler's x slice 32,896 B, xq / xsc / xss
    24,576 B; one struct class a width."""
    assert pool.lds_struct(8).__dsl_size_of__() == 32896
    assert pool.xq_lds_struct(8).__dsl_size_of__() == 24576
    assert pool.lds_struct(3) is pool.lds_struct(3)
    assert pool.xq_lds_struct(1).__dsl_size_of__() == 640 * 4 + 128 * 4
    structs = {"pool": pool.lds_struct(8), "xq": pool.xq_lds_struct(8)}
    assert context.lds_total_bytes(structs) <= spec.LDS_BYTES


def test_stage_regions_and_constants():
    for mod in (router, pool):
        assert set(mod.STAGE_REGIONS) <= set(spec.REGIONS), mod.STAGE_REGIONS
    assert router.K_THREAD * THREADS == spec.HIDDEN
    assert pool.SLICES_A_PART * spec.P_PARTS == spec.NA
    assert spec.ppart_pair(0, 1, 0, 1) == spec.ppart_pair(0, 1, 0, 0) + 1


def test_load_xq_debug_dump_resolution(monkeypatch):
    """``load_xq(c)`` (as K2 calls it) takes the debug dump from the
    context's ``debug_dump``, else its build key's, else off; the kwarg wins;
    anything but a Python bool is refused (it selects code at trace time)."""
    seen = []
    monkeypatch.setattr(pool, "_load_xq", lambda c, dbg, xe: seen.append(dbg))
    pool.load_xq({})
    pool.load_xq({"key": spec.K2Build(tokens=2, debug_dump=True)})
    pool.load_xq({"key": spec.K2Build(tokens=2)})
    pool.load_xq({"debug_dump": True, "key": spec.K2Build(tokens=2)})
    pool.load_xq({"debug_dump": True}, debug_dump=False)
    assert seen == [False, True, False, True, False]
    with pytest.raises(TypeError):
        pool.load_xq({"debug_dump": 1})


def test_reference_quant_matches_aiter():
    """The test's e2m1 rounding is aiter's (torchao) RNE on every midpoint,
    sub-midpoint and saturating value, both signs; the packing is aiter's."""
    from aiter.utility.fp4_utils import f32_to_mxfp4

    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    mids = (grid[1:] + grid[:-1]) / 2
    eps = torch.tensor(2.0**-20)
    v = torch.cat(
        [grid, mids, mids - eps, mids + eps, torch.tensor([7.0, 100.0, 0.1, 1e-30])]
    )
    v = torch.cat([v, -v, torch.randn(4096) * 3])
    if v.numel() % 2:
        v = torch.cat([v, torch.zeros(1)])
    ours = e2m1_codes(v)
    packed = (ours[0::2] | (ours[1::2] << 4)).to(torch.uint8)
    theirs = f32_to_mxfp4(v).view(torch.uint8)
    assert torch.equal(packed, theirs)
    # the RoundUp code: an exact 6 2^k amax keeps its power of two
    amax = torch.tensor([6.0, 6.0 * 2.0**-3, 6.0001, 0.0, 1e-30, 448.0])
    assert code_roundup(amax).tolist() == [127, 124, 128, FLOOR_CODE, FLOOR_CODE, 134]


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
    "which,S",
    [
        ("router", 1),
        ("router", 8),
        ("route", 1),
        ("route", 8),
        ("pool", 1),
        ("pool", 8),
        ("xq", 1),
        ("xq", 8),
        ("sigmoid", 1),
    ],
)
def test_probe_compiles(which, S, tmp_path, monkeypatch):
    """The GPU-less compile gate (README, Residency) on each probe."""
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
    if which in ("route", "sigmoid"):
        # the sigmoid's (and the renorm's) IEEE divide; v_exp_f32 for exp2
        assert "v_div_fixup_f32" in txt and "v_exp_f32" in txt
        assert "v_rcp_f32" not in txt or "v_div_scale_f32" in txt
    if which == "router":
        assert re.search(r"v_(pk_)?fma(c|k)?_f32", txt)
    if which == "pool":
        assert "v_mfma_f32_16x16x32_bf16" in txt
    if which == "xq":
        assert "v_cvt_scalef32_pk_fp4_bf16" in txt
        assert "v_rsq_f32" in txt


# ---------------------------------------------------------------- [1GPU]


def _put_xrow(bufs, x: torch.Tensor) -> None:
    """ffn_in rows -> the plain ``xrow`` region, every seam slice published."""
    buf, off = bufs.where(spec.XROW)
    n = x.numel() * 2
    buf[off : off + n].view(torch.bfloat16).copy_(x.reshape(-1))
    bufs.write_pairs("xrdy", 0, torch.ones(spec.NA, dtype=torch.int32))


@needs_gfx950
@pytest.mark.parametrize("S", [1, 3, 8])
def test_router_probe(S):
    """fp32 logits of this rank's 32 experts, in every rank's RLOG slot of
    this rank: within fp32 summation error of the fp64 dot, and the stock
    router GEMM's to its tolerance."""
    from vllm.model_executor.layers.fused_moe.router.rocm_fp32_router_gemm import (
        rocm_fp32_router_gemm,
    )

    rank = 5
    probe = _probe("router", S)
    bufs = context.ProbeBuffers(epoch=3, rank=rank)
    g = torch.Generator(device="cuda").manual_seed(7)
    w = torch.randn(spec.EXPERTS, spec.HIDDEN, generator=g, device="cuda") * 0.1768
    x = torch.randn(S, spec.HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    x[0, :64] = 0  # a zero stretch, and one tiny row
    if S > 2:
        x[2] *= 1e-3
    _put_xrow(bufs, x)
    probe(bufs, w_router=w)
    torch.accelerator.synchronize()
    assert not bufs.failed(), bufs.given_up_waits()
    got = torch.empty(S, spec.EXPERTS_RANK)
    for t in range(S):
        vals, tags = bufs.read_pairs(
            "rlog", spec.rlog_pair(rank, t, 0), spec.EXPERTS_RANK
        )
        assert (tags == bufs.tag()).all()
        got[t] = vals.cpu().view(torch.float32)
    rows = slice(spec.EXPERTS_RANK * rank, spec.EXPERTS_RANK * (rank + 1))
    xd, wd = x.double().cpu(), w[rows].double().cpu()
    ref = xd @ wd.T
    absdot = xd.abs() @ wd.abs().T
    # fp32 summation of 8192 exact products, rounded at depth 16 (the fma
    # chain) + 6 (the butterfly) + 8 (the waves) = 30: within gamma_30 =
    # 30 u / (1 - 30 u) of sum |x w| (u = 2^-24), ~1.8e-6 of it; the
    # "rel < 1e-6" bound taken relative to the absolute sum (a cancelling dot has
    # no relative bound)
    u = 2.0**-24
    bound = 30 * u / (1 - 30 * u) * absdot + 1e-30
    assert ((got.double() - ref).abs() <= bound).all()
    # the stock router GEMM to the README's tolerance, except where the
    # two fp32 orders differ only by accumulation noise on a cancelling dot
    stock = rocm_fp32_router_gemm(x, w)[:, rows].cpu()
    close = torch.isclose(got, stock, rtol=1e-5, atol=1e-6)
    noise = (got.double() - stock.double()).abs() <= 2**-20 * absdot
    assert (close | noise).all()
    assert close.float().mean().item() >= 0.99


def _tl_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """``tl.sigmoid`` of x, the stock routing's sigmoid (one Triton kernel)."""
    import triton

    y = torch.empty_like(x)
    _tl_sigmoid_kernel()[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), BLOCK=1024)
    return y


def _tl_sigmoid_kernel():
    # triton itself: vllm.triton_utils is a placeholder without a GPU
    import triton
    import triton.language as tl

    @triton.jit
    def tl_sigmoid_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        tl.store(y_ptr + off, tl.sigmoid(tl.load(x_ptr + off, mask=m)), mask=m)

    return tl_sigmoid_kernel


_SIGMOID_CHAIN = (
    r"%(\w+) = fmul float %\w+, f0xBFB8AA3B[^\n]*\n"
    r"\s*%(\w+) = (?:tail )?call float @(llvm\.exp2\.f32|llvm\.amdgcn\.exp2\.f32)"
    r"\(float %\1\)[^\n]*\n"
    r"\s*%(\w+) = fadd float %\2, 1\.000000e\+00[^\n]*\n"
    r"\s*%\w+ = fdiv float 1\.000000e\+00, %\4\b"
)


def _sigmoid_chains(ll: str) -> set[str]:
    """The exp2 intrinsics of the IR's x * -log2e -> exp2 -> 1 + e -> 1 / .
    chains (no fast-math flag on any of them: the regex admits none)."""
    return {m.group(3) for m in re.finditer(_SIGMOID_CHAIN, ll)}


@compiles
def test_ieee_sigmoid_lowers_as_tl_sigmoid(tmp_path, monkeypatch):
    """Sigmoid lowering without a GPU: ``tl.sigmoid`` on gfx950 (triton's HIP
    backend, compiled GPU-less) and ``ieee_sigmoid`` are the same chain: the
    fp32 multiply by f32(-log2 e), exp2, + 1, an IEEE ``fdiv`` (no fast-math
    flag), both in denormal mode 3. The only difference is the exp2
    intrinsic: ``llvm.exp2.f32`` is ``v_exp_f32`` plus a range fix-up below
    2^-126, where 1 + e rounds to 1 either way, so the sigmoids are equal bit
    for bit (``test_sigmoid_is_tl_sigmoid`` checks it on a GPU)."""
    triton = pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    src = ASTSource(
        fn=_tl_sigmoid_kernel(),
        signature={
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "n": "i32",
            "BLOCK": "constexpr",
        },
        constexprs={"BLOCK": 64},
    )
    tk = triton.compile(src, target=GPUTarget("hip", "gfx950", 64))
    assert _sigmoid_chains(tk.asm["llir"]) == {"llvm.exp2.f32"}
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    # a build of its own: an earlier compile in this process dumps nothing
    probe = _probe("sigmoid", 1, key=(("lowering", True),))
    try:
        probe.compile()
    except RuntimeError as e:
        if "hipErrorNoDevice" not in str(e) and "no ROCm" not in str(e):
            raise
    (ll,) = glob.glob(f"{tmp_path}/**/{probe.name}/*llvm_ir.ll", recursive=True)
    with open(ll) as f:
        assert _sigmoid_chains(f.read()) == {"llvm.amdgcn.exp2.f32"}
    txt, _ = _isa(str(tmp_path), probe.name)
    mode = re.compile(r"\.amdhsa_float_denorm_mode_32\s+(\d+)")
    ours, triton = mode.search(txt), mode.search(tk.asm["amdgcn"])
    assert ours is not None and triton is not None
    assert ours.group(1) == triton.group(1) == "3"


@needs_gfx950
def test_sigmoid_is_tl_sigmoid():
    """``ieee_sigmoid`` == ``tl.sigmoid`` bit for bit on 2^20 logits (the exp
    side included), incl. +-0, the saturation ranges, the
    exponent's denormal range, +-inf; NaN stays NaN."""
    g = torch.Generator(device="cuda").manual_seed(11)
    x = torch.randn(SIG_N, generator=g, device="cuda") * 6
    x[: SIG_N // 8] = torch.randn(SIG_N // 8, generator=g, device="cuda") * 60
    x[SIG_N // 8 : SIG_N // 4] = (
        torch.rand(SIG_N // 8, generator=g, device="cuda") - 0.5
    ) * 1e-3
    x[:12] = torch.tensor(
        [0.0, -0.0, 87.0, -87.0, 88.7, -88.7, 104.0, -104.0, 1e30, -1e30, 1e-40, -1e-40]
    )
    x[12:15] = torch.tensor([float("inf"), -float("inf"), float("nan")])
    y = torch.zeros_like(x)
    probe = _probe("sigmoid", 1)
    bufs = context.ProbeBuffers()
    probe(bufs, x=x, y=y)
    ref = _tl_sigmoid(x)
    torch.accelerator.synchronize()
    nan = torch.isnan(ref)
    assert torch.equal(torch.isnan(y), nan)
    yi, ri = y.view(torch.int32), ref.view(torch.int32)
    bad = (yi != ri) & ~nan
    assert not bad.any(), (x[bad][:8], y[bad][:8], ref[bad][:8])


def _put_logits(bufs, logits: torch.Tensor) -> None:
    """Every rank's RLOG rows (rank src's 32 experts of row t) as the router
    CTAs of the 8 ranks would have pushed them."""
    for src in range(spec.TP):
        for t in range(logits.shape[0]):
            cols = slice(spec.EXPERTS_RANK * src, spec.EXPERTS_RANK * (src + 1))
            bufs.write_pairs(
                "rlog", spec.rlog_pair(src, t, 0), logits[t, cols].contiguous()
            )


def _route_case(name: str, g):
    """(logits [S, 256] fp32, bias [256] fp32, pad [8] bool, stock-input
    logits): the routing probe's cases."""
    E = spec.EXPERTS
    dev = "cuda"
    pad = torch.zeros(spec.MAX_TOKENS, dtype=torch.bool, device=dev)
    if name == "random":
        lg = torch.randn(8, E, generator=g, device=dev) * 2
        bias = torch.randn(E, generator=g, device=dev) * 0.1
        return lg, bias, pad, lg
    if name == "one_row":
        lg = torch.randn(1, E, generator=g, device=dev) * 2
        bias = torch.randn(E, generator=g, device=dev) * 0.1
        return lg, bias, pad, lg
    if name == "ties":
        # a zero bias: the keys are the sigmoids, so equal logits are exact
        # ties, broken by the lower expert as tl.argmax does
        bias = torch.zeros(E, device=dev)
        lg = torch.randn(8, E, generator=g, device=dev) - 3
        lg[0] = 0.3  # all 256 tied: experts 0 .. 7
        tied = torch.tensor([3, 7, 17, 64, 65, 100, 200, 201, 202, 255])
        lg[1, tied] = 2.0  # 10 tied for 8 slots
        lg[2] = (torch.rand(E, generator=g, device=dev) - 0.5) * 120  # 1.0 / 0.0
        base = torch.tensor(1.25)
        near = torch.tensor([base.item()] * 16)
        for i in range(1, 16):  # 1-ulp steps: equal or adjacent sigmoids
            near[i] = torch.nextafter(near[i - 1], torch.tensor(9.0))
        lg[3, torch.arange(16) * 16 + 5] = near.to(dev)
        lg[4, [10, 70, 130, 190, 250, 33]] = 3.0  # 6 above, then 5 tied for 2
        lg[4, [1, 66, 129, 194, 254]] = 2.0
        lg[5] = 40.0  # every sigmoid exactly 1.0
        lg[6] = -200.0  # every sigmoid exactly 0.0 (sum = eps only)
        return lg, bias, pad, lg
    if name == "pad":
        lg = torch.randn(8, E, generator=g, device=dev) * 2
        bias = torch.randn(E, generator=g, device=dev) * 0.1
        pad[5:8] = True
        lg[6, 17] = float("nan")  # garbage in pad rows
        lg[7] = float("nan")
        return lg, bias, pad, lg
    if name == "nan":
        # NaN logits in live rows rank last and weigh nothing: the stock
        # result on the same rows with those experts at -1e30 (sigmoid 0)
        # and a -10 bias (ranked last there too)
        lg = torch.randn(4, E, generator=g, device=dev) * 2
        bias = torch.randn(E, generator=g, device=dev) * 0.1
        nan_e = torch.tensor([0, 5, 63, 64, 128, 200, 255])
        bias[nan_e] = -10.0
        stock = lg.clone()
        lg[0, nan_e] = float("nan")
        lg[1, nan_e[:3]] = float("nan")
        lg[3, nan_e[3:]] = float("nan")
        stock[torch.isnan(lg)] = -1e30
        return lg, bias, pad, stock
    raise ValueError(name)


@needs_gfx950
@pytest.mark.parametrize("case", ["random", "one_row", "ties", "pad", "nan"])
def test_route_probe(case):
    """ROUTE ids and weights ``torch.equal`` to ``fused_musespark1x_routing``
    on the same logits (pad rows: ids -1, weights exactly 0, NaN or not)."""
    from vllm.models.muse_spark.common.ops.fused_routing import (
        fused_musespark1x_routing,
    )

    g = torch.Generator(device="cuda").manual_seed(13)
    logits, bias, pad, stock_in = _route_case(case, g)
    S = logits.shape[0]
    probe = _probe("route", S)
    bufs = context.ProbeBuffers(epoch=21, rank=2)
    _put_logits(bufs, logits)
    probe(bufs, bias=bias, pad=pad)
    torch.accelerator.synchronize()
    assert not bufs.failed(), bufs.given_up_waits()
    vals, tags = bufs.read_pairs("route", 0, S * 16)
    assert (tags == bufs.tag()).all()
    vals = vals.cpu().view(S, 16)
    ids, wts = vals[:, : spec.TOPK], vals[:, spec.TOPK :].view(torch.float32)
    w_ref, id_ref = fused_musespark1x_routing(
        stock_in,
        bias,
        spec.TOPK,
        spec.ROUTE_SCALE,
        router_score_division_eps=spec.ROUTE_EPS,
        is_padding=pad[:S],
    )
    w_ref, id_ref = w_ref.cpu(), id_ref.cpu()
    p = pad[:S].cpu()
    live = ~p & ~torch.isnan(logits).all(1).cpu()
    assert torch.equal(ids[live], id_ref[live])
    assert torch.equal(wts[live], w_ref[live])
    assert (ids[p] == -1).all() and (wts[p] == 0).all()
    finite_pad = p & ~torch.isnan(logits).any(1).cpu()
    assert torch.equal(wts[finite_pad], w_ref[finite_pad])
    if case == "ties":
        assert ids[0].tolist() == list(range(8))
        assert ids[1].tolist() == [3, 7, 17, 64, 65, 100, 200, 201]
        assert ids[4, 6:].tolist() == [1, 66]


@needs_gfx950
@pytest.mark.parametrize("S", [1, 8])
def test_pool_probe(S):
    """This rank's 512 pooled columns, in POOL: one bf16 rounding of an fp32
    dot over K = 8192 (4 parts added in part order), against the stock GEMM."""
    rank = 3
    probe = _probe("pool", S)
    bufs = context.ProbeBuffers(epoch=5, rank=rank)
    g = torch.Generator(device="cuda").manual_seed(17)
    w = (
        torch.randn(spec.MOE_HIDDEN, spec.HIDDEN, generator=g, device="cuda") * 0.02
    ).to(torch.bfloat16)
    x = torch.randn(S, spec.HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    _put_xrow(bufs, x)
    probe(bufs, w_pool=w)
    torch.accelerator.synchronize()
    assert not bufs.failed(), bufs.given_up_waits()
    cols = slice(spec.OWN_COLS * rank, spec.OWN_COLS * (rank + 1))
    got = torch.empty(S, spec.OWN_COLS, dtype=torch.bfloat16)
    for t in range(S):
        vals, tags = bufs.read_pairs(
            "pool", spec.pool_pair(t, spec.OWN_COLS * rank), spec.OWN_COLS // 2
        )
        assert (tags == bufs.tag()).all()
        # int32 value words at a stride of 2: contiguous before the bf16 view
        got[t] = vals.cpu().contiguous().view(torch.bfloat16)
    stock = torch.nn.functional.linear(x, w)[:, cols].cpu()
    exact = (x.double() @ w.double().T)[:, cols].cpu().to(torch.bfloat16)
    w_own = w[cols]
    _check_gemv(got, stock, x, w_own)
    _check_gemv(got, exact, x, w_own)
    assert _cos(got, stock) > 0.99999


@needs_gfx950
@pytest.mark.parametrize("weight", ["random", "zero"])
@pytest.mark.parametrize("S", [1, 8])
def test_quant_probe(S, weight):
    """``load_xq``: the normed rows against ``fused_rmsnorm_expand`` (<= 1 ulp
    on >= 99.9 %, max 2, incl. an eps-scaled row and an all-zero weight); the
    a1 bytes and codes bit-exact against the RoundUp reference on the kernel's
    own normed rows and against aiter's HIP MXFP4 quant; every CTA's LDS
    equal, the debug dump equal to it; pad rows (NaN pooled) exact zeros."""
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip

    from vllm.models.muse_spark.amd.ops.fused_norm_expand import (
        fused_rmsnorm_expand,
    )

    probe = _probe("xq", S)
    bufs = context.ProbeBuffers(epoch=9, rank=1)
    g = torch.Generator(device="cuda").manual_seed(19)
    H = spec.MOE_HIDDEN
    pooled = torch.randn(S, H, generator=g, device="cuda") * 0.7
    pooled[0, :32] = 0  # a zero group: the floor's code
    pad = torch.zeros(spec.MAX_TOKENS, dtype=torch.bool, device="cuda")
    if S == 8:
        pooled[2] *= 3e-3  # mean(x^2) ~ PRE_EXP_EPS: the eps matters
        pooled[3] *= 40.0
        pad[6:8] = True
        pooled[6, 100] = float("nan")
        pooled[7] = float("nan")
    pooled = pooled.to(torch.bfloat16)
    if weight == "zero":
        w = torch.zeros(H, dtype=torch.bfloat16, device="cuda")
    else:
        w = (torch.randn(H, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
    for t in range(S):
        bufs.write_pairs("pool", spec.pool_pair(t, 0), pooled[t])
    xe = torch.full((S, H), float("nan"), dtype=torch.bfloat16, device="cuda")
    out = torch.zeros(XQ_CTAS * XQ_OUT, dtype=torch.int32, device="cuda")
    probe(bufs, w_preexp=w, pad=pad, xe=xe, out=out)
    torch.accelerator.synchronize()
    assert not bufs.failed(), bufs.given_up_waits()
    o = out.cpu().view(XQ_CTAS, XQ_OUT)
    # every CTA the same bytes; CTA 0's debug dump is its LDS
    assert torch.equal(o[0], o[1])
    buf, off = bufs.where(spec.XQD)
    dump = buf[off : off + S * ROW_WORDS * 4].view(torch.int32).cpu()
    assert torch.equal(dump, o[0, : S * ROW_WORDS])
    rows = o[0, : S * ROW_WORDS].view(S, ROW_WORDS)
    xq = rows[:, : spec.XQ_WORDS].contiguous().view(torch.uint8).view(S, H // 2)
    codes = rows[:, spec.XQ_WORDS :]
    assert (codes >= 0).all() and (codes <= 254).all()
    tot = o[0, S * ROW_WORDS : S * ROW_WORDS + S].view(torch.float32)
    p = pad[:S].cpu()
    live = ~p
    xe_k = xe.cpu()
    # the sums of squares (fp32, order differs from the fp64 one)
    ss = pooled.double().pow(2).sum(1).cpu()
    torch.testing.assert_close(tot[live].double(), ss[live], rtol=1e-5, atol=1e-30)
    # the normed rows vs the stock kernel on the same pooled rows
    stock = fused_rmsnorm_expand(
        pooled, w, spec.PRE_EXP_EPS, spec.GAIN, False, spec.TOPK
    ).cpu()
    u = bf16_ulps(xe_k[live], stock[live])
    assert u.max() <= 2 and (u <= 1).float().mean() >= 0.999, u.max()
    # pad rows: exact zeros, the floor's code, fp4 zeros
    assert (xe_k[p].float() == 0).all()
    assert (codes[p] == FLOOR_CODE).all() and (xq[p] == 0).all()
    # the a1 bytes: bit-exact against the RoundUp reference on the kernel's xe
    q_ref, c_ref = mxfp4_reference(xe_k)
    assert torch.equal(codes, c_ref)
    assert torch.equal(xq, q_ref)
    # and against aiter's HIP quant (same RoundUp; groups with amax 0 aside,
    # whose code depends on the 1e-10 floor of the fused moe-sort kernel)
    y, sc = per_1x32_mx_quant_hip(
        xe[live.cuda()].contiguous(), quant_dtype=dtypes.fp4x2
    )
    y = y.view(torch.uint8).cpu()
    sc = sc.view(torch.uint8).cpu().int()
    assert torch.equal(xq[live], y)
    nz = xe_k[live].float().view(-1, H // 32, 32).abs().amax(-1) > 0
    assert torch.equal(codes[live][nz], sc[nz])
