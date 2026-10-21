# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A torch reference of K2 (one Muse Spark decoder layer from the gated
attention core to the deferred FFN boundary), per TP rank where the kernel
shards, and the comparison helpers the numerics tests share.

Every function takes plain tensors (CPU or GPU) and reads its sizes from them,
so the CPU smoke tests run it at toy shapes and the TP8 tests at the real ones.
The rules are the stock ROCm decode's where the kernel matches it bit for bit,
and the kernel's own where no stock callable exists (rank-order sums, the
top-k tree, the RoundUp quant inside the experts):

    o_proj      a rank's bf16 partial of ``core @ w_o.T`` (K sharded)
    AR          the 8 partials summed in fp32 in rank order, one bf16 rounding
    boundary    post_attn_norm (weightless, 1e-8), ``alpha s + beta y``,
                pre_ffn_norm (``w + 1``, 1e-5) -> bf16 ffn_in
    router      fp32 logits; ``1 / (1 + exp2(-x log2e))`` (``tl.sigmoid``)
    routing     8 argmax steps over ``sigmoid + bias``, the lowest index on a
                tie; the picks' scores summed serially in pick order, ``+
                1e-15``; ``w = sig (route_scale / total)``
    pooler      bf16 ``ffn_in @ w_pool.T``
    pre-expert  bf16 ``(p rsqrt(mean p^2 + 1e-5)) (w + 1)``
    a1, a2      MXFP4 RoundUp (aiter's HIP ``fused_mx_quant_moe_sort``: amax
                floored at 1e-10, ``ceil`` of the e8m0 exponent of
                ``amax f32(1/6)``), elements E2M1 round-to-nearest-even
    experts     a rank's gate/up rows (gate first), SiLU (aiter's
                ``g rcp(1 + exp2(-g log2e)) u``), bf16 h, a2, the rank's bf16
                down partial of each pick (unweighted)
    reduce      the 8 ranks' rows summed in fp32 in rank order, not rounded
    finalize    ``x = R (w + 1)``, ``x rsqrt(mean x^2 + 1e-8) tw``, the 8 picks
                summed as ``((0+1)+(2+3))+((4+5)+(6+7))`` -> bf16 moe
    unpooler    bf16 ``moe @ w_unpool.T``

Pad rows are exact zeros everywhere the kernel selects them (``stream_out``,
ffn_in, xe, moe, branch; ids -1 and weights 0), whatever their inputs hold.

Rules that differ from a naive reference: the activation scale is RoundUp
(AITER's HIP quant), not the checkpoint's "even" weight rule; E2M1 rounds
exactly to nearest even by midpoints (``fp4_round``'s ``+1e-6`` odd penalty
rounded values just below a midpoint up); routing takes the 8 lowest-index
argmax steps and a serial pick-order sum (``fused_routing.py:100-118``), not
``torch.topk`` and a vector sum.

Phases B and C (DESIGN_BC sections 1.3, 1.4, 4) add the attention front and
core, and what their tests feed the kernels:

    ffn_boundary  the previous layer's B_ffn boundary into this one
                  (``fused_norm_residual_norm_fwd`` with the post weight first)
    qk_norm_rope  the fused insert's QK-norm and GPT-J RoPE in the gfx950
                  rounding order of its HEADS_PER_GROUP 1 instance (1.3)
    front         the qkv rows as K1 leaves them, and ``kv_insert``
    gated_norm    ``rms_norm_gated`` in Triton's gfx950 reduction order (1.4,
                  ``gated_sumsq``: the runtime-specialized instance's)
    gqa_decode    fp64 GQA decode through a block table, sliding or full
    FakeMrv2      MRV2's persistent block tables, slot mappings, sequence
                  lengths, positions and pad flags, written in place a step
    kv_cache      KV cache fixtures (LBHNC / LHBNC), page assignment, and
    poison_unread NaN in every cache byte the split core must never read

The rounding-order references reproduce every rounding point of the stock
gfx950 code but compute the hardware's approximate ops (``v_rsq_f32``,
``v_sqrt_f32``, ``v_exp_f32``) correctly rounded, so they may differ from the
GPU in the last fp32 bit there; the GPU probes compare with the stock kernels
themselves for bit-exactness.
"""

import math
from collections.abc import Collection, Mapping, Sequence
from types import SimpleNamespace

import torch

from vllm.models.muse_spark.amd.specialized_decode.mono import spec

BF16 = torch.bfloat16
F32 = torch.float32
F64 = torch.float64

# ---------------------------------------------------------------- comparisons


def rb(x: torch.Tensor) -> torch.Tensor:
    """Round to bf16 and back to fp32 (one bf16 store)."""
    return x.to(BF16).to(F32)


def bf16_ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16 ulps between a and b (bf16 or rounded to it; +0 and -0 coincide)."""

    def ordered(x):
        i = x.to(BF16).contiguous().view(torch.int16).int()
        return torch.where(i < 0, -(i & 0x7FFF), i)

    return (ordered(a) - ordered(b)).abs()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-300)).item()


def check_ulps(got, ref, *, frac: float = 0.999, max_ulps: int = 2, what="") -> None:
    """<= 1 bf16 ulp on >= ``frac`` of the elements, never more than
    ``max_ulps``."""
    u = bf16_ulps(got, ref)
    assert u.max().item() <= max_ulps, (what, u.max().item())
    assert (u <= 1).float().mean().item() >= frac, (what, (u > 1).sum().item())


def check_gemv(got, ref, x, w, *, frac: float = 0.999, what="") -> None:
    """bf16 GEMV rows ``got`` vs ``ref`` of ``x @ w.T``: <= 1 bf16 ulp on >=
    ``frac``, and past 1 ulp only within fp32 accumulation noise
    (``|got - ref| <= 2^-20 sum_k |x w|``): two fp32 dots summed in different
    orders can land 2 ulps apart on a cancelling sum, while a dropped K block
    still fails (test_probe_seam's rule)."""
    got, ref = got.cpu(), ref.cpu()
    u = bf16_ulps(got, ref)
    absdot = (x.double().abs() @ w.double().abs().T).cpu()
    noise = (got.double() - ref.double()).abs() <= 2**-20 * absdot
    assert ((u <= 1) | noise).all(), (what, u.max().item())
    assert (u <= 1).float().mean().item() >= frac, (what, (u > 1).sum().item())


# ---------------------------------------------------------------- MXFP4

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=F64)
# the midpoints between consecutive E2M1 magnitudes; an exact tie goes to the
# even code (0.25 -> 0, 0.75 -> 1.0, 1.25 -> 1.0, 1.75 -> 2.0, 2.5 -> 2.0,
# 3.5 -> 4.0, 5.0 -> 4.0)
E2M1_MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=F32)
FLOOR_CODE = 92  # the code of an all-zero group: amax floored at 1e-10


def e8m0_roundup_code(amax: torch.Tensor) -> torch.Tensor:
    """The RoundUp of aiter's HIP ``fused_mx_quant_moe_sort``: the biased e8m0 code
    of a group's amax, floored at 1e-10, ``ceil`` of the exponent of ``amax
    f32(1/6)`` (dequant scale ``2^(code - 127)``); at most 254, as the
    kernels clamp it."""
    amax = torch.clamp(amax.float(), min=spec.AMAX_FLOOR)
    u = (amax * torch.tensor(1.0 / 6.0, dtype=F32)).view(torch.int32)
    e = (u >> 23) & 0xFF
    up = ((u & 0x7FFFFF) != 0) & (e < 0xFF)
    return (e + up.int()).clamp(max=254).int()


def e2m1_codes(v: torch.Tensor) -> torch.Tensor:
    """fp32 values already divided by their scale -> E2M1 nibbles 0 .. 15:
    nearest, an exact midpoint to the even code, saturating at 6, the sign bit
    kept (also on a value that rounds to zero: the hardware convert's)."""
    mag = v.abs().float()
    mids = E2M1_MIDS.to(v.device)
    code = torch.bucketize(mag, mids)  # a midpoint -> the lower code
    tie = (code < len(mids)) & (mag == mids[code.clamp(max=len(mids) - 1)])
    code = torch.where(tie & (code % 2 == 1), code + 1, code)
    return code.int() | (torch.signbit(v).int() << 3)


def mx_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[..., n] -> (packed uint8 [..., n/2] with element 2i in the low nibble,
    codes int32 [..., n/32]): the RoundUp a1 / a2 quant."""
    g = x.float().unflatten(-1, (-1, 32))
    code = e8m0_roundup_code(g.abs().amax(-1))
    scale = (code << 23).view(F32)
    nib = e2m1_codes(g / scale.unsqueeze(-1)).flatten(-2)
    return (nib[..., 0::2] | (nib[..., 1::2] << 4)).to(torch.uint8), code


def fp4_values(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., n] of packed E2M1 (element 2i in the low nibble) -> fp64
    [..., 2n]."""
    p = packed.to(torch.int64)
    nib = torch.stack([p & 0xF, p >> 4], -1).flatten(-2)
    mag = E2M1.to(packed.device)[nib & 7]
    return torch.where((nib & 8) != 0, -mag, mag)


def mx_dequant(packed: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """MXFP4 (uint8 [..., n/2], e8m0 codes [..., n/32], any int dtype) -> fp64
    [..., n] (exact)."""
    sc = torch.exp2(codes.to(F64) - 127).repeat_interleave(32, -1)
    return fp4_values(packed) * sc


# ---------------------------------------------------------------- the stages


def _pad_mask(pad, rows: int, device) -> torch.Tensor:
    """The pad flags as bool [rows] (none when ``pad`` is None)."""
    if pad is None:
        return torch.zeros(rows, dtype=torch.bool, device=device)
    return pad[:rows].to(device=device, dtype=torch.bool)


def _zero_rows(x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """Rows of ``pad`` := 0 by selection (a NaN there does not leak)."""
    return torch.where(pad.view(-1, *([1] * (x.dim() - 1))), torch.zeros_like(x), x)


def oproj_partial(core: torch.Tensor, w_o: torch.Tensor) -> torch.Tensor:
    """A rank's o_proj partial: bf16 ``core @ w_o.T`` (fp32 accumulation; K =
    the rank's heads)."""
    return (core.double() @ w_o.double().T).to(BF16)


def rank_order_sum(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    """fp32 sum of ``parts`` in rank (list) order."""
    acc = parts[0].float()
    for p in parts[1:]:
        acc = acc + p.float()
    return acc


def attn_out(partials: Sequence[torch.Tensor]) -> torch.Tensor:
    """The seam's all-reduce: the bf16 partials summed in fp32 in rank order,
    rounded once (the custom-AR rule)."""
    return rank_order_sum(partials).to(BF16)


def _rsq(x: torch.Tensor, eps: float) -> torch.Tensor:
    """fp32 ``rsqrt(mean(x^2) + eps)`` over the last dim."""
    x = x.float()
    return torch.rsqrt((x * x).sum(-1, keepdim=True) / x.shape[-1] + eps)


def boundary(attn, stream, alpha, beta, w_preffn, pad=None):
    """The attention boundary: (``stream_out`` fp32, ffn_in bf16).
    ``post_attn_norm`` is weightless; ``pre_ffn_norm`` has gain ``w + 1``."""
    pad = _pad_mask(pad, stream.shape[0], stream.device)
    x = attn.float()
    y = x * _rsq(x, spec.POST_ATTN_EPS)
    n = _zero_rows(alpha.float() * stream.float() + beta.float() * y, pad)
    f = (n * _rsq(n, spec.PRE_FFN_EPS)) * (w_preffn.float() + spec.GAIN)
    return n, _zero_rows(f.to(BF16), pad)


def router_logits(ffn_in: torch.Tensor, w_router: torch.Tensor) -> torch.Tensor:
    """fp32 logits ``ffn_in @ w_router.T`` (fp64 accumulation, rounded once)."""
    return (ffn_in.double() @ w_router.double().T).float()


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """``tl.sigmoid``'s fp32 math: ``1 / (1 + exp2(x f32(-log2e)))``."""
    x = x.float()
    e = torch.exp2(x * torch.tensor(-spec.LOG2E, dtype=F32))
    return torch.tensor(1.0, dtype=F32) / (1.0 + e)


NAN_KEY = -3.0e38  # a NaN score sorts below every finite one


def route(logits, bias, pad=None, *, topk: int = spec.TOPK):
    """The top-k routing of every row: (ids int32 [S, topk], weights fp32).
    Pad rows: ids -1, weights 0."""
    S, E = logits.shape
    pad = _pad_mask(pad, S, "cpu")
    sig = sigmoid(logits.cpu())
    key = sig + bias.cpu().float()
    key = torch.where(torch.isnan(key), torch.tensor(NAN_KEY), key)
    sig = torch.where(torch.isnan(sig), torch.zeros_like(sig), sig)
    ids = torch.empty(S, topk, dtype=torch.int32)
    w = torch.empty(S, topk, dtype=F32)
    neg = torch.tensor(float("-inf"))
    eps = torch.tensor(spec.ROUTE_EPS, dtype=F32)
    scale = torch.tensor(spec.ROUTE_SCALE, dtype=F32)
    for t in range(S):
        taken = torch.zeros(E, dtype=torch.bool)
        total = torch.tensor(0.0, dtype=F32)
        picks = []
        for _ in range(topk):
            # torch.argmax returns the first maximum: the lowest index on a tie
            i = int(torch.argmax(torch.where(taken, neg, key[t])))
            taken[i] = True
            picks.append(i)
            total = total + sig[t, i]  # serial, pick order, fp32
        inv = scale / (total + eps)
        ids[t] = torch.tensor(picks, dtype=torch.int32)
        w[t] = sig[t, picks] * inv
        if pad[t]:
            ids[t] = -1
            w[t] = 0.0
    return ids.to(logits.device), w.to(logits.device)


def pooler(ffn_in: torch.Tensor, w_pool: torch.Tensor) -> torch.Tensor:
    """bf16 ``ffn_in @ w_pool.T`` (fp32 accumulation)."""
    return (ffn_in.double() @ w_pool.double().T).to(BF16)


def pre_expert(pooled, w_preexp, pad=None) -> torch.Tensor:
    """The pre-expert norm: bf16 ``(p rsqrt(mean p^2 + 1e-5)) (w + 1)``."""
    pad = _pad_mask(pad, pooled.shape[0], pooled.device)
    p = pooled.float()
    xe = (p * _rsq(p, spec.PRE_EXP_EPS)) * (w_preexp.float() + spec.GAIN)
    return _zero_rows(xe.to(BF16), pad)


def silu_mul(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """The SiLU-and-mul of aiter in fp32, ``g rcp(1 + exp2(-g log2e)) u``, the
    stage-1 output rounded to bf16."""
    g, u = g.float(), u.float()
    e = torch.exp2(g * torch.tensor(-spec.LOG2E, dtype=F32))
    return ((g * (1.0 / (1.0 + e))) * u).to(BF16)


def live(ids: torch.Tensor) -> torch.Tensor:
    """A pick is live iff its id is an expert (pads are -1)."""
    return (ids >= 0) & (ids < spec.EXPERTS)


def expert_rows(xe, ids, w13, c13, w2, c2, *, dtype=F64) -> torch.Tensor:
    """One rank's bf16 down partial of every pick, unweighted: bf16 [S, topk,
    Hm]. ``w13`` uint8 [E, 2 I, Hm/2] (gate rows, then up) and ``c13`` its
    e8m0 codes [E, 2 I, Hm/32]; ``w2`` [E, Hm, I/2], ``c2`` [E, Hm, I/32]: the
    rank's shard in the checkpoint (unshuffled) layout. Dead picks' rows are
    zero. The a1 quant is of ``xe`` (the bf16 pre-expert rows), the a2 quant
    of the bf16 h."""
    S, K = ids.shape
    hm = xe.shape[-1]
    inter = w13.shape[1] // 2
    out = torch.zeros(S, K, hm, dtype=BF16, device=xe.device)
    a1q, a1c = mx_quant(xe)
    a1 = mx_dequant(a1q, a1c).to(dtype)
    deq: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    ok = live(ids)
    for t in range(S):
        for k in range(K):
            if not bool(ok[t, k]):
                continue
            e = int(ids[t, k])
            if e not in deq:
                deq[e] = (
                    mx_dequant(w13[e], c13[e]).to(dtype),
                    mx_dequant(w2[e], c2[e]).to(dtype),
                )
            w_gu, w_d = deq[e]
            gu = (a1[t] @ w_gu.T).float()  # fp32 accumulators
            h = silu_mul(gu[:inter], gu[inter:])
            a2 = mx_dequant(*mx_quant(h)).to(dtype)
            out[t, k] = (a2 @ w_d.T).to(BF16)
    return out


def owner_sum(rows_by_rank: Sequence[torch.Tensor]) -> torch.Tensor:
    """R: the ranks' bf16 rows summed in fp32 in rank order, never rounded
    (README, Numerics: the deviation)."""
    return rank_order_sum(rows_by_rank)


def topk_tree(xs: torch.Tensor) -> torch.Tensor:
    """The fixed sum over dim 1 (8 picks): ((0+1)+(2+3))+((4+5)+(6+7))."""
    assert xs.shape[1] == 8, xs.shape
    a = (xs[:, 0] + xs[:, 1]) + (xs[:, 2] + xs[:, 3])
    b = (xs[:, 4] + xs[:, 5]) + (xs[:, 6] + xs[:, 7])
    return a + b


def finalize(R, w_postexp, tw, pad=None) -> torch.Tensor:
    """The post-expert norm (input scale: ``x = R (w + 1)``), the route weight,
    the top-k sum -> bf16 moe. A pad row's pick and a pick of route weight 0
    contribute an exact 0 (selected: NaN in their rows does not leak)."""
    S = R.shape[0]
    pad = _pad_mask(pad, S, R.device)
    x = R.float() * (w_postexp.float() + spec.GAIN)
    xs = x * (_rsq(x, spec.POST_EXP_EPS) * tw.float().unsqueeze(-1))
    dead = pad.view(S, 1, 1) | (tw == 0).view(S, -1, 1)
    xs = torch.where(dead, torch.zeros_like(xs), xs)
    return _zero_rows(topk_tree(xs).to(BF16), pad)


def owner_sumsq(R, w_postexp) -> torch.Tensor:
    """A pick's sum of ``(R (w + 1))^2`` over the pooled width, fp32 [S, 8]
    (fp64 accumulation: the owners' fixed order is within fp32 rounding)."""
    x = R.double() * (w_postexp.double() + spec.GAIN)
    return x.square().sum(-1).float()


def unpooler(moe: torch.Tensor, w_unpool: torch.Tensor, pad=None) -> torch.Tensor:
    """bf16 ``moe @ w_unpool.T``; pad rows 0."""
    pad = _pad_mask(pad, moe.shape[0], moe.device)
    return _zero_rows((moe.double() @ w_unpool.double().T).to(BF16), pad)


# ---------------------------------------------------------------- the chain


def k2(core, stream, pad, w) -> dict:
    """K2 on every rank's inputs: ``core`` is a list (a rank's [S, CORE] bf16),
    ``w`` holds the replicated weights (``attn_alpha``, ``attn_beta``,
    ``w_preffn``, ``w_router``, ``bias``, ``w_pool``, ``w_preexp``,
    ``w_postexp``, ``w_unpool``) and the per-rank lists ``w_o``, ``w13``,
    ``c13``, ``w2``, ``c2`` (checkpoint layout, ``expert_rows``). Returns
    every intermediate the kernel exposes (``partials``, ``attn_out``,
    ``stream_out``, ``ffn_in``, ``logits``, ``ids``, ``tw``, ``pooled``,
    ``xe``, ``a1``, ``rows``, ``R``, ``moe``, ``branch``)."""
    tp = len(core)
    partials = [oproj_partial(core[r], w["w_o"][r]) for r in range(tp)]
    attn = attn_out(partials)
    stream_out, ffn_in = boundary(
        attn, stream, w["attn_alpha"], w["attn_beta"], w["w_preffn"], pad
    )
    logits = router_logits(ffn_in, w["w_router"])
    ids, tw = route(logits, w["bias"], pad)
    pooled = pooler(ffn_in, w["w_pool"])
    xe = pre_expert(pooled, w["w_preexp"], pad)
    rows = [
        expert_rows(xe, ids, w["w13"][r], w["c13"][r], w["w2"][r], w["c2"][r])
        for r in range(tp)
    ]
    R = owner_sum(rows)
    moe = finalize(R, w["w_postexp"], tw, pad)
    return {
        "partials": partials,
        "attn_out": attn,
        "stream_out": stream_out,
        "ffn_in": ffn_in,
        "logits": logits,
        "ids": ids,
        "tw": tw,
        "pooled": pooled,
        "xe": xe,
        "a1": mx_quant(xe),
        "rows": rows,
        "R": R,
        "moe": moe,
        "branch": unpooler(moe, w["w_unpool"], pad),
    }


# ---------------------------------------------------------------- B / C: fp32


def fma32(a, b, c) -> torch.Tensor:
    """Correctly rounded fp32 ``a * b + c`` (``v_fma_f32`` / ``v_pk_fma_f32``,
    ``fmath.fma``) on fp32 tensors. The fp64 product is exact; TwoSum recovers
    the fp64 sum's rounding error, and a sum that landed exactly on an fp32
    midpoint goes to the side of that error (no double rounding)."""
    a, b, c = torch.broadcast_tensors(*(torch.as_tensor(v).float() for v in (a, b, c)))
    p = a.double() * b.double()
    cd = c.double()
    s = p + cd
    bv = s - p
    err = (p - (s - bv)) + (cd - bv)
    r = s.float()
    inf = torch.full_like(r, math.inf)
    side = torch.where(
        s > r.double(), torch.nextafter(r, inf), torch.nextafter(r, -inf)
    )
    tie = (s == (r.double() + side.double()) * 0.5) & (err != 0) & s.isfinite()
    fix = torch.where(err > 0, torch.maximum(r, side), torch.minimum(r, side))
    return torch.where(tie, fix, r)


def _f32(v: float) -> torch.Tensor:
    return torch.tensor(v, dtype=F32)


# ---------------------------------------------------------------- B: K1


def ffn_boundary(branch, stream, w_postffn, alpha, beta, w_preqkv, pad=None):
    """The B_ffn boundary into a layer (K1, DESIGN_BC 4 / 5.4): (``stream_out``
    fp32, ``attn_in`` bf16). The previous layer's ``post_ffn_norm`` scales its
    input first, ``x = RN(f32(branch) (w_post + 1))``, ``y = x rsqrt(mean x^2 +
    1e-8)``; ``n = alpha s + beta y``; ``attn_in = bf16(n rsqrt(mean n^2 +
    1e-5) (w_pre + 1))``. Pad rows: exact zeros."""
    pad = _pad_mask(pad, stream.shape[0], stream.device)
    x = branch.float() * (w_postffn.float() + spec.GAIN)
    y = x * _rsq(x, spec.POST_FFN_EPS)
    n = _zero_rows(alpha.float() * stream.float() + beta.float() * y, pad)
    z = (n * _rsq(n, spec.RMS_EPS)) * (w_preqkv.float() + spec.GAIN)
    return n, _zero_rows(z.to(BF16), pad)


def stream_tolerance(branch, stream, w_postffn, alpha, beta) -> torch.Tensor:
    """The ``stream_out`` bound (DESIGN_BC 4): ``1e-5 (|alpha s| + |beta y|)``,
    elementwise; a relative bound fails where ``alpha s + beta y`` cancels."""
    x = branch.float() * (w_postffn.float() + spec.GAIN)
    y = x * _rsq(x, spec.POST_FFN_EPS)
    return 1e-5 * ((alpha.float() * stream.float()).abs() + (beta.float() * y).abs())


def rope_table(max_pos: int, theta: float = spec.ROPE_THETA, device="cpu"):
    """A ``cos_sin_cache`` as ``RotaryEmbedding(64, 64, ..., dtype=float)``
    builds it: fp32 [max_pos, 64], cos in columns 0 .. 31, sin in 32 .. 63."""
    inv = 1.0 / (theta ** (torch.arange(0, 64, 2, dtype=F32) / 64))
    freqs = torch.outer(torch.arange(max_pos, dtype=F32), inv)
    return torch.cat([freqs.cos(), freqs.sin()], -1).to(device)


def qk_norm_rope(heads: torch.Tensor, cs: torch.Tensor | None) -> torch.Tensor:
    """The fused insert's QK-norm and RoPE of bf16 heads [N, 64] (DESIGN_BC
    1.3): per 8-dim chunk (a lane) a serial sum of rounded squares, the 8-lane
    tree ``((s0+s4)+(s2+s6)) + ((s1+s5)+(s3+s7))``, ``rsq(fma(sum, 2^-6,
    eps))``, ``RN(x r)``; then, with ``cs`` (fp32 [N, 64]: each head's
    position's table row), the uneven GPT-J pair table; one bf16 rounding."""
    x = heads.float().reshape(-1, 8, 8)  # [N, lane li, dim k]: dim 8 li + k
    sq = x * x
    s = sq[..., 0]
    for k in range(1, 8):
        s = s + sq[..., k]
    tot = ((s[:, 0] + s[:, 4]) + (s[:, 2] + s[:, 6])) + (
        (s[:, 1] + s[:, 5]) + (s[:, 3] + s[:, 7])
    )
    r = torch.rsqrt(tot * _f32(2.0**-6) + _f32(spec.RMS_EPS))
    x = x * r.view(-1, 1, 1)
    if cs is not None:
        c = cs[:, :32].float().reshape(-1, 8, 4)  # pair i of lane li: 4 li + i
        sn = cs[:, 32:].float().reshape(-1, 8, 4)
        xe, xo = x[..., 0::2], x[..., 1::2]  # dims 2i, 2i + 1 of the chunk
        out = torch.empty_like(x)
        out[..., 0] = xe[..., 0] * c[..., 0] - xo[..., 0] * sn[..., 0]  # no fma
        out[..., 1] = fma32(xo[..., 0], c[..., 0], xe[..., 0] * sn[..., 0])
        for i in (1, 2, 3):
            out[..., 2 * i] = fma32(xe[..., i], c[..., i], -(xo[..., i] * sn[..., i]))
        for i in (1, 2):
            out[..., 2 * i + 1] = fma32(xo[..., i], c[..., i], xe[..., i] * sn[..., i])
        # dim 7: the fused product is x6 s3, the rounded one x7 c3
        out[..., 7] = fma32(xe[..., 3], sn[..., 3], xo[..., 3] * c[..., 3])
        x = out
    return x.reshape(heads.shape).to(BF16)


def front(qkv, positions, cos_sin=None, pad=None, *, zero_pad=True) -> torch.Tensor:
    """K1's ``qkvo`` from bf16 GEMV rows ``qkv`` [S, QKV_ROWS]: q and k heads
    normalized (and roped with ``cos_sin``, positions clamped to its rows; a
    pad row's position is 0), v and o_gate verbatim; pad rows zero (stock
    writes values there). ``zero_pad=False`` keeps a pad row's values: the K
    and V K1 inserts for it (DESIGN_BC 5.6 stores ``bf16(x)``, not the zeroed
    row, when the pad row's slot is >= 0)."""
    S = qkv.shape[0]
    pad = _pad_mask(pad, S, qkv.device)
    out = qkv.clone()
    nq = spec.HEADS + spec.KV_HEADS  # q heads, then k heads: rows [0, ROW_V)
    heads = qkv[:, : spec.ROW_V].reshape(S * nq, spec.HEAD_DIM)
    cs = None
    if cos_sin is not None:
        pos = torch.where(pad, 0, positions[:S].long().to(qkv.device))
        pos = pos.clamp(0, cos_sin.shape[0] - 1)
        cs = cos_sin[pos].repeat_interleave(nq, 0)
    out[:, : spec.ROW_V] = qk_norm_rope(heads, cs).view(S, -1)
    return _zero_rows(out, pad) if zero_pad else out


def kv_insert(kv: torch.Tensor, qkvo: torch.Tensor, slots) -> torch.Tensor:
    """The KV insert of ``qkvo``'s k and v heads into ``kv`` [blocks,
    KV_HEADS, KV_BLOCK, KV_C], in place: slot >= 0 only (and inside the
    cache, as K1 clamps); returns ``kv``. The insert does not look at the pad
    flags: with pad rows, pass ``front(..., pad, zero_pad=False)``."""
    for t in range(qkvo.shape[0]):
        slot = int(slots[t])
        blk, n = divmod(slot, spec.KV_BLOCK)
        if slot < 0 or blk >= kv.shape[0]:
            continue
        for h in range(spec.KV_HEADS):
            k = qkvo[t, spec.ROW_K + 64 * h : spec.ROW_K + 64 * (h + 1)]
            v = qkvo[t, spec.ROW_V + 64 * h : spec.ROW_V + 64 * (h + 1)]
            kv[blk, h, n, : spec.V_OFF] = k.to(kv.device)
            kv[blk, h, n, spec.V_OFF :] = v.to(kv.device)
    return kv


# ---------------------------------------------------------------- B / C: core


def gated_sumsq(x: torch.Tensor) -> torch.Tensor:
    """fp32 [N] sum of squares of each row of 64 (``x`` [..., 64]) in the
    order of the gfx950 ``tl.sum(x * x)`` that ``rms_norm_gated`` runs as
    served (DESIGN_BC 1.4): the runtime specialization lays a row over 32
    lanes, two adjacent columns a lane, so the leaf is ``s_c = RN(x_2c^2) +
    RN(x_2c+1^2)``, then the butterfly over lanes xor 16, 8, 4, 2, 1, i.e.
    pairwise halving of s: ``s[:16] + s[16:]``, then 8, 4, 2, 1."""
    sq = x.float().reshape(-1, 64) ** 2
    s = sq[:, 0::2] + sq[:, 1::2]
    while s.shape[1] > 1:
        h = s.shape[1] // 2
        s = s[:, :h] + s[:, h:]
    return s[:, 0]


def gated_norm(attn, og, pad=None) -> torch.Tensor:
    """``rms_norm_gated(attn, og, None, None, "sigmoid", eps=1e-5)`` per (token,
    head) row of 64 in the order of the gfx950 instance served (DESIGN_BC
    1.4): ``tot = gated_sumsq(x)``; ``rstd = 1 / sqrt(fma(tot, 2^-6, eps))``;
    ``y = RN(RN(x rstd) sigmoid(g))``, one bf16 rounding. Pad rows: zeros.
    bf16, ``attn``'s shape."""
    S = attn.shape[0]
    pad = _pad_mask(pad, S, attn.device)
    x = attn.float().reshape(-1, 64)
    tot = gated_sumsq(x)
    rstd = _f32(1.0) / torch.sqrt(tot * _f32(2.0**-6) + _f32(spec.RMS_EPS))
    y = (x.reshape(-1, 64) * rstd.view(-1, 1)) * sigmoid(og.reshape(-1, 64))
    return _zero_rows(y.reshape(attn.shape).to(BF16), pad)


def gqa_decode(
    q, kv, block_table, seq_lens, *, window: bool, scale=spec.QK_SCALE, pad=None
) -> torch.Tensor:
    """fp64 decode attention of each row's q heads (bf16 [S, HEADS * 64] or
    [S, HEADS, 64], as K1 leaves them) over its keys ``lo <= k < L`` through
    the block table: ``L = clamp(seq_lens[t], 0, 16 * table row)``, ``lo = L -
    WINDOW`` for a sliding layer (else 0), key k on page ``block_table[t, k //
    16]`` row ``k % 16``; q head h reads KV head ``h // 8``. ``kv`` is a
    layer's cache [blocks, KV_HEADS, KV_BLOCK, KV_C], any strides. Pad rows
    and ``L == 0``: zeros. fp64 [S, HEADS, 64] (exact softmax)."""
    S = q.shape[0]
    pad = _pad_mask(pad, S, "cpu")
    out = torch.zeros(S, spec.HEADS, spec.HEAD_DIM, dtype=F64)
    cap = block_table.shape[1] * spec.KV_BLOCK
    for t in range(S):
        L = min(max(int(seq_lens[t]), 0), cap)
        if bool(pad[t]) or L == 0:
            continue
        lo = max(0, L - spec.WINDOW) if window else 0
        k = torch.arange(lo, L)
        blocks = block_table[t].cpu().long()[k // spec.KV_BLOCK]
        rows = kv[blocks.to(kv.device), :, (k % spec.KV_BLOCK).to(kv.device)]
        rows = rows.cpu().double()  # [keys, KV_HEADS, KV_C]
        K, V = rows[..., : spec.V_OFF], rows[..., spec.V_OFF :]
        qh = q[t].cpu().double().reshape(spec.KV_HEADS, spec.Q_PER_KV, -1)
        p = torch.softmax(torch.einsum("gjd,ngd->gjn", qh, K) * scale, -1)
        out[t] = torch.einsum("gjn,ngd->gjd", p, V).reshape(spec.HEADS, -1)
    return out


# ---------------------------------------------------------------- KV fixtures

PAD_SLOT = -1  # MRV2's PAD_SLOT_ID: no insert


def kv_cache(
    num_blocks: int,
    *,
    layout: str = "LBHNC",
    null: float = math.nan,
    device="cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """A layer's bf16 cache [num_blocks, KV_HEADS, KV_BLOCK, KV_C] as AITER FA
    may lay it out: "LBHNC" (a block's heads adjacent, contiguous) or "LHBNC"
    (head-major storage: ``stride(1) > stride(0)``). Random normal values;
    block 0, the null block, holds ``null`` (NaN: a read of it shows)."""
    dims = (num_blocks, spec.KV_HEADS, spec.KV_BLOCK, spec.KV_C)
    if layout == "LBHNC":
        kv = torch.randn(*dims, generator=generator)
    elif layout == "LHBNC":
        kv = torch.randn(dims[1], dims[0], *dims[2:], generator=generator)
        kv = kv.transpose(0, 1)
    else:
        raise ValueError(f"KV layout {layout!r} (LBHNC or LHBNC)")
    kv = kv.to(device=device, dtype=BF16)
    kv[0] = null
    return kv


def assign_pages(
    seq_lens: Sequence[int],
    num_blocks: int,
    *,
    extra: int = 0,
    generator: torch.Generator | None = None,
) -> list[list[int]]:
    """Distinct random physical blocks (never 0, the null block) for each
    request's ``cdiv(L, 16) + extra`` pages, in page order."""
    need = [-(-L // spec.KV_BLOCK) + extra for L in seq_lens]
    perm = torch.randperm(num_blocks - 1, generator=generator) + 1
    if sum(need) > perm.numel():
        raise ValueError(f"{sum(need)} pages > {num_blocks - 1} blocks")
    out, at = [], 0
    for n in need:
        out.append(perm[at : at + n].tolist())
        at += n
    return out


def poison_unread(kv, block_table, seq_lens, *, window: bool, rows=None):
    """NaN into every byte of ``kv`` (in place) that row t's attention must not
    read: its keys below ``lo`` and at or past ``L`` on the pages it reads
    (the first window page's head, the last page's tail, in K and V), and the
    whole of its table's pages wholly below the window. Rows' pages must be
    distinct (``assign_pages``). Returns ``kv``."""
    rows = range(block_table.shape[0]) if rows is None else rows
    cap = block_table.shape[1] * spec.KV_BLOCK
    for t in rows:
        L = min(max(int(seq_lens[t]), 0), cap)
        if L == 0:
            continue
        lo, b0, nb, _, _ = spec.split_plan(L, window, 1)
        bt = block_table[t].tolist()
        for j in range(b0):
            if bt[j] != 0:
                kv[bt[j]] = math.nan
        kv[bt[b0], :, : lo - b0 * spec.KV_BLOCK] = math.nan
        last = b0 + nb - 1
        kv[bt[last], :, L - last * spec.KV_BLOCK :] = math.nan
    return kv


class FakeMrv2:
    """Model Runner V2's persistent per-step inputs of the kernel layers
    (DESIGN_BC 1.5), written in place a step as MRV2 writes them, so a captured
    graph's addresses stay valid: per KV group a block table int32 [max_reqs,
    max_blocks] (``BlockTables.input_block_tables[g]``) and a slot-mapping row
    (``slot_mappings`` int64 [G, max_tokens]); ``seq_lens`` int32 [max_reqs];
    ``positions`` int64 and ``is_padding`` bool [max_tokens].

    ``groups``: KV group -> its layers' names; ``sliding``: the sliding
    groups. A sliding group's entries wholly below the window of the
    request's first scheduled token are nulled (block 0); entries past a
    request's last page keep what an earlier step left (``stale`` fills them
    instead); padded rows are zeroed."""

    def __init__(
        self,
        groups: Mapping[str, Sequence[str]],
        sliding: Collection[str],
        *,
        max_reqs: int = spec.MAX_ROWS,
        max_blocks: int = 256,
        max_tokens: int = spec.MAX_ROWS,
        device="cpu",
    ):
        self.groups = {g: tuple(names) for g, names in groups.items()}
        self.sliding = frozenset(sliding)
        self.block_tables = {
            g: torch.zeros(max_reqs, max_blocks, dtype=torch.int32, device=device)
            for g in self.groups
        }
        self.slot_mappings = torch.full(
            (len(self.groups), max_tokens), PAD_SLOT, dtype=torch.int64, device=device
        )
        self.seq_lens = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self.positions = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        self.is_padding = torch.ones(max_tokens, dtype=torch.bool, device=device)

    def _view(self, width: int, q: int) -> SimpleNamespace:
        reqs = width // q
        md, sm = {}, {}
        for gi, (g, names) in enumerate(self.groups.items()):
            for name in names:
                sm[name] = self.slot_mappings[gi, :width]
                md[name] = SimpleNamespace(
                    block_table=self.block_tables[g][:reqs],
                    seq_lens=self.seq_lens[:reqs],
                    slot_mapping=sm[name],
                    decode_metadata=SimpleNamespace(max_query_len=q),
                    num_prefills=0,
                    num_extends=0,
                    num_decode_tokens=width,
                    num_decodes=reqs,
                )
        return SimpleNamespace(
            width=width,
            attn_metadata=md,
            slot_mapping=sm,
            positions=self.positions[:width],
            pad=self.is_padding[:width],
        )

    def capture(self, width: int, q: int = 1) -> SimpleNamespace:
        """A capture / dummy run: rows zeroed, ``seq_lens`` q (each dummy
        request's query), slots ``PAD_SLOT``, positions 0, every row padding."""
        reqs = width // q
        for bt in self.block_tables.values():
            bt[:reqs].zero_()
        self.slot_mappings.fill_(PAD_SLOT)
        self.seq_lens.zero_()
        self.seq_lens[:reqs] = q
        self.positions[:width].zero_()
        self.is_padding[:width].fill_(True)
        return self._view(width, q)

    def step(
        self,
        seq_lens: Sequence[int],
        pages: Mapping[str, Sequence[Sequence[int]]],
        width: int,
        q: int = 1,
        *,
        stale: int | None = None,
    ) -> SimpleNamespace:
        """One decode (q = 1) or uniform verify step of ``len(seq_lens)``
        requests padded to ``width`` rows. ``seq_lens[r]``: request r's context
        including this step's q tokens; ``pages[g][r]``: its physical blocks in
        group g, page order. Returns the step's view: ``attn_metadata`` and
        ``slot_mapping`` by layer name (slices of the persistent buffers),
        ``positions``, ``pad``."""
        reqs, padded = len(seq_lens), width // q
        if padded * q != width or reqs > padded:
            raise ValueError(f"{reqs} requests of {q} rows in a width {width}")
        for gi, (g, bt) in enumerate(self.block_tables.items()):
            self.slot_mappings[gi].fill_(PAD_SLOT)
            for r in range(padded):
                if r >= reqs:
                    bt[r].zero_()
                    continue
                L = seq_lens[r]
                n = -(-L // spec.KV_BLOCK)
                ids = list(pages[g][r][:n])
                if g in self.sliding:  # block_table.py: num_computed_tokens - reach
                    null = max(L - q - (spec.WINDOW - 1), 0) // spec.KV_BLOCK
                    ids[:null] = [0] * null
                bt[r, :n] = torch.tensor(ids, dtype=torch.int32)
                if stale is not None:
                    bt[r, n:] = stale
                for i in range(q):
                    pos = L - q + i
                    blk = pages[g][r][pos // spec.KV_BLOCK]
                    self.slot_mappings[gi, r * q + i] = (
                        blk * spec.KV_BLOCK + pos % spec.KV_BLOCK
                    )
        self.seq_lens.zero_()
        self.seq_lens[:reqs] = torch.tensor(seq_lens, dtype=torch.int32)
        self.positions[:width].zero_()
        self.is_padding[:width].fill_(True)
        for r, L in enumerate(seq_lens):
            self.positions[r * q : (r + 1) * q] = torch.arange(L - q, L)
            self.is_padding[r * q : (r + 1) * q] = False
        return self._view(width, q)
