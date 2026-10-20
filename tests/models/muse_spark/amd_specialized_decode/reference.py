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
"""

from collections.abc import Sequence

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
