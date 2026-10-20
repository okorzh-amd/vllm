# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2 (one Muse Spark layer from the gated attention core to the deferred FFN
boundary) on MI355X against the ops vLLM runs for the same decode step, random
weights in the post-load formats at TP8 shapes.

[CPU] The torch reference (``reference.py``): the RoundUp quant and exact
E2M1 rounding, the lowest-index serial routing, pad rows, the TP sharding of
o_proj and the experts at toy shapes; the host readback of every mailbox
region against ``spec``'s pair formulas.

[TP8] Each stage is checked tightly on its own inputs as the kernel saw them
(the mailbox readbacks), so an eps, gain-center or order mistake in one stage
cannot hide behind upstream noise; the chain (``moe``, ``branch_out``) against
vLLM's stock decode. Every case has an eps-scaled row (the attention output,
the pooled rows, one route's expert rows) and one case all-zero norm weights
(gain ``w + 1``). Then: the ranks agree bit for bit; one CUDA graph of two
steps at two widths replayed 300 times, bit-identical; pad rows (NaN inputs)
come out as exact zeros and leave the live rows unchanged; an all-pad step
completes; a stalled rank fails the step without a hang; the timing gate
against the stock span; the warm-up compile; the startup self-test.
"""

import contextlib
import statistics
from types import SimpleNamespace

import pytest
import ray
import torch
import torch.distributed as dist

from tests.models.muse_spark.amd_specialized_decode import reference as ref
from tests.utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)
from vllm.models.muse_spark.amd.specialized_decode.mono import spec
from vllm.models.muse_spark.amd.specialized_decode.mono.abi import (
    FAILED,
    PEER_HEAD,
    RECORD_WORDS,
    STARTUP_WAIT_SECONDS,
    WAIT_BUDGET,
    WAIT_SECONDS,
)
from vllm.platforms import current_platform

bf = torch.bfloat16
f32 = torch.float32
H, HM, CORE, INTER = spec.HIDDEN, spec.MOE_HIDDEN, spec.CORE, spec.INTER
E, TOPK, TP, MT, MP = spec.EXPERTS, spec.TOPK, spec.TP, spec.MAX_TOKENS, spec.MAX_PICKS
SLOT = spec.k2_slot(1)  # the first kernel layer's (layer 0 stays stock)
STALLED = 3
COS_STAGE = 0.99999
COS_CHAIN = 0.9999
REL_CHAIN = 1e-2
CASES = ("base", "eps", "zero_w")
EPS_EXPERT = 17  # the "eps" case's forced expert, its w2 scaled by 2^-EPS_W2_SHIFT
# unscaled, a route's R at these weights has mean square ~370 (CPU reference,
# 8 ranks): 2^-17 puts mean((R (w + 1))^2) at ~2e-8, i.e. at POST_EXP_EPS
EPS_W2_SHIFT = 17


def _on_gfx950() -> bool:
    if not torch.cuda.is_available() or not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


needs_gfx950 = pytest.mark.skipif(
    not _on_gfx950(), reason="the Muse mono kernels need gfx950 (MI355X)"
)


# ---------------------------------------------------------------- [CPU]
# The reference's rules.


def test_reference_quant_rules():
    """RoundUp codes (AITER's HIP quant) and E2M1 nearest-even, against hand
    values; where the naive rules (checkpoint "even" scale, an odd
    penalty of 1e-6) differ."""
    amax = torch.tensor([6.0, 6.0 * 2.0**-3, 6.0001, 6.5, 0.0, 1e-30, 448.0])
    assert ref.e8m0_roundup_code(amax).tolist() == [127, 124, 128, 128, 92, 92, 134]
    assert ref.FLOOR_CODE == 92

    # the checkpoint's "even" weight-scale rule keeps
    # 6.5 at scale 1 (code 127) and saturates it; RoundUp moves to scale 2
    bits = (torch.tensor([6.5]).view(torch.int32) + 0x200000) & 0x7F800000
    even = torch.floor(torch.log2(bits.view(f32))) - 2 + 127
    assert even.item() == 127 and ref.e8m0_roundup_code(torch.tensor([6.5])) == 128

    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    v = torch.tensor(grid + [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, 1e9])
    want = list(range(8)) + [0, 2, 2, 4, 4, 6, 6, 7, 7]
    assert ref.e2m1_codes(v).tolist() == want
    assert ref.e2m1_codes(-v).tolist() == [c | 8 for c in want]
    # just below a midpoint rounds down (fp4_round's 1e-6 odd penalty rounded
    # 0.75 - 2^-22 up to 1.0)
    d = 2.0**-22
    below = torch.tensor([0.75 - d, 1.75 - d, 3.5 - d * 4])
    assert ref.e2m1_codes(below).tolist() == [1, 3, 5]
    assert ref.e2m1_codes(torch.tensor([-0.0, -0.1])).tolist() == [8, 8]

    # quantize / dequantize: grid values times a power of two are exact
    g = torch.Generator().manual_seed(1)
    x = torch.tensor(grid)[torch.randint(0, 8, (4, 64), generator=g)]
    x = x * torch.where(torch.rand(4, 64, generator=g) < 0.5, -1.0, 1.0) * 2.0**-5
    x[:, 0] = 6.0 * 2.0**-5  # every group's amax a 6 2^k
    x[:, 32] = 6.0 * 2.0**-5
    packed, codes = ref.mx_quant(x)
    assert packed.shape == (4, 32) and codes.shape == (4, 2)
    assert torch.equal(ref.mx_dequant(packed, codes).float(), x)
    # random rows: within half an E2M1 step of the scale (relative to amax)
    y = torch.randn(3, 4096, generator=g)
    deq = ref.mx_dequant(*ref.mx_quant(y)).view(3, -1, 32)
    amax = y.view(3, -1, 32).abs().amax(-1, keepdim=True)
    assert ((deq - y.view(3, -1, 32).double()).abs() <= amax / 3 + 1e-12).all()


def test_reference_quant_matches_aiter():
    """The E2M1 rounding and packing are aiter's (torchao RNE) on midpoints,
    sub-midpoints and saturating values of both signs."""
    fp4_utils = pytest.importorskip("aiter.utility.fp4_utils")
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    mids = (grid[1:] + grid[:-1]) / 2
    e = torch.tensor(2.0**-20)
    v = torch.cat([grid, mids, mids - e, mids + e, torch.tensor([7.0, 100.0, 0.1])])
    v = torch.cat(
        [v, -v, torch.randn(4096, generator=torch.Generator().manual_seed(2))]
    )
    if v.numel() % 2:
        v = torch.cat([v, torch.zeros(1)])
    nib = ref.e2m1_codes(v)
    ours = (nib[0::2] | (nib[1::2] << 4)).to(torch.uint8)
    assert torch.equal(ours, fp4_utils.f32_to_mxfp4(v).view(torch.uint8))


def test_reference_routing_rules():
    """8 argmax steps, the lowest index on a tie; a serial pick-order sum plus
    1e-15; pad rows -1 / 0 whatever their logits; NaN scores sort last."""
    S = 4
    logits = torch.zeros(S, E)
    bias = torch.zeros(E)
    logits[1] = torch.randn(E, generator=torch.Generator().manual_seed(3))
    bias[200] = 1.0  # row 0's ties: 200 first, then 0 .. 6
    logits[2, :] = float("nan")
    logits[2, 40:48] = 2.0  # only 8 finite scores
    logits[3] = float("nan")
    pad = torch.tensor([False, False, False, True])
    ids, w = ref.route(logits, bias, pad)
    assert ids[0].tolist() == [200, 0, 1, 2, 3, 4, 5, 6]
    assert ids[2].tolist() == list(range(40, 48))
    assert ids[3].tolist() == [-1] * TOPK and (w[3] == 0).all()
    # row 1 (no ties): the top-8 of sigmoid + bias, in descending order
    key = ref.sigmoid(logits[1]) + bias
    assert ids[1].tolist() == torch.topk(key, TOPK).indices.tolist()
    # the weights: sig * (1 / total), total summed serially in pick order
    sig = ref.sigmoid(logits[1])[ids[1].long()]
    total = torch.tensor(0.0)
    for k in range(TOPK):
        total = total + sig[k]
    inv = torch.tensor(1.0) / (total + torch.tensor(1e-15, dtype=f32))
    assert torch.equal(w[1], sig * inv)
    # the sigmoid is tl.sigmoid's fp32 math, close to torch's
    # (the f32 rounding of x log2e moves exp by ~|x| 6e-8 relative)
    x = torch.linspace(-30, 30, 1001)
    torch.testing.assert_close(ref.sigmoid(x), torch.sigmoid(x), rtol=1e-5, atol=0)


def _toy(tp: int, seed: int = 0):
    """K2's inputs at toy shapes (H 64, a rank's CORE 16, Hm 64, I 32 a rank,
    256 experts so the routing is the real one), TP ``tp``."""
    g = torch.Generator().manual_seed(seed)
    h, core_r, hm, i_r, S = 64, 16, 64, 32, 3

    def rn(*shape, s=1.0):
        return torch.randn(*shape, generator=g) * s

    def u8(*shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, generator=g, dtype=torch.int64).to(
            torch.uint8
        )

    w = {
        "attn_alpha": rn(h).abs() * 0.5 + 0.5,
        "attn_beta": rn(h).abs() * 0.5,
        "w_preffn": (rn(h) * 0.1).to(bf),
        "w_router": rn(E, h, s=0.2),
        "bias": rn(E, s=0.02),
        "w_pool": (rn(hm, h) / 8).to(bf),
        "w_preexp": (rn(hm) * 0.1).to(bf),
        "w_postexp": (rn(hm) * 0.1).to(bf),
        "w_unpool": (rn(h, hm) / 8).to(bf),
        "w_o": [(rn(h, core_r) / 8).to(bf) for _ in range(tp)],
        "w13": [u8(E, 2 * i_r, hm // 2) for _ in range(tp)],
        "c13": [u8(E, 2 * i_r, hm // 32, lo=118, hi=123) for _ in range(tp)],
        "w2": [u8(E, hm, i_r // 2) for _ in range(tp)],
        "c2": [u8(E, hm, i_r // 32, lo=118, hi=123) for _ in range(tp)],
    }
    core = [rn(S, core_r).to(bf) for _ in range(tp)]
    stream = rn(S, h)
    return w, core, stream


def _unshard(w, core):
    """The TP2 toy weights as one rank's: o_proj K and the experts'
    intermediate concatenated (gate rows then up rows of each rank)."""
    i_r = w["w13"][0].shape[1] // 2
    one = dict(w)
    one["w_o"] = [torch.cat(w["w_o"], 1)]
    for n in ("w13", "c13"):
        gates = [x[:, :i_r] for x in w[n]]
        ups = [x[:, i_r:] for x in w[n]]
        one[n] = [torch.cat(gates + ups, 1)]
    one["w2"] = [torch.cat(w["w2"], 2)]
    one["c2"] = [torch.cat(w["c2"], 2)]
    return one, [torch.cat(core, 1)]


def test_reference_shards_like_one_rank():
    """TP2 against the same layer on one rank: the o_proj partials and the
    experts' down partials sum to the unsharded results within their bf16
    roundings (MXFP4 groups never straddle a shard: 32 | I a rank)."""
    w, core, stream = _toy(2)
    pad = torch.tensor([False, False, False])
    two = ref.k2(core, stream, pad, w)
    w1, core1 = _unshard(w, core)
    one = ref.k2(core1, stream, pad, w1)
    assert ref.cos(two["attn_out"], one["attn_out"]) > 0.9999
    # the experts on the same a1 rows and picks
    xe, ids = one["xe"], one["ids"]
    rows2 = [
        ref.expert_rows(xe, ids, w["w13"][r], w["c13"][r], w["w2"][r], w["c2"][r])
        for r in range(2)
    ]
    rows1 = one["rows"][0]
    assert ref.rel_l2(ref.owner_sum(rows2), rows1.float()) < 1e-2
    assert ref.cos(ref.owner_sum(rows2), rows1.float()) > 0.9999
    # the finalize of either: one fp32 R, so the same moe up to R's roundings
    # (the chains themselves may pick another expert: at toy widths one
    # a1 code flip moves a logit)
    moe2 = ref.finalize(ref.owner_sum(rows2), w["w_postexp"], one["tw"])
    assert ref.cos(moe2, one["moe"]) > 0.9999
    assert ref.cos(two["stream_out"], one["stream_out"]) > 0.9999


def test_reference_pad_rows_are_zero_and_isolated():
    """A pad row with NaN inputs: exact zeros in every row-wise output, -1 ids;
    the live rows bit-equal to the same step without the pad row."""
    w, core, stream = _toy(2, seed=4)
    pad = torch.tensor([False, False, True])
    dirty = [c.clone() for c in core]
    for c in dirty:
        c[2] = float("nan")
    s_dirty = stream.clone()
    s_dirty[2] = float("nan")
    got = ref.k2(dirty, s_dirty, pad, w)
    for name in ("stream_out", "ffn_in", "xe", "moe", "branch"):
        assert (got[name][2].float() == 0).all(), name
    assert (got["ids"][2] == -1).all() and (got["tw"][2] == 0).all()
    live = ref.k2([c[:2] for c in core], stream[:2], pad[:2], w)
    for name in ("stream_out", "ffn_in", "logits", "ids", "tw", "moe", "branch"):
        assert torch.equal(got[name][:2], live[name]), name


def test_reference_finalize_and_gain():
    """The finalize's top-k tree and selections; every norm's gain is ``w +
    1`` (an all-zero weight is the plain norm) and its eps is the stage's."""
    g = torch.Generator().manual_seed(5)
    R = torch.randn(2, TOPK, 32, generator=g)
    tw = torch.rand(2, TOPK, generator=g)
    tw[0, 3] = 0.0
    R[0, 3] = float("nan")  # a pick of weight 0: selected out
    zero = torch.zeros(32, dtype=bf)
    moe = ref.finalize(R, zero, tw)
    assert torch.isfinite(moe.float()).all()
    x = R.double()
    xs = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + spec.POST_EXP_EPS)
    xs = torch.nan_to_num(xs * tw.double().unsqueeze(-1), nan=0.0)
    assert ref.cos(moe, xs.sum(1)) > 0.99999
    # eps: a row with mean(x^2) ~ 1e-8 sees POST_ATTN_EPS (1e-8), not 1e-5
    attn = torch.full((1, 64), 1e-4).to(bf)
    s = torch.zeros(1, 64)
    one = torch.ones(64)
    out, _ = ref.boundary(attn, s, one, one, torch.zeros(64, dtype=bf))
    want = 1e-4 / (1e-8 + 1e-8) ** 0.5
    assert abs(out[0, 0].item() - want) < 1e-3 * want
    assert torch.equal(
        ref.topk_tree(torch.arange(8.0).view(1, 8)), torch.tensor([28.0])
    )


# The host readback of the mailbox regions (the TP8 tests' helpers), against
# spec's pair formulas on a fake runner.


def _region(runner, name: str) -> tuple[torch.Tensor, int]:
    """(byte buffer, byte offset) of ``name`` as the kernels address it: a
    scratch region, or a peer region in the current epoch's parity half."""
    lay = spec.scratch_layout()
    if name in lay:
        return runner.scratch, lay[name][0]
    e = int(runner.epoch.item()) & spec.EPOCH_MASK
    off = PEER_HEAD + (e & 1) * spec.peer_half_bytes() + spec.peer_layout()[name][0]
    return runner.peers.bytes, off


def _pairs(runner, name: str, index: int, n: int):
    """(values, tags) of pairs ``index`` .. ``index + n`` of region ``name``
    (host copies)."""
    from vllm.models.muse_spark.amd.specialized_decode.mono.common.debug import (
        read_pairs,
    )

    buf, off = _region(runner, name)
    v, t = read_pairs(buf, off + 8 * index, n)
    return v.cpu(), t.cpu()


def _tag(runner) -> int:
    return spec.step_tag(int(runner.epoch.item()), SLOT)


def readback(runner, S: int) -> dict:
    """Every mailbox region of the last launch (rows < S), decoded, and the
    tags each pair carried: ``ar`` bf16 [TP, S, H] (src's partial), ``ass`` /
    ``bss`` fp32 [NA, S], ``rlog`` fp32 [S, E] (expert 32 src + i), ``route``
    (ids int32 [S, 8], weights fp32), ``pool`` bf16 [S, HM], ``rsx`` bf16 [TP,
    S, 8, OWN_COLS] (src's down rows of this rank's columns), ``pss`` fp32
    [RED_TASKS, S, 8], ``sqx`` fp32 [TP, S, 8], ``moe`` bf16 [S, HM],
    ``branch`` bf16 [S, H]; plain ``xrow`` bf16 [S, H] and ``xqd`` (packed a1
    bytes uint8 [S, HM / 2], codes int32 [S, HM / 32])."""
    out, tags = {}, {}

    def rows(name, n_rows, row_pairs):
        v, t = _pairs(runner, name, 0, n_rows * row_pairs)
        tags[name] = t.view(n_rows, row_pairs)
        return v.view(n_rows, row_pairs)

    out["ar"] = rows("ar", TP * MT, H // 2).view(TP, MT, H // 2)[:, :S]
    tags["ar"] = tags["ar"].view(TP, MT, -1)[:, :S]
    out["ar"] = out["ar"].contiguous().view(bf).view(TP, S, H)
    for name in ("ass", "bss"):
        v = rows(name, spec.NA, MT)
        out[name] = v[:, :S].contiguous().view(f32)
        tags[name] = tags[name][:, :S]
    v = rows("rlog", TP * MT, spec.EXPERTS_RANK).view(TP, MT, -1)[:, :S]
    tags["rlog"] = tags["rlog"].view(TP, MT, -1)[:, :S]
    out["rlog"] = v.contiguous().view(f32).permute(1, 0, 2).reshape(S, E)
    v = rows("route", MT, 16)[:S]
    tags["route"] = tags["route"][:S]
    out["ids"] = v[:, :TOPK].clone()
    out["tw"] = v[:, TOPK:].contiguous().view(f32)
    for name, width in (("pool", HM), ("moe", HM), ("branch", H)):
        v = rows(name, MT, width // 2)[:S]
        tags[name] = tags[name][:S]
        out[name] = v.contiguous().view(bf).view(S, width)
    v = rows("rsx", TP * MP, spec.OWN_COLS // 2).view(TP, MP, -1)[:, : S * TOPK]
    tags["rsx"] = tags["rsx"].view(TP, MP, -1)[:, : S * TOPK]
    out["rsx"] = v.contiguous().view(bf).view(TP, S, TOPK, spec.OWN_COLS)
    v = rows("pss", spec.RED_TASKS, MP)[:, : S * TOPK]
    tags["pss"] = tags["pss"][:, : S * TOPK]
    out["pss"] = v.contiguous().view(f32).view(spec.RED_TASKS, S, TOPK)
    v = rows("sqx", TP, MP)[:, : S * TOPK]
    tags["sqx"] = tags["sqx"][:, : S * TOPK]
    out["sqx"] = v.contiguous().view(f32).view(TP, S, TOPK)
    lay = spec.scratch_layout()
    o = lay[spec.XROW][0]
    xrow = runner.scratch[o : o + spec.XROW_BYTES].cpu().view(bf).view(MT, H)
    out["xrow"] = xrow[:S].clone()
    o = lay[spec.XQD][0]
    words = runner.scratch[o : o + MT * spec.XQD_ROW_WORDS * 4].cpu().view(torch.int32)
    words = words.view(MT, spec.XQD_ROW_WORDS)[:S]
    out["xq"] = words[:, : spec.XQ_WORDS].contiguous().view(torch.uint8)
    out["xsc"] = words[:, spec.XQ_WORDS :].clone()
    out["tags"] = tags
    return out


def test_readback_matches_the_pair_formulas():
    """``readback`` decodes each region at spec's pair indexes (a fake runner
    whose pairs hold their own index and tag), parity halves included."""
    for epoch in (6, 7):
        runner = SimpleNamespace(
            scratch=torch.zeros(spec.scratch_bytes(), dtype=torch.uint8),
            peers=SimpleNamespace(
                bytes=torch.zeros(PEER_HEAD + spec.peer_bytes(), dtype=torch.uint8)
            ),
            epoch=torch.tensor([epoch], dtype=torch.int32),
        )
        tag = _tag(runner)
        for name, (_, nbytes) in {
            **spec.scratch_layout(),
            **spec.peer_layout(),
        }.items():
            if name in (spec.XROW, spec.XQD):
                continue
            buf, off = _region(runner, name)
            n = nbytes // 8
            pairs = buf[off : off + nbytes].view(torch.int32).view(n, 2)
            pairs[:, 0] = torch.arange(n, dtype=torch.int32)
            pairs[:, 1] = tag
        S = 3
        rb = readback(runner, S)
        for name, t in rb["tags"].items():
            assert (t == tag).all(), name

        def idx(x):
            return x.contiguous().view(torch.int32)

        t, col, src, i, pick, cc, j = 2, 130, 5, 7, 13, 34, 9
        assert idx(rb["ar"][src, t, col : col + 2]).item() == spec.ar_pair(src, t, col)
        assert idx(rb["rlog"][t, 32 * src + i]).item() == spec.rlog_pair(src, t, i)
        assert rb["ids"][t, 3].item() == spec.route_id_pair(t, 3)
        assert idx(rb["tw"][t, 3]).item() == spec.route_weight_pair(t, 3)
        assert idx(rb["pool"][t, col : col + 2]).item() == spec.pool_pair(t, col)
        assert idx(rb["moe"][t, col : col + 2]).item() == spec.moe_pair(t, col)
        assert idx(rb["branch"][t, col : col + 2]).item() == spec.branch_pair(t, col)
        r = rb["rsx"][src, pick // TOPK, pick % TOPK, cc : cc + 2]
        assert idx(r).item() == spec.rsx_pair(src, pick, cc)
        assert idx(rb["sqx"][src, 1, 5]).item() == spec.sqx_pair(src, 13)
        assert idx(rb["pss"][j, 1, 5]).item() == spec.pss_pair(j, 13)
        assert idx(rb["ass"][j, t]).item() == spec.ss_pair(j, t)


# ---------------------------------------------------------------- [TP8]


def _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
    return device


def _gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The unquantized GEMM dispatch of vLLM (``rocm_unquantized_gemm`` on gfx950),
    as ``UnquantizedLinearMethod.apply`` runs a linear layer's weight: the o_proj
    reference only."""
    from vllm.model_executor.layers.utils import dispatch_unquantized_gemm

    return dispatch_unquantized_gemm()(torch.nn.Module(), x, w, None)


def _linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The pooler's and the unpooler's GEMM: a plain ``F.linear``, as
    ``MuseSparkTorchLinearMethod`` runs them (amd/model.py: not the AITER
    tuned-GEMM dispatch, whose split-K kernels accumulate in bf16)."""
    return torch.nn.functional.linear(x, w)


@contextlib.contextmanager
def _tp_checks(what: str):
    """Rank-local checks as one TP verdict. A check that fails on some ranks
    only must not leave the others in the next collective: vLLM's custom
    all-reduce and NCCL spin with no timeout, and the exclusive node hangs.
    Every rank enters and leaves the block at the same point; on exit the ranks
    vote over the TP CPU (gloo) group and all raise if any failed (the failing
    ranks their own error). No collective inside the block."""
    from vllm.distributed import get_tp_group
    from vllm.models.muse_spark.amd.specialized_decode.runtime.peer import tp_agree

    err: Exception | None = None
    try:
        yield
    except Exception as e:  # voted below, then raised on every rank
        err = e
    if not tp_agree(err is None, get_tp_group().cpu_group):
        if err is not None:
            raise err
        raise AssertionError(f"another TP rank failed {what}")


def _weights(rank: int, device, case: str = "base") -> dict:
    """One rank's K2 weights (post-load formats; replicated tensors from seed
    7, the rank's shards from 100 + rank) and the experts' unshuffled bytes
    and codes (``ckpt``). ``eps``: the pooler scaled so mean(pooled^2) ~
    PRE_EXP_EPS, expert EPS_EXPERT always picked and its w2 scaled by
    2^-EPS_W2_SHIFT (that route's mean square ~ POST_EXP_EPS), the residual
    gate -2.5 (beta ~ 2.4e-4, see ``_inputs``). ``zero_w``:
    every norm weight zero."""
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    from vllm.models.muse_spark.common.layers import (
        MuseSpark1xNormPreservingResidualConnection,
    )

    gc = torch.Generator(device=device).manual_seed(7)
    gr = torch.Generator(device=device).manual_seed(100 + rank)

    def rn(g, *shape, s=1.0, dtype=bf):
        return (torch.randn(*shape, generator=g, device=device) * s).to(dtype)

    def u8(g, *shape, lo=0, hi=256):
        return torch.randint(
            lo, hi, shape, generator=g, device=device, dtype=torch.int64
        ).to(torch.uint8)

    conn = MuseSpark1xNormPreservingResidualConnection(H).to(device)
    with torch.no_grad():
        conn.gate.copy_(rn(gc, H, dtype=f32))
        if case == "eps":
            # beta = sigmoid(-2.5 / 0.3) ~ 2.4e-4 (alpha ~ 1) on every channel:
            # the pre-FFN norm's input of a row with a small stream is small
            # too (``_inputs``)
            conn.gate.fill_(-2.5)
    conn.materialize_coefficients()
    w = {
        "attn_alpha": conn.alpha.contiguous(),
        "attn_beta": conn.beta.contiguous(),
        "w_preffn": rn(gc, H, s=0.1),
        # the router multiplier folded in (0.1768) at load
        "w_router": rn(gc, E, H, s=0.02 * 0.1768, dtype=f32),
        "bias": rn(gc, E, s=0.02, dtype=f32),
        "w_pool": rn(gc, HM, H, s=H**-0.5),
        "w_preexp": rn(gc, HM, s=0.1),
        "w_postexp": rn(gc, HM, s=0.1),
        "w_unpool": rn(gc, H, HM, s=HM**-0.5),
        "w_o": rn(gr, H, CORE, s=(TP * CORE) ** -0.5),
    }
    w13 = u8(gr, E, 2 * INTER, HM // 2)
    c13 = u8(gr, E, 2 * INTER, HM // 32, lo=118, hi=123)
    w2 = u8(gr, E, HM, INTER // 2)
    c2 = u8(gr, E, HM, INTER // 32, lo=118, hi=123)
    if case == "eps":
        w["w_pool"] = (w["w_pool"].float() * 3e-3).to(bf)
        w["bias"][EPS_EXPERT] = 10.0
        c2[EPS_EXPERT] -= EPS_W2_SHIFT
    if case == "zero_w":
        for n in ("w_preffn", "w_preexp", "w_postexp"):
            w[n] = torch.zeros_like(w[n])

    def shuf_scale(c):
        # e8m0_shuffle on the [E N, K / 32] view of the uint8 codes, the dtype
        # Quark loads them in (oracle/mxfp4.py AITER_MXFP4_MXFP4 branch): the
        # stock MoE gets exactly the tensor the served layer holds
        s0, s1, n = c.shape
        return e8m0_shuffle(c.reshape(s0 * s1, n)).view(s0, s1, -1).contiguous()

    w["w13"] = shuffle_weight(w13, layout=(16, 16)).contiguous()
    w["w2"] = shuffle_weight(w2, layout=(16, 16)).contiguous()
    # the stock MoE takes typed views (fp4x2 weights, e8m0 scales); the
    # kernel the same bytes by address
    w["w13_fp4"] = w["w13"].view(torch.float4_e2m1fn_x2)
    w["w2_fp4"] = w["w2"].view(torch.float4_e2m1fn_x2)
    # as oracle/mxfp4.py marks the loaded weights: aiter's fused_moe picks its
    # kernel (and the weight layout it assumes) from this attribute
    w["w13_fp4"].is_shuffled = True
    w["w2_fp4"].is_shuffled = True
    w["w13s"] = shuf_scale(c13)
    w["w2s"] = shuf_scale(c2)
    w["ckpt"] = (w13, c13, w2, c2)
    return w


def _inputs(rank: int, device, S: int, case: str = "base"):
    """(core [S, CORE] bf16 of this rank, stream [S, H] fp32 replicated). The
    "eps" case scales row 0's core on every rank by 1e-4: mean(attn_out^2) ~
    POST_ATTN_EPS; and the last row's stream by 3e-3: with that case's beta ~
    2.4e-4 the pre-FFN norm's input alpha stream + beta y has mean square
    ~9e-6 ~ PRE_FFN_EPS."""
    gc = torch.Generator(device=device).manual_seed(11 + S)
    gr = torch.Generator(device=device).manual_seed(200 + 8 * S + rank)
    core = torch.randn(S, CORE, generator=gr, device=device)
    if case == "eps":
        core[0] *= 1e-4
    stream = torch.randn(S, H, generator=gc, device=device)
    if case == "eps":
        stream[S - 1] *= 3e-3
    return core.to(bf), stream


def _pad(device, rows=()) -> torch.Tensor:
    """The pad flags: whole words (``ld_pad`` reads the word of a row's byte)."""
    p = torch.zeros(16, dtype=torch.bool, device=device)
    for r in rows:
        p[r] = True
    return p


KERNEL_WEIGHTS = (
    "w_o",
    "attn_alpha",
    "attn_beta",
    "w_preffn",
    "w_router",
    "bias",
    "w_pool",
    "w_preexp",
    "w13",
    "w13s",
    "w2",
    "w2s",
    "w_postexp",
    "w_unpool",
)


def _launch(runner, w, core, stream, pad, key=None, *, begin=True):
    """One K2 step (the epoch bumped first): -> (stream_out, branch_out)."""
    from vllm.models.muse_spark.amd.specialized_decode.mono import layer

    S = core.shape[0]
    key = key or spec.K2Build(tokens=S, **runner.fault)
    stream_out = torch.empty_like(stream)
    branch_out = torch.empty(S, H, dtype=bf, device=stream.device)
    if begin:
        runner.step_begin()
    layer.k2_launch(
        key,
        {
            **{n: w[n] for n in KERNEL_WEIGHTS},
            "core": core,
            "stream": stream,
            "pad": pad,
            "stream_out": stream_out,
            "branch_out": branch_out,
            "slot": SLOT,
            **runner.launch_args(),
            **runner.peer_args(),
        },
    )
    return stream_out, branch_out


def _assert_healthy(runner) -> None:
    """No wait gave up, no alarm: FAILED and every CTA's record zero (the
    header's WAIT_BUDGET word holds the budget), the output rank's flags 0.
    Collective (``_tp_checks``): only the output rank's flags are ever set
    (``runner.alarm``), so the local verdicts differ by construction."""
    with _tp_checks("the health check"):
        h = runner.health.cpu().clone()
        assert h[FAILED] == 0, h[RECORD_WORDS:].view(-1, RECORD_WORDS).nonzero()
        h[WAIT_BUDGET] = 0
        assert not h.any()
        assert not runner.flags.any()


def _gather(x: torch.Tensor) -> list[torch.Tensor]:
    out = [torch.empty_like(x) for _ in range(TP)]
    dist.all_gather(out, x.contiguous())
    return out


def _all_ranks_equal(x: torch.Tensor, what: str) -> None:
    """Every rank gathers the same tensors, so the verdict is the same on every
    rank (no vote needed)."""
    outs = _gather(x)
    assert all(torch.equal(outs[0], o) for o in outs), f"ranks disagree on {what}"


def _stock_boundary(attn, stream, w):
    from vllm.models.muse_spark.common.ops.fused_rmsnorm import (
        fused_norm_residual_norm_fwd,
    )

    ffn_in, stream_out = fused_norm_residual_norm_fwd(
        attn,
        stream,
        None,
        w["w_preffn"],
        spec.POST_ATTN_EPS,
        spec.PRE_FFN_EPS,
        alpha=w["attn_alpha"],
        beta=w["attn_beta"],
        post_gain_center=0.0,
        pre_gain_center=spec.GAIN,
        out_dtype=bf,
    )
    return ffn_in, stream_out


def _route_ids(ids: torch.Tensor, device) -> torch.Tensor:
    """The kernel's picks (``route`` readback, pads -1) as the stock MoE's
    ``[S 8, 1]`` int32 ids, a pad row's picks on expert 0 (only live rows are
    compared)."""
    flat = ids.to(device=device, dtype=torch.int32).clamp(min=0)
    return flat.reshape(-1, 1).contiguous()


def _stock_experts(ffn_in, flat_ids, tw, w, unit=None):
    """The routed experts vLLM runs, on ``ffn_in`` with the given picks
    (``flat_ids`` int32 [S 8, 1] on the device: the kernel's, so routing cannot
    differ): pooler, the expanded pre-expert norm, vLLM's AITER MoE op with the
    arguments ``rocm_aiter_fused_experts`` passes for Muse's MXFP4 w4a4 layers
    (one unit-weight top-1 route a row, as ``MuseSparkAiterExperts.apply``
    calls it), the rank-sum in fp32, the post-expert finalize.
    ``unit``: the cached ones buffer (``MuseSparkAiterExperts._unit_weights``),
    allocated here when None. -> (rows bf16 [S, 8, HM] of this rank, moe bf16
    [S, HM])."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
        ActivationMethod,
        QuantMethod,
    )
    from vllm.models.muse_spark.amd.ops.fused_norm_expand import (
        fused_rmsnorm_expand,
        take_expanded,
    )
    from vllm.models.muse_spark.common.ops.fused_finalize import (
        fused_post_expert_finalize,
    )

    S = ffn_in.shape[0]
    xe = fused_rmsnorm_expand(
        _linear(ffn_in, w["w_pool"]),
        w["w_preexp"],
        spec.PRE_EXP_EPS,
        spec.GAIN,
        False,
        TOPK,
    )
    expanded = take_expanded(xe, TOPK)
    if unit is None:
        unit = torch.ones(S * TOPK, 1, dtype=f32, device=ffn_in.device)
    # rocm_aiter_fused_experts for Muse (MXFP4 w4a4, SiLU, no bias, no pads:
    # hidden_pad = intermediate_pad = 0, gate_mode "" = AITER's "separated")
    rows = rocm_aiter_ops.fused_moe(
        expanded,
        w["w13_fp4"],
        w["w2_fp4"],
        unit,
        flat_ids,
        activation_method=ActivationMethod.SILU.value,
        quant_method=QuantMethod.BLOCK_1X32.value,
        w1_scale=w["w13s"],
        w2_scale=w["w2s"],
        output_dtype=bf,
        moe_sorting_dispatch_policy=rocm_aiter_ops.get_moe_dispatch_policy(),
        swiglu_limit=0.0,
    ).view(S, TOPK, HM)
    R = tensor_model_parallel_all_reduce(rows.float())
    moe = torch.empty(S, HM, dtype=bf, device=ffn_in.device)
    fused_post_expert_finalize(
        R, w["w_postexp"], tw, spec.POST_EXP_EPS, spec.GAIN, True, moe
    )
    return rows, moe


def _eps_bites(ms: torch.Tensor, eps: float, other: float) -> torch.Tensor:
    """How much the norm scale ``rsqrt(m + eps)`` of rows with mean square
    ``ms`` moves when ``eps`` is swapped for ``other`` (a factor >= 1)."""
    ms = ms.double()
    hi, lo = max(eps, other), min(eps, other)
    return torch.sqrt((ms + hi) / (ms + lo))


def _assert_eps_rows_bite(attn_k, stream_out, rb, R, w) -> None:
    """The "eps" case's premise (design 6.2 (i)): its scaled rows sit where a
    swapped 1e-8 / 1e-5 eps moves the norm by O(1), so the tight per-stage
    checks would catch the swap -- the attention output's row 0, the pre-FFN
    norm's input of the last row, the pooled rows, the forced expert's routes.
    A calibration that misses this makes the case vacuous, so it fails here
    instead of passing silently."""
    big, small = spec.PRE_FFN_EPS, spec.POST_ATTN_EPS  # 1e-5, 1e-8
    a = attn_k[0].double().square().mean()
    assert _eps_bites(a, small, big).item() > 1.25, ("attn row 0", a.item())
    n = stream_out[-1].double().square().mean()
    assert _eps_bites(n, big, small).item() > 1.25, ("pre-FFN last row", n.item())
    pm = rb["pool"].double().square().mean(-1)
    assert (_eps_bites(pm, big, small) > 1.25).all(), ("pooled rows", pm)
    hit = (rb["ids"] == EPS_EXPERT).to(R.device)
    x = R.double() * (w["w_postexp"].double() + spec.GAIN)
    xm = x.square().mean(-1)[hit]
    assert xm.numel() > 0 and (_eps_bites(xm, small, big) > 1.25).all(), (
        "forced route",
        xm,
    )


def _check_step(runner, w, core, stream, pad, stream_out, branch_out, case) -> dict:
    """Every stage of the last launch against vLLM's ops on the inputs the
    kernel saw (design 6.2); returns the readback. The checks are rank-local
    (each rank its own shard) and the references collective, so every check
    block is voted (``_tp_checks``) before the next collective."""
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.model_executor.layers.fused_moe.router.rocm_fp32_router_gemm import (
        rocm_fp32_router_gemm,
    )
    from vllm.models.muse_spark.common.ops.fused_finalize import (
        fused_post_expert_finalize,
    )
    from vllm.models.muse_spark.common.ops.fused_routing import (
        fused_musespark1x_routing,
    )

    dev = core.device
    rank = runner.rank
    S = core.shape[0]
    p = pad[:S].cpu()
    live = ~p
    lv = live.to(dev)
    where = f"{case} S={S}"
    _assert_healthy(runner)
    rb = readback(runner, S)
    tag = _tag(runner)
    # ---- o_proj: this rank's partial vs the stock GEMM; attn_out = the
    # partials in rank order (the seam's input), vs the stock all-reduce
    part = _gemm(core, w["w_o"])
    with _tp_checks(f"{where}: tags, o_proj"):
        for name, t in rb["tags"].items():
            assert (t == tag).all(), (case, S, name, t[t != tag][:4])
        # pad rows: the kernel pushes zeros (load_core selects them), stock
        # computes them
        ref.check_gemv(
            rb["ar"][rank][live], part[lv], core[lv], w["w_o"], what=f"{case} o_proj"
        )
        assert (rb["ar"][:, p].float() == 0).all(), case
    attn_k = ref.attn_out(list(rb["ar"])).to(dev)
    attn_stock = tensor_model_parallel_all_reduce(part.clone())
    _all_ranks_equal(rb["ar"].to(dev), "ar")
    with _tp_checks(f"{where}: boundary, router, routing, pooler"):
        assert ref.cos(attn_k[lv], attn_stock[lv]) > COS_STAGE
        # the seam's sums of squares: the 64 slices' partials
        ss = attn_k.double().square().sum(-1).cpu()
        torch.testing.assert_close(
            rb["ass"].double().sum(0)[live], ss[live], rtol=1e-5, atol=0
        )
        # ---- boundary on the kernel's own attn_out
        ffn_ref, stream_ref = _stock_boundary(attn_k, stream, w)
        got_s, ref_s = stream_out.cpu(), stream_ref.cpu()
        x = attn_k.float().cpu()
        y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + spec.POST_ATTN_EPS)
        mag = (w["attn_alpha"].cpu() * stream.cpu()).abs() + (
            w["attn_beta"].cpu() * y
        ).abs()
        err = (got_s - ref_s).abs()
        assert (err[live] <= 1e-5 * mag[live] + 1e-30).all(), (case, err[live].max())
        assert (got_s[p] == 0).all()
        ns = got_s.double()
        torch.testing.assert_close(
            rb["bss"].double().sum(0)[live],
            ns.square().sum(-1)[live],
            rtol=1e-5,
            atol=0,
        )
        rs = torch.rsqrt(ns.square().mean(-1, keepdim=True) + spec.PRE_FFN_EPS)
        own = (ns * rs * (w["w_preffn"].cpu().double() + spec.GAIN)).float().to(bf)
        assert ref.bf16_ulps(rb["xrow"][live], own[live]).max() <= 1, case
        u = ref.bf16_ulps(rb["xrow"][live], ffn_ref.cpu()[live])
        assert (u == 0).float().mean().item() >= 0.999, case
        big = ref_s.abs() >= 1e-3 * mag
        assert (u[big[live]] <= 1).all(), case
        assert (rb["xrow"][p].float() == 0).all()
        # chain guard: against the stock all-reduce's boundary
        _, stream_stock = _stock_boundary(attn_stock, stream, w)
        assert ref.cos(got_s[live], stream_stock.cpu()[live]) > COS_STAGE
        # ---- router logits on the kernel's own ffn_in: the README
        # tolerance, except where the two fp32 orders differ only by
        # accumulation noise on a cancelling dot (test_probe_pool's rule)
        xrow = rb["xrow"].to(dev)
        logits = rocm_fp32_router_gemm(xrow, w["w_router"]).cpu()
        absdot = rb["xrow"].double().abs() @ w["w_router"].double().abs().T.cpu()
        close = torch.isclose(rb["rlog"], logits, rtol=1e-5, atol=1e-6)
        noise = (rb["rlog"].double() - logits.double()).abs() <= 2**-20 * absdot
        assert (close | noise)[live].all(), (case, "router")
        assert close[live].float().mean().item() >= 0.99, (case, "router")
        # ---- routing on the kernel's own logits
        tw_s, ids_s = fused_musespark1x_routing(
            rb["rlog"].to(dev),
            w["bias"],
            TOPK,
            spec.ROUTE_SCALE,
            router_score_division_eps=spec.ROUTE_EPS,
            is_padding=pad[:S].contiguous(),
        )
        assert torch.equal(rb["ids"], ids_s.cpu()), case
        assert torch.equal(rb["tw"], tw_s.cpu().float()), case
        if case == "eps":
            assert (rb["ids"][live] == EPS_EXPERT).any(-1).all()
        # ---- pooler on the kernel's own ffn_in
        pooled = _linear(xrow, w["w_pool"]).cpu()
        ref.check_gemv(
            rb["pool"][live],
            pooled[live],
            rb["xrow"][live].float(),
            w["w_pool"].cpu(),
            what=f"{case} pool",
        )
        assert ref.cos(rb["pool"][live], pooled[live]) > COS_STAGE
    # ---- the owner reduce: R = the 8 ranks' rows of this rank's columns in
    # rank order (fp32, never rounded); the finalize on the gathered R
    R_own = ref.owner_sum(list(rb["rsx"].float()))  # [S, 8, 512]
    R = torch.cat(_gather(R_own.to(dev)), -1)  # [S, 8, HM], rank r's 512 r ..
    with _tp_checks(f"{where}: owner reduce, finalize"):
        sq = ref.owner_sumsq(R.cpu(), w["w_postexp"].cpu())
        for src in range(TP):
            lo = src * spec.OWN_COLS
            part_sq = ref.owner_sumsq(
                R[..., lo : lo + spec.OWN_COLS].cpu(),
                w["w_postexp"][lo : lo + spec.OWN_COLS].cpu(),
            )
            torch.testing.assert_close(
                rb["sqx"][src][live], part_sq[live], rtol=1e-5, atol=1e-30
            )
        torch.testing.assert_close(
            rb["sqx"].double().sum(0)[live], sq.double()[live], rtol=1e-5, atol=1e-30
        )
        torch.testing.assert_close(
            rb["pss"].double().sum(0)[live],
            rb["sqx"][rank].double()[live],
            rtol=1e-5,
            atol=1e-30,
        )
        moe_ref = torch.empty(S, HM, dtype=bf, device=dev)
        fused_post_expert_finalize(
            R,
            w["w_postexp"],
            rb["tw"].to(dev),
            spec.POST_EXP_EPS,
            spec.GAIN,
            True,
            moe_ref,
        )
        ref.check_ulps(rb["moe"][live], moe_ref.cpu()[live], what=f"{case} finalize")
        assert (rb["moe"][p].float() == 0).all()
        if case == "eps":
            _assert_eps_rows_bite(attn_k, got_s, rb, R, w)
    # ---- the experts' chain (stock MoE on the kernel's ffn_in and picks)
    rows_s, moe_s = _stock_experts(
        xrow, _route_ids(rb["ids"], dev), rb["tw"].to(dev), w
    )
    mine = rb["rsx"].to(dev)  # [src, t, k, this rank's 512]: src's rows
    # every rank's stock rows (its expert shard), then this rank's columns:
    # rsx[src] holds src's rows of the columns THIS rank owns
    lo = rank * spec.OWN_COLS
    rows_all = torch.stack(_gather(rows_s.contiguous()))[..., lo : lo + spec.OWN_COLS]
    moe_k = rb["moe"].to(dev)
    with _tp_checks(f"{where}: experts, unpooler"):
        assert ref.cos(mine[:, lv], rows_all[:, lv]) > COS_CHAIN, case
        assert ref.cos(moe_k[lv], moe_s[lv]) > COS_CHAIN, case
        assert ref.rel_l2(moe_k[lv], moe_s[lv]) < REL_CHAIN, case
        # ---- unpooler on the kernel's own moe; gather = the branch rows
        branch = _linear(moe_k, w["w_unpool"]).cpu()
        ref.check_gemv(
            branch_out.cpu()[live],
            branch[live],
            rb["moe"][live].float(),
            w["w_unpool"].cpu(),
            what=f"{case} unpool",
        )
        assert torch.equal(branch_out.cpu(), rb["branch"]), case
        assert (branch_out.cpu()[p].float() == 0).all()
        branch_s = _linear(moe_s, w["w_unpool"])
        assert ref.cos(branch_out[lv], branch_s[lv]) > COS_CHAIN, case
        assert ref.rel_l2(branch_out[lv], branch_s[lv]) < REL_CHAIN, case
    # ---- every rank the same rows
    for name, t in (
        ("stream_out", stream_out),
        ("branch_out", branch_out),
        ("xrow", xrow),
        ("route", torch.cat([rb["ids"], rb["tw"].view(torch.int32)], -1).to(dev)),
        ("moe", moe_k),
    ):
        _all_ranks_equal(t, name)
    return rb


def _check_a1_dump(runner, w, S: int) -> None:
    """A debug build's ``xqd``: CTA 0's a1 rows against the RoundUp reference
    of the stock pre-expert norm on the kernel's own pooled rows. The kernel's
    normed rows are not exposed in K2 (``test_quant_probe`` checks those bytes
    bit-exact on the kernel's own rows), so a group whose bf16 rows differ by
    an ulp may differ: >= 99.9 % of the groups bit-exact, the rest within one
    E2M1 step of the scale."""
    from vllm.models.muse_spark.amd.ops.fused_norm_expand import (
        fused_rmsnorm_expand,
    )

    rb = readback(runner, S)
    xe = fused_rmsnorm_expand(
        rb["pool"].to(w["w_preexp"].device),
        w["w_preexp"],
        spec.PRE_EXP_EPS,
        spec.GAIN,
        False,
        TOPK,
    ).cpu()
    q, c = ref.mx_quant(xe)
    same_code = (rb["xsc"] == c).float().mean().item()
    groups_k = rb["xq"].view(S, -1, 16)
    groups_r = q.view(S, -1, 16)
    same_group = ((groups_k == groups_r).all(-1) & (rb["xsc"] == c)).float().mean()
    assert same_code >= 0.999 and same_group.item() >= 0.999, (same_code, same_group)
    dk = ref.mx_dequant(rb["xq"], rb["xsc"])
    dr = xe.double()
    amax = dr.view(S, -1, 32).abs().amax(-1).repeat_interleave(32, -1)
    assert ((dk - dr).abs() <= amax / 2 + 1e-30).all()


@ray.remote(num_gpus=1, max_calls=1)
def _k2_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.muse_spark.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    # the first launch of each build compiles it: the startup budget
    runner.set_wait_budget(STARTUP_WAIT_SECONDS)
    for case in CASES:
        w = _weights(rank, device, case)
        for S in (1, 3, 8):
            core, stream = _inputs(rank, device, S, case)
            pad = _pad(device)
            stream_out, branch_out = _launch(runner, w, core, stream, pad)
            torch.accelerator.synchronize()
            _check_step(runner, w, core, stream, pad, stream_out, branch_out, case)
        if case == "base":
            # the a1 bytes (debug build: CTA 0 dumps its LDS rows)
            core, stream = _inputs(rank, device, 8)
            _launch(
                runner,
                w,
                core,
                stream,
                _pad(device),
                spec.K2Build(tokens=8, debug_dump=True),
            )
            torch.accelerator.synchronize()
            _assert_healthy(runner)
            with _tp_checks("the a1 dump"):
                _check_a1_dump(runner, w, 8)

    # One CUDA graph of two steps at two widths (each step's epoch bump
    # captured with it), replayed: every replay matches the first.
    w = _weights(rank, device)
    c1, s1 = _inputs(rank, device, 1)
    c8, s8 = _inputs(rank, device, 8)
    pad = _pad(device)
    outs = {}
    _launch(runner, w, c1, s1, pad)  # warm-up (code objects loaded)
    _launch(runner, w, c8, s8, pad)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs[1] = _launch(runner, w, c1, s1, pad)
        outs[8] = _launch(runner, w, c8, s8, pad)
    graph.replay()
    torch.accelerator.synchronize()
    want = {s: [t.clone() for t in o] for s, o in outs.items()}
    for _ in range(300):
        graph.replay()
    torch.accelerator.synchronize()
    with _tp_checks("the graph replays"):
        for s, o in outs.items():
            assert all(torch.equal(a, b) for a, b in zip(o, want[s])), s
    _assert_healthy(runner)


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_k2_tp8_matches_vllm_ops_and_replays(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _k2_worker)


@ray.remote(num_gpus=1, max_calls=1)
def _pad_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.muse_spark.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    # the first launch of each build compiles it: the startup budget
    runner.set_wait_budget(STARTUP_WAIT_SECONDS)
    w = _weights(rank, device)
    core, stream = _inputs(rank, device, 8)
    pad = _pad(device, (5, 6, 7))
    clean_s, clean_b = _launch(runner, w, core, stream, pad)
    torch.accelerator.synchronize()
    _assert_healthy(runner)
    _check_step(runner, w, core, stream, pad, clean_s, clean_b, "pad")
    dirty_c, dirty_s = core.clone(), stream.clone()
    dirty_c[5:] = float("nan")
    dirty_s[5:] = float("nan")
    dirty_c[6, 3] = float("inf")
    got_s, got_b = _launch(runner, w, dirty_c, dirty_s, pad)
    torch.accelerator.synchronize()
    _assert_healthy(runner)
    with _tp_checks("the NaN pad rows"):
        assert (got_s[5:] == 0).all() and (got_b[5:].float() == 0).all()
        assert torch.equal(got_s[:5], clean_s[:5])
        assert torch.equal(got_b[:5], clean_b[:5])
        rb = readback(runner, 8)
        assert (rb["ids"][5:] == -1).all() and (rb["tw"][5:] == 0).all()
        assert (rb["moe"][5:].float() == 0).all()
    # every row a pad row (a capture warm-up's step): nothing in the union,
    # the launch completes with zero outputs
    for S in (1, 8):
        core, stream = _inputs(rank, device, S)
        all_pad = torch.ones(16, dtype=torch.bool, device=device)
        s_out, b_out = _launch(runner, w, core, stream, all_pad)
        torch.accelerator.synchronize()
        _assert_healthy(runner)
        with _tp_checks(f"the all-pad step S={S}"):
            assert (s_out == 0).all() and (b_out.float() == 0).all(), S


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_pad_rows_are_finite_zeros(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _pad_worker)


def _stock_span(w, core, stream, pad, out: dict, unit: torch.Tensor) -> None:
    """The decode vLLM runs from o_proj's input to the unpooler's output (design
    7.0): o_proj + all-reduce, the boundary, the router GEMM and routing, the
    pooler, the expanded pre-expert norm, aiter's MoE, the fp32 all-reduce,
    the finalize, the unpooler -- and nothing else: the routing's ids go to the
    MoE as a view and ``unit`` is the experts' cached ones buffer, as in the
    served layer. Writes ``out["stream"]``, ``out["branch"]``."""
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.model_executor.layers.fused_moe.router.rocm_fp32_router_gemm import (
        rocm_fp32_router_gemm,
    )
    from vllm.models.muse_spark.common.ops.fused_routing import (
        fused_musespark1x_routing,
    )

    S = core.shape[0]
    attn = tensor_model_parallel_all_reduce(_gemm(core, w["w_o"]))
    ffn_in, stream_out = _stock_boundary(attn, stream, w)
    logits = rocm_fp32_router_gemm(ffn_in, w["w_router"])
    tw, ids = fused_musespark1x_routing(
        logits,
        w["bias"],
        TOPK,
        spec.ROUTE_SCALE,
        router_score_division_eps=spec.ROUTE_EPS,
        is_padding=pad[:S],
    )
    _, moe = _stock_experts(ffn_in, ids.view(S * TOPK, 1), tw, w, unit)
    out["stream"] = stream_out
    out["branch"] = _linear(moe, w["w_unpool"])


def _replay_ms(graph, n: int = 300) -> float:
    """Median of ``n`` replays, each between two CUDA events."""
    ev = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(n)
    ]
    for a, b in ev:
        a.record()
        graph.replay()
        b.record()
    torch.accelerator.synchronize()
    return statistics.median(a.elapsed_time(b) for a, b in ev)


@ray.remote(num_gpus=1, max_calls=1)
def _timing_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.distributed.parallel_state import graph_capture
    from vllm.models.muse_spark.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    w = _weights(rank, device)
    gate = {1: 0.6, 8: 0.8}
    for S, ratio in gate.items():
        core, stream = _inputs(rank, device, S)
        pad = _pad(device)
        unit = torch.ones(S * TOPK, 1, dtype=f32, device=device)
        # each width's first launch compiles its build (a cross-rank compile
        # skew): the startup budget, the serving budget only for the graphs
        runner.set_wait_budget(STARTUP_WAIT_SECONDS)
        _launch(runner, w, core, stream, pad)  # warm-ups: compile, load, JIT
        out: dict = {}
        _stock_span(w, core, stream, pad, out, unit)
        torch.accelerator.synchronize()
        _assert_healthy(runner)
        runner.set_wait_budget(WAIT_SECONDS)
        with graph_capture(device=device) as ctx:
            mono = torch.cuda.CUDAGraph()
            with torch.cuda.graph(mono, stream=ctx.stream):
                _launch(runner, w, core, stream, pad)
            stock = torch.cuda.CUDAGraph()
            with torch.cuda.graph(stock, stream=ctx.stream):
                _stock_span(w, core, stream, pad, out, unit)
        t = torch.tensor([_replay_ms(mono), _replay_ms(stock)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)  # the slowest rank's
        _assert_healthy(runner)
        k2_ms, stock_ms = t.tolist()
        print(
            f"S={S}: K2 {k2_ms * 1e3:.1f} us, stock span {stock_ms * 1e3:.1f} us",
            flush=True,
        )
        assert k2_ms <= ratio * stock_ms, (S, k2_ms, stock_ms)


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_k2_is_faster_than_the_stock_span(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _timing_worker)


@ray.remote(num_gpus=1, max_calls=1)
def _fault_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.muse_spark.amd.specialized_decode.mono.common.debug import (
        given_up_waits,
    )
    from vllm.models.muse_spark.amd.specialized_decode.mono.common.sync import (
        WAIT_TICKS,
    )
    from vllm.models.muse_spark.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    # rank STALLED stalls 1.5 serving budgets in the epoch-2 launch of slot
    # SLOT (the epoch-1 launch compiles the build: the startup budget)
    runner.fault = {
        "fault_rank": STALLED,
        "fault_epoch": 2,
        "fault_slot": SLOT,
        "fault_ticks": WAIT_TICKS * 3 // 2,
    }
    w = _weights(rank, device)
    core, stream = _inputs(rank, device, 8)
    pad = _pad(device)
    _launch(runner, w, core, stream, pad)  # epoch 1: no stall
    torch.accelerator.synchronize()
    _assert_healthy(runner)
    runner.set_wait_budget(WAIT_SECONDS)
    _launch(runner, w, core, stream, pad)  # epoch 2: every launch returns
    torch.accelerator.synchronize()
    waits = given_up_waits(runner.health.cpu(), spec.REGIONS)
    gave_up = torch.tensor([int(bool(waits))], device=device)
    dist.all_reduce(gave_up)
    assert gave_up.item() > 0, "no rank's wait gave up"
    if rank == 0:
        assert runner.flags.any(), "the output rank's flags missed the failure"


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_stalled_rank_fails_the_step_without_a_hang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _fault_worker)


@needs_gfx950
def test_warmup_compiles_without_launching() -> None:
    """The JIT warm-up compiles every build before capture without running it
    (a launch on its null arguments would fault)."""
    from vllm.models.muse_spark.amd.specialized_decode.mono.common.sync import (
        WAIT_TICKS,
    )
    from vllm.models.muse_spark.amd.specialized_decode.runtime.layers import (
        compile_build,
    )

    fault = dict(fault_rank=STALLED, fault_epoch=2, fault_slot=SLOT)
    for s in range(1, spec.MAX_ROWS + 1):
        compile_build(spec.K2Build(tokens=s))
    compile_build(spec.K2Build(tokens=8, fault_ticks=WAIT_TICKS * 3 // 2, **fault))
    compile_build(spec.K2Build(tokens=8, debug_dump=True))
    torch.accelerator.synchronize()


@ray.remote(num_gpus=1, max_calls=1)
def _self_test_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = _init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.muse_spark.amd.specialized_decode.runtime.decode import (
        SpecializedDecode,
    )
    from vllm.models.muse_spark.amd.specialized_decode.runtime.layers import (
        compile_build,
    )
    from vllm.models.muse_spark.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    w = _weights(rank, device)

    def kernel(width: int, idx: int):
        # a KernelLayer's launch contract: static arguments and builds
        static = {n: w[n] for n in KERNEL_WEIGHTS} | {
            "slot": spec.k2_slot(idx),
            **runner.launch_args(),
            **runner.peer_args(),
        }
        return SimpleNamespace(
            static=static, builds=lambda: [spec.K2Build(tokens=width, **runner.fault)]
        )

    sd = object.__new__(SpecializedDecode)
    sd.runner = runner
    sd.kernels = {width: {i: kernel(width, i) for i in (1, 2)} for width in (1, 8)}
    sd.self_test()  # all-pad then live, each width: passes on every rank
    _assert_healthy(runner)
    assert int(runner.health[WAIT_BUDGET]) == STARTUP_WAIT_SECONDS
    sd.after_capture()
    assert int(runner.health[WAIT_BUDGET]) == WAIT_SECONDS
    # a stalled rank in the next self-test step: every rank raises (the
    # fault builds compiled first, so the 2 s budget sees the stall, not a
    # compile skew between the ranks)
    runner.fault = {
        "fault_rank": STALLED,
        "fault_epoch": int(runner.epoch.item()) + 1,
        "fault_slot": spec.k2_slot(1),
        "fault_ticks": 3 * 100_000_000,  # 3 s of s_memrealtime
    }
    for by in sd.kernels.values():
        for build in by[1].builds():
            compile_build(build)
    dist.barrier()
    runner.set_wait_budget(2)
    with pytest.raises(ValueError, match="gave up waits|another TP rank"):
        sd.self_test()


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_self_test_and_after_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _self_test_worker)
