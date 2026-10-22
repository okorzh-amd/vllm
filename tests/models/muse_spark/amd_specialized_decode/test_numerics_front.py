# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1 (``mono/k1.py``, Phase B1, DESIGN_BC 5, WP-B4) in a kernel layer's chain
on MI355X: K1 -> the stock attention core -> K2(gated), against the ops vLLM
runs for the same decode step, random weights at TP8 shapes (9.3).

The reference is composed from ops, as Phase A's (``test_numerics``):
``fused_norm_residual_norm_fwd`` (the B_ffn boundary) -> the unquantized GEMM
dispatch (``dispatch_unquantized_gemm``: wvSplitK / hipBLASLt) -> the fused
QK-norm / RoPE / KV insert (``fused_muse_spark_qknorm_rope_kv_insert``) ->
``paged_attention_v1`` (a sliding layer on ``_build_swa_decode_tables``'
window-trimmed table) -> ``rms_norm_gated`` -> the stock span from o_proj to
the unpooler (``test_numerics._stock_span``). The mono side runs vLLM's
core on K1's own rows and cache, as a B1 deployment does.

[CPU] The cases hold what they claim (slots inside each row's pages, at the
row's position, a context past the window); the K1 launch carries exactly
``spec.K1_ARGS``.
[TP8] One sliding (RoPE) and one full (NoPE) layer at widths 1, 3, 8. Each
K1 stage tight on the kernel's own inputs: ``stream_out`` within the
magnitude-scaled bound of DESIGN_BC 4, the ``arow`` rows <= 1 bf16 ulp from
the fp64 pre-qkv norm of the kernel's own ``stream_out`` and bit-equal to
stock on >= 99.9 %, the GEMV rows (``qkvd``, a debug build) vs fp64 and vs
the GEMM dispatch, ``qkvo`` live rows and the whole cache ``torch.equal`` to
the stock insert run on ``qkvd``. The chain: K2's ``stream_out`` cos >
0.99999 against the stock chain, ``branch_out`` cos > 0.9999 against
vLLM's experts and unpooler on the kernel's own ``ffn_in`` rows and picks
(Phase A's reference, ``test_numerics._check_step``), live rows; the ranks
agree bit for bit; NaN pad rows (slot -1) leave the live rows and the cache
unchanged and come out as zeros; one CUDA graph of the chain (K1, the stock
core, K2) replayed 300 times bit-identical, the cache included; no wait gives
up. A stalled K1 (``fault_slot`` = layer 1's K1) fails the step without a
hang: every launch returns, the stalled rank's given-up waits name a K1
region, the output rank's flags are set, and the stalled rank's cache is
byte-identical (a FAILED rank inserts no K / V). Diagnostic: K1 against the
stock front (boundary + qkv GEMM + insert) in CUDA graphs, rotating 8 weight
sets; printed only (the B1 gate is a serving span, DESIGN_BC 11.2).
"""

import os
from types import SimpleNamespace

import pytest
import ray
import regex as re
import torch
import torch.distributed as dist

from tests.models.muse_spark.amd_specialized_decode import reference as ref
from tests.models.muse_spark.amd_specialized_decode import test_numerics as tn
from tests.utils import multi_gpu_test, multi_process_parallel
from vllm.models.muse_spark.amd.specialized_decode.mono import spec
from vllm.models.muse_spark.amd.specialized_decode.mono.abi import (
    STARTUP_WAIT_SECONDS,
    WAIT_SECONDS,
)

bf = torch.bfloat16
f32 = torch.float32
H, CORE, QKV, MT = spec.HIDDEN, spec.CORE, spec.QKV_ROWS, spec.MAX_TOKENS
STALLED = tn.STALLED
WIDTHS = (1, 3, 8)
# TUNE L1: the K1 tuning variant (``K1Build.layout``) the K1 builds here use:
# MUSE_K1_LAYOUT, read at call time in each Ray worker (a local Ray cluster's
# workers inherit the driver's environment as of ``ray.init``); the default
# is the E3 build. The ``*_k1_variant`` / ``*_r256`` tests set it; set by
# hand it moves every test here.
K1_VARIANT_ENV = {"layout": "MUSE_K1_LAYOUT"}
K1_VARIANT_DEFAULT = {"layout": "g192"}
# the non-default layouts the TP8 chain runs
K1_VARIANTS = ("r256",)


def _k1_variant() -> dict:
    """This process's K1 variant fields (``K1Build`` keyword arguments)."""
    return {
        f: os.environ.get(e, K1_VARIANT_DEFAULT[f]) for f, e in K1_VARIANT_ENV.items()
    }


def _use_k1_variant(monkeypatch: pytest.MonkeyPatch, layout: str) -> None:
    """Select a K1 variant for this test's Ray workers (before ``ray.init``)."""
    monkeypatch.setenv(K1_VARIANT_ENV["layout"], layout)


def _k1_regions() -> re.Pattern:
    """A given-up wait record on one of this variant's K1 regions."""
    names = spec.K1_REGIONS_BY_LAYOUT[_k1_variant()["layout"]]
    return re.compile(rf": ({'|'.join(names)}) pair")


KINDS = ("sliding", "full")
# the kernel layer each kind runs as: layer 1 is full, 2 sliding (DESIGN_BC
# 3.1), so each has its own K1 / K2 tag slots
LAYER = {"full": 1, "sliding": 2}
MAX_POS = 1 << 20  # the served RoPE table's rows
BLOCKS = 1024  # a rank's cache: 8 MiB
# each row's context length, this step's token included: the window's edges
# (a sliding layer's keys k >= L - 2048) and pages' edges
LENS = {
    1: (2100,),
    3: (1, 17, 2049),
    8: (1, 16, 17, 255, 300, 2048, 2049, 2500),
}
PAD_ROWS = (5, 6, 7)  # the pad case's rows (S = 8)
COS_STREAM = 0.99999
# branch_out against vLLM's experts on the kernel's own ffn_in rows and picks,
# as Phase A (test_numerics._check_step, COS_CHAIN): not against the stock
# chain, which is itself order-dependent there. vLLM's GEMM dispatch gives the
# same rows other bits at another width (S = 8 batched vs one row at a time),
# and the MXFP4 experts turn 1-ulp ffn_in flips into whole-stock branch cos
# 0.9986 at the qkv GEMM and 0.9994 at the o_proj GEMM alone (rows down to
# 0.9968) while stream_out stays > 0.9999999 (wp_logs_bc/DEBUG.md (b)). K1's
# GEMV is held to the dispatch in _check_front, the seam's ffn_in by Phase A.
COS_BRANCH = 0.9999
# K1's arguments besides the step's tensors and the launch state
K1_WEIGHTS = (
    "w_postffn",
    "ffn_alpha",
    "ffn_beta",
    "w_preqkv",
    "w_qkv",
    "cos_sin",
    "max_pos",
)

needs_gfx950 = tn.needs_gfx950


# ---------------------------------------------------------------- the cases


def _case(S: int, pad_rows=(), seed: int = 0, device="cpu") -> SimpleNamespace:
    """One step's per-row metadata: ``lens`` (host ints, this token included),
    ``positions`` = L - 1, distinct pages a row (``reference.assign_pages``,
    never block 0), the block table int32 [S, pages], ``slots`` = the page of
    the row's position x 16 + its offset; pad rows as MRV2 leaves them (slot
    -1, position 0, length 0). The tensors on ``device`` (a CUDA graph
    captures no host copy)."""
    g = torch.Generator().manual_seed(41 + 8 * S + seed)
    lens = list(LENS[S])
    pages = ref.assign_pages(lens, BLOCKS, generator=g)
    bt = torch.zeros(S, max(len(p) for p in pages), dtype=torch.int32)
    for t, p in enumerate(pages):
        bt[t, : len(p)] = torch.tensor(p, dtype=torch.int32)
    pos = torch.tensor([L - 1 for L in lens], dtype=torch.int64)
    slots = bt[torch.arange(S), pos // spec.KV_BLOCK].long() * spec.KV_BLOCK
    slots += pos % spec.KV_BLOCK
    for t in pad_rows:
        lens[t], pos[t], slots[t] = 0, 0, ref.PAD_SLOT
    return SimpleNamespace(
        S=S,
        lens=lens,
        seq_lens=torch.tensor(lens, dtype=torch.int32, device=device),
        bt=bt.to(device),
        positions=pos.to(device),
        slots=slots.to(device),
        pad_rows=tuple(pad_rows),
    )


def _front_weights(rank: int, device, seed: int = 0) -> dict:
    """One rank's K1 weights: the previous layer's ``post_ffn_norm`` weight
    and ``add_ffn`` coefficients and this layer's ``pre_qkv_proj_norm``
    weight (replicated, seed 17 + seed), this rank's qkv_proj shard bf16
    [2304, 8192] (seed 100 + rank + 8 seed: DESIGN_BC 9.3)."""
    from vllm.models.muse_spark.common.layers import (
        MuseSpark1xNormPreservingResidualConnection,
    )

    gc = torch.Generator(device=device).manual_seed(17 + seed)
    gr = torch.Generator(device=device).manual_seed(100 + rank + 8 * seed)

    def rn(g, *shape, s=1.0, dtype=bf):
        return (torch.randn(*shape, generator=g, device=device) * s).to(dtype)

    conn = MuseSpark1xNormPreservingResidualConnection(H).to(device)
    with torch.no_grad():
        conn.gate.copy_(rn(gc, H, dtype=f32))
    conn.materialize_coefficients()
    return {
        "w_postffn": rn(gc, H, s=0.1),
        "ffn_alpha": conn.alpha.contiguous(),
        "ffn_beta": conn.beta.contiguous(),
        "w_preqkv": rn(gc, H, s=0.1),
        "w_qkv": rn(gr, QKV, H, s=H**-0.5),
    }


def _with_table(fw: dict, table: torch.Tensor | None) -> dict:
    """``fw`` with a layer's RoPE table (None: a NoPE layer, 0 and 0 rows)."""
    return {
        **fw,
        "cos_sin": 0 if table is None else table,
        "max_pos": 0 if table is None else int(table.shape[0]),
    }


def _step_inputs(device, S: int, pad_rows=()) -> tuple[torch.Tensor, torch.Tensor]:
    """(branch bf16, stream fp32) [S, H], replicated (the previous layer's
    deferred boundary); pad rows NaN."""
    g = torch.Generator(device=device).manual_seed(11 + S)
    branch = torch.randn(S, H, generator=g, device=device) * 0.7
    stream = torch.randn(S, H, generator=g, device=device) * 2.0
    for t in pad_rows:
        branch[t] = float("nan")
        stream[t] = float("nan")
    return branch.to(bf), stream


def _k1_args(runner, fw, kv, case, dev, branch, stream, slot):
    """K1's arguments (``spec.K1_ARGS``): the layer's weights, the cache's
    (``runtime.layers.kv_args``: byte strides), the step's tensors, the
    launch state; -> (args, stream_out, qkvo)."""
    from vllm.models.muse_spark.amd.specialized_decode.runtime.layers import kv_args

    S = case.S
    stream_out = torch.empty_like(stream)
    qkvo = torch.empty(S, QKV, dtype=bf, device=dev)
    args = {
        **{n: fw[n] for n in K1_WEIGHTS},
        **kv_args(kv),
        "branch": branch,
        "stream": stream,
        "pad": tn._pad(dev, case.pad_rows),
        "positions": case.positions.to(dev),
        "slots": case.slots.to(dev),
        "stream_out": stream_out,
        "qkvo": qkvo,
        "slot": slot,
        "rank": runner.rank,
        **runner.launch_args(),
    }
    return args, stream_out, qkvo


# ---------------------------------------------------------------- [CPU]


def test_cases_hold_what_they_claim():
    """Every live slot is in its row's last page at its position; slots are
    distinct and never in block 0; some row's context passes the window and
    some ends on a page edge; pad rows are MRV2's."""
    for S in WIDTHS:
        for pad_rows in ((), PAD_ROWS if S == 8 else ()):
            c = _case(S, pad_rows)
            live = [t for t in range(S) if t not in pad_rows]
            for t in live:
                L = c.lens[t]
                assert c.positions[t] == L - 1
                page = int(c.bt[t, (L - 1) // spec.KV_BLOCK])
                assert page > 0
                assert int(c.slots[t]) == page * 16 + (L - 1) % 16
            ls = c.slots[live].tolist()
            assert len(set(ls)) == len(ls)
            for t in pad_rows:
                assert c.slots[t] == ref.PAD_SLOT and c.lens[t] == 0
    big = _case(8)
    assert max(big.lens) > spec.WINDOW and spec.WINDOW in big.lens
    assert any(L % spec.KV_BLOCK == 0 for L in big.lens)
    assert any(L % spec.KV_BLOCK == 1 for L in big.lens)  # a new page


def test_k1_variant_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """The variant tests' plumbing: unset, the E3 build; each parametrized
    variant a valid, distinct K1 build whose stall regex names its own
    partials region."""
    for e in K1_VARIANT_ENV.values():
        monkeypatch.delenv(e, raising=False)
    assert _k1_variant() == K1_VARIANT_DEFAULT
    assert spec.K1Build(tokens=1, rope=True, **_k1_variant()) == spec.K1Build(1, True)
    assert _k1_regions().search("x: qpart pair 3")
    names = set()
    for layout in K1_VARIANTS:
        _use_k1_variant(monkeypatch, layout)
        key = spec.K1Build(tokens=8, rope=False, **_k1_variant())
        assert key.layout == layout
        names.add(key.kernel_name())
        part = spec.K1_QPART[layout]
        assert _k1_regions().search(f"x: {part} pair 3")
        if layout == "r256":
            assert not _k1_regions().search("x: qpart pair 3")
    assert len(names) == len(K1_VARIANTS)


def test_k1_launch_carries_the_abi():
    """The test's K1 launch dict holds exactly ``spec.K1_ARGS`` (byte
    strides from the runtime's ``kv_args``)."""
    state = dict.fromkeys(("scratch", "epoch", "health", "alarm"), 0)
    runner = SimpleNamespace(rank=3, launch_args=lambda: state)
    kv = torch.zeros(4, spec.KV_HEADS, spec.KV_BLOCK, spec.KV_C, dtype=bf)
    case = _case(3)
    fw = _with_table({n: 0 for n in K1_WEIGHTS[:5]}, None)
    branch, stream = torch.zeros(3, H, dtype=bf), torch.zeros(3, H)
    args, _, qkvo = _k1_args(runner, fw, kv, case, "cpu", branch, stream, 1)
    assert set(args) == set(spec.K1_ARGS)
    assert args["kv_st"] == spec.KV_C * 2 and args["kv_blocks"] == 4
    assert args["cos_sin"] == 0 and args["max_pos"] == 0
    assert qkvo.shape == (3, QKV)


# ---------------------------------------------------------------- [TP8]


def _kv(rank: int, device) -> torch.Tensor:
    """A rank's cache with every context key in place (its KV heads)."""
    g = torch.Generator().manual_seed(300 + rank)
    return ref.kv_cache(BLOCKS, null=0.0, generator=g).to(device)


def _k1(runner, fw, kv, case, branch, stream, slot, key):
    """One K1 launch: -> (stream_out, qkvo)."""
    from vllm.models.muse_spark.amd.specialized_decode.mono import k1

    args, stream_out, qkvo = _k1_args(
        runner, fw, kv, case, stream.device, branch, stream, slot
    )
    k1.k1_launch(key, args)
    return stream_out, qkvo


def _k2g(runner, w, attn, qkvo, stream, pad, slot):
    """One K2(gated) launch on (attn, qkvo): -> (stream_out, branch_out)."""
    from vllm.models.muse_spark.amd.specialized_decode.mono import layer

    S = attn.shape[0]
    stream_out = torch.empty_like(stream)
    branch_out = torch.empty(S, H, dtype=bf, device=stream.device)
    layer.k2_launch(
        spec.K2Build(tokens=S, core="gated"),
        {
            **{n: w[n] for n in tn.KERNEL_WEIGHTS},
            "attn": attn,
            "qkvo": qkvo,
            "stream": stream,
            "pad": pad,
            "stream_out": stream_out,
            "branch_out": branch_out,
            "slot": slot,
            **runner.launch_args(),
            **runner.peer_args(),
        },
    )
    return stream_out, branch_out


def _stock_core(qkvo, kv, bt, seq_lens, max_len: int, window: bool):
    """VLLM's decode core on a qkv block: ``paged_attention_v1`` on the
    cache's K / V views, a sliding layer on ``_build_swa_decode_tables``'
    window-trimmed table (window 2048), a full layer on the whole table.
    bf16 [S, CORE]."""
    import aiter  # noqa: F401

    from vllm.v1.attention.backends.rocm_aiter_fa import (
        _PARTITION_SIZE_ROCM,
        AiterFlashAttentionMetadataBuilder,
    )

    S, dev = qkvo.shape[0], kv.device
    key_cache, value_cache = kv.transpose(1, 2).split(spec.V_OFF, dim=-1)
    table, lens, sliding = bt, seq_lens, 0
    if window:
        nwb = spec.WINDOW // spec.KV_BLOCK + 1
        builder = SimpleNamespace(
            aot_sliding_window=(spec.WINDOW - 1, 0),
            block_size=spec.KV_BLOCK,
            _swa_block_table=torch.zeros(S, nwb, dtype=torch.int32, device=dev),
            _swa_seq_lens=torch.zeros(S, dtype=torch.int32, device=dev),
            _swa_block_offsets=torch.arange(nwb, dtype=torch.int64, device=dev),
        )
        table, lens, max_len = (
            AiterFlashAttentionMetadataBuilder._build_swa_decode_tables(
                builder, bt, seq_lens, S, max_len
            )
        )
        sliding = spec.WINDOW
    parts = -(-max(max_len, 1) // _PARTITION_SIZE_ROCM)
    work = torch.empty(
        S * spec.HEADS * parts * 64 * 2 + 2 * S * spec.HEADS * parts * 4,
        dtype=torch.uint8,
        device=dev,
    )
    q = qkvo[:, :CORE].reshape(S, spec.HEADS, 64).contiguous()
    out = torch.empty_like(q)
    one = torch.ones(1, dtype=f32, device=dev)
    # the kernel reads cu_query_lens[t + 1] (query_len = its difference): decode
    # passes ``query_start_loc[:num_decodes]``, a view of the [S + 1] offsets,
    # so the row past the end is real; a bare arange(S) reads past its end and
    # a stale value > 1 there skips the row (output never written)
    qsl = torch.arange(S + 1, dtype=torch.int32, device=dev)[:S]
    torch.ops.aiter.paged_attention_v1(
        out, work, q, key_cache, value_cache, spec.QK_SCALE, table,
        qsl, lens, max_len, None,
        "auto", "NHD", 0.0, one, one, None, _PARTITION_SIZE_ROCM, 1, sliding,
    )  # fmt: skip
    return out.view(S, CORE)


def _gate(attn, qkvo):
    """VLLM's gated core (o_proj's input): ``rms_norm_gated(attn, og)``."""
    from vllm.third_party.flash_linear_attention.ops.kda import (
        rms_norm_gated,
    )

    S = attn.shape[0]
    return rms_norm_gated(
        attn.view(S, spec.HEADS, 64),
        qkvo[:, spec.ROW_G :].view(S, spec.HEADS, 64),
        None,
        None,
        "sigmoid",
        eps=spec.RMS_EPS,
    ).reshape(S, CORE)


def _stock_insert(rows, positions, table, slots, kv) -> None:
    """The stock fused QK-norm / RoPE / KV insert in place on ``rows`` (bf16
    [S, QKV], contiguous) and ``kv``: the decode instance (HPG 1)."""
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


def _stock_boundary(fw, branch, stream):
    """The stock B_ffn boundary into the layer: (attn_in bf16, stream fp32)."""
    from vllm.models.muse_spark.common.ops.fused_rmsnorm import (
        fused_norm_residual_norm_fwd,
    )

    return fused_norm_residual_norm_fwd(
        branch,
        stream,
        fw["w_postffn"],
        fw["w_preqkv"],
        spec.POST_FFN_EPS,
        spec.RMS_EPS,
        alpha=fw["ffn_alpha"],
        beta=fw["ffn_beta"],
        post_gain_center=spec.GAIN,
        pre_gain_center=spec.GAIN,
        post_input_scale=True,
        out_dtype=bf,
    )


def _stock_front(fw, table, kv, case, branch, stream):
    """The stock front: (qkv rows after the insert, stream); ``kv`` in place."""
    dev = stream.device
    attn_in, stream_out = _stock_boundary(fw, branch, stream)
    qkv = tn._gemm(attn_in, fw["w_qkv"])
    _stock_insert(qkv, case.positions.to(dev), table, case.slots.to(dev), kv)
    return qkv, stream_out


def _scratch_rows(runner, name: str, cols: int, S: int) -> torch.Tensor:
    """Rows 0 .. S-1 of a plain bf16 scratch region (``arow``, ``qkvd``)."""
    off, nbytes = spec.scratch_layout()[name]
    return runner.scratch[off : off + nbytes].view(bf).view(MT, cols)[:S].cpu()


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int16)


def _check_front(runner, fw, table, kv0, kv, case, branch, stream, stream_k1, qkvo):
    """Each K1 stage on the kernel's own inputs (module docstring). Live rows;
    the pad rows' outputs are zeros. Rank-local checks (``_tp_checks``)."""
    S = case.S
    dev = stream.device
    live = torch.tensor([t not in case.pad_rows for t in range(S)])
    attn_in, ref_stream = _stock_boundary(fw, branch, stream)
    got_stream = stream_k1.cpu()
    ref_stream = ref_stream.cpu()
    # the boundary: stream_out within DESIGN_BC 4's magnitude-scaled bound
    tol = ref.stream_tolerance(
        branch.cpu(),
        stream.cpu(),
        fw["w_postffn"].cpu(),
        fw["ffn_alpha"].cpu(),
        fw["ffn_beta"].cpu(),
    )
    err = (got_stream - ref_stream).abs()
    assert (err[live] <= tol[live] + 1e-30).all(), err[live].max()
    # arow: <= 1 ulp from the fp64 pre-qkv norm of the kernel's own stream,
    # bit-equal to stock on >= 99.9 %
    arow = _scratch_rows(runner, spec.AROW, H, S)
    n = got_stream.double()
    rs = torch.rsqrt(n.square().mean(-1, keepdim=True) + spec.RMS_EPS)
    own = (n * rs * (fw["w_preqkv"].cpu().double() + spec.GAIN)).float().to(bf)
    assert ref.bf16_ulps(arow[live], own[live]).max().item() <= 1
    frac = (ref.bf16_ulps(arow[live], attn_in.cpu()[live]) == 0).float().mean()
    assert frac.item() >= 0.999, frac.item()
    # the GEMV rows: vs fp64 and the GEMM dispatch on the same arow rows
    rows = _scratch_rows(runner, spec.QKVD, QKV, S)
    w = fw["w_qkv"].cpu()
    ref64 = (arow.double() @ w.double().T).to(bf)
    ref.check_gemv(rows[live], ref64[live], arow[live], w, what=f"qkvd S={S}")
    lin = tn._gemm(arow.to(dev), fw["w_qkv"]).cpu()
    eq = (_bits(rows[live]) == _bits(lin[live])).float().mean().item()
    print(f"S={S}: qkvd == the GEMM dispatch on {eq:.6f}", flush=True)
    assert eq >= 0.99, eq
    # the epilogue: the stock insert on the kernel's own rows, same cache
    stock = rows.to(dev).contiguous()
    kv_stock = kv0.clone()
    _stock_insert(stock, case.positions.to(dev), table, case.slots.to(dev), kv_stock)
    got = qkvo.cpu()
    assert torch.equal(_bits(got[live]), _bits(stock.cpu()[live]))
    assert (_bits(got[~live]) == 0).all()
    assert (got_stream[~live] == 0).all()
    assert torch.equal(_bits(kv), _bits(kv_stock))


def _chain(runner, w, fw, table, kv, case, window, branch, stream, *, dump=False):
    """The mono chain of one kernel layer, the step's epoch bumped first:
    K1 -> vLLM's core on K1's rows and cache -> K2(gated). -> (stream_k1,
    qkvo, attn, stream_out, branch_out)."""
    S, dev = case.S, stream.device
    layer_idx = LAYER["sliding" if window else "full"]
    key = spec.K1Build(
        tokens=S, rope=window, debug_dump=dump, **runner.fault_k1, **_k1_variant()
    )
    runner.step_begin()
    stream_k1, qkvo = _k1(
        runner, fw, kv, case, branch, stream, spec.k1_slot(layer_idx), key
    )
    attn = _stock_core(
        qkvo, kv, case.bt.to(dev), case.seq_lens.to(dev), max(case.lens), window
    )
    pad = tn._pad(dev, case.pad_rows)
    out = _k2g(runner, w, attn, qkvo, stream_k1, pad, spec.k2_slot(layer_idx))
    return (stream_k1, qkvo, attn, *out)


def _own_branch(runner, w, S: int) -> torch.Tensor:
    """The branch vLLM's experts and unpooler give on the last K2 launch's own
    ``ffn_in`` rows (``xrow``) and picks (``route``): Phase A's ``branch_s``
    (``test_numerics._check_step``). Read before any later launch; collective
    (the experts all-reduce)."""
    rb = tn.readback(runner, S)
    dev = runner.scratch.device
    _, moe = tn._stock_experts(
        rb["xrow"].to(dev), tn._route_ids(rb["ids"], dev), rb["tw"].to(dev), w
    )
    return tn._linear(moe, w["w_unpool"])


def _stock_chain(w, fw, table, kv, case, window, branch, stream) -> dict:
    """The stock chain from the same inputs (``kv`` in place)."""
    S, dev = case.S, stream.device
    qkv, stream1 = _stock_front(fw, table, kv, case, branch, stream)
    attn = _stock_core(
        qkv, kv, case.bt.to(dev), case.seq_lens.to(dev), max(case.lens), window
    )
    core = _gate(attn, qkv)
    out: dict = {}
    unit = torch.ones(S * spec.TOPK, 1, dtype=f32, device=dev)
    tn._stock_span(w, core, stream1, tn._pad(dev, case.pad_rows), out, unit)
    return out


@ray.remote(num_gpus=1, max_calls=1)
def _chain_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = tn._init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.muse_spark.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    if rank == 0:
        print(f"K1 variant {_k1_variant()}", flush=True)
    # the first launch of each build compiles it: the startup budget
    runner.set_wait_budget(STARTUP_WAIT_SECONDS)
    w = tn._weights(rank, device)
    fw0 = _front_weights(rank, device)
    rope = ref.rope_table(MAX_POS, device=device)
    for kind in KINDS:
        window = kind == "sliding"
        table = rope if window else None
        fw = _with_table(fw0, table)
        for S in WIDTHS:
            case = _case(S, device=device)
            branch, stream = _step_inputs(device, S)
            kv0 = _kv(rank, device)
            kv = kv0.clone()
            stream_k1, qkvo, _, s_out, b_out = _chain(
                runner, w, fw, table, kv, case, window, branch, stream, dump=True
            )
            torch.accelerator.synchronize()
            own = _own_branch(runner, w, S)
            tn._assert_healthy(runner)
            with tn._tp_checks(f"the K1 stages ({kind}, S={S})"):
                _check_front(
                    runner, fw, table, kv0, kv, case, branch, stream, stream_k1, qkvo
                )
            kv_stock = kv0.clone()
            want = _stock_chain(w, fw, table, kv_stock, case, window, branch, stream)
            torch.accelerator.synchronize()
            with tn._tp_checks(f"the chain ({kind}, S={S})"):
                c_s = ref.cos(s_out, want["stream"])
                c_b = ref.cos(b_out, own)
                c_e = ref.cos(b_out, want["branch"])  # printed: COS_BRANCH's note
                print(
                    f"{kind} S={S}: stream_out cos {c_s:.8f}, branch cos {c_b:.7f}"
                    f" (vs the whole stock chain {c_e:.7f})"
                )
                assert c_s > COS_STREAM and c_b > COS_BRANCH, (c_s, c_b)
            for x, what in ((stream_k1, "K1 stream_out"), (s_out, "stream_out")):
                tn._all_ranks_equal(x, f"{what} ({kind}, S={S})")
            tn._all_ranks_equal(b_out, f"branch_out ({kind}, S={S})")

    # pad rows (S = 8, slot -1, length 0) with NaN inputs: the live rows and
    # the cache as with clean pad rows, the pad rows' outputs exact zeros
    window = True
    fw = _with_table(fw0, rope)
    case = _case(8, PAD_ROWS, device=device)
    clean_b, clean_s = _step_inputs(device, 8)
    dirty_b, dirty_s = _step_inputs(device, 8, PAD_ROWS)
    kv0 = _kv(rank, device)
    kv_c, kv_d = kv0.clone(), kv0.clone()
    clean = _chain(runner, w, fw, rope, kv_c, case, window, clean_b, clean_s)
    dirty = _chain(runner, w, fw, rope, kv_d, case, window, dirty_b, dirty_s)
    own = _own_branch(runner, w, 8)  # the dirty launch's
    # the stock chain all-reduces: outside the verdict block (no collective in
    # a ``_tp_checks``: a rank that failed before it would skip it)
    want = _stock_chain(w, fw, rope, kv0.clone(), case, window, clean_b, clean_s)
    torch.accelerator.synchronize()
    tn._assert_healthy(runner)
    live = [t for t in range(8) if t not in PAD_ROWS]
    with tn._tp_checks("the NaN pad rows"):
        assert torch.equal(_bits(kv_c), _bits(kv_d))
        for i in (0, 1, 3, 4):  # stream_k1, qkvo, stream_out, branch_out
            g, c = dirty[i], clean[i]
            assert torch.equal(g[live], c[live]), i
            assert (g[list(PAD_ROWS)].float() == 0).all(), i
        assert ref.cos(dirty[3][live], want["stream"][live]) > COS_STREAM
        assert ref.cos(dirty[4][live], own[live]) > COS_BRANCH

    # one CUDA graph of the chain (epoch bump, K1, the stock core, K2) at two
    # widths, replayed: every replay matches the first, the caches included
    ins = {}
    for S, kind in ((1, "full"), (8, "sliding")):
        window = kind == "sliding"
        table = rope if window else None
        ins[S] = (
            _with_table(fw0, table),
            table,
            _kv(rank, device),
            _case(S, device=device),
            window,
            *_step_inputs(device, S),
        )

    def step(S):
        fw, table, kv, case, window, branch, stream = ins[S]
        return _chain(runner, w, fw, table, kv, case, window, branch, stream)

    # warm-ups (the S = 1 NoPE K1 build compiles here: a cross-rank compile
    # skew, so the startup budget), then the serving budget for the graph
    step(1)
    step(8)
    torch.accelerator.synchronize()
    tn._assert_healthy(runner)
    runner.set_wait_budget(WAIT_SECONDS)
    graph = torch.cuda.CUDAGraph()
    outs = {}
    with torch.cuda.graph(graph):
        outs[1] = step(1)
        outs[8] = step(8)
    graph.replay()
    torch.accelerator.synchronize()
    first = {S: [t.clone() for t in o] for S, o in outs.items()}
    caches = {S: ins[S][2].clone() for S in ins}
    for _ in range(300):
        graph.replay()
    torch.accelerator.synchronize()
    with tn._tp_checks("the chain's graph replays"):
        for S, o in outs.items():
            assert all(torch.equal(a, b) for a, b in zip(o, first[S])), S
            assert torch.equal(_bits(ins[S][2]), _bits(caches[S])), S
    tn._assert_healthy(runner)


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_layer_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _chain_worker)


@ray.remote(num_gpus=1, max_calls=1)
def _stall_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = tn._init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.muse_spark.amd.specialized_decode.mono.common.debug import (
        given_up_waits,
    )
    from vllm.models.muse_spark.amd.specialized_decode.mono.common.sync import (
        WAIT_TICKS,
    )
    from vllm.models.muse_spark.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    # rank STALLED's CTA 0 stalls 1.5 serving budgets in the epoch-2 launch of
    # layer 1's K1 (a full layer: NoPE); the epoch-1 launch compiles the build
    # under the startup budget
    runner.fault_k1 = {
        "fault_rank": STALLED,
        "fault_epoch": 2,
        "fault_slot": spec.k1_slot(LAYER["full"]),
        "fault_ticks": WAIT_TICKS * 3 // 2,
    }
    w = tn._weights(rank, device)
    fw = _with_table(_front_weights(rank, device), None)
    kv = _kv(rank, device)
    branch, stream = _step_inputs(device, 8)
    case = _case(8, device=device)
    _chain(runner, w, fw, None, kv, case, False, branch, stream)  # epoch 1
    torch.accelerator.synchronize()
    tn._assert_healthy(runner)
    runner.set_wait_budget(WAIT_SECONDS)
    # epoch 2 inserts at other slots (other pages): a write shows
    case = _case(8, seed=1, device=device)
    before = kv.clone()
    _chain(runner, w, fw, None, kv, case, False, branch, stream)
    torch.accelerator.synchronize()  # every launch returned
    waits = given_up_waits(runner.health.cpu(), spec.REGIONS)
    gave_up = torch.tensor([int(bool(waits))], device=device)
    dist.all_reduce(gave_up)
    with tn._tp_checks("the stalled K1"):
        assert gave_up.item() > 0, "no rank's wait gave up"
        if rank == STALLED:
            assert any(_k1_regions().search(x) for x in waits), waits
            # FAILED before the epilogue's insert: not one K / V byte moved
            assert torch.equal(_bits(kv), _bits(before))
        else:
            assert not torch.equal(_bits(kv), _bits(before)), "K1 inserted nothing"
        if rank == 0:
            assert runner.flags.any(), "the output rank's flags missed the failure"


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_stalled_k1_fails_without_a_hang(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _stall_worker)


@needs_gfx950
@multi_gpu_test(num_gpus=8)
@pytest.mark.parametrize("layout", K1_VARIANTS)
def test_layer_chain_k1_variant(monkeypatch: pytest.MonkeyPatch, layout: str) -> None:
    """``test_layer_chain`` on a K1 tuning variant (TUNE L1): the r256 layout
    (boundary CTAs run 6 GEMV tasks past their publish; 2 to 4 contributors a
    tile in ``qpart4``), at every width and kind, pad rows, the NaN pad rows
    and a CUDA graph of two widths."""
    _use_k1_variant(monkeypatch, layout)
    multi_process_parallel(monkeypatch, 8, 1, _chain_worker)


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_stalled_k1_fails_without_a_hang_r256(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stall test on the r256 layout, whose stalled CTA 0 also runs GEMV
    tasks and an epilogue unit: every launch returns, a ``qpart4`` / ``ardy``
    / ``fss`` / ``nss`` wait gives up, the stalled rank inserts nothing."""
    _use_k1_variant(monkeypatch, "r256")
    multi_process_parallel(monkeypatch, 8, 1, _stall_worker)


def _mono_sets(runner, sets, case, branch, stream, key) -> None:
    """The epoch bump, then one K1 a (weights, cache) set, each on its own K1
    slot as consecutive layers' are (no launch reads another's pairs)."""
    runner.step_begin()
    for i, (fw, kv) in enumerate(sets):
        _k1(runner, fw, kv, case, branch, stream, spec.k1_slot(2 + i), key)


def _stock_sets(sets, table, case, branch, stream) -> None:
    """The stock front of every set."""
    for fw, kv in sets:
        _stock_front(fw, table, kv, case, branch, stream)


@ray.remote(num_gpus=1, max_calls=1)
def _speed_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = tn._init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.distributed.parallel_state import graph_capture
    from vllm.models.muse_spark.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    rope = ref.rope_table(MAX_POS, device=device)
    # 8 weight / cache sets (8 x 37.7 MB of qkv weights a rank, past the
    # MALL), a sliding layer's each
    sets = [
        (_with_table(_front_weights(rank, device, seed=i), rope), _kv(rank, device))
        for i in range(8)
    ]
    for S, gate in ((1, 0.70), (8, 0.6)):
        case = _case(S, device=device)
        branch, stream = _step_inputs(device, S)
        key = spec.K1Build(tokens=S, rope=True, **_k1_variant())
        runner.set_wait_budget(STARTUP_WAIT_SECONDS)
        mono = (runner, sets, case, branch, stream, key)
        stock = (sets, rope, case, branch, stream)
        _mono_sets(*mono)  # warm-ups: compile, load, JIT
        _stock_sets(*stock)
        torch.accelerator.synchronize()
        tn._assert_healthy(runner)
        runner.set_wait_budget(WAIT_SECONDS)
        # the epoch bump timed alone and taken off: without it a replay's
        # polls would pass on the previous replay's pairs (same tags)
        with graph_capture(device=device) as ctx:
            g_mono = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_mono, stream=ctx.stream):
                _mono_sets(*mono)
            g_begin = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_begin, stream=ctx.stream):
                runner.step_begin()
            g_stock = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_stock, stream=ctx.stream):
                _stock_sets(*stock)
        t = torch.tensor(
            [tn._replay_ms(g_mono), tn._replay_ms(g_begin), tn._replay_ms(g_stock)],
            device=device,
        )
        dist.all_reduce(t, op=dist.ReduceOp.MAX)  # the slowest rank's
        tn._assert_healthy(runner)
        mono_ms, begin_ms, stock_ms = t.tolist()
        k1_us = (mono_ms - begin_ms) * 1e3 / len(sets)
        stock_us = stock_ms * 1e3 / len(sets)
        print(
            f"S={S}: K1 {k1_us:.1f} us, stock front {stock_us:.1f} us "
            f"({k1_us / stock_us:.2f}x; the serving-span gate is {gate}x)",
            flush=True,
        )


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_front_speed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Diagnostic (DESIGN_BC 9.3, E12): printed, not gated."""
    multi_process_parallel(monkeypatch, 8, 1, _speed_worker)
