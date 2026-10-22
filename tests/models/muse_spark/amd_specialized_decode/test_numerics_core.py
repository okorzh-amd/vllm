# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2(split) (``K2Build(core="split")``, the in-kernel split-KV attention
core, DESIGN_BC 7, WP-C3) at TP8 against the decode vLLM runs, real shapes,
random weights in the post-load formats.

The front is the stock one (``fused_norm_residual_norm_fwd``, the qkv GEMM as
served, the fused QK-norm / RoPE / KV insert into the layer's cache) or K1
(``mono/k1.py``, Phase B1; skipped while the tree has none). Every step runs
the same front twice on two copies of the cache: once into the stock core
(``paged_attention_v1``, a sliding layer on ``_build_swa_decode_tables``'
window-trimmed table) and K2(gated) (B6: bit for bit K2(tensor) on vLLM's
gated core), once into K2(split). The metadata is MRV2's persistent buffers
(``reference.FakeMrv2``: one full and one sliding KV group, each its own
block table and slot mapping, a sliding table nulled below the window).

[CPU] The test's own pieces: the cases make the edges bite (the page and
window edges, a window start aligned and unaligned to a page, a context past
32k, pad rows, every width class); the replay schedule crosses a page and the
2048 window on MRV2's tables (a sliding entry nulled mid-run); the fp64 gate.

[TP8] ``test_layer_chain_split``: on the live rows the split core's gated rows
(``xo``) agree with ``rms_norm_gated`` of the stock attention (cos) and are
never further from the fp64 gated attention than stock's (E9);
``stream_out`` agrees with the K2(gated) step and with vLLM's stock span from
the stock gated core (cos 0.99999); the tail is Phase A's (``stream_out`` and
``branch_out`` bit for bit K2(tensor) on the split core's own gated rows) and
``branch_out`` agrees with vLLM's experts on its own ffn_in rows and picks
(cos 0.9999, Phase A's reference, DESIGN_BC 9.3; the whole chain's branch is
printed: the stock span itself moves to cos ~0.99 between the two cores,
wp_logs_bc/DEBUG.md (c)); the pad rows
are exact zeros, and NaN in their qkv rows changes nothing; the ranks agree
bit for bit; the cache bytes after the split step equal the gated step's (the
split core never writes the cache); no wait gave up.

[TP8] ``test_graph_replay_follows_metadata``: one CUDA graph of ``step_begin;
front; K2(split)`` for a full and a sliding layer chained (and one of the
same two layers' gated chain, the stock core's max length the table's: a
sliding layer's trimmed table caps it at the window on the device, as a
captured decode does), captured on the persistent buffers; between
replays the host advances ``seq_lens``, positions, slots and the block tables
as MRV2 does (crossing pages and the sliding window); every replay equals an
eager run of the same step on the same contents, 300 replays, health zero.

[TP8] ``test_layer_chain_split_oproj_early`` / ``test_graph_replay_oproj_early``
(TUNE L2, ``K2Build.oproj_issue="early"``: the combine CTAs' o_proj tasks run
on the CTAs below C0): the chain test's checks on the "early" build, plus
``stream_out``, ``branch_out`` and ``xo`` bit for bit a "late" launch on the
same inputs (every task's arithmetic is unchanged, only its CTA moves); the
replay test with the "early" build in the split graph.
``test_split_oproj_early_every_width``: "early" vs "late" bit for bit at
every width 1..8 and both kinds (``EARLY_CASES``).

[TP8] ``test_layer_chain_split_preset`` (TUNE L3, ``mono_split_preset``: the
split core's N and MIN_PER; "coarse" is the runtime default): the chain
test's checks on each ``CHAIN_PRESETS`` build. ``test_split_presets_every_width``:
every width and kind, each ``WIDTH_PRESETS`` build's xo within the core
targets of the design's split build, its "early" build bit for bit its
"late" one. ``test_graph_replay_split_preset``: the replay test on
``REPLAY_PRESETS`` builds.
"""

import importlib.util
import math
import os
from types import SimpleNamespace

import pytest
import ray
import torch

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

from tests.models.muse_spark.amd_specialized_decode import (  # noqa: E402
    reference as ref,
)
from tests.models.muse_spark.amd_specialized_decode import (  # noqa: E402
    test_numerics as tn,
)
from tests.models.muse_spark.amd_specialized_decode import (  # noqa: E402
    test_numerics_front as tnf,
)
from tests.models.muse_spark.amd_specialized_decode import (  # noqa: E402
    test_probe_split as tps,
)
from tests.utils import multi_gpu_test, multi_process_parallel  # noqa: E402
from vllm.models.muse_spark.amd.specialized_decode.mono import spec  # noqa: E402
from vllm.models.muse_spark.amd.specialized_decode.mono.abi import (  # noqa: E402
    STARTUP_WAIT_SECONDS,
)

bf = torch.bfloat16
f32, f64 = torch.float32, torch.float64
H, CORE, QKV = spec.HIDDEN, spec.CORE, spec.QKV_ROWS
FULL, SLIDING = "full", "sliding"
KINDS = (FULL, SLIDING)
# the kernel layers of the fake model: layer 1 full (NoPE), layer 2 sliding
LAYER = {FULL: 1, SLIDING: 2}
NAME = {FULL: "layers.1.self_attn.attn", SLIDING: "layers.2.self_attn.attn"}
MAX_POS = 1 << 16  # the RoPE table's rows: every position below
MAX_BLOCKS = 2112  # a block table's row: 33,792 keys (> the 32,773 case)
COS_CORE = 0.99999
REPLAYS = 300
# (requests' context lengths, width): rows past the requests are MRV2's pad
# rows (seq_lens 0, slot -1). 2065 / 2063: a window starting mid-page; 2064
# and 4096 + 16: at a page start; 2047 .. 2050 around the window; 32773 past
# 32k; 1, 2, 16, 17 the first page's edges
CASES = (
    ((2065,), 1),
    ((32773,), 1),
    ((17,), 1),
    ((2049, 17, 2064), 3),
    ((1, 16, 257, 2047, 2050, 4103), 8),
    ((32773, 2048, 2063, 17, 3000, 2, 4112, 900), 8),
)
# TUNE L2's early-vs-late launches: every width (S = 6, 7 hold their second
# extra slot on part of the CTAs below C0, S = 8 on all; S = 5 puts extras on
# seam CTAs), pad rows at S = 2, 5, 7, a context past 32k at S = 1 and 6
EARLY_CASES = tuple(
    (lens[:n], S)
    for lens, n, S in (
        ((32773,), 1, 1),
        ((2065, 17), 1, 2),
        ((2049, 17, 2064), 3, 3),
        ((1, 16, 257, 4103), 4, 4),
        ((2047, 2050, 900, 3000), 4, 5),
        ((32773, 2048, 2063, 17, 3000, 2), 6, 6),
        ((4112, 900, 1, 16, 257, 2047), 6, 7),
        ((17, 2065, 2049, 1, 16, 257, 4103, 2050), 8, 8),
    )
)
# the replay graph's two requests: request 0 crosses the window (and a page
# every 16 steps), request 1 is past it from the start
REPLAY_START = (2030, 4090)


def _on_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


def _has_k1() -> bool:
    return (
        importlib.util.find_spec(
            "vllm.models.muse_spark.amd.specialized_decode.mono.k1"
        )
        is not None
    )


needs_gfx950 = pytest.mark.skipif(
    not _on_gfx950(), reason="the Muse mono kernels run on gfx950 (MI355X)"
)
needs_k1 = pytest.mark.skipif(
    not _has_k1(), reason="mono.k1 (WP-B4, Phase B1) is not in this tree"
)


# ---------------------------------------------------------------- helpers


def gate64(attn, og) -> torch.Tensor:
    """The output norm and gate in fp64 (no rounding): per (token, head) row
    of 64, ``a / sqrt(mean a^2 + eps) sigmoid(g)``. [S, HEADS, 64]."""
    a = attn.double().reshape(attn.shape[0], spec.HEADS, spec.HEAD_DIM)
    g = og.double().reshape(a.shape)
    r = torch.rsqrt(a.square().mean(-1, keepdim=True) + spec.RMS_EPS)
    return a * r * torch.sigmoid(g)


def _fake_mrv2(device) -> ref.FakeMrv2:
    """MRV2's persistent buffers of the two kernel layers: each its own KV
    group, the sliding one's table nulled below the window."""
    return ref.FakeMrv2(
        {FULL: [NAME[FULL]], SLIDING: [NAME[SLIDING]]},
        {SLIDING},
        max_blocks=MAX_BLOCKS,
        device=device,
    )


def _schedule(steps: int = REPLAYS) -> list[tuple[int, ...]]:
    """The replay graph's per-step context lengths."""
    return [tuple(L + i for L in REPLAY_START) for i in range(steps)]


def _pages(lens, nblocks: int, seed: int) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    return ref.assign_pages(lens, nblocks, generator=g)


def _front_weights(rank: int, device) -> dict:
    """One rank's K1 weights (the previous layer's FFN boundary, this layer's
    pre-qkv norm, its qkv_proj shard [2304, 8192], the RoPE table):
    replicated tensors from seed 17, the shard from 300 + rank."""
    from vllm.models.muse_spark.common.layers import (
        MuseSpark1xNormPreservingResidualConnection,
    )

    gc = torch.Generator(device=device).manual_seed(17)
    gr = torch.Generator(device=device).manual_seed(300 + rank)

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
        "cos_sin": ref.rope_table(MAX_POS, device=device),
    }


def _boundary_inputs(device, S: int, seed: int):
    """(branch bf16, stream fp32) [S, H], replicated: the previous layer's
    deferred boundary."""
    g = torch.Generator(device=device).manual_seed(seed)
    branch = torch.randn(S, H, generator=g, device=device).to(bf)
    stream = torch.randn(S, H, generator=g, device=device)
    return branch, stream


def _stock_front(fw, kind, branch, stream, positions, slots, kv):
    """The stock front of a kernel layer: the deferred boundary, the qkv GEMM
    (as served), the fused QK-norm / RoPE / KV insert in place. ->
    (stream_out, qkvo)."""
    from vllm import _custom_ops as ops
    from vllm.models.muse_spark.common.ops.fused_rmsnorm import (
        fused_norm_residual_norm_fwd,
    )

    attn_in, stream_out = fused_norm_residual_norm_fwd(
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
    qkvo = tn._gemm(attn_in, fw["w_qkv"])
    rope = kind == SLIDING
    ops.fused_muse_spark_qknorm_rope_kv_insert(
        qkvo,
        positions,
        spec.HEADS,
        spec.KV_HEADS,
        spec.HEAD_DIM,
        spec.RMS_EPS,
        fw["cos_sin"] if rope else None,
        spec.HEAD_DIM if rope else 0,
        slots,
        kv,
        spec.KV_BLOCK,
    )
    return stream_out, qkvo


def _k1_front(runner, fw, kind, branch, stream, positions, slots, pad, kv):
    """K1's front (Phase B1) with the runtime's arguments. -> (stream_out,
    qkvo)."""
    from vllm.models.muse_spark.amd.specialized_decode.runtime import layers

    S = branch.shape[0]
    rope = kind == SLIDING
    stream_out = torch.empty_like(stream)
    qkvo = torch.empty(S, QKV, dtype=bf, device=stream.device)
    layers.k1_launch(
        spec.K1Build(tokens=S, rope=rope),
        {
            **{n: fw[n] for n in ("w_postffn", "ffn_alpha", "ffn_beta", "w_preqkv")},
            "w_qkv": fw["w_qkv"],
            "cos_sin": fw["cos_sin"] if rope else 0,
            "max_pos": MAX_POS if rope else 0,
            **layers.kv_args(kv),
            "branch": branch,
            "stream": stream,
            "pad": pad,
            "positions": positions,
            "slots": slots,
            "stream_out": stream_out,
            "qkvo": qkvo,
            "slot": spec.k1_slot(LAYER[kind]),
            **runner.launch_args(),
            "rank": runner.rank,
        },
    )
    return stream_out, qkvo


def _front(front, runner, fw, kind, branch, stream, step, kv):
    name = NAME[kind]
    if front == "k1":
        return _k1_front(
            runner,
            fw,
            kind,
            branch,
            stream,
            step.positions,
            step.slot_mapping[name],
            step.pad,
            kv,
        )
    return _stock_front(
        fw, kind, branch, stream, step.positions, step.slot_mapping[name], kv
    )


def _stock_attention(qkvo, kv, md, kind, max_len=None):
    """The stock core's attention output bf16 [S, CORE] (``paged_attention_v1``
    as decode runs it, ``test_probe_split.stock_attention``); ``max_len``
    defaults to the step's longest context (a graph passes the table's)."""
    S = qkvo.shape[0]
    lens = md.seq_lens.tolist() if max_len is None else [max_len]
    ns = SimpleNamespace(
        case=SimpleNamespace(S=S, window=kind == SLIDING, lens=tuple(lens)),
        kv=kv,
        bt=md.block_table,
        seq_lens=md.seq_lens,
        qkvo=qkvo,
    )
    return tps.stock_attention(ns).reshape(S, CORE)


def _stock_gated(attn, qkvo):
    """VLLM's gated core (o_proj's input) from the attention output."""
    from vllm.third_party.flash_linear_attention.ops.kda import (
        rms_norm_gated,
    )

    S = attn.shape[0]
    return rms_norm_gated(
        attn.view(S, spec.HEADS, 64),
        qkvo[:, spec.ROW_G :].reshape(S, spec.HEADS, 64),
        None,
        None,
        "sigmoid",
        eps=spec.RMS_EPS,
    ).reshape(S, CORE)


def _split_args(qkvo, kv, md) -> dict:
    """K2(split)'s core arguments as ``KernelLayer`` passes them."""
    from vllm.models.muse_spark.amd.specialized_decode.runtime import layers

    bt = md.block_table
    return {
        "qkvo": qkvo,
        **layers.kv_args(kv),
        "block_table": bt,
        "bt_rows": layers.table_rows(bt),
        "bt_stride": bt.stride(0),
        "seq_lens": md.seq_lens,
    }


def _k2(runner, w, key, core_args, stream, pad, kind):
    """One K2 launch of build ``key`` (no epoch bump): (stream_out,
    branch_out)."""
    from vllm.models.muse_spark.amd.specialized_decode.mono import layer

    S = key.tokens
    stream_out = torch.empty_like(stream)
    branch_out = torch.empty(S, H, dtype=bf, device=stream.device)
    layer.k2_launch(
        key,
        {
            **{n: w[n] for n in tn.KERNEL_WEIGHTS},
            **core_args,
            "stream": stream,
            "pad": pad,
            "stream_out": stream_out,
            "branch_out": branch_out,
            "slot": spec.k2_slot(LAYER[kind]),
            **runner.launch_args(),
            **runner.peer_args(),
        },
    )
    return stream_out, branch_out


def _split_key(
    S: int, kind: str, oproj: str = "late", preset: str = "base"
) -> spec.K2Build:
    """The split build of width S and ``kind``: ``oproj`` its
    ``oproj_issue``, ``preset`` its TUNE L3 split preset (``split_fields``)."""
    sw = kind == SLIDING
    return spec.K2Build(
        tokens=S,
        core="split",
        window=sw,
        oproj_issue=oproj,
        **spec.split_fields(preset, S, sw),
    )


def _split_step(
    front,
    runner,
    w,
    fw,
    kind,
    branch,
    stream,
    step,
    kv,
    *,
    begin=True,
    oproj="late",
    preset="base",
):
    """``step_begin; front; K2(split)`` of one layer: (stream_out, branch_out,
    qkvo). ``oproj``: the build's ``oproj_issue``; ``preset``: its split
    preset (TUNE L3)."""
    S = branch.shape[0]
    if begin:
        runner.step_begin()
    s1, qkvo = _front(front, runner, fw, kind, branch, stream, step, kv)
    md = step.attn_metadata[NAME[kind]]
    key = _split_key(S, kind, oproj, preset)
    out = _k2(runner, w, key, _split_args(qkvo, kv, md), s1, step.pad, kind)
    return (*out, qkvo)


def _gated_step(
    front, runner, w, fw, kind, branch, stream, step, kv, max_len=None, *, begin=True
):
    """``step_begin; front; stock core; K2(gated)`` of one layer:
    (stream_out, branch_out, qkvo, attn, front's stream_out)."""
    S = branch.shape[0]
    if begin:
        runner.step_begin()
    s1, qkvo = _front(front, runner, fw, kind, branch, stream, step, kv)
    attn = _stock_attention(qkvo, kv, step.attn_metadata[NAME[kind]], kind, max_len)
    key = spec.K2Build(tokens=S, core="gated")
    out = _k2(runner, w, key, {"attn": attn, "qkvo": qkvo}, s1, step.pad, kind)
    return (*out, qkvo, attn, s1)


def _xo(runner, S: int) -> torch.Tensor:
    """The split core's gated rows (the combine's ``xo``) of the last launch."""
    off, nbytes = spec.scratch_layout()[spec.XO]
    return runner.scratch[off : off + nbytes].view(bf).view(-1, CORE)[:S].clone()


def _bytes(kv: torch.Tensor) -> torch.Tensor:
    return kv.view(torch.int16)


# ---------------------------------------------------------------- [CPU]


def test_cases_make_the_edges_bite():
    """Both kinds at every width class (1, 3, 8), the window's edges, a window
    start aligned and unaligned to a page, the page edges, a context past
    32k, MRV2 pad rows, distinct physical pages that fit the table."""
    lens = [L for c, _ in CASES for L in c]
    assert {w for _, w in CASES} == {1, 3, 8}
    assert any(len(c) < w for c, w in CASES)  # pad rows
    for L in (1, 2, 16, 17, 2047, 2048, 2049, 2050):
        assert L in lens, L
    starts = [L - spec.WINDOW for L in lens if L > spec.WINDOW]
    assert any(s % spec.KV_BLOCK == 0 for s in starts)  # aligned
    assert any(s % spec.KV_BLOCK for s in starts)  # mid-page
    assert max(lens) > 32768
    assert -(-max(lens) // spec.KV_BLOCK) <= MAX_BLOCKS
    for c, w in CASES:
        assert len(c) <= w <= spec.MAX_ROWS
        need = sum(-(-L // spec.KV_BLOCK) for L in c)
        pages = _pages(c, need + 8, 0)
        flat = [b for row in pages for b in row]
        assert len(set(flat)) == len(flat) and 0 not in flat


def test_early_cases_cover_every_placement():
    """TUNE L2's early-vs-late cases: every width once (so every extra-slot
    count, a slot held by part of the CTAs below C0 and one held by all, and
    extras on seam CTAs), pad rows, a context past 32k, distinct pages."""
    assert [w for _, w in EARLY_CASES] == list(range(1, spec.MAX_ROWS + 1))
    assert any(len(c) < w for c, w in EARLY_CASES)
    assert max(L for c, _ in EARLY_CASES for L in c) > 32768
    held = {
        (w, j): sum(
            j + 2 < len(spec.oproj_tasks(w, b, "early")) for b in range(spec.BLOCKS)
        )
        for _, w in EARLY_CASES
        for j in range(spec.oproj_slots(w))
    }
    assert any(0 < n < spec.C0(w) for (w, j), n in held.items() if j == 1)
    assert any(n == spec.C0(w) for (w, j), n in held.items() if j == 1)
    for c, w in EARLY_CASES:
        assert 1 <= len(c) <= w <= spec.MAX_ROWS
        assert -(-max(c) // spec.KV_BLOCK) <= MAX_BLOCKS
        need = sum(-(-L // spec.KV_BLOCK) for L in c)
        pages = _pages(c, need + 8, 0)
        flat = [b for row in pages for b in row]
        assert len(set(flat)) == len(flat) and 0 not in flat


def test_presets_run_tuned_builds():
    """TUNE L3: every TP8 preset is valid and tunes a build at some width the
    chain (CASES) or the every-width test (EARLY_CASES) runs; the default
    preset runs in the chain under both o_proj placements."""
    for p, _ in CHAIN_PRESETS:
        spec.check_split_preset(p)
        tuned = [
            S for _, S in CASES for w in (False, True) if spec.split_fields(p, S, w)
        ]
        assert tuned, p
    assert {(spec.DEFAULT_SPLIT_PRESET, o) for o in spec.OPROJ_ISSUES} <= set(
        CHAIN_PRESETS
    )
    assert spec.DEFAULT_SPLIT_PRESET in WIDTH_PRESETS
    for p in WIDTH_PRESETS:
        assert any(
            spec.split_fields(p, S, w) for _, S in EARLY_CASES for w in (False, True)
        )


def test_replay_schedule_crosses_a_page_and_the_window():
    """The replay graph's host steps on MRV2's persistent buffers: the same
    buffers every step (a graph's baked addresses), seq_lens, positions and
    slots advancing, request 0 crossing pages and the 2048 window, the
    sliding table nulling an entry below the window mid-run."""
    sched = _schedule()
    need = [-(-L // spec.KV_BLOCK) for L in sched[-1]]
    pages = _pages(sched[-1], sum(need) + 8, 1)
    mrv2 = _fake_mrv2("cpu")
    ptrs = None
    nulled = []
    crossed = 0
    prev_slot = None
    for lens in sched:
        step = mrv2.step(list(lens), {FULL: pages, SLIDING: pages}, 2)
        md = step.attn_metadata[NAME[SLIDING]]
        now = tuple(
            t.data_ptr()
            for t in (
                md.block_table,
                md.seq_lens,
                step.positions,
                step.slot_mapping[NAME[FULL]],
            )
        )
        assert ptrs is None or now == ptrs
        ptrs = now
        assert md.seq_lens.tolist() == list(lens)
        assert step.positions.tolist() == [L - 1 for L in lens]
        n0 = -(-lens[0] // spec.KV_BLOCK)  # request 0's pages this step
        nulled.append(int((md.block_table[0, :n0] == 0).sum()))
        slot = int(step.slot_mapping[NAME[FULL]][0])
        if (
            prev_slot is not None
            and slot // spec.KV_BLOCK != prev_slot // spec.KV_BLOCK
        ):
            crossed += 1
        prev_slot = slot
    assert sched[0][0] < spec.WINDOW < sched[-1][0]
    assert nulled[0] == 0 and nulled[-1] > 0  # the window moved past a page
    assert crossed >= 10
    full = mrv2.block_tables[FULL][0, : need[0]]
    assert (full != 0).all()  # a full group's table is never nulled


def test_gate64_is_the_rounded_gate():
    """The fp64 gate rounds to ``reference.gated_norm`` (FLA's gfx950 order)
    within one bf16 ulp, pad rows aside."""
    g = torch.Generator().manual_seed(5)
    attn = torch.randn(8, CORE, generator=g).to(bf)
    attn[2] *= 3e-3  # mean square ~ eps: the eps bites
    og = (torch.randn(8, CORE, generator=g) * 4).to(bf)
    got = gate64(attn, og).to(bf).reshape(8, CORE)
    want = ref.gated_norm(attn, og)
    assert ref.bf16_ulps(got, want).max().item() <= 1
    assert math.isclose(ref.cos(got, want), 1.0, abs_tol=1e-6)


def test_launches_carry_the_abis(monkeypatch):
    """The [TP8] tests' launches pass exactly their builds' ABI names (K1:
    ``spec.K1_ARGS``; K2: ``spec.K2_ARGS_BY_CORE``), the split's table
    arguments as ``KernelLayer`` binds them: the layer's group's persistent
    table (its buffer's rows, its row stride) and ``seq_lens``."""
    from vllm.models.muse_spark.amd.specialized_decode.mono import layer
    from vllm.models.muse_spark.amd.specialized_decode.runtime import layers

    got: dict = {}
    monkeypatch.setattr(layers, "k1_launch", lambda k, a: got.update(k1=(k, a)))
    monkeypatch.setattr(layer, "k2_launch", lambda k, a: got.update({k.core: (k, a)}))
    state = dict.fromkeys(("scratch", "epoch", "health", "alarm"), 0)
    runner = SimpleNamespace(
        rank=3, launch_args=lambda: state, peer_args=lambda: {"peers": 0, "rank": 3}
    )
    lens, S = (17, 2065), 3
    mrv2 = _fake_mrv2("cpu")
    pages = _pages(lens, 256, 0)
    step = mrv2.step(list(lens), {FULL: pages, SLIDING: pages}, S)
    kv = torch.zeros(4, spec.KV_HEADS, spec.KV_BLOCK, spec.KV_C, dtype=bf)
    branch, stream = torch.zeros(S, H, dtype=bf), torch.zeros(S, H)
    w = {n: 0 for n in tn.KERNEL_WEIGHTS}
    for kind in KINDS:
        fw = {n: 0 for n in ("w_postffn", "ffn_alpha", "ffn_beta", "w_preqkv")}
        fw.update(w_qkv=0, cos_sin=torch.zeros(MAX_POS, 64))
        got.clear()
        _, qkvo = _front("k1", runner, fw, kind, branch, stream, step, kv)
        key1, a1 = got["k1"]
        assert key1 == spec.K1Build(tokens=S, rope=kind == SLIDING)
        assert set(a1) == set(spec.K1_ARGS) and a1["slots"] is not None
        assert (a1["max_pos"] == MAX_POS) == (kind == SLIDING), a1["max_pos"]
        assert a1["slot"] == spec.k1_slot(LAYER[kind])
        *_, qkvo = _split_step(
            "k1", runner, w, fw, kind, branch, stream, step, kv, begin=False
        )
        key2, a2 = got["split"]
        assert key2 == spec.K2Build(tokens=S, core="split", window=kind == SLIDING)
        assert set(a2) == set(spec.K2_ARGS_BY_CORE["split"])
        table = mrv2.block_tables[kind]
        assert a2["block_table"].data_ptr() == table.data_ptr()
        assert a2["bt_rows"] == table.shape[0] and a2["bt_stride"] == MAX_BLOCKS
        assert a2["seq_lens"].data_ptr() == mrv2.seq_lens.data_ptr()
        assert a2["slot"] == spec.k2_slot(LAYER[kind]) and a2["qkvo"] is qkvo
        gated = spec.K2Build(tokens=S, core="gated")
        _k2(runner, w, gated, {"attn": qkvo, "qkvo": qkvo}, stream, step.pad, kind)
        assert set(got["gated"][1]) == set(spec.K2_ARGS_BY_CORE["gated"])


# ---------------------------------------------------------------- [TP8]


def _check_chain(runner, w, kind, case, gated, split, s_front, step, kvs):
    """The 9.3 checks of one step (``gated``/``split``: the two runs'
    outputs; ``kvs``: their caches)."""
    lens, S = case
    dev = s_front.device
    sg, bg, qkvo_g, attn, _ = gated
    ss, bs, qkvo_s, xo = split
    pad = step.pad.cpu()
    live = ~pad
    lv = live.to(dev)
    md = step.attn_metadata[NAME[kind]]
    where = f"{kind} {lens} S={S}"
    tn._assert_healthy(runner)
    # the branch vLLM's experts give on the split launch's own ffn_in rows and
    # picks (read back before any later launch; collective)
    own = tnf._own_branch(runner, w, S)
    # the stock span (collective) from vLLM's gated core on the same rows, and
    # from the split core's own gated rows (diagnostic: stock's sensitivity)
    stock_core = _stock_gated(attn.clone(), qkvo_g)
    span: dict = {}
    tn._stock_span(w, stock_core, s_front, step.pad, span, None)
    span_xo: dict = {}
    tn._stock_span(w, xo, s_front, step.pad, span_xo, None)
    # Phase A's K2(tensor) on the split core's own gated rows: the same tail
    runner.step_begin()
    tensor = spec.K2Build(tokens=S)
    st, bt = _k2(runner, w, tensor, {"core": xo}, s_front, step.pad, kind)
    torch.accelerator.synchronize()
    tn._assert_healthy(runner)
    with tn._tp_checks(f"{where}: the split core"):
        assert torch.equal(qkvo_s[lv], qkvo_g[lv]), "the front is not repeatable"
        assert torch.equal(_bytes(kvs[0]), _bytes(kvs[1])), "the KV cache differs"
        # the gated rows: vs vLLM's gated core on the stock attention, and
        # never further from the fp64 gated attention than stock (E9)
        q = qkvo_g[:, :CORE].reshape(S, spec.HEADS, 64)
        a64 = ref.gqa_decode(
            q, kvs[1], md.block_table, md.seq_lens, window=kind == SLIDING, pad=pad
        )
        g64 = gate64(a64, qkvo_g[:, spec.ROW_G :].cpu())[live.cpu()]
        got = xo.cpu().reshape(S, spec.HEADS, 64)[live]
        stock = stock_core.cpu().reshape(S, spec.HEADS, 64)[live]
        assert ref.cos(got, stock) > COS_CORE, (where, ref.cos(got, stock))
        e_got = (got.double() - g64).abs().max().item()
        e_stock = (stock.double() - g64).abs().max().item()
        ulp = 2.0 ** (math.floor(math.log2(g64.abs().max().item())) - 7)
        assert e_got <= e_stock + 2 * ulp, (where, e_got, e_stock)
        u = tps.row_ulps(got, stock)
        print(
            f"{where}: xo vs stock gate: max {u.max().item():.2f} row ulps, "
            f"<= 1 on {(u <= 1).float().mean().item():.5f}; fp64 err "
            f"{e_got:.3g} (stock {e_stock:.3g})",
            flush=True,
        )
        assert (xo[pad.to(xo.device)].float() == 0).all()
        # the layer's outputs. stream_out: vs the K2(gated) step and the stock
        # span. The tail after the core is Phase A's: bit for bit K2(tensor)
        # on the split core's own gated rows. branch_out: vs vLLM's experts on
        # its own ffn_in rows and picks (Phase A's reference, DESIGN_BC 9.3),
        # not vs the gated / stock chain: two cores that differ by an ulp on
        # ~40 % of the rows' elements (split vs paged_attention_v1, both held
        # above) take the stock span itself to branch cos 0.973 .. 0.995
        # (``stock span`` printed; wp_logs_bc/DEBUG.md (c))
        assert ref.cos(ss[lv], sg[lv]) > tn.COS_STAGE, where
        assert ref.cos(ss[lv], span["stream"][lv]) > tn.COS_STAGE, where
        assert torch.equal(st, ss) and torch.equal(bt, bs), (where, "K2(tensor)")
        assert ref.cos(bs[lv], own[lv]) > tn.COS_CHAIN, where
        assert (ss[~lv] == 0).all() and (bs[~lv].float() == 0).all(), where
        sb = span["branch"][lv]
        print(
            f"{where}: branch cos vs own experts {ref.cos(bs[lv], own[lv]):.7f}; "
            f"whole chain vs K2(gated) {ref.cos(bs[lv], bg[lv]):.5f}, vs stock "
            f"span {ref.cos(bs[lv], span['branch'][lv]):.5f}; stock span on "
            f"xo vs on stock's core {ref.cos(span_xo['branch'][lv], sb):.5f}",
            flush=True,
        )
    tn._all_ranks_equal(ss, f"stream_out ({where})")
    tn._all_ranks_equal(bs, f"branch_out ({where})")


def _chain(
    front,
    monkeypatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
    oproj="late",
    preset="base",
):
    device = tn._init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.muse_spark.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    # the first launch of each build compiles it: the startup budget
    runner.set_wait_budget(STARTUP_WAIT_SECONDS)
    w = tn._weights(rank, device)
    fw = _front_weights(rank, device)
    mrv2 = _fake_mrv2(device)
    for ci, (lens, S) in enumerate(CASES):
        need = sum(-(-L // spec.KV_BLOCK) for L in lens)
        nblocks = need + 64
        pages = _pages(lens, nblocks, 100 + ci)
        step = mrv2.step(list(lens), {FULL: pages, SLIDING: pages}, S)
        branch, stream = _boundary_inputs(device, S, 50 + ci)
        for kind in KINDS:
            g = torch.Generator().manual_seed(1000 * ci + 7 * rank + LAYER[kind])
            kv = ref.kv_cache(nblocks, device=device, generator=g)
            kvs = (kv, kv.clone())
            gated = _gated_step(
                front, runner, w, fw, kind, branch, stream, step, kvs[0]
            )
            ss, bs, qkvo_s = _split_step(
                front,
                runner,
                w,
                fw,
                kind,
                branch,
                stream,
                step,
                kvs[1],
                oproj=oproj,
                preset=preset,
            )
            torch.accelerator.synchronize()
            split = (ss, bs, qkvo_s, _xo(runner, S))
            _check_chain(runner, w, kind, (lens, S), gated, split, gated[4], step, kvs)
            md = step.attn_metadata[NAME[kind]]
            if oproj != "late":
                # TUNE L2: the "late" build on the same inputs, bit for bit
                runner.step_begin()
                late = _k2(
                    runner,
                    w,
                    _split_key(S, kind, preset=preset),
                    _split_args(qkvo_s, kvs[1], md),
                    gated[4],
                    step.pad,
                    kind,
                )
                torch.accelerator.synchronize()
                tn._assert_healthy(runner)
                with tn._tp_checks(f"{kind} {lens} S={S}: {oproj} vs late"):
                    assert torch.equal(late[0], ss), "stream_out"
                    assert torch.equal(late[1], bs), "branch_out"
                    assert torch.equal(_xo(runner, S), split[3]), "xo"
            if len(lens) == S:
                continue
            # NaN in the pad rows' qkv rows: nothing else moves
            dirty = qkvo_s.clone()
            dirty[step.pad] = float("nan")
            key = _split_key(S, kind, oproj, preset)
            runner.step_begin()
            again = _k2(
                runner, w, key, _split_args(dirty, kvs[1], md), gated[4], step.pad, kind
            )
            torch.accelerator.synchronize()
            tn._assert_healthy(runner)
            with tn._tp_checks(f"{kind} {lens}: NaN pad rows"):
                assert torch.equal(again[0], ss) and torch.equal(again[1], bs)


@ray.remote(num_gpus=1, max_calls=1)
def _chain_stock_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _chain("stock", monkeypatch, tp_size, pp_size, rank, distributed_init_port)


@ray.remote(num_gpus=1, max_calls=1)
def _chain_k1_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _chain("k1", monkeypatch, tp_size, pp_size, rank, distributed_init_port)


@needs_gfx950
@multi_gpu_test(num_gpus=8)
@pytest.mark.parametrize(
    "front", ["stock", pytest.param("k1", marks=needs_k1)], ids=["stock", "k1"]
)
def test_layer_chain_split(front: str, monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _chain_k1_worker if front == "k1" else _chain_stock_worker
    multi_process_parallel(monkeypatch, 8, 1, worker)


@ray.remote(num_gpus=1, max_calls=1)
def _chain_stock_early_worker(
    monkeypatch, tp_size, pp_size, rank, distributed_init_port
):
    _chain("stock", monkeypatch, tp_size, pp_size, rank, distributed_init_port, "early")


@ray.remote(num_gpus=1, max_calls=1)
def _chain_k1_early_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _chain("k1", monkeypatch, tp_size, pp_size, rank, distributed_init_port, "early")


@needs_gfx950
@multi_gpu_test(num_gpus=8)
@pytest.mark.parametrize(
    "front", ["stock", pytest.param("k1", marks=needs_k1)], ids=["stock", "k1"]
)
def test_layer_chain_split_oproj_early(
    front: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _chain_k1_early_worker if front == "k1" else _chain_stock_early_worker
    multi_process_parallel(monkeypatch, 8, 1, worker)


def _early_vs_late(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    """TUNE L2: at every width and kind, the "early" K2(split) build's
    ``stream_out``, ``branch_out`` and ``xo`` bit for bit the "late" build's
    on the same front rows and cache (each o_proj task's arithmetic is
    unchanged, only its CTA moves); no wait gave up."""
    device = tn._init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.muse_spark.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    runner.set_wait_budget(STARTUP_WAIT_SECONDS)
    w = tn._weights(rank, device)
    fw = _front_weights(rank, device)
    mrv2 = _fake_mrv2(device)
    for ci, (lens, S) in enumerate(EARLY_CASES):
        need = sum(-(-L // spec.KV_BLOCK) for L in lens)
        nblocks = need + 64
        pages = _pages(lens, nblocks, 200 + ci)
        step = mrv2.step(list(lens), {FULL: pages, SLIDING: pages}, S)
        branch, stream = _boundary_inputs(device, S, 70 + ci)
        for kind in KINDS:
            g = torch.Generator().manual_seed(2000 * ci + 7 * rank + LAYER[kind])
            kv = ref.kv_cache(nblocks, device=device, generator=g)
            md = step.attn_metadata[NAME[kind]]
            runner.step_begin()
            s1, qkvo = _front("stock", runner, fw, kind, branch, stream, step, kv)
            out = {}
            for oproj in spec.OPROJ_ISSUES:
                runner.step_begin()
                key = _split_key(S, kind, oproj)
                args = _split_args(qkvo, kv, md)
                so, bo = _k2(runner, w, key, args, s1, step.pad, kind)
                torch.accelerator.synchronize()
                tn._assert_healthy(runner)
                out[oproj] = (so, bo, _xo(runner, S))
            where = f"{kind} {lens} S={S}: early vs late"
            (sl, bl, xl), (se, be, xe) = out["late"], out["early"]
            with tn._tp_checks(where):
                assert torch.equal(se, sl), "stream_out"
                assert torch.equal(be, bl), "branch_out"
                assert torch.equal(xe, xl), "xo"
            tn._all_ranks_equal(se, f"stream_out ({where})")


@ray.remote(num_gpus=1, max_calls=1)
def _early_vs_late_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _early_vs_late(monkeypatch, tp_size, pp_size, rank, distributed_init_port)


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_split_oproj_early_every_width(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _early_vs_late_worker)


# TUNE L3: the split presets the TP8 chain runs (every atom's build at the
# CASES widths 1, 3, 8; "narrow" = short+wide at 1, 3 and coarse at 8), the
# default ("coarse") under both o_proj placements, short+wide "early"
CHAIN_PRESETS = (
    ("narrow", "late"),
    ("coarse", "late"),
    ("coarse", "early"),
    ("short+wide", "early"),
)


# the preset and o_proj placement ``_chain_preset_worker`` runs: read in each
# Ray worker (a local Ray cluster's workers inherit the driver's environment
# as of ``ray.init``), set per parameter by ``test_layer_chain_split_preset``
PRESET_ENV, PRESET_OPROJ_ENV = "MUSE_SPLIT_PRESET", "MUSE_SPLIT_OPROJ"


@ray.remote(num_gpus=1, max_calls=1)
def _chain_preset_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _chain(
        "stock",
        monkeypatch,
        tp_size,
        pp_size,
        rank,
        distributed_init_port,
        # no default: a worker that missed the driver's environment must not
        # silently run the design's build (TUNE L3 review)
        os.environ[PRESET_OPROJ_ENV],
        os.environ[PRESET_ENV],
    )


@needs_gfx950
@multi_gpu_test(num_gpus=8)
@pytest.mark.parametrize(
    "preset,oproj", CHAIN_PRESETS, ids=[f"{p}-{o}" for p, o in CHAIN_PRESETS]
)
def test_layer_chain_split_preset(
    preset: str, oproj: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TUNE L3: ``test_layer_chain_split``'s checks (stock front) on a split
    preset's build: xo vs vLLM's gated core and fp64 (E9), stream_out vs
    K2(gated) and the stock span, the tail bit for bit K2(tensor) on its own
    xo, branch vs vLLM's experts, NaN pad rows inert, ranks equal; an
    "early" build also bit for bit the preset's "late" build."""
    monkeypatch.setenv(PRESET_ENV, preset)
    monkeypatch.setenv(PRESET_OPROJ_ENV, oproj)
    multi_process_parallel(monkeypatch, 8, 1, _chain_preset_worker)


# TUNE L3 at every width (EARLY_CASES): each preset's "late" build vs the
# design's (the same keys split another way: fp32 rounding only) and its
# "early" build bit for bit its "late" one
WIDTH_PRESETS = ("narrow", "coarse")


def _presets_every_width(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    """At every width and kind, each ``WIDTH_PRESETS`` build: xo within the
    core targets of the design's split build (cos, and never further from the
    fp64 gated attention than it by more than 2 ulps), stream_out cos, pad
    rows zero; its "early" build bit for bit its "late" one; health zero."""
    device = tn._init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.muse_spark.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    runner.set_wait_budget(STARTUP_WAIT_SECONDS)
    w = tn._weights(rank, device)
    fw = _front_weights(rank, device)
    mrv2 = _fake_mrv2(device)
    for ci, (lens, S) in enumerate(EARLY_CASES):
        need = sum(-(-L // spec.KV_BLOCK) for L in lens)
        nblocks = need + 64
        pages = _pages(lens, nblocks, 300 + ci)
        step = mrv2.step(list(lens), {FULL: pages, SLIDING: pages}, S)
        branch, stream = _boundary_inputs(device, S, 90 + ci)
        pad = step.pad.cpu()
        live = ~pad
        lv = live.to(device)
        for kind in KINDS:
            g = torch.Generator().manual_seed(3000 * ci + 7 * rank + LAYER[kind])
            kv = ref.kv_cache(nblocks, device=device, generator=g)
            md = step.attn_metadata[NAME[kind]]
            runner.step_begin()
            s1, qkvo = _front("stock", runner, fw, kind, branch, stream, step, kv)
            args = _split_args(qkvo, kv, md)
            runner.step_begin()
            sb, _ = _k2(runner, w, _split_key(S, kind), args, s1, step.pad, kind)
            torch.accelerator.synchronize()
            tn._assert_healthy(runner)
            xb = _xo(runner, S)
            q = qkvo[:, :CORE].reshape(S, spec.HEADS, 64)
            a64 = ref.gqa_decode(
                q, kv, md.block_table, md.seq_lens, window=kind == SLIDING, pad=pad
            )
            g64 = gate64(a64, qkvo[:, spec.ROW_G :].cpu())[live.cpu()]
            base = xb.cpu().reshape(S, spec.HEADS, 64)[live]
            e_base = (base.double() - g64).abs().max().item() if live.any() else 0.0
            for preset in WIDTH_PRESETS:
                if not spec.split_fields(preset, S, kind == SLIDING):
                    continue  # the design's build at this width and kind
                out = {}
                for oproj in spec.OPROJ_ISSUES:
                    runner.step_begin()
                    key = _split_key(S, kind, oproj, preset)
                    so, bo = _k2(runner, w, key, args, s1, step.pad, kind)
                    torch.accelerator.synchronize()
                    tn._assert_healthy(runner)
                    out[oproj] = (so, bo, _xo(runner, S))
                where = f"{kind} {lens} S={S} {preset}"
                (sl, bl, xl), (se, be, xe) = out["late"], out["early"]
                with tn._tp_checks(where):
                    assert torch.equal(se, sl), "early vs late: stream_out"
                    assert torch.equal(be, bl), "early vs late: branch_out"
                    assert torch.equal(xe, xl), "early vs late: xo"
                    assert (xl[pad.to(device)].float() == 0).all()
                    if live.any():
                        got = xl.cpu().reshape(S, spec.HEADS, 64)[live]
                        assert ref.cos(got, base) > COS_CORE, ref.cos(got, base)
                        e_got = (got.double() - g64).abs().max().item()
                        top = g64.abs().max().item()
                        ulp = 2.0 ** (math.floor(math.log2(top)) - 7)
                        assert e_got <= e_base + 2 * ulp, (e_got, e_base)
                        assert ref.cos(sl[lv], sb[lv]) > tn.COS_STAGE
                        u = tps.row_ulps(got, base)
                        print(
                            f"{where}: xo vs base max {u.max().item():.2f} row "
                            f"ulps; fp64 err {e_got:.3g} (base {e_base:.3g})",
                            flush=True,
                        )
                tn._all_ranks_equal(sl, f"stream_out ({where})")


@ray.remote(num_gpus=1, max_calls=1)
def _presets_every_width_worker(
    monkeypatch, tp_size, pp_size, rank, distributed_init_port
):
    _presets_every_width(monkeypatch, tp_size, pp_size, rank, distributed_init_port)


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_split_presets_every_width(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _presets_every_width_worker)


def _replay(
    front,
    monkeypatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
    oproj="late",
    preset="base",
):
    device = tn._init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.muse_spark.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    runner.set_wait_budget(STARTUP_WAIT_SECONDS)
    w = tn._weights(rank, device)
    fw = _front_weights(rank, device)
    sched = _schedule()
    S = len(REPLAY_START)
    need = sum(-(-L // spec.KV_BLOCK) for L in sched[-1])
    nblocks = need + 64
    pages = _pages(sched[-1], nblocks, 7)
    mrv2 = _fake_mrv2(device)
    step = mrv2.step(list(sched[0]), {FULL: pages, SLIDING: pages}, S)
    g = torch.Generator().manual_seed(77 + rank)
    kv = {k: ref.kv_cache(nblocks, device=device, generator=g) for k in KINDS}
    kv_g = {k: t.clone() for k, t in kv.items()}
    branch, stream = _boundary_inputs(device, S, 9)
    cap = MAX_BLOCKS * spec.KV_BLOCK

    def chain():
        """Two kernel layers: the full one, then the sliding one on its
        deferred boundary (one step_begin)."""
        s1, b1, _ = _split_step(
            front,
            runner,
            w,
            fw,
            FULL,
            branch,
            stream,
            step,
            kv[FULL],
            oproj=oproj,
            preset=preset,
        )
        s2, b2, _ = _split_step(
            front,
            runner,
            w,
            fw,
            SLIDING,
            b1,
            s1,
            step,
            kv[SLIDING],
            begin=False,
            oproj=oproj,
            preset=preset,
        )
        return [s1, b1, s2, b2]

    def gated():
        """The same two layers on the stock core and K2(gated). The stock
        core's max length is the table's (``cap``), a host constant: a sliding
        layer's ``_build_swa_decode_tables`` trims the table and the lengths
        on the device and caps the max length at the window."""
        s1, b1 = _gated_step(
            front, runner, w, fw, FULL, branch, stream, step, kv_g[FULL], cap
        )[:2]
        s2, b2 = _gated_step(
            front,
            runner,
            w,
            fw,
            SLIDING,
            b1,
            s1,
            step,
            kv_g[SLIDING],
            cap,
            begin=False,
        )[:2]
        return [s1, b1, s2, b2]

    chain()  # warm-ups: compile, load, JIT
    gated()
    torch.accelerator.synchronize()
    graphs, outs = {}, {}
    for name, fn in (("split", chain), ("gated", gated)):
        graphs[name] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[name]):
            outs[name] = fn()
    torch.accelerator.synchronize()
    for i, lens in enumerate(sched):
        mrv2.step(list(lens), {FULL: pages, SLIDING: pages}, S)  # in place
        for name, fn in (("split", chain), ("gated", gated)):
            graphs[name].replay()
            torch.accelerator.synchronize()
            got = [t.clone() for t in outs[name]]
            want = fn()  # eager: the same step's K / V rewritten in place
            torch.accelerator.synchronize()
            with tn._tp_checks(f"replay {i} ({name}, lens {lens})"):
                for a, b in zip(got, want):
                    assert torch.equal(a, b), (name, i, lens)
                    assert torch.isfinite(a.float()).all(), (name, i)
    tn._assert_healthy(runner)


@ray.remote(num_gpus=1, max_calls=1)
def _replay_stock_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _replay("stock", monkeypatch, tp_size, pp_size, rank, distributed_init_port)


@ray.remote(num_gpus=1, max_calls=1)
def _replay_k1_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _replay("k1", monkeypatch, tp_size, pp_size, rank, distributed_init_port)


@needs_gfx950
@multi_gpu_test(num_gpus=8)
@pytest.mark.parametrize(
    "front", ["stock", pytest.param("k1", marks=needs_k1)], ids=["stock", "k1"]
)
def test_graph_replay_follows_metadata(
    front: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _replay_k1_worker if front == "k1" else _replay_stock_worker
    multi_process_parallel(monkeypatch, 8, 1, worker)


@ray.remote(num_gpus=1, max_calls=1)
def _replay_early_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _replay(
        "stock", monkeypatch, tp_size, pp_size, rank, distributed_init_port, "early"
    )


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_graph_replay_oproj_early(monkeypatch: pytest.MonkeyPatch) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _replay_early_worker)


# TUNE L3 review: the replay graph (S = 2, request 0 crossing the window and a
# page every 16 steps) on tuned split builds: full N 16, sliding N 17
# ("early"); MIN_PER 16, sliding N 9 (the default "coarse")
REPLAY_PRESETS = (("short+wide", "early"), ("coarse", "late"))


@ray.remote(num_gpus=1, max_calls=1)
def _replay_preset_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    _replay(
        "stock",
        monkeypatch,
        tp_size,
        pp_size,
        rank,
        distributed_init_port,
        os.environ[PRESET_OPROJ_ENV],
        os.environ[PRESET_ENV],
    )


@needs_gfx950
@multi_gpu_test(num_gpus=8)
@pytest.mark.parametrize(
    "preset,oproj", REPLAY_PRESETS, ids=[f"{p}-{o}" for p, o in REPLAY_PRESETS]
)
def test_graph_replay_split_preset(
    preset: str, oproj: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``test_graph_replay_follows_metadata`` (stock front) on a split
    preset's builds: every replay equals an eager run, health zero."""
    assert all(
        spec.split_fields(preset, len(REPLAY_START), w) for w in (False, True)
    ), preset
    monkeypatch.setenv(PRESET_ENV, preset)
    monkeypatch.setenv(PRESET_OPROJ_ENV, oproj)
    multi_process_parallel(monkeypatch, 8, 1, _replay_preset_worker)
