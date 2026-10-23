# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the ROCm AITER FA window-trimmed decode tables.

``AiterFlashAttentionMetadataBuilder._build_swa_decode_tables`` drops the
blocks a sliding-window decode can never read and rebases the seq lens, for
plain decode (q == 1) and spec-decode verify (q > 1, per-request q). The
kernels (AITER unified_attention, gluon gfx950 and triton) derive each query
row's position as ``seqused_k - query_len + i`` and mask with key/query
differences only, so the trimmed (table, seq_len) must expose exactly the
same physical KV slots to every query row as the full (table, seq_len).
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.rocm_aiter_fa import (
    AiterFlashAttentionImpl,
    AiterFlashAttentionMetadataBuilder,
)

MAX_Q = 8


def _visible_slots(
    table_row: torch.Tensor, seq_len: int, q: int, window: int, block_size: int
) -> list[torch.Tensor]:
    """Physical KV slots each query row attends to, as the kernels see them:
    row i sits at ``seq_len - q + i``, keys k < seq_len with
    ``0 <= pos - k < window``, slot = table[k // bs] * bs + k % bs."""
    context = seq_len - q
    rows = []
    for i in range(q):
        pos = context + i
        lo, hi = max(0, pos - window + 1), min(pos, seq_len - 1)
        k = torch.arange(lo, hi + 1, dtype=torch.int64)
        if k.numel():
            # Never relies on the kernels' clamp of the column index.
            assert int((k // block_size).max()) < table_row.shape[0]
        rows.append(table_row[k // block_size].to(torch.int64) * block_size
                    + k % block_size)  # fmt: skip
    return rows


def _builder_ns(n: int, window: int, block_size: int, max_q: int = MAX_Q):
    width = cdiv(window + max_q - 1, block_size) + 1
    return SimpleNamespace(
        aot_sliding_window=(window - 1, 0),
        block_size=block_size,
        _swa_block_table=torch.full((n, width), -7, dtype=torch.int32),
        _swa_seq_lens=torch.full((n,), -7, dtype=torch.int32),
        _swa_block_offsets=torch.arange(width, dtype=torch.int64),
    )


def _full_tables(seq_lens: list[int], block_size: int, pad_cols: int = 3):
    """Unique physical block ids per (request, logical block)."""
    width = cdiv(max(seq_lens), block_size) + pad_cols
    perm = torch.randperm(len(seq_lens) * width, dtype=torch.int64)
    return perm.view(len(seq_lens), width).to(torch.int32)


def _edge_lens(q: int, window: int, block_size: int, rng) -> list[int]:
    cands = {
        q, q + 1, window - 1, window, window + 1,
        window + q - 2, window + q - 1, window + q, window + q + 1,
        window + q - 1 + block_size - 1, window + q - 1 + block_size,
        2 * window, 2 * window + 1, 4096, 4097,
    }  # fmt: skip
    for m in range(1, 6):
        b = (window // block_size + m) * block_size
        cands.update({b - 1, b, b + 1, b + q - 1, b + q})
    cands.update(int(x) for x in rng.integers(1, 6 * window + 64, size=24))
    return sorted(c for c in cands if c >= q)


def _check(seq_lens, query_lens, window, block_size, max_q_cap=MAX_Q):
    n = len(seq_lens)
    full = _full_tables(seq_lens, block_size)
    lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    qsl = torch.zeros(n + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(query_lens, dtype=torch.int32).cumsum(0)
    builder = _builder_ns(n, window, block_size, max_q_cap)
    max_seq_len = max(seq_lens)
    table, lens, max_len = AiterFlashAttentionMetadataBuilder._build_swa_decode_tables(
        builder, full, lens_t, n, max_seq_len,
        query_start_loc=qsl, max_query_len=max(query_lens),
    )  # fmt: skip
    assert table is not None and lens is not None
    # Persistent storage, not a fresh tensor.
    assert table.data_ptr() == builder._swa_block_table.data_ptr()
    assert lens.data_ptr() == builder._swa_seq_lens.data_ptr()
    assert max_len <= max_seq_len
    for r in range(n):
        L, q, Lt = seq_lens[r], query_lens[r], int(lens[r])
        assert q <= Lt <= L and (L - Lt) % block_size == 0
        assert Lt <= max_len
        assert cdiv(Lt, block_size) <= table.shape[1]
        want = _visible_slots(full[r], L, q, window, block_size)
        got = _visible_slots(table[r], Lt, q, window, block_size)
        for i in range(q):
            assert torch.equal(want[i], got[i]), (r, L, q, Lt, i)
    return table, lens, max_len


@pytest.mark.parametrize(
    "window, block_size", [(2048, 16), (128, 16), (100, 16), (7, 4), (3, 16)]
)
@pytest.mark.parametrize("q", list(range(1, MAX_Q + 1)))
def test_trimmed_table_sees_same_keys_uniform_q(window, block_size, q):
    import numpy as np

    rng = np.random.default_rng(1000 * window + 10 * block_size + q)
    torch.manual_seed(int(rng.integers(1 << 30)))
    seq_lens = _edge_lens(q, window, block_size, rng)
    if (window, block_size) == (2048, 16):
        seq_lens += [32768, 32768 + 5, 200000, 200000 + 3]
    _check(seq_lens, [q] * len(seq_lens), window, block_size)


@pytest.mark.parametrize("window, block_size", [(2048, 16), (100, 16), (7, 4)])
@pytest.mark.parametrize("seed", range(4))
def test_trimmed_table_sees_same_keys_mixed_q(window, block_size, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n = 48
    query_lens = [int(x) for x in rng.integers(1, MAX_Q + 1, size=n)]
    seq_lens = []
    for q in query_lens:
        kind = int(rng.integers(0, 4))
        if kind == 0:  # inside the first window
            seq_lens.append(int(rng.integers(q, window + q + 1)))
        elif kind == 1:  # around the first trimmed block
            seq_lens.append(window + q - 1 + int(rng.integers(-2, 2 * block_size)))
        else:
            seq_lens.append(int(rng.integers(q, 8 * window + 1)))
        seq_lens[-1] = max(seq_lens[-1], q)
    # The max q need not sit on every request: the max must not leak.
    query_lens[0] = MAX_Q
    _check(seq_lens, query_lens, window, block_size)


def _old_q1_tables(builder, block_table, seq_lens, num_decodes, max_seq_len):
    """The q == 1 trim as it was before spec-decode verify support."""
    window = builder.aot_sliding_window[0] + 1
    lens = seq_lens[:num_decodes]
    first_block = torch.clamp(lens - window, min=0) // builder.block_size
    width = cdiv(window, builder.block_size) + 1
    index = (
        first_block.to(torch.int64)[:, None] + torch.arange(width, dtype=torch.int64)
    ).clamp_(max=block_table.shape[1] - 1)
    table = torch.gather(block_table[:num_decodes], 1, index)
    return (
        table,
        lens - first_block * builder.block_size,
        min(max_seq_len, window + builder.block_size - 1),
    )


@pytest.mark.parametrize("max_q_cap", [1, MAX_Q])
def test_q1_trim_is_unchanged(max_q_cap):
    import numpy as np

    window, block_size = 2048, 16
    rng = np.random.default_rng(7)
    seq_lens = _edge_lens(1, window, block_size, rng) + [32768, 200000]
    n = len(seq_lens)
    full = _full_tables(seq_lens, block_size)
    lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    qsl = torch.arange(n + 1, dtype=torch.int32)
    for kwargs in ({}, {"query_start_loc": qsl, "max_query_len": 1}):
        builder = _builder_ns(n, window, block_size, max_q_cap)
        table, lens, max_len = (
            AiterFlashAttentionMetadataBuilder._build_swa_decode_tables(
                builder, full, lens_t, n, max(seq_lens), **kwargs
            )
        )
        old_table, old_lens, old_max = _old_q1_tables(
            builder, full, lens_t, n, max(seq_lens)
        )
        assert table is not None and lens is not None
        assert torch.equal(table[:, : old_table.shape[1]], old_table)
        assert torch.equal(lens, old_lens)
        assert max_len == old_max
        if max_q_cap == 1:
            assert table.shape == old_table.shape


def test_capacity_fallback_to_full_table():
    window, block_size = 2048, 16
    n = 4
    full = _full_tables([5000] * n, block_size)
    lens = torch.full((n,), 5000, dtype=torch.int32)
    qsl = torch.arange(0, (n + 1) * 2, 2, dtype=torch.int32)
    # A q == 1 sized table cannot hold a q == 2 verify window.
    builder = _builder_ns(n, window, block_size, max_q=1)
    out = AiterFlashAttentionMetadataBuilder._build_swa_decode_tables(
        builder, full, lens, n, 5000, query_start_loc=qsl, max_query_len=2
    )
    assert out == (None, None, 0)
    # Too many decodes for the persistent rows.
    builder = _builder_ns(n - 1, window, block_size)
    out = AiterFlashAttentionMetadataBuilder._build_swa_decode_tables(
        builder, full, lens, n, 5000, query_start_loc=qsl, max_query_len=2
    )
    assert out == (None, None, 0)


class _CountOps(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops: list[str] = []

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        if not func.is_view:  # views (slice, unsqueeze) launch nothing
            self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("q, budget", [(1, 7), (MAX_Q, 9)])
def test_trim_launch_budget(q, budget):
    """Every aten op here is a device launch on GPU; the trim runs once per
    sliding KV-cache group per step, so keep it to the minimum (q == 1 was 10
    ops before, verify 12)."""
    n, window, block_size = 32, 2048, 16
    full = _full_tables([30000] * n, block_size)
    lens = torch.full((n,), 30000, dtype=torch.int32)
    qsl = torch.arange(n + 1, dtype=torch.int32) * q
    builder = _builder_ns(n, window, block_size)
    with _CountOps() as counter:
        out = AiterFlashAttentionMetadataBuilder._build_swa_decode_tables(
            builder, full, lens, n, 30000, query_start_loc=qsl, max_query_len=q
        )
    assert out[0] is not None
    assert len(counter.ops) <= budget, counter.ops
    assert not any("_to_copy" in o or "copy_" in o for o in counter.ops)


# --------------------------------------------------------------------------
# Builder / forward wiring
# --------------------------------------------------------------------------

WINDOW, BLOCK, HEADS, KV_HEADS, HEAD = 2048, 16, 4, 2, 64


@pytest.fixture
def cpu_vllm_config():
    """A default VllmConfig that also builds on a GPU-less host."""
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(config):
        yield config


def _make_builder(
    monkeypatch,
    vllm_config,
    num_spec_tokens: int | None,
    verify_trim: bool = True,
    alibi: bool = False,
):
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    monkeypatch.setenv(
        "VLLM_ROCM_AITER_FA_SWA_VERIFY_TRIM", "1" if verify_trim else "0"
    )
    layer = Attention.__new__(Attention)
    torch.nn.Module.__init__(layer)
    layer.impl = AiterFlashAttentionImpl.__new__(AiterFlashAttentionImpl)
    layer.impl.sliding_window = (WINDOW - 1, 0)
    layer.impl.alibi_slopes = torch.ones(HEADS) if alibi else None
    layer.impl.num_kv_heads = KV_HEADS
    layer.impl.head_size = HEAD
    layer.kv_sharing_target_layer_name = None
    layer.kv_cache_update_bypass = False
    monkeypatch.setattr(
        vllm_config.compilation_config, "static_forward_context", {"attn": layer}
    )
    monkeypatch.setattr(
        vllm_config,
        "model_config",
        SimpleNamespace(
            dtype=torch.bfloat16,
            get_num_attention_heads=lambda _: HEADS,
            get_num_kv_heads=lambda _: KV_HEADS,
            get_head_size=lambda: HEAD,
        ),
    )
    spec_cfg = None
    if num_spec_tokens is not None:
        spec_cfg = SimpleNamespace(
            num_speculative_tokens=num_spec_tokens, parallel_drafting=False
        )
    monkeypatch.setattr(vllm_config, "speculative_config", spec_cfg)
    spec = SlidingWindowSpec(
        block_size=BLOCK,
        num_kv_heads=KV_HEADS,
        head_size=HEAD,
        dtype=torch.bfloat16,
        sliding_window=WINDOW,
    )
    return AiterFlashAttentionMetadataBuilder(
        spec, ["attn"], vllm_config, torch.device("cpu")
    )


@pytest.mark.parametrize("verify_trim", [False, True])
@pytest.mark.parametrize("num_spec_tokens", [None, 3, 7])
def test_builder_sizes_table_for_verify(
    monkeypatch, cpu_vllm_config, num_spec_tokens, verify_trim
):
    builder = _make_builder(
        monkeypatch, cpu_vllm_config, num_spec_tokens, verify_trim=verify_trim
    )
    max_q = 1 + (num_spec_tokens or 0)
    assert builder.reorder_batch_threshold == max_q
    assert builder._swa_verify_trim is verify_trim
    assert builder._swa_block_table is not None
    width = builder._swa_block_table.shape[1]
    if verify_trim:
        assert width == cdiv(WINDOW + max_q - 1, BLOCK) + 1
    else:  # default: exactly the q == 1 table of before
        assert width == cdiv(WINDOW, BLOCK) + 1
    if num_spec_tokens is None:  # unchanged width without spec decode
        assert width == cdiv(WINDOW, BLOCK) + 1


def test_builder_alibi_only_keeps_verify_trim_off(monkeypatch, cpu_vllm_config):
    # forward never routes alibi layers to the trimmed table: do not pay for it
    builder = _make_builder(monkeypatch, cpu_vllm_config, 7, alibi=True)
    assert builder._swa_verify_trim is False
    assert builder._swa_block_table.shape[1] == cdiv(WINDOW, BLOCK) + 1


def _common(seq_lens, query_lens):
    from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata

    return create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        BLOCK,
        torch.device("cpu"),
        arange_block_indices=True,
    )


VERIFY_BATCH = (
    [100, 2047, 2048, 2055, 2056, 4096, 32768, 200000],
    [8, 8, 8, 8, 8, 8, 8, 8],
)
MIXED_BATCH = (
    [100, 2047, 2055, 2056, 4096, 32768, 200000, 9],
    [1, 8, 3, 8, 5, 2, 8, 7],
)


class _CallCounter:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.fn(*args, **kwargs)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("verify_trim", [False, True])
def test_build_skips_verify_trim_unless_enabled(
    monkeypatch, cpu_vllm_config, verify_trim, causal
):
    """Default (and non-causal) verify steps issue no trim work at all; q == 1
    decode keeps trimming either way."""
    builder = _make_builder(monkeypatch, cpu_vllm_config, 7, verify_trim=verify_trim)
    counter = _CallCounter(builder._build_swa_decode_tables)
    builder._build_swa_decode_tables = counter
    common = _common(*MIXED_BATCH)
    common.causal = causal
    md = builder.build(common_prefix_len=0, common_attn_metadata=common)
    dmd = md.decode_metadata
    assert dmd is not None and dmd.max_query_len == max(MIXED_BATCH[1])
    trimmed = verify_trim and causal
    assert counter.calls == int(trimmed)
    assert (dmd.swa_block_table is not None) is trimmed
    assert (dmd.swa_seq_lens is not None) is trimmed

    seq_lens = MIXED_BATCH[0]
    md = builder.build(
        common_prefix_len=0, common_attn_metadata=_common(seq_lens, [1] * len(seq_lens))
    )
    assert counter.calls == int(trimmed) + 1
    assert md.decode_metadata.swa_block_table is not None


@pytest.mark.parametrize("batch", [VERIFY_BATCH, MIXED_BATCH])
def test_build_trims_verify_decodes(monkeypatch, cpu_vllm_config, batch):
    seq_lens, query_lens = batch
    builder = _make_builder(monkeypatch, cpu_vllm_config, 7)
    common = _common(seq_lens, query_lens)
    md = builder.build(common_prefix_len=0, common_attn_metadata=common)
    assert md.num_decodes == len(seq_lens) and md.num_prefills == 0
    dmd = md.decode_metadata
    assert dmd is not None and dmd.max_query_len == max(query_lens)
    assert dmd.swa_block_table is not None and dmd.swa_seq_lens is not None
    assert dmd.swa_block_table.data_ptr() == builder._swa_block_table.data_ptr()
    assert dmd.swa_max_seq_len == min(
        max(seq_lens), WINDOW + max(query_lens) - 1 + BLOCK - 1
    )
    for r, (L, q) in enumerate(zip(seq_lens, query_lens)):
        Lt = int(dmd.swa_seq_lens[r])
        assert Lt <= dmd.swa_max_seq_len
        want = _visible_slots(common.block_table_tensor[r], L, q, WINDOW, BLOCK)
        got = _visible_slots(dmd.swa_block_table[r], Lt, q, WINDOW, BLOCK)
        assert all(torch.equal(a, b) for a, b in zip(want, got)), (L, q)
        if L - q - (WINDOW - 1) >= BLOCK:  # long context really got trimmed
            assert Lt < L


class _Recorder:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["out"]


@pytest.mark.parametrize(
    "case",
    [
        "swa_verify",
        "swa_verify_trim_off",
        "full_verify",
        "swa_verify_alibi",
        "swa_q1_sinks",
    ],
)
def test_forward_routes_verify_to_trimmed_table(monkeypatch, cpu_vllm_config, case):
    builder = _make_builder(
        monkeypatch,
        cpu_vllm_config,
        7,
        verify_trim=case != "swa_verify_trim_off",
    )
    if case == "swa_q1_sinks":
        seq_lens, query_lens = [100, 4096, 200000], [1, 1, 1]
    else:
        seq_lens, query_lens = MIXED_BATCH
    common = _common(seq_lens, query_lens)
    md = builder.build(common_prefix_len=0, common_attn_metadata=common)
    dmd = md.decode_metadata
    assert dmd is not None
    assert (dmd.swa_block_table is not None) is (case != "swa_verify_trim_off")

    impl = AiterFlashAttentionImpl(
        num_heads=HEADS,
        head_size=HEAD,
        scale=HEAD**-0.5,
        num_kv_heads=KV_HEADS,
        alibi_slopes=[0.5] * HEADS if case == "swa_verify_alibi" else None,
        sliding_window=None if case == "full_verify" else WINDOW,
        kv_cache_dtype="auto",
        sinks=torch.zeros(HEADS) if case == "swa_q1_sinks" else None,
    )
    recorder = _Recorder()
    fake = types.ModuleType("aiter.ops.triton.unified_attention")
    fake.unified_attention = recorder
    monkeypatch.setitem(sys.modules, "aiter.ops.triton.unified_attention", fake)

    num_tokens = sum(query_lens)
    num_blocks = int(common.block_table_tensor.max()) + 1
    kv_cache = torch.zeros(num_blocks, KV_HEADS, BLOCK, 2 * HEAD)
    query = torch.zeros(num_tokens, HEADS, HEAD)
    output = torch.zeros(num_tokens, HEADS, HEAD)
    layer = SimpleNamespace(
        _k_scale=torch.ones(1), _v_scale=torch.ones(1), layer_name="attn"
    )
    impl.forward(layer, query, None, None, kv_cache, md, output)

    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["max_seqlen_q"] == max(query_lens)
    assert call["window_size"] == impl.sliding_window
    if case == "swa_verify":
        assert call["block_table"] is dmd.swa_block_table
        assert call["seqused_k"] is dmd.swa_seq_lens
        assert call["max_seqlen_k"] == dmd.swa_max_seq_len
        assert call["max_seqlen_k"] < max(seq_lens)
    else:
        n = len(seq_lens)
        assert torch.equal(call["block_table"], md.block_table[:n])
        assert torch.equal(call["seqused_k"], md.seq_lens[:n])
        assert call["max_seqlen_k"] == md.max_seq_len
