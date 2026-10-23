# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU numerics: AITER unified_attention on the window-trimmed decode tables.

Runs the spec-decode verify call of ``AiterFlashAttentionImpl`` (causal,
sliding window, q rows per request) on the full block table and on the
table ``_build_swa_decode_tables`` trims to the window, for uniform and
mixed q, and asserts both agree (and match an fp32 reference).

Two dispatches: "shipped" is AITER as installed (on gfx950 a sliding-window
call runs one KV segment that already starts at the window, so full and
trimmed do the same work). "forced_split" (gfx950 only) is the split-KV
grid the trimmed table is meant for: it lifts both AITER gates, the
``_gfx950_split_grid`` sliding-window reject and the ``kv_split["SW"]``
``MAX_SEGMENTS = 1`` cap, and asserts every call really ran > 1 segment.

Muse Spark TP8 shape: 16 q / 2 kv heads per rank, head 64, bf16, window
2048, block 16.
"""

import contextvars
import importlib
import zlib
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv

if not current_platform.is_rocm():
    pytest.skip("ROCm AITER only", allow_module_level=True)

UA = pytest.importorskip("aiter.ops.triton.attention.unified_attention")
UAU = importlib.import_module("aiter.ops.triton.utils.unified_attention_utils")

from vllm.v1.attention.backends.rocm_aiter_fa import (  # noqa: E402
    AiterFlashAttentionMetadataBuilder,
)

WINDOW, BLOCK, HQ, HKV, D = 2048, 16, 16, 2, 64
CONTEXTS = [100, 2047, 2048, 2055, 4096, 32768, 200000]
ATOL = 1e-3


def _make(seq_lens, query_lens, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    nblk = [cdiv(L, BLOCK) for L in seq_lens]
    width = max(nblk) + 2
    num_blocks = sum(nblk) + 8
    k = torch.randn(num_blocks, BLOCK, HKV, D, device=dev, generator=g)
    v = torch.randn(num_blocks, BLOCK, HKV, D, device=dev, generator=g)
    k, v = k.bfloat16(), v.bfloat16()
    perm = torch.randperm(num_blocks - 8, device=dev, generator=g).to(torch.int32)
    # Unused columns point at real blocks too: a stray read would show up.
    bt = torch.randint(
        0, num_blocks, (len(seq_lens), width), device=dev, generator=g
    ).to(torch.int32)
    off = 0
    for r, n in enumerate(nblk):
        bt[r, :n] = perm[off : off + n]
        off += n
    T = sum(query_lens)
    q = torch.randn(T, HQ, D, device=dev, generator=g).bfloat16()
    cu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=dev)
    cu[1:] = torch.tensor(query_lens, device=dev).cumsum(0)
    sk = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    return q, k, v, bt, cu, sk


def _trim(bt, sk, cu, max_q):
    n = sk.shape[0]
    width = cdiv(WINDOW + 8 - 1, BLOCK) + 1  # builder width for K = 7
    builder = SimpleNamespace(
        aot_sliding_window=(WINDOW - 1, 0),
        block_size=BLOCK,
        _swa_block_table=torch.zeros(n, width, dtype=torch.int32, device="cuda"),
        _swa_seq_lens=torch.zeros(n, dtype=torch.int32, device="cuda"),
        _swa_block_offsets=torch.arange(width, dtype=torch.int64, device="cuda"),
    )
    table, lens, max_len = AiterFlashAttentionMetadataBuilder._build_swa_decode_tables(
        builder, bt, sk, n, int(sk.max()), query_start_loc=cu, max_query_len=max_q
    )
    assert table is not None
    return table, lens, max_len


def _ua(q, k, v, cu, max_q, sk, max_k, bt):
    out = torch.empty_like(q)
    UA.unified_attention(
        q=q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=max_q,
        seqused_k=sk, max_seqlen_k=max_k, softmax_scale=D**-0.5, causal=True,
        window_size=(WINDOW - 1, 0), block_table=bt, softcap=0,
        q_descale=None, k_descale=None, v_descale=None,
    )  # fmt: skip
    return out


def _reference(q, k, v, bt, seq_lens, query_lens):
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    t = 0
    for r, (L, ql) in enumerate(zip(seq_lens, query_lens)):
        lo = max(0, L - ql - (WINDOW - 1))
        pos = torch.arange(lo, L, device=q.device)
        slots = bt[r, pos // BLOCK].long() * BLOCK + pos % BLOCK
        kk = k.view(-1, HKV, D)[slots].float().repeat_interleave(HQ // HKV, 1)
        vv = v.view(-1, HKV, D)[slots].float().repeat_interleave(HQ // HKV, 1)
        qq = q[t : t + ql].float()
        s = torch.einsum("qhd,khd->hqk", qq, kk) * D**-0.5
        qpos = (L - ql + torch.arange(ql, device=q.device))[:, None]
        ok = (pos[None, :] <= qpos) & (qpos - pos[None, :] < WINDOW)
        s = s.masked_fill(~ok[None], float("-inf"))
        out[t : t + ql] = torch.einsum("hqk,khd->qhd", s.softmax(-1), vv)
        t += ql
    return out


_LIFT_SW_CAP = contextvars.ContextVar("lift_sw_cap", default=False)


def force_sliding_window_split(monkeypatch) -> list[int]:
    """Make AITER's gfx950 gluon path split sliding-window calls over the KV.

    Two gates keep it unsplit: ``_gfx950_split_grid`` returns False for a
    sliding window unless every query is a single token, and the
    ``kv_split`` config entry every SW > 0 call lands on ("SW") sets
    ``MAX_SEGMENTS = 1``. Lift both, and record the NUM_SEGMENTS of every
    3d launch so the caller can check the split really happened.
    """
    orig_grid = UA._gfx950_split_grid
    orig_cfg = UA.get_unified_attention_config
    orig_segments = UAU.compute_segment_params
    segments: list[int] = []

    def split_grid(params):
        return True if params.sliding_window > 0 else orig_grid(params)

    def get_config(op, params, *args, **kwargs):
        lift = op == "kv_split" and params.sliding_window > 0
        token = _LIFT_SW_CAP.set(lift)
        try:
            return orig_cfg(op, params, *args, **kwargs)
        finally:
            _LIFT_SW_CAP.reset(token)

    def segment_params(config, params):
        if _LIFT_SW_CAP.get():
            config.pop("MAX_SEGMENTS", None)
        return orig_segments(config, params)

    orig_3d = UA._unified_attention_3d_gfx950

    def record_3d(params, segm_output, segm_max, segm_expsum, num_segments, *a):
        segments.append(num_segments)
        return orig_3d(params, segm_output, segm_max, segm_expsum, num_segments, *a)

    # unified_attention imports both by name: patch the copies it calls.
    monkeypatch.setattr(UA, "_gfx950_split_grid", split_grid)
    monkeypatch.setattr(UA, "get_unified_attention_config", get_config)
    monkeypatch.setattr(UAU, "compute_segment_params", segment_params)
    monkeypatch.setattr(UA, "_unified_attention_3d_gfx950", record_3d)
    return segments


@pytest.fixture(params=["shipped", "forced_split"])
def dispatch(request, monkeypatch):
    segments = None
    if request.param == "forced_split":
        if getattr(UA, "DEVICE_ARCH", None) != "gfx950":
            pytest.skip("forced split targets the gfx950 gluon path")
        segments = force_sliding_window_split(monkeypatch)
    return SimpleNamespace(name=request.param, segments=segments)


def _cases():
    for q in (1, 2, 4, 8):
        yield f"uniform_q{q}", CONTEXTS, [q] * len(CONTEXTS)
    yield "mixed_a", CONTEXTS, [8, 1, 3, 8, 2, 8, 5]
    yield "mixed_b", CONTEXTS + CONTEXTS, [1, 2, 4, 8, 7, 6, 5, 8, 8, 1, 2, 3, 4, 8]


@pytest.mark.parametrize(
    "name, seq_lens, query_lens", list(_cases()), ids=[c[0] for c in _cases()]
)
def test_trimmed_matches_full(dispatch, name, seq_lens, query_lens):
    seed = zlib.crc32(name.encode()) & 0xFFFF
    q, k, v, bt, cu, sk = _make(seq_lens, query_lens, seed=seed)
    max_q = max(query_lens)
    full = _ua(q, k, v, cu, max_q, sk, int(sk.max()), bt)
    table, lens, max_len = _trim(bt, sk, cu, max_q)
    assert max_len < int(sk.max())
    trimmed = _ua(q, k, v, cu, max_q, lens, max_len, table)
    torch.cuda.synchronize()
    if dispatch.segments is not None:
        # one 3d launch per call, each split over the KV
        assert len(dispatch.segments) == 2, dispatch.segments
        assert min(dispatch.segments) > 1, dispatch.segments
    ref = _reference(q, k, v, bt, seq_lens, query_lens)
    diff = (full.float() - trimmed.float()).abs().max().item()
    assert diff <= ATOL, f"{name}/{dispatch.name}: full vs trimmed {diff:.3e}"
    for tag, o in (("full", full), ("trimmed", trimmed)):
        err = (o.float() - ref).abs().max().item()
        assert err <= 1e-2, f"{name}/{dispatch.name}: {tag} vs fp32 ref {err:.3e}"
