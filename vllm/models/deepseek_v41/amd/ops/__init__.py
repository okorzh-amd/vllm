# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx942 (MI300X, MI325X) decode helpers for DeepSeek-V4.1.

``topk`` holds the sparse indexer's top-512 hooks, built from
``topk512_gfx942.cu``, and ``cand_logits`` the indexer logits of the DSpark
candidate positions. ``sparse_indexer`` is the indexer layer that runs them on
decode steps. ``topk_pack`` and ``swa_decode_meta`` build the decode
attention metadata, ``fused_markov`` is DSpark's Markov sampling in one
launch, and ``draft_lm_head`` keeps the DSpark draft's 5-row logits GEMM off
wvSplitK. All of them run only when ``enabled()`` is true.
"""

import functools
import os

from vllm.platforms import current_platform

# --monokernel (KernelConfig.monokernel) of this process's model. The helpers
# run in forward passes, where vLLM's current config is not set, so the model
# records the flag when it is created (record_monokernel).
_monokernel = False


def record_monokernel(vllm_config) -> None:
    """Record whether this process's model runs with --monokernel. The mono
    decode layer calls this when the model is created, before any helper
    runs."""
    global _monokernel
    _monokernel = bool(vllm_config.kernel_config.monokernel)
    enabled.cache_clear()
    c1ab_swameta_enabled.cache_clear()


@functools.cache
def enabled() -> bool:
    """True with --monokernel on a gfx942 GPU. Every helper keeps the result
    of vLLM's op it replaces, so each one returns False for a call it cannot
    take and vLLM then runs its own op. gfx950 always keeps vLLM's ops here."""
    if not _monokernel or not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx942

    return on_gfx942()


@functools.cache
def c1ab_swameta_enabled() -> bool:
    """VLLM_C1AB_SWAMETA=1: also run swa_decode_meta's one-launch SWA decode
    metadata build on gfx950 with --monokernel, for the target's SWA groups
    and the DSpark draft's non-causal groups alike. Only swa_decode_meta reads
    this; topk, topk_pack, cand_logits, fused_markov and draft_lm_head keep
    the gfx942-only ``enabled()``. Unset, gfx950 keeps vLLM's 8-launch build.
    The kernel is plain Triton (no arch intrinsics) and writes the same
    tensors, ragged + indptr included, so nothing it compiles depends on the
    flag."""
    if os.environ.get("VLLM_C1AB_SWAMETA", "0") != "1":
        return False
    if not _monokernel or not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()
