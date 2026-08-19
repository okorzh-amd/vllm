# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np

from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec


class CPUOffloadingMetrics:
    STORES_SKIPPED = "vllm:kv_offload_stores_skipped"
    CPU_CACHE_USAGE_PERC = "vllm:kv_offload_cpu_cache_usage_perc"
    CPU_ALLOCATION_SIZE = "vllm:kv_offload_cpu_allocation_size"
    CPU_CACHE_WRITE_USAGE_PERC = "vllm:kv_offload_cpu_cache_write_usage_perc"
    CPU_CACHE_READ_USAGE_PERC = "vllm:kv_offload_cpu_cache_read_usage_perc"


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """
    Spec for loading/storing a KV block to CPU memory.

    With per-group pools the block ids are group-local slot ids and no longer
    address the buffer by themselves, since each group's row has its own width
    and the group's extents are scattered through the arena. ``block_offsets``
    then carries the byte offset of each slot. The ids are still needed: they
    are dense and consecutive, which is what makes writer rotation across
    replicated ranks uniform.
    """

    def __init__(
        self, block_ids: list[int], block_offsets: list[int] | None = None
    ) -> None:
        super().__init__(block_ids)
        self.block_offsets: np.ndarray | None = (
            None if block_offsets is None else np.array(block_offsets, dtype=np.int64)
        )
