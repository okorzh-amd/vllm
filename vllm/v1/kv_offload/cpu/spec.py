# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
from typing_extensions import override

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.group_arena import GroupArena
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.per_group_manager import PerGroupCPUOffloadingManager
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

logger = init_logger(__name__)


class CPUOffloadingSpec(OffloadingSpec):
    BLOCK_SIZE_ALIGNMENT = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        definitions: dict[str, OffloadingMetricMetadata] = {
            CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by active "
                    "transfers (0.0 = idle, 1.0 = saturated). Sustained high "
                    "values indicate transfers (stores or promotions) may be "
                    "dropped due to insufficient capacity."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "in-flight stores that have not yet "
                    "completed (0.0 = idle, 1.0 = saturated)."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "in-flight loads that have not yet "
                    "completed (0.0 = idle, 1.0 = saturated)."
                ),
            ),
            CPUOffloadingMetrics.CPU_ALLOCATION_SIZE: OffloadingHistogramMetadata(
                documentation=(
                    "Histogram of the number of CPU blocks requested by each "
                    "KV offload prepare_store call."
                ),
                buckets=(1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144),
            ),
        }
        store_threshold = int(extra_config.get("store_threshold", 0))
        if store_threshold >= 2:
            definitions[CPUOffloadingMetrics.STORES_SKIPPED] = (
                OffloadingCounterMetadata(
                    documentation=(
                        "Number of KV offload stores skipped because the reuse "
                        "threshold was not reached."
                    ),
                )
            )
        return definitions

    def __init__(self, config: OffloadingConfig):
        super().__init__(config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        world_size = config.parallel.world_size
        self.num_blocks = 0
        self.kv_bytes_per_chunk = 0
        self.cpu_page_size_per_worker = 0
        self.replicated_layout = config.replicated_layout and self._uses_shared_region()

        # Per-group pools give each KV cache group a host row sized from its
        # own canonical pages, instead of one row of
        # worker_kv_bytes_per_block * world_size shared by every group. On a
        # hybrid model that removes both the replication of KV that is
        # identical on every rank and the padding of narrow groups up to the
        # widest one.
        self.arena: GroupArena | None = None
        if config.per_group_pools:
            assert config.group_canonical_bytes_per_block is not None
            if not self._uses_shared_region():
                raise ValueError(
                    "per_group_cpu_pools needs the shared /dev/shm offload "
                    "region, which this platform does not use."
                )
            extent_bytes = self.extra_config.get("cpu_pool_extent_bytes")
            self.arena = GroupArena(
                group_row_bytes=[
                    group_bytes * self.blocks_per_chunk
                    for group_bytes in config.group_canonical_bytes_per_block
                ],
                budget_bytes=int(cpu_bytes_to_use),
                extent_bytes=None if extent_bytes is None else int(extent_bytes),
            )
            # Deduplication is now per layer, driven by each mapping's writer
            # election; the aggregate flag would additionally silence every
            # non-zero rank's stores.
            self.replicated_layout = False
            direct_row = (
                config.worker_kv_bytes_per_block * world_size * self.blocks_per_chunk
            )
            logger.info(
                "Per-group CPU pools: rows %s, against %d per group with the "
                "direct layout — %.2fx less host memory to hold one block of "
                "every group. Groups that store less often than every block "
                "(recurrent checkpoints) widen that further at runtime.",
                self.arena.group_row_bytes,
                direct_row,
                (direct_row * len(self.arena.group_row_bytes))
                / max(sum(self.arena.group_row_bytes), 1),
            )

        if config.worker_kv_bytes_per_block > 0 and world_size > 0:
            num_copies = 1 if self.replicated_layout else world_size
            kv_bytes_per_block = config.worker_kv_bytes_per_block * num_copies
            kv_bytes_per_chunk = kv_bytes_per_block * self.blocks_per_chunk

            # calculate cpu_page_size_per_worker
            self.cpu_page_size_per_worker = kv_bytes_per_chunk // num_copies

            # calculate num_blocks
            aligned_kv_bytes_per_chunk = round_up(
                kv_bytes_per_chunk, self.BLOCK_SIZE_ALIGNMENT
            )
            self.num_blocks = int(cpu_bytes_to_use) // aligned_kv_bytes_per_chunk

            # Expose aligned_kv_bytes_per_chunk as
            # kv_bytes_per_chunk. Note that this might contain
            # some padding. i.e. each offloaded block is of the form,
            # |--- W0-B0---|---- W1-B0---| ... |---- Wn-B0---| *** maybe-pad *** |
            # or |--- B0 (single copy) ---| *** maybe-pad *** |
            self.kv_bytes_per_chunk = aligned_kv_bytes_per_chunk

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._worker: CPUOffloadingWorker | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")
        self.cache_policy_module_path: str | None = self.extra_config.get(
            "cache_policy_module_path"
        )

    @override
    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            # store_threshold: how many times a block must appear in lookup()
            # before it is eligible for CPU offloading.  Values < 2 disable
            # filtering (a threshold of 1 equals no filter; 0 is the default).
            store_threshold = int(self.extra_config.get("store_threshold", 0))

            # Maximum entries in the internal tracker's LRU table.
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))

            if self.arena is not None:
                self._manager = PerGroupCPUOffloadingManager(
                    arena=self.arena,
                    cache_policy=self.eviction_policy,
                    cache_policy_module_path=self.cache_policy_module_path,
                    enable_events=self.kv_events_config.enable_kv_cache_events,
                    store_threshold=store_threshold,
                    max_tracker_size=max_tracker_size,
                )
                return self._manager

            self._manager = CPUOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,
                cache_policy_module_path=self.cache_policy_module_path,
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
            )
        return self._manager

    def _uses_shared_region(self) -> bool:
        """Whether the worker CPU buffer is the shared mmap region (vs a private
        per-rank tensor); replicated-layout dedup is gated on this being True."""
        return current_platform.is_cuda_alike()

    def create_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        if self.arena is not None:
            return self._create_per_group_worker(kv_caches)

        mmap_region: SharedOffloadRegion | None = None
        # num_blocks == 0 would size the region to zero bytes, which cannot be
        # mmap'd; fall back to the tensor path (empty tensors) as before.
        if self._uses_shared_region() and self.num_blocks > 0:
            # Replicated layout puts all ranks on slot 0 (single MLA copy);
            # otherwise each rank takes its own slot by physical device index.
            if self.replicated_layout:
                rank = 0
            else:
                world_size = self.config.parallel.world_size
                rank = torch.accelerator.current_device_index() % world_size
            mmap_region = SharedOffloadRegion(
                engine_id=self.config.engine_id,
                num_blocks=self.num_blocks,
                rank=rank,
                kv_bytes_per_block=self.kv_bytes_per_chunk,
                cpu_page_size=self.cpu_page_size_per_worker,
            )
        try:
            return CPUOffloadingWorker(
                kv_caches=kv_caches,
                blocks_per_chunk=self.blocks_per_chunk,
                num_cpu_blocks=self.num_blocks,
                mmap_region=mmap_region,
            )
        except Exception:
            if mmap_region is not None:
                mmap_region.cleanup()
            raise

    def _create_per_group_worker(
        self, kv_caches: CanonicalKVCaches
    ) -> CPUOffloadingWorker:
        assert self.arena is not None
        # Fail closed rather than silently reverting to the direct layout: the
        # scheduler already sized its pools from the canonical widths, so a
        # worker on a different layout would write to the wrong offsets.
        derived: list[int] = []
        expected: list[int] = []
        for row, group_refs in zip(
            self.arena.group_row_bytes, kv_caches.group_data_refs
        ):
            if not group_refs:
                continue
            group_bytes = 0
            for ref in group_refs:
                if ref.mapping is None:
                    raise RuntimeError(
                        "per_group_cpu_pools was requested but the live KV "
                        "cache layout could not be certified for canonical "
                        "offload. Remove per_group_cpu_pools from "
                        "kv_connector_extra_config."
                    )
                group_bytes += ref.mapping.canonical_page_size_bytes
            derived.append(group_bytes * self.blocks_per_chunk)
            expected.append(row)
        assert all(d <= e for d, e in zip(derived, expected)), (
            f"worker-derived group rows {derived} exceed the rows the "
            f"scheduler sized the arena with {expected}"
        )

        world_size = self.config.parallel.world_size
        rank = torch.accelerator.current_device_index() % world_size
        mmap_region = SharedOffloadRegion(
            engine_id=self.config.engine_id,
            num_blocks=0,
            rank=rank,
            kv_bytes_per_block=0,
            cpu_page_size=0,
            arena_bytes=self.arena.total_bytes,
            stripe_count=world_size,
            numa_interleave=self.extra_config.get("cpu_pool_numa_policy", "interleave")
            == "interleave",
        )
        try:
            return CPUOffloadingWorker(
                kv_caches=kv_caches,
                blocks_per_chunk=self.blocks_per_chunk,
                num_cpu_blocks=0,
                mmap_region=mmap_region,
                canonical_layout=True,
                per_group_pools=True,
            )
        except Exception:
            mmap_region.cleanup()
            raise

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if not self._worker:
            if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
                raise Exception(
                    "CPU Offloading is currently only supported on CUDA-alike "
                    "and XPU GPUs"
                )
            self._worker = self.create_worker(kv_caches)

        assert self._worker is not None
        return self._worker
