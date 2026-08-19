# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One CPU offload pool per KV cache group, over a shared byte arena."""

from collections import defaultdict
from collections.abc import Collection, Iterable

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec, CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.group_arena import GroupArena
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

logger = init_logger(__name__)

# Extent quota prior, in slots per group. Keeps the very first growth
# decisions from being dictated by a handful of stores while still letting
# measured demand take over quickly.
_QUOTA_PRIOR_SLOTS = 8


class PerGroupCPUOffloadingManager(OffloadingManager):
    """Routes each offload key to its KV cache group's own pool.

    Each group gets a `CPUOffloadingManager` over slots of that group's row
    width, all drawn from one `GroupArena`. Extents are handed out on demand
    and capped by a demand-proportional quota, so a group that stores on every
    chunk cannot permanently crowd out one that stores only at checkpoints.

    Slot ids are group-local, so the emitted spec also carries the arena byte
    offset of each slot.
    """

    def __init__(
        self,
        arena: GroupArena,
        cache_policy: str = "lru",
        cache_policy_module_path: str | None = None,
        enable_events: bool = False,
        store_threshold: int = 0,
        max_tracker_size: int = 64_000,
    ) -> None:
        self.arena = arena
        num_groups = arena.num_groups
        self._pools = [
            CPUOffloadingManager(
                num_blocks=arena.num_slots(group_idx),
                cache_policy=cache_policy,
                cache_policy_module_path=cache_policy_module_path,
                enable_events=enable_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
                policy_capacity=arena.max_slots(group_idx),
            )
            for group_idx in range(num_groups)
        ]

        # Cumulative host bytes each group has asked to store. Used only to
        # apportion extents; a cap, never a reservation.
        self._demand_bytes = [0] * num_groups

    def _pool_for(self, key: OffloadKey) -> CPUOffloadingManager:
        return self._pools[get_offload_group_idx(key)]

    def _group_keys(
        self, keys: Collection[OffloadKey]
    ) -> dict[int, list[tuple[int, OffloadKey]]]:
        """Split keys by group, keeping each key's index in the input.

        The caller's order is the order the worker will pair CPU slots with
        GPU blocks, so it has to survive the round trip through the pools.
        """
        grouped: dict[int, list[tuple[int, OffloadKey]]] = defaultdict(list)
        for idx, key in enumerate(keys):
            grouped[get_offload_group_idx(key)].append((idx, key))
        return grouped

    def _may_grow(self, group_idx: int) -> bool:
        """Whether this group is still inside its demand-proportional quota."""
        rows = self.arena.group_row_bytes
        weights = [
            self._demand_bytes[g] + _QUOTA_PRIOR_SLOTS * rows[g]
            for g in range(self.arena.num_groups)
        ]
        total = sum(weights)
        quota = max(1, int(self.arena.num_extents * weights[group_idx] / total))
        return self.arena.num_extents_assigned(group_idx) < quota

    def _reserve(self, group_idx: int, num_slots: int) -> None:
        """Back up to num_slots free slots with extents, within quota."""
        pool = self._pools[group_idx]
        while pool.num_free_blocks < num_slots and self._may_grow(group_idx):
            if self.arena.grow(group_idx) == 0:
                break
            pool.set_num_blocks(self.arena.num_slots(group_idx))

    # --- OffloadingManager interface ---

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return self._pools[0].on_new_request(req_context)

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        return self._pool_for(key).lookup(key, req_context)

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        for group_idx, entries in self._group_keys(keys).items():
            self._pools[group_idx].touch([key for _, key in entries], req_context)

    @override
    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        for group_idx, entries in self._group_keys(keys).items():
            self._pools[group_idx].complete_load(
                [key for _, key in entries], req_context
            )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        for group_idx, entries in self._group_keys(keys).items():
            self._pools[group_idx].complete_store(
                [key for _, key in entries], req_context, success=success
            )

    @override
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        num_keys = len(keys)
        slot_ids = [0] * num_keys
        offsets = [0] * num_keys
        for group_idx, entries in self._group_keys(keys).items():
            spec = self._pools[group_idx].prepare_load(
                [key for _, key in entries], req_context
            )
            assert isinstance(spec, CPULoadStoreSpec)
            group_slots = spec.block_ids.tolist()
            group_offsets = self.arena.offsets(group_idx, group_slots)
            for (idx, _), slot, offset in zip(entries, group_slots, group_offsets):
                slot_ids[idx] = slot
                offsets[idx] = offset
        return CPULoadStoreSpec(slot_ids, offsets)

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        grouped = self._group_keys(keys)

        # Ordered by group so the flat spec matches the group-major GPU block
        # list the scheduler builds alongside it.
        outputs: list[tuple[int, PrepareStoreOutput]] = []
        for group_idx in sorted(grouped):
            entries = grouped[group_idx]
            group_keys = [key for _, key in entries]
            pool = self._pools[group_idx]
            self._reserve(group_idx, pool.count_unstored(group_keys))
            output = pool.prepare_store(group_keys, req_context)
            if output is None:
                # A partial store would hand the worker CPU slots for some
                # groups and none for others, which the scheduler's
                # all-or-nothing job accounting cannot express. Undo the
                # groups that did succeed.
                for done_idx, done in outputs:
                    self._pools[done_idx].complete_store(
                        done.keys_to_store, req_context, success=False
                    )
                return None
            outputs.append((group_idx, output))

        keys_to_store: list[OffloadKey] = []
        evicted_keys: list[OffloadKey] = []
        slot_ids: list[int] = []
        offsets: list[int] = []
        for group_idx, output in outputs:
            spec = output.store_spec
            assert isinstance(spec, CPULoadStoreSpec)
            group_slots = spec.block_ids.tolist()
            keys_to_store.extend(output.keys_to_store)
            evicted_keys.extend(output.evicted_keys)
            slot_ids.extend(group_slots)
            offsets.extend(self.arena.offsets(group_idx, group_slots))
            self._demand_bytes[group_idx] += (
                len(group_slots) * (self.arena.group_row_bytes[group_idx])
            )

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=CPULoadStoreSpec(slot_ids, offsets),
            evicted_keys=evicted_keys,
        )

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        for pool in self._pools:
            pool.on_request_finished(req_context)

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        for pool in self._pools:
            yield from pool.take_events()

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        for pool in self._pools:
            pool.on_schedule_end(context)

    @override
    def reset_cache(self) -> None:
        # Extents stay where they are. Returning them would remap slot ids to
        # different bytes, so a store still in flight from before the reset
        # could land in another group's row, whose width it does not match --
        # a hazard the uniform layout cannot have. The split is demand-derived
        # and worth keeping across a cache reset anyway.
        for pool in self._pools:
            pool.reset_cache()

    @override
    def shutdown(self) -> None:
        for pool in self._pools:
            pool.shutdown()

    @override
    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = OffloadingConnectorStats()
        rows = self.arena.group_row_bytes
        used_bytes = 0
        write_bytes = 0
        for group_idx, pool in enumerate(self._pools):
            group_stats = pool.get_stats()
            if group_stats is not None:
                stats.aggregate(group_stats)
            # The per-group gauges are fractions of pools with different row
            # widths, so they are recomputed arena-wide in bytes rather than
            # averaged.
            used_bytes += pool.num_used_blocks * rows[group_idx]
            write_bytes += pool.num_write_pending_blocks * rows[group_idx]

        total = float(self.arena.total_bytes) or 1.0
        usage = used_bytes / total
        write_usage = write_bytes / total
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC, usage)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC, write_usage)
        stats.set_gauge(
            CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC,
            max(usage - write_usage, 0.0),
        )
        logger.debug(
            "CPU offload arena: extents per group %s, slots per group %s",
            [self.arena.num_extents_assigned(g) for g in range(len(self._pools))],
            [pool.num_blocks for pool in self._pools],
        )
        return stats
