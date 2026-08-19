# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte arena that hands each KV cache group its own host slot width.

The uniform host row (``worker_kv_bytes_per_block * world_size``) pays twice on
a hybrid model: once for KV that is bit-identical on every rank, and once more
because the row is padded to the widest group. Sizing a row per group removes
both, but the widths then differ, so the pool can no longer be a flat array of
equal blocks.

Groups do not consume the pool in a ratio that is knowable up front: a
full-attention group stores every chunk while a recurrent group stores only its
checkpoints, and the checkpoint cadence is a runtime property. A static split
would therefore have to be conservative enough to waste most of the win. This
arena instead carves fixed-size extents and assigns them to groups on demand,
so the split converges on the ratio the workload actually asks for, while slot
ids stay dense per group (an extent holds a whole number of that group's
slots), which keeps writer rotation uniform.
"""

import mmap

from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)

# Extents must comfortably exceed the widest group row, or a group whose row is
# a large fraction of an extent wastes the remainder of every extent it takes.
_MIN_SLOTS_PER_EXTENT = 16
_MIN_EXTENT_BYTES = 1 << 30


class GroupArena:
    """On-demand extent allocator over one contiguous host region.

    Address space only: the arena hands out byte offsets and never touches the
    memory behind them.
    """

    ALIGNMENT: int = mmap.PAGESIZE

    def __init__(
        self,
        group_row_bytes: list[int],
        budget_bytes: int,
        extent_bytes: int | None = None,
    ) -> None:
        assert group_row_bytes, "arena needs at least one group"
        assert all(row > 0 for row in group_row_bytes)
        self.group_row_bytes = [
            round_up(row, self.ALIGNMENT) for row in group_row_bytes
        ]

        widest = max(self.group_row_bytes)
        if extent_bytes is None:
            extent_bytes = max(_MIN_EXTENT_BYTES, _MIN_SLOTS_PER_EXTENT * widest)
        self.extent_bytes = round_up(extent_bytes, self.ALIGNMENT)
        assert self.extent_bytes >= widest, (
            f"cpu_pool_extent_bytes={self.extent_bytes} is smaller than the "
            f"widest group row ({widest})"
        )

        num_groups = len(self.group_row_bytes)
        self.num_extents = budget_bytes // self.extent_bytes
        if self.num_extents < num_groups:
            raise ValueError(
                f"CPU offload pool of {budget_bytes} bytes holds only "
                f"{self.num_extents} extents of {self.extent_bytes} bytes, "
                f"which cannot give each of the {num_groups} KV cache groups "
                f"even one. Raise cpu_bytes_to_use or lower "
                f"cpu_pool_extent_bytes."
            )
        self.total_bytes = self.num_extents * self.extent_bytes

        self.slots_per_extent = [
            self.extent_bytes // row for row in self.group_row_bytes
        ]
        # Every group starts with one extent so a group that only stores late
        # (e.g. a recurrent group waiting for its first checkpoint) is not
        # locked out by whichever group happened to fill the arena first.
        self._extent_bases: list[list[int]] = [[] for _ in range(num_groups)]
        self._next_extent = 0
        for group_idx in range(num_groups):
            assigned = self.grow(group_idx)
            assert assigned > 0

        logger.info(
            "CPU offload arena: %.2f GB in %d extents of %.2f GB; "
            "group rows %s; slots/extent %s",
            self.total_bytes / 1e9,
            self.num_extents,
            self.extent_bytes / 1e9,
            self.group_row_bytes,
            self.slots_per_extent,
        )

    @property
    def num_groups(self) -> int:
        return len(self.group_row_bytes)

    def max_slots(self, group_idx: int) -> int:
        """Slots this group could hold if it took every extent."""
        return self.num_extents * self.slots_per_extent[group_idx]

    def num_slots(self, group_idx: int) -> int:
        """Slots currently backed by extents assigned to this group."""
        return len(self._extent_bases[group_idx]) * self.slots_per_extent[group_idx]

    def num_extents_assigned(self, group_idx: int) -> int:
        return len(self._extent_bases[group_idx])

    def grow(self, group_idx: int) -> int:
        """Assign one more extent to a group. Returns the slots gained (0 if
        the arena is fully assigned)."""
        if self._next_extent >= self.num_extents:
            return 0
        self._extent_bases[group_idx].append(self._next_extent * self.extent_bytes)
        self._next_extent += 1
        return self.slots_per_extent[group_idx]

    def offset(self, group_idx: int, slot_id: int) -> int:
        """Byte offset of a group-local slot within the arena."""
        slots_per_extent = self.slots_per_extent[group_idx]
        extent_idx, within = divmod(slot_id, slots_per_extent)
        return (
            self._extent_bases[group_idx][extent_idx]
            + within * self.group_row_bytes[group_idx]
        )

    def offsets(self, group_idx: int, slot_ids: list[int]) -> list[int]:
        slots_per_extent = self.slots_per_extent[group_idx]
        row = self.group_row_bytes[group_idx]
        bases = self._extent_bases[group_idx]
        return [
            bases[slot_id // slots_per_extent] + (slot_id % slots_per_extent) * row
            for slot_id in slot_ids
        ]
