# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-group CPU offload pools: arena addressing and pool routing."""

import mmap

import pytest

from vllm.v1.kv_offload.base import (
    LookupResult,
    get_offload_group_idx,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec, CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.group_arena import GroupArena
from vllm.v1.kv_offload.cpu.per_group_manager import PerGroupCPUOffloadingManager

from .test_manager import _EMPTY_REQ_CTX

PAGE = mmap.PAGESIZE
# A narrow group that stores on every chunk beside a wide one that stores
# rarely — the shape the uniform row cannot express.
NARROW = 4 * PAGE
WIDE = 32 * PAGE
EXTENT = 128 * PAGE


def make_arena(
    rows: list[int] | None = None,
    budget_extents: int = 8,
    extent_bytes: int = EXTENT,
) -> GroupArena:
    return GroupArena(
        group_row_bytes=rows if rows is not None else [NARROW, WIDE],
        budget_bytes=budget_extents * extent_bytes,
        extent_bytes=extent_bytes,
    )


def key(group_idx: int, n: int) -> bytes:
    return make_offload_key(n.to_bytes(8, "big"), group_idx)


def test_rows_are_page_aligned_and_slots_are_dense():
    arena = make_arena(rows=[NARROW + 1, WIDE])
    assert arena.group_row_bytes == [NARROW + PAGE, WIDE]
    assert arena.slots_per_extent == [EXTENT // (NARROW + PAGE), EXTENT // WIDE]

    # Consecutive slot ids inside one extent step by exactly the row width;
    # writer rotation depends on the ids being dense.
    row = arena.group_row_bytes[0]
    offsets = [arena.offset(0, slot) for slot in range(arena.num_slots(0))]
    assert offsets == [offsets[0] + i * row for i in range(len(offsets))]


def test_slots_of_different_groups_never_overlap():
    arena = make_arena()
    for group_idx in range(2):
        for _ in range(3):
            arena.grow(group_idx)

    spans: list[tuple[int, int]] = []
    for group_idx, row in enumerate(arena.group_row_bytes):
        for slot in range(arena.num_slots(group_idx)):
            start = arena.offset(group_idx, slot)
            spans.append((start, start + row))
    spans.sort()
    assert spans[0][0] >= 0
    assert spans[-1][1] <= arena.total_bytes
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))


def test_every_group_starts_with_an_extent():
    """A group that only stores late (a recurrent group waiting for its first
    checkpoint) must not be locked out by whichever group filled the arena."""
    arena = make_arena(rows=[NARROW, WIDE, WIDE])
    assert all(arena.num_extents_assigned(g) == 1 for g in range(3))


def test_budget_too_small_for_one_extent_per_group_is_rejected():
    with pytest.raises(ValueError, match="cannot give each"):
        GroupArena([NARROW, WIDE], budget_bytes=EXTENT, extent_bytes=EXTENT)


def test_growth_stops_when_the_arena_is_assigned():
    arena = make_arena(budget_extents=3)
    assert arena.grow(0) > 0  # third extent
    assert arena.grow(0) == 0


def make_manager(**kwargs) -> PerGroupCPUOffloadingManager:
    return PerGroupCPUOffloadingManager(arena=make_arena(**kwargs))


def store(manager, keys):
    output = manager.prepare_store(keys, _EMPTY_REQ_CTX)
    if output is not None:
        manager.complete_store(output.keys_to_store, _EMPTY_REQ_CTX)
    return output


def test_store_spec_offsets_resolve_to_the_owning_group():
    manager = make_manager()
    keys = [key(0, 0), key(0, 1), key(1, 0)]
    output = store(manager, keys)
    assert output is not None
    spec = output.store_spec
    assert isinstance(spec, CPULoadStoreSpec)
    assert spec.block_offsets is not None

    arena = manager.arena
    for offset, offload_key in zip(spec.block_offsets.tolist(), output.keys_to_store):
        group_idx = get_offload_group_idx(offload_key)
        group_offsets = [
            arena.offset(group_idx, slot) for slot in range(arena.num_slots(group_idx))
        ]
        assert offset in group_offsets


def test_spec_preserves_caller_order_across_groups():
    """The scheduler pairs the flat CPU list with a group-major GPU block
    list positionally, so interleaved input order has to come back intact."""
    manager = make_manager()
    keys = [key(1, 0), key(0, 0), key(1, 1), key(0, 1)]
    output = store(manager, keys)
    assert output is not None
    assert output.keys_to_store == sorted(keys, key=get_offload_group_idx)

    load_spec = manager.prepare_load(output.keys_to_store, _EMPTY_REQ_CTX)
    assert isinstance(load_spec, CPULoadStoreSpec)
    store_spec = output.store_spec
    assert isinstance(store_spec, CPULoadStoreSpec)
    assert load_spec.block_offsets.tolist() == store_spec.block_offsets.tolist()


def test_groups_do_not_evict_each_other():
    """Filling one group's pool must not cost the other group its blocks."""
    manager = make_manager(budget_extents=2)
    arena = manager.arena
    capacity = arena.num_slots(0)

    survivor = key(1, 0)
    assert store(manager, [survivor]) is not None

    for i in range(capacity * 2):
        assert store(manager, [key(0, i)]) is not None

    assert manager.lookup(survivor, _EMPTY_REQ_CTX) == LookupResult.HIT


def test_partial_group_failure_rolls_back_the_groups_that_succeeded():
    """prepare_store is all or nothing: the scheduler emits one job covering
    every group, so a half-allocated store would strand CPU slots."""
    manager = make_manager(budget_extents=2)
    arena = manager.arena

    # Pin every slot the wide group can get, so its next store fails.
    pinned = [key(1, i) for i in range(arena.num_slots(1))]
    assert store(manager, pinned) is not None
    manager.prepare_load(pinned, _EMPTY_REQ_CTX)  # ref_cnt > 0 => unevictable

    output = manager.prepare_store([key(0, 99), key(1, 999)], _EMPTY_REQ_CTX)
    assert output is None

    assert manager.lookup(key(0, 99), _EMPTY_REQ_CTX) == LookupResult.MISS


def test_usage_gauge_weights_groups_by_row_width():
    """Group pools hold rows of different widths, so a slot count would
    misreport how much of the host pool an in-flight store is holding."""
    manager = make_manager()
    assert manager.prepare_store([key(1, 0)], _EMPTY_REQ_CTX) is not None
    reduced = manager.get_stats().reduce()
    expected = WIDE / manager.arena.total_bytes
    assert reduced[CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC] == pytest.approx(expected)
    assert reduced[CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC] == pytest.approx(
        expected
    )


def test_reset_clears_blocks_but_keeps_the_split():
    """Returning extents would remap slot ids onto different bytes, so a store
    still in flight could land in another group's differently sized row."""
    manager = make_manager()
    assert store(manager, [key(0, 0), key(1, 0)]) is not None
    for _ in range(3):
        manager.arena.grow(0)
    extents = [manager.arena.num_extents_assigned(g) for g in range(2)]

    manager.reset_cache()

    assert manager.lookup(key(0, 0), _EMPTY_REQ_CTX) == LookupResult.MISS
    assert [manager.arena.num_extents_assigned(g) for g in range(2)] == extents
