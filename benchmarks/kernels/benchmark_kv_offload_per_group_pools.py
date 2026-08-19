# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host capacity and transfer throughput of per-group CPU offload pools.

Two questions this answers for a given model geometry:

1. How many tokens of context a host pool of a given size holds, under the
   direct per-rank layout, under the canonical layout with a uniform row, and
   under per-group rows.
2. Whether addressing CPU slots by arena byte offset costs anything in the
   DMA path relative to the row-index arithmetic it replaces.

Defaults describe Kimi-K3 at TP8: 24 MLA layers whose latent is replicated on
every rank, and 69 recurrent layers whose state is not, sharing one padded
884,736 B page and a 1536-token block.

Example:
    python benchmarks/kernels/benchmark_kv_offload_per_group_pools.py \
        --host-pool-gb 1024 --transfer-blocks 64
"""

import argparse
import time
import uuid

import torch

from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalPageMapping,
    CopyRun,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import (
    GroupArenaLayout,
    SingleDirectionOffloadingHandler,
    pin_mmap_region,
)
from vllm.v1.kv_offload.cpu.group_arena import GroupArena
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

GiB = 1 << 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--page-bytes", type=int, default=884_736)
    parser.add_argument("--tokens-per-block", type=int, default=1536)
    parser.add_argument(
        "--replicated-layers",
        type=int,
        default=24,
        help="layers whose KV is bit-identical on every rank (MLA latent)",
    )
    parser.add_argument(
        "--private-layers",
        type=int,
        default=69,
        help="layers whose KV differs per rank (recurrent state)",
    )
    parser.add_argument(
        "--private-groups",
        type=int,
        default=3,
        help="how many KV cache groups the private layers are split across",
    )
    parser.add_argument("--host-pool-gb", type=float, default=1024.0)
    parser.add_argument(
        "--recurrent-store-blocks",
        type=int,
        nargs="+",
        default=[1, 5, 21],
        help=(
            "blocks between stores of a recurrent group, i.e. its checkpoint "
            "cadence in blocks. 1 means it stores on every block like a "
            "full-attention group; larger values are the realistic case, and "
            "the ratio the change is worth depends on it."
        ),
    )
    parser.add_argument(
        "--transfer-blocks",
        type=int,
        default=0,
        help="blocks per timed transfer; 0 skips the GPU throughput section",
    )
    parser.add_argument("--transfer-layers", type=int, default=8)
    parser.add_argument("--iters", type=int, default=5)
    return parser.parse_args()


def report_capacity(args: argparse.Namespace) -> None:
    page = args.page_bytes
    world = args.world_size
    private_per_group = args.private_layers // args.private_groups

    # vLLM allocates one physical tensor per layer slot, shared by one layer
    # from every group and padded to the widest page, so a manager block costs
    # the same in every group.
    layers_per_group = max(args.replicated_layers, private_per_group)
    direct_row = layers_per_group * page * world

    # Canonical layout with a uniform row: the row is a max over the groups
    # sharing each tensor, so the replicated group's dedup buys nothing.
    canonical_uniform_row = max(
        args.replicated_layers * page,  # one copy
        private_per_group * page * world,
    )

    per_group_rows = [args.replicated_layers * page] + [
        private_per_group * page * world
    ] * args.private_groups

    budget = int(args.host_pool_gb * GiB)

    print(f"\n{'=' * 78}")
    print(
        f"Host capacity: {args.replicated_layers} replicated + "
        f"{args.private_layers} private layers, page {page} B, "
        f"world_size {world}, pool {args.host_pool_gb:g} GiB"
    )
    print(f"{'=' * 78}")
    print(f"row bytes per block: direct {direct_row:,} for every group")
    print(f"                     canonical uniform {canonical_uniform_row:,}")
    print(f"                     per-group {per_group_rows}")
    print()
    print(
        f"{'recurrent cadence':>18}{'direct B/tok':>16}{'uniform B/tok':>16}"
        f"{'per-group B/tok':>18}{'tokens/pool':>16}{'gain':>8}"
    )

    for interval in args.recurrent_store_blocks:
        # A recurrent group holds one slot per checkpoint, not per block, so
        # what the layouts are worth depends on how often it checkpoints.
        # Slots per token, per group, at this cadence:
        replicated_slots = 1 / args.tokens_per_block
        private_slots = 1 / (args.tokens_per_block * interval)

        direct = direct_row * (replicated_slots + args.private_groups * private_slots)
        uniform = canonical_uniform_row * (
            replicated_slots + args.private_groups * private_slots
        )
        per_group = per_group_rows[0] * replicated_slots + sum(
            row * private_slots for row in per_group_rows[1:]
        )
        label = "every block" if interval == 1 else f"1 in {interval} blocks"
        print(
            f"{label:>18}{direct:>16,.0f}{uniform:>16,.0f}{per_group:>18,.0f}"
            f"{budget / per_group:>16,.0f}{direct / per_group:>7.2f}x"
        )

    print(
        "\nThe cadence is a runtime property, which is why the arena assigns "
        "extents\non demand rather than splitting the pool up front."
    )

    arena = GroupArena(per_group_rows, budget_bytes=budget)
    print(
        f"arena: {arena.num_extents} extents of "
        f"{arena.extent_bytes / GiB:.2f} GiB, slots/extent {arena.slots_per_extent}, "
        f"max slots {[arena.max_slots(g) for g in range(arena.num_groups)]}"
    )


def _whole_page_mapping(page: int, num_writers: int, rank: int) -> CanonicalPageMapping:
    run = CopyRun(0, 0, page, 1, page, page)
    return CanonicalPageMapping(page, page, (run,), num_writers, rank, True)


def _time_transfers(handler, src_spec, dst_spec, iters: int) -> float:
    """Seconds per transfer, after one warm-up."""
    for i in range(iters + 1):
        if i == 1:
            torch.accelerator.synchronize()
            start = time.perf_counter()
        assert handler.transfer_async(i, src_spec, dst_spec)
        while not handler.get_finished():
            pass
    torch.accelerator.synchronize()
    return (time.perf_counter() - start) / iters


def report_throughput(args: argparse.Namespace) -> None:
    if args.transfer_blocks <= 0:
        return
    if not torch.cuda.is_available():
        print("\nNo accelerator visible; skipping the transfer section.")
        return

    page = args.page_bytes
    num_blocks = args.transfer_blocks
    num_layers = args.transfer_layers
    gpu_tensors = [
        torch.randint(-128, 128, (num_blocks, page), dtype=torch.int8, device="cuda")
        for _ in range(num_layers)
    ]
    refs = [
        [
            CanonicalKVCacheRef(
                tensor_idx=i,
                page_size_bytes=page,
                mapping=_whole_page_mapping(page, 1, 0),
            )
            for i in range(num_layers)
        ]
    ]

    row = num_layers * page
    # One extent per 4 slots keeps the timed slots spread over several
    # extents, so the offset gather is exercised, not just a flat stride.
    extent = 4 * row
    arena = GroupArena([row], budget_bytes=(num_blocks + 4) * row, extent_bytes=extent)
    while arena.num_slots(0) < num_blocks:
        assert arena.grow(0) > 0
    region = SharedOffloadRegion(
        engine_id=f"bench-{uuid.uuid4()}",
        num_blocks=0,
        rank=0,
        kv_bytes_per_block=0,
        cpu_page_size=0,
        arena_bytes=arena.total_bytes,
        stripe_count=1,
    )
    try:
        pin_mmap_region(region)
        layout = GroupArenaLayout(
            base_ptr=region.base_ptr,
            layer_offsets=[[i * page for i in range(num_layers)]],
        )
        slots = list(range(num_blocks))
        gpu_spec = GPULoadStoreSpec(
            slots, group_sizes=(num_blocks,), block_indices=(0,)
        )
        cpu_spec = CPULoadStoreSpec(slots, arena.offsets(0, slots))
        moved = num_blocks * num_layers * page

        print(f"\n{'=' * 78}")
        print(
            f"Transfer: {num_blocks} blocks x {num_layers} layers x {page} B "
            f"= {moved / GiB:.2f} GiB per direction"
        )
        print(f"{'=' * 78}")
        # Both handlers share the gpu_tensors list, and shutdown() clears it,
        # so build them before timing either direction.
        handlers = {
            gpu_to_cpu: SingleDirectionOffloadingHandler(
                gpu_tensors=gpu_tensors,
                cpu_tensors=[],
                blocks_per_chunk=1,
                layer_refs_per_group=refs,
                gpu_to_cpu=gpu_to_cpu,
                canonical_layout=True,
                arena=layout,
            )
            for gpu_to_cpu in (True, False)
        }
        for gpu_to_cpu, handler in handlers.items():
            src, dst = (gpu_spec, cpu_spec) if gpu_to_cpu else (cpu_spec, gpu_spec)
            seconds = _time_transfers(handler, src, dst, args.iters)
            label = "D2H (store)" if gpu_to_cpu else "H2D (load)"
            print(
                f"{label:<14}{seconds * 1e3:>9.2f} ms"
                f"{moved / seconds / 1e9:>10.1f} GB/s"
                f"{seconds / (num_blocks * num_layers) * 1e6:>12.2f} us/descriptor"
            )
    finally:
        region.cleanup()


def main() -> None:
    args = parse_args()
    report_capacity(args)
    report_throughput(args)


if __name__ == "__main__":
    main()
