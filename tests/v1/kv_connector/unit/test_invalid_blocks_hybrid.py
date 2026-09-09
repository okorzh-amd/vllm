# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Block-level KV load-failure recovery on models with several KV cache groups.

A token is computed only if every group holds valid KV for it, so a failure in
one group has to rewind the request past the whole token span, in every group.
"""

from collections.abc import Callable
from unittest.mock import Mock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    SupportsHMA,
)
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from .utils import (
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
    make_kv_cache_config,
)

pytestmark = pytest.mark.cpu_test


def _scheduler_with_group_block_sizes(
    vllm_config, kv_cache_config: KVCacheConfig
) -> Scheduler:
    """Scheduler for groups whose block sizes differ.

    ``create_scheduler`` takes both sizes from the cache config, which only
    works when every group shares that block size. Derive them the way
    EngineCore does instead: the scheduler block size is the LCM of the group
    block sizes and the hashing granularity is their GCD.
    """
    block_size, hash_block_size = resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    vllm_config.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=block_size,
        hash_block_size=hash_block_size,
    )


def _two_groups(block_size_0: int, block_size_1: int) -> KVCacheConfig:
    """Two full-attention groups whose block sizes may differ."""
    return KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                [f"layer{i}"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
            for i, block_size in enumerate((block_size_0, block_size_1))
        ],
    )


def _make_get_num_new_matched_tokens(
    num_matched: int,
) -> Callable[[Request, int], tuple[int, bool]]:
    def get_num_new_matched_tokens(request: Request, _: int) -> tuple[int, bool]:
        return num_matched, False

    return get_num_new_matched_tokens


def _run_until_failure(
    scheduler: Scheduler,
    num_prompt_tokens: int,
    num_external_computed_tokens: int,
    pick_invalid_blocks: Callable[[tuple[list[int], ...]], set[int]],
) -> Request:
    """Schedule one request with a sync external load, then fail some blocks.

    Returns the request after the scheduler has handled the failure.
    """
    # Block hashes must be computed at the scheduler's hashing granularity,
    # which is the GCD of the group block sizes on a hybrid model.
    request = create_request(
        num_tokens=num_prompt_tokens, block_size=scheduler.hash_block_size
    )
    scheduler.add_request(request=request)

    # An HMA-aware connector: the scheduler asserts a single KV cache group for
    # connectors that do not declare support.
    scheduler.connector = Mock(spec=type("C", (KVConnectorBase_V1, SupportsHMA), {}))
    scheduler.connector.get_num_new_matched_tokens.side_effect = (
        _make_get_num_new_matched_tokens(num_external_computed_tokens)
    )
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.request_finished_all_groups.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()

    scheduler_output = scheduler.schedule()
    assert request.status == RequestStatus.RUNNING

    block_ids = scheduler_output.scheduled_new_reqs[0].block_ids
    scheduler.update_from_output(
        scheduler_output,
        create_model_runner_output(
            [request],
            invalid_block_ids=pick_invalid_blocks(block_ids),
            use_eos=False,
        ),
    )
    return request


@pytest.fixture
def vllm_config():
    config = create_vllm_config()
    config.kv_transfer_config.kv_load_failure_policy = "recompute"
    return config


def test_single_group_rewind_is_unchanged(vllm_config):
    """The single-group path must keep truncating at the failed block."""
    scheduler = create_scheduler(vllm_config)
    block_size = scheduler.block_size

    request = _run_until_failure(
        scheduler,
        num_prompt_tokens=100 * block_size,
        num_external_computed_tokens=99 * block_size,
        pick_invalid_blocks=lambda block_ids: {block_ids[0][50]},
    )

    assert request.status == RequestStatus.RUNNING
    assert request.num_computed_tokens == 50 * block_size


@pytest.mark.parametrize("failing_group", [0, 1])
def test_hybrid_equal_block_sizes_rewinds_from_either_group(
    vllm_config, failing_group: int
):
    """A failure in any group rewinds the request. Regression test for #50687.

    Before group-aware recovery the scheduler unpacked a single group and died
    with ValueError on any model with more than one.
    """
    block_size = vllm_config.cache_config.block_size
    scheduler = create_scheduler(
        vllm_config,
        kv_cache_config=make_kv_cache_config(
            block_size=block_size, swa_enabled=True, sw_size=1024, num_blocks=10000
        ),
    )
    assert len(scheduler.kv_cache_config.kv_cache_groups) == 2

    request = _run_until_failure(
        scheduler,
        num_prompt_tokens=50 * block_size,
        num_external_computed_tokens=49 * block_size,
        pick_invalid_blocks=lambda block_ids: {block_ids[failing_group][20]},
    )

    assert request.status == RequestStatus.RUNNING
    assert request.num_computed_tokens == 20 * block_size


def test_hybrid_rewind_is_aligned_to_the_scheduler_block_size(vllm_config):
    """A narrow group's failure rewinds past the wide group's whole block.

    Group 1's block 11 covers tokens 88-95, which sit inside group 0's block 5
    (80-95). Resuming at 88 would leave group 0 mid-block, so the request must
    rewind to 80.
    """
    scheduler = _scheduler_with_group_block_sizes(vllm_config, _two_groups(16, 8))
    assert scheduler.block_size == 16

    request = _run_until_failure(
        scheduler,
        num_prompt_tokens=64 * 16,
        num_external_computed_tokens=32 * 16,
        pick_invalid_blocks=lambda block_ids: {block_ids[1][11]},
    )

    assert request.num_computed_tokens == 80


def test_hybrid_earliest_failure_across_groups_wins(vllm_config):
    """Two groups fail at once; the earlier token position decides the rewind."""
    scheduler = _scheduler_with_group_block_sizes(vllm_config, _two_groups(16, 8))

    request = _run_until_failure(
        scheduler,
        num_prompt_tokens=64 * 16,
        num_external_computed_tokens=32 * 16,
        # group 0 block 10 -> token 160; group 1 block 60 -> token 480
        pick_invalid_blocks=lambda block_ids: {block_ids[0][10], block_ids[1][60]},
    )

    assert request.num_computed_tokens == 160


def test_null_block_is_not_treated_as_invalid(vllm_config):
    """A sparse group parks dropped prefixes on the shared null block.

    It holds no tokens and is common to every request, so reporting it must not
    rewind anything.
    """
    block_size = vllm_config.cache_config.block_size
    # A window this short leaves the sliding-window group parked on the null
    # block for most of the prompt.
    scheduler = create_scheduler(
        vllm_config,
        kv_cache_config=make_kv_cache_config(
            block_size=block_size,
            swa_enabled=True,
            sw_size=4 * block_size,
            num_blocks=10000,
        ),
    )
    null_block_id = scheduler.kv_cache_manager.block_pool.null_block.block_id

    request = _run_until_failure(
        scheduler,
        num_prompt_tokens=50 * block_size,
        num_external_computed_tokens=49 * block_size,
        pick_invalid_blocks=lambda block_ids: {null_block_id},
    )

    sliding_window_blocks = scheduler.kv_cache_manager.get_block_ids(
        request.request_id
    )[1]
    assert null_block_id in sliding_window_blocks
    assert request.status == RequestStatus.RUNNING
    # Untouched: the whole prompt is still considered computed.
    assert request.num_computed_tokens == 50 * block_size
