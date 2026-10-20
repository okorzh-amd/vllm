# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Muse Spark specialized decode executor (layer mode): one routing decision
per step, then the layer kernels inside the stock layer loop.

Built by the model state once the weights are loaded (``MuseSparkModelState``):
checks the loaded weights, builds the per-rank launch state (``runner``) and the
layer kernels of every width, and compiles every build, before memory profiling
and graph capture. ``prepare_kv`` binds it to every KV cache vLLM allocates and
runs a startup self-test of every build; ``after_capture`` votes on the capture
warm-ups and tightens the in-kernel wait budget. Each step ``plan`` returns a
``Step`` or None; ``Step.layer(i)`` is the kernel replacing decoder layer i
(None for vLLM's layers). Weight reloads are not supported."""

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, NamedTuple, cast

import torch

from vllm.config import CUDAGraphMode
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.models.muse_spark.amd.specialized_decode.mono import spec
from vllm.models.muse_spark.amd.specialized_decode.mono.abi import (
    FAILED,
    STARTUP_WAIT_SECONDS,
    WAIT_SECONDS,
)
from vllm.models.muse_spark.amd.specialized_decode.runtime import HINT, DecodeConfig
from vllm.models.muse_spark.amd.specialized_decode.runtime.guards import (
    FailStop,
    kernel_layers,
    refuse_caches,
    refuse_model,
)
from vllm.models.muse_spark.amd.specialized_decode.runtime.layers import (
    KernelLayer,
    compile_build,
    k2_launch,
)
from vllm.models.muse_spark.amd.specialized_decode.runtime.peer import tp_agree
from vllm.models.muse_spark.amd.specialized_decode.runtime.runner import MonoRunner

logger = init_logger(__name__)


class Step(NamedTuple):
    """A planned step: ``width`` rows of whole uniform decode (or DSpark verify)
    requests."""

    width: int
    attn_metadata: Mapping[str, Any]  # by layer name
    slot_mapping: Mapping[str, torch.Tensor]  # by layer name (Phase B's K1)
    kernels: Mapping[int, KernelLayer]  # by decoder layer, at this width
    pad: torch.Tensor  # bool/uint8 [>= width], device, graph-stable

    def layer(self, idx: int) -> Callable | None:
        """The kernel replacing decoder layer idx in this step, or None (vLLM's):
        ``(positions, x) -> MuseSpark1xResidualStream``."""
        kernel = self.kernels.get(idx)
        return None if kernel is None else partial(kernel, self)


def step_reason(
    attn_metadata, names: Mapping[str, str], width: int, q: int
) -> str | None:
    """None if every named layer's ``AiterFlashAttentionMetadata`` (one layer of
    each KV kind among the kernel layers; identical on every TP rank) is a pure
    decode / uniform verify step of whole requests filling ``width`` rows, else
    why not."""
    if not isinstance(attn_metadata, dict):
        return "no metadata"  # profile and dummy runs, DBO's list
    for name in names.values():
        m: Any = attn_metadata.get(name)
        if m is None or m.decode_metadata is None:
            return "no decode metadata"
        if m.num_prefills or m.num_extends:
            return "not pure decode"
        if m.decode_metadata.max_query_len != q:
            return "not uniform"
        if m.num_decode_tokens != width or m.num_decodes * q != width:
            return "not uniform / padded width"
    return None


def agree(reasons: list[str], group, what: str) -> None:
    """Raises on every TP rank if any rank has ``reasons``: a rank that refused
    alone would leave the others spinning in-kernel."""
    if not tp_agree(not reasons, group):
        why = "; ".join(reasons) if reasons else "another TP rank refused"
        raise ValueError(
            f"Muse Spark specialized decode does not support {what}: {why} ({HINT})"
        )


def _health_reasons(runner: MonoRunner) -> list[str]:
    """This rank's given-up waits, once its FAILED word is set (device read)."""
    if not int(runner.health[FAILED].item()):
        return []
    from vllm.models.muse_spark.amd.specialized_decode.mono.common.debug import (
        given_up_waits,
    )

    waits = given_up_waits(runner.health, spec.REGIONS)
    return [f"rank {runner.rank} gave up waits: {'; '.join(waits) or 'unrecorded'}"]


class SpecializedDecode:
    """The executor: builds the layer kernels, binds them to each KV cache and
    plans every step."""

    def __init__(self, backbone, config: DecodeConfig, vllm_config=None):
        tp = get_tp_group()
        layers = kernel_layers(backbone)
        agree(refuse_model(backbone, layers), tp.cpu_group, "the loaded weights")
        if vllm_config is None:
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
        self.backbone, self.config, self.layers = backbone, config, layers
        self.vllm_config = vllm_config
        runner = self.runner = MonoRunner(next(backbone.parameters()).device)
        runner.first_layer = min(layers)
        L = backbone.layers
        # the first kernel layer of each KV kind: their metadata decide every
        # step, their caches key the bind
        self.first = {
            kind: min(i for i, k in layers.items() if k == kind)
            for kind in sorted(set(layers.values()))
        }
        self.names = {
            kind: L[i].self_attn.attn.layer_name for kind, i in self.first.items()
        }
        self.kernels: dict[int, dict[int, KernelLayer]] = {
            w: {i: KernelLayer(L[i], L[i - 1], runner, w) for i in sorted(layers)}
            for w in config.widths
        }
        builds = sorted(
            {
                b
                for by in self.kernels.values()
                for k in by.values()
                for b in k.builds()
            },
            key=lambda b: b.tokens,
        )
        # before memory profiling and capture; agreed, so a rank that cannot
        # compile stops every rank
        failure = None
        try:
            for build in builds:
                compile_build(build)
        except Exception as err:  # reported on every rank below
            failure = err
        try:
            agree(
                [f"compiling the kernels failed: {failure!r}"] if failure else [],
                tp.cpu_group,
                "this deployment",
            )
        except ValueError as err:
            raise err from failure
        self.failstop = FailStop(
            runner.health[FAILED : FAILED + 1], runner.rank, tp.world_size, tp.cpu_group
        )
        # (data_ptr, shape, stride) of each KV kind's cache prepare_kv bound
        self._kv: tuple | None = None
        logger.info(
            "Muse Spark specialized decode: %d kernel layers (%d..%d), widths %s, "
            "%d builds; other steps run vLLM's decode",
            len(layers),
            min(layers),
            max(layers),
            list(config.widths),
            len(builds),
        )

    def _kv_key(self) -> tuple | None:
        """(data_ptr, shape, stride) of the first kernel layer of each KV kind's
        cache; None while any is unbound (empty)."""
        key = []
        for i in self.first.values():
            kv = getattr(self.backbone.layers[i].self_attn.attn, "kv_cache", None)
            if not isinstance(kv, torch.Tensor) or kv.numel() == 0:
                return None
            key.append((kv.data_ptr(), tuple(kv.shape), kv.stride()))
        return tuple(key)

    def prepare_kv(self) -> None:
        """From ``MuseSparkModelState.initialize_kv_cache``, at every bind (the
        CUDA-graph memory-profiling cache, then the real one): checks the caches,
        binds them, raises the wait budget until the capture ends and runs the
        startup self-test."""
        self._kv = None
        key = self._kv_key()
        if key is None:
            return
        group = get_tp_group().cpu_group
        agree(
            refuse_caches(self.backbone, self.layers, self.vllm_config),
            group,
            "the KV caches",
        )
        self._kv = key
        self.runner.set_wait_budget(STARTUP_WAIT_SECONDS)
        self.self_test()
        # the served-steps log counts from here, not the self-test's steps
        self.runner.reset_count()
        self._log_graphs()

    def self_test(self) -> None:
        """Collective, outside any capture, on every TP rank: for every width two
        steps, each one launch of every distinct build at that width (the first
        kernel layer's weights) on a zero core and stream: (a) every row a pad row
        (no expert in the union: a capture warm-up's step), (b) no pad row (every
        stage live). The ranks then agree that none gave up a wait and that every
        output is finite (and zero in (a)). This also loads every code object
        before any capture (``compile_build`` only compiles)."""
        runner = self.runner
        device = runner.scratch.device
        kernels = {w: by[min(by)] for w, by in self.kernels.items()}
        outputs = []
        for w, kernel in kernels.items():
            core = torch.zeros(w, spec.CORE, dtype=torch.bfloat16, device=device)
            stream = torch.zeros(w, spec.HIDDEN, dtype=torch.float32, device=device)
            for all_pad in (True, False):
                pad = torch.full_like(runner.no_pad, int(all_pad))
                stream_out = torch.empty_like(stream)
                branch_out = torch.empty(
                    w, spec.HIDDEN, dtype=torch.bfloat16, device=device
                )
                runner.step_begin()
                for build in dict.fromkeys(kernel.builds()):
                    k2_launch(
                        build,
                        {
                            **kernel.static,
                            "core": core,
                            "stream": stream,
                            "pad": pad,
                            "stream_out": stream_out,
                            "branch_out": branch_out,
                        },
                    )
                outputs.append((w, all_pad, stream_out, branch_out))
        torch.accelerator.synchronize()
        reasons = _health_reasons(runner)
        for w, all_pad, *outs in outputs:
            for out in outs:
                if not bool(torch.isfinite(out).all()):
                    reasons.append(
                        f"rank {runner.rank}: width {w} gave non-finite rows"
                    )
                elif all_pad and bool(out.any()):
                    reasons.append(f"rank {runner.rank}: width {w} pad rows not zero")
        agree(reasons, get_tp_group().cpu_group, "the startup self-test")

    def after_capture(self) -> None:
        """From ``MuseSparkModelState.capture_inner_cudagraphs``, after the decoder
        graphs of a capture (profile-only or real): the ranks agree that no
        eager warm-up launch before the capture gave up a wait, then waits give
        up after WAIT_SECONDS. The next ``prepare_kv`` raises the budget again
        until the next capture ends. Under enforce-eager nothing is captured and
        the budget stays at STARTUP_WAIT_SECONDS."""
        torch.accelerator.synchronize()
        agree(
            _health_reasons(self.runner),
            get_tp_group().cpu_group,
            "the CUDA-graph capture warm-up",
        )
        self.runner.set_wait_budget(WAIT_SECONDS)
        # ... nor the warm-ups' (enforce-eager: from the KV bind)
        self.runner.reset_count()

    def _log_graphs(self) -> None:
        """Logs every width without a FULL decode graph of its own, applying
        MRV2's capture filters to the resolved capture sizes: MRV2 replays the
        next larger FULL decode graph with pad requests (``_init_candidates``
        maps a token count to the smallest graph at or above it), so such a
        width's steps run that graph's build; only widths above every FULL
        decode graph run the kernels eagerly."""
        from vllm.utils.math_utils import round_up

        cc = self.vllm_config.compilation_config
        q = self.config.q
        max_reqs = self.vllm_config.scheduler_config.max_num_seqs
        mode = cc.cudagraph_mode
        graphs = set()
        if mode is not None and mode.decode_mode() == CUDAGraphMode.FULL:
            for s in cc.cudagraph_capture_sizes or ():
                w = round_up(s, q)
                if (
                    w <= max_reqs * q
                    and w <= cc.max_cudagraph_capture_size
                    and w // q <= max_reqs
                ):
                    graphs.add(w)
        padded = {
            w: min(g for g in graphs if g > w)
            for w in self.config.widths
            if w not in graphs and any(g > w for g in graphs)
        }
        eager = [w for w in self.config.widths if w not in graphs and w not in padded]
        if padded:
            logger.info(
                "Muse Spark specialized decode: widths %s have no FULL decode "
                "graph of their own; their steps replay the graph of width %s "
                "with pad rows",
                list(padded),
                list(padded.values()),
            )
        if eager:
            logger.info(
                "Muse Spark specialized decode: widths %s have no FULL decode "
                "graph; their steps run the kernels eagerly",
                eager,
            )

    def plan(self, positions: torch.Tensor, intermediate_tensors) -> Step | None:
        """This step's plan, or None to run vLLM's layers.

        Steps that run vLLM's layers:
        - pipeline-parallel stage inputs, and steps before a KV cache is bound
          (profiling, dummy runs);
        - PIECEWISE steps (captured without attention metadata);
        - widths without kernels (more than MAX_ROWS rows);
        - steps that are not pure decode / uniform verify steps of whole requests
          (``step_reason``).

        Eager steps run the kernels too (enforce-eager, widths without a graph,
        MRV2's warm-ups before each capture). Every predicate is the same on all
        TP ranks: the kernels' mailboxes need every rank on one path. Under FULL
        graphs this runs at capture only and the plan is baked in.
        """
        if intermediate_tensors is not None or self._kv is None:
            return None
        if not is_forward_context_available():
            return None
        ctx = get_forward_context()
        if ctx.cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
            return None
        width = positions.size(-1)
        kernels = self.kernels.get(width)
        if kernels is None:
            return None
        md = ctx.attn_metadata
        if step_reason(md, self.names, width, self.config.q) is not None:
            return None
        if self._kv_key() != self._kv:
            raise RuntimeError(
                "Muse Spark specialized decode: the KV cache was re-bound without "
                "prepare_kv (ModelState.initialize_kv_cache)"
            )
        # Maintained per step only with VLLM_MOE_SKIP_PADDING=1 (refused
        # otherwise); captures and warm-ups fill it all True.
        pad = ctx.is_padding if ctx.is_padding is not None else self.runner.no_pad
        # step_reason accepted md: one step's dicts (not DBO's lists)
        return Step(
            width,
            cast(Mapping[str, Any], md),
            cast(Mapping[str, torch.Tensor], ctx.slot_mapping),
            kernels,
            pad,
        )

    def after_step(self) -> None:
        """Once per executed step, from compute_logits (eager on every rank, also
        under graph replay). Raises on every rank once a rank gave up an
        in-kernel wait."""
        self.failstop.after_step()
        self.runner.count_step()
