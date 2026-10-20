# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The layer kernel (Phase A): takes and returns what
``MuseSpark1xDecoderLayer.forward`` does, a ``MuseSpark1xResidualStream`` (or
the first layer's plain stream) in, the layer's deferred FFN boundary out.

vLLM runs the layer's residual boundary and attention up to o_proj's input (the
gated core); one K2 launch then runs o_proj and its all-reduce, the attention
boundary, the router and routing, the pooler, the MXFP4 experts, the expert
reduce, the post-expert norm and the unpooler.

Weights are views of the loaded parameters, taken once when the layer kernel is
built: the model refuses weight reloads once the executor exists
(``hooks.refuse_weight_reload``), so the parameters never move. Replicated
weights are passed whole; the kernel offsets them by its ``rank`` argument.

The kernel module (``mono.layer``, FlyDSL) is imported only inside the
functions that build or launch it.
"""

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from vllm.models.muse_spark.amd.specialized_decode.mono import spec
from vllm.models.muse_spark.amd.specialized_decode.mono.spec import K2Build
from vllm.models.muse_spark.amd.specialized_decode.runtime.runner import MonoRunner

if TYPE_CHECKING:
    from vllm.models.muse_spark.amd.specialized_decode.runtime.decode import Step
    from vllm.models.muse_spark.common.layers import MuseSpark1xResidualStream

# the arguments a launch adds to the layer's static ones
STEP_ARGS = ("core", "stream", "pad", "stream_out", "branch_out")


def k2_tensors(layer, runner: MonoRunner) -> dict[str, torch.Tensor]:
    """A kernel layer's weights by K2 argument name (design section 4): the
    loaded parameters and buffers themselves, no copies. ``refuse_model`` has
    checked every shape, dtype and layout."""
    attn, ffn = layer.self_attn, layer.feed_forward
    experts = ffn.experts.routed_experts
    return {
        "w_o": attn.o_proj.weight,
        "attn_alpha": layer.add_attn.alpha,
        "attn_beta": layer.add_attn.beta,
        "w_preffn": layer.pre_ffn_norm.weight,
        "w_router": ffn.router.weight,  # x output multiplier (folded at load)
        "bias": ffn.expert_bias,
        "w_pool": ffn.pooler.weight,
        "w_preexp": ffn.pre_expert_norm.weight,
        "w13": experts.w13_weight,
        "w13s": experts.w13_weight_scale,
        "w2": experts.w2_weight,
        "w2s": experts.w2_weight_scale,
        "w_postexp": ffn.post_expert_norm.weight,
        "w_unpool": ffn.unpooler.weight,
    }


def k2_launch(key: K2Build, args: Mapping[str, torch.Tensor | int]) -> None:
    """One K2 launch on the current stream (``mono.layer.k2_launch``)."""
    from vllm.models.muse_spark.amd.specialized_decode.mono import layer as k2

    k2.k2_launch(key, args)


def compile_build(key: K2Build) -> None:
    """Compiles one launch's kernel without running it: FlyDSL's compile-only
    mode returns before the launch. Loading the code object happens at the first
    launch (``SpecializedDecode.self_test``, before any capture)."""
    from vllm.models.muse_spark.amd.specialized_decode.mono import layer as k2

    launch = k2.build(key)
    # FlyDSL's compile-only switch has no FLYDSL_ prefix
    old = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"
    try:
        if torch.cuda.is_available():
            launch(*k2.ABI.zeros(), stream=torch.cuda.current_stream())
        else:  # a GPU-less compile: FlyDSL's default stream
            launch(*k2.ABI.zeros())
    finally:
        if old is None:
            del os.environ["COMPILE_ONLY"]
        else:
            os.environ["COMPILE_ONLY"] = old


class KernelLayer:
    """Decoder layer ``layer`` at one launch width: vLLM's boundary and attention
    up to o_proj's input, then K2. ``prev`` is the decoder layer before it, whose
    FFN boundary this layer's input carries."""

    def __init__(self, layer, prev, runner: MonoRunner, width: int):
        self.layer, self.prev, self.runner, self.width = layer, prev, runner, width
        # the step's first kernel layer bumps the epoch (a graph captures it)
        self.first = layer.layer_idx == runner.first_layer
        self.slot = spec.k2_slot(layer.layer_idx)
        # everything but the step's tensors, packed once
        self.static = k2_tensors(layer, runner) | {
            "slot": self.slot,
            **runner.launch_args(),
            **runner.peer_args(),
        }

    def key(self) -> K2Build:
        """The build this layer launches (the fault hook from the runner: tests)."""
        return K2Build(tokens=self.width, **self.runner.fault)

    def builds(self) -> list[K2Build]:
        """The builds this layer launches at its width, compiled before capture."""
        return [self.key()]

    def __call__(
        self,
        step: "Step",
        positions: torch.Tensor,
        x: "MuseSpark1xResidualStream | torch.Tensor",
    ) -> "MuseSpark1xResidualStream":
        from vllm.models.muse_spark.amd.model import norm_residual_then_norm
        from vllm.models.muse_spark.common.layers import MuseSpark1xResidualStream

        layer = self.layer
        if self.first:
            self.runner.step_begin()
        if isinstance(x, torch.Tensor):
            stream = x
            attn_in = layer.self_attn.pre_qkv_proj_norm(
                stream, out_dtype=torch.bfloat16
            )
        else:
            prev = self.prev
            if x.post_norm is not prev.post_ffn_norm or x.connection is not (
                prev.add_ffn
            ):
                raise RuntimeError(
                    f"Muse Spark specialized decode: layer {layer.layer_idx} got "
                    "another layer's residual boundary"
                )
            attn_in, stream = norm_residual_then_norm(
                x, layer.self_attn.pre_qkv_proj_norm, out_dtype=torch.bfloat16
            )
        core = layer.self_attn(
            positions=positions, hidden_states=attn_in, skip_o_proj=True
        )
        s = self.width
        # what the stock callables give, and K2's buffer reads assume
        assert core.is_contiguous() and core.dtype == torch.bfloat16, core.dtype
        assert tuple(core.shape) == (s, spec.CORE), tuple(core.shape)
        assert stream.is_contiguous() and stream.dtype == torch.float32, stream.dtype
        assert tuple(stream.shape) == (s, spec.HIDDEN), tuple(stream.shape)
        stream_out = torch.empty_like(stream)
        branch_out = torch.empty(
            s, spec.HIDDEN, dtype=torch.bfloat16, device=stream.device
        )
        k2_launch(
            self.key(),
            {
                **self.static,
                "core": core,
                "stream": stream,
                "pad": step.pad,
                "stream_out": stream_out,
                "branch_out": branch_out,
            },
        )
        return MuseSpark1xResidualStream(
            stream=stream_out,
            branch=branch_out,
            post_norm=layer.post_ffn_norm,
            connection=layer.add_ffn,
        )
