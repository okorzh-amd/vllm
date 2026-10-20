# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark's specialized decode on CPU: the switch, its widths and kernel
layers, its three refusal stages (config, loaded weights, KV caches), plan()
routing, the layer kernel's launch contract, the model hooks, the model state
and the fail-stop vote. The kernel module (``mono.layer``) is stubbed."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("aiter")

from vllm.config import CompilationMode, CUDAGraphMode  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import MoEActivation  # noqa: E402
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (  # noqa: E402
    Mxfp4MoeBackend,
)
from vllm.models.muse_spark.amd import model as model_mod  # noqa: E402
from vllm.models.muse_spark.amd.model import (  # noqa: E402
    MuseSpark1xForCausalLM,
    MuseSpark1xModel,
    MuseSpark1xMoE,
)
from vllm.models.muse_spark.amd.muse_moe_runner import (  # noqa: E402
    MuseSparkQuarkOCPMXFusedMoE,
)
from vllm.models.muse_spark.amd.specialized_decode import runtime  # noqa: E402
from vllm.models.muse_spark.amd.specialized_decode.mono import spec  # noqa: E402
from vllm.models.muse_spark.amd.specialized_decode.runtime import (  # noqa: E402
    DecodeConfig,
    specialized_decode_config,
)
from vllm.models.muse_spark.amd.specialized_decode.runtime import (  # noqa: E402
    decode as decode_mod,
)
from vllm.models.muse_spark.amd.specialized_decode.runtime import (  # noqa: E402
    layers as layers_mod,
)
from vllm.models.muse_spark.amd.specialized_decode.runtime import (  # noqa: E402
    runner as runner_mod,
)
from vllm.models.muse_spark.amd.specialized_decode.runtime.decode import (  # noqa: E402
    SpecializedDecode,
    Step,
    step_reason,
)
from vllm.models.muse_spark.amd.specialized_decode.runtime.guards import (  # noqa: E402
    FULL,
    SLIDING,
    FailStop,
    kernel_layers,
    refuse_caches,
    refuse_model,
)
from vllm.models.muse_spark.amd.specialized_decode.runtime.layers import (  # noqa: E402
    STEP_ARGS,
    KernelLayer,
)
from vllm.models.muse_spark.common.layers import (  # noqa: E402
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xResidualStream,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum  # noqa: E402
from vllm.v1.attention.backends.rocm_aiter_fa import (  # noqa: E402
    AiterFlashAttentionImpl,
)

from .conftest import dspark, quark_config, vllm_config  # noqa: E402

BF, F32, U8 = torch.bfloat16, torch.float32, torch.uint8
ALL_WIDTHS = (1, 2, 3, 4, 5, 6, 7, 8)


# --------------------------------------------------------------------------
# Stage 1: the switch and the config


def test_auto_runs_the_stock_decode(monkeypatch):
    """Without the switch nothing is checked, even on an unsupported deployment."""

    def fail(_):
        raise AssertionError("checked")

    monkeypatch.setattr(runtime, "refuse_config", fail)
    monkeypatch.setattr(runtime, "unsupported", fail)
    cfg = vllm_config(
        additional_config={}, **{"parallel_config.tensor_parallel_size": 4}
    )
    assert specialized_decode_config(cfg) is None
    cfg.additional_config = {"decode_backend": "auto"}
    assert specialized_decode_config(cfg) is None
    cfg.additional_config = None
    assert runtime.decode_backend(cfg) == "auto"
    assert not runtime.requested(cfg)
    assert specialized_decode_config(cfg) is None


def test_monokernel_flag_enables(on_mi355x):
    cfg = vllm_config(additional_config={}, **{"kernel_config.monokernel": True})
    assert runtime.requested(cfg)
    assert specialized_decode_config(cfg) == DecodeConfig(q=1, widths=ALL_WIDTHS)


@pytest.mark.parametrize("monokernel", [False, True])
def test_an_unknown_backend_is_refused(monokernel):
    cfg = vllm_config(
        additional_config={"decode_backend": "fast"},
        **{"kernel_config.monokernel": monokernel},
    )
    with pytest.raises(ValueError, match="decode_backend must be"):
        specialized_decode_config(cfg)


def test_supported_configs(on_mi355x, monkeypatch):
    assert specialized_decode_config(vllm_config()) == DecodeConfig(1, ALL_WIDTHS)
    # widths the scheduler can never batch are not built
    cfg = vllm_config(**{"scheduler_config.max_num_seqs": 2})
    assert specialized_decode_config(cfg) == DecodeConfig(1, (1, 2))
    # DSpark, K = 7: one request a launch
    cfg = vllm_config(speculative_config=dspark())
    assert specialized_decode_config(cfg) == DecodeConfig(8, (8,))
    cfg = vllm_config(
        speculative_config=dspark(), **{"scheduler_config.max_num_seqs": 1}
    )
    assert specialized_decode_config(cfg) == DecodeConfig(8, (8,))
    # breakable CUDA graphs are not a refusal
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    assert specialized_decode_config(vllm_config()) is not None
    # FULL_DECODE_ONLY too
    cfg = vllm_config(
        **{"compilation_config.cudagraph_mode": CUDAGraphMode.FULL_DECODE_ONLY}
    )
    assert specialized_decode_config(cfg) is not None
    with pytest.raises(ValueError, match="num_speculative_tokens 3"):
        specialized_decode_config(
            vllm_config(speculative_config=dspark(num_speculative_tokens=3))
        )


def _text(**over):
    from .conftest import muse_text_config

    return muse_text_config(**over)


_KINDS = _text().layer_types
_REFUSALS = [
    # 1-9 parallelism
    ({"parallel_config.tensor_parallel_size": 4}, {}, "TP 4 (TP8 only)"),
    ({"parallel_config.pipeline_parallel_size": 2}, {}, "pipeline parallelism"),
    ({"parallel_config.data_parallel_size": 2}, {}, "data parallelism"),
    ({"parallel_config.enable_expert_parallel": True}, {}, "expert parallelism"),
    ({"parallel_config.enable_eplb": True}, {}, "EPLB"),
    ({"parallel_config.decode_context_parallel_size": 8}, {}, "decode context"),
    ({"parallel_config.prefill_context_parallel_size": 2}, {}, "prefill context"),
    ({"parallel_config.use_ubatching": True}, {}, "DBO"),
    ({"parallel_config.use_sequence_parallel_moe": True}, {}, "sequence-parallel"),
    # 10-16 features
    ({"lora_config": object()}, {}, "LoRA"),
    ({"kv_transfer_config": object()}, {}, "KV transfer / KV offloading"),
    (
        {"aux_output_config.enable_return_routed_experts": True},
        {},
        "routed-expert capture",
    ),
    ({"model_config.enable_sleep_mode": True}, {}, "sleep mode"),
    ({"offload_config.uva.cpu_offload_gb": 4}, {}, "CPU weight offload"),
    ({"offload_config.prefetch.offload_group_size": 2}, {}, "CPU weight offload"),
    ({"weight_transfer_config": object()}, {}, "weight updates"),
    ({"model_config.enable_prompt_embeds": True}, {}, "prompt embeds"),
    # 17-20 speculative decoding
    ({"speculative_config": dspark(method="eagle3")}, {}, "method 'eagle3'"),
    (
        {"speculative_config": dspark(num_speculative_tokens=3)},
        {},
        "num_speculative_tokens 3 (0 or 7 only)",
    ),
    (
        {"speculative_config": dspark(enable_adaptive_verification=True)},
        {},
        "adaptive verification",
    ),
    ({"speculative_config": dspark(dynamic=True)}, {}, "dynamic speculative"),
    # 21-23 runner and graphs
    ({"use_v2_model_runner": False}, {}, "V1 model runner"),
    (
        {"compilation_config.mode": CompilationMode.STOCK_TORCH_COMPILE},
        {},
        "torch.compile",
    ),
    (
        {"compilation_config.cudagraph_mode": CUDAGraphMode.PIECEWISE},
        {},
        "PIECEWISE decode CUDA graphs",
    ),
    # 24-28 dtypes, KV cache, attention
    ({"model_config.dtype": torch.float16}, {}, "model dtype torch.float16"),
    ({"cache_config.cache_dtype": "fp8"}, {}, "KV cache dtype 'fp8'"),
    ({"cache_config.block_size": 32}, {}, "KV block size 32"),
    (
        {"attention_config.backend": AttentionBackendEnum.TRITON_ATTN},
        {},
        "attention backend TRITON_ATTN",
    ),
    ({"attention_config.backend": None}, {}, "attention backend auto"),
    (
        {
            "attention_config.backend_per_kind": {
                "sliding_window": AttentionBackendEnum.TRITON_ATTN
            }
        },
        {},
        "attention backend sliding_window=TRITON_ATTN",
    ),
    ({}, {"shuffle": True}, "the AITER shuffled KV cache layout"),
    # 29-31 MoE env, quantization
    ({}, {"VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE": "fp16"}, "REDUCE_DTYPE='fp16'"),
    ({"quant_config": None}, {}, "quantization 'none'"),
    (
        {"quant_config": SimpleNamespace(get_name=lambda: "fp8")},
        {},
        "quantization 'fp8'",
    ),
    # 32-36 the model
    ({"model_config.hf_text_config.hidden_size": 4096}, {}, "model shapes"),
    ({"model_config.hf_text_config.num_shared_experts": 1}, {}, "model shapes"),
    (
        {"model_config.hf_text_config.gated_delta_net": SimpleNamespace(enable=True)},
        {},
        "gated_delta_net layers",
    ),
    (
        {"model_config.hf_text_config.mlp_layer_types": ["dense"] + ["sparse"] * 61},
        {},
        "dense MLP layers",
    ),
    (
        {"model_config.hf_text_config.layer_types": ["chunked_attention"] + _KINDS[1:]},
        {},
        "layer type 'chunked_attention'",
    ),
    (
        {
            "model_config.hf_text_config.layer_rope_theta": [500000.0] * 62,
        },
        {},
        "RoPE on full-attention layers",
    ),
    # 38-39
    ({}, {"VLLM_MOE_SKIP_PADDING": "0"}, "VLLM_MOE_SKIP_PADDING=0"),
    (
        {"parallel_config.enable_batch_sharded_sampling": True},
        {},
        "batch-sharded sampling",
    ),
]


@pytest.mark.parametrize(
    "over, env, reason",
    _REFUSALS,
    ids=[f"{i}-{r[2][:24]}" for i, r in enumerate(_REFUSALS)],
)
def test_refusals(on_mi355x, monkeypatch, over, env, reason):
    from vllm._aiter_ops import rocm_aiter_ops

    for k, v in env.items():
        if k == "shuffle":
            monkeypatch.setattr(
                rocm_aiter_ops,
                "is_shuffle_kv_cache_enabled",
                classmethod(lambda cls: True),
            )
        else:
            monkeypatch.setenv(k, v)
    reasons = runtime.refuse_config(vllm_config(**over))
    assert len(reasons) == 1 and reason in reasons[0], reasons


def test_unsupported_is_refuse_config():
    assert runtime.unsupported is runtime.refuse_config


def test_platform_reasons_off_rocm(monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    assert runtime.platform_reasons() == ["this platform (ROCm only)"]


def test_platform_reasons_report_a_failing_cu_query(monkeypatch):
    """Any error of the platform probes is a refusal reason, not a crash of the
    model's __init__."""
    import sys
    import types

    from vllm.models.muse_spark.amd.specialized_decode.mono import common
    from vllm.platforms import current_platform, rocm

    def no_device(*args, **kwargs):
        raise AssertionError("no HIP device")

    ops = types.ModuleType(f"{common.__name__}.ops")
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    monkeypatch.setattr(common, "ops", ops, raising=False)
    monkeypatch.setattr(current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(rocm, "on_gfx950", lambda: True)
    monkeypatch.setattr(current_platform, "num_compute_units", no_device)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    assert runtime.platform_reasons() == ["the GPU's CU count (no HIP device)"]
    monkeypatch.setattr(current_platform, "num_compute_units", lambda i: 32)
    assert runtime.platform_reasons() == [
        "32 CUs (the kernels need 256 resident CTAs, one a CU)"
    ]


def test_every_refusal_is_reported_at_once(monkeypatch):
    monkeypatch.setattr(
        runtime, "platform_reasons", lambda: ["this GPU (gfx950 / MI355X only)"]
    )
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    cfg = vllm_config(
        **{
            "parallel_config.tensor_parallel_size": 4,
            "use_v2_model_runner": False,
            "lora_config": object(),
            "compilation_config.cudagraph_mode": CUDAGraphMode.PIECEWISE,
        }
    )
    with pytest.raises(ValueError) as e:
        specialized_decode_config(cfg)
    msg = str(e.value)
    assert msg.startswith("Muse Spark specialized decode (--monokernel")
    for reason in (
        "TP 4",
        "V1 model runner",
        "this GPU",
        "LoRA",
        "PIECEWISE decode CUDA graphs",
        "decode_backend=auto without --monokernel",
    ):
        assert reason in msg


def test_reduce_dtype_bf16_is_accepted(on_mi355x, monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE", "bf16")
    assert specialized_decode_config(vllm_config()) is not None


def test_eager_mode_warns(on_mi355x, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        runtime.logger, "warning_once", lambda msg, *a, **k: warnings.append(msg)
    )
    assert specialized_decode_config(vllm_config()) is not None
    assert not warnings
    for mode in (CUDAGraphMode.NONE, None):
        cfg = vllm_config(**{"compilation_config.cudagraph_mode": mode})
        assert runtime.eager_decode(cfg)
        assert specialized_decode_config(cfg) is not None
    assert len(warnings) == 2 and "eagerly" in warnings[0]


# --------------------------------------------------------------------------
# Stage 2: the loaded model


def _meta(*shape, dtype=BF):
    return torch.empty(*shape, dtype=dtype, device="meta")


def _norm(n: int, eps: float, *, weight=True, post=False, before=False):
    cls = MuseSpark1xRMSNormWithInputScale if before else MuseSpark1xRMSNorm
    norm = cls(
        n,
        eps=eps,
        has_weight=weight,
        is_post_norm=post,
        zero_centered_gamma=True,
        post_norm_gain_center_type="one",
        residual_id=None if post else 3,
    )
    if weight:
        norm.weight = nn.Parameter(torch.zeros(n, dtype=BF), requires_grad=False)
    return norm


def _connection(ready=True):
    conn = MuseSpark1xNormPreservingResidualConnection(spec.HIDDEN, temperature=0.3)
    conn.gate.data.zero_()
    if ready:
        conn.materialize_coefficients()
    return conn


def _moe(backend=Mxfp4MoeBackend.AITER_MXFP4_MXFP4, quantized=True):
    """A ``MuseSpark1xMoE`` with Muse's loaded tensors (meta) and its quant
    method: AITER MXFP4 x MXFP4 by default."""
    moe = MuseSpark1xMoE.__new__(MuseSpark1xMoE)
    nn.Module.__init__(moe)
    if quantized:
        method = MuseSparkQuarkOCPMXFusedMoE.__new__(MuseSparkQuarkOCPMXFusedMoE)
        method.mxfp4_backend = backend
    else:
        method = SimpleNamespace()  # the stock BF16 experts
    w13, w2 = _meta(256, 1024, 2048, dtype=U8), _meta(256, 4096, 256, dtype=U8)
    w13.is_shuffled = w2.is_shuffled = True
    moe.experts = SimpleNamespace(
        routed_experts=SimpleNamespace(
            quant_method=method,
            w13_weight=w13,
            w13_weight_scale=_meta(256, 1024, 128, dtype=U8),
            w2_weight=w2,
            w2_weight_scale=_meta(256, 4096, 16, dtype=U8),
            expert_map=None,
            activation=MoEActivation.SILU,
        )
    )
    moe.router = SimpleNamespace(weight=_meta(256, 8192, dtype=F32))
    # as MuseSpark1xMoE.fold_router_output_multiplier leaves it: the multiplier
    # stays set, the flag records the fold
    moe.router_output_multiplier = 0.1767
    moe._router_multiplier_folded = True
    moe.expert_bias = _meta(256, dtype=F32)
    moe.route_scale = 1.0
    moe.is_sequence_parallel = False
    moe.pooler = SimpleNamespace(weight=_meta(4096, 8192))
    moe.pre_expert_norm = _norm(4096, 1e-5)
    moe.post_expert_norm = _norm(4096, 1e-8, post=True, before=True)
    moe.unpooler = SimpleNamespace(weight=_meta(8192, 4096))
    return moe


class _Attn:
    """``MuseSpark1xAttention`` as the kernel layer sees it."""

    def __init__(self, i: int, kv=None):
        self.o_proj = SimpleNamespace(weight=_meta(8192, 1024))
        self.num_heads, self.head_dim, self.num_kv_heads = 16, 64, 2
        self.fused_qk_norm_rope_kv_insert = True
        self.pre_qkv_proj_norm = _norm(8192, 1e-5)
        self.attn = SimpleNamespace(
            impl=AiterFlashAttentionImpl.__new__(AiterFlashAttentionImpl),
            layer_name=f"model.layers.{i}.self_attn.attn",
            kv_cache=torch.tensor([]) if kv is None else kv,
        )
        self.calls: list = []

    def __call__(self, positions, hidden_states, skip_o_proj=False):
        assert skip_o_proj
        self.calls.append((positions, hidden_states))
        return torch.zeros(hidden_states.size(0), spec.CORE, dtype=BF)


def _layer(i: int, moe=None, kv=None):
    return SimpleNamespace(
        layer_idx=i,
        self_attn=_Attn(i, kv),
        feed_forward=_moe() if moe is None else moe,
        add_attn=_connection(),
        add_ffn=_connection(),
        post_attn_norm=_norm(8192, 1e-8, weight=False, post=True),
        pre_ffn_norm=_norm(8192, 1e-5),
        post_ffn_norm=_norm(8192, 1e-8, post=True, before=True),
    )


def _backbone(n=4, unquantized=(0,), kinds=None):
    kinds = kinds or _KINDS
    layers = [
        _layer(i, _moe(quantized=False) if i in unquantized else None) for i in range(n)
    ]
    return SimpleNamespace(
        layers=layers,
        start_layer=0,
        end_layer=n,
        config=SimpleNamespace(layer_types=kinds[:n]),
    )


def test_kernel_layers_are_the_mxfp4_layers():
    """Detected on the loaded model: the checkpoint keeps the first and the last
    layer's experts in BF16 (Quark ``exclude``)."""
    bb = _backbone(62, unquantized=(0, 61))
    layers = kernel_layers(bb)
    assert list(layers) == list(range(1, 61))
    assert layers[1] == FULL and layers[5] == FULL and layers[57] == FULL
    assert layers[2] == SLIDING and layers[60] == SLIDING
    # another MXFP4 backend, a dense layer, another pipeline stage
    bb.layers[3].feed_forward = _moe(Mxfp4MoeBackend.AITER_MXFP4_BF16)
    bb.layers[4].feed_forward = SimpleNamespace()
    bb.start_layer, bb.end_layer = 2, 31
    assert list(kernel_layers(bb)) == [2] + list(range(5, 31))


def test_refuse_model_accepts_the_checkpoint_layout():
    bb = _backbone()
    layers = kernel_layers(bb)
    assert layers == {1: FULL, 2: SLIDING, 3: SLIDING}
    assert refuse_model(bb, layers) == []
    # the fp4 / e8m0 dtypes of a later aiter are accepted too
    if hasattr(torch, "float4_e2m1fn_x2"):
        X = bb.layers[2].feed_forward.experts.routed_experts
        X.w13_weight = _meta(256, 1024, 2048, dtype=torch.float4_e2m1fn_x2)
        X.w13_weight.is_shuffled = True
        X.w2_weight_scale = _meta(256, 4096, 16, dtype=torch.float8_e8m0fnu)
        assert refuse_model(bb, layers) == []


def _break(bb, what):
    L = bb.layers[2]
    F, A = L.feed_forward, L.self_attn
    X = F.experts.routed_experts
    if what == "multiplier":
        F._router_multiplier_folded = False
    elif what == "coefficients":
        L.add_attn = _connection(ready=False)
    elif what == "o_proj":
        A.o_proj.weight = _meta(8192, 2048)
    elif what == "o_proj strided":
        A.o_proj.weight = _meta(1024, 8192).t()
    elif what == "w13 shuffle":
        X.w13_weight = _meta(256, 1024, 2048, dtype=U8)
    elif what == "insert":
        A.fused_qk_norm_rope_kv_insert = False
    elif what == "impl":
        A.attn.impl = SimpleNamespace()
    elif what == "expert map":
        X.expert_map = torch.arange(256)
    elif what == "router dtype":
        F.router.weight = _meta(256, 8192)
    elif what == "missing pooler":
        del F.pooler
    elif what == "pre_ffn eps":
        L.pre_ffn_norm.variance_epsilon = 1e-6
    elif what == "post_expert side":
        F.post_expert_norm = _norm(4096, 1e-8, post=True, before=False)
    elif what == "post_attn weight":
        L.post_attn_norm = _norm(8192, 1e-8, post=True)
    elif what == "activation":
        X.activation = MoEActivation.GELU
    elif what == "route scale":
        F.route_scale = 2.5
    elif what == "heads":
        A.num_kv_heads = 1
    elif what == "pre_expert variance size":
        F.pre_expert_norm.variance_size_override = 2048
    elif what == "post_attn variance size":
        L.post_attn_norm.variance_size_override = 4096
    elif what == "prev add_ffn":
        bb.layers[1].add_ffn = _connection(ready=False)
    else:
        raise AssertionError(what)


@pytest.mark.parametrize(
    "what, reason",
    [
        ("multiplier", "layer 2: router output multiplier not folded"),
        ("coefficients", "layer 2: add_attn coefficients not materialized"),
        (
            "o_proj",
            "layer 2 o_proj: (8192, 2048) torch.bfloat16, kernel reads contiguous "
            "(8192, 1024) torch.bfloat16",
        ),
        ("o_proj strided", "layer 2 o_proj: (8192, 1024) torch.bfloat16 strided"),
        ("w13 shuffle", "layer 2: w13 not AITER-shuffled"),
        ("insert", "layer 2: the fused QK-norm/RoPE/KV insert is off"),
        ("impl", "layer 2: attention impl SimpleNamespace (ROCM_AITER_FA only)"),
        ("expert map", "layer 2: an expert map (EP)"),
        ("router dtype", "layer 2 router: (256, 8192) torch.bfloat16"),
        ("missing pooler", "layer 2 pooler: missing"),
        ("pre_ffn eps", "layer 2: pre_ffn_norm (gain 1, eps 1e-5, weight after)"),
        (
            "post_expert side",
            "layer 2: post_expert_norm (gain 1, eps 1e-8, weight first)",
        ),
        ("post_attn weight", "layer 2: post_attn_norm (weightless, eps 1e-8)"),
        ("activation", "layer 2: expert activation"),
        ("route scale", "layer 2: route_scale 2.5"),
        ("heads", "layer 2: attention heads"),
        (
            "pre_expert variance size",
            "layer 2: pre_expert_norm (gain 1, eps 1e-5, weight after)",
        ),
        ("post_attn variance size", "layer 2: post_attn_norm (weightless, eps 1e-8)"),
        ("prev add_ffn", "layer 1: add_ffn coefficients not materialized"),
    ],
)
def test_refuse_model_names_every_mismatch(what, reason):
    bb = _backbone()
    layers = kernel_layers(bb)
    _break(bb, what)
    reasons = refuse_model(bb, layers)
    assert len(reasons) == 1 and reasons[0].startswith(reason), reasons


def test_refuse_model_reports_every_reason_and_layer_zero():
    bb = _backbone()
    layers = kernel_layers(bb)
    for what in ("multiplier", "insert", "impl"):
        _break(bb, what)
    bb.layers[3].feed_forward.route_scale = 2.0
    assert len(refuse_model(bb, layers)) == 4
    assert refuse_model(bb, {}) == ["no layer has AITER_MXFP4_MXFP4 routed experts"]
    reasons = refuse_model(_backbone(unquantized=()), {0: SLIDING})
    assert reasons == ["layer 0 as a kernel layer (its input is the embedding stream)"]


def test_refuse_model_accepts_the_models_own_router_fold(monkeypatch):
    """``MuseSpark1xMoE.fold_router_output_multiplier`` (what a real load runs)
    folds the multiplier into the router weight but keeps it set and records the
    fold in ``_router_multiplier_folded``: stage 2 accepts that model, and
    refuses it until the fold ran."""
    monkeypatch.setattr(model_mod, "_supports_router_bf16x3_device", lambda: False)
    bb = _backbone()
    layers = kernel_layers(bb)
    for L in bb.layers:
        F = L.feed_forward
        F.router_output_multiplier, F._router_multiplier_folded = 0.1767, False
        F.moe_router = SimpleNamespace(router_output_multiplier=0.1767)
    assert "layer 2: router output multiplier not folded" in refuse_model(bb, layers)
    for L in bb.layers:
        L.feed_forward.fold_router_output_multiplier()
        assert L.feed_forward.router_output_multiplier == 0.1767
    assert refuse_model(bb, layers) == []
    # a router without a multiplier has nothing to fold
    for L in bb.layers:
        F = L.feed_forward
        F.router_output_multiplier, F._router_multiplier_folded = None, False
    assert refuse_model(bb, layers) == []


def test_refuse_model_on_an_unmaterialized_model_names_it():
    """A dummy-style load that skipped materialization is a reason, never an
    AttributeError (the coefficient buffers do not exist yet)."""
    bb = _backbone()
    layers = kernel_layers(bb)
    for L in bb.layers:
        L.add_attn, L.add_ffn = _connection(ready=False), _connection(ready=False)
        L.feed_forward._router_multiplier_folded = False
    reasons = refuse_model(bb, layers)
    assert "layer 1: add_attn coefficients not materialized" in reasons
    assert "layer 0: add_ffn coefficients not materialized" in reasons
    assert "layer 3: router output multiplier not folded" in reasons


# --------------------------------------------------------------------------
# Stage 3: the KV caches


def _cfg_mode(mode):
    return SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=mode))


def _kv(blocks=4, block=16, dtype=BF, width=128):
    return _meta(blocks, 2, block, width, dtype=dtype)


def test_refuse_caches_accepts_real_layouts():
    bb = _backbone()
    layers = kernel_layers(bb)
    # one pool shared by every layer of a KV group (aliasing), ~11 GB a view
    pool = _kv(blocks=(11 << 30) // (2 * 16 * 128 * 2))
    for i in layers:
        bb.layers[i].self_attn.attn.kv_cache = pool
    assert refuse_caches(bb, layers, _cfg_mode(CUDAGraphMode.FULL_DECODE_ONLY)) == []
    assert refuse_caches(bb, layers, _cfg_mode(CUDAGraphMode.NONE)) == []


@pytest.mark.parametrize(
    "kv, mode, reason",
    [
        (_kv(dtype=torch.float8_e4m3fn), None, "layer 2: KV cache torch.float8_e4m3fn"),
        (_kv(dtype=torch.float16), None, "layer 2: KV cache torch.float16 (bf16"),
        (_meta(4, 1, 16, 128), None, "layer 2: KV cache shape (4, 1, 16, 128)"),
        (_meta(4, 2, 16, 64), None, "layer 2: KV cache shape (4, 2, 16, 64)"),
        (_kv(block=32), None, "layer 2: KV kernel block 32 (--block-size 16 only)"),
        (_meta(4, 2, 16, 256)[..., ::2], None, "layer 2: KV cache not dense"),
        (None, CUDAGraphMode.PIECEWISE, "PIECEWISE decode CUDA graphs"),
    ],
)
def test_refuse_caches(kv, mode, reason):
    bb = _backbone()
    layers = kernel_layers(bb)
    for i in layers:
        bb.layers[i].self_attn.attn.kv_cache = _kv()
    if kv is not None:
        bb.layers[2].self_attn.attn.kv_cache = kv
    reasons = refuse_caches(
        bb, layers, _cfg_mode(mode or CUDAGraphMode.FULL_AND_PIECEWISE)
    )
    assert len(reasons) == 1 and reasons[0].startswith(reason), reasons


# --------------------------------------------------------------------------
# plan(): which steps run the kernels


def _md(width: int, q: int, **over):
    m = SimpleNamespace(
        decode_metadata=SimpleNamespace(max_query_len=q),
        num_prefills=0,
        num_extends=0,
        num_decode_tokens=width,
        num_decodes=width // q,
    )
    for k, v in over.items():
        setattr(m, k, v)
    return m


NAMES = {FULL: "full.attn", SLIDING: "sliding.attn"}


def _metadata(width: int, q: int, full=None, **over):
    """Both KV kinds' metadata of a step: identical unless ``full`` is given."""
    return {
        "sliding.attn": _md(width, q, **over),
        "full.attn": _md(width, q, **(full or over)),
        "other": object(),
    }


@pytest.mark.parametrize(
    "width, q, md, why",
    [
        (1, 1, _metadata(1, 1), None),
        (8, 1, _metadata(8, 1), None),
        (8, 8, _metadata(8, 8), None),  # one DSpark verify of 8 rows
        (4, 1, _metadata(4, 1, num_prefills=1), "not pure decode"),
        (4, 1, _metadata(4, 1, num_extends=1), "not pure decode"),
        (4, 1, _metadata(4, 1, full={"num_prefills": 1}), "not pure decode"),
        (
            8,
            8,
            _metadata(8, 8, decode_metadata=SimpleNamespace(max_query_len=4)),
            "not uniform",
        ),  # fmt: skip
        (4, 1, _metadata(4, 1, num_decode_tokens=3), "not uniform / padded width"),
        (4, 1, _metadata(4, 1, num_decodes=3), "not uniform / padded width"),
        (4, 1, _metadata(4, 1, decode_metadata=None), "no decode metadata"),
        (4, 1, {"sliding.attn": _md(4, 1)}, "no decode metadata"),
        (4, 1, None, "no metadata"),
        (4, 1, [_metadata(2, 1), _metadata(2, 1)], "no metadata"),  # DBO
    ],
)
def test_step_reason(width, q, md, why):
    assert step_reason(md, NAMES, width, q) == why


class _Kernel:
    def __call__(self, step, positions, x):
        return "kernel", step.width, x


def _decode(monkeypatch, mode=CUDAGraphMode.NONE, md=None, q=1, is_padding=True):
    """An executor with kernels on layers 1 and 2 at its widths, bound to a KV
    cache, under a forward context of ``mode`` (MRV2 captures FULL graphs and
    warms up with NONE)."""
    decode = object.__new__(SpecializedDecode)
    widths = (8,) if q == 8 else (1, 2, 4, 8)
    decode.config = DecodeConfig(q=q, widths=widths)
    decode.names = NAMES
    decode.runner = SimpleNamespace(no_pad=torch.zeros(16, dtype=U8))
    decode.kernels = {w: {1: _Kernel(), 2: _Kernel()} for w in widths}
    decode._kv = (1, 2)
    monkeypatch.setattr(SpecializedDecode, "_kv_key", lambda self: (1, 2))
    context = SimpleNamespace(
        cudagraph_runtime_mode=mode,
        attn_metadata=_metadata(4, q) if md is None else md,
        slot_mapping={"full.attn": torch.zeros(4, dtype=torch.int64)},
        is_padding=torch.ones(8, dtype=torch.bool) if is_padding else None,
    )
    monkeypatch.setattr(decode_mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(decode_mod, "get_forward_context", lambda: context)
    return decode, context


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
def test_plan_runs_the_kernels_of_its_width(monkeypatch, mode):
    decode, ctx = _decode(monkeypatch, mode)
    step = decode.plan(torch.zeros(4), None)
    assert step is not None and step.width == 4
    assert step.kernels is decode.kernels[4] and step.attn_metadata is ctx.attn_metadata
    assert step.pad is ctx.is_padding
    assert step.layer(1)(None, "x") == ("kernel", 4, "x")
    assert step.layer(0) is None and step.layer(3) is None
    # M-RoPE positions [3, T]: the width is the last dim
    assert decode.plan(torch.zeros(3, 4), None) is not None


def test_plan_runs_a_dspark_verify_step(monkeypatch):
    decode, _ = _decode(monkeypatch, md=_metadata(8, 8), q=8)
    step = decode.plan(torch.zeros(8), None)
    assert step is not None and step.width == 8
    decode, _ = _decode(monkeypatch, md=_metadata(4, 4), q=8)
    assert decode.plan(torch.zeros(4), None) is None


def test_plan_runs_a_piecewise_warmup_with_all_pad_rows(monkeypatch):
    """MRV2 warms a PIECEWISE descriptor up with cudagraph_runtime_mode NONE,
    uniform-decode metadata and is_padding all True: the kernels run with no
    live row (nu == 0) on every rank."""
    decode, ctx = _decode(monkeypatch, CUDAGraphMode.NONE)
    step = decode.plan(torch.zeros(4), None)
    assert step is not None and bool(step.pad[:4].all())


@pytest.mark.parametrize(
    "mode, tokens, md, intermediate",
    [
        (CUDAGraphMode.PIECEWISE, 4, None, None),
        (CUDAGraphMode.NONE, 16, _metadata(16, 1), None),  # no kernels that wide
        (CUDAGraphMode.NONE, 3, _metadata(3, 1), None),  # no width 3 built here
        (CUDAGraphMode.NONE, 4, _metadata(4, 1, num_prefills=1), None),
        (CUDAGraphMode.FULL, 4, _metadata(4, 1, num_extends=1), None),
        (CUDAGraphMode.NONE, 4, None, object()),  # a pipeline stage's input
        (CUDAGraphMode.NONE, 4, {}, None),  # a profile / dummy run
    ],
)
def test_plan_runs_other_steps_stock(monkeypatch, mode, tokens, md, intermediate):
    decode, _ = _decode(monkeypatch, mode, md)
    assert decode.plan(torch.zeros(tokens), intermediate) is None


def test_plan_runs_stock_without_a_forward_context(monkeypatch):
    decode, _ = _decode(monkeypatch)
    monkeypatch.setattr(decode_mod, "is_forward_context_available", lambda: False)
    assert decode.plan(torch.zeros(4), None) is None


def test_plan_runs_stock_before_a_kv_cache_is_bound(monkeypatch):
    decode, _ = _decode(monkeypatch)
    decode._kv = None
    assert decode.plan(torch.zeros(4), None) is None


def test_plan_refuses_a_kv_cache_bound_behind_its_back(monkeypatch):
    decode, _ = _decode(monkeypatch)
    monkeypatch.setattr(SpecializedDecode, "_kv_key", lambda self: (3, 4))
    with pytest.raises(RuntimeError, match="without prepare_kv"):
        decode.plan(torch.zeros(4), None)


def test_plan_uses_runner_no_pad_without_is_padding(monkeypatch):
    decode, _ = _decode(monkeypatch, is_padding=False)
    step = decode.plan(torch.zeros(4), None)
    assert step.pad is decode.runner.no_pad
    # ops.ld_pad reads whole words: >= MAX_TOKENS bytes, a 4-byte multiple
    assert runner_mod.NO_PAD_BYTES >= spec.MAX_TOKENS
    assert runner_mod.NO_PAD_BYTES % 4 == 0


def test_plan_carries_slot_mapping(monkeypatch):
    decode, ctx = _decode(monkeypatch)
    assert decode.plan(torch.zeros(4), None).slot_mapping is ctx.slot_mapping


def test_kv_key_tracks_every_kv_kind():
    bb = _backbone()
    decode = object.__new__(SpecializedDecode)
    decode.backbone = bb
    decode.first = {FULL: 1, SLIDING: 2}
    assert decode._kv_key() is None  # unbound (empty) caches
    full, sliding = torch.zeros(2, 2, 16, 128, dtype=BF), torch.zeros(3, 2, 16, 128)
    bb.layers[1].self_attn.attn.kv_cache = full
    assert decode._kv_key() is None  # the sliding kind still unbound
    bb.layers[2].self_attn.attn.kv_cache = sliding
    key = decode._kv_key()
    assert key == (
        (full.data_ptr(), (2, 2, 16, 128), full.stride()),
        (sliding.data_ptr(), (3, 2, 16, 128), sliding.stride()),
    )


# --------------------------------------------------------------------------
# The layer kernel and the layer loop


class _Runner:
    """A ``MonoRunner`` as the layer kernel sees it, recording the step bumps."""

    def __init__(self, calls, first_layer=1):
        self.calls, self.first_layer, self.fault = calls, first_layer, {}
        self.scratch = torch.zeros(8, dtype=U8)
        self.epoch = torch.zeros(1, dtype=torch.int32)
        self.health = torch.zeros(8, dtype=torch.int32)
        self.peer_table = torch.zeros(8, dtype=torch.int64)
        self.alarm, self.rank = 1234, 3

    def step_begin(self):
        self.calls.append("step_begin")
        self.epoch.add_(1)

    launch_args = runner_mod.MonoRunner.launch_args
    peer_args = runner_mod.MonoRunner.peer_args


def _kernel_loop(monkeypatch, width=2):
    calls: list = []
    bb = _backbone()
    runner = _Runner(calls)
    kernels = {
        i: KernelLayer(bb.layers[i], bb.layers[i - 1], runner, width) for i in (1, 2, 3)
    }

    def launch(key, args):
        calls.append(("launch", key, args))

    def boundary(x, pre_norm, *, out_dtype):
        assert out_dtype == BF
        return torch.zeros(width, spec.HIDDEN, dtype=BF), x.stream

    monkeypatch.setattr(layers_mod, "k2_launch", launch)
    monkeypatch.setattr(model_mod, "norm_residual_then_norm", boundary)
    pad = torch.zeros(16, dtype=U8)
    step = Step(width, {}, {}, kernels, pad)
    return bb, runner, step, calls


def _stream(layer, width=2):
    return MuseSpark1xResidualStream(
        stream=torch.zeros(width, spec.HIDDEN),
        branch=torch.zeros(width, spec.HIDDEN, dtype=BF),
        post_norm=layer.post_ffn_norm,
        connection=layer.add_ffn,
    )


def test_step_begin_precedes_every_launch(monkeypatch):
    bb, runner, step, calls = _kernel_loop(monkeypatch)
    positions = torch.zeros(2, dtype=torch.int64)
    for _ in range(2):
        x = _stream(bb.layers[0])
        for i in (1, 2, 3):
            x = step.layer(i)(positions, x)
            assert x.post_norm is bb.layers[i].post_ffn_norm
            assert x.connection is bb.layers[i].add_ffn
            assert x.stream.dtype == F32 and x.branch.dtype == BF
            assert tuple(x.branch.shape) == (2, spec.HIDDEN)
    kinds = [c if isinstance(c, str) else c[0] for c in calls]
    assert kinds == ["step_begin", "launch", "launch", "launch"] * 2
    assert int(runner.epoch[0]) == 2
    for n, (_, key, args) in enumerate(c for c in calls if c != "step_begin"):
        layer = bb.layers[1 + n % 3]
        assert key == spec.K2Build(tokens=2)
        assert set(args) == set(spec.K2_ARGS)  # every argument, nothing else
        assert args["slot"] == spec.k2_slot(layer.layer_idx)
        assert args["pad"] is step.pad and args["rank"] == 3
        assert args["w_o"] is layer.self_attn.o_proj.weight
        assert args["w13"] is layer.feed_forward.experts.routed_experts.w13_weight
        assert args["attn_alpha"] is layer.add_attn.alpha
        assert args["peers"] is runner.peer_table and args["alarm"] == 1234
        assert args["stream_out"].dtype == F32
    # the static arguments are packed once, the step's added at each launch
    k = step.kernels[1]
    assert set(k.static) | set(STEP_ARGS) == set(spec.K2_ARGS)
    assert not set(k.static) & set(STEP_ARGS)


def test_kernel_layer_rejects_a_foreign_boundary(monkeypatch):
    bb, _, step, _ = _kernel_loop(monkeypatch)
    x = _stream(bb.layers[1])  # layer 1's boundary, handed to layer 3
    with pytest.raises(RuntimeError, match="another layer's residual boundary"):
        step.layer(3)(torch.zeros(2), x)
    x = MuseSpark1xResidualStream(
        stream=x.stream,
        branch=x.branch,
        post_norm=bb.layers[2].post_ffn_norm,
        connection=bb.layers[1].add_ffn,
    )
    with pytest.raises(RuntimeError, match="another layer's residual boundary"):
        step.layer(3)(torch.zeros(2), x)


def test_kernel_layer_fault_builds_follow_the_runner(monkeypatch):
    bb, runner, step, calls = _kernel_loop(monkeypatch)
    fault = {"fault_rank": 3, "fault_epoch": 2, "fault_slot": 129, "fault_ticks": 5}
    runner.fault = fault
    step.layer(1)(torch.zeros(2), _stream(bb.layers[0]))
    assert calls[-1][1] == spec.K2Build(tokens=2, **fault)
    assert step.kernels[1].builds() == [spec.K2Build(tokens=2, **fault)]


def test_layer_loop_uses_step_kernels(monkeypatch):
    """The stock layer loop runs layer 0 stock and layer 1 through the step's
    kernel; the final norm and the EAGLE3 aux capture see the kernel's stream."""
    calls: list = []
    stock_out = _stream(_layer(0))
    kernel_out = _stream(_layer(1))

    def stock(positions, x):
        calls.append(("stock", positions, x))
        return stock_out

    def kernel(step, positions, x):
        calls.append(("kernel", positions, x))
        return kernel_out

    bb = object.__new__(MuseSpark1xModel)
    nn.Module.__init__(bb)
    bb.layers = [stock, SimpleNamespace()]
    bb.start_layer, bb.end_layer = 0, 2
    bb.residual_dtype, bb.default_dtype = F32, BF
    bb.fuse_across_layers = True
    bb.aux_hidden_state_layers = (2,)
    bb.norm = "final norm"
    monkeypatch.setattr(
        model_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(model_mod, "norm_then_gated_residual", lambda x: ("aux", x))
    monkeypatch.setattr(
        model_mod,
        "norm_residual_then_norm",
        lambda x, norm, out_dtype: (("final", x, norm), None),
    )
    positions = torch.arange(2)
    embeds = torch.ones(2, spec.HIDDEN, dtype=BF)
    step = Step(2, {}, {}, {1: kernel}, torch.zeros(16, dtype=U8))
    out, aux = bb.forward(None, positions, None, inputs_embeds=embeds, step=step)
    assert [c[0] for c in calls] == ["stock", "kernel"]
    assert calls[0][2].dtype == F32 and calls[1][2] is stock_out
    assert calls[1][1] is positions
    assert out == ("final", kernel_out, "final norm")
    assert aux == [("aux", kernel_out)]
    # no step: the stock loop (layer 1 is not callable here)
    calls.clear()
    with pytest.raises(TypeError):
        bb.forward(None, positions, None, inputs_embeds=embeds)


# --------------------------------------------------------------------------
# The model hooks and the model state


def _causal_lm(config=None, decode=None):
    m = object.__new__(MuseSpark1xForCausalLM)
    nn.Module.__init__(m)
    m.init_specialized_decode(config)
    m.specialized_decode = decode
    return m


class _Decode:
    def __init__(self, step="step"):
        self.calls: list = []
        self.step = step

    def plan(self, positions, intermediate_tensors):
        self.calls.append(("plan", positions, intermediate_tensors))
        return self.step

    def after_step(self):
        self.calls.append("after_step")

    def prepare_kv(self):
        self.calls.append("prepare_kv")

    def after_capture(self):
        self.calls.append("after_capture")


class _Folded(nn.Module):
    """A MoE's router as ``materialize_residual_path_constants`` sees it: the
    multiplier stays set and ``_router_multiplier_folded`` records the fold."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2), requires_grad=False)
        self.router_output_multiplier = 0.5
        self._router_multiplier_folded = False
        self.folds = 0

    def fold_router_output_multiplier(self):
        self.folds += 1
        if not self._router_multiplier_folded:
            self.weight.mul_(self.router_output_multiplier)
            self._router_multiplier_folded = True


def test_model_hooks():
    config = DecodeConfig(1, ALL_WIDTHS)
    decode = _Decode()
    m = _causal_lm(config, decode)
    seen = []

    def model(*args, **kwargs):
        seen.append((args, kwargs))
        return "hidden"

    m.model = model
    pos = torch.zeros(4)
    assert m.forward("ids", pos) == "hidden"
    assert seen == [(("ids", pos, None, None), {"step": "step"})]
    assert decode.calls == [("plan", pos, None)]

    # compute_logits polls the fail-stop once a call
    m.lm_head = SimpleNamespace(weight=torch.zeros(1, dtype=BF))
    m.logits_processor = lambda head, hidden: None
    assert m.compute_logits(torch.zeros(1, 4)) is None
    assert decode.calls[1:] == ["after_step"]

    # once the executor exists the weights may not move
    with pytest.raises(RuntimeError, match="reloading or updating weights"):
        m.process_weights_after_loading()

    # mono off: plan is None, nothing polled
    off = _causal_lm()
    off.model = lambda *args, **kwargs: kwargs["step"]
    assert off.forward("ids", pos) is None
    off.after_specialized_decode_step()
    # a model built without __init__ (tests of other features) has no hooks
    bare = object.__new__(MuseSpark1xForCausalLM)
    assert bare.plan_specialized_decode(pos, None) is None
    bare.after_specialized_decode_step()
    bare.refuse_weight_reload()


@pytest.mark.parametrize("on", [True, False])
def test_dummy_loads_materialize_only_with_the_decode_on(on):
    """--load-format dummy never calls load_weights; with the decode on
    process_weights_after_loading materializes the residual constants and folds
    the router multiplier exactly once, with it off it changes nothing."""
    m = _causal_lm(DecodeConfig(1, (1,)) if on else None)
    m.conn, m.router = _connection(ready=False), _Folded()
    for _ in range(2):
        m.process_weights_after_loading()
    ready = getattr(m.conn, "_coefficients_ready", False)
    assert ready is on
    assert m.router.folds == (1 if on else 0)
    assert float(m.router.weight[0, 0]) == (0.5 if on else 1.0)
    # a real load materialized everything: nothing is redone
    m = _causal_lm(DecodeConfig(1, (1,)))
    m.conn, m.router = _connection(ready=True), _Folded()
    m.router._router_multiplier_folded = True
    m.process_weights_after_loading()
    assert m.router.folds == 0


def test_model_state_lifecycle(monkeypatch):
    from vllm.models.muse_spark.amd.model_state import MuseSparkModelState
    from vllm.models.muse_spark.amd.multimodal import (
        MuseSparkForConditionalGeneration,
    )
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    for config in (DecodeConfig(1, (1,)), None):
        assert _causal_lm(config).get_model_state_cls() is MuseSparkModelState
    assert issubclass(MuseSparkModelState, DefaultModelState)

    monkeypatch.setattr(DefaultModelState, "__init__", lambda self, *a, **k: None)
    built = []
    lm = _causal_lm(DecodeConfig(1, (1,)))

    def build(vllm_config=None):
        built.append(vllm_config)
        lm.specialized_decode = decode

    decode = _Decode()
    lm.build_specialized_decode = build
    # the multimodal model hands its language model over
    omni = SimpleNamespace(get_language_model=lambda: lm)
    state = MuseSparkModelState("cfg", omni, None, "cpu")
    assert built == ["cfg"] and state.language_model is lm
    state.initialize_kv_cache(None, None)
    state.capture_inner_cudagraphs(None, None, None, None)
    assert decode.calls == ["prepare_kv", "after_capture"]

    # mono off: built never, the hooks are no-ops
    off = _causal_lm()
    off.build_specialized_decode = lambda *a: built.append("off")
    state = MuseSparkModelState("cfg", off, None, "cpu")
    state.initialize_kv_cache(None, None)
    state.capture_inner_cudagraphs(None, None, None, None)
    assert built == ["cfg"]

    # the multimodal model inherits every hook
    for name in (
        "forward",
        "compute_logits",
        "process_weights_after_loading",
        "get_model_state_cls",
        "init_specialized_decode",
        "build_specialized_decode",
        "plan_specialized_decode",
        "after_specialized_decode_step",
        "refuse_weight_reload",
        "materialize_for_specialized_decode",
    ):
        assert getattr(MuseSparkForConditionalGeneration, name) is getattr(
            MuseSpark1xForCausalLM, name
        ), name


def test_attention_returns_the_core_with_skip_o_proj():
    import inspect

    from vllm.models.muse_spark.amd.model import MuseSpark1xAttention

    params = inspect.signature(MuseSpark1xAttention.forward).parameters
    assert params["skip_o_proj"].default is False
    params = inspect.signature(MuseSpark1xModel.forward).parameters
    assert params["step"].default is None


# --------------------------------------------------------------------------
# Fail-stop, the runner, the epoch


class _Event:
    def record(self):
        pass

    def query(self):
        return True


def _failstop(monkeypatch, others_failed=lambda step: False):
    real_zeros = torch.zeros
    monkeypatch.setattr(
        torch, "zeros", lambda *a, pin_memory=False, **k: real_zeros(*a, **k)
    )
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    capturing = {"on": False}
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: capturing["on"]
    )
    votes = []

    def all_reduce(t, group=None):
        votes.append(t.clone())
        if others_failed(len(votes)):
            t[5] = 1

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    failed = real_zeros(1, dtype=torch.int32)
    return FailStop(failed, rank=2, world=8, group="gloo"), failed, capturing, votes


def test_failstop_votes_and_raises(monkeypatch):
    stop, failed, capturing, votes = _failstop(monkeypatch)
    for _ in range(FailStop.CHECK_EVERY):
        stop.after_step()
    assert len(votes) == 1 and not votes[0].any()
    # a capture runs compute_logits without a step: not counted
    capturing["on"] = True
    for _ in range(3 * FailStop.CHECK_EVERY):
        stop.after_step()
    capturing["on"] = False
    failed[0] = 1  # this rank gave up a wait
    for _ in range(FailStop.CHECK_EVERY - 1):
        stop.after_step()
    with pytest.raises(RuntimeError, match=r"TP ranks \[2\] gave up"):
        stop.after_step()
    assert len(votes) == 2 and int(votes[1][2]) == 1


def test_failstop_raises_on_every_rank_together(monkeypatch):
    """A healthy rank raises at the same step as the rank that failed."""
    stop, _, _, votes = _failstop(monkeypatch, others_failed=lambda n: n == 2)
    for _ in range(2 * FailStop.CHECK_EVERY - 1):
        stop.after_step()
    with pytest.raises(RuntimeError, match=r"TP ranks \[5\]"):
        stop.after_step()


def test_runner_budget_and_served_log(monkeypatch):
    from vllm.models.muse_spark.amd.specialized_decode.mono.abi import (
        HEALTH_WORDS,
        STARTUP_WAIT_SECONDS,
        WAIT_BUDGET,
        WAIT_SECONDS,
    )

    runner = object.__new__(runner_mod.MonoRunner)
    runner.health = torch.zeros(HEALTH_WORDS, dtype=torch.int32)
    runner.epoch = torch.zeros(1, dtype=torch.int32)
    runner.steps, runner.epoch_mark = 0, None
    runner.set_wait_budget(STARTUP_WAIT_SECONDS)
    assert int(runner.health[WAIT_BUDGET]) == 120 and not runner.health[2:].any()
    runner.set_wait_budget(WAIT_SECONDS)
    assert int(runner.health[WAIT_BUDGET]) == 10

    logs = []
    monkeypatch.setattr(runner_mod, "LOG_EVERY", 4)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(runner_mod.logger, "info", lambda msg, *a: logs.append(a))
    runner.epoch[0] = 2**31 - 2  # served across the int32 wrap
    for served in (True, True, False, True, True, True, True, False):
        if served:
            runner.step_begin()
        runner.count_step()
    assert logs == [(2, 4), (5, 8)]
    # restarted at the end of the startup: counts from the device epoch
    logs.clear()
    runner.reset_count()
    for served in (True, True, False, True):
        if served:
            runner.step_begin()
        runner.count_step()
    assert logs == [(3, 4)]


def test_epoch_wrap_arithmetic():
    """Tags stay positive int32s and the peer parity alternates across the
    22-bit mask's wrap and the int32 counter's sign wrap."""
    runner = object.__new__(runner_mod.MonoRunner)
    slot = spec.k2_slot(61)
    for start in (spec.EPOCH_MASK - 1, 2**31 - 2):
        runner.epoch = torch.tensor([start], dtype=torch.int32)
        seen = []
        for _ in range(4):
            runner.step_begin()
            e = int(runner.epoch[0])
            tag = spec.step_tag(e, slot)
            assert 0 < tag < 2**31
            seen.append((e & 1, tag))
        assert [p for p, _ in seen] == [seen[0][0], 1 - seen[0][0]] * 2
        assert len({t for _, t in seen}) == 4
    assert int(runner.epoch[0]) == -(2**31) + 2  # wrapped
    assert spec.step_tag(-(2**31), slot) == spec.step_tag(0, slot)


def test_quark_stub_is_a_quark_config():
    from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

    assert isinstance(quark_config(), QuarkConfig)


# --------------------------------------------------------------------------
# The executor's lifecycle (build, KV binds, self-test, capture), single rank


class _ExecRunner:
    """``MonoRunner`` without peer memory (a single CPU rank)."""

    def __init__(self, device):
        from vllm.models.muse_spark.amd.specialized_decode.mono.abi import (
            HEALTH_WORDS,
            STARTUP_WAIT_SECONDS,
            WAIT_BUDGET,
        )

        self.scratch = torch.zeros(8, dtype=U8)
        self.epoch = torch.zeros(1, dtype=torch.int32)
        self.health = torch.zeros(HEALTH_WORDS, dtype=torch.int32)
        self.health[WAIT_BUDGET] = STARTUP_WAIT_SECONDS
        self.no_pad = torch.zeros(runner_mod.NO_PAD_BYTES, dtype=U8)
        self.peer_table = torch.zeros(8, dtype=torch.int64)
        self.rank, self.alarm, self.fault, self.first_layer = 0, 0, {}, -1
        self.steps, self.epoch_mark = 0, None

    step_begin = runner_mod.MonoRunner.step_begin
    set_wait_budget = runner_mod.MonoRunner.set_wait_budget
    reset_count = runner_mod.MonoRunner.reset_count
    launch_args = runner_mod.MonoRunner.launch_args
    peer_args = runner_mod.MonoRunner.peer_args


def _executor(monkeypatch, fill=0.0, compile_error=None):
    """A ``SpecializedDecode`` over a 4-layer fake backbone (kernel layers 1..3)
    at widths 1..3, with the kernel module, the TP group and the fail-stop
    stubbed. ``fill``: what the stub launch writes into ``branch_out``."""
    from vllm.models.muse_spark.amd.specialized_decode.mono.abi import FAILED

    bb = _backbone()
    bb.parameters = lambda: iter([torch.zeros(1)])
    compiled, launches, logs = [], [], []

    def compile_build(key):
        if compile_error is not None:
            raise compile_error
        compiled.append(key)

    def launch(key, args):
        tokens = key.tokens
        pad = args["pad"][:tokens]
        launches.append((key, int(args["epoch"][0]), bool(pad.all())))
        assert set(args) == set(spec.K2_ARGS)
        args["stream_out"].zero_()
        args["branch_out"].fill_(0.0 if bool(pad.all()) else fill)

    tp = SimpleNamespace(cpu_group=None, world_size=1, rank_in_group=0)
    monkeypatch.setattr(decode_mod, "get_tp_group", lambda: tp)
    monkeypatch.setattr(decode_mod, "MonoRunner", _ExecRunner)
    monkeypatch.setattr(decode_mod, "compile_build", compile_build)
    monkeypatch.setattr(decode_mod, "k2_launch", launch)
    monkeypatch.setattr(
        decode_mod, "FailStop", lambda failed, rank, world, group: failed
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda *a: None)
    monkeypatch.setattr(decode_mod.logger, "info", lambda msg, *a: logs.append(a))
    cfg = vllm_config()
    decode = SpecializedDecode(bb, DecodeConfig(1, (1, 2, 3)), cfg)
    assert decode.failstop.data_ptr() == decode.runner.health[FAILED:].data_ptr()
    return decode, bb, compiled, launches, logs


def _bind(bb):
    full, sliding = (
        torch.zeros(2, 2, 16, 128, dtype=BF),
        torch.zeros(3, 2, 16, 128, dtype=BF),
    )
    for i in (1, 2, 3):
        bb.layers[i].self_attn.attn.kv_cache = full if i == 1 else sliding


def test_executor_lifecycle(monkeypatch):
    from vllm.models.muse_spark.amd.specialized_decode.mono.abi import (
        FAILED,
        WAIT_BUDGET,
    )

    decode, bb, compiled, launches, logs = _executor(monkeypatch)
    runner = decode.runner
    # every build compiled once, before any KV cache (memory profiling)
    assert compiled == [spec.K2Build(tokens=w) for w in (1, 2, 3)]
    assert runner.first_layer == 1 and not launches
    assert decode.names == {
        FULL: "model.layers.1.self_attn.attn",
        SLIDING: "model.layers.2.self_attn.attn",
    }
    assert decode.kernels[2][3].prev is bb.layers[2] and decode.kernels[2][3].width == 2
    assert decode.kernels[1][1].first and not decode.kernels[1][2].first
    assert logs[-1][:3] == (3, 1, 3)

    # the memory-profiling run: nothing bound yet
    decode.prepare_kv()
    assert decode._kv is None and not launches

    # a bind: two self-test steps a width, all-pad then all-live, then the log
    _bind(bb)
    runner.set_wait_budget(10)
    decode.prepare_kv()
    assert decode._kv == decode._kv_key() is not None
    assert int(runner.health[WAIT_BUDGET]) == 120
    assert [(k.tokens, e, p) for k, e, p in launches] == [
        (1, 1, True), (1, 2, False), (2, 3, True), (2, 4, False),
        (3, 5, True), (3, 6, False),
    ]  # fmt: skip
    # no FULL decode graph of 3 rows: MRV2 replays the 4-row graph, padded
    assert logs[-1] == ([3], [4])
    # with no larger FULL decode graph the width runs eagerly
    cc = decode.vllm_config.compilation_config
    sizes = cc.cudagraph_capture_sizes
    cc.cudagraph_capture_sizes = [1, 2]
    decode._log_graphs()
    assert logs[-1] == ([3],)
    cc.cudagraph_capture_sizes = sizes
    # the served-steps log starts after the self-test's steps
    assert (runner.steps, runner.epoch_mark) == (0, 6)

    # the capture's vote, then the serving budget
    runner.step_begin()  # a capture warm-up step: not served
    runner.steps = 3
    decode.after_capture()
    assert int(runner.health[WAIT_BUDGET]) == 10
    assert (runner.steps, runner.epoch_mark) == (0, 7)
    # a re-bind raises it again until the next capture ends
    decode.prepare_kv()
    assert int(runner.health[WAIT_BUDGET]) == 120

    # a rank that gave up a wait during the warm-ups stops the startup
    runner.health[FAILED] = 1
    with pytest.raises(ValueError, match="the CUDA-graph capture warm-up: rank 0"):
        decode.after_capture()
    with pytest.raises(ValueError, match="the startup self-test: rank 0 gave up"):
        decode.prepare_kv()


def test_executor_self_test_rejects_bad_outputs(monkeypatch):
    decode, bb, _, _, _ = _executor(monkeypatch, fill=float("nan"))
    _bind(bb)
    with pytest.raises(ValueError, match="width 1 gave non-finite rows"):
        decode.prepare_kv()


def test_executor_refuses_before_any_collective(monkeypatch):
    with pytest.raises(ValueError, match="compiling the kernels failed"):
        _executor(monkeypatch, compile_error=RuntimeError("no gfx950"))
    monkeypatch.setattr(
        decode_mod, "refuse_model", lambda bb, layers: ["layer 2: w13 not shuffled"]
    )
    monkeypatch.setattr(
        decode_mod,
        "MonoRunner",
        lambda device: pytest.fail("peer memory before the weights' vote"),
    )
    with pytest.raises(ValueError, match="the loaded weights: layer 2: w13"):
        SpecializedDecode(_backbone(), DecodeConfig(1, (1,)), vllm_config())


def test_executor_after_step_polls_and_counts(monkeypatch):
    decode = object.__new__(SpecializedDecode)
    calls = []
    decode.failstop = SimpleNamespace(after_step=lambda: calls.append("failstop"))
    decode.runner = SimpleNamespace(count_step=lambda: calls.append("count"))
    decode.after_step()
    assert calls == ["failstop", "count"]


# --------------------------------------------------------------------------
# Mono off: the model's edits are inert (the stock decode, call for call)


def _recording_layers(n: int, calls: list):
    """``n`` stock decoder-layer stand-ins: each records its call and returns
    a fresh residual stream (so the next layer's input is identifiable)."""

    def make(i):
        def stock(*args, **kwargs):
            out = _stream(_layer(i))
            calls.append((i, args, kwargs, out))
            return out

        return stock

    return [make(i) for i in range(n)]


def _stock_backbone(layers, fuse_across_layers: bool):
    bb = object.__new__(MuseSpark1xModel)
    nn.Module.__init__(bb)
    bb.layers = layers
    bb.start_layer, bb.end_layer = 0, len(layers)
    bb.residual_dtype, bb.default_dtype = F32, BF
    bb.fuse_across_layers = fuse_across_layers
    bb.aux_hidden_state_layers = ()
    bb.norm = lambda x, out_dtype: ("norm", x)
    return bb


@pytest.mark.parametrize("fuse_across_layers", [True, False])
def test_mono_off_layer_loop_calls_exactly_the_stock_layers(
    monkeypatch, fuse_across_layers
):
    """With no step (the switch off), or a step whose layers all run stock,
    every decoder layer is called once, in order, exactly as the stock loop
    calls it: ``layer(positions, x)`` with the previous layer's output (or its
    unfused boundary), no keyword, and the final norm sees the last one."""
    monkeypatch.setattr(
        model_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    unfused: list = []

    def unfuse(x):
        unfused.append((x, torch.zeros(2, spec.HIDDEN)))
        return unfused[-1][1]

    monkeypatch.setattr(model_mod, "norm_then_gated_residual", unfuse)
    monkeypatch.setattr(
        model_mod,
        "norm_residual_then_norm",
        lambda x, norm, out_dtype: (("final", x), None),
    )
    positions = torch.arange(2)
    embeds = torch.ones(2, spec.HIDDEN, dtype=BF)
    n = 4
    for step in (None, Step(2, {}, {}, {}, torch.zeros(16, dtype=U8))):
        calls: list = []
        unfused.clear()
        bb = _stock_backbone(_recording_layers(n, calls), fuse_across_layers)
        out = bb.forward(None, positions, None, inputs_embeds=embeds, step=step)
        assert [c[0] for c in calls] == list(range(n))
        for _, args, kwargs, _ in calls:
            assert kwargs == {} and len(args) == 2 and args[0] is positions
        first = calls[0][1][1]
        assert first.dtype == F32 and torch.equal(first, embeds.float())
        # what layer i returned is what layer i + 1 (or the final norm) got
        outs = [unfused[i][0] for i in range(len(unfused))]
        if fuse_across_layers:
            assert unfused == []
            for i in range(1, n):
                assert calls[i][1][1] is calls[i - 1][3]
            assert out == ("final", calls[-1][3])
        else:
            assert len(outs) == n
            assert all(o is c[3] for o, c in zip(outs, calls))
            for i in range(1, n):
                assert calls[i][1][1] is unfused[i - 1][1]
            assert out == ("norm", unfused[-1][1])


def test_mono_off_model_hooks_are_inert(monkeypatch):
    """The CausalLM's hooks with the switch off: forward hands the model no
    step, compute_logits polls nothing, process_weights_after_loading neither
    refuses nor materializes, and the attention runs o_proj by default."""
    from vllm.models.muse_spark.amd.model import MuseSpark1xAttention
    from vllm.models.muse_spark.common import layers as common_layers

    off = _causal_lm()  # specialized_decode_config None, no executor
    seen = []

    def model(*args, **kwargs):
        seen.append((args, kwargs))
        return "hidden"

    off.model = model
    pos = torch.zeros(4)
    assert off.forward("ids", pos) == "hidden"
    assert seen == [(("ids", pos, None, None), {"step": None})]
    off.lm_head = SimpleNamespace(weight=torch.zeros(1, dtype=BF))
    logits = torch.zeros(1, 3)
    off.logits_processor = lambda head, hidden: logits
    off.mul_tanh_soft_cap = None
    off.metap_output_multiplier = 1.0
    off.output_soft_cap_temp = None
    assert off.compute_logits(torch.zeros(1, 4)) is logits
    monkeypatch.setattr(
        common_layers,
        "materialize_residual_path_constants",
        lambda m: pytest.fail("materialized with the switch off"),
    )
    off.process_weights_after_loading()  # no error, nothing materialized

    # attention: o_proj runs unless skip_o_proj is asked for (default False)
    attn = object.__new__(MuseSpark1xAttention)
    nn.Module.__init__(attn)
    attn.num_heads, attn.head_dim = 2, 4
    attn.qkv_proj = lambda h: (h, None)
    attn._qk_norm_rope_kv_insert = lambda qkv, positions: (qkv, None, None, None)
    attn.fused_qk_norm_rope_kv_insert = True
    attn.attn = lambda q, k, v: q
    attn.fuse_o_norm_gate = False
    attn.use_o_norm = False
    attn.use_o_gate = False
    o_calls = []

    def o_proj(x):
        o_calls.append(x)
        return x * 2, None

    attn.o_proj = o_proj
    h = torch.ones(3, 8)
    assert torch.equal(attn(torch.zeros(3), h), h * 2) and len(o_calls) == 1
    assert torch.equal(attn(torch.zeros(3), h, skip_o_proj=True), h)
    assert len(o_calls) == 1


def test_importing_the_model_loads_no_kernel_module():
    """model.py imports only the switch and the hooks: no FlyDSL kernel module
    (mono.layer, the stages, the sync helpers) and no executor at import."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import vllm.models.muse_spark.amd.model\n"
        "pkg = 'vllm.models.muse_spark.amd.specialized_decode'\n"
        "bad = [m for m in sys.modules if m.startswith(pkg + '.mono.')\n"
        "       and m not in (pkg + '.mono.spec', pkg + '.mono.abi')]\n"
        "bad += [m for m in (pkg + '.runtime.decode', pkg + '.runtime.layers')\n"
        "        if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert r.returncode == 0, r.stderr[-4000:]
