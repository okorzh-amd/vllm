# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The layer kernels: take and return what ``MuseSpark1xDecoderLayer.forward``
does, a ``MuseSpark1xResidualStream`` (or the first layer's plain stream) in,
the layer's deferred FFN boundary out.

Phase A: vLLM runs the layer's residual boundary and attention up to o_proj's
input (the gated core); one K2 launch then runs o_proj and its all-reduce, the
attention boundary, the router and routing, the pooler, the MXFP4 experts, the
expert reduce, the post-expert norm and the unpooler.

Phases B and C (``Phases``, off by default) change what runs before o_proj:
- the front: vLLM's boundary, qkv GEMV and fused QK-norm / RoPE / KV insert
  (``MuseSpark1xAttention.front``), or one K1 launch (B1);
- K2's core (``K2Build.core``): o_proj's input from vLLM ("tensor", Phase A),
  vLLM's attention output with the gated norm in K2's core load ("gated", B2),
  or the split-KV attention core in K2 ("split", C; q = 1 widths only, a DSpark
  verify step keeps the gated core).

K1 and the split core read this layer's KV cache and its KV group's per-step
metadata (slot mapping, block table, sequence lengths): MRV2's persistent
buffers, which ``SpecializedDecode.plan`` checks, so a FULL graph's baked
addresses stay valid at every replay.

Weights are views of the loaded parameters, taken once when the layer kernel is
built: the model refuses weight reloads once the executor exists
(``hooks.refuse_weight_reload``), so the parameters never move. Replicated
weights are passed whole; the kernel offsets them by its ``rank`` argument.

The kernel modules (``mono.layer``, ``mono.k1``, FlyDSL) are imported only
inside the functions that build or launch them.
"""

import importlib
import importlib.util
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.models.muse_spark.amd.specialized_decode.mono import spec
from vllm.models.muse_spark.amd.specialized_decode.mono.spec import K1Build, K2Build
from vllm.models.muse_spark.amd.specialized_decode.runtime import HINT
from vllm.models.muse_spark.amd.specialized_decode.runtime.guards import FULL, SLIDING
from vllm.models.muse_spark.amd.specialized_decode.runtime.runner import MonoRunner

if TYPE_CHECKING:
    from vllm.models.muse_spark.amd.specialized_decode.runtime.decode import Step
    from vllm.models.muse_spark.common.layers import MuseSpark1xResidualStream

_MONO = "vllm.models.muse_spark.amd.specialized_decode.mono"

# the fronts and K2's cores (spec.K2_CORES)
STOCK, K1 = "stock", "k1"
FRONTS = (STOCK, K1)
TENSOR, GATED, SPLIT = spec.K2_CORES
# additional_config keys selecting them (with decode_backend), and the narrowest
# width K1 serves (DESIGN_BC 11.3: K1 only where it passes its gate)
FRONT_KEY, CORE_KEY, K1_MIN_WIDTH_KEY = "mono_front", "mono_core", "mono_k1_min_width"
# ... and the split core's o_proj placement (K2Build.oproj_issue, TUNE L2)
CORE_OPROJ_KEY = "mono_core_oproj"
# ... and K1's GEMV task layout (K1Build.layout, spec.K1_LAYOUTS; TUNE L1)
K1_LAYOUT_KEY = "mono_k1_layout"
# ... and the split core's split / combine retune (TUNE L3; spec.split_fields):
# a preset name (spec.SPLIT_PRESETS, "+"-composed) at every width, or a dict
# {"w" or "lo-hi": preset} by launch width (widths it leaves out: "base").
# Absent: spec.DEFAULT_SPLIT_PRESET at every width (with the split core).
SPLIT_PRESET_KEY = "mono_split_preset"
BASE_SPLIT = ("base",) * spec.MAX_ROWS
DEFAULT_SPLIT = (spec.DEFAULT_SPLIT_PRESET,) * spec.MAX_ROWS


def split_presets(value) -> tuple[str, ...]:
    """``mono_split_preset``'s value -> the preset of widths 1 .. MAX_ROWS.
    ValueError for an unknown preset, a malformed width or a width named
    twice."""
    bad = ValueError(
        f"Muse Spark specialized decode: {SPLIT_PRESET_KEY} {value!r} (a preset "
        f"of {spec.SPLIT_PRESETS}, '+'-composed, or a dict of them by width "
        f"'w' / 'lo-hi' in 1..{spec.MAX_ROWS}; {HINT})"
    )
    if isinstance(value, str):
        items = {f"1-{spec.MAX_ROWS}": value}
    elif isinstance(value, Mapping):
        items = dict(value)
    else:
        raise bad
    out: list[str | None] = [None] * spec.MAX_ROWS
    for rng, preset in items.items():
        try:
            spec.check_split_preset(preset)
            lo, dash, hi = str(rng).partition("-")
            if dash and not hi:  # "3-": a range needs both ends
                raise ValueError(rng)
            lo_i, hi_i = int(lo), int(hi or lo)
        except ValueError:
            raise bad from None
        if not 1 <= lo_i <= hi_i <= spec.MAX_ROWS:
            raise bad
        for w in range(lo_i, hi_i + 1):
            if out[w - 1] is not None:
                raise bad
            out[w - 1] = preset
    return tuple(p or "base" for p in out)


# the arguments a launch adds to the layer's static ones, by K2 core
STEP_ARGS_BY_CORE = {
    TENSOR: ("core", "stream", "pad", "stream_out", "branch_out"),
    GATED: ("attn", "qkvo", "stream", "pad", "stream_out", "branch_out"),
    SPLIT: (
        "qkvo",
        *spec.K2_KV_ARGS,
        "block_table",
        "bt_rows",
        "bt_stride",
        "seq_lens",
        "stream",
        "pad",
        "stream_out",
        "branch_out",
    ),
}
STEP_ARGS = STEP_ARGS_BY_CORE[TENSOR]
# ... and to K1's: the step's tensors and this layer's KV cache (``bind_kv``)
K1_STEP_ARGS = (
    "branch",
    "stream",
    "pad",
    "positions",
    "slots",
    *spec.K2_KV_ARGS,
    "stream_out",
    "qkvo",
)


@dataclass(frozen=True)
class Phases:
    """The kernels serving a kernel layer: ``front`` "stock" (vLLM's) or "k1"
    (B1), ``core`` K2's core at q = 1 widths (``spec.K2_CORES``). With K1 on,
    widths below ``k1_min_width`` keep vLLM's front (the 11.3 cut matrix's "K1
    only where it wins"). ``core_oproj``: the split core's o_proj placement
    (``spec.OPROJ_ISSUES``; "early" needs ``core`` "split"). The default is
    Phase A. ``split_preset``: the design's split at every width unless
    given (``decode_phases`` gives the split core ``DEFAULT_SPLIT`` when
    ``mono_split_preset`` is absent)."""

    front: str = STOCK
    core: str = TENSOR
    k1_min_width: int = 1
    core_oproj: str = "late"
    # K1's GEMV task layout (TUNE L1; "g192": the E3 build)
    k1_layout: str = "g192"
    # the split core's split / combine retune (TUNE L3): a preset a width
    # (``split_presets``; all "base": the design's N and MIN_PER)
    split_preset: tuple[str, ...] = BASE_SPLIT

    def __post_init__(self):
        if self.front not in FRONTS:
            raise ValueError(
                f"Muse Spark specialized decode: {FRONT_KEY} {self.front!r} "
                f"(one of {FRONTS}; {HINT})"
            )
        if self.core not in spec.K2_CORES:
            raise ValueError(
                f"Muse Spark specialized decode: {CORE_KEY} {self.core!r} "
                f"(one of {spec.K2_CORES}; {HINT})"
            )
        width = self.k1_min_width
        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or not 1 <= width <= spec.MAX_ROWS
            or (width != 1 and self.front != K1)
        ):
            raise ValueError(
                f"Muse Spark specialized decode: {K1_MIN_WIDTH_KEY} {width!r} "
                f"(an int in 1..{spec.MAX_ROWS}, with {FRONT_KEY} {K1!r}; {HINT})"
            )
        if self.core_oproj not in spec.OPROJ_ISSUES or (
            self.core_oproj != "late" and self.core != SPLIT
        ):
            raise ValueError(
                f"Muse Spark specialized decode: {CORE_OPROJ_KEY} "
                f"{self.core_oproj!r} (one of {spec.OPROJ_ISSUES}; not 'late' "
                f"only with {CORE_KEY} {SPLIT!r}; {HINT})"
            )
        if self.k1_layout not in spec.K1_LAYOUTS or (
            self.k1_layout != "g192" and self.front != K1
        ):
            raise ValueError(
                f"Muse Spark specialized decode: {K1_LAYOUT_KEY} "
                f"{self.k1_layout!r} (one of {spec.K1_LAYOUTS}; not 'g192' only "
                f"with {FRONT_KEY} {K1!r}; {HINT})"
            )
        sp = self.split_preset
        if (
            not isinstance(sp, tuple)
            or len(sp) != spec.MAX_ROWS
            or (sp != BASE_SPLIT and self.core != SPLIT)
        ):
            raise ValueError(
                f"Muse Spark specialized decode: {SPLIT_PRESET_KEY} {sp!r} (one "
                f"preset a width 1..{spec.MAX_ROWS}; not all 'base' only with "
                f"{CORE_KEY} {SPLIT!r}; {HINT})"
            )
        for p in sp:
            spec.check_split_preset(p)

    def k2_split(self, q: int, width: int, window: bool) -> dict:
        """The split core's K2Build split fields at q rows a request and a
        launch width (``spec.split_fields`` of the width's preset; {} where
        the split core does not run)."""
        if self.k2_core(q) != SPLIT:
            return {}
        return spec.split_fields(self.split_preset[width - 1], width, window)

    def front_at(self, width: int) -> str:
        """The front at a launch width: K1 from ``k1_min_width`` on."""
        return K1 if self.front == K1 and width >= self.k1_min_width else STOCK

    def k2_core(self, q: int) -> str:
        """K2's core at q rows a request: the split core masks no query row
        causally, so a DSpark verify step (q > 1) keeps the gated core load."""
        return GATED if self.core == SPLIT and q != 1 else self.core

    def k2_oproj(self, q: int) -> str:
        """K2's o_proj placement at q rows a request: the split core's
        (``core_oproj``) where it runs, else "late"."""
        return self.core_oproj if self.k2_core(q) == SPLIT else "late"

    def reads_kv(self, q: int) -> bool:
        """A kernel reads the KV cache and the per-step metadata (slots, block
        tables, sequence lengths): K1 or the split core."""
        return self.front == K1 or self.k2_core(q) == SPLIT


PHASE_A = Phases()


def decode_phases(vllm_config) -> Phases:
    """``additional_config["mono_front"]`` / ``["mono_core"]`` /
    ``["mono_k1_min_width"]`` / ``["mono_core_oproj"]`` / ``["mono_k1_layout"]``
    / ``["mono_split_preset"]``, Phase A's when absent (or when
    ``additional_config`` is not a dict); with the split core an absent
    ``mono_split_preset`` is ``spec.DEFAULT_SPLIT_PRESET``."""
    extra = vllm_config.additional_config
    extra = extra if isinstance(extra, dict) else {}
    core = extra.get(CORE_KEY, TENSOR)
    preset = extra.get(SPLIT_PRESET_KEY)
    return Phases(
        extra.get(FRONT_KEY, STOCK),
        core,
        extra.get(K1_MIN_WIDTH_KEY, 1),
        extra.get(CORE_OPROJ_KEY, "late"),
        k1_layout=extra.get(K1_LAYOUT_KEY, "g192"),
        split_preset=(
            split_presets(preset)
            if preset is not None
            else DEFAULT_SPLIT
            if core == SPLIT
            else BASE_SPLIT
        ),
    )


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


def k1_tensors(layer, prev) -> dict[str, torch.Tensor | int]:
    """A kernel layer's K1 weights (DESIGN_BC 5.2): the previous layer's FFN
    boundary (post_ffn_norm, add_ffn) and this layer's pre-qkv norm, qkv_proj
    and RoPE table, as views of the loaded tensors (a packed copy would cost
    2.26 GB a rank). A NoPE layer passes no table (0, 0 rows)."""
    attn = layer.self_attn
    rope = attn.rotary_emb
    table = None if rope is None else rope.cos_sin_cache
    return {
        "w_postffn": prev.post_ffn_norm.weight,
        "ffn_alpha": prev.add_ffn.alpha,
        "ffn_beta": prev.add_ffn.beta,
        "w_preqkv": attn.pre_qkv_proj_norm.weight,
        "w_qkv": attn.qkv_proj.weight,
        "cos_sin": 0 if table is None else table,
        "max_pos": 0 if table is None else int(table.shape[0]),
    }


def kv_args(kv: torch.Tensor) -> dict[str, torch.Tensor | int]:
    """A KV cache [blocks, KV_HEADS, KV_BLOCK, KV_C] as K1 and the split core
    address it: block, head and token strides in bytes (a head stride may pass
    4 GiB: Int64), and the block count block ids are clamped below."""
    n = kv.element_size()
    return {
        "kv": kv,
        "kv_sb": kv.stride(0) * n,
        "kv_sh": kv.stride(1) * n,
        "kv_st": kv.stride(2) * n,
        "kv_blocks": int(kv.shape[0]),
    }


def table_rows(block_table: torch.Tensor) -> int:
    """The rows of a block table's buffer from the view's first row: the split
    core's bound on the table (``bt_rows``), the whole persistent table even
    when the step's view holds fewer rows."""
    entries = block_table.untyped_storage().nbytes() // block_table.element_size()
    return (entries - block_table.storage_offset()) // block_table.stride(0)


def _kernel_module(key: K1Build | K2Build):
    """The kernel module of ``key`` and its ABI: ``mono.k1`` for a K1Build,
    ``mono.layer`` for a K2Build of a core it builds (``ABI_BY_CORE``). A
    kernel this tree does not hold raises: the startup stops on every rank
    (``SpecializedDecode``'s agreed compile), so nothing ever launches a build
    under another build's ABI."""
    if isinstance(key, K1Build):
        if importlib.util.find_spec(_MONO + ".k1") is None:
            raise RuntimeError(
                "Muse Spark specialized decode: K1 (mono.k1, Phase B1) is not in "
                f"this build; drop additional_config {FRONT_KEY!r}"
            )
        k1 = importlib.import_module(_MONO + ".k1")
        return k1, k1.ABI
    if isinstance(key, K2Build):
        from vllm.models.muse_spark.amd.specialized_decode.mono import layer as k2

        abis = getattr(k2, "ABI_BY_CORE", {TENSOR: k2.ABI})
        if key.core not in abis:
            raise RuntimeError(
                f"Muse Spark specialized decode: K2's {key.core!r} core is not in "
                f"this build (mono.layer builds {sorted(abis)}); set "
                f"additional_config {CORE_KEY!r} to one of them"
            )
        return k2, abis[key.core]
    raise TypeError(f"a build key is a K1Build or a K2Build, not {type(key).__name__}")


def k1_launch(key: K1Build, args: Mapping[str, torch.Tensor | int]) -> None:
    """One K1 launch on the current stream (``mono.k1.k1_launch``)."""
    importlib.import_module(_MONO + ".k1").k1_launch(key, args)


def k2_launch(key: K2Build, args: Mapping[str, torch.Tensor | int]) -> None:
    """One K2 launch on the current stream (``mono.layer.k2_launch``)."""
    from vllm.models.muse_spark.amd.specialized_decode.mono import layer as k2

    k2.k2_launch(key, args)


def compile_build(key: K1Build | K2Build) -> None:
    """Compiles one launch's kernel without running it: FlyDSL's compile-only
    mode returns before the launch. Loading the code object happens at the first
    launch (``SpecializedDecode.self_test``, before any capture)."""
    module, abi = _kernel_module(key)
    launch = module.build(key)
    # FlyDSL's compile-only switch has no FLYDSL_ prefix
    old = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"
    try:
        if torch.cuda.is_available():
            launch(*abi.zeros(), stream=torch.cuda.current_stream())
        else:  # a GPU-less compile: FlyDSL's default stream
            launch(*abi.zeros())
    finally:
        if old is None:
            del os.environ["COMPILE_ONLY"]
        else:
            os.environ["COMPILE_ONLY"] = old


def _table_key(t: torch.Tensor | int) -> tuple | None:
    """A RoPE table's identity as K1 bakes it in (None: no table)."""
    if isinstance(t, int):
        return None
    return t.data_ptr(), t.dtype, tuple(t.shape)


def _check_rows(what: str, t: torch.Tensor, rows: int, cols: int, dtype) -> None:
    """What the stock callables give, and the kernels' buffer reads assume."""
    assert t.is_contiguous() and t.dtype == dtype, (what, t.dtype)
    assert tuple(t.shape) == (rows, cols), (what, tuple(t.shape))


def _check_split_metadata(
    block_table: torch.Tensor, seq_lens: torch.Tensor, rows: int
) -> None:
    """What the split core reads of a step's metadata for token t < rows: the
    table's row t, ``bt_stride`` int32 entries at ``t * bt_stride`` (the row
    is the whole stride: the core clamps L to ``bt_stride`` pages, so a view
    narrower than its stride would let it read columns past the view), and
    ``seq_lens[t]`` int32 (``SpecializedDecode.plan`` checks that both are
    MRV2's persistent buffers; this checks the views a launch passes, the
    self-test's too)."""
    assert block_table.dtype == torch.int32 and block_table.ndim == 2, (
        "block_table",
        block_table.dtype,
        tuple(block_table.shape),
    )
    assert (
        block_table.stride(1) == 1
        and block_table.stride(0) == block_table.shape[1]
        and block_table.shape[0] >= rows
    ), (
        "block_table",
        tuple(block_table.shape),
        block_table.stride(),
    )
    assert seq_lens.dtype == torch.int32 and seq_lens.ndim == 1, (
        "seq_lens",
        seq_lens.dtype,
        tuple(seq_lens.shape),
    )
    assert seq_lens.stride(0) == 1 and seq_lens.shape[0] >= rows, (
        "seq_lens",
        tuple(seq_lens.shape),
        seq_lens.stride(),
    )


class KernelLayer:
    """Decoder layer ``layer`` (attention ``kind``) at one launch width of
    ``q`` rows a request: the front (vLLM's or K1), the attention core (vLLM's
    or K2's), then K2. ``prev`` is the decoder layer before it, whose FFN
    boundary this layer's input carries."""

    def __init__(
        self,
        layer,
        prev,
        runner: MonoRunner,
        width: int,
        kind: str,
        q: int = 1,
        phases: Phases = PHASE_A,
    ):
        if kind not in (FULL, SLIDING):
            raise ValueError(f"attention kind {kind!r} ({FULL!r} or {SLIDING!r})")
        self.layer, self.prev, self.runner, self.width = layer, prev, runner, width
        self.kind, self.q, self.phases = kind, q, phases
        self.core = phases.k2_core(q)  # K2's core at this width
        self.name = layer.self_attn.attn.layer_name  # its metadata's key
        # the step's first kernel layer bumps the epoch (a graph captures it)
        self.first = layer.layer_idx == runner.first_layer
        self.slot = spec.k2_slot(layer.layer_idx)
        # everything but the step's tensors, packed once
        self.static = k2_tensors(layer, runner) | {
            "slot": self.slot,
            **runner.launch_args(),
            **runner.peer_args(),
        }
        self.static_k1 = (
            k1_tensors(layer, prev)
            | {
                "slot": spec.k1_slot(layer.layer_idx),
                "rank": runner.rank,
                **runner.launch_args(),
            }
            if phases.front_at(width) == K1
            else None
        )
        # this layer's KV cache arguments, from bind_kv (K1, the split core)
        self.kv_args: dict[str, torch.Tensor | int] | None = None

    def k1_key(self) -> K1Build | None:
        """The K1 build this layer launches, None on vLLM's front (the fault
        hook from the runner: tests)."""
        if self.static_k1 is None:
            return None
        return K1Build(
            tokens=self.width,
            rope=self.kind == SLIDING,
            **self.runner.fault_k1,
            layout=self.phases.k1_layout,
        )

    def key(self) -> K2Build:
        """The K2 build this layer launches (the fault hook from the runner:
        tests)."""
        return K2Build(
            tokens=self.width,
            core=self.core,
            window=self.core == SPLIT and self.kind == SLIDING,
            oproj_issue=self.phases.k2_oproj(self.q),
            **self.phases.k2_split(self.q, self.width, self.kind == SLIDING),
            **self.runner.fault,
        )

    def builds(self) -> list[K1Build | K2Build]:
        """The builds this layer launches at its width, in launch order,
        compiled before capture."""
        k1 = self.k1_key()
        return [self.key()] if k1 is None else [k1, self.key()]

    def bind_kv(self) -> None:
        """At every KV bind (``prepare_kv``, after ``refuse_caches``): this
        layer's cache arguments, and the RoPE table K1 holds checked against
        the module's (``RotaryEmbedding`` reassigns ``cos_sin_cache`` when a
        forward runs on another dtype or device)."""
        if not self.phases.reads_kv(self.q):
            return
        self.kv_args = kv_args(self.layer.self_attn.attn.kv_cache)
        if self.static_k1 is None:
            return
        rope = self.layer.self_attn.rotary_emb
        held = _table_key(self.static_k1["cos_sin"])
        if held != _table_key(0 if rope is None else rope.cos_sin_cache):
            raise RuntimeError(
                f"Muse Spark specialized decode: layer {self.layer.layer_idx}'s "
                "RoPE table (rotary_emb.cos_sin_cache) was reassigned after the "
                "kernels took it"
            )

    def _core_args(
        self, step: "Step", positions: torch.Tensor, attn_in, qkvo
    ) -> dict[str, torch.Tensor | int]:
        """K2's core arguments from the front's output: ``attn_in`` (vLLM's
        front) or ``qkvo`` (K1's rows)."""
        A, s = self.layer.self_attn, self.width
        if self.core == TENSOR:
            if qkvo is None:  # Phase A
                core = A(positions=positions, hidden_states=attn_in, skip_o_proj=True)
            else:
                core = A.attend(qkvo)
            _check_rows("core", core, s, spec.CORE, torch.bfloat16)
            return {"core": core}
        if qkvo is None:
            qkvo = A.front(positions, attn_in)
        _check_rows("qkvo", qkvo, s, spec.QKV_ROWS, torch.bfloat16)
        if self.core == GATED:
            attn = A.attend(qkvo, gate=False)
            _check_rows("attn", attn, s, spec.CORE, torch.bfloat16)
            return {"attn": attn, "qkvo": qkvo}
        md = step.attn_metadata[self.name]
        bt, seq_lens = md.block_table, md.seq_lens
        _check_split_metadata(bt, seq_lens, s)
        assert self.kv_args is not None, "bind_kv before the first split launch"
        return {
            "qkvo": qkvo,
            **self.kv_args,
            "block_table": bt,
            "bt_rows": table_rows(bt),
            "bt_stride": bt.stride(0),
            "seq_lens": seq_lens,
        }

    def __call__(
        self,
        step: "Step",
        positions: torch.Tensor,
        x: "MuseSpark1xResidualStream | torch.Tensor",
    ) -> "MuseSpark1xResidualStream":
        from vllm.models.muse_spark.amd.model import norm_residual_then_norm
        from vllm.models.muse_spark.common.layers import MuseSpark1xResidualStream

        layer, s = self.layer, self.width
        k1 = self.k1_key()
        if self.first:
            self.runner.step_begin()
        attn_in = qkvo = None
        if isinstance(x, torch.Tensor):
            if k1 is not None:
                # unreachable while layer 0 and pipeline parallelism are refused
                raise RuntimeError(
                    f"Muse Spark specialized decode: layer {layer.layer_idx} got a "
                    "plain stream; K1 takes the previous layer's deferred boundary"
                )
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
            if k1 is None:
                attn_in, stream = norm_residual_then_norm(
                    x, layer.self_attn.pre_qkv_proj_norm, out_dtype=torch.bfloat16
                )
            else:
                stream = x.stream
                _check_rows("branch", x.branch, s, spec.HIDDEN, torch.bfloat16)
                _check_rows("stream", stream, s, spec.HIDDEN, torch.float32)
                assert self.static_k1 is not None and self.kv_args is not None
                stream_k1 = torch.empty_like(stream)
                qkvo = torch.empty(
                    s, spec.QKV_ROWS, dtype=torch.bfloat16, device=stream.device
                )
                k1_launch(
                    k1,
                    {
                        **self.static_k1,
                        **self.kv_args,
                        "branch": x.branch,
                        "stream": stream,
                        "pad": step.pad,
                        "positions": positions,
                        "slots": step.slot_mapping[self.name],
                        "stream_out": stream_k1,
                        "qkvo": qkvo,
                    },
                )
                stream = stream_k1
        core_args = self._core_args(step, positions, attn_in, qkvo)
        _check_rows("stream", stream, s, spec.HIDDEN, torch.float32)
        stream_out = torch.empty_like(stream)
        branch_out = torch.empty(
            s, spec.HIDDEN, dtype=torch.bfloat16, device=stream.device
        )
        k2_launch(
            self.key(),
            {
                **self.static,
                **core_args,
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


def self_test_step(kernel, pad: torch.Tensor, live: bool) -> list[torch.Tensor]:
    """One self-test step (``SpecializedDecode.self_test``) of a kernel layer's
    builds on a zero stream and branch, returning every output. It reads only
    the layer's launch contract: ``builds()`` ([K1Build,] K2Build), ``static``
    and, with K1, ``static_k1``. Nothing touches the real KV cache: K1 inserts
    nothing (slots -1) into a test-local zero cache of one block, and the split
    core reads that block through a zero block table, with a context of one
    token on a live row and none on a pad row."""
    *k1, k2 = kernel.builds()
    s, dev = k2.tokens, kernel.static["scratch"].device
    bf, f32 = torch.bfloat16, torch.float32
    stream = torch.zeros(s, spec.HIDDEN, dtype=f32, device=dev)
    qkvo = torch.zeros(s, spec.QKV_ROWS, dtype=bf, device=dev)
    local_kv = kv_args(
        torch.zeros(1, spec.KV_HEADS, spec.KV_BLOCK, spec.KV_C, dtype=bf, device=dev)
    )
    outputs = []
    for key in k1:
        stream_k1 = torch.empty_like(stream)
        k1_launch(
            key,
            {
                **kernel.static_k1,
                **local_kv,
                "branch": torch.zeros(s, spec.HIDDEN, dtype=bf, device=dev),
                "stream": stream,
                "pad": pad,
                "positions": torch.zeros(s, dtype=torch.int64, device=dev),
                "slots": torch.full((s,), -1, dtype=torch.int64, device=dev),
                "stream_out": stream_k1,
                "qkvo": qkvo,
            },
        )
        outputs += [stream_k1, qkvo]
        stream = stream_k1
    if k2.core == TENSOR:
        core_args = {"core": torch.zeros(s, spec.CORE, dtype=bf, device=dev)}
    elif k2.core == GATED:
        attn = torch.zeros(s, spec.CORE, dtype=bf, device=dev)
        core_args = {"attn": attn, "qkvo": qkvo}
    else:
        table = torch.zeros(s, 1, dtype=torch.int32, device=dev)
        seq_lens = torch.full((s,), int(live), dtype=torch.int32, device=dev)
        _check_split_metadata(table, seq_lens, s)
        core_args = {
            "qkvo": qkvo,
            **local_kv,
            "block_table": table,
            "bt_rows": table_rows(table),
            "bt_stride": table.stride(0),
            "seq_lens": seq_lens,
        }
    stream_out = torch.empty_like(stream)
    branch_out = torch.empty(s, spec.HIDDEN, dtype=bf, device=dev)
    k2_launch(
        k2,
        {
            **kernel.static,
            **core_args,
            "stream": stream,
            "pad": pad,
            "stream_out": stream_out,
            "branch_out": branch_out,
        },
    )
    return outputs + [stream_out, branch_out]
