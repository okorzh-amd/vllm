# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2(gated) (``K2Build(core="gated")``, DESIGN_BC 6, WP-B6) compiles for
gfx950 without a GPU, every width and the fault and debug variants, and keeps
the Phase A builds as they were.

[CPU] The ABI by core (``layer.ABI_BY_CORE``: ``attn`` and ``qkvo`` in place
of ``core``), the launch packing by the build's core, the refusal of a core
key K2Build does not know, and the gate's row placement (every (token, head)
once a CTA). The split core's ABI and builds: ``test_compile_split``.

[CPU] FlyDSL: the ISA gate of ``test_compile`` (no private segment, at most
256 VGPRs, the LDS of the tensor build, no readfirstlane waterfall); the gate's
code (Triton's DPP tree, one raw ``v_sqrt_f32`` a row, IEEE divides, a
``ds_write_b16`` a pass); and its order (6.2, rule 8): the pad words and the
core's 2-byte loads all go out before the o_proj weights with no wait among
them, no gate math or LDS write before the weights, and no wait between the
first weight load and the last ``xl`` write retires a weight load. The trace
is Phase A's program order with the same hand-over barriers. A tensor build
compiled after gated builds in the same process still hashes to the Phase A
baseline.
"""

import json

import pytest
import regex as re
import torch

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

import flydsl.expr.gpu as fx_gpu  # noqa: E402

from tests.models.muse_spark.amd_specialized_decode import (  # noqa: E402
    test_compile as tc,
)
from vllm.models.muse_spark.amd.specialized_decode.mono import (  # noqa: E402
    layer,
    spec,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.attention import (  # noqa: E402
    back,
    gate,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common import (  # noqa: E402
    plan,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.sync import (  # noqa: E402
    WAIT_TICKS,
)

compiles = tc.compiles

# every gated build the runtime and the tests launch: the widths (DSpark's
# q = 8 is width 8), the stalled-rank fault builds and a debug build
FAULT = dict(fault_rank=3, fault_epoch=2, fault_slot=spec.k2_slot(1))
GATED = [spec.K2Build(tokens=s, core="gated") for s in range(1, spec.MAX_ROWS + 1)] + [
    spec.K2Build(tokens=8, core="gated", fault_ticks=WAIT_TICKS * 3 // 2, **FAULT),
    spec.K2Build(tokens=1, core="gated", fault_ticks=WAIT_TICKS * 3 // 2, **FAULT),
    spec.K2Build(tokens=8, core="gated", debug_dump=True),
]
# the o_proj weight loads a lane has in flight when the core load starts:
# 2 tasks x 4 dwordx4 (``back.o_loads``: one 128-wide K chunk a wave)
O_LOADS = 2 * 4

_NT_LOAD = re.compile(r"^\s*buffer_load_dwordx4\b.*\bnt\s*$", re.M)
_CORE_LOAD = re.compile(r"^\s*buffer_load_(?:ushort|short_d16)\b", re.M)
_XL_WRITE = re.compile(r"^\s*ds_write_b16\b", re.M)
_VMCNT = re.compile(r"^\s*s_waitcnt\b.*\bvmcnt\((\d+)\)", re.M)
_VMEM_LOAD = re.compile(r"^\s*(?:buffer|global)_load_", re.M)


def _waits_before_weights(txt: str, first_core: int, first_nt: int) -> list[str]:
    """The ``vmcnt`` waits from the pad-word load (the last vector load before
    the first core load) to the first o_proj weight load. Any wait there, even
    one that retires only the pad words or a context load, holds the weights
    back until that load lands (6.2: the weights go out right behind the core
    operands)."""
    pads = list(_VMEM_LOAD.finditer(txt, 0, first_core))
    assert pads, "no pad-word load before the core loads"
    return [
        f"{m.group(0).strip()} after the pad load"
        for m in _VMCNT.finditer(txt, pads[-1].start(), first_nt)
    ]


def _compile(key: spec.K2Build, dump, monkeypatch) -> tuple[str, dict]:
    """``build(key)`` then its launcher on its core's ``ABI.zeros()`` in
    FlyDSL's compile-only mode, the IR dumped under ``dump``."""
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(dump))
    monkeypatch.setenv("COMPILE_ONLY", "1")
    layer._BUILDS.pop(key, None)  # trace it again (a test may record the trace)
    launch = layer.build(key)
    try:
        launch(*layer.ABI_BY_CORE[key.core].zeros())
    except RuntimeError as e:  # the load, past the compile, needs a device
        if "hipErrorNoDevice" not in str(e) and "no ROCm" not in str(e):
            raise
    return tc._isa(str(dump), key.kernel_name())


def _baseline() -> dict:
    return json.loads(tc.BASELINE.read_text())


# ---------------------------------------------------------------- [CPU]


def test_abi_by_core():
    """One ABI a built core, spec.K2_ARGS_BY_CORE renamed as the tensor
    build's (``stream`` -> ``stream_in``); "tensor" is Phase A's ``ABI``."""
    assert tuple(layer.ABI_BY_CORE) == back.CORES == ("tensor", "gated", "split")
    assert layer.ABI is layer.ABI_BY_CORE["tensor"]
    for core, abi in layer.ABI_BY_CORE.items():
        want = [layer.KERNEL_NAMES.get(n, n) for n in spec.K2_ARGS_BY_CORE[core]]
        assert list(abi.names) == want, core
        assert "stream" not in abi.names and "stream_in" in abi.names
    g = layer.ABI_BY_CORE["gated"].names
    assert g[:2] == ("attn", "qkvo") and "core" not in g
    assert g[2:] == layer.ABI.names[1:]


def test_a_core_without_a_variant_is_refused():
    """No key compiles another core's kernel under its name and ABI (B0
    review): every K2Build core has its variant (Phase C added "split"; its
    core load is the split's, no early loads), and the core dispatch refuses
    a core K2Build does not know."""
    for key in (
        spec.K2Build(tokens=1, core="split"),
        spec.K2Build(tokens=8, core="split", window=True),
    ):
        assert layer.build(key) is layer._BUILDS[key]
    assert back.core_loads({}, "split") is None
    with pytest.raises(ValueError, match="one of"):
        back.core_loads({}, "fused")
    # a gated core load without its early loads (they would trail the weights)
    with pytest.raises(ValueError, match="core_loads"):
        back.load_core({}, "gated")
    with pytest.raises(TypeError):
        layer.build((8, "gated"))


def test_k2_launch_packs_by_the_core(monkeypatch):
    """``k2_launch`` packs a gated launch's arguments by the gated names (no
    ``core``), a tensor launch's by Phase A's."""
    calls = []

    def fake_build(key):
        return lambda *a, stream: calls.append((key, a, stream))

    monkeypatch.setattr(layer, "build", fake_build)
    monkeypatch.setattr(layer.torch.cuda, "current_stream", lambda: "cur")
    names = spec.K2_ARGS_BY_CORE["gated"]
    t = {n: torch.empty(4, dtype=torch.uint8) for n in names}
    args = {**t, "rank": 5, "slot": spec.k2_slot(7), "alarm": 1234}
    key = spec.K2Build(tokens=3, core="gated")
    layer.k2_launch(key, args)
    [(got_key, packed, stream)] = calls
    assert got_key is key and stream == "cur" and len(packed) == len(names)
    for name, v in zip(names, packed):
        want = args[name]
        assert v == (want.data_ptr() if isinstance(want, torch.Tensor) else want)
    with pytest.raises(TypeError, match="missing"):
        layer.k2_launch(key, {n: v for n, v in args.items() if n != "qkvo"})
    with pytest.raises(TypeError, match="unexpected"):
        layer.k2_launch(key, {**args, "core": t["attn"]})
    # the tensor ABI refuses the gated arguments
    with pytest.raises(TypeError, match="unexpected"):
        layer.k2_launch(spec.K2Build(tokens=3), args)


def test_gate_rows_cover_every_head():
    """Pass k of wave w is (t, h) = divmod(8 k + w, 16): over the 2S passes
    the 8 waves gate every (token, head) of the launch exactly once; the LDS
    rows are back's; a lane holds 4S core operands (<= 32 VGPRs)."""
    assert gate.XL_ROW == back.XL_ROW
    for s in range(1, spec.MAX_ROWS + 1):
        assert gate.passes(s) == 2 * s
        seen = [
            gate.pass_row(k, w)
            for k in range(gate.passes(s))
            for w in range(spec.WAVES)
        ]
        assert seen == [divmod(8 * k + w, 16) for k in range(2 * s) for w in range(8)]
        assert sorted(seen) == [(t, h) for t in range(s) for h in range(spec.HEADS)]
        assert 2 * gate.passes(s) == 4 * s <= 32
    assert gate.STAGE == back.STAGE and gate.STAGE_REGIONS == ()


# ---------------------------------------------------------------- [CPU] FlyDSL


def _window(txt: str) -> tuple[int, int, int]:
    """(first core load, first o_proj weight load, last xl write) offsets."""
    core = _CORE_LOAD.search(txt)
    nt = _NT_LOAD.search(txt)
    writes = list(_XL_WRITE.finditer(txt))
    assert core and nt and writes, "no core load / weight load / xl write"
    return core.start(), nt.start(), writes[-1].start()


@compiles
@pytest.mark.parametrize("key", GATED, ids=lambda k: k.kernel_name())
def test_gated_builds_compile_without_a_gpu(key, tmp_path, monkeypatch):
    """Every gated build passes the ISA gate with the tensor build's LDS, has
    the gate's code, and keeps 6.2's order: the core loads, then the weights,
    then the gate, never waiting for a weight load before the last xl
    write."""
    txt, res = _compile(key, tmp_path, monkeypatch)
    print(f"{key.kernel_name()}: {res}")
    s = key.tokens
    assert res["private_segment_fixed_size"] == 0, res
    assert res["next_free_vgpr"] <= spec.VGPR_MAX, res
    assert res["group_segment_fixed_size"] <= spec.LDS_BYTES, res
    assert res["group_segment_fixed_size"] <= layer.lds_bytes(s), res
    # the same LDS as the tensor build of the width (the core rows are back's)
    tensor = _baseline()["builds"][spec.K2Build(tokens=s).kernel_name()]
    assert (
        res["group_segment_fixed_size"]
        == tensor["resources"]["group_segment_fixed_size"]
    )
    loops = tc._WATERFALL.findall(txt)
    assert not loops, f"{len(loops)} waterfall compares, e.g. {loops[0]}"
    for op in (
        "v_mfma_f32_16x16x32_bf16",
        "v_mfma_scale_f32_16x16x128_f8f6f4",
        "v_sqrt_f32",  # the gate's raw sqrt
        "v_div_fixup_f32",
        "v_permlane32_swap",  # the served gate's butterfly (tl_sum64)
        "v_permlane16_swap",
        "row_shr:8",
        "row_shr:4",
    ):
        assert op in txt, op
    # the gate's numerics (DESIGN_BC 1.4): one raw v_sqrt_f32 a row (the
    # tensor build has none) and the IEEE divides of rstd and the sigmoid (an
    # approximate lowering would drop their v_div_fixup_f32)
    assert len(re.findall(r"^\s*v_sqrt_f32", txt, re.M)) == gate.passes(s)
    assert len(re.findall(r"^\s*v_div_fixup_f32", txt, re.M)) >= 2 * gate.passes(s)
    # the core loads: x and g a pass, 2 bytes a lane; a 2-byte LDS write a pass
    assert len(_CORE_LOAD.findall(txt)) == 2 * gate.passes(s)
    assert len(_XL_WRITE.findall(txt)) == gate.passes(s)
    first_core, first_nt, last_xl = _window(txt)
    assert first_core < first_nt < last_xl
    assert not _CORE_LOAD.search(txt, first_nt), "a core load after the weights"
    assert not _XL_WRITE.search(txt, 0, first_nt), "an xl write before the weights"
    # every o_proj weight load is issued before the gate's first write
    first_xl = _XL_WRITE.search(txt).start()
    assert len(_NT_LOAD.findall(txt, first_nt, first_xl)) == O_LOADS
    # the weights are not held back behind the core operands (nor the pads)
    bad = _waits_before_weights(txt, first_core, first_nt)
    assert not bad, bad
    # rule 8 (6.2's ISA pin): no wait in the gate retires a weight load (in
    # order: vmcnt(n) with n < O_LOADS would; vmcnt(0) above all)
    waits = [int(n) for n in _VMCNT.findall(txt, first_nt, last_xl)]
    assert waits and min(waits) >= O_LOADS, waits


@compiles
@pytest.mark.parametrize("tokens", [1, 8])
def test_gated_trace_is_the_program_order(tokens, tmp_path, monkeypatch):
    """The gated build's trace is Phase A's: the stages in order (the gated
    core load is o_proj's first stage), a barrier after every LDS stage, every
    access named and declared, every K2 region put and polled."""
    seen = []  # the stage count when each barrier was traced
    real = fx_gpu.barrier

    def barrier(*a, **k):
        rec = plan._record
        seen.append(len(rec.stages) if rec is not None else -1)
        return real(*a, **k)

    monkeypatch.setattr(fx_gpu, "barrier", barrier)
    with plan.recording() as rec:
        _compile(spec.K2Build(tokens=tokens, core="gated"), tmp_path, monkeypatch)
    assert tuple(rec.stages) == layer.STAGES, rec.stages
    assert not rec.unnamed, rec.unnamed
    for i, name in enumerate(rec.stages[:-1]):
        if name not in tc.NO_LDS:
            assert i + 1 in seen, f"no barrier between {name} and {rec.stages[i + 1]}"
    puts: set = set()
    polls: set = set()
    for a in rec.accesses:
        mod = tc.STAGE_MODULES[a.stage]
        assert a.region in mod.STAGE_REGIONS, (a.stage, a.region)
        (puts if a.kind == "put" else polls).add(a.region)
    regions = set(spec.K2_REGIONS)
    assert puts == regions and polls == regions, (puts, polls)


@compiles
@pytest.mark.parametrize("tokens", [1, 8])
def test_tensor_build_unchanged_after_gated(tokens, tmp_path, monkeypatch):
    """A Phase A build compiled after a gated build of its width, in one
    process (shared caches: the build table, the cached LDS structs), still
    has the Phase A final ISA."""
    base = _baseline()
    if tc.toolchain() != base["toolchain"]:
        pytest.skip(f"the baseline pins {base['toolchain']}: see test_compile")
    _compile(spec.K2Build(tokens=tokens, core="gated"), tmp_path / "g", monkeypatch)
    key = spec.K2Build(tokens=tokens)
    txt, res = _compile(key, tmp_path / "t", monkeypatch)
    rec = base["builds"][key.kernel_name()]
    assert {k: res[k] for k in rec["resources"]} == rec["resources"]
    assert tc.isa_digest(txt, key.kernel_name()) == rec["sha256"]


def test_gated_and_tensor_builds_are_distinct():
    """The two variants of a width are two cached launchers (``build`` traces
    nothing until the first call) under two symbols."""
    tk, gk = spec.K2Build(tokens=2), spec.K2Build(tokens=2, core="gated")
    t, g = layer.build(tk), layer.build(gk)
    assert t is not g and layer.build(gk) is g
    assert tk.kernel_name() + "_g" == gk.kernel_name()
