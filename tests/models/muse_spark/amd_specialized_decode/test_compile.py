# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2 (``mono/layer.py``) compiles for gfx950 without a GPU, every build the
runtime can launch: widths 1 .. 8, a fault build (``sync.stall`` in the
prologue) and a debug build (CTA 0 dumps its a1 rows).

The ISA gate (design 5.13): no private segment (a spilling persistent kernel
can lose residency and deadlock), at most 256 VGPRs (2 waves a SIMD), the LDS
of one CU. And the trace: the stages run in the design's program order (5.2),
every stage that touches the stage LDS union or ``red`` ends with a barrier
before the next stage starts (the hand-over rule), every mailbox access names
a region its stage module declares, and every region is both put and polled.

Phases B and C must leave every Phase A build bit-identical:
``test_k2_tensor_isa_unchanged`` compares each ``core="tensor"`` build's final
ISA with ``k2_isa_baseline.json``, recorded from the Phase A sources (head
ca9640e2e2) before any B / C kernel edit. Regenerating it is only valid from
those sources (the file's ``procedure``): running this module as a script,
``python3 test_compile.py --write-baseline``, records the tree it runs on.

K1 (``mono/k1.py``, Phase B1, DESIGN_BC 5) compiles for gfx950 the same way,
every build the runtime and the tests launch (widths 1 .. 8 x RoPE / NoPE, the
fault and debug builds), within the same gate plus one CTA a CU (5.9: more
than 128 VGPRs or more than half a CU's LDS). Its final ISA keeps rule 8 (no
poll or publish with another task's load in flight, DESIGN_BC 10.4) in each
role's code: the boundary's operands and the epilogue's step inputs before
the first poll, only the cos / sin row between the boundary's publish and
the epilogue's poll, no plain load past it, every qkv weight before the GEMV's
``ardy`` poll and the staging loads before its first LDS write; and the
epilogue's 1.3 arithmetic (one ``v_rsq_f32``, the serial ``row_shr:1``
square chain with no fma, a ``v_sub_f32`` of two ``v_mul_f32`` on RoPE
builds).
"""

import dataclasses
import glob
import hashlib
import importlib.metadata
import inspect
import json
import os
import sys
from pathlib import Path

import pytest
import regex as re
import torch

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

import flydsl.expr.gpu as fx_gpu  # noqa: E402

from tests.models.muse_spark.amd_specialized_decode import (  # noqa: E402
    test_probe_qkv as tq,
)
from vllm.models.muse_spark.amd.specialized_decode.mono import (  # noqa: E402
    layer,
    spec,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.attention import (  # noqa: E402
    back,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common import (  # noqa: E402
    context,
    plan,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.sync import (  # noqa: E402
    WAIT_TICKS,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.stages import (  # noqa: E402
    experts,
    pool,
    reduce,
    router,
    seam,
    unpool,
)


def _on_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


def _compiles_for_gfx950() -> bool:
    """A GPU-less compile must target gfx950: FlyDSL falls back to gfx942
    without a device and then fails to select gfx950's permlane swaps."""
    arch = os.environ.get("FLYDSL_GPU_ARCH") or os.environ.get("ARCH")
    return arch == "gfx950" or _on_gfx950()


compiles = pytest.mark.skipif(
    not _compiles_for_gfx950(),
    reason="GPU-less compile needs FLYDSL_GPU_ARCH=gfx950 (or a gfx950 GPU)",
)

# every build the runtime and the tests launch: the widths, the stalled-rank
# fault builds (test_numerics: S = 8; test_engine_fault: the S = 1 width) and
# the a1 dump of test_numerics
FAULT = dict(fault_rank=3, fault_epoch=2, fault_slot=spec.k2_slot(1))
BUILDS = [spec.K2Build(tokens=s) for s in range(1, spec.MAX_ROWS + 1)] + [
    spec.K2Build(tokens=8, fault_ticks=WAIT_TICKS * 3 // 2, **FAULT),
    spec.K2Build(tokens=1, fault_ticks=WAIT_TICKS * 3 // 2, **FAULT),
    spec.K2Build(tokens=8, debug_dump=True),
]

# The Phase A ISA baseline (DESIGN_BC 9.1): kernel name -> key, sha256, resources
BASELINE = Path(__file__).with_name("k2_isa_baseline.json")
# the packages whose code generation the hashes pin
TOOLCHAIN = ("flydsl", "amd-aiter")

# a 64-bit compare of a VGPR buffer resource against its first lane's: a
# readfirstlane waterfall around a load whose base is not wave-uniform
_WATERFALL = re.compile(r"v_cmp_eq_u64_e\d+\s+\S+,\s*s\[\d+:\d+\],\s*v\[")

# the stage modules, by the names their ``plan.enter_stage`` records
STAGE_MODULES = {
    "o_proj": back,
    "seam": seam,
    "route": router,
    "router": router,
    "pool": pool,
    "xq": pool,
    "load_route": experts,
    "ug": experts,
    "load_inter": experts,
    "down": experts,
    "owner": reduce,
    "unpool": unpool,
    "gather": unpool,
}
# stages that touch neither the stage LDS union nor ``red`` (no hand-over
# barrier is owed after them)
NO_LDS = {"seam", "gather"}


def _isa(dump: str, name: str) -> tuple[str, dict]:
    """The final ISA of kernel ``name`` under the dump dir and its resources."""
    for path in glob.glob(f"{dump}/**/*final_isa.s", recursive=True):
        with open(path) as f:
            txt = f.read()
        if re.search(rf"\.amdhsa_kernel {re.escape(name)}\b", txt):
            res = {}
            for k in (
                "next_free_vgpr",
                "accum_offset",
                "next_free_sgpr",
                "group_segment_fixed_size",
                "private_segment_fixed_size",
            ):
                m = re.search(rf"\.amdhsa_{k}\s+(\d+)", txt)
                res[k] = int(m.group(1)) if m else None
            return txt, res
    raise AssertionError(f"no final ISA of {name} under {dump}")


def isa_digest(txt: str, name: str) -> str:
    """sha256 of a final ISA with its kernel symbol ``name`` written
    ``<kernel>`` (every line naming it: the label, ``.globl``, ``.amdhsa_kernel``,
    the resource ``.set``s, the metadata), so the hash pins the code and its
    resources, not the symbol."""
    norm = re.sub(rf"\b{re.escape(name)}\b", "<kernel>", txt)
    return hashlib.sha256(norm.encode()).hexdigest()


def toolchain() -> dict[str, str]:
    return {p: importlib.metadata.version(p) for p in TOOLCHAIN}


def _baseline_builds() -> list[spec.K2Build]:
    if not BASELINE.exists():
        return []
    return [
        spec.K2Build(**b["key"])
        for b in json.loads(BASELINE.read_text())["builds"].values()
    ]


def _compile(key: spec.K2Build, dump, monkeypatch) -> tuple[str, dict]:
    """``build(key)`` then its launcher on ``ABI.zeros()`` in FlyDSL's
    compile-only mode (what ``runtime.layers.compile_build`` does, without the
    current HIP stream it passes), the IR dumped under ``dump``."""
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(dump))
    monkeypatch.setenv("COMPILE_ONLY", "1")
    layer._BUILDS.pop(key, None)  # trace it again (a test may record the trace)
    launch = layer.build(key)
    try:
        launch(*layer.ABI.zeros())
    except RuntimeError as e:  # the load, past the compile, needs a device
        if "hipErrorNoDevice" not in str(e) and "no ROCm" not in str(e):
            raise
    return _isa(str(dump), key.kernel_name())


# ---------------------------------------------------------------- [CPU]


def test_abi_is_the_k2_arguments():
    """The kernel's parameters are spec.K2_ARGS in order (the residual stream
    renamed ``stream_in``: the launcher's own ``stream`` is the HIP stream);
    ``k2_launch`` packs by the K2_ARGS names, tensors by address."""
    names = list(layer.ABI.names)
    assert names == [layer.KERNEL_NAMES.get(n, n) for n in spec.K2_ARGS]
    assert "stream" not in names and names[spec.K2_ARGS.index("stream")] == (
        "stream_in"
    )
    assert len(layer.ABI.zeros()) == len(spec.K2_ARGS)


def test_k2_launch_packs_by_name(monkeypatch):
    calls = []

    def fake_build(key):
        return lambda *a, stream: calls.append((key, a, stream))

    monkeypatch.setattr(layer, "build", fake_build)
    monkeypatch.setattr(layer.torch.cuda, "current_stream", lambda: "cur")
    t = {n: torch.empty(4, dtype=torch.uint8) for n in spec.K2_ARGS}
    args = {**t, "rank": 5, "slot": spec.k2_slot(7), "alarm": 1234}
    key = spec.K2Build(tokens=3)
    layer.k2_launch(key, args)
    [(got_key, packed, stream)] = calls
    assert got_key is key and stream == "cur"
    for name, v in zip(spec.K2_ARGS, packed):
        want = args[name]
        assert v == (want.data_ptr() if isinstance(want, torch.Tensor) else want)
    with pytest.raises(TypeError, match="missing"):
        layer.k2_launch(key, {n: v for n, v in args.items() if n != "pad"})
    with pytest.raises(TypeError, match="unexpected"):
        layer.k2_launch(key, {**args, "out": 0})


def test_lds_fits_every_width():
    """The stage union (design 5.13) at every width: the composite ``xq_ug``
    holds the a1 rows and the ug epilogue's buffer; the total fits a CU."""
    for s in range(1, spec.MAX_ROWS + 1):
        lds = layer.lds_structs(s)
        assert set(lds) == {"back", "seam", "pool", "xq_ug", "down", "reduce", "unpool"}
        assert set(lds["xq_ug"]) == {"xq", "ug"}
        assert layer.lds_bytes(s) <= spec.LDS_BYTES, (s, layer.lds_bytes(s))
    sizes = context.stage_bytes(layer.lds_structs(8))
    assert sizes["xq_ug"] == 24576 + 2048 and sizes["pool"] == 32896, sizes
    assert max(sizes.values()) == sizes["pool"]


def test_build_refuses_a_foreign_key():
    with pytest.raises(TypeError):
        layer.build((8,))


# ---------------------------------------------------------------- [CPU] FlyDSL


@compiles
@pytest.mark.parametrize("key", BUILDS, ids=lambda k: k.kernel_name())
def test_k2_builds_compile_without_a_gpu(key, tmp_path, monkeypatch):
    """Every build passes the ISA gate; its symbol is the build's name."""
    txt, res = _compile(key, tmp_path, monkeypatch)
    print(f"{key.kernel_name()}: {res}")
    assert res["private_segment_fixed_size"] == 0, res
    assert res["next_free_vgpr"] <= spec.VGPR_MAX, res
    assert res["group_segment_fixed_size"] <= spec.LDS_BYTES, res
    assert res["group_segment_fixed_size"] <= layer.lds_bytes(key.tokens), res
    # the stages are all there: o_proj / pooler / unpooler (bf16 MFMA), ug /
    # down (scaled FP4 MFMA), the a1 / a2 packs, the routing's IEEE divide
    for op in (
        "v_mfma_f32_16x16x32_bf16",
        "v_mfma_scale_f32_16x16x128_f8f6f4",
        "v_cvt_scalef32_pk_fp4_bf16",
        "v_cvt_scalef32_pk_fp4_f32",
        "v_div_fixup_f32",
        "s_memrealtime",
        "s_barrier",
    ):
        assert op in txt, op
    # the 22-bit epoch mask (the int32 epoch may wrap), folded into the tag's
    # (e & 0x3FFFFF) * 256 as a mask of e << 8
    assert re.search(r"0x3fffff(00)?\b", txt, re.IGNORECASE)
    # every mailbox / weight load has a wave-uniform buffer resource: no
    # readfirstlane waterfall (``experts.load_inter``'s rule; the seam's and the
    # owner's polls after a lane-predicated put use ``sync.uniform_addr``)
    loops = _WATERFALL.findall(txt)
    assert not loops, f"{len(loops)} waterfall compares, e.g. {loops[0]}"


@compiles
def test_compile_build_runs_without_a_gpu(tmp_path, monkeypatch):
    """The runtime's warm-up compile (``runtime.layers.compile_build``) also
    works GPU-less (FlyDSL's default stream when there is no device) and
    restores ``COMPILE_ONLY``."""
    from vllm.models.muse_spark.amd.specialized_decode.runtime import layers

    key = spec.K2Build(tokens=1)
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    monkeypatch.delenv("COMPILE_ONLY", raising=False)
    layer._BUILDS.pop(key, None)
    try:
        layers.compile_build(key)
    except RuntimeError as e:  # the load, past the compile, needs a device
        if "hipErrorNoDevice" not in str(e) and "no ROCm" not in str(e):
            raise
    assert "COMPILE_ONLY" not in os.environ
    _, res = _isa(str(tmp_path), key.kernel_name())
    assert res["private_segment_fixed_size"] == 0, res


@compiles
def test_fault_and_debug_builds_add_their_code(tmp_path, monkeypatch):
    """The fault build compiles ``sync.stall`` in (a clock spin of its own);
    the debug build stores CTA 0's a1 rows; the plain builds carry neither."""
    plain, _ = _compile(spec.K2Build(tokens=8), tmp_path / "p", monkeypatch)
    fault, _ = _compile(BUILDS[spec.MAX_ROWS], tmp_path / "f", monkeypatch)
    debug, _ = _compile(BUILDS[-1], tmp_path / "d", monkeypatch)

    def count(txt, pat):
        return len(re.findall(pat, txt))

    clock = r"s_memrealtime"
    assert count(fault, clock) > count(plain, clock)
    stores = r"\b(buffer|global)_store_"
    assert count(debug, stores) > count(plain, stores)


@compiles
@pytest.mark.parametrize("tokens", [1, 8])
def test_trace_is_the_program_order(tokens, tmp_path, monkeypatch):
    """The trace of one build: the stages in 5.2's order, a barrier after every
    LDS stage before the next stage enters, every access named and declared by
    its stage module, every region put and polled."""
    seen = []  # the stage count when each barrier was traced
    real = fx_gpu.barrier

    def barrier(*a, **k):
        rec = plan._record
        seen.append(len(rec.stages) if rec is not None else -1)
        return real(*a, **k)

    monkeypatch.setattr(fx_gpu, "barrier", barrier)
    with plan.recording() as rec:
        _compile(spec.K2Build(tokens=tokens), tmp_path, monkeypatch)
    assert tuple(rec.stages) == layer.STAGES, rec.stages
    assert not rec.unnamed, rec.unnamed
    # stage i entered at count i + 1: a barrier traced at count i + 1 runs
    # after stage i's entry and before stage i + 1's
    for i, name in enumerate(rec.stages[:-1]):
        if name not in NO_LDS:
            assert i + 1 in seen, f"no barrier between {name} and {rec.stages[i + 1]}"
    puts: set = set()
    polls: set = set()
    for a in rec.accesses:
        mod = STAGE_MODULES[a.stage]
        assert a.region in mod.STAGE_REGIONS, (a.stage, a.region)
        space = (
            plan.Space.PEER if a.region in spec.PEER_REGIONS else (plan.Space.SCRATCH)
        )
        assert a.space == space, a
        (puts if a.kind == "put" else polls).add(a.region)
    # (K2's regions; Phase B's K1 and Phase C's split core add their own)
    regions = set(spec.K2_REGIONS)
    assert puts == regions and polls == regions, (puts, polls)


@compiles
@pytest.mark.parametrize("key", _baseline_builds(), ids=lambda k: k.kernel_name())
def test_k2_tensor_isa_unchanged(key, tmp_path, monkeypatch):
    """Every Phase A build (``core="tensor"``) still compiles to the final ISA
    the Phase A sources did (the symbol normalized): Phases B and C change
    nothing a Phase A launch runs (DESIGN_BC 5.2, 5.4, 6, 7)."""
    base = json.loads(BASELINE.read_text())
    now = toolchain()
    if now != base["toolchain"]:
        pytest.skip(
            f"the baseline pins {base['toolchain']}, this image has {now}: "
            f"regenerate it per its procedure"
        )
    rec = base["builds"][key.kernel_name()]
    assert key.core == "tensor" and not key.window
    txt, res = _compile(key, tmp_path, monkeypatch)
    assert {k: res[k] for k in rec["resources"]} == rec["resources"], key
    assert isa_digest(txt, key.kernel_name()) == rec["sha256"], (
        f"{key.kernel_name()}: the final ISA changed from the Phase A build"
    )


def test_isa_baseline_covers_every_phase_a_build():
    """The baseline holds exactly this module's Phase A builds (``BUILDS``),
    each keyed by its kernel name, with the Phase A resources."""
    base = json.loads(BASELINE.read_text())
    assert set(base) >= {"procedure", "source", "image", "toolchain", "builds"}
    assert set(TOOLCHAIN) == set(base["toolchain"])
    assert sorted(_baseline_builds(), key=str) == sorted(BUILDS, key=str)
    for name, rec in base["builds"].items():
        assert spec.K2Build(**rec["key"]).kernel_name() == name
        assert len(rec["sha256"]) == 64
        assert rec["resources"]["private_segment_fixed_size"] == 0
        assert rec["resources"]["next_free_vgpr"] <= spec.VGPR_MAX


def _write_baseline(out: Path, source: str, image: str) -> None:
    """Records every ``BUILDS`` entry's final-ISA digest and resources, compiled
    GPU-less from the tree this module imports ``vllm`` from."""
    builds = {}
    for key in BUILDS:
        name = key.kernel_name()
        dump = Path(os.environ.get("TMPDIR", "/tmp")) / f"k2_isa_baseline/{name}"
        os.environ.update(
            FLYDSL_RUNTIME_ENABLE_CACHE="0",
            FLYDSL_DUMP_IR="1",
            FLYDSL_DUMP_DIR=str(dump),
            COMPILE_ONLY="1",
        )
        layer._BUILDS.pop(key, None)
        try:
            layer.build(key)(*layer.ABI.zeros())
        except RuntimeError as e:
            if "hipErrorNoDevice" not in str(e) and "no ROCm" not in str(e):
                raise
        txt, res = _isa(str(dump), name)
        builds[name] = {
            "key": dataclasses.asdict(key),
            "sha256": isa_digest(txt, name),
            "resources": res,
        }
        print(name, builds[name]["sha256"], res, flush=True)
    base = json.loads(out.read_text()) if out.exists() else {}
    base.update(source=source, image=image, toolchain=toolchain(), builds=builds)
    out.write_text(json.dumps(base, indent=1, sort_keys=True) + "\n")


# ---------------------------------------------------------------- K1 (Phase B1)

# every K1 build the runtime and the tests launch: the widths of both kinds,
# the stalled-rank fault builds of a full layer's K1 (test_numerics_front: S =
# 8; test_engine_fault: the S = 1 width) and the GEMV-row dumps of
# test_numerics_front
K1_FAULT = dict(fault_rank=3, fault_epoch=2, fault_slot=spec.k1_slot(1))
K1_BUILDS = [
    spec.K1Build(tokens=s, rope=rope)
    for s in range(1, spec.MAX_ROWS + 1)
    for rope in (False, True)
] + [
    spec.K1Build(tokens=8, rope=False, fault_ticks=WAIT_TICKS * 3 // 2, **K1_FAULT),
    spec.K1Build(tokens=1, rope=False, fault_ticks=WAIT_TICKS * 3 // 2, **K1_FAULT),
    spec.K1Build(tokens=8, rope=True, debug_dump=True),
    spec.K1Build(tokens=8, rope=False, debug_dump=True),
]

# a poll's back-off (``sync._spin``; the fault hook's stall sleeps 127)
_POLL_SLEEP = re.compile(r"^\s*s_sleep 1\b", re.M)
_BARRIER = re.compile(r"^\s*s_barrier\b", re.M)
_ROW_SHR1 = re.compile(r"v_add_f32_dpp[^\n]*row_shr:1\b")
_STORE_SHORT = re.compile(r"^\s*buffer_store_short\b", re.M)
_DS_WRITE4 = re.compile(r"^\s*ds_write_b128\b", re.M)


def _compile_k1(key: spec.K1Build, dump, monkeypatch) -> tuple[str, dict]:
    """``k1.build(key)`` then its launcher on ``k1.ABI.zeros()`` in FlyDSL's
    compile-only mode (``runtime.layers.compile_build``'s call), the IR
    dumped under ``dump``."""
    from vllm.models.muse_spark.amd.specialized_decode.mono import k1

    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(dump))
    monkeypatch.setenv("COMPILE_ONLY", "1")
    k1._BUILDS.pop(key, None)  # trace it again (a test may record the trace)
    launch = k1.build(key)
    try:
        launch(*k1.ABI.zeros())
    except RuntimeError as e:  # the load, past the compile, needs a device
        if "hipErrorNoDevice" not in str(e) and "no ROCm" not in str(e):
            raise
    return _isa(str(dump), key.kernel_name())


def test_k1_abi_is_the_k1_arguments():
    """K1's parameters are spec.K1_ARGS in order (the residual stream renamed
    ``stream_in``), ``K1_INT32_ARGS`` typed Int32 and every other one an Int64
    address or byte stride, in the kernel and its launcher."""
    from flydsl.expr.typing import Int32, Int64

    from vllm.models.muse_spark.amd.specialized_decode.mono import k1

    names = list(k1.ABI.names)
    assert names == [k1.KERNEL_NAMES.get(n, n) for n in spec.K1_ARGS]
    assert "stream" not in names and "peers" not in names
    assert len(k1.ABI.zeros()) == len(spec.K1_ARGS) == 25
    launch = k1.build(spec.K1Build(tokens=2, rope=True))
    params = inspect.signature(launch.func).parameters
    for name, arg in zip(spec.K1_ARGS, names):
        want = Int32 if name in spec.K1_INT32_ARGS else Int64
        assert params[arg].annotation is want, (name, params[arg].annotation)
    assert k1.build(spec.K1Build(tokens=2, rope=True)) is launch  # cached


def test_k1_launch_packs_by_name(monkeypatch):
    from vllm.models.muse_spark.amd.specialized_decode.mono import k1

    calls = []

    def fake_build(key):
        return lambda *a, stream: calls.append((key, a, stream))

    monkeypatch.setattr(k1, "build", fake_build)
    monkeypatch.setattr(k1.torch.cuda, "current_stream", lambda: "cur")
    t = {n: torch.empty(4, dtype=torch.uint8) for n in spec.K1_ARGS}
    ints = {"kv_sb": 1 << 33, "kv_sh": 8192, "kv_st": 256, "kv_blocks": 7}
    args = {**t, **ints, "cos_sin": 0, "max_pos": 0, "rank": 5, "slot": 3}
    args["alarm"] = 1234
    key = spec.K1Build(tokens=3, rope=False)
    k1.k1_launch(key, args)
    [(got_key, packed, stream)] = calls
    assert got_key is key and stream == "cur"
    for name, v in zip(spec.K1_ARGS, packed):
        want = args[name]
        assert v == (want.data_ptr() if isinstance(want, torch.Tensor) else want)
    with pytest.raises(TypeError, match="missing"):
        k1.k1_launch(key, {n: v for n, v in args.items() if n != "slots"})
    with pytest.raises(TypeError, match="unexpected"):
        k1.k1_launch(key, {**args, "peers": 0})


def test_k1_lds_holds_one_cta_a_cu():
    """The stage union is the GEMV's staged rows (5.9: 12 S padded K part
    rows) and the residency pin: every width holds more than half a CU's LDS
    (one CTA a CU), within a CU's; the pin decides it at S <= 5 only."""
    from vllm.models.muse_spark.amd.specialized_decode.mono import k1
    from vllm.models.muse_spark.amd.specialized_decode.mono.front import qkv

    for s in range(1, spec.MAX_ROWS + 1):
        lds = k1.lds_structs(s)
        assert set(lds) == {qkv.STAGE, "pin"}
        rows = context.lds_total_bytes({qkv.STAGE: qkv.lds_struct(s)}, False)
        assert rows == 8192 + spec.K1_TPC * s * qkv.XA_ROW * 2
        got = k1.lds_bytes(s)
        assert spec.LDS_BYTES // 2 < got <= spec.LDS_BYTES, (s, got)
        assert got == max(rows, k1.RESIDENT_LDS), (s, got)
        assert (rows > spec.LDS_BYTES // 2) == (s >= 6), s


def test_k1_build_refuses_a_foreign_key():
    from vllm.models.muse_spark.amd.specialized_decode.mono import k1

    with pytest.raises(TypeError, match="K1Build"):
        k1.build(spec.K2Build(tokens=1))


def test_k1_builds_are_the_runtime_keys():
    """The builds compiled here are the runtime's (``KernelLayer.k1_key``:
    tokens, rope = sliding) and the tests' fault / dump builds; names are
    distinct."""
    names = [k.kernel_name() for k in K1_BUILDS]
    assert len(set(names)) == len(names)
    plain = {(k.tokens, k.rope) for k in K1_BUILDS if not (k.faulty or k.debug_dump)}
    assert plain == {(s, r) for s in range(1, spec.MAX_ROWS + 1) for r in (False, True)}
    for k in K1_BUILDS:
        if k.faulty:
            assert k.fault_slot == spec.k1_slot(1) and not k.rope  # layer 1: full


def _k1_layout(txt: str) -> dict:
    """The positions of K1's four polls and its weight stream in a final ISA:
    the boundary's ``fss`` and ``nss``, the epilogue's ``qpart``, the GEMV's
    ``ardy`` (LLVM lays the roles out in program order: the boundary CTAs'
    branch, then the GEMV CTAs'), the first and the last nontemporal weight
    load."""
    polls = [m.start() for m in _POLL_SLEEP.finditer(txt)]
    assert len(polls) == 4, f"{len(polls)} polls (fss, nss, qpart, ardy)"
    nt = [m.start() for m in tq._LOAD_NT.finditer(txt)]
    assert len(nt) == 4 * spec.K1_ROUNDS, len(nt)  # 6 rounds x 4 dwordx4
    fss, nss, qpart, ardy = polls
    assert qpart < nt[0] and nt[-1] < ardy, "roles out of program order"
    return {"fss": fss, "nss": nss, "qpart": qpart, "ardy": ardy, "nt": nt}


@compiles
@pytest.mark.parametrize("key", K1_BUILDS, ids=lambda k: k.kernel_name())
def test_k1_builds_compile_without_a_gpu(key, tmp_path, monkeypatch):
    """The 5.9 gate (no private segment, <= 256 VGPRs, a CU's LDS, one CTA a
    CU, no readfirstlane waterfall) on every build; rule 8 in each role's
    code; the epilogue's 1.3 arithmetic."""
    from vllm.models.muse_spark.amd.specialized_decode.mono import k1
    from vllm.models.muse_spark.amd.specialized_decode.mono.front import qkv

    txt, res = _compile_k1(key, tmp_path, monkeypatch)
    print(f"{key.kernel_name()}: {res}")
    assert res["private_segment_fixed_size"] == 0, res
    assert res["next_free_vgpr"] <= spec.VGPR_MAX, res
    assert res["group_segment_fixed_size"] == k1.lds_bytes(key.tokens), res
    assert res["group_segment_fixed_size"] <= spec.LDS_BYTES, res
    # one CTA a CU (5.9): two would need 2 x 4 waves a SIMD or 2 x the LDS
    assert (
        res["next_free_vgpr"] > 128
        or res["group_segment_fixed_size"] > spec.LDS_BYTES // 2
    ), res
    loops = _WATERFALL.findall(txt)
    assert not loops, f"{len(loops)} waterfall compares, e.g. {loops[0]}"
    lay = _k1_layout(txt)
    rope = key.rope
    # -- the boundary CTAs: the 7 boundary operands and the epilogue's 3 step
    # inputs before the first poll; past it, before the epilogue's poll, only
    # the cos / sin row, after the slice's publish (its barrier drains
    # everything the CTA issued)
    plain = [m.start() for m in tq._PLAIN_LOAD.finditer(txt)]
    assert len([p for p in plain if p < lay["fss"]]) >= 10
    mid = [p for p in plain if lay["fss"] < p < lay["qpart"]]
    publish = [m.start() for m in _BARRIER.finditer(txt, 0, lay["qpart"])][-1]
    assert len(mid) == (2 if rope else 0), len(mid)
    assert all(p > publish for p in mid)
    front = tq._code(txt[: lay["qpart"]])
    assert sum(1 for x in front if x.startswith("v_rsq_f32")) == 2  # two norms
    assert not tq._fmas(front), tq._fmas(front)
    # -- the epilogue: no plain load past its poll; 1.3 steps 2-6
    epi_txt = txt[lay["qpart"] : lay["nt"][0]]
    assert not tq._PLAIN_LOAD.findall(epi_txt)
    epi = tq._code(epi_txt)
    assert sum(1 for x in epi if x.startswith("v_rsq_f32")) == 1
    assert len(_ROW_SHR1.findall(epi_txt)) == 7  # the serial square chain
    fmas = tq._fmas(epi)
    # the eps fma (and RoPE's 3): the squares and their sums are not fused
    assert len(fmas) == (4 if rope else 1), fmas
    assert tq._sub_of_two_muls(epi) == rope
    assert ("quad_perm:[1,0,3,2]" in epi_txt) == rope
    shorts = len(_STORE_SHORT.findall(epi_txt))  # qkvo, the cache (qkvd)
    assert shorts == (3 if key.debug_dump else 2), shorts
    assert len(_STORE_SHORT.findall(txt)) == shorts
    # -- the GEMV CTAs: every weight before the ardy poll (its vmcnt(0) waits
    # for them, the CTA's only loads), none after; the staging loads all
    # before the first staging ds_write; 6 rounds x 4 MFMAs past the poll
    gemv = txt[lay["ardy"] :]
    first_mfma = tq._MFMA.search(gemv).start()
    stage = [m.start() for m in tq._LOAD_DEV.finditer(gemv, 0, first_mfma)]
    assert len(stage) == -(-spec.K1_TPC * key.tokens * qkv.X4 // spec.THREADS)
    writes = [m.start() for m in _DS_WRITE4.finditer(gemv, 0, first_mfma)]
    assert writes and stage[-1] < writes[0]
    assert len(tq._MFMA.findall(gemv)) == len(tq._MFMA.findall(txt)) == 24
    assert not tq._PLAIN_LOAD.findall(gemv)


@compiles
def test_k1_compile_build_runs_without_a_gpu(tmp_path, monkeypatch):
    """The runtime's warm-up compile of a K1 key (``compile_build``: mono.k1,
    k1.ABI) works GPU-less and restores ``COMPILE_ONLY``."""
    from vllm.models.muse_spark.amd.specialized_decode.mono import k1
    from vllm.models.muse_spark.amd.specialized_decode.runtime import layers

    key = spec.K1Build(tokens=1, rope=True)
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    monkeypatch.delenv("COMPILE_ONLY", raising=False)
    k1._BUILDS.pop(key, None)
    assert layers._kernel_module(key) == (k1, k1.ABI)
    try:
        layers.compile_build(key)
    except RuntimeError as e:  # the load, past the compile, needs a device
        if "hipErrorNoDevice" not in str(e) and "no ROCm" not in str(e):
            raise
    assert "COMPILE_ONLY" not in os.environ
    _, res = _isa(str(tmp_path), key.kernel_name())
    assert res["private_segment_fixed_size"] == 0, res


@compiles
def test_k1_fault_and_debug_builds_add_their_code(tmp_path, monkeypatch):
    """The fault build compiles ``sync.stall`` in; the debug build stores
    the GEMV rows (``qkvd``); the plain build carries neither."""
    plain, _ = _compile_k1(
        spec.K1Build(tokens=8, rope=False), tmp_path / "p", monkeypatch
    )
    fault, _ = _compile_k1(K1_BUILDS[-4], tmp_path / "f", monkeypatch)
    debug, _ = _compile_k1(K1_BUILDS[-1], tmp_path / "d", monkeypatch)
    clock = r"s_memrealtime"
    assert len(re.findall(clock, fault)) > len(re.findall(clock, plain))
    assert len(_STORE_SHORT.findall(debug)) == len(_STORE_SHORT.findall(plain)) + 1


@compiles
@pytest.mark.parametrize("rope", [False, True])
def test_k1_trace_is_the_program_order(rope, tmp_path, monkeypatch):
    """The trace of one build: the boundary, the epilogue, the GEMV (5.3);
    every access named and declared by its stage module; every K1 region put
    and polled, in scratch only (K1 is rank-local)."""
    from vllm.models.muse_spark.amd.specialized_decode.mono import k1

    with plan.recording() as rec:
        _compile_k1(spec.K1Build(tokens=3, rope=rope), tmp_path, monkeypatch)
    assert tuple(rec.stages) == k1.STAGES, rec.stages
    assert not rec.unnamed, rec.unnamed
    puts: set = set()
    polls: set = set()
    for a in rec.accesses:
        assert a.region in k1.STAGE_MODULES[a.stage].STAGE_REGIONS, a
        assert a.space == plan.Space.SCRATCH, a
        (puts if a.kind == "put" else polls).add(a.region)
    regions = set(spec.K1_REGIONS)
    assert puts == regions and polls == regions, (puts, polls)


# ---------------------------------------------------------------- K1 tuning variants

# TUNE L1 (the opt-in ``K1Build.layout``; the defaults above are unchanged):
# the r256 task layout (DESIGN_BC 11.3 rung 3, M2's 10/6 rebalance) at every
# width and kind; r256's debug and fault builds (the tests' numerics and
# stall runs)
K1_VARIANT_LAYOUTS = ("r256",)
K1_VARIANTS = [
    spec.K1Build(tokens=s, rope=rope, layout=lay)
    for lay in K1_VARIANT_LAYOUTS
    for s in range(1, spec.MAX_ROWS + 1)
    for rope in (False, True)
] + [
    spec.K1Build(tokens=8, rope=True, debug_dump=True, layout="r256"),
    spec.K1Build(tokens=8, rope=False, debug_dump=True, layout="r256"),
    spec.K1Build(tokens=1, rope=True, debug_dump=True, layout="r256"),
    spec.K1Build(
        tokens=8,
        rope=False,
        fault_ticks=WAIT_TICKS * 3 // 2,
        layout="r256",
        **K1_FAULT,
    ),
]


def test_k1_variant_keys_and_lds():
    """Every variant has its own symbol (the defaults keep theirs), every
    width holds more than half a CU's LDS (one CTA a CU), and r256's GEMV
    CTAs stage 10 K parts."""
    from vllm.models.muse_spark.amd.specialized_decode.mono import k1
    from vllm.models.muse_spark.amd.specialized_decode.mono.front import qkv

    names = [k.kernel_name() for k in K1_BUILDS + K1_VARIANTS]
    assert len(set(names)) == len(names)
    assert k1.max_tasks("g192") == spec.K1_TPC and k1.max_tasks("r256") == 10
    for s in range(1, spec.MAX_ROWS + 1):
        assert k1.lds_bytes(s) == k1.key_lds_bytes(spec.K1Build(s, True))
        for lay in K1_VARIANT_LAYOUTS:
            got = k1.lds_bytes(s, lay)
            assert spec.LDS_BYTES // 2 < got <= spec.LDS_BYTES, (s, lay, got)
            rows = 8192 + k1.max_tasks(lay) * s * qkv.XA_ROW * 2
            assert got == max(rows, k1.RESIDENT_LDS), (s, lay, got)


@compiles
@pytest.mark.parametrize("key", K1_VARIANTS, ids=lambda k: k.kernel_name())
def test_k1_variants_compile_without_a_gpu(key, tmp_path, monkeypatch):
    """The 5.9 gate on every variant build, and rule 8 in each role's code:
    r256's boundary CTAs issue their weights only past the slice's publish
    (no weight in flight at the fss / nss polls or the publish), their ardy
    poll holds only those weights, the epilogue's qpart4 poll follows their
    last MFMA (nothing but cos / sin in flight) and loads nothing plain past
    it; the GEMV CTAs' weights all precede their ardy poll."""
    from vllm.models.muse_spark.amd.specialized_decode.mono import k1
    from vllm.models.muse_spark.amd.specialized_decode.mono.front import qkv

    txt, res = _compile_k1(key, tmp_path, monkeypatch)
    print(f"{key.kernel_name()}: {res}")
    assert res["private_segment_fixed_size"] == 0, res
    assert res["next_free_vgpr"] <= spec.VGPR_MAX, res
    assert res["group_segment_fixed_size"] == k1.key_lds_bytes(key), res
    assert res["group_segment_fixed_size"] <= spec.LDS_BYTES, res
    assert res["group_segment_fixed_size"] > spec.LDS_BYTES // 2, res  # one a CU
    loops = _WATERFALL.findall(txt)
    assert not loops, f"{len(loops)} waterfall compares, e.g. {loops[0]}"
    polls = [m.start() for m in _POLL_SLEEP.finditer(txt)]
    nt = [m.start() for m in tq._LOAD_NT.finditer(txt)]
    mfma = [m.start() for m in tq._MFMA.finditer(txt)]
    roles = qkv.roles(key.layout)
    assert len(nt) == 4 * sum(r.rounds for r in roles), len(nt)
    assert len(mfma) == 4 * sum(r.rounds for r in roles), len(mfma)
    if key.layout == "g192":
        assert len(polls) == 4, polls  # fss, nss, qpart, ardy
        fss, nss, qpart, ardy = polls
        segments = [(nt, ardy, len(txt))]
        epi_end = nt[0]
    else:
        assert len(polls) == 5, polls  # fss, nss, ardy (boundary), qpart, ardy
        fss, nss, ardy_b, qpart, ardy = polls
        nb = 4 * roles[0].rounds
        bound_nt, gemv_nt = nt[:nb], nt[nb:]
        # the boundary CTAs' weights: past the slice's publish (its vmcnt(0)
        # and barrier follow the nss poll), before their own ardy poll
        publish = [m.start() for m in _BARRIER.finditer(txt, nss, bound_nt[0])]
        assert publish, "no publish barrier between the nss poll and the weights"
        assert nss < bound_nt[0] and bound_nt[-1] < ardy_b
        # their MFMAs before the epilogue's poll; the GEMV CTAs' after
        assert all(ardy_b < m < qpart for m in mfma[:nb]), "boundary GEMV order"
        assert qpart < gemv_nt[0] and gemv_nt[-1] < ardy
        assert all(m > ardy for m in mfma[nb:])
        # (the boundary CTAs' segment ends at their last MFMA: the
        # epilogue's cos / sin loads follow it, before the qpart poll)
        segments = [(bound_nt, ardy_b, mfma[nb - 1]), (gemv_nt, ardy, len(txt))]
        plain_mid = tq._PLAIN_LOAD.findall(txt, mfma[nb - 1], qpart)
        assert len(plain_mid) == (2 if key.rope else 0), plain_mid
        epi_end = gemv_nt[0]
    # the boundary: no weight before its polls; two norms, nothing fused
    assert nt[0] > nss
    front = tq._code(txt[:qpart])
    assert sum(1 for x in front if x.startswith("v_rsq_f32")) == 2
    # the epilogue: no plain load past its poll; one rsq; RoPE's 3 fmas
    epi_txt = txt[qpart:epi_end]
    assert not tq._PLAIN_LOAD.findall(epi_txt)
    epi = tq._code(epi_txt)
    assert sum(1 for x in epi if x.startswith("v_rsq_f32")) == 1
    assert len(tq._fmas(epi)) == (4 if key.rope else 1), tq._fmas(epi)
    shorts = len(_STORE_SHORT.findall(epi_txt))
    assert shorts == (3 if key.debug_dump else 2), shorts
    assert len(_STORE_SHORT.findall(txt)) == shorts
    # each GEMV role: its weights before its poll, its B operand past it
    for (w, poll, end), role in zip(segments, roles):
        assert w[-1] < poll
        first = tq._MFMA.search(txt, poll, end).start()
        dev = [m.start() for m in tq._LOAD_DEV.finditer(txt, poll, first)]
        writes = [m.start() for m in _DS_WRITE4.finditer(txt, poll, first)]
        want = -(-role.n * key.tokens * qkv.X4 // spec.THREADS)
        assert len(dev) == want, (len(dev), want, role)
        assert writes and dev[-1] < writes[0]
        assert not tq._PLAIN_LOAD.findall(txt, poll, end)


@compiles
def test_k1_r256_trace_is_the_program_order(tmp_path, monkeypatch):
    """r256's trace: the boundary, the boundary CTAs' GEMV, the epilogue, the
    GEMV CTAs' GEMV; every access named and declared by its stage module;
    the layout's K1 regions (qpart4, not qpart) put and polled, scratch
    only."""
    from vllm.models.muse_spark.amd.specialized_decode.mono import k1

    key = spec.K1Build(tokens=3, rope=True, layout="r256")
    with plan.recording() as rec:
        _compile_k1(key, tmp_path, monkeypatch)
    want = ("boundary", "qkv", "epilogue", "qkv")
    assert tuple(rec.stages) == want, rec.stages
    assert not rec.unnamed, rec.unnamed
    puts: set = set()
    polls: set = set()
    for a in rec.accesses:
        assert a.region in k1.STAGE_MODULES[a.stage].STAGE_REGIONS, a
        assert a.space == plan.Space.SCRATCH, a
        (puts if a.kind == "put" else polls).add(a.region)
    regions = set(spec.K1_REGIONS_BY_LAYOUT["r256"])
    assert puts == regions and polls == regions, (puts, polls)


if __name__ == "__main__":
    # python3 test_compile.py --write-baseline <source> <image>: see BASELINE's
    # "procedure" (only ever from the Phase A sources)
    if len(sys.argv) != 4 or sys.argv[1] != "--write-baseline":
        sys.exit(__doc__)
    _write_baseline(BASELINE, sys.argv[2], sys.argv[3])
