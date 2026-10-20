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
"""

import glob
import os

import pytest
import regex as re
import torch

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

import flydsl.expr.gpu as fx_gpu  # noqa: E402

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
    assert puts == set(spec.REGIONS) and polls == set(spec.REGIONS), (puts, polls)
