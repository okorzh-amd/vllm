# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2(split) (``K2Build(core="split", window=...)``, DESIGN_BC 7, WP-C3)
compiles for gfx950 without a GPU, every width, both attention kinds and the
fault and debug variants, and keeps the Phase A builds as they were.

[CPU] The split ABI (``layer.ABI_BY_CORE["split"]``: ``qkvo``, the KV cache
and its byte strides, the block table and the sequence lengths in place of
``core``; the Int32 arguments typed so), the launch packing, the LDS union
(the split's and the combine's structs join it, the other cores' union is
Phase A's) and the program order of a split build; the CTA roles of 7.2.

[CPU] FlyDSL: the ISA gate of ``test_compile`` (no private segment, at most
256 VGPRs and more than 128, so one CTA a CU, the LDS of one CU, pinned at
S = 1, no readfirstlane waterfall anywhere in the kernel); no SMEM but the
kernel arguments (rule 9: ``pstat`` / ``opart`` / ``xo`` are never read
through the scalar cache) and every merge load of the combine at device scope
(the LLVM IR); the split loop keeps both pages in flight (each page's first QK
MFMA waits for its own loads only); and 7.2's order (rule 8): no o_proj weight
load before the split's and the combine's code and their device-scope stores,
then the CTA's 8 weight loads, then ``load_core``'s ``crdy`` poll and its
device-scope ``xo`` loads. The trace is the split's stages, then Phase A's,
with the hand-over barriers, every region put and polled. A tensor build
compiled after split builds in the same process still hashes to the Phase A
baseline.
"""

import inspect
import json

import pytest
import regex as re
import torch

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

import flydsl.expr.gpu as fx_gpu  # noqa: E402
from flydsl.expr.typing import Int32, Int64  # noqa: E402

from tests.models.muse_spark.amd_specialized_decode import (  # noqa: E402
    test_compile as tc,
)
from tests.models.muse_spark.amd_specialized_decode import (  # noqa: E402
    test_probe_combine as tpc,
)
from tests.models.muse_spark.amd_specialized_decode import (  # noqa: E402
    test_probe_split as tps,
)
from vllm.models.muse_spark.amd.specialized_decode.mono import (  # noqa: E402
    layer,
    spec,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.attention import (  # noqa: E402
    back,
    combine,
    split,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common import (  # noqa: E402
    context,
    plan,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (  # noqa: E402
    CM_DEV,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.sync import (  # noqa: E402
    WAIT_TICKS,
)

compiles = tc.compiles

# every split build the runtime and the tests launch: the q = 1 widths of both
# kinds, the stalled-rank fault builds and a debug build
FAULT = dict(fault_rank=3, fault_epoch=2, fault_slot=spec.k2_slot(1))
SPLIT = [
    spec.K2Build(tokens=s, core="split", window=w)
    for s in range(1, spec.MAX_ROWS + 1)
    for w in (False, True)
] + [
    spec.K2Build(
        tokens=8, core="split", window=True, fault_ticks=WAIT_TICKS * 3 // 2, **FAULT
    ),
    spec.K2Build(tokens=1, core="split", fault_ticks=WAIT_TICKS * 3 // 2, **FAULT),
    spec.K2Build(tokens=8, core="split", debug_dump=True),
]
# the o_proj weight loads a lane issues (``back.o_loads``, 2 tasks x 4 dwordx4)
O_LOADS = 2 * 4
# K2(split)'s LDS at S = 1 (DESIGN_BC 7.7: the split struct, 17,408 B, exceeds
# every Phase A stage at S <= 4); from S = 5 on the pooler's sets it, as in the
# tensor build
LDS_S1 = 29504
# a one-CTA-a-CU kernel: more than 128 VGPRs leaves at most 2 waves a SIMD,
# i.e. the 8 waves of one 512-thread CTA a CU (the M2 review's residency rule)
VGPR_ONE_CTA = 128

_NT_LOAD = re.compile(r"^\s*buffer_load_dwordx4\b.*\bnt\s*$", re.M)
_PV = re.compile(r"^\s*v_mfma_f32_16x16x16_bf16\b", re.M)
_SQRT = re.compile(r"^\s*v_sqrt_f32", re.M)
_STORE = re.compile(r"^\s*buffer_store\S*.*$", re.M)
# a device-scope poll load (sc1 without sc0, which the health words carry)
_POLL = re.compile(r"^\s*buffer_load_dword(?:x2)?\s(?!.*\bsc0\b).*\bsc1\b", re.M)
_XO_LOAD = re.compile(r"^\s*buffer_load_dwordx4\b(?!.*\bnt\b).*\bsc1\b", re.M)
_OPROJ_MFMA = re.compile(r"^\s*v_mfma_f32_16x16x32_bf16\b", re.M)
_SMEM = re.compile(
    r"^\s*s_(?:buffer_)?load_\w+\s+s\[?[\d:]+\]?,\s*(s\[\d+:\d+\])", re.M
)


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


def _line(txt: str, at: int) -> int:
    return txt.count("\n", 0, at) + 1


# ---------------------------------------------------------------- [CPU]


def test_split_abi():
    """The split ABI is spec.K2_ARGS_BY_CORE["split"] renamed as the tensor
    build's (``stream`` -> ``stream_in``), its launch state Phase A's; the
    launcher types the Int32 arguments (block count, table rows and stride,
    rank, slot) Int32 and every other one Int64."""
    assert tuple(layer.ABI_BY_CORE) == back.CORES == spec.K2_CORES
    abi = layer.ABI_BY_CORE["split"]
    want = [layer.KERNEL_NAMES.get(n, n) for n in spec.K2_ARGS_BY_CORE["split"]]
    assert list(abi.names) == want
    assert "core" not in abi.names and "stream" not in abi.names
    assert (
        abi.names[0] == "qkvo"
        and abi.names[-len(layer.ABI.names) + 1 :] == (layer.ABI.names[1:])
    )
    launch = layer.build(spec.K2Build(tokens=2, core="split", window=True))
    py = getattr(launch, "func", None) or launch._func  # as plan._python_params
    params = inspect.signature(py).parameters
    assert tuple(params)[:-1] == abi.names and tuple(params)[-1] == "stream"
    int32 = set(spec.K2_INT32_ARGS_BY_CORE["split"])
    for name in abi.names:
        want_t = Int32 if name in int32 else Int64
        assert params[name].annotation is want_t, name


def test_k2_launch_packs_by_the_split_core(monkeypatch):
    """``k2_launch`` packs a split launch's arguments by the split names: the
    KV strides and counts as given, tensors by address; another core's ABI
    refuses them."""
    calls = []

    def fake_build(key):
        return lambda *a, stream: calls.append((key, a, stream))

    monkeypatch.setattr(layer, "build", fake_build)
    monkeypatch.setattr(layer.torch.cuda, "current_stream", lambda: "cur")
    names = spec.K2_ARGS_BY_CORE["split"]
    ints = {
        "kv_sb": 2 * 4096,
        "kv_sh": 2 * 2048 * 8,
        "kv_st": 256,
        "kv_blocks": 77,
        "bt_rows": 16,
        "bt_stride": 300,
        "rank": 5,
        "slot": spec.k2_slot(7),
        "alarm": 1234,
    }
    t = {n: torch.empty(4, dtype=torch.uint8) for n in names if n not in ints}
    args = {**t, **ints}
    key = spec.K2Build(tokens=3, core="split", window=True)
    layer.k2_launch(key, args)
    [(got_key, packed, stream)] = calls
    assert got_key is key and stream == "cur" and len(packed) == len(names)
    for name, v in zip(names, packed):
        want = args[name]
        assert v == (want.data_ptr() if isinstance(want, torch.Tensor) else want)
    with pytest.raises(TypeError, match="missing"):
        layer.k2_launch(key, {n: v for n, v in args.items() if n != "seq_lens"})
    with pytest.raises(TypeError, match="unexpected"):
        layer.k2_launch(key, {**args, "attn": t["qkvo"]})
    with pytest.raises(TypeError, match="unexpected"):
        layer.k2_launch(spec.K2Build(tokens=3, core="gated"), args)


def test_split_lds_union_and_stages():
    """A split build's union adds the split struct (17,408 B) and the
    combine's (2 KiB) to Phase A's members; the other cores' union is Phase
    A's. The trace order: the split, the combine, then Phase A's stages."""
    for s in range(1, spec.MAX_ROWS + 1):
        base = layer.lds_structs(s)
        lds = layer.lds_structs(s, "split")
        assert set(base) == set(layer.lds_structs(s, "gated"))
        assert set(lds) == set(base) | {split.STAGE, combine.STAGE}
        assert all(lds[k] == base[k] for k in base)  # the same (cached) structs
        sizes = context.stage_bytes(lds)
        assert sizes[split.STAGE] == 17408 and sizes[combine.STAGE] == 2048
        assert layer.lds_bytes(s, "split") <= spec.LDS_BYTES
        assert layer.lds_bytes(s, "split") >= layer.lds_bytes(s)
    assert layer.lds_bytes(1, "split") == LDS_S1 > layer.lds_bytes(1)
    for s in range(5, spec.MAX_ROWS + 1):
        assert layer.lds_bytes(s, "split") == layer.lds_bytes(s)
    assert layer.stages("tensor") == layer.stages("gated") == layer.STAGES
    assert layer.stages("split") == layer.SPLIT_STAGES
    assert (split.STAGE, combine.STAGE, *layer.STAGES) == layer.SPLIT_STAGES
    assert split.STAGE_REGIONS == combine.STAGE_REGIONS == spec.SPLIT_REGIONS


def test_split_roles_place_every_unit_and_combine():
    """7.2's roles over the grid at every width and kind: split units on CTAs
    [0, n_units), one (token, q head) combine on each CTA of [C0, 256); a CTA
    holding both runs the unit first; a unit never waits and a combine waits
    only on units, so with every CTA resident the core cannot deadlock. Every
    CTA's o_proj weights come after its last role stage: at entry (no role),
    after ``srdy`` (a unit), or after ``crdy`` (a combine)."""
    for s in range(1, spec.MAX_ROWS + 1):
        for window in (False, True):
            n, c0 = spec.n_units(s, window), spec.C0(s)
            assert 0 < n <= spec.BLOCKS and 128 <= c0 < spec.BLOCKS
            combines = [divmod(b - c0, spec.HEADS) for b in range(c0, spec.BLOCKS)]
            assert combines == [(t, h) for t in range(s) for h in range(spec.HEADS)]
            units = [spec.split_unit(u, spec.nsplit(s, window)) for u in range(n)]
            assert len(set(units)) == n
            assert {(t, kvh) for t, kvh, _ in units} == {
                (t, kvh) for t in range(s) for kvh in range(spec.KV_HEADS)
            }
            issue = {}
            for b in range(spec.BLOCKS):
                roles = [r for r, on in (("srdy", b < n), ("crdy", b >= c0)) if on]
                issue[b] = roles[-1] if roles else "entry"
            # the CTAs that start their weights late (the 7.2 tail): 16 S
            # combines, plus the split-only units
            assert sum(v == "crdy" for v in issue.values()) == spec.HEADS * s
            assert sum(v == "srdy" for v in issue.values()) == max(min(n, c0), 0)
            # the full kinds hold a unit on every CTA but a few (2 S N <= 256):
            # ``split_roles`` drops the CTA test when n == 256
            if not window:
                assert spec.BLOCKS - n < 2 * s


def test_split_builds_are_distinct():
    """A width's two kinds and its other cores are distinct cached launchers
    under distinct symbols (``build`` traces nothing until the first call)."""
    keys = [
        spec.K2Build(tokens=2),
        spec.K2Build(tokens=2, core="gated"),
        spec.K2Build(tokens=2, core="split"),
        spec.K2Build(tokens=2, core="split", window=True),
    ]
    launches = [layer.build(k) for k in keys]
    assert len({id(f) for f in launches}) == len(keys)
    assert layer.build(keys[3]) is launches[3]
    assert [k.kernel_name() for k in keys] == [
        "muse_mono_k2_s2",
        "muse_mono_k2_s2_g",
        "muse_mono_k2_s2_full",
        "muse_mono_k2_s2_sw",
    ]


# ---------------------------------------------------------------- [CPU] FlyDSL


@compiles
@pytest.mark.parametrize("key", SPLIT, ids=lambda k: k.kernel_name())
def test_split_builds_compile_without_a_gpu(key, tmp_path, monkeypatch):
    """Every split build passes the ISA gate and keeps 7.2's order (rule 8)
    and rule 9; the split loop keeps two pages in flight."""
    txt, res = _compile(key, tmp_path, monkeypatch)
    print(f"{key.kernel_name()}: {res}")
    s = key.tokens
    n = spec.nsplit(s, key.window)
    assert res["private_segment_fixed_size"] == 0, res
    assert VGPR_ONE_CTA < res["next_free_vgpr"] <= spec.VGPR_MAX, res
    assert res["group_segment_fixed_size"] <= spec.LDS_BYTES, res
    assert res["group_segment_fixed_size"] <= layer.lds_bytes(s, "split"), res
    if s == 1:
        assert res["group_segment_fixed_size"] == LDS_S1, res
    if s >= 5:  # the tensor build's (the pooler's struct is the union's max)
        tensor = json.loads(tc.BASELINE.read_text())["builds"]
        want = tensor[spec.K2Build(tokens=s).kernel_name()]["resources"]
        assert res["group_segment_fixed_size"] == want["group_segment_fixed_size"]
    # no readfirstlane waterfall anywhere (the split loop, the combine's and
    # load_core's polls after a lane-predicated put, the xo loads behind
    # the wave-0 store, and every Phase A stage)
    loops = tc._WATERFALL.findall(txt)
    assert not loops, f"{len(loops)} waterfall compares, e.g. {loops[0]}"
    # rule 9: no scalar loads but the kernel arguments' (s[0:1])
    assert "s_buffer_load" not in txt
    bases = set(_SMEM.findall(txt))
    assert bases == {"s[0:1]"}, bases
    for op in (
        "v_mfma_f32_16x16x32_bf16",
        "v_mfma_f32_16x16x16_bf16",  # the split's PV
        "v_mfma_scale_f32_16x16x128_f8f6f4",
        "v_exp_f32",
        "v_sqrt_f32",  # the combine's gate
        "v_div_fixup_f32",
        "row_shr:8",
        "s_memrealtime",
    ):
        assert op in txt, op
    # -- 7.2 / rule 8: the CTA's first o_proj weight load comes after the
    # split's math (both unrolled pages' PV MFMAs), the combine's gate, and
    # every store of both (opart, pstat, srdy, xo, crdy: device scope or
    # wider); no weight load is in flight at the split's or the combine's
    # publish, nor at the combine's srdy poll
    first_nt = _NT_LOAD.search(txt).start()
    assert len(_PV.findall(txt, 0, first_nt)) == 8 and not _PV.search(txt, first_nt)
    assert len(_SQRT.findall(txt, 0, first_nt)) == 1 and not _SQRT.search(txt, first_nt)
    stores = _STORE.findall(txt, 0, first_nt)
    assert all(re.search(r"\bsc1\b", st) for st in stores), stores
    assert sum("buffer_store_short" in st for st in stores) == 1  # xo
    # then the 8 weight loads together, before load_core's first crdy poll
    poll = _POLL.search(txt, first_nt)
    assert poll, "no crdy poll after the weights"
    assert len(_NT_LOAD.findall(txt, first_nt, poll.start())) == O_LOADS
    # and the xo rows at device scope (one dwordx4 a thread and round) after
    # the poll, before o_proj's first MFMA
    first_o = _OPROJ_MFMA.search(txt, poll.start())
    rounds = -(-s * split.OKW // spec.THREADS)
    assert len(_XO_LOAD.findall(txt, poll.start(), first_o.start())) == rounds
    # -- the split loop (7.3): the 2x unrolled trip's two first QK MFMAs each
    # wait for their own page's 6 loads only, the other page's stay in flight
    firsts = list(tps._QK_FIRST.finditer(txt, 0, first_nt))
    assert len(firsts) == 2, [m.group(0) for m in firsts]
    for mf in firsts:
        issued = list(tps._VMEM_LOAD.finditer(txt, 0, mf.start()))[-tps.PAGE_LOADS :]
        assert len(issued) == tps.PAGE_LOADS
        waits = [
            int(w.group(1)) for w in tps._VMCNT.finditer(txt, issued[0].start(),
                                                         mf.start())
        ]  # fmt: skip
        assert waits and min(waits) >= tps.PAGE_LOADS, (_line(txt, mf.start()), waits)
    # -- rule 9 in the IR: the combine's merge loads, exactly cdiv(N, 64)
    # stats and cdiv(N, 8) partials, every one at device scope
    ir = tpc._merge_ir_loads(str(tmp_path), key.kernel_name())
    assert ir[combine.PSTAT_BYTES] == [("v2f32", CM_DEV)] * (-(-n // 64)), ir
    assert ir[combine.OPART_BYTES] == [("f32", CM_DEV)] * (-(-n // spec.WAVES)), ir


@compiles
@pytest.mark.parametrize(
    "key",
    [
        spec.K2Build(tokens=s, core="split", window=w)
        for s in (1, 8)
        for w in (False, True)
    ],
    ids=lambda k: k.kernel_name(),
)
def test_split_trace_is_the_program_order(key, tmp_path, monkeypatch):
    """The split build's trace: the split and the combine, then Phase A's
    stages (the split core load is o_proj's first stage), a barrier after
    every LDS stage, every access named and declared (o_proj's core load
    polls the split's ``crdy``), every K2 and split region put and polled."""
    seen = []  # the stage count when each barrier was traced
    real = fx_gpu.barrier

    def barrier(*a, **k):
        rec = plan._record
        seen.append(len(rec.stages) if rec is not None else -1)
        return real(*a, **k)

    monkeypatch.setattr(fx_gpu, "barrier", barrier)
    with plan.recording() as rec:
        _compile(key, tmp_path, monkeypatch)
    assert tuple(rec.stages) == layer.SPLIT_STAGES, rec.stages
    assert not rec.unnamed, rec.unnamed
    for i, name in enumerate(rec.stages[:-1]):
        if name not in tc.NO_LDS:
            assert i + 1 in seen, f"no barrier between {name} and {rec.stages[i + 1]}"
    declared = {
        **{st: set(m.STAGE_REGIONS) for st, m in tc.STAGE_MODULES.items()},
        split.STAGE: set(split.STAGE_REGIONS),
        combine.STAGE: set(combine.STAGE_REGIONS),
    }
    declared[back.STAGE] |= set(split.STAGE_REGIONS)  # load_core polls crdy
    puts: set = set()
    polls: set = set()
    for a in rec.accesses:
        assert a.region in declared[a.stage], (a.stage, a.region)
        space = plan.Space.PEER if a.region in spec.PEER_REGIONS else plan.Space.SCRATCH
        assert a.space == space, a
        (puts if a.kind == "put" else polls).add(a.region)
    regions = set(spec.K2_REGIONS) | set(spec.SPLIT_REGIONS)
    assert puts == regions and polls == regions, (puts, polls)
    # the split core's flags: srdy put by the split, polled by the combine;
    # crdy put by the combine, polled by o_proj's core load
    acc = {(a.stage, a.region, a.kind) for a in rec.accesses if a.region in regions}
    assert {x for x in acc if x[1] in spec.SPLIT_REGIONS} == {
        (split.STAGE, "srdy", "put"),
        (combine.STAGE, "srdy", "poll"),
        (combine.STAGE, "crdy", "put"),
        (back.STAGE, "crdy", "poll"),
    }


@compiles
@pytest.mark.parametrize("tokens", [1, 8])
def test_tensor_build_unchanged_after_split(tokens, tmp_path, monkeypatch):
    """A Phase A build compiled after split builds of its width, in one
    process (shared caches: the build table, the cached LDS structs), still
    has the Phase A final ISA."""
    base = json.loads(tc.BASELINE.read_text())
    if tc.toolchain() != base["toolchain"]:
        pytest.skip(f"the baseline pins {base['toolchain']}: see test_compile")
    for window in (False, True):
        key = spec.K2Build(tokens=tokens, core="split", window=window)
        _compile(key, tmp_path / key.kernel_name(), monkeypatch)
    key = spec.K2Build(tokens=tokens)
    txt, res = _compile(key, tmp_path / "t", monkeypatch)
    rec = base["builds"][key.kernel_name()]
    assert {k: res[k] for k in rec["resources"]} == rec["resources"]
    assert tc.isa_digest(txt, key.kernel_name()) == rec["sha256"]
