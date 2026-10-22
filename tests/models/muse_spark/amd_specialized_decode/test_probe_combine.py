# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2(split)'s combine (``mono/attention/combine.py``, DESIGN_BC 7.4,
WP-C2) in a probe kernel on synthetic partials.

The probe runs the 256-CTA grid of a ``K2Build(core="split")``: CTA u <
``n_units`` is a stand-in split unit that copies its unit's partials from an
input (as the split stores them: ``pstat`` / ``opart``, CM_DEV) and publishes
``srdy[u]`` (``pub[u]``: 0 never, 1 at once, w > 1 after spinning w ticks
first, a late unit); CTA ``C0 + k`` runs ``stage_combine(c, k)``.
The partials are the split's format: per unit and q head the base-2 running
max m, the sum l of the bf16 P, and O = sum P V, unnormalized.

[CPU] ``unit_plan`` (the kernel's very function, on ints) equals
``spec.split_plan``; the host split of a real decode problem combines to
fp64 attention within the core's targets (DESIGN_BC 4), so the partials the
GPU tests feed are the split's; the cases make the edges bite; the probe
compiles GPU-less for gfx950 (the gate, no waterfall, no SMEM, exactly the
static N's merge loads, each at device scope, in the LLVM IR) and traces
exactly ``srdy`` polled, ``crdy`` put.
[1GPU] The attention output (dumped before the gate) vs the fp64 combine of
the same partials (<= 1 bf16 ulp up to fp32 summation noise); the ``xo`` row
``torch.equal`` the runtime-compiled ``rms_norm_gated`` of that output;
pad rows and empty rows (L = 0, every live unit (NEG, 0)) exact zeros;
partials past ``live`` poisoned with NaN and their flags stale (never read
or waited on); every ``crdy`` pair carries the launch's tag; 200 launches
(scratch poisoned between them) give equal bytes; late units (their
partials stored tens of us after the combine starts polling) give the same
bytes, so a merge load the poll does not order would read the poison; a
live unit's missing flag gives up (FAILED, ``srdy`` named) and the launch
still completes.
"""

import glob
import math
import os
from typing import Any

import pytest
import regex as re
import torch

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

import flydsl.expr as fx  # noqa: E402
from aiter.ops.flydsl.kernels import buffer_ops as bo  # noqa: E402
from flydsl.expr import rocdl  # noqa: E402
from flydsl.expr.typing import T  # noqa: E402

from tests.models.muse_spark.amd_specialized_decode import (  # noqa: E402
    reference,
)
from vllm.models.muse_spark.amd.specialized_decode.mono import spec  # noqa: E402
from vllm.models.muse_spark.amd.specialized_decode.mono.abi import (  # noqa: E402
    WAIT_BUDGET,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.attention import (  # noqa: E402
    combine,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common import (  # noqa: E402
    context,
    plan,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (  # noqa: E402
    CM_DEV,
    ld_i32,
    memrealtime,
    rsrc,
    traced,
    uniform,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.plan import (  # noqa: E402
    Space,
    enter_stage,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.sync import (  # noqa: E402
    publish,
)

bf = torch.bfloat16
F32, F64 = torch.float32, torch.float64
CORE, QKV, ROW_G = spec.CORE, spec.QKV_ROWS, spec.ROW_G
J = spec.Q_PER_KV  # q heads a unit
D = spec.HEAD_DIM
UNIT_WORDS = J * D  # opart words a unit: one a thread of the stand-in
NEG = combine.NEG
BT_STRIDE = 4096  # the table's row (entries): L clamps at 65536
BUILDS = [(1, False), (1, True), (3, False), (8, False), (8, True)]
G_EDGES = (-100.0, -88.0, 0.0, 88.0, 100.0)


def _on_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


def _compiles_for_gfx950() -> bool:
    arch = os.environ.get("FLYDSL_GPU_ARCH") or os.environ.get("ARCH")
    return arch == "gfx950" or _on_gfx950()


needs_gfx950 = pytest.mark.skipif(
    not _on_gfx950(), reason="the Muse mono probes run on gfx950 (MI355X)"
)
compiles = pytest.mark.skipif(
    not _compiles_for_gfx950(),
    reason="GPU-less compile needs FLYDSL_GPU_ARCH=gfx950 (or a gfx950 GPU)",
)

# ---------------------------------------------------------------- the probe

PROBE_ARGS = (
    "qkvo",
    "pad",
    "seq_lens",
    "bt_stride",
    "pstat_in",
    "opart_in",
    "pub",
    "attn",
)


@traced
def _stand_in_split(c, u):
    """Split unit u's stores as ``stage_split`` leaves them: its 8 heads'
    ``opart`` (a word a thread) and ``pstat`` words from the inputs, CM_DEV;
    then ``srdy[u]`` if ``pub[u]`` (> 1: the stores and the flag ``pub[u]``
    ticks of ``s_memrealtime`` late)."""
    enter_stage("split")
    tid = c["tid"]
    a, scr = c["args"], c["scr"]
    how = uniform(ld_i32(a["pub"], u))
    if how > 1:
        t0 = memrealtime()
        now = memrealtime()
        while now - t0 < fx.Int64(how):
            rocdl.s_sleep(1)
            now = memrealtime()
    at = u * UNIT_WORDS + tid
    v = bo.buffer_load(rsrc(a["opart_in"]), at, vec_width=1, dtype=T.i32)
    bo.buffer_store(v, rsrc(scr[spec.OPART]), at, cache_modifier=CM_DEV)
    if tid < 2 * J:
        at2 = u * 2 * J + tid
        w = bo.buffer_load(rsrc(a["pstat_in"]), at2, vec_width=1, dtype=T.i32)
        bo.buffer_store(w, rsrc(scr[spec.PSTAT]), at2, cache_modifier=CM_DEV)
    if how != 0:
        publish(c["put"], scr["srdy"], spec.srdy_pair(u), fx.Int32(1), tid == 0)


@traced
def _grid(c):
    s, b = c["S"], c["bid"]
    w = c["key"].window
    if b < spec.n_units(s, w):
        _stand_in_split(c, b)
    if b >= spec.C0(s):
        combine.stage_combine(c, b - spec.C0(s), attn_out=c["args"]["attn"])


def _body(c):
    """The probe's ``bt_stride`` comes as an Int64 (the argv table); K2's is
    an Int32."""
    mine = dict(c)
    mine["args"] = {**c["args"], "bt_stride": fx.Int32(c["args"]["bt_stride"])}
    _grid(mine)


def _probe(S: int, window: bool) -> context.Probe:
    kind = "sw" if window else "full"
    return context.probe_kernel(
        f"muse_probe_combine_s{S}_{kind}",
        S,
        _body,
        {"combine": combine.lds_struct(S)},
        PROBE_ARGS,
        blocks=spec.BLOCKS,
        build=spec.K2Build(tokens=S, core="split", window=window),
    )


# ---------------------------------------------------------------- [CPU]


def test_stage_contract():
    """The LDS member, the regions, the placement, every index in range."""
    assert combine.lds_struct(8).__dsl_size_of__() == spec.WAVES * D * 4
    assert combine.lds_struct(3) is combine.lds_struct(3)
    assert set(combine.STAGE_REGIONS) == set(spec.SPLIT_REGIONS)
    lay = spec.scratch_layout()
    for S in range(1, spec.MAX_ROWS + 1):
        for w in (False, True):
            n = spec.nsplit(S, w)
            # combine CTAs [C0, 256): one (t, h) each, every unit of the
            # (t, kvh) a split CTA
            ks = range(spec.BLOCKS - spec.C0(S))
            assert sorted((k // 16, k % 16) for k in ks) == [
                (t, h) for t in range(S) for h in range(16)
            ]
            last = (2 * (S - 1) + 1) * n + n - 1
            assert last < spec.n_units(S, w) <= spec.BLOCKS
            assert 4 * (spec.pstat_word(last, J - 1, 1) + 1) <= lay[spec.PSTAT][1]
            assert 4 * (spec.opart_word(last, J - 1, D - 1) + 1) <= lay[spec.OPART][1]
            assert spec.crdy_pair(S - 1, 15) < 16 * spec.MAX_TOKENS
            assert n <= 2 * 64  # the stats: two lane-parallel loads at most
    assert 2 * (spec.xo_elem(spec.MAX_TOKENS - 1, 15, D - 1) + 1) == lay[spec.XO][1]
    assert lay[spec.PSTAT][1] == combine.PSTAT_BYTES
    assert lay[spec.OPART][1] == combine.OPART_BYTES


def test_unit_plan_is_spec_split_plan():
    """The kernel's ``unit_plan`` (Python ints) is ``spec.split_plan``, at
    every width and kind, over the window and page edges and long context."""
    Ls = set(range(0, 4200)) | {32767, 32768, 32773, 65535, 65536, 131072}
    for S in range(1, spec.MAX_ROWS + 1):
        for w in (False, True):
            n = spec.nsplit(S, w)
            for L in Ls:
                got = combine.unit_plan(L, w, n)
                assert got == spec.split_plan(L, w, n), (S, w, L)
                assert got[4] <= n and (L == 0) == (got[4] == 0)


def test_unit_plan_is_spec_split_plan_for_every_preset():
    """TUNE L3: the kernel's ``unit_plan`` with a preset's N and MIN_PER is
    ``spec.split_plan`` with them, at every width and kind, and every index
    of the largest N stays in its region."""
    Ls = set(range(0, 300)) | set(range(2000, 2120)) | set(range(4090, 4130))
    Ls |= {32767, 32768, 32773, 65536, 131072}
    lay = spec.scratch_layout()
    seen = set()
    for p in spec.SPLIT_PRESETS:
        for S in range(1, spec.MAX_ROWS + 1):
            for w in (False, True):
                k = spec.K2Build(
                    tokens=S, core="split", window=w, **spec.split_fields(p, S, w)
                )
                n, mp = k.nsplit(), k.split_min_per
                last = k.n_units() - 1
                assert 4 * (spec.pstat_word(last, J - 1, 1) + 1) <= lay[spec.PSTAT][1]
                assert (
                    4 * (spec.opart_word(last, J - 1, D - 1) + 1) <= lay[spec.OPART][1]
                )
                if (w, n, mp) in seen:
                    continue
                seen.add((w, n, mp))
                for L in Ls:
                    got = combine.unit_plan(L, w, n, mp)
                    assert got == spec.split_plan(L, w, n, mp), (p, S, w, L)
                    assert got[4] <= n and (L == 0) == (got[4] == 0)


def _units(L: int, window: bool, n: int):
    """[(first key, end key)] of the units p < live: unit p covers pages [b0 +
    p per, min(b0 + (p + 1) per, b0 + nb)), keys clipped to [lo, L)."""
    lo, b0, nb, per, live = spec.split_plan(L, window, n)
    out = []
    for p in range(live):
        pa, pe = b0 + p * per, min(b0 + (p + 1) * per, b0 + nb)
        out.append((max(16 * pa, lo), min(16 * pe, L)))
    return out


def _split_unit(q, K, V):
    """A unit's partials of 8 q heads over its keys, as ``stage_split``
    computes them: fp32 logits x f32(QK_SCALE LOG2E) (base 2), m, P =
    bf16(exp2(s - m)), l = sum f32(P), O = sum P V (fp32 sums). q bf16 [8,
    64], K / V bf16 [keys, 64]. -> (m [8], l [8], O [8, 64]) fp32."""
    sc = (q.float() @ K.float().T) * torch.tensor(spec.QK_SCALE_LOG2E, dtype=F32)
    m = sc.max(1).values
    P = torch.exp2(sc - m[:, None]).to(bf).float()
    return m, P.sum(1), P @ V.float()


def _problem(S: int, window: bool, Ls, seed: int, pad_rows=()) -> dict[str, Any]:
    """A decode problem at context lengths ``Ls`` (row t: ``seq_lens[t]``;
    clamped to the table as the kernel does) and its split partials.

    Returns a dict: qkvo bf16 [S, QKV] (og with sigmoid's edges; NaN pad
    rows), pad bool [8],
    seq int32 [S], pstat / opart fp32 (the scratch regions' shape: units past
    each (t, kvh)'s live poisoned with NaN), pub int32 [BLOCKS] (every unit),
    live [S, 2], attn fp64 [S, 16, 64] (exact attention, zeros on pad / empty
    rows), amag (softmax @ |V|) and parts: (t, kvh) -> [(m, l, O)] of the
    live units."""
    g = torch.Generator().manual_seed(seed)
    n = spec.nsplit(S, window)
    qkvo = (torch.randn(S, QKV, generator=g) * 2).to(bf)
    og = qkvo[:, ROW_G:].float().reshape(-1, D)
    og[1::7] = torch.tensor(G_EDGES * 12 + (0.0,) * 4)
    qkvo[:, ROW_G:] = og.reshape(S, CORE).to(bf)
    pad = torch.zeros(spec.MAX_TOKENS, dtype=torch.bool)
    pad[list(pad_rows)] = True
    qkvo[list(pad_rows)] = math.nan  # a pad row's og never reaches xo
    seq = torch.tensor([Ls[t % len(Ls)] for t in range(S)], dtype=torch.int32)
    pstat = torch.full((spec.BLOCKS * J * 2,), math.nan)
    opart = torch.full((spec.BLOCKS * UNIT_WORDS,), math.nan)
    live = torch.zeros(S, 2, dtype=torch.int64)
    attn = torch.zeros(S, spec.HEADS, D, dtype=F64)
    amag = torch.zeros_like(attn)
    parts: dict[tuple[int, int], list] = {}
    for t in range(S):
        L = 0 if pad[t] else min(max(int(seq[t]), 0), BT_STRIDE * 16)
        lo = max(0, L - spec.WINDOW) if window else 0
        for kvh in range(spec.KV_HEADS):
            q = qkvo[t, J * D * kvh : J * D * (kvh + 1)].reshape(J, D)
            # keys as a sliding layer's: unit-scale logits, a spread of maxima
            K = (torch.randn(L, D, generator=g) * 0.25).to(bf)
            V = torch.randn(L, D, generator=g).to(bf)
            units = _units(L, window, n)
            live[t, kvh] = len(units)
            parts[t, kvh] = []
            for p, (k0, k1) in enumerate(units):
                m, lsum, out = _split_unit(q, K[k0:k1], V[k0:k1])
                u = (2 * t + kvh) * n + p
                pstat.view(-1, J, 2)[u, :, 0] = m
                pstat.view(-1, J, 2)[u, :, 1] = lsum
                opart.view(-1, J, D)[u] = out
                parts[t, kvh].append((m, lsum, out))
            if L:
                s = (q.double() @ K[lo:L].double().T) * spec.QK_SCALE
                sm = torch.softmax(s, -1)
                attn[t, J * kvh : J * (kvh + 1)] = sm @ V[lo:L].double()
                amag[t, J * kvh : J * (kvh + 1)] = sm @ V[lo:L].double().abs()
    pub = torch.ones(spec.BLOCKS, dtype=torch.int32)
    return dict(
        S=S, window=window, n=n, qkvo=qkvo, pad=pad, seq=seq, pstat=pstat,
        opart=opart, pub=pub, live=live, attn=attn, amag=amag, parts=parts,
    )  # fmt: skip


def _combine_ref(P):
    """The fp64 combine of ``P``'s partials: (o fp64 [S, 16, 64], mag fp64
    [S, 16, 64] = sum_p w_p |O_p| / L, the summation-noise scale)."""
    S = P["S"]
    o = torch.zeros(S, spec.HEADS, D, dtype=F64)
    mag = torch.zeros_like(o)
    for (t, kvh), units in P["parts"].items():
        if not units:
            continue
        m = torch.stack([u[0] for u in units]).double()  # [p, 8]
        lsum = torch.stack([u[1] for u in units]).double()
        op = torch.stack([u[2] for u in units]).double()  # [p, 8, 64]
        ok = lsum > 0
        M = torch.where(ok, m, -math.inf).max(0).values
        w = torch.where(ok, torch.exp2(m - M), 0.0)
        tot = (lsum * w).sum(0)
        has = (tot > 0)[:, None]
        num = (op * w[..., None]).sum(0)
        den = torch.where(has, tot[:, None], 1.0)
        o[t, J * kvh : J * (kvh + 1)] = torch.where(has, num / den, 0.0)
        a = (op.abs() * w[..., None]).sum(0)
        mag[t, J * kvh : J * (kvh + 1)] = torch.where(has, a / den, 0.0)
    return o, mag


def _ord(x: torch.Tensor) -> torch.Tensor:
    """bf16 -> integers in value order (+0 and -0 both 0)."""
    b = x.contiguous().view(torch.int16).int()
    m = b & 0x7FFF
    return torch.where(b < 0, -m, m)


def _ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (_ord(a) - _ord(b)).abs()


def _check_close(got, ref, mag, *, max_ulps: int = 1, what: str = "") -> None:
    """bf16 ``got`` vs fp64 ``ref``: <= ``max_ulps`` bf16 ulps of bf16(ref),
    past it only within fp32 summation noise (``2^-18 mag``: a cancelling sum
    lands ulps apart where a bf16 ulp is below the fp32 error); >= 99.9 %
    within 1 ulp (a floor of 2 elements)."""
    u = _ulps(got, ref.to(bf))
    noise = (got.double() - ref).abs() <= 2**-18 * mag
    assert ((u <= max_ulps) | noise).all(), (what, u.max())
    assert (u > 1).sum().item() <= max(2, math.ceil(0.001 * u.numel())), (what, u)


CASES: dict[tuple[int, bool], list[tuple[list[int], tuple[int, ...]]]] = {
    # (S, window): [(Ls, pad rows)]
    # 16384: live = N = 128 (both stats loads and every partial load live)
    (1, False): [([32773], ()), ([17], ()), ([2048], ()), ([1], ()), ([16384], ())],
    (1, True): [([4103], ()), ([2049], ()), ([2047], ()), ([300], ())],
    (3, False): [([129, 70000, 4103], ()), ([0, 256, 2065], (1,))],
    (8, False): [([1, 17, 128, 129, 2048, 4103, 32773, 16], (2, 7))],
    (8, True): [([1, 2047, 2048, 2049, 2063, 2065, 4103, 300], (5,))],
}


@pytest.mark.parametrize("S,window", BUILDS)
def test_split_partials_combine_to_attention(S, window):
    """[CPU] The host split's partials, combined in fp64 and rounded to bf16,
    are the decode attention up to the bf16 P (E9, the only approximation):
    within that rounding's bound (2^-9 relative a weight, on numerator and
    denominator, then the output's bf16 rounding: < 2^-7 softmax @ |V|)
    everywhere, cos > 0.99999 (DESIGN_BC 4's ulp target is vs stock, which
    rounds its normalized P too: the [1GPU] split probe's). Pad rows and
    L = 0 give zeros."""
    for i, (Ls, pads) in enumerate(CASES[S, window]):
        P = _problem(S, window, Ls, 7 + i, pads)
        o, _ = _combine_ref(P)
        live = (P["live"] > 0).repeat_interleave(J, 1)  # [S, 16]
        got, want = o[live].to(bf), P["attn"][live]
        assert ((got.double() - want).abs() <= 2**-7 * P["amag"][live]).all()
        assert reference.cos(got.double(), want) > 0.99999
        assert (o[~live] == 0).all() and (P["attn"][~live] == 0).all()


def test_cases_make_the_edges_bite():
    """[CPU] The cases hold one-unit and many-unit (t, kvh)s, live < N with
    poisoned units past it, live = N at N = 128, L clamped at the table, a
    pad row with L > 0, an L = 0 live row, logit maxima spread past exp2's
    range across units, and sigmoid's saturating gates."""
    seen = set()
    for (S, window), cases in CASES.items():
        n = spec.nsplit(S, window)
        for i, (Ls, pads) in enumerate(cases):
            P = _problem(S, window, Ls, 7 + i, pads)
            lv = P["live"]
            seen |= {"one"} if (lv == 1).any() else set()
            seen |= {"many"} if (lv > 8).any() else set()
            seen |= {"live<N"} if ((lv > 0) & (lv < n)).any() else set()
            seen |= {"live=128"} if n == 128 and (lv == n).any() else set()
            seen |= {"clamp"} if (P["seq"] > BT_STRIDE * 16).any() else set()
            seen |= {"pad"} if any(int(P["seq"][t]) > 0 for t in pads) else set()
            seen |= {"L0"} if any(int(P["seq"][t]) == 0 for t in range(S)) else set()
            pst = P["pstat"].view(-1, J, 2)
            for t, kvh in P["parts"]:
                u0, nl = (2 * t + kvh) * n, int(lv[t, kvh])
                assert pst[u0 + nl : u0 + n].isnan().all()  # never read
            og = P["qkvo"][:, ROW_G:].float()
            if all((og == v).any() for v in G_EDGES):
                seen.add("gates")
    want = {"one", "many", "live<N", "live=128", "clamp", "pad", "L0", "gates"}
    assert seen == want, seen
    # the edges of the synthetic stats (``_edge_stats``) bite in the reference
    P = _edge_stats(_problem(8, False, CASES[8, False][0][0], 3))
    o, _ = _combine_ref(P)
    assert (o[0, :J] == 0).all()  # every live unit (NEG, 0): L = 0
    many = [u for tk, u in P["parts"].items() if tk != (0, 0) and len(u) >= 3]
    assert many
    for units in many:
        m = torch.stack([u[0] for u in units if (u[1] > 0).all()])
        assert (m.max(0).values - m[0] > 200).all()  # unit 0's exp2 underflows


def _edge_stats(P: dict[str, Any]) -> dict[str, Any]:
    """``P`` with synthetic stats on top of the attention's: row 0's KV head 0
    has every live unit (NEG, 0, O = 0) (L = 0 after the merge); in every
    other (t, kvh) with two or more units, unit 1 is (NEG, 0, 0), and with
    three or more unit 0's max is 300 below its own (its weight underflows in
    fp32's exp2)."""
    n = P["n"]
    pst = P["pstat"].view(-1, J, 2)
    opt = P["opart"].view(-1, J, D)
    for (t, kvh), units in P["parts"].items():
        u0 = (2 * t + kvh) * n
        for p in range(len(units)):
            m, lsum, out = units[p]
            if (t, kvh) == (0, 0) or len(units) >= 2 and p == 1:
                m, lsum, out = torch.full_like(m, NEG), lsum * 0, out * 0
            elif len(units) >= 3 and p == 0:
                m = m - 300
            units[p] = (m, lsum, out)
            pst[u0 + p, :, 0], pst[u0 + p, :, 1], opt[u0 + p] = m, lsum, out
    return P


def _merge_ir_loads(dump: str, name: str) -> dict[int, list[tuple[str, int]]]:
    """The buffer loads of the LLVM IR of ``name`` through a resource sized
    ``PSTAT_BYTES`` or ``OPART_BYTES`` (only the merge's): num_records ->
    [(loaded type, cache policy)]."""
    for path in glob.glob(f"{dump}/**/*llvm_ir.ll", recursive=True):
        with open(path) as f:
            txt = f.read()
        if not re.search(rf"define amdgpu_kernel void @{re.escape(name)}\(", txt):
            continue
        sized = {combine.PSTAT_BYTES, combine.OPART_BYTES}
        rsrcs = {
            m.group(1): int(m.group(2))
            for m in re.finditer(
                r"(%\d+) = call ptr addrspace\(8\) @llvm\.amdgcn\.make\.buffer"
                r"\.rsrc\.p8\.p0\(ptr %\d+, i16 0, i64 (\d+),",
                txt,
            )
            if int(m.group(2)) in sized
        }
        out: dict[int, list[tuple[str, int]]] = {b: [] for b in sized}
        for m in re.finditer(
            r"@llvm\.amdgcn\.raw\.ptr\.buffer\.load\.(\w+)\(ptr addrspace\(8\) "
            r"(%\d+), i32 [^,]+, i32 [^,]+, i32 (\d+)\)",
            txt,
        ):
            if m.group(2) in rsrcs:
                out[rsrcs[m.group(2)]].append((m.group(1), int(m.group(3))))
        return out
    raise AssertionError(f"no LLVM IR of {name} under {dump}")


def _isa(dump: str, name: str) -> tuple[str, dict]:
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


def _compile(probe: context.Probe) -> None:
    try:
        probe.compile()
    except RuntimeError as e:  # the load, past the compile, needs a device
        if "hipErrorNoDevice" not in str(e) and "no ROCm" not in str(e):
            raise


def _merge_loads(txt: str) -> tuple[int, int]:
    """(dword, dwordx2) buffer loads at device scope (``sc1``) in ``txt``."""
    x1 = re.findall(r"^\s*buffer_load_dword\s[^\n]*\bsc1\b", txt, re.M)
    x2 = re.findall(r"^\s*buffer_load_dwordx2\s[^\n]*\bsc1\b", txt, re.M)
    return len(x1), len(x2)


@compiles
@pytest.mark.parametrize("S,window", BUILDS)
def test_combine_probe_compiles(S, window, tmp_path, monkeypatch):
    """The compile gate on the probe (the stand-in split and the combine on
    one CTA, as K2 runs both there), and the traced hand-offs."""
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    probe = _probe(S, window)
    with plan.recording() as rec:
        _compile(probe)
    txt, res = _isa(str(tmp_path), probe.name)
    n = spec.nsplit(S, window)
    x1, x2 = _merge_loads(txt)
    print(f"{probe.name}: N={n} {res} sc1 loads dword={x1} dwordx2={x2}")
    assert res["private_segment_fixed_size"] == 0, res
    assert res["next_free_vgpr"] <= spec.VGPR_MAX, res
    assert res["group_segment_fixed_size"] <= spec.LDS_BYTES, res
    waterfall = re.compile(r"v_cmp_eq_u64_e\d+\s+\S+,\s*s\[\d+:\d+\],\s*v\[")
    assert not waterfall.findall(txt)
    # no scalar (SMEM) load of a plain region: the buffer loads only
    assert not re.search(r"^\s*s_buffer_load", txt, re.M)
    # rule 9 on the merge, in the IR: exactly cdiv(N, 64) stats (a dwordx2 a
    # lane) and cdiv(N, 8) partials (a dword a lane; masked loads stay loads),
    # every one at device scope (CM_DEV), none through SMEM
    ir_loads = _merge_ir_loads(str(tmp_path), probe.name)
    assert ir_loads[combine.PSTAT_BYTES] == [("v2f32", CM_DEV)] * (-(-n // 64))
    assert ir_loads[combine.OPART_BYTES] == [("f32", CM_DEV)] * (-(-n // spec.WAVES))
    assert x1 >= -(-n // spec.WAVES) and x2 >= -(-n // 64), (x1, x2, n)
    # the gate: one raw sqrt and the IEEE divides (rstd, sigmoid, 1 / L)
    assert len(re.findall(r"^\s*v_sqrt_f32", txt, re.M)) == 1
    assert len(re.findall(r"^\s*v_div_fixup_f32", txt, re.M)) >= 3
    assert "v_exp_f32" in txt and "v_rsq_f32" not in txt
    if not rec.accesses:  # traced earlier in this process (FlyDSL's own cache)
        return
    assert not rec.unnamed
    acc = {(a.stage, a.region, a.space, a.kind) for a in rec.accesses}
    assert {a[1:] for a in acc if a[0] == combine.STAGE} == {
        ("srdy", Space.SCRATCH, "poll"),
        ("crdy", Space.SCRATCH, "put"),
    }


# ---------------------------------------------------------------- [1GPU]


def _assert_healthy(bufs) -> None:
    assert not bufs.failed(), bufs.given_up_waits()
    h = bufs.health.cpu().clone()
    h[WAIT_BUDGET] = 0
    assert not h.any(), bufs.given_up_waits()
    assert not bufs.flags().any()


def _region(bufs, name: str, dtype) -> torch.Tensor:
    off, nbytes = spec.scratch_layout()[name]
    return bufs.scratch[off : off + nbytes].view(dtype)


def _poison(bufs) -> None:
    """NaN in ``xo``, ``pstat`` and ``opart``: a launch's output is its own."""
    for r in (spec.XO, spec.PSTAT, spec.OPART):
        _region(bufs, r, torch.uint8).fill_(0xFF)


def _launch(probe, bufs, P, attn):
    probe(
        bufs,
        qkvo=P["d_qkvo"],
        pad=P["d_pad"],
        seq_lens=P["d_seq"],
        bt_stride=BT_STRIDE,
        pstat_in=P["d_pstat"],
        opart_in=P["d_opart"],
        pub=P["d_pub"],
        attn=attn,
    )


def _to_device(P: dict[str, Any]) -> dict[str, Any]:
    for k in ("qkvo", "pad", "seq", "pstat", "opart", "pub"):
        P["d_" + k] = P[k].cuda()
    return P


def _run(probe, bufs, P):
    """One launch: (xo bf16 [S, 1024], attn bf16 [S, 1024]) on the CPU."""
    S = P["S"]
    attn = torch.full((S, CORE), math.nan, dtype=bf, device="cuda")
    _poison(bufs)
    _launch(probe, bufs, P, attn)
    torch.accelerator.synchronize()
    xo = _region(bufs, spec.XO, bf)[: S * CORE].view(S, CORE)
    return xo.cpu().clone(), attn.cpu()


def _stock_gate(attn, qkvo):
    """The runtime-compiled ``rms_norm_gated`` of (attn, og), og read in place
    at the qkv stride 2304 as served (a served-shape call first: Triton's
    cache then holds decode's specialization)."""
    from vllm.third_party.flash_linear_attention.ops.kda import (
        rms_norm_gated,
    )

    def stock(a, q):
        t = a.shape[0]
        # rms_norm_gated writes y over x (kda.py: ``y = x`` when
        # out_dtype is None): on a copy, the inputs are reused below
        return rms_norm_gated(
            a.clone().view(t, spec.HEADS, D),
            q[:, ROW_G:].view(t, spec.HEADS, D),
            None,
            None,
            "sigmoid",
            eps=spec.RMS_EPS,
        ).reshape(t, CORE)

    a, q = attn.cuda(), qkvo.cuda()
    stock(a[:1], q[:1])
    return stock(a, q).cpu()


def _check(bufs, P, xo, attn) -> None:
    """The attention output vs the fp64 combine of the partials, the gated
    row bit for bit vs the stock gate on it, zeros where nothing is live,
    every crdy pair this launch's."""
    S = P["S"]
    o, mag = _combine_ref(P)
    rows = (~P["pad"][:S])[:, None].expand(S, spec.HEADS)
    lv = (P["live"] > 0).repeat_interleave(J, 1) & rows
    got = attn.view(S, spec.HEADS, D)
    _check_close(got[lv], o[lv], mag[lv], what=f"S={S} attn")
    assert (got[~lv].view(torch.int16) == 0).all()  # +0: pad and empty rows
    live_rows = ~P["pad"][:S]
    want = _stock_gate(attn[live_rows], P["qkvo"][live_rows])
    same = (xo[live_rows].view(torch.int16) == want.view(torch.int16)).all(1)
    assert same.all(), f"S={S}: {(~same).sum().item()} gated rows differ"
    assert (xo[~live_rows].view(torch.int16) == 0).all()
    vals, tags = bufs.read_pairs("crdy", 0, 16 * S)
    assert (tags.cpu() == bufs.tag()).all() and (vals.cpu() == 1).all()
    _assert_healthy(bufs)


@needs_gfx950
@pytest.mark.parametrize("S,window", BUILDS)
def test_combine_matches_the_partials_and_the_stock_gate(S, window):
    """Attention partials of real decode problems (all units published), then
    the same with only the live units' flags written (the rest stale)."""
    probe = _probe(S, window)
    bufs = context.ProbeBuffers()
    for i, (Ls, pads) in enumerate(CASES[S, window]):
        P = _to_device(_problem(S, window, Ls, 100 + i, pads))
        xo, attn = _run(probe, bufs, P)
        _check(bufs, P, xo, attn)
        # stale flags past live: no wait on them
        bufs.epoch += 1
        pub = torch.zeros(spec.BLOCKS, dtype=torch.int32)
        n = P["n"]
        for (t, kvh), units in P["parts"].items():
            u0 = (2 * t + kvh) * n
            pub[u0 : u0 + len(units)] = 1
        P["d_pub"] = pub.cuda()
        xo2, attn2 = _run(probe, bufs, P)
        _check(bufs, P, xo2, attn2)
        assert torch.equal(xo2.view(torch.int16), xo.view(torch.int16))
        bufs.epoch += 1


@needs_gfx950
@pytest.mark.parametrize("S,window", [(1, False), (8, False), (8, True)])
def test_combine_edge_stats(S, window):
    """(NEG, 0) units among the live ones, a (t, kvh) whose every live unit
    is (NEG, 0) (zeros), a unit whose max underflows exp2."""
    probe = _probe(S, window)
    bufs = context.ProbeBuffers(epoch=5)
    Ls, pads = CASES[S, window][0]
    P = _to_device(_edge_stats(_problem(S, window, Ls, 31, pads)))
    xo, attn = _run(probe, bufs, P)
    _check(bufs, P, xo, attn)
    assert (attn[0, : J * D].view(torch.int16) == 0).all()


@needs_gfx950
@pytest.mark.parametrize("S,window", [(1, False), (8, True)])
def test_combine_is_deterministic(S, window):
    """200 launches, the scratch poisoned and the epoch bumped between them:
    the same bytes every time (a stale cross-XCD read of a partial or a flag
    would show as NaN or a different row)."""
    probe = _probe(S, window)
    bufs = context.ProbeBuffers(epoch=9)
    Ls, pads = CASES[S, window][0]
    P = _to_device(_problem(S, window, Ls, 77, pads))
    xo0, attn0 = _run(probe, bufs, P)
    _check(bufs, P, xo0, attn0)
    for _ in range(200):
        bufs.epoch += 1
        xo, attn = _run(probe, bufs, P)
        assert torch.equal(xo.view(torch.int16), xo0.view(torch.int16))
        assert torch.equal(attn.view(torch.int16), attn0.view(torch.int16))
    _assert_healthy(bufs)


LATE_TICKS = 2000  # 20 us of s_memrealtime (100 MHz)


@needs_gfx950
@pytest.mark.parametrize("S,window", [(1, False), (8, True)])
def test_combine_waits_for_late_units(S, window):
    """Live units that store their partials and flag 20-80 us after the
    launch starts (every combine CTA is then polling): the same bytes as the
    on-time launch, 20 times. With no late producer every flag is already
    set at the first poll, so a merge load issued ahead of its flag would
    still read the right data; here it reads the poison (NaN)."""
    probe = _probe(S, window)
    bufs = context.ProbeBuffers(epoch=21)
    Ls, pads = CASES[S, window][0]
    P = _to_device(_problem(S, window, Ls, 55, pads))
    xo0, attn0 = _run(probe, bufs, P)
    _check(bufs, P, xo0, attn0)
    pub = P["pub"].clone()
    n = P["n"]
    for (t, kvh), units in P["parts"].items():
        for p in range(len(units)):
            pub[(2 * t + kvh) * n + p] = LATE_TICKS * (1 + p % 4)
    P["d_pub"] = pub.cuda()
    for _ in range(20):
        bufs.epoch += 1
        xo, attn = _run(probe, bufs, P)
        assert torch.equal(xo.view(torch.int16), xo0.view(torch.int16))
        assert torch.equal(attn.view(torch.int16), attn0.view(torch.int16))
    _check(bufs, P, xo, attn)


@needs_gfx950
def test_a_missing_live_flag_gives_up():
    """A live unit that never publishes: its combine CTAs give up after the
    budget (FAILED, the wait record names ``srdy``), and the launch returns
    with every crdy pair published."""
    S, window = 1, False
    probe = _probe(S, window)
    bufs = context.ProbeBuffers(epoch=11, budget=1)
    P = _to_device(_problem(S, window, [2048], 5))
    pub = P["pub"].clone()
    pub[3] = 0  # (t 0, kvh 0)'s unit 3 of 16 live
    P["d_pub"] = pub.cuda()
    _run(probe, bufs, P)
    assert bufs.failed()
    assert any("srdy" in w for w in bufs.given_up_waits()), bufs.given_up_waits()
    vals, tags = bufs.read_pairs("crdy", 0, 16 * S)
    assert (tags.cpu() == bufs.tag()).all()
