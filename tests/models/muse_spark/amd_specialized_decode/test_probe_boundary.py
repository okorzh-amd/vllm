# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1's residual boundary (``mono/front/boundary.py``, DESIGN_BC 5.4, WP-B2)
in probe kernels with K1's context (``peers=False``, ``route_lds=False``, a
``K1Build``): the boundary alone on its NA CTAs, and the boundary CTAs' entry
as K1 runs it (``boundary_loads`` + ``epilogue_inputs``, then the stage).

[CPU] Every probe compiles GPU-less for gfx950 with no private segment,
within the VGPR and LDS budget, without a readfirstlane waterfall, every
plain load issued before the first poll (rule 8: nothing but the stage's own
operands in flight at a poll); the trace records the hand-offs (``fss`` /
``nss`` put and polled, ``ardy`` put, no peer access). The test inputs make
the edges bite, checked on an fp64 host model of the stock order and on B0's
reference: a row where the post-FFN eps (1e-8) moves the bf16 rows, a row
where the pre-qkv eps (1e-5) does, the gain centers.
[1GPU] ``stream_out`` / ``arow`` vs the stock
``fused_norm_residual_norm_fwd`` (post weight first, both gain centers 1.0):
``stream_out`` within the magnitude-scaled bound of DESIGN_BC 4, ``arow``
bit-equal on >= 99.9 % and <= 1 ulp wherever the blend does not cancel (and
<= 1 ulp on 100 % vs the pre-qkv norm of the kernel's own ``stream_out``);
4 rows; the eps rows; zero weights; NaN pad rows -> exact zeros; the K1 tags
on every flag; the epilogue units' pad / position / slot words.
"""

import glob
import os

import pytest
import regex as re
import torch

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

import flydsl.expr as fx  # noqa: E402
from aiter.ops.flydsl.kernels import buffer_ops as bo  # noqa: E402

from tests.models.muse_spark.amd_specialized_decode import (  # noqa: E402
    reference,
)
from vllm.models.muse_spark.amd.specialized_decode.mono import spec  # noqa: E402
from vllm.models.muse_spark.amd.specialized_decode.mono.abi import (  # noqa: E402
    WAIT_BUDGET,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common import (  # noqa: E402
    context,
    plan,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (  # noqa: E402
    rsrc,
    traced,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.plan import (  # noqa: E402
    Space,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.front import (  # noqa: E402
    boundary,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.stages import (  # noqa: E402
    seam,
)

MT = spec.MAX_TOKENS
H = spec.HIDDEN
NA = spec.NA
LAYER = 5  # the probe launches are layer 5's K1 (its tag slot)
EPI_WORDS = 8  # words a CTA of the entry probe writes: its epilogue inputs


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

# ---------------------------------------------------------------- probe bodies


def _boundary_body(c):
    """Slice j = CTA b: the boundary's loads, then the stage."""
    j = c["bid"]
    pre = boundary.boundary_loads(c, j)
    boundary.stage_boundary(c, j, pre)


@traced
def _entry_body(c):
    """A weight-free K1 CTA's entry (DESIGN_BC 5.3): ``boundary_loads`` and
    ``epilogue_inputs`` first, the stage, then the epilogue inputs of CTA b
    -> ``out`` words b * 8 .. (t, g5, pad, pos lo / hi, slot lo / hi, 0),
    one store block (lanes 0 .. 7 of wave 0)."""
    b, tid = c["bid"], c["tid"]
    pre = boundary.boundary_loads(c, b)
    epi = boundary.epilogue_inputs(c, b)
    boundary.stage_boundary(c, b, pre)
    pos, slot = epi["pos"], epi["slot"]
    words = (
        epi["t"],
        epi["g5"],
        epi["pad"].select(fx.Int32(1), fx.Int32(0)),
        fx.Int32(pos),
        fx.Int32(pos >> 32),
        fx.Int32(slot),
        fx.Int32(slot >> 32),
    )
    v = fx.Int32(0)
    for k, w in enumerate(words):
        v = (tid == k).select(fx.Int32(w), v)
    if tid < EPI_WORDS:
        bo.buffer_store(v, rsrc(c["args"]["out"]), b * EPI_WORDS + tid)


BOUNDARY_ARGS = (
    "branch",
    "stream",
    "pad",
    "w_postffn",
    "ffn_alpha",
    "ffn_beta",
    "w_preqkv",
    "stream_out",
)
ENTRY_ARGS = BOUNDARY_ARGS + ("positions", "slots", "out")


def _probe(which: str, S: int) -> context.Probe:
    k1 = dict(
        peers=False,
        route_lds=False,
        build=spec.K1Build(tokens=S, rope=False),
        blocks=NA,
    )
    if which == "boundary":
        return context.probe_kernel(
            f"muse_probe_boundary_s{S}", S, _boundary_body, None, BOUNDARY_ARGS, **k1
        )
    assert which == "entry", which
    return context.probe_kernel(
        f"muse_probe_k1_entry_s{S}", S, _entry_body, None, ENTRY_ARGS, **k1
    )


# ---------------------------------------------------------------- [CPU]


def test_stage_contract():
    """The regions of 5.7 / 5.8, the ABI names the stage reads, the pair
    indexes it puts at."""
    assert set(boundary.STAGE_REGIONS) <= set(spec.K1_REGIONS)
    assert set(boundary.STAGE_REGIONS) <= set(spec.REGIONS)
    assert not set(boundary.STAGE_REGIONS) & set(spec.PEER_REGIONS)  # rank-local
    assert set(ENTRY_ARGS) - {"out"} <= set(spec.K1_ARGS)
    assert spec.AROW in spec.IN_LAUNCH_PLAIN  # stored and read at device scope
    # the per-slice partials fill fss / nss exactly; the flags ardy
    lay = spec.scratch_layout()
    assert (spec.ss_pair(NA - 1, MT - 1) + 1) * spec.PAIR_BYTES == lay["fss"][1]
    assert (spec.ss_pair(NA - 1, MT - 1) + 1) * spec.PAIR_BYTES == lay["nss"][1]
    assert (spec.ardy_pair(NA - 1) + 1) * spec.PAIR_BYTES == lay["ardy"][1]
    assert (spec.arow_elem(MT - 1, H - 2) + 2) * 2 == lay[spec.AROW][1]
    # every epilogue unit's token is a boundary CTA's, and its row group spans
    # the 5 row blocks (spec.epilogue_row)
    assert spec.E_UNITS * spec.MAX_ROWS <= NA
    assert seam.blend_norm is boundary.blend_norm  # one body, two stages


def _ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return reference.bf16_ulps(a, b)


def _boundary64(branch, stream, wpost, alpha, beta, wpre, post_eps, pre_eps, gain):
    """The stock order in fp64 from the fp32 inputs (x = RN(f32(branch)
    (w_post + gain)) rounded to fp32 as stock's product is): (stream', bf16
    attn_in)."""
    x = (branch.float() * (wpost.float() + gain)).double()
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + post_eps)
    n = alpha.double() * stream.double() + beta.double() * y
    r = torch.rsqrt(n.square().mean(-1, keepdim=True) + pre_eps)
    return n, (n * r * (wpre.double() + gain)).float().to(torch.bfloat16)


CASES = ("plain", "post_eps", "pre_eps", "zero_weight", "pad")


def _inputs(S: int, case: str, seed: int):
    """(branch bf16, stream fp32, w_post bf16, alpha, beta fp32, w_pre bf16,
    dead rows). ``post_eps``: row S - 1's branch x1e-4, so mean(x^2) ~ the
    post-FFN eps 1e-8; ``pre_eps``: the gate -2.5 on every channel (beta ~
    2.4e-4, alpha ~ 1) and row S - 1's stream x3e-3, so mean(stream'^2) ~ the
    pre-qkv eps 1e-5; ``zero_weight``: both weights 0 (the gain centers alone);
    ``pad``: rows 5 .. 7 pad with NaN branch and stream."""
    g = torch.Generator().manual_seed(seed)
    branch = torch.randn(S, H, generator=g) * 0.7
    stream = torch.randn(S, H, generator=g) * (1.0 if case == "pre_eps" else 2.0)
    gate = torch.randn(H, generator=g).to(torch.bfloat16).float()
    if case == "pre_eps":
        gate = torch.full_like(gate, -2.5)
    beta = torch.sigmoid(gate / 0.3)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-gate / 0.3) * (1 + beta), min=1e-3))
    wpost = torch.randn(H, generator=g) * 0.1
    wpre = torch.randn(H, generator=g) * 0.1
    if case == "zero_weight":
        wpost, wpre = torch.zeros(H), torch.zeros(H)
    if case == "post_eps":
        branch[S - 1] *= 1e-4
    if case == "pre_eps":
        stream[S - 1] *= 3e-3
    dead = (5, 6, 7) if case == "pad" else ()
    branch = branch.to(torch.bfloat16)
    for t in dead:
        branch[t] = float("nan")
        stream[t] = float("nan")
    return (
        branch,
        stream,
        wpost.to(torch.bfloat16),
        alpha,
        beta,
        wpre.to(torch.bfloat16),
        dead,
    )


@pytest.mark.parametrize("S,case", [(3, "post_eps"), (3, "pre_eps"), (4, "plain")])
def test_inputs_make_the_edges_bite(S, case):
    """The GPU cases are built where their edge matters: the eps under test
    (or a wrong one in its place) moves the eps row's bf16 rows, on the fp64
    model of the stock order; the model agrees with B0's reference. Where the
    blend cancels (``|stream'|`` far below ``|alpha s| + |beta y|``) a bf16
    row's ulp distance is unbounded, so the ulp checks skip those elements."""
    branch, stream, wpost, alpha, beta, wpre, _ = _inputs(S, case, 11 * S)
    eps = (spec.POST_FFN_EPS, spec.RMS_EPS)
    n, z = _boundary64(branch, stream, wpost, alpha, beta, wpre, *eps, spec.GAIN)
    rn, rz = reference.ffn_boundary(branch, stream, wpost, alpha, beta, wpre)
    tol = reference.stream_tolerance(branch, stream, wpost, alpha, beta)
    mag = tol.double() * 1e5  # |alpha s| + |beta y|
    assert ((rn.double() - n).abs() <= tol).all()
    u = _ulps(rz, z)
    assert ((u <= 1) | (n.abs() < 1e-3 * mag)).all(), u.max()
    assert (u == 0).float().mean().item() >= 0.999
    if case == "plain":
        # the gain centers: w + 0 instead of w + 1 moves every row
        _, z0 = _boundary64(branch, stream, wpost, alpha, beta, wpre, *eps, 0.0)
        assert (_ulps(z0, z) > 1).float().mean().item() > 0.5
        return
    swap = (
        (spec.RMS_EPS, spec.RMS_EPS)  # 1e-5 in place of the post-FFN 1e-8
        if case == "post_eps"
        else (spec.POST_FFN_EPS, spec.POST_FFN_EPS)  # 1e-8 in place of 1e-5
    )
    _, zs = _boundary64(branch, stream, wpost, alpha, beta, wpre, *swap, spec.GAIN)
    u = _ulps(zs[S - 1], z[S - 1])
    assert (u > 4).float().mean().item() > 0.5, u.float().mean()
    # only the eps row moves (past the cancelling elements, where the other
    # eps' 1e-5 relative change of y is amplified)
    u = _ulps(zs[: S - 1], z[: S - 1])
    flat = n[: S - 1].abs() >= 1e-2 * mag[: S - 1]
    assert (u[flat] <= 1).all(), u[flat].max()


# readfirstlane waterfall around a load whose base is not wave-uniform
_WATERFALL = re.compile(r"v_cmp_eq_u64_e\d+\s+\S+,\s*s\[\d+:\d+\],\s*v\[")
# a plain (non-mailbox) buffer load, and a mailbox poll (device scope, one
# pair a lane)
_PLAIN = re.compile(r"^\s*buffer_load_\w+\s(?![^\n]*\bsc1\b)", re.M)
_POLL = re.compile(r"^\s*buffer_load_\w+\s[^\n]*\boffen\b[^\n]*\bsc1\b", re.M)


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


def _body(txt: str, name: str) -> str:
    """The kernel's instructions: from its symbol to ``s_endpgm``."""
    start = re.search(rf"^{re.escape(name)}:", txt, re.M)
    assert start, name
    end = txt.index("s_endpgm", start.end())
    return txt[start.end() : end]


def _compile(probe: context.Probe) -> None:
    try:
        probe.compile()
    except RuntimeError as e:  # the load, past the compile, needs a device
        if "hipErrorNoDevice" not in str(e) and "no ROCm" not in str(e):
            raise


# plain loads before the first poll, at least: the boundary's 7 (stream,
# alpha, beta, the two weights, branch, pad) and, in the entry probe, the
# epilogue's 3 (pad, position, slot); the probe's argument table adds more
ENTRY_LOADS = {"boundary": 7, "entry": 10}


@compiles
@pytest.mark.parametrize(
    "which,S", [("boundary", 1), ("boundary", 4), ("boundary", 8), ("entry", 8)]
)
def test_compiles(which, S, tmp_path, monkeypatch):
    """The GPU-less compile gate (README, Residency), rule 8 in the ISA (every
    plain load issued before the first poll, whose ``vmcnt(0)`` drains them),
    the traced hand-offs."""
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    probe = _probe(which, S)
    with plan.recording() as rec:
        _compile(probe)
    txt, res = _isa(str(tmp_path), probe.name)
    print(f"{probe.name}: {res}")
    assert res["private_segment_fixed_size"] == 0, res
    assert res["next_free_vgpr"] <= spec.VGPR_MAX, res
    assert res["group_segment_fixed_size"] <= spec.LDS_BYTES, res
    # no LDS beyond the context's fixed ``red`` (the stage is in registers)
    assert res["group_segment_fixed_size"] <= context.lds_total_bytes(None, False)
    loops = _WATERFALL.findall(txt)
    assert not loops, f"{len(loops)} waterfall compares, e.g. {loops[0]}"
    body = _body(txt, probe.name)
    # two norms: the hardware rsq of each
    assert len(re.findall(r"v_rsq_f32", body)) >= 2
    # rule 8: every plain load is issued before the first poll (none after),
    # so a poll's vmcnt(0) waits only for the stage's own operands
    first_poll = _POLL.search(body)
    assert first_poll, "no mailbox poll"
    assert len(_PLAIN.findall(body[: first_poll.start()])) >= ENTRY_LOADS[which]
    late = _PLAIN.findall(body[first_poll.start() :])
    assert not late, late
    if not rec.accesses:  # traced earlier in this process (FlyDSL's own cache)
        return
    acc = {(a.stage, a.region, a.space, a.kind) for a in rec.accesses}
    assert not rec.unnamed
    for r in ("fss", "nss"):
        assert ("boundary", r, Space.SCRATCH, "put") in acc
        assert ("boundary", r, Space.SCRATCH, "poll") in acc
    assert ("boundary", "ardy", Space.SCRATCH, "put") in acc
    assert not any(a.space == Space.PEER for a in rec.accesses)
    assert {a.region for a in rec.accesses} == set(boundary.STAGE_REGIONS)
    assert rec.stages == ["boundary"]


# ---------------------------------------------------------------- [1GPU]


def _assert_healthy(bufs) -> None:
    """No wait gave up and no alarm (the header's ``WAIT_BUDGET`` word holds
    the probe's budget)."""
    assert not bufs.failed(), bufs.given_up_waits()
    h = bufs.health.cpu().clone()
    h[WAIT_BUDGET] = 0
    assert not h.any(), bufs.given_up_waits()
    assert not bufs.flags().any()
    assert not bufs.peer.any()  # rank-local: no peer traffic


def _pad(S: int, rows) -> torch.Tensor:
    """The pad flags, 8 bytes (``ld_pad`` reads whole words)."""
    p = torch.zeros(MT, dtype=torch.bool)
    p[list(rows)] = True
    assert all(r < S for r in rows)
    return p.cuda()


def _arow(bufs) -> torch.Tensor:
    buf, off = bufs.where(spec.AROW)
    n = spec.scratch_layout()[spec.AROW][1]
    return buf[off : off + n].cpu().view(torch.bfloat16).view(MT, H)


def _run(probe, bufs, S, inputs, **extra):
    branch, stream, wpost, alpha, beta, wpre, dead = inputs
    stream_out = torch.full((S, H), 7.0, device="cuda")
    probe(
        bufs,
        branch=branch.cuda(),
        stream=stream.cuda(),
        pad=_pad(S, dead),
        w_postffn=wpost.cuda(),
        ffn_alpha=alpha.cuda(),
        ffn_beta=beta.cuda(),
        w_preqkv=wpre.cuda(),
        stream_out=stream_out,
        **extra,
    )
    torch.accelerator.synchronize()
    return stream_out.cpu()


def _check_boundary(bufs, S, inputs, got_stream) -> None:
    """``stream_out`` and the ``arow`` rows against the stock boundary; pad
    rows exact zeros, rows past S untouched; every slice published with this
    launch's (K1) tag."""
    from vllm.models.muse_spark.common.ops.fused_rmsnorm import (
        fused_norm_residual_norm_fwd,
    )

    branch, stream, wpost, alpha, beta, wpre, dead = inputs
    live = [t for t in range(S) if t not in dead]
    ref_in, ref_stream = fused_norm_residual_norm_fwd(
        branch.cuda(),
        stream.cuda(),
        wpost.cuda(),
        wpre.cuda(),
        spec.POST_FFN_EPS,
        spec.RMS_EPS,
        alpha=alpha.cuda(),
        beta=beta.cuda(),
        post_gain_center=spec.GAIN,
        pre_gain_center=spec.GAIN,
        post_input_scale=True,
        out_dtype=torch.bfloat16,
    )
    ref_in, ref_stream = ref_in.cpu(), ref_stream.cpu()
    arow = _arow(bufs)
    # stream_out: fp32 up to the sum-of-squares order and the hardware rsq,
    # bounded by the magnitude of the blend's terms (DESIGN_BC 4)
    tol = reference.stream_tolerance(branch, stream, wpost, alpha, beta)
    err = (got_stream - ref_stream).abs()
    assert (err[live] <= tol[live] + 1e-30).all(), err[live].max()
    # attn_in vs the pre-qkv norm (fp64) of the kernel's own stream_out: only
    # the rsq and the sum order differ, so <= 1 bf16 ulp on 100 % (an eps or
    # gain-center mistake moves it by O(1))
    n = got_stream.double()
    rs = torch.rsqrt(n.square().mean(-1, keepdim=True) + spec.RMS_EPS)
    own = (n * rs * (wpre.double() + spec.GAIN)).float().to(torch.bfloat16)
    u = _ulps(arow[live], own[live])
    assert u.max().item() <= 1, u.max()
    # attn_in vs stock: bit-equal on >= 99.9 %, <= 1 ulp wherever the blend
    # does not cancel (where it does, stream'`s relative error is unbounded)
    u = _ulps(arow[live], ref_in[live])
    frac = (u == 0).float().mean().item()
    print(f"S={S}: arow bit-equal to stock on {frac:.6f}")
    assert frac >= 0.999
    mag = tol * 1e5  # |alpha s| + |beta y|
    cancels = ref_stream[live].abs() < 1e-3 * mag[live]
    assert ((u <= 1) | cancels).all(), u.max()
    for t in dead:
        assert (got_stream[t] == 0).all() and (arow[t].view(torch.int16) == 0).all()
    assert (arow[S:].view(torch.int16) == 0).all()
    vals, tags = bufs.read_pairs("ardy", 0, NA)
    assert (vals == 1).all() and (tags == bufs.tag()).all()
    for r in ("fss", "nss"):
        _, tags = bufs.read_pairs(r, 0, NA * MT)
        tags = tags.cpu().view(NA, MT)
        assert (tags[:, :S] == bufs.tag()).all() and (tags[:, S:] == 0).all()
    for r in ("ass", "bss", "xrdy"):  # K2's seam regions: untouched
        _, tags = bufs.read_pairs(r, 0, 1)
        assert (tags == 0).all()


@needs_gfx950
@pytest.mark.parametrize(
    "S,case,epoch",
    [
        (4, "plain", 1),
        (1, "plain", 2),
        (3, "post_eps", 3),
        (3, "pre_eps", 4),
        (8, "zero_weight", 5),
        (8, "pad", 6),
        (8, "plain", (1 << 22) + 7),
    ],
)
def test_boundary_probe(S, case, epoch):
    """The boundary on its NA CTAs vs the stock boundary: 4 rows; one row;
    the post-FFN eps row (branch x1e-4); the pre-qkv eps row (stream x3e-3,
    beta ~ 2.4e-4); zero weights (the gain centers); NaN pad rows -> exact
    zeros; an epoch past the 22-bit wrap."""
    probe = _probe("boundary", S)
    bufs = context.ProbeBuffers(epoch=epoch, slot=spec.k1_slot(LAYER))
    inputs = _inputs(S, case, 11 * S)
    got_stream = _run(probe, bufs, S, inputs)
    _assert_healthy(bufs)
    _check_boundary(bufs, S, inputs, got_stream)
    if case in ("post_eps", "pre_eps"):  # the premise, on the kernel's rows
        x = inputs[0][S - 1].double() * (inputs[2].double() + spec.GAIN)
        ms = x if case == "post_eps" else got_stream[S - 1].double()
        ms = ms.square().mean()
        eps = spec.POST_FFN_EPS if case == "post_eps" else spec.RMS_EPS
        assert ms.item() < 10 * eps, ms.item()


@needs_gfx950
@pytest.mark.parametrize("S,dead", [(3, ()), (8, (2, 7))])
def test_entry_probe(S, dead):
    """The weight-free CTAs' entry as K1 runs it: the boundary is unchanged
    by the epilogue loads beside it, and CTA b < E_UNITS S holds token b //
    E_UNITS's pad flag, position and slot (int64: a position past 2^31, slot
    -1, a slot past 2^32); CTAs past E_UNITS S load row S - 1."""
    probe = _probe("entry", S)
    bufs = context.ProbeBuffers(epoch=12, slot=spec.k1_slot(LAYER))
    inputs = (*_inputs(S, "plain", 7 + S)[:6], dead)
    for t in dead:
        inputs[0][t] = float("nan")
        inputs[1][t] = float("nan")
    positions = torch.tensor([3, (1 << 31) + 5, 2047, 9, 0, 1 << 40, 77, 12])[:S]
    slots = torch.tensor([17, -1, (1 << 32) + 3, 0, 15, 16, -1, 4095])[:S]
    out = torch.full((NA * EPI_WORDS,), -7, dtype=torch.int32, device="cuda")
    got_stream = _run(
        probe, bufs, S, inputs, positions=positions.cuda(), slots=slots.cuda(), out=out
    )
    _assert_healthy(bufs)
    _check_boundary(bufs, S, inputs, got_stream)
    got = out.cpu().view(NA, EPI_WORDS)
    for b in range(NA):
        t = min(b // spec.E_UNITS, S - 1)
        p, s = int(positions[t]), int(slots[t])
        want = [
            t,
            b % spec.E_UNITS,
            int(t in dead),
            p & 0xFFFFFFFF,
            p >> 32,
            s & 0xFFFFFFFF,
            s >> 32,
            0,
        ]
        want = [w - (1 << 32) if w >= 1 << 31 else w for w in want]
        assert got[b].tolist() == want, (b, got[b].tolist(), want)
