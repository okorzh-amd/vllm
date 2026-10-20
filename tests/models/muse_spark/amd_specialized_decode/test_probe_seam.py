# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2 phases 1 and 2a in probe kernels (``context.probe_kernel``): o_proj and
the AR push (``mono/attention/back.py``), the AR reduce and the attention
boundary (``mono/stages/seam.py``), alone and in K2's program order.

[CPU] Every probe compiles GPU-less for gfx950 with no private segment, within
the VGPR and LDS budget; the trace records the stages' hand-offs (``ar``
pushed by o_proj and polled by the seam, ``ass`` / ``bss`` / ``xrdy``).
[1GPU] o_proj partials vs fp64 and ``F.linear`` (<= 1 bf16 ulp up to fp32
accumulation noise); the seam's ``stream_out`` / ``xrow`` vs
``fused_norm_residual_norm_fwd`` on the attention output rebuilt in rank order
from the ``ar`` pairs (and ``xrow`` vs the pre-FFN norm of the kernel's own
``stream_out``), incl. a row at the post-attention eps and an all-zero pre-FFN
weight; pad rows with NaN inputs -> exact zeros; no wait given up, no alarm.
With aliased peers a push to "every rank" lands in this rank's own slot; the
test writes the other ranks' pairs.
"""

import glob
import math
import os

import pytest
import regex as re
import torch

flydsl = pytest.importorskip("flydsl")
pytest.importorskip("aiter")

from vllm.models.muse_spark.amd.specialized_decode.mono import spec  # noqa: E402
from vllm.models.muse_spark.amd.specialized_decode.mono.abi import (  # noqa: E402
    WAIT_BUDGET,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.attention import (  # noqa: E402
    back,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common import (  # noqa: E402
    context,
    plan,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (  # noqa: E402
    traced,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.plan import (  # noqa: E402
    Space,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.stages import (  # noqa: E402
    seam,
)

MT = spec.MAX_TOKENS
H = spec.HIDDEN
OWN = 3  # the probe's rank


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


def _o_proj(c):
    """Phase 1 as K2 runs it: both tasks' weights first, the core, the two
    GEMVs and pushes."""
    b = c["bid"]
    ops0 = back.o_loads(c, b)
    ops1 = back.o_loads(c, b + spec.BLOCKS)
    back.load_core(c)
    back.stage_o(c, b, ops0)
    back.stage_o(c, b + spec.BLOCKS, ops1)


def _seam(c, j):
    pre = seam.seam_loads(c, j)
    seam.stage_seam(c, j, pre)


def _oproj_body(c):
    _o_proj(c)


def _seam_body(c):
    _seam(c, c["bid"])


@traced
def _back_seam_body(c):
    """K2's program order through phase 2a: every CTA's o_proj, then the seam
    on CTAs [0, NA)."""
    _o_proj(c)
    if c["bid"] < spec.NA:
        _seam(c, c["bid"])


OPROJ_ARGS = ("core", "pad", "w_o")
SEAM_ARGS = ("stream", "pad", "attn_alpha", "attn_beta", "w_preffn", "stream_out")
BACK_SEAM_ARGS = ("core", "w_o") + SEAM_ARGS


def _probe(which: str, S: int) -> context.Probe:
    if which == "oproj":
        return context.probe_kernel(
            f"muse_probe_oproj_s{S}",
            S,
            _oproj_body,
            {"back": back.lds_struct(S)},
            OPROJ_ARGS,
            blocks=spec.BLOCKS,
        )
    if which == "seam":
        return context.probe_kernel(
            f"muse_probe_seam_s{S}",
            S,
            _seam_body,
            {"seam": seam.lds_struct(S)},
            SEAM_ARGS,
            blocks=spec.NA,
        )
    assert which == "back_seam", which
    return context.probe_kernel(
        f"muse_probe_back_seam_s{S}",
        S,
        _back_seam_body,
        {"back": back.lds_struct(S), "seam": seam.lds_struct(S)},
        BACK_SEAM_ARGS,
        blocks=spec.BLOCKS,
    )


# ---------------------------------------------------------------- [CPU]


def test_stage_contract():
    """The LDS members of 5.13, the regions of 5.10, the build-time asserts."""
    assert back.lds_struct(8).__dsl_size_of__() == 16512
    assert back.lds_struct(1).__dsl_size_of__() == 1032 * 2
    assert back.lds_struct(8) is back.lds_struct(8)
    assert seam.lds_struct(8).__dsl_size_of__() == 256
    lds = {"back": back.lds_struct(8), "seam": seam.lds_struct(8)}
    assert context.lds_total_bytes(lds) <= spec.LDS_BYTES
    assert set(back.STAGE_REGIONS) <= set(spec.PEER_REGIONS)
    assert set(seam.STAGE_REGIONS) <= set(spec.REGIONS)
    # the seam's per-slice partials fit their regions at every row
    assert spec.ss_pair(spec.NA - 1, MT - 1) + 1 == spec.NA * MT
    # the last o_proj push lands in the last pair of the AR half
    last = spec.ar_pair(spec.TP - 1, MT - 1, H - 2)
    assert (last + 1) * spec.PAIR_BYTES == spec.peer_layout()["ar"][1]


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


@compiles
@pytest.mark.parametrize(
    "which,S",
    [("oproj", 8), ("seam", 8), ("back_seam", 1), ("back_seam", 3), ("back_seam", 8)],
)
def test_compiles(which, S, tmp_path, monkeypatch):
    """The GPU-less compile gate (README, Residency) and the traced hand-offs."""
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
    if which != "seam":
        assert "v_mfma_f32_16x16x32_bf16" in txt
        # red (8 KiB) + the core rows
        assert res["group_segment_fixed_size"] >= back.lds_struct(S).__dsl_size_of__()
    if which != "oproj":
        assert "v_rsq_f32" in txt
    if not rec.accesses:  # traced earlier in this process (FlyDSL's own cache)
        return
    acc = {(a.stage, a.region, a.space, a.kind) for a in rec.accesses}
    assert not rec.unnamed
    if which != "seam":
        assert ("o_proj", "ar", Space.PEER, "put") in acc
        assert {a.region for a in rec.accesses if a.stage == "o_proj"} == {"ar"}
    if which != "oproj":
        assert ("seam", "ar", Space.PEER, "poll") in acc
        for r in ("ass", "bss"):
            assert ("seam", r, Space.SCRATCH, "put") in acc
            assert ("seam", r, Space.SCRATCH, "poll") in acc
        assert ("seam", "xrdy", Space.SCRATCH, "put") in acc
        seen = {a.region for a in rec.accesses if a.stage == "seam"}
        assert seen == set(seam.STAGE_REGIONS)
    if which == "back_seam":
        assert rec.stages[:3] == ["o_proj"] * 3 and rec.stages[3] == "seam"


# ---------------------------------------------------------------- host helpers


def _ord(x: torch.Tensor) -> torch.Tensor:
    """bf16 -> integers in value order (+0 and -0 both 0)."""
    b = x.contiguous().view(torch.int16).int()
    m = b & 0x7FFF
    return torch.where(b < 0, -m, m)


def _ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (_ord(a) - _ord(b)).abs()


def _assert_healthy(bufs) -> None:
    """No wait gave up and no alarm: FAILED and every CTA's wait record zero.
    (The header's ``WAIT_BUDGET`` word holds the probe's budget, so the health
    words as a whole are never all zero.)"""
    assert not bufs.failed(), bufs.given_up_waits()
    h = bufs.health.cpu().clone()
    h[WAIT_BUDGET] = 0
    assert not h.any(), bufs.given_up_waits()
    assert not bufs.flags().any()


def _check_gemv(got: torch.Tensor, ref: torch.Tensor, x, w) -> None:
    """bf16 GEMV rows ``got`` vs ``ref`` (both bf16 [rows, n], CPU) of
    ``x @ w.T``: <= 1 bf16 ulp on >= 99.9 %, and past 1 ulp only within fp32
    accumulation noise (``|got - ref| <= 2^-20 sum_k |x w|``): two fp32 dots
    summed in different orders can land 2 ulps apart on a cancelling
    (near-zero) sum, where a bf16 ulp is far below the fp32 rounding error."""
    u = _ulps(got, ref)
    absdot = (x.double().abs() @ w.double().abs().T).cpu()
    noise = (got.double() - ref.double()).abs() <= 2**-20 * absdot
    assert ((u <= 1) | noise).all(), u.max()
    # >= 99.9 % within 1 ulp, as a count with a floor of 2: at S = 1 a probe
    # has only 512 outputs, where one cancelling dot (bounded by the noise
    # rule above) would otherwise fail the fraction
    assert (u > 1).sum().item() <= max(2, math.ceil(0.001 * u.numel())), u.max()


def _pad(S: int, rows) -> torch.Tensor:
    """The pad flags, 8 bytes (``ld_pad`` reads whole words)."""
    p = torch.zeros(MT, dtype=torch.bool)
    p[list(rows)] = True
    assert all(r < S for r in rows)
    return p.cuda()


def _ar_rows(bufs, src: int, S: int):
    """Rank ``src``'s AR slot as read on this rank: (bf16 [MT, H], tags [MT, H/2])."""
    vals, tags = bufs.read_pairs("ar", spec.ar_pair(src, 0, 0), MT * H // 2)
    rows = vals.cpu().contiguous().view(torch.bfloat16).view(MT, H)
    return rows, tags.cpu().view(MT, H // 2)


def _write_ar(bufs, src: int, rows: torch.Tensor) -> None:
    """Rank ``src``'s partials (bf16 [S, H]) in this rank's AR region."""
    bufs.write_pairs("ar", spec.ar_pair(src, 0, 0), rows.contiguous().to(bufs.device))


def _rank_order(parts: torch.Tensor) -> torch.Tensor:
    """The seam's attn_out: the 8 bf16 partials [TP, S, H] summed in fp32 in
    rank order, rounded to bf16 once (the K3 custom-AR rule)."""
    acc = parts[0].float()
    for src in range(1, spec.TP):
        acc = acc + parts[src].float()
    return acc.to(torch.bfloat16)


def _boundary_inputs(S: int, g: torch.Generator, zero_w: bool, low_gate=False):
    """``low_gate``: the gate -2.5 on every channel (beta ~ 2.4e-4, alpha ~ 1),
    so a row with a small stream has a small pre-FFN norm input."""
    stream = torch.randn(S, H, generator=g) * 2
    gate = torch.randn(H, generator=g).to(torch.bfloat16).float()
    if low_gate:
        gate = torch.full_like(gate, -2.5)
    beta = torch.sigmoid(gate / 0.3)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-gate / 0.3) * (1 + beta), min=1e-3))
    w = torch.zeros(H) if zero_w else torch.randn(H, generator=g) * 0.1
    return stream, alpha, beta, w.to(torch.bfloat16)


def _check_boundary(bufs, S, attn_out, stream, alpha, beta, w, stream_out, dead):
    """``stream_out`` and the ``xrow`` rows against the stock boundary on the
    kernel's own attn_out; pad rows exact zeros; every slice published."""
    from vllm.models.muse_spark.common.ops.fused_rmsnorm import (
        fused_norm_residual_norm_fwd,
    )

    live = [t for t in range(S) if t not in dead]
    ref_in, ref_stream = fused_norm_residual_norm_fwd(
        attn_out.cuda(),
        stream.cuda(),
        None,
        w.cuda(),
        spec.POST_ATTN_EPS,
        spec.PRE_FFN_EPS,
        alpha=alpha.cuda(),
        beta=beta.cuda(),
        post_gain_center=0.0,
        pre_gain_center=spec.GAIN,
        out_dtype=torch.bfloat16,
    )
    ref_in, ref_stream = ref_in.cpu(), ref_stream.cpu()
    got_stream = stream_out.cpu()
    xbuf, xoff = bufs.where(spec.XROW)
    xrow = xbuf[xoff : xoff + spec.XROW_BYTES].cpu().view(torch.bfloat16).view(MT, H)
    # stream_out: fp32 up to the sum-of-squares order and the hardware rsq;
    # bounded by the magnitude of the blend's terms (alpha s + beta y can
    # cancel, so a bound relative to the result alone is not sound)
    x = attn_out.float()
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + spec.POST_ATTN_EPS)
    mag = (alpha * stream).abs() + (beta * y).abs()
    err = (got_stream - ref_stream).abs()
    assert (err[live] <= 1e-5 * mag[live] + 1e-30).all(), err[live].max()
    # ffn_in vs the pre-FFN norm (fp64) of the kernel's own stream_out: only
    # the rsq and the sum order differ, a common factor within ~1e-7, so
    # <= 1 bf16 ulp on 100 % (an eps or gain-center mistake moves it by O(1))
    n = got_stream.double()
    rs = torch.rsqrt(n.square().mean(-1, keepdim=True) + spec.PRE_FFN_EPS)
    own = (n * rs * (w.double() + spec.GAIN)).float().to(torch.bfloat16)
    u = _ulps(xrow[live], own[live])
    assert u.max().item() <= 1, u.max()
    # ffn_in vs the stock boundary: bit-equal on >= 99.9 %, <= 1 ulp wherever
    # the blend does not cancel (|stream'| >= 1e-3 (|alpha s| + |beta y|): there
    # stream_out's error above is < 2^-8 of the value). Under cancellation the
    # relative error of stream' is not bounded, so neither is ffn_in's ulp gap.
    u = _ulps(xrow[live], ref_in[live])
    assert (u == 0).float().mean().item() >= 0.999
    cancels = ref_stream[live].abs() < 1e-3 * mag[live]
    assert ((u <= 1) | cancels).all(), u.max()
    for t in dead:
        assert (got_stream[t] == 0).all() and (xrow[t].view(torch.int16) == 0).all()
    vals, tags = bufs.read_pairs("xrdy", 0, spec.NA)
    assert (vals == 1).all() and (tags == bufs.tag()).all()
    for r in ("ass", "bss"):
        _, tags = bufs.read_pairs(r, 0, spec.NA * MT)
        tags = tags.cpu().view(spec.NA, MT)
        assert (tags[:, :S] == bufs.tag()).all() and (tags[:, S:] == 0).all()


def test_host_helpers():
    """The tests' own plumbing, on CPU memory: bf16 ulp distance across signs
    and zeros; a rank's AR slot written and read back at ``spec.ar_pair``."""
    a = torch.tensor([1.0, -1.0, 0.0, -0.0, 2.0], dtype=torch.bfloat16)
    up = torch.tensor([1.0078125, -1.0078125, -0.0, 0.0, 1.9921875])
    assert _ulps(a, up.to(torch.bfloat16)).tolist() == [1, 1, 0, 0, 1]
    tiny = torch.tensor([9.2e-41], dtype=torch.bfloat16)  # a positive subnormal
    assert _ulps(-tiny, tiny).item() == 2 * _ord(tiny).item()
    bufs = context.ProbeBuffers("cpu", epoch=4, rank=OWN)
    S = 3
    rows = torch.randn(S, H).to(torch.bfloat16)
    _write_ar(bufs, 6, rows)
    got, tags = _ar_rows(bufs, 6, S)
    assert torch.equal(got[:S].view(torch.int16), rows.view(torch.int16))
    assert (tags[:S] == bufs.tag()).all() and (tags[S:] == 0).all()
    assert (_ar_rows(bufs, 5, S)[1] == 0).all()
    parts = torch.randn(spec.TP, S, H).to(torch.bfloat16)
    want = parts.float()[0]
    for src in range(1, spec.TP):
        want = want + parts.float()[src]
    assert torch.equal(_rank_order(parts), want.to(torch.bfloat16))


# ---------------------------------------------------------------- [1GPU]


@needs_gfx950
@pytest.mark.parametrize(
    "S,dead,epoch",
    [(1, (), 1), (3, (), 2), (8, (5, 6, 7), 3), (8, (), (1 << 22) + 4)],
)
def test_oproj_gemv_probe(S, dead, epoch):
    """o_proj partials vs fp64 and ``F.linear`` (``_check_gemv``), in this
    rank's AR slot of this epoch's parity half; pad rows (NaN core) exact zeros; rows
    past S and the other ranks' slots untouched."""
    probe = _probe("oproj", S)
    bufs = context.ProbeBuffers(epoch=epoch, rank=OWN)
    g = torch.Generator().manual_seed(S * 100 + epoch % 7)
    w_o = (torch.randn(H, spec.CORE, generator=g) * 0.03).to(torch.bfloat16)
    core = torch.randn(S, spec.CORE, generator=g).to(torch.bfloat16)
    for t in dead:
        core[t] = float("nan")
    core, w_o = core.cuda(), w_o.cuda()
    probe(bufs, core=core, pad=_pad(S, dead), w_o=w_o)
    torch.accelerator.synchronize()
    _assert_healthy(bufs)
    rows, tags = _ar_rows(bufs, OWN, S)
    live = [t for t in range(S) if t not in dead]
    ref64 = (core[live].double() @ w_o.double().T).float().cpu().to(torch.bfloat16)
    lin = torch.nn.functional.linear(core[live], w_o).cpu()
    _check_gemv(rows[live], ref64, core[live], w_o)
    _check_gemv(rows[live], lin, core[live], w_o)
    for t in dead:
        assert (rows[t].view(torch.int16) == 0).all()
    assert (tags[:S] == bufs.tag()).all() and (tags[S:] == 0).all()
    for src in range(spec.TP):
        if src != OWN:
            assert (_ar_rows(bufs, src, S)[1] == 0).all()
    par = (epoch & spec.EPOCH_MASK) & 1
    other = context.PEER_HEAD + spec.peer_half_bytes() * (1 - par)
    assert not bufs.peer[other : other + spec.peer_half_bytes()].any()


@needs_gfx950
@pytest.mark.parametrize(
    "S,case",
    [
        (1, "plain"),
        (3, "eps_row"),
        (8, "zero_weight"),
        (8, "pad"),
        (8, "eps_row"),
        (3, "pre_ffn_eps"),
    ],
)
def test_seam_probe(S, case):
    """The AR reduce and the boundary on given partials (all 8 ranks' slots
    written by the test): ``stream_out`` / ``xrow`` vs the stock boundary on
    the rank-order attn_out; a row at the post-attention eps; a row whose
    pre-FFN norm input is at the pre-FFN eps; an all-zero pre-FFN weight (gain
    1); NaN pad rows -> zeros."""
    probe = _probe("seam", S)
    bufs = context.ProbeBuffers(epoch=5 + S, rank=OWN)
    g = torch.Generator().manual_seed(17 * S + len(case))
    parts = (torch.randn(spec.TP, S, H, generator=g) * 0.5).to(torch.bfloat16)
    dead = (5, 6, 7) if case == "pad" else ()
    if case == "eps_row":  # mean(attn_out^2) ~ POST_ATTN_EPS
        parts[:, S - 1] = (parts[:, S - 1].float() * 1e-4).to(torch.bfloat16)
    stream, alpha, beta, w = _boundary_inputs(
        S, g, case == "zero_weight", low_gate=case == "pre_ffn_eps"
    )
    if case == "pre_ffn_eps":
        # alpha s + beta y with beta ~ 2.4e-4: mean square ~9e-6 ~ PRE_FFN_EPS,
        # where a swapped 1e-8 moves the norm's rsqrt ~1.45x
        stream[S - 1] *= 1.5e-3
    for t in dead:
        parts[:, t] = float("nan")
        stream[t] = float("nan")
    for src in range(spec.TP):
        _write_ar(bufs, src, parts[src])
    stream_out = torch.full((S, H), 7.0, device="cuda")
    probe(
        bufs,
        stream=stream.cuda(),
        pad=_pad(S, dead),
        attn_alpha=alpha.cuda(),
        attn_beta=beta.cuda(),
        w_preffn=w.cuda(),
        stream_out=stream_out,
    )
    torch.accelerator.synchronize()
    _assert_healthy(bufs)
    attn_out = _rank_order(parts)
    _check_boundary(bufs, S, attn_out, stream, alpha, beta, w, stream_out, dead)
    if case == "pre_ffn_eps":  # the premise: the eps swap would move the norm
        ms = stream_out[S - 1].double().square().mean()
        bite = torch.sqrt((ms + spec.PRE_FFN_EPS) / (ms + spec.POST_ATTN_EPS))
        assert bite.item() > 1.25, ms.item()


@needs_gfx950
@pytest.mark.parametrize("S,dead", [(1, ()), (8, (5, 6, 7))])
def test_back_seam_probe(S, dead):
    """Phases 1 and 2a in K2's order on the full grid: this rank's partials
    from the kernel, the other ranks' from the test; the boundary is checked
    on the rank-order sum of the ``ar`` readback (what the seam polled); NaN
    pad rows in the core, the stream and the peers' partials -> zeros."""
    probe = _probe("back_seam", S)
    bufs = context.ProbeBuffers(epoch=9, rank=OWN)
    g = torch.Generator().manual_seed(1000 + S)
    w_o = (torch.randn(H, spec.CORE, generator=g) * 0.03).to(torch.bfloat16)
    core = torch.randn(S, spec.CORE, generator=g).to(torch.bfloat16)
    parts = (torch.randn(spec.TP, S, H, generator=g) * 0.5).to(torch.bfloat16)
    stream, alpha, beta, w = _boundary_inputs(S, g, False)
    for t in dead:
        core[t] = float("nan")
        stream[t] = float("nan")
        parts[:, t] = float("nan")
    for src in range(spec.TP):
        if src != OWN:
            _write_ar(bufs, src, parts[src])
    stream_out = torch.full((S, H), 7.0, device="cuda")
    probe(
        bufs,
        core=core.cuda(),
        w_o=w_o.cuda(),
        stream=stream.cuda(),
        pad=_pad(S, dead),
        attn_alpha=alpha.cuda(),
        attn_beta=beta.cuda(),
        w_preffn=w.cuda(),
        stream_out=stream_out,
    )
    torch.accelerator.synchronize()
    _assert_healthy(bufs)
    live = [t for t in range(S) if t not in dead]
    own, tags = _ar_rows(bufs, OWN, S)
    lin = torch.nn.functional.linear(core[live].cuda(), w_o.cuda()).cpu()
    _check_gemv(own[live], lin, core[live], w_o)
    assert (tags[:S] == bufs.tag()).all()
    parts[OWN] = own[:S]
    for t in dead:
        assert (own[t].view(torch.int16) == 0).all()
    _check_boundary(
        bufs, S, _rank_order(parts), stream, alpha, beta, w, stream_out, dead
    )
