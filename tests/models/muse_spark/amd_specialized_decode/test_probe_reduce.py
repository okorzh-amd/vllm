# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2 phase 4 (``mono.stages.reduce`` / ``unpool``): the
column-owner per-route reduce, the post-expert norm's sum-of-squares exchange,
the route weighting and top-k sum, the unpooler and the branch gather.

[CPU] The hand-off protocol on the host with the stages' own index helpers:
every rsx / pss / sqx / upart / moe / branch pair that is polled has exactly
one producer, and the all-gathers tile their regions once. A host mirror of
the owner's summation order against an fp64 stock formula (incl. the
eps-scaled route, the zero weight, pad rows and NaN in dead picks), and of the
unpooler's K split against one GEMM. Both probes compile GPU-less for gfx950
with no private segment, within the VGPR and LDS budget, and touch exactly
their ``STAGE_REGIONS``.
[1GPU] ``test_finalize_probe``: owner CTAs on fake 8-rank partials (every src
slot of ``rsx`` and the other ranks' ``sqx`` written by the test) against
``fused_post_expert_finalize`` (6.2 finalize row). ``test_unpool_probe``: the
full grid's unpooler + gather against the stock GEMM (6.2 unpooler row).
"""

import glob
import os

import pytest
import regex as re
import torch

flydsl = pytest.importorskip("flydsl")
pytest.importorskip("aiter")

import flydsl.expr as fx  # noqa: E402
from flydsl.expr import gpu  # noqa: E402

from vllm.models.muse_spark.amd.specialized_decode.mono import spec  # noqa: E402
from vllm.models.muse_spark.amd.specialized_decode.mono.common import (  # noqa: E402
    context,
    plan,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (  # noqa: E402
    ld_i32,
    traced,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.stages import (  # noqa: E402
    reduce,
    unpool,
)

TP, MT, MP, TOPK = spec.TP, spec.MAX_TOKENS, spec.MAX_PICKS, spec.TOPK
H, OWN = spec.MOE_HIDDEN, spec.OWN_COLS


def _on_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


def _compiles_for_gfx950() -> bool:
    """A GPU-less compile must target gfx950 (FlyDSL falls back to gfx942
    without a device)."""
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


@traced
def _owner_body(c):
    """Owner CTA j = bid of the probe's rank: the route table (as ``load_route``
    leaves it) from ``route``, then ``stage_owner``."""
    tid = c["tid"]
    if tid < c["S"] * 16:
        fx.ptr_store(ld_i32(c["args"]["route"], tid), c["rt"]["route"] + tid)
    gpu.barrier()
    reduce.stage_owner(c, c["bid"])


@traced
def _unpool_body(c):
    """The full grid's phase 4b in K2's order: weights, unpooler, gather."""
    b = c["bid"]
    ops = unpool.unpool_loads(c, b)
    unpool.stage_unpool(c, b, ops)
    if b < spec.G_TASKS:
        unpool.stage_gather(c, b)


def _owner_probe(S: int, key: tuple = ()):
    return context.probe_kernel(
        f"muse_probe_owner_s{S}",
        S,
        _owner_body,
        {"reduce": reduce.lds_struct(S)},
        ("pad", "route", "w_postexp"),
        blocks=spec.RED_TASKS,
        key=key,
    )


def _unpool_probe(S: int, key: tuple = ()):
    return context.probe_kernel(
        f"muse_probe_unpool_s{S}",
        S,
        _unpool_body,
        {"unpool": unpool.lds_struct(S)},
        ("pad", "w_unpool", "branch_out"),
        blocks=spec.BLOCKS,
        key=key,
    )


# ---------------------------------------------------------------- host mirrors


def _butterfly8(v: torch.Tensor) -> torch.Tensor:
    """``butterfly(v, (4, 2, 1))`` over the last dim (8 lanes), lane 0's order
    (every lane gets the same bits: the adds are commutative)."""
    a = v[..., 0:4] + v[..., 4:8]
    b = a[..., 0:2] + a[..., 2:4]
    return b[..., 0] + b[..., 1]


def _rank_sum(parts: torch.Tensor) -> torch.Tensor:
    """fp32 sum over dim 0 in index order (rank order)."""
    acc = parts[0].float()
    for i in range(1, parts.shape[0]):
        acc = acc + parts[i].float()
    return acc


def owner_mirror(part, w, tw, pad):
    """The owner stage's math in its order, every rank's columns at once.

    ``part``: bf16 [TP, S, 8, 4096] (src rank, row, pick k, column), ``w``:
    bf16 [4096], ``tw``: fp32 [S, 8], ``pad``: bool [S]. -> (moe bf16 [S,
    4096], ss fp32 [S, 8], own fp32 [TP, S, 8] (each owner's ``sqx``))."""
    S = part.shape[1]
    R = _rank_sum(part)  # [S, 8, 4096]
    x = R * (w.float() + spec.GAIN)
    sq = x * x
    lanes = (sq[..., 0::2] + sq[..., 1::2]).view(S, TOPK, H // 16, 8)
    pss = _butterfly8(lanes)  # [S, 8, 256]: (owner o, slice j) = 32 o + j
    pss = pss.view(S, TOPK, TP, spec.RED_TASKS)
    sl = pss.view(S, TOPK, TP, 8, reduce.PSS_PER_LANE)  # lane rp: 4 rp ..
    lane = ((sl[..., 0] + sl[..., 1]) + sl[..., 2]) + sl[..., 3]
    own = _butterfly8(lane)  # [S, 8, TP]
    ss = _rank_sum(own.permute(2, 0, 1))
    scale = torch.rsqrt(ss * (1.0 / H) + spec.POST_EXP_EPS) * tw
    dead = pad[:, None] | (tw == 0)
    y = torch.where(dead[..., None], torch.zeros_like(x), x * scale[..., None])
    m = reduce.topk_tree([y[:, k] for k in range(TOPK)])
    m = torch.where(pad[:, None], torch.zeros_like(m), m)
    return m.to(torch.bfloat16), ss, own.permute(2, 0, 1)


def stock_finalize_fp64(part, w, tw, pad):
    """The stock formula in fp64 on the rank-order fp32 R; dead picks 0."""
    R = _rank_sum(part).double()
    dead = pad[:, None] | (tw == 0)
    R = torch.where(dead[..., None], torch.zeros_like(R), R)
    x = R * (w.double() + 1.0)
    scale = torch.rsqrt((x * x).mean(-1) + spec.POST_EXP_EPS) * tw.double()
    return (x * scale[..., None]).sum(1)


def _bf16_ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16 ulp distance (sign-magnitude ordered)."""

    def order(t):
        i = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)

    return (order(a) - order(b)).abs()


def _assert_ulps(got, want, frac=0.999, max_ulps=2, what=""):
    u = _bf16_ulps(got.cpu(), want.cpu())
    ok = (u <= 1).float().mean().item()
    assert ok >= frac, f"{what}: {ok:.5f} within 1 ulp (max {u.max().item()})"
    if max_ulps is not None:
        assert u.max().item() <= max_ulps, f"{what}: max {u.max().item()} ulps"


def _case(S, seed=0, zero_weight=False, device="cpu"):
    """Fake 8-rank partials, weights and route weights: the finalize row of 6.2
    (pad rows and a tw == 0 pick with NaN rows, an eps-scaled route)."""
    g = torch.Generator().manual_seed(seed)
    part = torch.randn(TP, S, TOPK, H, generator=g).to(torch.bfloat16)
    if zero_weight:
        w = torch.zeros(H, dtype=torch.bfloat16)
    else:
        w = (0.1 * torch.randn(H, generator=g)).to(torch.bfloat16)
    tw = torch.rand(S, TOPK, generator=g) + 0.05
    tw = tw / tw.sum(1, keepdim=True)
    pad = torch.zeros(S, dtype=torch.bool)
    # one route's R x ~3.5e-5: mean(x^2) ~ POST_EXP_EPS
    part[:, 0, 1] = (part[:, 0, 1].float() * 3.5e-5).to(torch.bfloat16)
    if S > 1:
        pad[S - 1] = True
        tw[S - 1] = 0
        part[:, S - 1] = float("nan")
    tw[0, 3] = 0  # a live row's pick of weight 0, NaN rows
    part[:, 0, 3] = float("nan")
    return (t.to(device) for t in (part, w, tw, pad))


# ---------------------------------------------------------------- [CPU]


def test_lds_structs_match_the_design():
    """LDS sizes: xs f32[s 8 16] (4096 B at S = 8), xl i32[s (1024 + 8) / 2]
    (16512 B); one CTA's LDS within a CU's."""
    for s in range(1, spec.MAX_ROWS + 1):
        assert reduce.lds_struct(s).__dsl_size_of__() == s * 8 * 16 * 4
        assert unpool.lds_struct(s).__dsl_size_of__() == s * 1032 * 2
        assert reduce.lds_struct(s) is reduce.lds_struct(s)
        lds = {"reduce": reduce.lds_struct(s), "unpool": unpool.lds_struct(s)}
        assert context.lds_total_bytes(lds) <= spec.LDS_BYTES
    assert reduce.lds_struct(8).__dsl_size_of__() == 4096
    assert unpool.lds_struct(8).__dsl_size_of__() == 16512


def _down_pushes(S):
    """``experts.stage_down``'s rsx pushes: (owner rank, pair) of
    every down CTA b of every src rank."""
    out = []
    for src in range(TP):
        for b in range(spec.DOWN_TASKS):
            o = b // spec.RED_TASKS
            for tid in range(S * 64):
                pick, rp = tid // 8, tid % 8
                out.append((o, spec.rsx_pair(src, pick, (b % 32) * 16 + 2 * rp)))
    return out


@pytest.mark.parametrize("S", [1, 3, 8])
def test_owner_protocol(S):
    """stage_owner's pairs, by its own index helpers: every rsx pair a down CTA
    pushes is polled by exactly one (owner CTA, lane); every pss pair polled is
    put by an owner CTA of the rank, each slice summed once a pick; every sqx
    pair polled is pushed by owner CTA 0 of its src; the moe pushes of all
    ranks tile rows 0 .. S of ``moe`` once."""
    picks = S * TOPK
    want = sorted(_down_pushes(S))
    polled, pss_put, moe = [], set(), []
    for rank in range(TP):
        for j in range(spec.RED_TASKS):
            for tid in range(picks * reduce.PICK_LANES):
                pick, rp = tid // reduce.PICK_LANES, tid % reduce.PICK_LANES
                cc = reduce.owner_col(j, rp)
                assert 0 <= cc < OWN and cc % 2 == 0
                for src in range(TP):
                    polled.append((rank, spec.rsx_pair(src, pick, cc)))
                if rp == 0:
                    pss_put.add((rank, spec.pss_pair(j, pick)))
            for tid in range(S * reduce.PICK_LANES):
                tt, rp = tid // reduce.PICK_LANES, tid % reduce.PICK_LANES
                col = reduce.global_col(rank, reduce.owner_col(j, rp))
                e = reduce.moe_elem(tt, col)
                assert e == 2 * spec.moe_pair(tt, col)
                moe.append(e // 2)
    assert sorted(polled) == want
    # pss: lane rp's slices, over the 8 lanes: every owner slice once
    got = sorted(q for rp in range(8) for q in reduce.pss_lane_slices(rp))
    assert got == list(range(spec.RED_TASKS))
    for rank in range(TP):
        for pick in range(picks):
            for j in range(spec.RED_TASKS):
                assert (rank, spec.pss_pair(j, pick)) in pss_put
    assert len(pss_put) == TP * spec.RED_TASKS * picks
    # sqx: owner CTA 0 of rank src pushes (src, pick); every owner polls all
    sqx = {spec.sqx_pair(src, pick) for src in range(TP) for pick in range(picks)}
    assert len(sqx) == TP * picks and max(sqx) < TP * MP
    # moe: the 8 ranks' owners tile the S rows exactly once, in the region
    assert sorted(moe) == list(range(S * H // 2))
    assert max(moe) < spec.peer_layout()["moe"][1] // 8
    assert max(p for _, p in want) < spec.peer_layout()["rsx"][1] // 8
    last = spec.pss_pair(spec.RED_TASKS - 1, picks - 1)
    assert last < spec.scratch_layout()["pss"][1] // 8


@pytest.mark.parametrize("S", [1, 3, 8])
def test_unpool_protocol(S):
    """The unpooler's placement and pairs: every (row group, K quarter) on
    exactly one CTA; the ranks' rows tile the 8192 branch columns; every upart
    pair a q = 0 CTA polls is put by its group's q > 0 CTA, once; each CTA's
    moe polls are its K quarter of every row; the branch pushes tile rows 0 ..
    S once and the gather reads each pair once into its own branch_out word."""
    tasks = [unpool.task(b) for b in range(spec.BLOCKS)]
    assert sorted(tasks) == [
        (g, q) for g in range(spec.U_GROUPS) for q in range(spec.U_PARTS)
    ]
    rows = sorted(
        unpool.row0(rank, g) for rank in range(TP) for g in range(spec.U_GROUPS)
    )
    assert rows == list(range(0, spec.HIDDEN, spec.ROWS))
    upart_put, upart_poll, branch = [], [], []
    for b in range(spec.BLOCKS):
        g, q = unpool.task(b)
        moe = [
            spec.moe_pair(t, spec.U_K * q) + tid
            for t in range(S)
            for tid in range(spec.THREADS)
        ]
        want = [
            spec.moe_pair(t, col)
            for t in range(S)
            for col in range(spec.U_K * q, spec.U_K * (q + 1), 2)
        ]
        assert moe == want
        if q > 0:
            for tid in range(S * spec.ROWS):
                upart_put.append(spec.upart_pair(g, q, tid // 16, tid % 16))
        else:
            for tid in range(S * 8):
                tt, rp = tid // 8, tid % 8
                for h in range(2):
                    for qq in range(1, spec.U_PARTS):
                        upart_poll.append(spec.upart_pair(g, qq, tt, 2 * rp + h))
                for rank in range(TP):
                    col = unpool.row0(rank, g) + 2 * rp
                    e = unpool.branch_elem(tt, col)
                    assert e == 2 * spec.branch_pair(tt, col)
                    branch.append(e // 2)
    assert sorted(upart_put) == sorted(upart_poll)
    assert len(set(upart_put)) == len(upart_put)
    assert max(upart_put) < spec.scratch_layout()["upart"][1] // 8
    assert sorted(branch) == list(range(S * spec.HIDDEN // 2))
    gather = [
        spec.branch_pair(t, unpool.gather_col(j, lane))
        for j in range(spec.G_TASKS)
        for t in range(S)
        for lane in range(unpool.GATHER_LANES)
    ]
    assert sorted(gather) == list(range(S * spec.HIDDEN // 2))


@pytest.mark.parametrize("zero_weight", [False, True])
def test_owner_math_on_host(zero_weight):
    """The owner stage's order (rank-order R, 16-column butterflies, the 32
    slices, the 8 owners in rank order, the top-k tree) against the stock
    formula in fp64: <= 1 bf16 ulp on >= 99.9 %, max 2, the eps-scaled route
    included; pad rows and the weight-0 pick with NaN rows -> exactly 0 / no
    NaN; the 8 owners' sums add up to the whole row's."""
    S = 8
    part, w, tw, pad = _case(S, seed=1, zero_weight=zero_weight)
    m, ss, own = owner_mirror(part, w, tw, pad)
    ref = stock_finalize_fp64(part, w, tw, pad)
    assert not m.isnan().any()
    assert (m[pad].view(torch.int16) == 0).all()
    live = ~pad
    _assert_ulps(m[live], ref[live].to(torch.bfloat16), what="owner mirror")
    # the eps-scaled route: its mean(x^2) is of the order of the eps
    R = _rank_sum(part)
    x = R * (w.float() + 1.0)
    ms = (x[0, 1] ** 2).mean().item()
    assert 1e-9 < ms < 1e-7, ms
    # the exchange: the 8 owners' sums are the row's sum of squares
    full = (x.double() ** 2).sum(-1)
    ok = live[:, None] & (tw != 0)
    assert torch.allclose(ss[ok].double(), full[ok], rtol=1e-5)
    assert torch.allclose(own.sum(0)[ok].double(), full[ok], rtol=1e-5)


def test_unpool_math_on_host():
    """The unpooler's K split (four fp32 quarters added in order, one bf16
    rounding) against one fp64 dot rounded once: <= 1 ulp on >= 99.9 %."""
    g = torch.Generator().manual_seed(3)
    S = 8
    moe = torch.randn(S, H, generator=g).to(torch.bfloat16)
    w = (torch.randn(256, H, generator=g) / 64).to(torch.bfloat16)
    parts = [
        moe[:, spec.U_K * q : spec.U_K * (q + 1)].float()
        @ w[:, spec.U_K * q : spec.U_K * (q + 1)].float().T
        for q in range(spec.U_PARTS)
    ]
    got = (((parts[0] + parts[1]) + parts[2]) + parts[3]).to(torch.bfloat16)
    ref = (moe.double() @ w.double().T).to(torch.bfloat16)
    _assert_ulps(got, ref, what="unpool K split")


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


_ACCESSES = {
    "owner": {
        ("owner", "rsx", "poll"),
        ("owner", "pss", "put"),
        ("owner", "pss", "poll"),
        ("owner", "sqx", "put"),
        ("owner", "sqx", "poll"),
        ("owner", "moe", "put"),
    },
    "unpool": {
        ("unpool", "moe", "poll"),
        ("unpool", "upart", "put"),
        ("unpool", "upart", "poll"),
        ("unpool", "branch", "put"),
        ("gather", "branch", "poll"),
    },
}


@compiles
@pytest.mark.parametrize("S", [1, 8])
@pytest.mark.parametrize("which", ["owner", "unpool"])
def test_probe_compiles(which, S, tmp_path, monkeypatch):
    """The GPU-less compile gate (README, Residency) on each probe, and the trace:
    the stages put / poll exactly the regions they name (``STAGE_REGIONS``)."""
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    make = _owner_probe if which == "owner" else _unpool_probe
    # its own key: a fresh trace in this process (the recorder sees it)
    probe = make(S, key=(("trace", str(tmp_path)),))
    with plan.recording() as rec:
        try:
            probe.compile()
        except RuntimeError as e:  # the load, past the compile, needs a device
            if "hipErrorNoDevice" not in str(e) and "no ROCm" not in str(e):
                raise
    txt, res = _isa(str(tmp_path), probe.name)
    print(f"{probe.name}: {res}")
    assert res["private_segment_fixed_size"] == 0, res
    assert res["next_free_vgpr"] <= spec.VGPR_MAX, res
    assert res["group_segment_fixed_size"] <= spec.LDS_BYTES, res
    got = {(a.stage, a.region, a.kind) for a in rec.accesses}
    assert got == _ACCESSES[which], got
    names = set(reduce.STAGE_REGIONS if which == "owner" else unpool.STAGE_REGIONS)
    assert {a.region for a in rec.accesses} == names
    assert not rec.unnamed
    if which == "owner":
        assert "v_rsq_f32" in txt
    else:
        assert "v_mfma_f32_16x16x32_bf16" in txt


# ---------------------------------------------------------------- [1GPU]


def _host_pairs(bufs, region: str, index: int, n: int):
    """(values, tags) of ``n`` pairs, as contiguous host int32 tensors (the
    device views are strided: a bf16 payload view needs contiguous words)."""
    vals, tags = bufs.read_pairs(region, index, n)
    return vals.cpu().contiguous(), tags.cpu().contiguous()


def _route_table(tw: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """The LDS route table ``load_route`` leaves (RouteLds encoding): ids (-1
    on pad rows) and the fp32 bits of the weights."""
    S = tw.shape[0]
    rt = torch.zeros(MT * 16, dtype=torch.int32)
    for t in range(S):
        for k in range(TOPK):
            rt[spec.route_id_pair(t, k)] = -1 if pad[t] else 8 * t + k
        rt[spec.route_weight_pair(t, 0) : spec.route_weight_pair(t, 0) + TOPK] = (
            tw[t].float().contiguous().view(torch.int32)
        )
    return rt


@needs_gfx950
@pytest.mark.parametrize("zero_weight", [False, True])
@pytest.mark.parametrize("rank", [0, 5])
@pytest.mark.parametrize("S", [1, 3, 8])
def test_finalize_probe(S, rank, zero_weight):
    """Owner CTAs of rank ``rank`` on fake 8-rank partials vs
    ``fused_post_expert_finalize`` on the rank-order R (README, Numerics:
    tolerances): <= 1 bf16 ulp on >= 99.9 %, max 2, incl. the eps-scaled route
    and the zero weight; pad rows and the weight-0 pick with NaN partials -> 0; the
    rank's sqx; every pair carries the launch's tag; no wait gave up."""
    from vllm.models.muse_spark.common.ops.fused_finalize import (
        fused_post_expert_finalize,
    )

    probe = _owner_probe(S)
    part, w, tw, pad = _case(S, seed=10 + S + rank, zero_weight=zero_weight)
    m_mirror, ss_mirror, own_mirror = owner_mirror(part, w, tw, pad)
    bufs = context.ProbeBuffers(epoch=3 + S, rank=rank, slot=spec.k2_slot(1))
    picks = S * TOPK
    lo = rank * OWN
    for src in range(TP):  # every down CTA's push to this owner
        rows = part[src, :, :, lo : lo + OWN].reshape(picks, OWN)
        bufs.write_pairs("rsx", spec.rsx_pair(src, 0, 0), rows.contiguous())
        if src != rank:  # the other owners' sums
            bufs.write_pairs("sqx", spec.sqx_pair(src, 0), own_mirror[src].reshape(-1))
    pad_d = torch.zeros(MT, dtype=torch.bool)
    pad_d[:S] = pad
    probe(
        bufs,
        pad=pad_d.cuda(),
        route=_route_table(tw, pad).cuda(),
        w_postexp=w.cuda(),
    )
    torch.accelerator.synchronize()
    assert not bufs.failed(), bufs.given_up_waits()
    tag = bufs.tag()
    vals, tags = _host_pairs(bufs, "moe", 0, S * H // 2)
    got = vals.view(torch.bfloat16).view(S, H)
    mine = (tags.view(S, H // 2) == tag).repeat_interleave(2, 1)
    assert mine[:, lo : lo + OWN].all()
    assert not mine[:, :lo].any() and not mine[:, lo + OWN :].any()
    got = got[:, lo : lo + OWN]
    # pad rows exactly 0; no NaN leaks from the dead picks
    assert not got.isnan().any()
    assert (got[pad].view(torch.int16) == 0).all()
    # vs the stock finalize (the dead picks' NaN rows zeroed: stock has no
    # select, NaN * 0 is NaN)
    R = _rank_sum(part)
    dead = pad[:, None] | (tw == 0)
    R = torch.where(dead[..., None], torch.zeros_like(R), R).contiguous()
    out = torch.empty(S, H, dtype=torch.bfloat16, device="cuda")
    fused_post_expert_finalize(
        R.cuda(), w.cuda(), tw.cuda(), spec.POST_EXP_EPS, spec.GAIN, True, out
    )
    live = ~pad
    _assert_ulps(
        got[live], out.cpu()[live][:, lo : lo + OWN], what="vs fused_post_expert"
    )
    _assert_ulps(got[live], m_mirror[live][:, lo : lo + OWN], what="vs mirror")
    # the exchange: this rank's sqx (pushed by owner CTA 0) and the pss slices
    vals, tags = _host_pairs(bufs, "sqx", spec.sqx_pair(rank, 0), picks)
    assert (tags == tag).all()
    own = vals.view(torch.float32).view(S, TOPK)
    ok = live[:, None] & (tw != 0)
    assert torch.allclose(own[ok], own_mirror[rank][ok], rtol=1e-5)
    _, tags = _host_pairs(bufs, "pss", 0, spec.RED_TASKS * MP)
    assert (tags.view(spec.RED_TASKS, MP)[:, :picks] == tag).all()


@needs_gfx950
@pytest.mark.parametrize("rank", [0, 6])
@pytest.mark.parametrize("S", [1, 3, 8])
def test_unpool_probe(S, rank):
    """The full grid's unpooler + gather of rank ``rank`` on a given ``moe``
    (every owner's slice written by the test): the rank's 1024 branch columns
    vs the stock unpooler GEMM (unpooler tolerance: <= 1 bf16 ulp on >=
    99.9 %) and vs one fp64 dot (also max 2 ulps); the other ranks' columns
    gathered as pushed; a pad row with NaN in its moe and branch rows -> 0."""
    g = torch.Generator().manual_seed(20 + S + rank)
    probe = _unpool_probe(S)
    moe = torch.randn(S, H, generator=g).to(torch.bfloat16)
    w = (torch.randn(spec.HIDDEN, H, generator=g) / 64).to(torch.bfloat16)
    other = torch.randn(S, spec.HIDDEN, generator=g).to(torch.bfloat16)
    pad = torch.zeros(S, dtype=torch.bool)
    if S > 1:
        pad[S - 1] = True
        moe[S - 1] = float("nan")
        other[S - 1] = float("nan")
    bufs = context.ProbeBuffers(epoch=7 + S, rank=rank, slot=spec.k2_slot(60))
    bufs.write_pairs("moe", 0, moe.contiguous())
    lo = rank * unpool.U_N
    for t in range(S):  # the other ranks' unpooler pushes
        for src in range(TP):
            if src != rank:
                c0 = src * unpool.U_N
                bufs.write_pairs(
                    "branch",
                    spec.branch_pair(t, c0),
                    other[t, c0 : c0 + unpool.U_N].contiguous(),
                )
    pad_d = torch.zeros(MT, dtype=torch.bool)
    pad_d[:S] = pad
    w_d = w.cuda()
    out = torch.full((S, spec.HIDDEN), 7.0, dtype=torch.bfloat16, device="cuda")
    probe(bufs, pad=pad_d.cuda(), w_unpool=w_d, branch_out=out)
    torch.accelerator.synchronize()
    assert not bufs.failed(), bufs.given_up_waits()
    got = out.cpu()
    live = ~pad
    assert (got[pad].view(torch.int16) == 0).all()
    mine = got[live][:, lo : lo + unpool.U_N]
    # the stock unpooler: ReplicatedLinear with MuseSparkTorchLinearMethod, a
    # plain F.linear (amd/model.py: not the AITER tuned-GEMM dispatch)
    stock = torch.nn.functional.linear(moe[live].cuda(), w_d).cpu()
    _assert_ulps(
        mine, stock[:, lo : lo + unpool.U_N], max_ulps=None, what="vs stock GEMM"
    )
    exact = (moe[live].double() @ w.double().T).to(torch.bfloat16)
    _assert_ulps(mine, exact[:, lo : lo + unpool.U_N], what="vs fp64")
    rest = torch.ones(spec.HIDDEN, dtype=torch.bool)
    rest[lo : lo + unpool.U_N] = False
    assert torch.equal(
        got[live][:, rest].view(torch.int16), other[live][:, rest].view(torch.int16)
    )
    # the rank's branch pushes carry the launch's tag
    _, tags = _host_pairs(bufs, "branch", spec.branch_pair(0, lo), unpool.U_N // 2)
    assert (tags == bufs.tag()).all()
