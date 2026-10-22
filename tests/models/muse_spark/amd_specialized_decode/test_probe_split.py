# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2(split)'s split stage and core load (``mono/attention/split.py``,
DESIGN_BC 7.1, 7.3, WP-C1): the split-KV GQA decode units in a probe kernel.

[CPU] ``split_model`` is the split's arithmetic in torch (the unit and page
plan, the per-wave online softmax with bf16 P and ``l`` from the rounded P,
the NaN-safe masks, the wave merge); with the host combine ``merge`` it
equals B0's fp64 reference ``gqa_decode`` within the bf16 P's rounding, over
both kinds, mixed lengths, pad rows and a cache whose every unread byte is
NaN. The kernel's ``split.pages`` (on ints) is ``spec.split_plan`` and
holds pages exactly for the units the combine waits for. The probes compile
GPU-less for gfx950 (the end-to-end core probe too): the budgets, no readfirstlane
waterfall, the page loop's two pages in flight at its MFMAs, device-scope
stores of the plain regions, no scalar loads of them; the trace names the
regions the module declares.
[1GPU] ``stage_split`` on every split CTA: the partials match the model
(layout and arithmetic), the host-merged rows meet section 4's targets vs
``torch.ops.aiter.paged_attention_v1`` (sliding on the window-trimmed table,
as served) and vs fp64, ``srdy`` carries the launch's tag, 200 launches are
bit-identical; cases per 9.2 (L at the page and window edges, an unaligned
and an aligned window start, NaN below the window, in the last page's tail
and in block 0, pad rows with L > 0, mixed rows, S in {1, 3, 8}, LHBNC, a
page past 4 GiB, a 64-page block-id reload). ``load_core`` copies the
combine's rows behind ``crdy`` (pad rows zero) and gives a stale flag up
without hanging. Split + combine + load_core end to end: the combine's
attention output meets the core targets, and every CTA's rows equal
``rms_norm_gated`` of that output bit for bit (section 4's gate target).
"""

import glob
import math
import os
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import regex as re
import torch

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

import flydsl.expr as fx  # noqa: E402
from aiter.ops.flydsl.kernels import buffer_ops as bo  # noqa: E402
from flydsl.expr import range_constexpr  # noqa: E402

from tests.models.muse_spark.amd_specialized_decode import (  # noqa: E402
    reference,
)
from vllm.models.muse_spark.amd.specialized_decode.mono import spec  # noqa: E402
from vllm.models.muse_spark.amd.specialized_decode.mono.attention import (  # noqa: E402
    back,
    split,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common import (  # noqa: E402
    context,
    plan,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (  # noqa: E402
    lds_bytes,
    rsrc,
    traced,
)

bf = torch.bfloat16
F32, F64 = torch.float32, torch.float64
CORE, QKV, ROW_G = spec.CORE, spec.QKV_ROWS, spec.ROW_G
XL_WORDS = back.XL_ROW // 2  # i32 words a padded LDS core row
ROW_WORDS = CORE // 2  # i32 words a core row
PAD_BYTES = 16  # the probe's pad flags (whole words: ld_pad's)
NEG = split.NEG
QK_SCALE_LOG2E = torch.tensor(spec.QK_SCALE_LOG2E, dtype=F32)  # the kernel's f32
# 9.2's context lengths: the page edges (15, 16, 17, 255 .. 257), the window
# edges (2047 .. 2050: the window start crosses into page 1; 2063 .. 2065: a
# window starting at a page start, 2064), long contexts
LENS = (1, 2, 15, 16, 17, 255, 256, 257, 2047, 2048, 2049, 2050, 2063, 2064, 2065)
LONG = (4103, 32773)
BIG_BLOCKS = (1 << 32) // (spec.KV_HEADS * spec.KV_BLOCK * spec.KV_C * 2) + 64
RELOAD_LEN = 140_000  # S = 8 full: 8750 pages, per 547, 69 a wave (> 64 ids)
ARGS = (
    "qkvo",
    "kv",
    "kv_sb",
    "kv_sh",
    "kv_st",
    "kv_blocks",
    "block_table",
    "bt_rows",
    "bt_stride",
    "seq_lens",
    "pad",
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
    arch = os.environ.get("FLYDSL_GPU_ARCH") or os.environ.get("ARCH")
    return arch == "gfx950" or _on_gfx950()


def _combine():
    """C2's combine module, or None while it is not in the tree."""
    try:
        from vllm.models.muse_spark.amd.specialized_decode.mono.attention import (
            combine,
        )
    except ImportError:
        return None
    return combine


needs_gfx950 = pytest.mark.skipif(
    not _on_gfx950(), reason="the Muse mono probes run on gfx950 (MI355X)"
)
compiles = pytest.mark.skipif(
    not _compiles_for_gfx950(),
    reason="GPU-less compile needs FLYDSL_GPU_ARCH=gfx950 (or a gfx950 GPU)",
)
needs_combine = pytest.mark.skipif(
    _combine() is None, reason="mono.attention.combine (WP-C2) is not in this tree"
)


# ---------------------------------------------------------------- the model


def _plan(seq_lens, pad, S: int, window: bool, cap: int):
    """Per split unit (host): (t, kvh, pa, pe, lo, L) from ``spec.split_plan``
    (``test_split_pages_are_the_plan`` pins the kernel's ``split.pages`` to
    it)."""
    n = spec.nsplit(S, window)
    rows = []
    for u in range(spec.n_units(S, window)):
        t, kvh, p = spec.split_unit(u, n)
        L = 0 if bool(pad[t]) else min(max(int(seq_lens[t]), 0), cap)
        lo, b0, nb, per, _ = spec.split_plan(L, window, n)
        pa = b0 + p * per
        rows.append((t, kvh, pa, min(pa + per, b0 + nb), lo, L))
    return torch.tensor(rows, dtype=torch.int64)


def split_model(qkvo, kv, block_table, seq_lens, pad, S: int, window: bool):
    """The split stage's arithmetic in torch, every unit and wave at once ->
    (pstat fp32 [n_units, 8, 2] = (m, l), opart fp32 [n_units, 8, 64]): wave w
    of unit u takes pages pa + w + 8 i < pe; per page the bf16 QK in fp32, the
    base-2 scores at f32(QK_SCALE LOG2E), masked keys NEG, the running max and
    rescale, P rounded to bf16 and ``l`` summed from the rounded P, masked V
    zeroed; then the waves merged by exp2(m_w - M). Exact exp2 (the kernel's
    v_exp_f32 is within an ulp), the fp32 sums in torch's order."""
    dev = kv.device
    cap = block_table.shape[1] * spec.KV_BLOCK
    pl = _plan(seq_lens, pad, S, window, cap).to(dev)
    t, kvh, pa, pe, lo, L = pl.unbind(1)
    U = pl.shape[0]
    w = torch.arange(spec.WAVES, device=dev)
    first = pa[:, None] + w[None]  # [U, 8]
    nw = (pe[:, None] - first).clamp_min(0).add(spec.WAVES - 1) // spec.WAVES
    q = qkvo[:, :CORE].float().view(S, spec.KV_HEADS, spec.Q_PER_KV, 64)[t, kvh]
    m = torch.full((U, spec.WAVES, spec.Q_PER_KV), NEG, dtype=F32, device=dev)
    den = torch.zeros_like(m)
    acc = torch.zeros(U, spec.WAVES, spec.Q_PER_KV, 64, dtype=F32, device=dev)
    keys16 = torch.arange(spec.KV_BLOCK, device=dev)
    rows = block_table.to(dev).long()
    for i in range(int(nw.max()) if U else 0):
        pg = first + spec.WAVES * i
        live = i < nw
        blk = rows[t[:, None], pg.clamp(0, rows.shape[1] - 1)]
        page = kv[blk.clamp(0, kv.shape[0] - 1), kvh[:, None]]  # [U, 8, 16, 128]
        K = page[..., : spec.V_OFF].float()
        V = page[..., spec.V_OFF :].float()
        key = spec.KV_BLOCK * pg[..., None] + keys16
        ok = live[..., None] & (key >= lo[:, None, None]) & (key < L[:, None, None])
        okh = ok[:, :, None, :]  # [U, 8 waves, 1, 16 keys]
        sc = torch.einsum("uwkd,uhd->uwhk", K, q)
        s = torch.where(okh, sc * QK_SCALE_LOG2E, torch.tensor(NEG, device=dev))
        mn = torch.maximum(m, s.amax(-1))
        al = torch.exp2(m - mn)
        p = torch.where(okh, torch.exp2(s - mn[..., None]), 0.0).to(bf).float()
        den = den * al + p.sum(-1)
        vz = torch.where(ok[..., None], V, 0.0)
        acc = acc * al[..., None] + torch.einsum("uwhk,uwkd->uwhd", p, vz)
        m = mn
    big = m.amax(1, keepdim=True)
    wt = torch.exp2(m - big)
    lsum = (den * wt).sum(1)
    osum = (acc * wt[..., None]).sum(1)
    return torch.stack([big[:, 0], lsum], -1), osum


def merge(pstat, opart, S: int, window: bool) -> torch.Tensor:
    """The host combine (7.4's math in fp64) of the first ``n_units`` units'
    partials -> each (token, q head)'s attention output fp64 [S, 16, 64];
    zero where no unit is live (a pad row, L = 0)."""
    n = spec.nsplit(S, window)
    U = spec.n_units(S, window)
    ps = pstat[:U].double().reshape(S, spec.KV_HEADS, n, spec.Q_PER_KV, 2)
    op = opart[:U].double().reshape(S, spec.KV_HEADS, n, spec.Q_PER_KV, 64)
    mp, lp = ps[..., 0], ps[..., 1]
    wp = torch.where(lp > 0, torch.exp2(mp - mp.amax(2, keepdim=True)), 0.0)
    lt = (lp * wp).sum(2)
    o = (op * wp[..., None]).sum(2)
    o = torch.where(lt[..., None] > 0, o / lt.clamp_min(1e-300)[..., None], 0.0)
    return o.reshape(S, spec.HEADS, 64)


# ---------------------------------------------------------------- inputs


@dataclass(frozen=True)
class Case:
    S: int
    window: bool
    lens: tuple
    pad: tuple = ()  # pad rows (their L stays > 0)
    layout: str = "LBHNC"
    big: bool = False  # a cache past 4 GiB, row 0's last page in its top block
    seed: int = 0

    @property
    def name(self) -> str:
        kind = "sw" if self.window else "full"
        tags = [f"S{self.S}", kind, "-".join(map(str, self.lens))]
        if self.pad:
            tags.append("pad" + "".join(map(str, self.pad)))
        if self.layout != "LBHNC":
            tags.append(self.layout)
        if self.big:
            tags.append("4GiB")
        return "_".join(tags)


def _cases(gpu: bool) -> list[Case]:
    """9.2's cases (``gpu``: also the long and large ones)."""
    out = []
    lens = LENS + (LONG if gpu else LONG[:1])
    for window in (False, True):
        for i, L in enumerate(lens):
            out.append(Case(1, window, (L,), seed=i))
        out.append(Case(3, window, (2049, 17, 300), pad=(1,), seed=40))
        out.append(Case(3, window, (2064, 2063, 1), seed=41))
        out.append(
            Case(
                8, window, (1, 16, 257, 2047, 2050, 2065, 4103, 900), pad=(6,), seed=42
            )
        )
        out.append(
            Case(8, window, (2048, 255, 2066, 17, 3000, 2, 4103, 33), seed=43,
                 layout="LHBNC")
        )  # fmt: skip
    if gpu:
        out.append(Case(1, False, (131072,), seed=50))
        out.append(Case(8, False, (32773,) * 2 + (2049,) * 6, seed=51))
        out.append(Case(8, False, (RELOAD_LEN, 5, 17, 2048, 1, 2, 3, 4), seed=52))
        out.append(Case(1, False, (2065,), big=True, seed=53))
        out.append(Case(1, True, (4103,), big=True, seed=54))
    return out


def make(case: Case, device="cpu") -> SimpleNamespace:
    """A case's inputs, the cache still clean: qkvo bf16 [S, QKV], the layer's
    cache (block 0 NaN), a persistent-style table int32 [S + 2, width] (rows
    past the step's, entries past a row's pages and, on a sliding layer,
    below its window null -- block 0, as MRV2's gather leaves them),
    seq_lens int32 [S + 2] (junk past S), pad bool [16]; ``pages``: each
    row's blocks. ``poison(ns)`` then fills every byte the split must not
    read with NaN."""
    g = torch.Generator().manual_seed(1000 + case.seed)
    S = case.S
    need = [-(-L // spec.KV_BLOCK) for L in case.lens]
    width = max(need) + 3
    nblocks = max(sum(need) + 8, 64)
    if case.big:
        nblocks = BIG_BLOCKS
        dims = (nblocks, spec.KV_HEADS, spec.KV_BLOCK, spec.KV_C)
        kv = torch.empty(dims, dtype=bf, device=device)
        kv.normal_(generator=torch.Generator(device=device).manual_seed(case.seed))
        kv[0] = math.nan
    else:
        kv = reference.kv_cache(nblocks, layout=case.layout, device=device, generator=g)
    pages = reference.assign_pages(case.lens, nblocks, generator=g)
    if case.big:  # row 0's last page at the cache's top block (> 4 GiB in)
        top = nblocks - 1
        for row in pages:
            if top in row:
                row[row.index(top)] = pages[0][-1]
        pages[0][-1] = top
        assert top * kv.stride(0) * 2 >= 1 << 32
    bt = torch.zeros(S + 2, width, dtype=torch.int32)
    for t, (L, row) in enumerate(zip(case.lens, pages)):
        bt[t, : len(row)] = torch.tensor(row, dtype=torch.int32)
        if case.window:  # entries wholly below the window: the null block
            null = max(L - spec.WINDOW, 0) // spec.KV_BLOCK
            bt[t, :null] = 0
    seq_lens = torch.full((S + 2,), 7777, dtype=torch.int32)
    seq_lens[:S] = torch.tensor(case.lens, dtype=torch.int32)
    pad = torch.zeros(PAD_BYTES, dtype=torch.bool)
    pad[list(case.pad)] = True
    pad[S:] = True
    qkvo = torch.randn(S, QKV, generator=g).to(bf)
    return SimpleNamespace(
        case=case,
        qkvo=qkvo.to(device),
        kv=kv,
        bt=bt.to(device),
        seq_lens=seq_lens.to(device),
        pad=pad.to(device),
        pages=pages,
    )


def poison(ns) -> None:
    """NaN into every cache byte the split must not read (in place): a live
    row's keys below its window and past L in the pages it reads, its pages
    wholly below the window; every page of a pad row."""
    case = ns.case
    live = [t for t in range(case.S) if t not in case.pad]
    reference.poison_unread(
        ns.kv, ns.bt.cpu(), ns.seq_lens.cpu(), window=case.window, rows=live
    )
    for t in case.pad:
        ns.kv[torch.tensor(ns.pages[t], device=ns.kv.device)] = math.nan


def live_rows(case: Case) -> torch.Tensor:
    """The rows with keys: not a pad row and L > 0."""
    return torch.tensor([t not in case.pad and case.lens[t] > 0 for t in range(case.S)])


def _q(ns):
    return ns.qkvo[:, :CORE].reshape(ns.case.S, spec.HEADS, 64)


# ---------------------------------------------------------------- [CPU]


@pytest.mark.parametrize("case", _cases(gpu=False), ids=lambda c: c.name)
def test_model_matches_reference(case):
    """The split's arithmetic (``split_model`` + ``merge``) on a poisoned
    cache equals the fp64 reference within the bf16 P's rounding; every unit
    of a pad row or past a row's live parts publishes (NEG, 0, 0); a live
    unit's l > 0; nothing is NaN."""
    ns = make(case)
    ref = reference.gqa_decode(
        _q(ns), ns.kv, ns.bt, ns.seq_lens, window=case.window, pad=ns.pad[: case.S]
    )
    poison(ns)
    pstat, opart = split_model(
        ns.qkvo, ns.kv, ns.bt, ns.seq_lens, ns.pad, case.S, case.window
    )
    assert torch.isfinite(pstat).all() and torch.isfinite(opart).all()
    got = merge(pstat, opart, case.S, case.window)
    live = live_rows(case)
    err = (got - ref).abs().amax().item()
    # bf16 P: each weight within 2^-9 relative; V entries ~ N(0, 1)
    assert err <= 4e-3, (case.name, err)
    if live.any():
        assert reference.cos(got[live], ref[live]) > 0.9999
    assert (got[~live] == 0).all()
    n = spec.nsplit(case.S, case.window)
    cap = ns.bt.shape[1] * spec.KV_BLOCK
    units = _plan(ns.seq_lens, ns.pad, case.S, case.window, cap).tolist()
    for u, (t, _, pa, pe, _, _) in enumerate(units):
        if pe <= pa:  # empty: NEG, 0, zeros
            assert pstat[u, :, 0].eq(NEG).all() and pstat[u, :, 1].eq(0).all(), u
            assert opart[u].eq(0).all(), u
        else:
            assert (pstat[u, :, 1] > 0).all(), (u, t)
    assert n * 2 * case.S == pstat.shape[0]


def test_split_pages_are_the_plan():
    """``split.pages`` (the kernel's very function, on Python ints) is
    ``spec.split_plan``'s part p, at every width and kind, over the page and
    window edges and long contexts; part p holds pages iff p < live, the
    units the combine waits for (``combine.unit_plan``'s live), so no unit
    with keys is ever left out of the merge. ``load_core`` writes back's
    padded ``xl`` rows."""
    Ls = set(range(0, 300)) | set(range(2000, 2120)) | set(range(4090, 4120))
    Ls |= {32767, 32768, 32773, 65536, 131072, RELOAD_LEN}
    for S in range(1, spec.MAX_ROWS + 1):
        for w in (False, True):
            n = spec.nsplit(S, w)
            for L in sorted(Ls):
                lo, b0, nb, per, live = spec.split_plan(L, w, n)
                for p in range(n):
                    got = split.pages(L, w, n, p)
                    want = (lo, b0 + p * per, min(b0 + (p + 1) * per, b0 + nb))
                    assert got == want, (S, w, L, p, got, want)
                    assert (got[1] < got[2]) == (p < live), (S, w, L, p)
    assert split.XL_ROW == back.XL_ROW and split.OKW == back.OKW
    # TUNE L3: every preset's (N, MIN_PER)
    seen = set()
    for pr in spec.SPLIT_PRESETS:
        for S in range(1, spec.MAX_ROWS + 1):
            for w in (False, True):
                k = spec.K2Build(
                    tokens=S, core="split", window=w, **spec.split_fields(pr, S, w)
                )
                n, mp = k.nsplit(), k.split_min_per
                if (w, n, mp) in seen or (n, mp) == (spec.nsplit(S, w), spec.MIN_PER):
                    continue
                seen.add((w, n, mp))
                for L in sorted(Ls):
                    lo, b0, nb, per, live = spec.split_plan(L, w, n, mp)
                    for p in range(n):
                        got = split.pages(L, w, n, p, mp)
                        want = (lo, b0 + p * per, min(b0 + (p + 1) * per, b0 + nb))
                        assert got == want, (pr, S, w, L, p, got, want)
                        assert (got[1] < got[2]) == (p < live), (pr, S, w, L, p)
    assert seen


def test_cases_make_the_edges_bite():
    """The CPU and GPU case lists cover what 9.2 asks: window starts aligned
    and unaligned to a page, last pages full and partial, live < N and live
    == N, a unit of more than MIN_PER pages, pad rows with L > 0, both
    layouts, a page past 4 GiB and the block-id reload (GPU), NaN in what is
    never read."""
    starts, tails, lives, pers = set(), set(), set(), set()
    for case in _cases(gpu=True):
        n = spec.nsplit(case.S, case.window)
        for t, L in enumerate(case.lens):
            if t in case.pad:
                continue
            lo, b0, nb, per, live = spec.split_plan(L, case.window, n)
            if case.window and L > spec.WINDOW:
                starts.add(lo % spec.KV_BLOCK == 0)
            tails.add(L % spec.KV_BLOCK == 0)
            lives.add(live == n)
            pers.add(per > spec.MIN_PER)
    assert starts == {True, False} and tails == {True, False}
    assert lives == {True, False} and pers == {True, False}
    gpu = _cases(gpu=True)
    assert any(c.big for c in gpu) and any(c.layout == "LHBNC" for c in gpu)
    assert any(c.pad and c.lens[c.pad[0]] > 0 for c in gpu)
    # the reload: some wave of the S = 8 full case has more than 64 pages
    _, _, nb, per, _ = spec.split_plan(RELOAD_LEN, False, spec.nsplit(8, False))
    assert -(-per // spec.WAVES) > split.BID_LANES
    ns = make(Case(8, True, (1, 16, 257, 2047, 2050, 2065, 4103, 900), pad=(6,)))
    poison(ns)
    assert ns.kv[0].isnan().all()  # the null block
    assert ns.kv[torch.tensor(ns.pages[6])].isnan().all()  # the pad row's pages
    assert (ns.bt[5, :1] == 0).all()  # below row 5's window: nulled
    assert ns.kv[ns.bt[3, -1 + -(-2047 // 16)].long(), :, 2047 % 16 :].isnan().all()


# ---------------------------------------------------------------- the probes


def _split_body(n: int):
    """The split units, CTA b < ``n`` running unit b (as K2(split) places
    them); the rest of the grid idle."""

    @traced
    def body(c):
        if c["bid"] < n:
            split.stage_split(c, c["bid"])

    return body


def _split_probe(S: int, window: bool):
    n = spec.n_units(S, window)
    return context.probe_kernel(
        f"muse_probe_split_s{S}_{'sw' if window else 'full'}",
        S,
        _split_body(n),
        {"split": split.lds_struct(S)},
        ARGS,
        blocks=spec.BLOCKS,
        build=spec.K2Build(tokens=S, core="split", window=window),
        key=(("units", n),),
    )


def _load_core_body(c):
    """``split.load_core``, then CTA b's LDS ``xl`` rows -> ``out`` rows b S
    .. (every CTA loads the same rows)."""
    s, tid = c["S"], c["tid"]
    split.load_core(c)
    xl = c["lds"]["back"]["xl"]
    b = fx.Int64(fx.Int32(c["bid"]))
    out = rsrc(c["args"]["out"] + b * (s * CORE * 2))
    for i in range_constexpr(s):
        v = fx.ptr_load(xl + (i * XL_WORDS + tid))
        bo.buffer_store(v, out, i * ROW_WORDS + tid)


def _load_core_probe(S: int):
    return context.probe_kernel(
        f"muse_probe_split_load_core_s{S}",
        S,
        _load_core_body,
        {"back": back.lds_struct(S)},
        ("pad", "out"),
        blocks=spec.BLOCKS,
    )


def _core_body():
    """7.2's core without the weights: K2's own ``layer.split_roles`` (the
    split units, the combine CTAs, each combine also dumping its bf16
    attention output, before the gate, to ``attn``), then every CTA's
    load_core; CTA b's xl rows -> ``out`` rows b S .."""
    from vllm.models.muse_spark.amd.specialized_decode.mono import layer

    @traced
    def body(c):
        s, tid = c["S"], c["tid"]
        b = c["bid"]
        layer.split_roles(c, attn_out=c["args"]["attn"])
        split.load_core(c)
        xl = c["lds"]["back"]["xl"]
        out = rsrc(c["args"]["out"] + fx.Int64(fx.Int32(b)) * (s * CORE * 2))
        for i in range_constexpr(s):
            v = fx.ptr_load(xl + (i * XL_WORDS + tid))
            bo.buffer_store(v, out, i * ROW_WORDS + tid)

    return body


def _core_key(S: int, window: bool, preset: str = "base") -> spec.K2Build:
    """The split build key of the core probe: ``preset``'s (TUNE L3) fields."""
    return spec.K2Build(
        tokens=S, core="split", window=window, **spec.split_fields(preset, S, window)
    )


def _core_probe(S: int, window: bool, preset: str = "base"):
    combine = _combine()
    key = _core_key(S, window, preset)
    return context.probe_kernel(
        f"muse_probe_split_core_s{S}_{'sw' if window else 'full'}" + key.split_suffix(),
        S,
        _core_body(),
        {
            "back": back.lds_struct(S),
            "split": split.lds_struct(S),
            "combine": combine.lds_struct(S),
        },
        (*ARGS, "out", "attn"),
        blocks=spec.BLOCKS,
        build=key,
        key=(
            ("units", key.n_units()),
            ("c0", key.c0()),
            ("min_per", key.split_min_per),
        ),
    )


def _isa(dump: str, name: str) -> tuple[str, dict]:
    for path in glob.glob(f"{dump}/**/*final_isa.s", recursive=True):
        with open(path) as f:
            txt = f.read()
        if re.search(rf"\.amdhsa_kernel {re.escape(name)}\b", txt):
            res = {}
            for k in (
                "next_free_vgpr",
                "group_segment_fixed_size",
                "private_segment_fixed_size",
            ):
                m = re.search(rf"\.amdhsa_{k}\s+(\d+)", txt)
                res[k] = int(m.group(1)) if m else None
            return txt, res
    raise AssertionError(f"no final ISA of {name} under {dump}")


def _compile(probe, dump, monkeypatch) -> tuple[str, dict]:
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(dump))
    try:
        probe.compile()
    except RuntimeError as e:  # the load, past the compile, needs a device
        if "hipErrorNoDevice" not in str(e) and "no ROCm" not in str(e):
            raise
    return _isa(str(dump), probe.name)


_WATERFALL = re.compile(r"v_cmp_eq_u64_e\d+\s+\S+,\s*s\[\d+:\d+\],\s*v\[")
_QK_FIRST = re.compile(r"^\s*v_mfma_f32_16x16x32_bf16\s+v\[\d+:\d+\],.*,\s*0\s*$", re.M)
_VMCNT = re.compile(r"^\s*s_waitcnt\b.*\bvmcnt\((\d+)\)", re.M)
_VMEM_LOAD = re.compile(r"^\s*buffer_load_", re.M)
PAGE_LOADS = 6  # a page's K (2 dwordx4) and V (4 dwordx2) a lane


def _gate(txt: str, res: dict) -> None:
    """The compile gate: no scratch, the VGPR and LDS budgets, no
    readfirstlane waterfall, no scalar buffer loads."""
    assert res["private_segment_fixed_size"] == 0, res
    assert res["next_free_vgpr"] <= spec.VGPR_MAX, res
    assert res["group_segment_fixed_size"] <= spec.LDS_BYTES, res
    assert not _WATERFALL.findall(txt)
    assert "s_buffer_load" not in txt


@compiles
@pytest.mark.parametrize("window", [False, True], ids=["full", "sw"])
@pytest.mark.parametrize("S", [1, 3, 8])
def test_split_probe_compiles(S, window, tmp_path, monkeypatch):
    """The gate on the split; its LDS is the split struct (17,408 B); the
    page loop is unrolled once (2 pages a trip: 4 QK and 8 PV MFMAs), and
    each page's first QK MFMA waits for its own loads only (the other page's
    6 stay in flight); every store (opart, pstat, the srdy pair) is device
    scope."""
    txt, res = _compile(_split_probe(S, window), tmp_path, monkeypatch)
    print(f"split S={S} window={window}: {res}")
    _gate(txt, res)
    assert res["group_segment_fixed_size"] == lds_bytes(split.lds_struct(S)) == 17408
    assert len(re.findall(r"^\s*v_mfma_f32_16x16x32_bf16\b", txt, re.M)) == 4
    assert len(re.findall(r"^\s*v_mfma_f32_16x16x16_bf16\b", txt, re.M)) == 8
    firsts = list(_QK_FIRST.finditer(txt))
    assert len(firsts) == 2, [m.group(0) for m in firsts]
    for mf in firsts:
        issued = list(_VMEM_LOAD.finditer(txt, 0, mf.start()))[-PAGE_LOADS:]
        assert len(issued) == PAGE_LOADS
        waits = [int(w.group(1)) for w in _VMCNT.finditer(txt, issued[0].start(),
                                                          mf.start())]  # fmt: skip
        assert waits and min(waits) >= PAGE_LOADS, (mf.group(0), waits)
    stores = re.findall(r"^\s*buffer_store\S*.*$", txt, re.M)
    assert len(stores) == 3 and all(re.search(r"\bsc1\b", s) for s in stores), stores


@compiles
@pytest.mark.parametrize("S", [1, 8])
def test_load_core_probe_compiles(S, tmp_path, monkeypatch):
    """The gate on load_core: the crdy poll and the xo rows at device scope
    (one dwordx4 a thread and round)."""
    txt, res = _compile(_load_core_probe(S), tmp_path, monkeypatch)
    print(f"load_core S={S}: {res}")
    _gate(txt, res)
    rounds = -(-S * split.OKW // spec.THREADS)
    xo = re.findall(r"^\s*buffer_load_dwordx4\b.*\bsc1\b", txt, re.M)
    assert len(xo) == rounds, xo
    # the poll: device scope (sc1 without sc0, which the health words carry)
    poll = re.compile(r"^\s*buffer_load_dword(?:x2)?\s(?!.*\bsc0\b).*\bsc1\b", re.M)
    assert poll.search(txt)


@compiles
@needs_combine
@pytest.mark.parametrize("window", [False, True], ids=["full", "sw"])
@pytest.mark.parametrize("S", [1, 8])
def test_core_probe_compiles(S, window, tmp_path, monkeypatch):
    """The gate on the end-to-end core probe ``test_split_combine_core``
    launches (split, combine, load_core on the full grid)."""
    txt, res = _compile(_core_probe(S, window), tmp_path, monkeypatch)
    print(f"core S={S} window={window}: {res}")
    _gate(txt, res)


# TUNE L3: the presets the core probe runs (``test_split_combine_core_preset``)
PROBE_PRESETS = ("short", "wide", "coarse")


@compiles
@needs_combine
@pytest.mark.parametrize("window", [False, True], ids=["full", "sw"])
@pytest.mark.parametrize("S", [1, 3, 8])
@pytest.mark.parametrize("preset", PROBE_PRESETS)
def test_core_probe_compiles_preset(preset, S, window, tmp_path, monkeypatch):
    """The gate on the TUNE L3 core probes (a preset that leaves the build
    the design's at this width and kind is skipped)."""
    if not spec.split_fields(preset, S, window):
        pytest.skip(f"{preset} is the design's build at S={S} window={window}")
    txt, res = _compile(_core_probe(S, window, preset), tmp_path, monkeypatch)
    print(f"core {preset} S={S} window={window}: {res}")
    _gate(txt, res)


@compiles
@pytest.mark.parametrize("S", [1, 8])
def test_trace_names_the_regions(S, tmp_path, monkeypatch):
    """The split stage puts ``srdy`` only, load_core (o_proj's first part)
    polls ``crdy`` only, both scratch regions ``STAGE_REGIONS`` declares."""
    n = spec.n_units(S, True)

    @traced
    def body(c):
        if c["bid"] < n:
            split.stage_split(c, c["bid"])
        split.load_core(c)

    probe = context.probe_kernel(
        f"muse_probe_split_trace_s{S}",
        S,
        body,
        {"back": back.lds_struct(S), "split": split.lds_struct(S)},
        ARGS,
        blocks=spec.BLOCKS,
        build=spec.K2Build(tokens=S, core="split", window=True),
    )
    with plan.recording() as rec:
        _compile(probe, tmp_path, monkeypatch)
    assert tuple(rec.stages) == (split.STAGE, split.CORE_STAGE), rec.stages
    assert not rec.unnamed, rec.unnamed
    seen = {(a.stage, a.region, a.kind) for a in rec.accesses}
    assert seen == {
        (split.STAGE, "srdy", "put"),
        (split.CORE_STAGE, "crdy", "poll"),
    }, seen
    assert all(a.space == plan.Space.SCRATCH for a in rec.accesses)
    assert set(split.STAGE_REGIONS) == {"srdy", "crdy"} <= set(spec.SPLIT_REGIONS)


def test_window_needs_a_split_key():
    """``stage_split`` reads its key window from the build: a probe or
    kernel without a split key is refused while tracing."""
    for key in (None, spec.K2Build(tokens=1), spec.K2Build(tokens=1, core="gated")):
        with pytest.raises(ValueError, match="split"):
            split.window_of({"key": key})
    assert split.window_of({"key": spec.K2Build(tokens=2, core="split", window=True)})
    assert not split.window_of({"key": spec.K2Build(tokens=2, core="split")})


# ---------------------------------------------------------------- [1GPU]


def _launch_split(probe, bufs, ns):
    kv = ns.kv
    probe(
        bufs,
        qkvo=ns.qkvo,
        kv=kv,
        kv_sb=kv.stride(0) * kv.element_size(),
        kv_sh=kv.stride(1) * kv.element_size(),
        kv_st=kv.stride(2) * kv.element_size(),
        kv_blocks=kv.shape[0],
        block_table=ns.bt,
        bt_rows=ns.bt.shape[0],
        bt_stride=ns.bt.stride(0),
        seq_lens=ns.seq_lens,
        pad=ns.pad,
    )


def _plain(bufs, name: str, dtype, shape):
    off, nbytes = spec.scratch_layout()[name]
    return bufs.scratch[off : off + nbytes].view(dtype).view(shape)


def _partials(bufs):
    pstat = _plain(bufs, spec.PSTAT, F32, (spec.BLOCKS, spec.Q_PER_KV, 2))
    opart = _plain(bufs, spec.OPART, F32, (spec.BLOCKS, spec.Q_PER_KV, 64))
    return pstat, opart


def stock_attention(ns) -> torch.Tensor:
    """What stock decode computes for the case: ``paged_attention_v1`` on the
    layer's cache views (``_split_kv_cache``), a sliding layer on
    ``_build_swa_decode_tables``' window-trimmed table with the window 2048,
    a full layer on the whole table. bf16 [S, 16, 64]."""
    import aiter  # noqa: F401

    from vllm.v1.attention.backends.rocm_aiter_fa import (
        _PARTITION_SIZE_ROCM,
        AiterFlashAttentionMetadataBuilder,
    )

    case, dev = ns.case, ns.kv.device
    S = case.S
    key_cache, value_cache = ns.kv.transpose(1, 2).split(64, dim=-1)
    bt, lens = ns.bt[:S], ns.seq_lens[:S]
    max_len = max(case.lens)
    if case.window:
        nwb = spec.WINDOW // spec.KV_BLOCK + 1
        builder = SimpleNamespace(
            aot_sliding_window=(spec.WINDOW - 1, 0),
            block_size=spec.KV_BLOCK,
            _swa_block_table=torch.zeros(S, nwb, dtype=torch.int32, device=dev),
            _swa_seq_lens=torch.zeros(S, dtype=torch.int32, device=dev),
            _swa_block_offsets=torch.arange(nwb, dtype=torch.int64, device=dev),
        )
        table, lens, max_len = (
            AiterFlashAttentionMetadataBuilder._build_swa_decode_tables(  # noqa: E501
                builder, bt, lens, S, max_len
            )
        )
        sliding = spec.WINDOW
    else:
        table, sliding = bt, 0
    parts = -(-max_len // _PARTITION_SIZE_ROCM)
    work = torch.empty(
        S * spec.HEADS * parts * 64 * 2 + 2 * S * spec.HEADS * parts * 4,
        dtype=torch.uint8,
        device=dev,
    )
    q = _q(ns).contiguous()
    out = torch.empty_like(q)
    one = torch.ones(1, dtype=F32, device=dev)
    # the kernel reads cu_query_lens[t + 1] (query_len = its difference): decode
    # passes ``query_start_loc[:num_decodes]``, a view of the [S + 1] offsets,
    # so the row past the end is real; a bare arange(S) reads past its end and
    # a stale value > 1 there skips the row (output never written)
    qsl = torch.arange(S + 1, dtype=torch.int32, device=dev)[:S]
    torch.ops.aiter.paged_attention_v1(
        out, work, q, key_cache, value_cache, spec.QK_SCALE, table,
        qsl, lens, max_len, None,
        "auto", "NHD", 0.0, one, one, None, _PARTITION_SIZE_ROCM, 1, sliding,
    )  # fmt: skip
    return out


def row_ulps(got, ref) -> torch.Tensor:
    """|got - ref| in bf16 ulps at the scale of each (token, head) row of
    ``ref`` (its largest element's ulp). Element ulps are no measure here:
    an output near zero (V cancelling) has a tiny ulp, and one bf16 rounding
    of P -- stock's and the split's alike -- moves it by many (the CPU
    emulation of ll4mi: 2-7 % of the elements past 2 element ulps vs the
    split, none past 1 row ulp)."""
    r = ref.double()
    top = r.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    return (got.double() - r).abs() / torch.exp2(torch.floor(torch.log2(top)) - 7)


def check_rows(got, ref, what: str) -> None:
    """<= 2 row ulps everywhere, <= 1 on >= 99.9 %, cos > 0.99999."""
    u = row_ulps(got, ref)
    assert u.max().item() <= 2, (what, u.max().item())
    assert (u <= 1).float().mean().item() >= 0.999, (what, (u > 1).sum().item())
    assert reference.cos(got, ref) > 0.99999, what


def check_core(got, stock, ref64, live, what: str) -> None:
    """Section 4's core targets on the live rows: vs stock within 2 bf16
    ulps at the row's scale (``row_ulps``) and cos > 0.99999; vs fp64 the max
    abs error at most stock's plus one bf16 ulp of the largest output (never
    worse than stock, E9). The element-ulp fraction is printed."""
    g, s, r = got[live].cpu(), stock[live].cpu(), ref64[live].cpu()
    check_rows(g, s, what)
    u = reference.bf16_ulps(g, s)
    print(f"{what}: <= 2 element ulps vs stock on {(u <= 2).float().mean():.5f}")
    e_got = (g.double() - r).abs().max().item()
    e_stock = (s.double() - r).abs().max().item()
    ulp = 2.0 ** (math.floor(math.log2(r.abs().max().item())) - 7)
    assert e_got <= e_stock + ulp, (what, e_got, e_stock)


LAUNCHES = 200  # a stale cross-XCD read cannot pass by luck


@needs_gfx950
@pytest.mark.parametrize("case", _cases(gpu=True), ids=lambda c: c.name)
def test_split_partials(case):
    """``stage_split`` on every split CTA: its partials match the model, the
    host-merged rows meet the core targets vs the stock decode and fp64,
    the pad rows' units are empty, every srdy pair carries the launch's tag,
    and 200 launches (a new epoch each) give the same bytes."""
    ns = make(case, "cuda")
    live = live_rows(case)
    stock = stock_attention(ns).cpu() if case.layout == "LBHNC" else None
    ref = reference.gqa_decode(
        _q(ns), ns.kv, ns.bt, ns.seq_lens, window=case.window, pad=ns.pad[: case.S]
    )
    poison(ns)
    U = spec.n_units(case.S, case.window)
    probe = _split_probe(case.S, case.window)
    bufs = context.ProbeBuffers()
    _launch_split(probe, bufs, ns)
    torch.accelerator.synchronize()
    assert not bufs.failed(), bufs.given_up_waits()
    pstat, opart = (x[:U].clone() for x in _partials(bufs))
    assert torch.isfinite(pstat).all() and torch.isfinite(opart).all()
    _, tags = bufs.read_pairs("srdy", 0, U)
    assert (tags == bufs.tag()).all()
    # the layout and arithmetic: the model up to the hardware exp2 (a bf16 P
    # may round the other way: l and o move by 2^-8 of that term) and the
    # MFMA's summation order; a misplaced head, dim or page is O(1) off
    mp, mo = split_model(ns.qkvo, ns.kv, ns.bt, ns.seq_lens, ns.pad, case.S,
                         case.window)  # fmt: skip
    assert torch.equal(pstat[:, :, 0] == NEG, mp[:, :, 0] == NEG)
    torch.testing.assert_close(pstat[..., 0], mp[..., 0], rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(pstat[..., 1], mp[..., 1], rtol=1e-2, atol=1e-6)
    scale = mo.abs().amax().clamp_min(1.0)
    assert ((opart - mo).abs().amax() / scale).item() < 1e-2
    got = merge(pstat, opart, case.S, case.window).to(bf).cpu()
    assert (got[~live].float() == 0).all()
    if live.any():
        if stock is not None:
            check_core(got, stock, ref, live, case.name)
        else:  # LHBNC: no stock kernel on that layout; fp64 only
            check_rows(got[live], ref[live].to(bf), case.name)
    for k in range(LAUNCHES):
        bufs.epoch += 1
        _launch_split(probe, bufs, ns)
        if k % 50 == 49:
            torch.accelerator.synchronize()
            p2, o2 = (x[:U] for x in _partials(bufs))
            assert torch.equal(p2, pstat) and torch.equal(o2, opart), k
    torch.accelerator.synchronize()
    assert not bufs.failed(), bufs.given_up_waits()
    _, tags = bufs.read_pairs("srdy", 0, U)
    assert (tags == bufs.tag()).all()


def _put_xo(bufs, rows) -> None:
    """The combine's output rows (bf16 [S, CORE]) into ``xo``."""
    S = rows.shape[0]
    _plain(bufs, spec.XO, bf, (spec.MAX_TOKENS, CORE))[:S] = rows


@needs_gfx950
@pytest.mark.parametrize("S", [1, 3, 8])
def test_load_core(S):
    """Every CTA's xl rows are the xo rows bit for bit once every crdy pair
    carries the tag, a pad row's zeros (NaN in its xo row); a stale crdy pair
    gives up within the budget, names ``crdy``, and the launch completes."""
    probe = _load_core_probe(S)
    bufs = context.ProbeBuffers(budget=1)
    g = torch.Generator(device="cuda").manual_seed(S)
    rows = torch.randn(S, CORE, generator=g, device="cuda").to(bf)
    pad = torch.zeros(PAD_BYTES, dtype=torch.bool, device="cuda")
    if S > 1:
        pad[S - 1] = True
        rows[S - 1] = math.nan
    _put_xo(bufs, rows)
    flags = torch.ones(spec.HEADS * S, dtype=torch.int32)
    bufs.write_pairs("crdy", 0, flags)
    out = torch.full((spec.BLOCKS * S, CORE), math.nan, dtype=bf, device="cuda")
    probe(bufs, pad=pad, out=out)
    torch.accelerator.synchronize()
    assert not bufs.failed(), bufs.given_up_waits()
    want = torch.where(pad[:S, None], torch.zeros_like(rows), rows)
    got = out.view(spec.BLOCKS, S, CORE)
    assert (got.view(torch.int16) == want.view(torch.int16)).all()
    # a stale flag (the previous epoch's tag) on (t, h) = (0, 5)
    bufs.write_pairs("crdy", 5, flags[:1], tag=bufs.tag() - spec.TAG_SLOTS)
    probe(bufs, pad=pad, out=out)
    torch.accelerator.synchronize()
    assert bufs.failed()
    waits = bufs.given_up_waits()
    assert waits and all("crdy pair 5" in w for w in waits), waits


def _core_launch(probe, bufs, ns, out, attn):
    kv = ns.kv
    probe(
        bufs,
        qkvo=ns.qkvo,
        kv=kv,
        kv_sb=kv.stride(0) * kv.element_size(),
        kv_sh=kv.stride(1) * kv.element_size(),
        kv_st=kv.stride(2) * kv.element_size(),
        kv_blocks=kv.shape[0],
        block_table=ns.bt,
        bt_rows=ns.bt.shape[0],
        bt_stride=ns.bt.stride(0),
        seq_lens=ns.seq_lens,
        pad=ns.pad,
        out=out,
        attn=attn,
    )


@needs_gfx950
@needs_combine
@pytest.mark.parametrize(
    "case",
    [c for c in _cases(gpu=True) if c.layout == "LBHNC" and not c.big],
    ids=lambda c: c.name,
)
def test_split_combine_core(case):
    """Split + combine + load_core (7.2's core, no weights) against section
    4's two targets: the combine's bf16 attention output (dumped before the
    gate) meets the core targets vs ``paged_attention_v1`` and fp64
    (``check_core``), and every CTA's xl rows equal the runtime-compiled
    ``rms_norm_gated`` of that very output bit for bit; pad rows exact
    zeros; no wait given up; 200 launches (a new epoch each) bit-identical."""
    _split_combine_core(case, "base")


@needs_gfx950
@needs_combine
@pytest.mark.parametrize(
    "case",
    [c for c in _cases(gpu=True) if c.layout == "LBHNC" and not c.big],
    ids=lambda c: c.name,
)
@pytest.mark.parametrize("preset", PROBE_PRESETS)
def test_split_combine_core_preset(preset, case):
    """TUNE L3: ``test_split_combine_core``'s checks on a preset's build (its
    N and MIN_PER through K2's ``split_roles``): the same core
    targets, gate bit for bit, determinism over 200 launches."""
    if not spec.split_fields(preset, case.S, case.window):
        pytest.skip(f"{preset} is the design's build here")
    _split_combine_core(case, preset)


def _split_combine_core(case, preset: str) -> None:
    from vllm.third_party.flash_linear_attention.ops.kda import (
        rms_norm_gated,
    )

    ns = make(case, "cuda")
    S = case.S
    heads = (S, spec.HEADS, 64)
    live = live_rows(case)
    stock = stock_attention(ns).cpu()
    ref = reference.gqa_decode(
        _q(ns), ns.kv, ns.bt, ns.seq_lens, window=case.window, pad=ns.pad[:S]
    )
    poison(ns)
    probe = _core_probe(S, case.window, preset)
    bufs = context.ProbeBuffers()
    out = torch.full((spec.BLOCKS * S, CORE), math.nan, dtype=bf, device="cuda")
    attn = torch.full((S, CORE), math.nan, dtype=bf, device="cuda")
    _core_launch(probe, bufs, ns, out, attn)
    torch.accelerator.synchronize()
    assert not bufs.failed(), bufs.given_up_waits()
    first, first_attn = out.clone(), attn.clone()
    got = attn.cpu().view(heads)
    assert (got[~live].view(torch.int16) == 0).all()
    if live.any():
        check_core(got, stock, ref, live, case.name)
    # the gate on the kernel's own attention output, as served (og read in
    # place at the qkv row stride)
    want = rms_norm_gated(
        attn.view(heads),
        ns.qkvo[:, ROW_G:].view(heads),
        None,
        None,
        "sigmoid",
        eps=spec.RMS_EPS,
    ).reshape(S, CORE)
    pad_rows = torch.tensor([t in case.pad for t in range(S)], device="cuda")
    want = torch.where(pad_rows[:, None], torch.zeros_like(want), want)
    rows = out.view(spec.BLOCKS, S, CORE)
    assert (rows.view(torch.int16) == want.view(torch.int16)[None]).all()
    if live.any():  # diagnostic: the whole chain vs the stock chain
        gated = rms_norm_gated(
            stock.cuda(), ns.qkvo[:, ROW_G:].view(heads), None, None, "sigmoid",
            eps=spec.RMS_EPS,
        ).reshape(S, CORE)  # fmt: skip
        lv = live.cuda()
        cos = reference.cos(rows[0][lv], gated[lv])
        print(f"{case.name}: gated rows vs the stock chain, cos {cos:.7f}")
    for k in range(LAUNCHES):
        bufs.epoch += 1
        _core_launch(probe, bufs, ns, out, attn)
        if k % 50 == 49:
            torch.accelerator.synchronize()
            assert torch.equal(out.view(torch.int16), first.view(torch.int16)), k
            assert torch.equal(attn.view(torch.int16), first_attn.view(torch.int16))
    torch.accelerator.synchronize()
    assert not bufs.failed(), bufs.given_up_waits()
