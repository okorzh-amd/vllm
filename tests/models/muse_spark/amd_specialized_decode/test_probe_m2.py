# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""WP-M2 (DESIGN_BC 9.2, 10.1, 12 open questions 1-3): the speed-of-light
probes Phases B and C are sized by, in probe kernels
(``context.probe_kernel``). They print the numbers DESIGN_BC 11.1 is
recomputed from; they gate nothing but their own sanity.

    wstream   K1's weight stream without the kernel around it: the qkv GEMV's
              2304 (tile, K part) tasks of 16 x 512 bf16 (16 KiB each, 37.7 MB
              a rank) as K1's GEMV CTAs issue them (rounds of 2 tasks, waves
              0-3 / 4-7, every task's weights in flight from entry, then the
              MFMAs). Layouts: ``g192`` (E3: 12 tasks on each of the
              192 CTAs [64, 256), none on [0, 64)) and ``r256`` (the 10/6
              rebalance of open question 1: 6 on [0, 64), 10 on [64, 256)).
              B operand: one constant LDS row of K part width, no polls
              (``lds``: 9.2's speed of light, the 192-vs-256 decision); or,
              as K1 gets it, behind the ``ardy`` poll (5.5: it retires when
              the CTA's weights have landed) from the ``arow`` rows at device
              scope (``ld_dev``), either straight into the MFMAs (``global``:
              ladder rung 1, 11.3) or through 5.5's one staging pass into LDS
              (``staged``); the host pre-publishes ``ardy``
    hop       a device-scope hand-off round trip, CTA 0 <-> CTA b: ``put`` /
              ``poll`` (CM_DEV) ping-pong, ROUNDS rounds a launch, both CTAs'
              XCD read from ``HW_REG_XCC_ID`` (same vs cross XCD)
    otail     K2 phase 1 (o_proj: 16 MiB a rank, both tasks' weights first,
              ``load_core``, ``stage_o``) with a per-CTA issue delay: the
              16 S combine CTAs [C0(S), 256) and the split CTAs issue their
              o_proj weights late (DESIGN_BC 7.2), and every CTA's core only
              ready at the combine's time (``load_core`` behind the ``crdy``
              poll, its weights in flight). The tail C pays is the span
              against every CTA issuing at entry with the same core time; the
              span with the core ready at entry is reported too (9.2's
              literal "256 vs 256 - 16 S plus 16 S late issuers")

Every CTA stamps ``s_memrealtime`` (100 MHz, device-wide) at entry, when its
arguments have landed (a probe reads them from a table, a real kernel from
SGPRs), when it issues its weights and when it is done, and records its XCD
and CU; a span is the last CTA's end minus the first CTA's argument arrival.
Every probe holds more than half a CU's LDS (``PIN_BYTES``), so it runs one
CTA a CU as K1 and K2 do (DESIGN_BC 5.9), checked from the stamps. Weights
rotate through more
sets than the 256 MB MALL holds, so every launch streams from HBM, as serving
does (DESIGN_BC E12: one-layer replays keep weights hot).

[CPU] the task maps (every task once; 2 contributors a tile under E3, 4 under
the rebalance), the scenario tables; every probe compiles GPU-less for gfx950
with no private segment, within the VGPR and LDS budget, one CTA a CU, no
readfirstlane waterfall, and the weight probe has every task's loads issued
before its first MFMA (and, behind the ``ardy`` poll, before the poll).
[1GPU] the numbers (printed; with ``MUSE_M2_OUT`` set also appended to that
file as JSON lines), each probe checked first: the weight probe's checksum
against fp64 (every task streamed once), the o_proj partials against fp64
(and carrying this launch's tag), no wait given up.
"""

import glob
import json
import os
import statistics

import pytest
import regex as re
import torch

flydsl = pytest.importorskip("flydsl")
pytest.importorskip("aiter")

import flydsl.expr as fx  # noqa: E402
from aiter.ops.flydsl.kernels import buffer_ops as bo  # noqa: E402
from flydsl._mlir.dialects import llvm  # noqa: E402
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl  # noqa: E402
from flydsl.expr.typing import T  # noqa: E402

from vllm.models.muse_spark.amd.specialized_decode.mono import spec  # noqa: E402
from vllm.models.muse_spark.amd.specialized_decode.mono.abi import (  # noqa: E402
    WAIT_BUDGET,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.attention import (  # noqa: E402
    back,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common import (  # noqa: E402
    context,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (  # noqa: E402
    batched_rounds,
    bf16_pair,
    ld_dev,
    memrealtime,
    mfma_bf16,
    rsrc,
    traced,
    uniform,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.plan import (  # noqa: E402
    Space,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.sync import (  # noqa: E402
    Addr,
    uniform_addr,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.stages import (  # noqa: E402
    gemv,
)

THREADS = spec.THREADS
BLOCKS = spec.BLOCKS
H = spec.HIDDEN
MT = spec.MAX_TOKENS
TICK_NS = 10  # s_memrealtime: 100 MHz
MALL_BYTES = 256 << 20  # MI355X's Infinity Cache
ROTATE_BYTES = 320 << 20  # weight sets a test cycles through: past the MALL

# ---------------------------------------------------------------- task maps

# The weight-stream layouts: (first CTA, end CTA, tasks a CTA, first task);
# CTA b of a role runs tasks tau0 + n (b - lo) + i, i < n.
W_LAYOUTS = {
    "g192": ((0, spec.K1_GEMV0, 0, 0), (spec.K1_GEMV0, BLOCKS, spec.K1_TPC, 0)),
    "r256": ((0, spec.NA, 6, 0), (spec.NA, BLOCKS, 10, 6 * spec.NA)),
}
W_BYTES = spec.QKV_ROWS * H * 2  # the qkv weight a rank: 37,748,736 B
TASK_BYTES = spec.ROWS * spec.KPART * 2  # 16 KiB
XL_ROW = spec.KPART + gemv.LDS_PAD  # bf16 a padded constant LDS row


def task_map(layout: str) -> dict[int, list[int]]:
    """CTA -> its tasks (tau = 16 tile + K part) under ``layout``."""
    out: dict[int, list[int]] = {}
    for lo, hi, n, tau0 in W_LAYOUTS[layout]:
        for b in range(lo, hi):
            out[b] = [tau0 + n * (b - lo) + i for i in range(n)]
    return out


def tile_contributors(layout: str) -> dict[int, set[int]]:
    """Tile -> the CTAs holding one of its K parts."""
    out: dict[int, set[int]] = {}
    for b, taus in task_map(layout).items():
        for tau in taus:
            out.setdefault(tau // spec.KPARTS, set()).add(b)
    return out


# The o_proj tail scenarios (DESIGN_BC 7.2): name -> (late0, late_ticks,
# split_n, split_ticks, core_ticks): CTAs [late0, 256) issue their o_proj
# weights late_ticks after entry, CTAs [0, min(split_n, late0)) split_ticks
# after it, the rest at entry; every CTA's core is ready core_ticks after entry
# (K2(split)'s ``load_core`` polls ``crdy``, published when the combine is
# done: no CTA's stage_o starts before it, early weights or not). SPLIT_US
# stands for stage_split, COMBINE_US for split + srdy hop + combine + crdy
# publish; the GPU numbers replace both.
SPLIT_US, COMBINE_US = 1.5, 4.0
LATE_US = (1.0, 2.0, 4.0, 6.0)  # the combine time, swept


def _ticks(us: float) -> int:
    return round(us * 1000 / TICK_NS)


def otail_scenarios(S: int) -> dict[str, tuple[int, int, int, int, int]]:
    """Scenario -> its delays; ``otail_baseline`` names the scenario each one's
    tail is taken against.

    entry                  Phase A: every CTA's weights and core at entry
    early_{d}us            every CTA's weights at entry, the core at d (7.2's
                           A/B variant: the combine CTAs issue before the srdy
                           poll; that variant's cost to the combine is C3's)
    combine_late_{d}us     7.2's order: the 16 S combine CTAs' weights at d,
                           the core at d (tail against early_{d}us)
    combine_late_{d}us_core0
                           the same weights, the core at entry (9.2's literal
                           probe; tail against entry: an upper bound)
    split_{kind}           7.2 with the split CTAs' weights at SPLIT_US and
                           the combine CTAs' at COMBINE_US, the core at
                           COMBINE_US (tail against early_{COMBINE_US}us)"""
    c0 = spec.C0(S)
    out = {"entry": (BLOCKS, 0, 0, 0, 0)}
    for d in LATE_US:
        out[f"early_{d:g}us"] = (BLOCKS, 0, 0, 0, _ticks(d))
        out[f"combine_late_{d:g}us"] = (c0, _ticks(d), 0, 0, _ticks(d))
        out[f"combine_late_{d:g}us_core0"] = (c0, _ticks(d), 0, 0, 0)
    for window, kind in ((True, "sliding"), (False, "full")):
        nu = spec.n_units(S, window)
        late = _ticks(COMBINE_US)
        out[f"split_{kind}"] = (c0, late, nu, _ticks(SPLIT_US), late)
    return out


def otail_baseline(name: str) -> str:
    """The scenario ``name``'s tail is taken against (``otail_scenarios``)."""
    if name.startswith("split_"):
        return f"early_{COMBINE_US:g}us"
    if name.startswith("combine_late_") and not name.endswith("_core0"):
        return "early_" + name[len("combine_late_") :]
    return "entry"


def delay_ticks(scenario: tuple[int, int, int, int, int]) -> torch.Tensor:
    """The per-CTA issue delay (ticks) ``_otail`` applies for a scenario."""
    late0, late_ticks, split_n, split_ticks, _ = scenario
    b = torch.arange(BLOCKS)
    split = torch.where(b < split_n, split_ticks, 0)
    return torch.where(b >= late0, late_ticks, split).int()


ROUNDS = 16  # hop round trips a launch
WARM = 2  # rounds dropped (cold instruction cache, first touch)
STAMPS = ("entry", "args", "issue", "done", "xcc", "cu")  # a CTA's stamp words
STAMP_WORDS = 8
OUT_ACC = BLOCKS * STAMP_WORDS  # the weight probe's accumulators after the stamps
# LDS every probe holds (its last word touched, so it stays allocated): with
# ``red``, past half a CU's, so one CTA a CU as K1 and K2 run (DESIGN_BC 5.9).
# The probes' own VGPRs and LDS would let 2-4 CTAs share a CU and leave CUs
# idle: neither the bandwidth nor the hop of the real kernels
PIN_BYTES = 81 << 10


@fx.struct
class PinLds:
    pin: fx.Array[fx.Int32, PIN_BYTES // 4, 16]


def _pinned(lds_structs: dict | None) -> dict:
    """A probe's LDS stages and the residency pin (a union member)."""
    return {**(lds_structs or {}), "pin": PinLds}


# ---------------------------------------------------------------- device helpers


def _st(out, i, v):
    """out[i] := v (plain store; an fp32 by its bits)."""
    bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
    bo.buffer_store(bits, rsrc(out), i)


def _tick():
    """The 100 MHz device clock, 31 bits (the host takes differences mod 2^31)."""
    return fx.Int32(memrealtime() & 0x7FFFFFFF)


def _xcc():
    """This CTA's XCD (``HW_REG_XCC_ID`` bits [3:0], wave-uniform)."""
    v = llvm.inline_asm(
        T.i32, [], "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)", "=s",
        has_side_effects=True,
    )  # fmt: skip
    return fx.Int32(v) & 0xF


def _cu():
    """This CTA's CU within its XCD (``HW_REG_HW_ID`` CU_ID, SH_ID, SE_ID:
    bits [15:8])."""
    v = llvm.inline_asm(
        T.i32, [], "s_getreg_b32 $0, hwreg(HW_REG_HW_ID)", "=s",
        has_side_effects=True,
    )  # fmt: skip
    return (fx.Int32(v) >> 8) & 0xFF


@traced
def _pin(c):
    """Thread 0 touches the residency pin's last word (``PIN_BYTES``)."""
    if c["tid"] == 0:
        fx.ptr_store(fx.Int32(0), c["lds"]["pin"]["pin"] + (PIN_BYTES // 4 - 1))


def _landed(v):
    """An opaque scalar move of ``v`` (an argument the probe's table holds):
    it waits for the table's load, and a clock read after it is ordered after
    it (both have side effects). A probe's arguments come through ``argv``, a
    load a real kernel does not have (its kernel arguments are SGPRs from the
    kernarg segment): spans are taken from this point."""
    llvm.inline_asm(
        T.i32, [fx.Int32(v).ir_value()], "s_mov_b32 $0, $1", "=s,s",
        has_side_effects=True,
    )  # fmt: skip


@traced
def _stamps(c, entry, args, issue, done):
    """Thread 0: this CTA's stamps (``STAMPS``) -> ``out`` words
    ``STAMP_WORDS`` b .. (``out`` re-read from lane 0: a store under ``tid ==
    0`` would otherwise merge its buffer resource into a waterfall)."""
    xcc = _xcc()
    cu = _cu()
    out = _uniform_out(c)
    if c["tid"] == 0:
        base = c["bid"] * STAMP_WORDS
        for i, v in enumerate((entry, args, issue, done, xcc, cu)):
            _st(out, base + i, v)


def _uniform_out(c):
    """The ``out`` argument re-read from lane 0 (a new wave-uniform value)."""
    return uniform_addr(Addr("out", Space.SCRATCH, c["args"]["out"])).value


def _const_x(t, k):
    """The weight probe's constant B row: x[t, k] = (((5 k + 3 t) % 16) + 1)
    / 16, exact in bf16, positive (the checksum has no cancellation)."""
    return fx.Float32((k * 5 + t * 3) % 16 + 1) * fx.Float32(0.0625)


def host_x(S: int, k: int) -> torch.Tensor:
    t = torch.arange(S)[:, None]
    kk = torch.arange(k)[None, :]
    return (((kk * 5 + t * 3) % 16 + 1) / 16).to(torch.bfloat16)


# ---------------------------------------------------------------- probe bodies


@traced
def _ardy_poll(c, tau0, n):
    """5.5's ``ardy`` poll: thread i < 4 n polls the flag of boundary slice
    4 kp + i % 4 of its task i // 4's K part (the host published every flag),
    then a barrier. Its ``_spin`` waits ``vmcnt(0)``: it retires when the
    CTA's weights have landed (5.3)."""
    tid = c["tid"]
    if tid < 4 * n:
        kp = (tau0 + tid // 4) % spec.KPARTS
        c["poll"]([(c["scr"]["ardy"], spec.ardy_pair(4 * kp + tid % 4), 1)])
    gpu.barrier()


def _stage_ij(v, s):
    """A staging element v -> (task i, token t, dwordx4 x of the K part)."""
    per = spec.KPART // 8
    return v // (s * per), v // per % s, v % per


@traced
def _stage_arow(c, tau0, n):
    """5.5's one staging pass: the CTA's n K parts of the ``arow`` rows (S
    rows each, device scope) -> LDS ``xl`` (task i's rows at i S), every
    load issued before any LDS store, one barrier."""
    s, tid = c["S"], c["tid"]
    arow = c["scr"][spec.AROW]
    xl = c["lds"]["wstream"]["xl"]
    total = n * s * spec.KPART // 8
    batched_rounds(
        tid, total,
        lambda v: fx.Vector(ld_dev(arow, _arow_word(tau0, v, s), 4)),
        lambda v, w: fx.ptr_store(w, xl + _xl_word(v, s)),
        batch=-(-total // THREADS),
    )  # fmt: skip
    gpu.barrier()


def _arow_word(tau0, v, s):
    i, t, x = _stage_ij(v, s)
    kp = (tau0 + i) % spec.KPARTS
    return (t * H + kp * spec.KPART + 8 * x) // 2


def _xl_word(v, s):
    i, t, x = _stage_ij(v, s)
    return ((i * s + t) * XL_ROW + 8 * x) // 2


@traced
def _wtasks(c, t0, t1, tau0, n, bsrc):
    """``n`` (even) tasks from ``tau0`` on this CTA: rounds of 2 (waves 0-3
    task 2 r, waves 4-7 task 2 r + 1; wave w % 4 takes K chunk w % 4 of the
    512-wide K part), every round's weights issued before anything else, then
    the B operands (``bsrc``, module docstring), then the MFMAs into one
    accumulator a wave. Stamped when every wave's accumulator is in ``red``;
    the accumulators -> ``out`` after."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    a = c["args"]
    w4 = wave % 4
    half = wave // 4
    g = lane // spec.ROWS
    t = fx.min(lane % spec.ROWS, s - 1)
    ops, kps = [], []
    for r in range_constexpr(n // 2):
        tau = tau0 + 2 * r + half
        kp = tau % spec.KPARTS
        row0 = tau // spec.KPARTS * spec.ROWS
        ops.append(
            gemv.loads(lane, w4, a["w"], row0, spec.KPART, ldk=H, k0=kp * spec.KPART)[0]
        )
        kps.append(kp)
    # every load stays where it is issued (LLVM would sink the B rows' and the
    # late rounds' loads to their MFMAs to save registers: not all in flight)
    rocdl.sched_barrier(0)
    xl = c["lds"]["wstream"]["xl"] if bsrc != "global" else None
    xs = []
    if const_expr(bsrc == "lds"):
        words = s * spec.KPART // 2
        for i in range_constexpr((words + THREADS - 1) // THREADS):
            e = fx.min(tid + THREADS * i, words - 1)
            tt = e // (spec.KPART // 2)
            k = e % (spec.KPART // 2) * 2
            v = bf16_pair(_const_x(tt, k), _const_x(tt, k + 1))
            if tid + THREADS * i < words:
                fx.ptr_store(v.bitcast(fx.Int32), xl + (tt * XL_ROW + k) // 2)
        gpu.barrier()
    else:
        _ardy_poll(c, tau0, n)
    if const_expr(bsrc == "global"):
        arow = c["scr"][spec.AROW]
        for r in range_constexpr(n // 2):
            base = (t * H + kps[r] * spec.KPART + w4 * gemv.KCH + g * 8) // 2
            xs.append([fx.Vector(ld_dev(arow, base + 16 * q, 4)) for q in range(4)])
        rocdl.sched_barrier(0)
    if const_expr(bsrc == "staged"):
        _stage_arow(c, tau0, n)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for r in range_constexpr(n // 2):
        for q in range_constexpr(4):
            if const_expr(bsrc == "global"):
                xb = xs[r][q]
            else:
                kk = w4 * gemv.KCH + g * 8 + 32 * q
                brow = (2 * r + half) * s + t if bsrc == "staged" else t
                xb = fx.Vector(
                    fx.ptr_load(
                        xl + (brow * XL_ROW + kk) // 2,
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                )
            wb = ops[r][q].bitcast(fx.BFloat16)
            acc = mfma_bf16(wb, xb.bitcast(fx.BFloat16), acc)
    fx.ptr_store(acc, c["red"] + (wave * 64 + lane) * 4)
    gpu.barrier()
    _stamps(c, t0, t1, t1, _tick())
    at = OUT_ACC + (c["bid"] * THREADS + tid) * 4
    bo.buffer_store(acc.bitcast(fx.Int32), rsrc(_uniform_out(c)), at)


def _wstream_body(layout: str, bsrc: str):
    roles = W_LAYOUTS[layout]

    @traced
    def body(c):
        """The weight stream of ``layout`` (module docstring), stamped; a CTA
        without tasks only stamps (the host zeroes ``out``)."""
        b = c["bid"]
        _pin(c)
        t0 = _tick()
        _landed(c["args"]["w"])
        t1 = _tick()
        for i in range_constexpr(len(roles)):
            lo, hi, n, tau0 = roles[i]
            if (b >= lo) & (b < hi):
                if const_expr(n > 0):
                    _wtasks(c, t0, t1, tau0 + n * (b - lo), n, bsrc)
                else:
                    _stamps(c, t0, t1, t1, _tick())

    return body


def _wstream_lds(rows: int):
    """``rows`` padded bf16 B rows: S (``lds``), or S a staged task
    (``staged``)."""

    @fx.struct
    class WstreamLds:
        xl: fx.Array[fx.Int32, rows * XL_ROW // 2, 16]

    return WstreamLds


@traced
def _hop(c):
    """CTA 0 pings CTA ``partner`` ROUNDS times (``ass`` pair r), the partner
    answers each (``bss`` pair r); thread 0 of each. Stamps: CTA 0's put and
    back times, the partner's seen times, both XCDs (kept in registers until
    the last round: a store in flight would hold up the next poll's vmcnt)."""
    _pin(c)
    tid, b32 = c["tid"], fx.Int32(c["bid"])
    p = uniform(fx.Int32(c["args"]["partner"]))
    ping, pong = c["scr"]["ass"], c["scr"]["bss"]
    xcc = _xcc()
    out = _uniform_out(c)
    if (b32 == 0) & (tid == 0):
        sent, back_ = [], []
        for r in range_constexpr(ROUNDS):
            sent.append(_tick())
            c["put"](ping, r, r + 1)
            c["poll"]([(uniform_addr(pong), r, 1)])
            back_.append(_tick())
        _st(out, 0, xcc)
        for r in range_constexpr(ROUNDS):
            _st(out, 8 + r, sent[r])
            _st(out, 8 + ROUNDS + r, back_[r])
    if (b32 == p) & (tid == 0):
        seen = []
        for r in range_constexpr(ROUNDS):
            v = c["poll"]([(uniform_addr(ping), r, 1)])[0][0]
            seen.append(_tick())
            c["put"](pong, r, v)
        _st(out, 1, xcc)
        for r in range_constexpr(ROUNDS):
            _st(out, 8 + 2 * ROUNDS + r, seen[r])


@traced
def _otail(c):
    """K2 phase 1 with this CTA's o_proj weights issued late: CTAs from
    ``late0`` wait ``late_ticks``, CTAs below ``split_n`` (and below
    ``late0``) ``split_ticks``, counted from the arguments' arrival (a spin on
    the clock; the scalars come with the argument table, no extra load before
    the weights). Then, the weights in flight, every CTA waits until
    ``core_ticks`` (the ``crdy`` poll ``load_core`` makes in K2(split), a
    spin on the clock here: no wait on the weights) before ``load_core``.
    Stamped."""
    b = c["bid"]
    b32 = fx.Int32(b)
    a = c["args"]
    _pin(c)
    entry = _tick()
    _landed(a["w_o"])
    t0 = memrealtime()
    late = b32 >= fx.Int32(a["late0"])
    split = b32 < fx.Int32(a["split_n"])
    d = late.select(
        fx.Int32(a["late_ticks"]),
        split.select(fx.Int32(a["split_ticks"]), fx.Int32(0)),
    )
    now = memrealtime()
    while now - t0 < fx.Int64(d):
        rocdl.s_sleep(1)
        now = memrealtime()
    issue = _tick()
    ops0 = back.o_loads(c, b)
    ops1 = back.o_loads(c, b + BLOCKS)
    # the weights stay ahead of the core wait (in flight through it)
    rocdl.sched_barrier(0)
    core = fx.Int64(fx.Int32(a["core_ticks"]))
    now = memrealtime()
    while now - t0 < core:
        rocdl.s_sleep(1)
        now = memrealtime()
    back.load_core(c)
    back.stage_o(c, b, ops0)
    back.stage_o(c, b + BLOCKS, ops1)
    _stamps(c, entry, fx.Int32(t0 & 0x7FFFFFFF), issue, _tick())


WSTREAM_CASES = (
    ("g192", "lds", 1),
    ("g192", "lds", 8),
    ("r256", "lds", 1),
    ("r256", "lds", 8),
    ("g192", "global", 1),
    ("g192", "global", 8),
    ("g192", "staged", 1),
    ("g192", "staged", 8),
)
# o_proj's K2 arguments, the stamps, then the 7.2 issue delays and the core's
# ready time (scalars)
OTAIL_ARGS = ("core", "pad", "w_o", "out")
OTAIL_DELAYS = ("late0", "late_ticks", "split_n", "split_ticks", "core_ticks")


def _probe(which: str, *case) -> context.Probe:
    if which == "wstream":
        layout, bsrc, S = case
        nmax = max(n for _, _, n, _ in W_LAYOUTS[layout])
        rows = {"lds": S, "staged": nmax * S}.get(bsrc)
        lds = {"wstream": _wstream_lds(rows)} if rows else None
        return context.probe_kernel(
            f"muse_probe_m2_wstream_{layout}_{bsrc}_s{S}",
            S,
            _wstream_body(layout, bsrc),
            _pinned(lds),
            ("w", "out"),
            blocks=BLOCKS,
        )
    if which == "hop":
        return context.probe_kernel(
            "muse_probe_m2_hop",
            1,
            _hop,
            _pinned(None),
            ("out", "partner"),
            blocks=BLOCKS,
        )
    assert which == "otail", which
    (S,) = case
    return context.probe_kernel(
        f"muse_probe_m2_otail_s{S}",
        S,
        _otail,
        _pinned({"back": back.lds_struct(S)}),
        OTAIL_ARGS + OTAIL_DELAYS,
        blocks=BLOCKS,
    )


# ---------------------------------------------------------------- [CPU]


def test_task_maps():
    """Both layouts cover the 2304 tasks once each, in consecutive runs a CTA;
    E3's has 2 contributors on every tile (``K1_MAXC``), the rebalance up to 4
    (open question 1: qpart then holds 4 contributors a tile, 589,824 B)."""
    for layout, maxc, per in (("g192", spec.K1_MAXC, (0, 12)), ("r256", 4, (6, 10))):
        m = task_map(layout)
        assert sorted(sum(m.values(), [])) == list(range(spec.K1_TASKS)), layout
        assert set(m) == set(range(BLOCKS))
        assert {len(v) for b, v in m.items() if b < spec.NA} == {per[0]}
        assert {len(v) for b, v in m.items() if b >= spec.NA} == {per[1]}
        for taus in m.values():
            assert len(taus) % 2 == 0  # rounds of 2
            if taus:
                assert taus == list(range(taus[0], taus[0] + len(taus)))
        contrib = tile_contributors(layout)
        assert len(contrib) == spec.QKV_TILES
        assert max(len(v) for v in contrib.values()) == maxc, layout
    assert all(len(v) == 2 for v in tile_contributors("g192").values())
    # E3: tile T's contributors are k1_contributor0(T) and the next GEMV CTA
    for tile, ctas in tile_contributors("g192").items():
        g0 = spec.k1_contributor0(tile)
        assert ctas == {spec.K1_GEMV0 + g0, spec.K1_GEMV0 + g0 + 1}, tile
    # the rebalance's qpart: 4 contributors a tile
    assert spec.QKV_TILES * 4 * MT * spec.ROWS * spec.PAIR_BYTES == 589_824
    # bytes a streaming CTA: 192 KiB (E3), 160 / 96 KiB (rebalance)
    assert spec.K1_TPC * TASK_BYTES == 192 << 10
    assert (10 * TASK_BYTES, 6 * TASK_BYTES) == (160 << 10, 96 << 10)
    assert spec.K1_TASKS * TASK_BYTES == W_BYTES == 37_748_736
    # more weight sets than the MALL holds
    assert -(-ROTATE_BYTES // W_BYTES) * W_BYTES > MALL_BYTES
    assert -(-ROTATE_BYTES // (spec.CORE * H * 2)) * spec.CORE * H * 2 > MALL_BYTES


def test_otail_scenarios():
    """The 7.2 roles: 16 S combine CTAs from C0(S), the split CTAs below
    them (32 sliding units at S = 1; every CTA on a full layer at S = 1 and 8
    and on a sliding one at S = 8), the delays in ticks; each tail's baseline
    has the same core time and every weight at entry; the hop rounds fit
    ``ass`` / ``bss``."""
    for S in (1, 8):
        c0 = spec.C0(S)
        sc = otail_scenarios(S)
        assert not delay_ticks(sc["entry"]).any() and sc["entry"][4] == 0
        d = delay_ticks(sc["combine_late_4us"])
        assert (d[c0:] == 400).all() and not d[:c0].any()
        assert sc["combine_late_4us"][4] == 400
        assert torch.equal(delay_ticks(sc["combine_late_4us_core0"]), d)
        assert sc["combine_late_4us_core0"][4] == 0
        for name, v in sc.items():
            base = sc[otail_baseline(name)]
            assert not delay_ticks(base).any(), name
            assert (
                base[4] == v[4]
                or name.endswith("_core0")
                or (name.startswith("early_"))
            ), name
        assert BLOCKS - c0 == 16 * S
        # the split CTAs: [0, n_units) below C0; a unit's CTA at or past C0
        # runs it and then combines, issuing after the combine (late0 wins)
        for window, kind in ((True, "sliding"), (False, "full")):
            nu = spec.n_units(S, window)
            want = torch.zeros(BLOCKS, dtype=torch.int32)
            want[: min(nu, c0)] = 150
            want[c0:] = 400
            assert torch.equal(delay_ticks(sc[f"split_{kind}"]), want), (S, kind)
            assert sc[f"split_{kind}"][4] == 400
        assert spec.n_units(1, True) == 32 and spec.n_units(S, False) == BLOCKS
        for v in sc.values():
            assert all(0 <= x < 2**31 for x in v)
    lay = spec.scratch_layout()
    assert lay["ass"][1] // 8 >= ROUNDS and lay["bss"][1] // 8 >= ROUNDS
    assert OUT_ACC + BLOCKS * THREADS * 4 < 1 << 22


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

# readfirstlane waterfall around a load whose base is not wave-uniform
_WATERFALL = re.compile(r"v_cmp_eq_u64_e\d+\s+\S+,\s*s\[\d+:\d+\],\s*v\[")
# a weight load (nontemporal), a device-scope B row load (sc1)
_LOAD_NT = re.compile(r"^\s*buffer_load_dwordx4\b[^\n]*\boffen\b[^\n]*\bnt\b", re.M)
_LOAD_DEV = re.compile(r"^\s*buffer_load_dwordx4\b[^\n]*\bsc1\b", re.M)
# a wait right before an MFMA (its vmcnt: the loads it lets stay in flight)
_FIRST_WAIT = re.compile(
    r"^\s*s_waitcnt vmcnt\((\d+)\)[^\n]*\n\s*v_mfma_f32_16x16x32_bf16", re.M
)
_MFMA = re.compile(r"^\s*v_mfma_f32_16x16x32_bf16\b", re.M)

COMPILE_CASES = [("wstream", *c) for c in WSTREAM_CASES] + [
    ("hop",),
    ("otail", 1),
    ("otail", 8),
]


@compiles
@pytest.mark.parametrize("case", COMPILE_CASES, ids=lambda c: "_".join(map(str, c)))
def test_compiles(case, tmp_path, monkeypatch):
    """The GPU-less compile gate (README, Residency) on each M2 probe; the
    weight probe issues every task's weights before anything else (4 dwordx4
    a lane a round, all in flight: before its first MFMA, and before the
    ``ardy`` poll that waits for them); ``global`` issues its B rows (device
    scope) after the poll and before its first MFMA; ``otail`` keeps its
    weights in flight through the core wait."""
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    probe = _probe(*case)
    _compile(probe)
    txt, res = _isa(str(tmp_path), probe.name)
    print(f"{probe.name}: {res}")
    assert res["private_segment_fixed_size"] == 0, res
    assert res["next_free_vgpr"] <= spec.VGPR_MAX, res
    assert res["group_segment_fixed_size"] <= spec.LDS_BYTES, res
    # one CTA a CU (DESIGN_BC 5.9): the residency pin is allocated
    assert res["group_segment_fixed_size"] > spec.LDS_BYTES // 2, res
    loops = _WATERFALL.findall(txt)
    assert not loops, f"{len(loops)} waterfall compares, e.g. {loops[0]}"
    assert "s_memrealtime" in txt
    if case[0] == "hop":
        assert "s_getreg_b32" in txt and "HW_REG_XCC_ID" in txt
        assert "v_mfma" not in txt
        return
    assert _MFMA.search(txt)
    first = txt[: _MFMA.search(txt).start()]
    if case[0] == "otail":
        # issue delay spin, the 8 weight loads, then the core wait's spin with
        # no vmcnt wait between (the weights stay in flight through it)
        nt = [m.end() for m in _LOAD_NT.finditer(first)]
        assert len(nt) == 8, len(nt)
        sleeps = [m.start() for m in re.finditer(r"\bs_sleep\b", first)]
        assert sleeps and sleeps[0] < nt[0], "the issue delay is not first"
        gate = [p for p in sleeps if p > nt[-1]]
        assert gate, "no core wait after the weights"
        assert "vmcnt" not in first[nt[-1] : gate[0]], "weights drained early"
        return
    layout, bsrc, _ = case[1:]
    rounds = [n // 2 for _, _, n, _ in W_LAYOUTS[layout] if n]
    nt = _LOAD_NT.findall(first)
    assert len(nt) in {4 * r for r in rounds}, len(nt)
    assert len(_MFMA.findall(txt)) == 4 * sum(rounds)
    if bsrc != "lds":
        # every weight before the ardy poll, B (device scope) after it
        poll = re.search(r"\bs_sleep\b", txt).start()
        assert len(_LOAD_NT.findall(txt[:poll])) in {4 * r for r in rounds}
        assert not _LOAD_NT.findall(txt[poll : _MFMA.search(txt).start()])
        dev = _LOAD_DEV.findall(txt[poll : _MFMA.search(txt).start()])
        if bsrc == "global":
            assert len(dev) in {4 * r for r in rounds}, len(dev)
        else:
            assert dev, "no staging load"
            assert "ds_write" in txt[poll : _MFMA.search(txt).start()]
    if bsrc == "staged":
        return
    # the first MFMA waits for its round's oldest loads only: vmcnt(4 rounds
    # - 1) (lds: the weights; global: the B rows, issued after the poll
    # drained the weights)
    waits = [int(m.group(1)) for m in _FIRST_WAIT.finditer(txt)]
    assert max(waits) == 4 * max(rounds) - 1, waits
    assert 4 * min(rounds) - 1 in waits, waits


# ---------------------------------------------------------------- [1GPU]


def _report(name: str, numbers: dict) -> None:
    """Print the probe's numbers (and append them to ``$MUSE_M2_OUT``)."""
    line = json.dumps({"probe": name, **numbers}, sort_keys=True)
    print(f"M2 {line}")
    path = os.environ.get("MUSE_M2_OUT")
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")


def _assert_healthy(bufs) -> None:
    assert not bufs.failed(), bufs.given_up_waits()
    h = bufs.health.cpu().clone()
    h[WAIT_BUDGET] = 0
    assert not h.any(), bufs.given_up_waits()
    assert not bufs.flags().any()


def _stamps_of(out: torch.Tensor) -> dict[str, torch.Tensor]:
    """The CTAs' stamps (``STAMPS``; clocks in ticks from the earliest
    entry, mod 2^31)."""
    s = out[: BLOCKS * STAMP_WORDS].cpu().view(BLOCKS, STAMP_WORDS).long()
    rel = (s[:, :4] - s[:, 0].min()) % (1 << 31)
    got = {k: rel[:, i] for i, k in enumerate(STAMPS[:4])}
    got["xcc"] = s[:, 4]
    got["cu"] = s[:, 5]
    return got


def _ctas_a_cu(st: dict) -> int:
    """The most CTAs any one CU (XCD, CU) ran: 1 as K1 / K2 run."""
    key = st["xcc"] * 256 + st["cu"]
    return int(torch.unique(key, return_counts=True)[1].max())


def _span(st: dict) -> int:
    """The grid's span from the arguments' arrival to the last CTA's end."""
    return int(st["done"].max() - st["args"].min())


def _us(ticks) -> float:
    return float(ticks) * TICK_NS / 1000


def _weight_sets(rows: int, cols: int, g: torch.Generator) -> list[torch.Tensor]:
    """Weight sets past the MALL, bf16 in [0, 1) (positive: no cancellation in
    the checksum), each a distinct buffer."""
    n = -(-ROTATE_BYTES // (rows * cols * 2))
    return [
        torch.rand(rows, cols, generator=g, device="cuda").to(torch.bfloat16)
        for _ in range(n)
    ]


LAUNCHES = 64  # timed launches a case (weights rotating)


@needs_gfx950
@pytest.mark.parametrize("case", WSTREAM_CASES, ids=lambda c: "_".join(map(str, c)))
def test_weight_stream(case):
    """Open question 1: the qkv weight stream's achieved bandwidth with 192 vs
    256 streaming CTAs; ladder rung 1: the B operand behind the ``ardy`` poll
    straight from ``arow`` vs 5.5's staging pass. Every launch's checksum
    equals fp64's (every task streamed once); the CTAs' clocks agree (entries
    within 50 us: a per-XCD clock would not)."""
    layout, bsrc, S = case
    probe = _probe("wstream", *case)
    bufs = context.ProbeBuffers()
    g = torch.Generator(device="cuda").manual_seed(0)
    sets = _weight_sets(spec.QKV_ROWS, H, g)
    if bsrc != "lds":
        # the boundary's output: arow rows, every ardy flag published (the
        # epoch stays: the flags hold this tag in every launch)
        x = (torch.rand(S, H, generator=g, device="cuda") + 0.5).to(torch.bfloat16)
        buf, off = bufs.where(spec.AROW)
        buf[off : off + S * H * 2].view(torch.bfloat16).copy_(x.view(-1))
        bufs.write_pairs("ardy", 0, torch.zeros(spec.NA, dtype=torch.int32))
        xcols = x.double()[torch.arange(16).clamp(max=S - 1)].sum(0)  # [H]
        want = [(w.double().sum(0) * xcols).sum().item() for w in sets]
    else:
        x = host_x(S, spec.KPART).double()
        xcols = x[torch.arange(16).clamp(max=S - 1)].sum(0).cuda()  # [512]
        want = [
            (w.double().view(-1, spec.KPART).sum(0) * xcols).sum().item() for w in sets
        ]
    out = torch.zeros(OUT_ACC + BLOCKS * THREADS * 4, dtype=torch.int32, device="cuda")
    for i in range(len(sets)):  # warm-up: the JIT and each set once
        probe(bufs, w=sets[i], out=out)
    torch.accelerator.synchronize()
    spans, raw, ctas, skew, argl = [], [], [], [], []
    for i in range(LAUNCHES):
        k = i % len(sets)
        out.zero_()
        probe(bufs, w=sets[k], out=out)
        torch.accelerator.synchronize()
        _assert_healthy(bufs)
        got = out[OUT_ACC:].view(torch.float32).double().sum().item()
        assert abs(got - want[k]) <= 2e-5 * abs(want[k]), (got, want[k])
        st = _stamps_of(out)
        assert _ctas_a_cu(st) == 1, "two CTAs shared a CU: not K1's placement"
        spans.append(_span(st))
        raw.append(int(st["done"].max()))
        ctas.append(int((st["done"] - st["args"]).max()))
        skew.append(int(st["entry"].max()))
        argl.append(int((st["args"] - st["entry"]).max()))
    assert max(skew) < 5000, f"CTA entries {max(skew)} ticks apart"
    span = statistics.median(spans)
    _report(
        "wstream",
        {
            "layout": layout,
            "bsrc": bsrc,
            "S": S,
            "span_us": _us(span),
            "span_us_p10_p90": [_us(q) for q in statistics.quantiles(spans, n=10)[::8]],
            "tb_per_s": W_BYTES / (span * TICK_NS * 1e-9) / 1e12,
            "span_from_entry_us": _us(statistics.median(raw)),
            "slowest_cta_us": _us(statistics.median(ctas)),
            "entry_skew_us": _us(statistics.median(skew)),
            "argv_wait_us": _us(statistics.median(argl)),
        },
    )


@needs_gfx950
def test_hop_latency():
    """Open question 2: a device-scope hand-off, CTA 0 <-> every CTA b, same
    vs cross XCD (read from the hardware, not assumed b % 8). Hop = half the
    round trip; one-way = the partner's seen stamp - CTA 0's put stamp."""
    probe = _probe("hop")
    bufs = context.ProbeBuffers(epoch=3)
    out = torch.zeros(8 + 3 * ROUNDS, dtype=torch.int32, device="cuda")
    rows = []
    for b in [1, *range(1, BLOCKS)]:  # the first launch is a warm-up
        bufs.epoch += 1  # a new tag: last launch's pairs are stale
        out.zero_()
        probe(bufs, out=out, partner=b)
        torch.accelerator.synchronize()
        _assert_healthy(bufs)
        o = out.cpu().long()
        sent, back_, seen = (
            o[8 + i * ROUNDS : 8 + (i + 1) * ROUNDS][WARM:] for i in range(3)
        )
        rtt = ((back_ - sent) % (1 << 31)).double()
        one = ((seen - sent) % (1 << 31)).double()
        rows.append((b, int(o[0]), int(o[1]), rtt.median().item(), one.median().item()))
    rows = rows[1:]
    xcc0 = rows[0][1]
    assert {r[1] for r in rows} == {xcc0}
    xccs = sorted({r[2] for r in rows})
    by_xcc = {x: statistics.median([r[3] / 2 for r in rows if r[2] == x]) for x in xccs}
    same = [r[3] / 2 for r in rows if r[2] == xcc0]
    cross = [r[3] / 2 for r in rows if r[2] != xcc0]
    one_same = [r[4] for r in rows if r[2] == xcc0]
    one_cross = [r[4] for r in rows if r[2] != xcc0]
    assert same and cross, xccs
    _report(
        "hop",
        {
            "xcds": len(xccs),
            "cta_xcd_is_b_mod_8": all(r[2] == (r[0] + xcc0) % 8 for r in rows),
            "hop_same_xcd_us": _us(statistics.median(same)),
            "hop_cross_xcd_us": _us(statistics.median(cross)),
            "hop_cross_xcd_max_us": _us(max(cross)),
            "one_way_same_xcd_us": _us(statistics.median(one_same)),
            "one_way_cross_xcd_us": _us(statistics.median(one_cross)),
            "hop_us_by_partner_xcd": {str(x): _us(v) for x, v in by_xcc.items()},
        },
    )


def _ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    def ordered(x):
        v = x.contiguous().view(torch.int16).int()
        m = v & 0x7FFF
        return torch.where(v < 0, -m, m)

    return (ordered(a) - ordered(b)).abs()


@needs_gfx950
@pytest.mark.parametrize("S", [1, 8])
def test_oproj_tail(S):
    """Open question 3 (DESIGN_BC 7.2): the o_proj stream's span with the
    combine (and split) CTAs issuing late, against every CTA issuing at entry
    with the same core time (``otail_baseline``). The partials are checked
    against fp64 once a scenario (<= 1 bf16 ulp up to fp32 accumulation
    noise), and carry this launch's tag (a stale partial cannot pass)."""
    probe = _probe("otail", S)
    bufs = context.ProbeBuffers(rank=0)
    g = torch.Generator(device="cuda").manual_seed(S)
    sets = _weight_sets(H, spec.CORE, g)
    core = torch.randn(S, spec.CORE, generator=g, device="cuda").to(torch.bfloat16)
    pad = torch.zeros(MT, dtype=torch.bool, device="cuda")
    out = torch.zeros(BLOCKS * STAMP_WORDS, dtype=torch.int32, device="cuda")
    numbers = {}
    for name, sc in otail_scenarios(S).items():
        delay = dict(zip(OTAIL_DELAYS, sc))
        for i in range(len(sets)):  # warm-up
            probe(bufs, core=core, pad=pad, w_o=sets[i], out=out, **delay)
        torch.accelerator.synchronize()
        spans, late_issue = [], []
        for i in range(LAUNCHES):
            k = i % len(sets)
            bufs.epoch += 1
            out.zero_()
            probe(bufs, core=core, pad=pad, w_o=sets[k], out=out, **delay)
            torch.accelerator.synchronize()
            _assert_healthy(bufs)
            st = _stamps_of(out)
            assert _ctas_a_cu(st) == 1, "two CTAs shared a CU: not K2's placement"
            spans.append(_span(st))
            late_issue.append(int(st["issue"].max() - st["args"].min()))
            if i == 0:
                vals, tags = bufs.read_pairs("ar", spec.ar_pair(0, 0, 0), S * H // 2)
                assert (tags == bufs.tag()).all(), name
                got = vals.cpu().contiguous().view(torch.bfloat16).view(S, H)
                ref64 = core.double() @ sets[k].double().T
                ref = ref64.to(torch.bfloat16).cpu()
                u = _ulps(got, ref)
                absdot = (core.double().abs() @ sets[k].double().abs().T).cpu()
                noise = (got.double() - ref.double()).abs() <= 2**-20 * absdot
                assert ((u <= 1) | noise).all(), u.max()
        numbers[name] = {
            "span_us": _us(statistics.median(spans)),
            "last_issue_us": _us(statistics.median(late_issue)),
        }
    base = numbers["entry"]["span_us"]
    for name, v in numbers.items():
        v["tail_us"] = v["span_us"] - numbers[otail_baseline(name)]["span_us"]
        v["against"] = otail_baseline(name)
    _report(
        "otail",
        {
            "S": S,
            "o_proj_tb_per_s_at_entry": spec.CORE * H * 2 / (base * 1e-6) / 1e12,
            "scenarios": numbers,
        },
    )
