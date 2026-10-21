# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The probe harness (``mono.common.context``) that the stage probes and
K2 build on: the stage context, the LDS composition, the mailbox
hand-offs (scratch and aliased peers) and the bounded waits.

[CPU] Every probe compiles GPU-less for gfx950 (FlyDSL compile-only) with no
private segment (a spilling persistent kernel can lose residency), within the
VGPR and LDS budget; the routing sigmoid lowers to an IEEE divide.
[1GPU] The probes run: put / poll round trips in both scopes, the peer parity,
LDS stage structs and composites, the route table encoding, ``ld_pad``, and a
wait that gives up after its ``WAIT_BUDGET`` and fails the rank.

Phases B and C (DESIGN_BC 10.1, WP-B1): K1's context (``peers=False``,
``route_lds=False``, a ``K1Build`` key) and the Phase B / C scratch regions
in ``c["scr"]``; the stock-order reductions ``ops.seq8_sum`` (the fused
QK-norm insert's, 1.3) and ``ops.tl_sum64`` (Triton's gated norm, 1.4) and the
raw ``ops.hw_sqrt``. [CPU] their ISA (no fma in the serial chain nor in the
gate's butterfly, a raw ``v_sqrt_f32``); [1GPU] ``torch.equal`` with the stock kernels
on 1e5 rows and with an fp32 emulation of their order.
"""

import glob
import os
import time

import pytest
import regex as re
import torch

flydsl = pytest.importorskip("flydsl")
pytest.importorskip("aiter")

import flydsl.expr as fx  # noqa: E402
from aiter.ops.flydsl.kernels import buffer_ops as bo  # noqa: E402
from flydsl.expr import gpu, range_constexpr  # noqa: E402
from flydsl.expr import math as fmath  # noqa: E402
from flydsl.expr.typing import T  # noqa: E402

from tests.models.muse_spark.amd_specialized_decode import (  # noqa: E402
    reference,
)
from vllm.models.muse_spark.amd.specialized_decode.mono import spec  # noqa: E402
from vllm.models.muse_spark.amd.specialized_decode.mono.abi import (  # noqa: E402
    FAILED,
    PEER_HEAD,
    RECORD_WORDS,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common import (  # noqa: E402
    context,
    route_lds,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.mx import (  # noqa: E402
    FP4,
    e8m0_roundup,
    fp4_tile_load,
    mfma_scaled,
    pk_fp4_bf16,
    pk_fp4_f32,
    scale_byte,
    w13_scale_word,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (  # noqa: E402
    CM_DEV,
    bf16_round,
    bf_hi,
    bf_lo,
    block_excl_scan,
    hw_rsq,
    hw_sigmoid_approx,
    hw_sqrt,
    ieee_sigmoid,
    ld_dev,
    ld_i32,
    ld_pad,
    row_sum,
    rsrc,
    seq8_sum,
    tl_sum64,
    traced,
    wave_sum,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.plan import (  # noqa: E402
    Space,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ranks import (  # noqa: E402
    peer_at,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.sync import (  # noqa: E402
    Addr,
    publish,
    uniform_addr,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.stages import (  # noqa: E402
    gemv,
)

THREADS = spec.THREADS
OUT_WORDS = 4096
PEER_VAL = 7777  # this rank's own rlog push
OWN_SRC = 3  # the probe's rank


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


needs_gfx950 = pytest.mark.skipif(
    not _on_gfx950(), reason="the Muse mono probes run on gfx950 (MI355X)"
)
compiles = pytest.mark.skipif(
    not _compiles_for_gfx950(),
    reason="GPU-less compile needs FLYDSL_GPU_ARCH=gfx950 (or a gfx950 GPU)",
)


def _st(out, i, v):
    """out[i] := v (plain store; an fp32 by its bits)."""
    bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
    bo.buffer_store(bits, rsrc(out), i)


def _b(pred):
    return pred.select(fx.Int32(1), fx.Int32(0))


# ---------------------------------------------------------------- probe bodies


@fx.struct
class OneLds:
    v: fx.Array[fx.Int32, THREADS, 16]


@fx.struct
class ALds:
    x: fx.Array[fx.Int32, 64, 16]


@fx.struct
class BLds:
    y: fx.Array[fx.Float32, 64, 16]


LDS = {"one": OneLds, "two": {"a": ALds, "b": BLds}}


@traced
def _roundtrip(c):
    """One CTA: every hand-off kind once, results in ``out`` (see the test)."""
    tid, rank = c["tid"], c["rank"]
    out, pad, xin = c["args"]["out"], c["args"]["pad"], c["args"]["xin"]
    scr, peer = c["scr"], c["peer"]
    # scratch, device scope: 64 pairs, each thread reads its neighbour's
    if tid < 64:
        c["put"](scr["xrdy"], tid, tid * 3 + 1)
    if tid < 64:
        v = c["poll"]([(scr["xrdy"], (tid + 1) % 64, 1)])[0][0]
        _st(out, tid, v)
    # peers (all aliased to this rank's buffer): this rank's rlog row to every
    # rank; the other ranks' rows were written by the host
    if tid == 0:
        for p in range_constexpr(spec.TP):
            c["put"](peer["rlog"][p], rank * (spec.MAX_TOKENS * 32) + 5, PEER_VAL)
    if tid < spec.TP:
        v = c["poll"]([(peer["rlog_own"], tid * (spec.MAX_TOKENS * 32) + 5, 1)])[0][0]
        _st(out, 64 + tid, v)
    # put_bf (element index 2 * pair) and put_words (raw bits, a NaN kept)
    if tid == 0:
        c["put_bf"](peer["ar"][0], 2 * 9, [fx.Float32(1.5), fx.Float32(-2.25)])
        c["put_words"](scr["inter"], 3, [fx.Int32(0x7FC00001), fx.Int32(0x12345678)])
    if tid == 0:
        w = c["poll"]([(peer["ar_own"], 9, 1)])[0][0]
        _st(out, 72, w)
        ws = c["poll"]([(scr["inter"], 3, 2)])[0]
        _st(out, 74, ws[0])
        _st(out, 75, ws[1])
    # a flag over plain stores (publish), read back at device scope
    if tid < 4:
        _st(scr["xrow"], tid, tid + 40)
    publish(c["put"], scr["xrdy"], 63, 1, tid == 0)
    # LDS: a stage struct, then a composite's two members (the union's bytes)
    one, a, b = c["lds"]["one"], c["lds"]["a"], c["lds"]["b"]
    fx.ptr_store(tid * 7, one["v"] + tid)
    gpu.barrier()
    _st(out, 128 + tid, fx.ptr_load(one["v"] + (tid + 1) % THREADS))
    gpu.barrier()
    if tid < 64:
        fx.ptr_store(tid + 100, a["x"] + tid)
        fx.ptr_store(fx.Float32(tid) * fx.Float32(0.5), b["y"] + tid)
    gpu.barrier()
    if tid < 64:
        _st(out, 640 + tid, fx.ptr_load(a["x"] + (63 - tid)))
        _st(out, 704 + tid, fx.ptr_load(b["y"] + (63 - tid)))
        _st(out, 2048 + tid, fx.ptr_load(c["lds"]["two"]["x"] + tid))
    # the route table: ids and fp32 weight bits
    rt = c["rt"]
    if tid < 8:
        route_lds.store_id(rt, 1, tid, tid * 100 - 1)
        route_lds.store_weight(rt, 1, tid, fx.Float32(tid) * fx.Float32(0.125))
    gpu.barrier()
    if tid < 8:
        e = route_lds.load_id(rt, 1, 7 - tid)
        _st(out, 768 + tid, e)
        _st(out, 776 + tid, route_lds.load_weight(rt, 1, 7 - tid))
        _st(out, 784 + tid, _b(route_lds.live(e)))
        _st(out, 792 + tid, route_lds.clamp_expert(e))
        _st(out, 808 + tid, _b(ld_pad(pad, tid)))
    if tid == 0:
        _st(out, 800, c["e"])
        _st(out, 801, c["par"])
        _st(out, 802, c["slot"])
        _st(out, 803, rank)
        _st(out, 804, c["S"])
    # the two sigmoids
    x = fx.Float32(bo.buffer_load(rsrc(xin), tid, vec_width=1, dtype=T.f32))
    _st(out, 1024 + tid, ieee_sigmoid(x))
    _st(out, 1536 + tid, hw_sigmoid_approx(x))
    gpu.barrier()


@traced
def _grid(c):
    """Every CTA: a device-scope hand-off to its neighbour CTA, and a push
    through ``peer_at`` (a traced rank) polled on this rank's own region."""
    tid, bid = c["tid"], c["bid"]
    out = c["args"]["out"]
    if tid == 0:
        c["put"](c["scr"]["ass"], bid, bid * 5 + 3)
        c["put"](peer_at(c["peer"]["sqx"], bid % spec.TP), bid, bid + 1000)
    if tid == 0:
        v = c["poll"]([(c["scr"]["ass"], (bid + 1) % spec.BLOCKS, 1)])[0][0]
        w = c["poll"]([(c["peer"]["sqx_own"], (bid + 7) % spec.BLOCKS, 1)])[0][0]
        _st(out, bid, v)
        _st(out, spec.BLOCKS + bid, w)


@traced
def _timeout(c):
    """CTA 0 waits on a pair nobody writes: it must give up after the budget."""
    if c["tid"] == 0:
        v = c["poll"]([(c["scr"]["pss"], 5, 1)])[0][0]
        _st(c["args"]["out"], 0, v)
    if c["tid"] == 1:
        _st(c["args"]["out"], 1, 1)


GK = 1024  # the helper probe's GEMV K (o_proj's)
HS = 4  # the helper probe's rows
SC_COLS = 128  # w13 scale codes a row


def _gemv_lds(s):
    @fx.struct
    class GemvLds:
        xl: fx.Array[fx.Int32, s * (GK + gemv.LDS_PAD) // 2, 16]

    return GemvLds


@traced
def _helpers(c):
    """The shared helpers once each: a 16-row bf16 GEMV (``gemv`` + ``red``),
    wave / block reductions, the MX scale and quant helpers, one FP4 MFMA."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    a = c["args"]
    out, red = a["out"], c["red"]
    xl = c["lds"]["gemv"]["xl"]
    ops = gemv.loads(lane, wave, a["w"], 0, GK)
    gemv.rows_to_lds(tid, a["x"], GK, s, xl, GK + gemv.LDS_PAD)
    acc = gemv.mfmas(lane, wave, xl, GK + gemv.LDS_PAD, GK, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < s * 16:
        _st(out, tid, row_sum(red, tid % 16, tid // 16))
    gpu.barrier()
    # reductions: a wave's sum, the CTA's exclusive scan
    _st(out, 64 + tid, wave_sum(fx.Float32(lane + wave)))
    pre, tot = block_excl_scan((tid * 7) % 5, lane, wave, c["rt"]["scan"])
    _st(out, 576 + tid, pre)
    if tid == 0:
        _st(out, 1088, tot)
    # MX: RoundUp codes, scale bytes of a shuffled scale, packing, one MFMA
    amax = fx.Float32(bo.buffer_load(rsrc(a["amax"]), tid, vec_width=1, dtype=T.f32))
    _st(out, 1100 + tid, e8m0_roundup(amax))
    row = fx.Int32(bo.buffer_load(rsrc(a["rc"]), 2 * tid, vec_width=1, dtype=T.i32))
    col = fx.Int32(bo.buffer_load(rsrc(a["rc"]), 2 * tid + 1, vec_width=1, dtype=T.i32))
    _st(out, 1612 + tid, scale_byte(rsrc(a["w13s"]), row, col, SC_COLS))
    word = w13_scale_word(
        row // 1024, (row % 1024) // 16, col // 4, row % 16 + 16 * (col % 4)
    )
    w = fx.Int32(bo.buffer_load(rsrc(a["w13s"]), word, vec_width=1, dtype=T.i32))
    _st(out, 2124 + tid, w)
    pk = pk_fp4_f32(fx.Int32(0), amax, -amax, fx.Float32(1.0), 0)
    pk = pk_fp4_bf16(pk, fx.Int32(0x3F80BF80), fx.Float32(1.0), 1)
    _st(out, 2636 + tid, pk)
    t4 = fp4_tile_load(rsrc(a["w13s"]), 0, 0, 128, 0, lane * 4)
    m = mfma_scaled(t4, t4, fx.Vector.filled(4, 0.0, fx.Float32), 127, 127, FP4, FP4)
    if tid < 64:
        for i in range_constexpr(4):
            _st(out, 3148 + 4 * tid + i, m[i])
    gpu.barrier()


def _probes(S: int = 2):
    return {
        "roundtrip": context.probe_kernel(
            f"muse_probe_roundtrip_s{S}",
            S,
            _roundtrip,
            LDS,
            ("out", "pad", "xin"),
        ),
        "grid": context.probe_kernel(
            "muse_probe_grid", 1, _grid, None, ("out",), blocks=spec.BLOCKS
        ),
        "timeout": context.probe_kernel(
            "muse_probe_timeout", 1, _timeout, None, ("out",)
        ),
        "helpers": context.probe_kernel(
            "muse_probe_helpers",
            HS,
            _helpers,
            {"gemv": _gemv_lds(HS)},
            ("out", "w", "x", "amax", "rc", "w13s"),
        ),
    }


# ---------------------------------------------------------------- B / C probes

ROW_PROBE_ROWS = spec.BLOCKS * spec.WAVES  # a row of 64 a wave
LAY = spec.scratch_layout()
# the Phase B / C mailbox regions' last pair, the plain regions' last word
BC_LAST_PAIR = tuple(LAY[r][1] // 8 - 1 for r in spec.SCRATCH_REGIONS_BC)
BC_PLAIN = tuple(n for n, _, _ in spec.SCRATCH_PLAIN_BC)
BC_LAST_WORD = tuple(LAY[r][1] // 4 - 1 for r in BC_PLAIN)
K1_KEY = spec.K1Build(tokens=2, rope=False)
K1_FAULT_KEY = spec.K1Build(
    tokens=2,
    rope=True,
    fault_rank=1,
    fault_epoch=3,
    fault_slot=spec.k1_slot(2),
    fault_ticks=1000,
)


def _bf16_at(ptr, i):
    """bf16 element i of a buffer, as an f32."""
    w = ld_i32(ptr, i // 2)
    return fx.Float32((i % 2 == 1).select(bf_hi(w), bf_lo(w)))


@traced
def _k1_exchanges(c):
    """K1's context: a device-scope hand-off over ``fss``, a plain ``arow``
    row (CM_DEV) published by an ``ardy`` flag, every Phase B / C region's
    last pair and plain word through ``c["scr"]``, the context scalars."""
    tid, scr = c["tid"], c["scr"]
    out = c["args"]["out"]
    if tid < 64:
        c["put"](scr["fss"], tid, tid * 3 + 1)
    if tid < 64:
        v = c["poll"]([(uniform_addr(scr["fss"]), (tid + 1) % 64, 1)])[0][0]
        _st(out, tid, v)
    if tid < 64:
        bo.buffer_store(
            fx.Int32(tid + 40), rsrc(scr["arow"]), tid, cache_modifier=CM_DEV
        )
    publish(c["put"], scr["ardy"], 5, 1, tid == 0)
    if tid < 64:
        f = c["poll"]([(uniform_addr(scr["ardy"]), 5, 1)])[0][0]
        _st(out, 64 + tid, f + fx.Int32(ld_dev(scr["arow"], (tid + 1) % 64)))
    if tid == 0:
        for i in range_constexpr(len(BC_LAST_PAIR)):
            c["put"](scr[spec.SCRATCH_REGIONS_BC[i]], BC_LAST_PAIR[i], 1000 + i)
        for i in range_constexpr(len(BC_PLAIN)):
            bo.buffer_store(
                fx.Int32(2000 + i),
                rsrc(scr[BC_PLAIN[i]]),
                BC_LAST_WORD[i],
                cache_modifier=CM_DEV,
            )
        _st(out, 128, c["e"])
        _st(out, 129, c["par"])
        _st(out, 130, c["slot"])
        _st(out, 131, c["rank"])
        _st(out, 132, c["S"])
    # the stage union (no RouteLds before it); ``out`` re-read from lane 0:
    # its buffer resource would otherwise merge with the tid == 0 stores' at
    # the join and the store become a readfirstlane waterfall
    one = c["lds"]["one"]
    fx.ptr_store(tid * 7, one["v"] + tid)
    gpu.barrier()
    out = uniform_addr(Addr("out", Space.SCRATCH, out)).value
    _st(out, 256 + tid, fx.ptr_load(one["v"] + (tid + 1) % THREADS))
    gpu.barrier()


def _k1_context(c):
    """``_k1_exchanges`` in a context with no peer regions and no route table
    (checked while tracing)."""
    if c["peer"] or c["rt"]:
        raise AssertionError(f"K1 context: peer {list(c['peer'])} rt {list(c['rt'])}")
    if c["key"] is None or not isinstance(c["key"], spec.K1Build):
        raise AssertionError(f"K1 context key {c['key']!r}")
    _k1_exchanges(c)


@traced
def _sqrt(c):
    """``hw_sqrt`` of word tid."""
    a = c["args"]
    x = fx.Float32(bo.buffer_load(rsrc(a["x"]), c["tid"], vec_width=1, dtype=T.f32))
    _st(a["y"], c["tid"], hw_sqrt(x))


def _row(c):
    """(row, lane, x): wave w of CTA b takes row 8 b + w of ``x`` (bf16 [2048,
    64]), lane d its dim d."""
    r = c["bid"] * spec.WAVES + c["wave"]
    return r, c["lane"], _bf16_at(c["args"]["x"], r * 64 + c["lane"])


@traced
def _seq8(c):
    """``seq8_sum`` of each row's rounded squares -> ``tot`` word r."""
    r, lane, x = _row(c)
    tot = seq8_sum(x * x, lane)
    if lane == 0:
        _st(c["args"]["tot"], r, tot)


@traced
def _qknorm(c):
    """The fused insert's QK-norm (DESIGN_BC 1.3 steps 1-5, 7; NoPE) of each
    row: ``tot`` word r and y = bf16(RN(x rsq(fma(tot, 2^-6, eps)))) as f32."""
    a = c["args"]
    r, lane, x = _row(c)
    tot = seq8_sum(x * x, lane)
    q = fx.Float32(fmath.fma(tot, fx.Float32(2.0**-6), fx.Float32(spec.RMS_EPS)))
    _st(a["y"], r * 64 + lane, bf16_round(x * hw_rsq(q)))
    if lane == 0:
        _st(a["tot"], r, tot)


@traced
def _tl64(c):
    """``tl_sum64`` of each row -> ``tot`` word r."""
    r, lane, x = _row(c)
    tot = tl_sum64(x, lane)
    if lane == 0:
        _st(c["args"]["tot"], r, tot)


@traced
def _gate(c):
    """FLA's gated norm (DESIGN_BC 1.4) of each row with gate row ``g``:
    ``tot`` word r and y = bf16(RN(RN(x rstd) sigmoid(g))) as f32, rstd = 1 /
    sqrt(fma(tot, 2^-6, eps)) with the raw sqrt and an IEEE divide."""
    a = c["args"]
    r, lane, x = _row(c)
    g = _bf16_at(a["g"], r * 64 + lane)
    tot = tl_sum64(x, lane)
    q = fx.Float32(fmath.fma(tot, fx.Float32(2.0**-6), fx.Float32(spec.RMS_EPS)))
    rstd = fx.Float32(1.0) / hw_sqrt(q)
    _st(a["y"], r * 64 + lane, bf16_round((x * rstd) * ieee_sigmoid(g)))
    if lane == 0:
        _st(a["tot"], r, tot)


def _b1_probes():
    rows = dict(blocks=spec.BLOCKS)
    k1 = dict(peers=False, route_lds=False)
    return {
        "k1ctx": context.probe_kernel(
            "muse_probe_k1ctx", 2, _k1_context, {"one": OneLds}, ("out",),
            build=K1_KEY, **k1,
        ),
        "k1ctx_fault": context.probe_kernel(
            "muse_probe_k1ctx_fault", 2, _k1_context, {"one": OneLds}, ("out",),
            build=K1_FAULT_KEY, **k1,
        ),
        "sqrt": context.probe_kernel("muse_probe_sqrt", 1, _sqrt, None, ("x", "y")),
        "seq8": context.probe_kernel(
            "muse_probe_seq8", 1, _seq8, None, ("x", "tot"), **rows
        ),
        "qknorm": context.probe_kernel(
            "muse_probe_qknorm", 1, _qknorm, None, ("x", "y", "tot"), **rows
        ),
        "tl64": context.probe_kernel(
            "muse_probe_tl64", 1, _tl64, None, ("x", "tot"), **rows
        ),
        "gate": context.probe_kernel(
            "muse_probe_gate", 1, _gate, None, ("x", "g", "y", "tot"), **rows
        ),
    }  # fmt: skip


# ---------------------------------------------------------------- [CPU]


def test_context_lds_layout():
    """The fixed red (8 KiB) + RouteLds (3904 B) + the union of the stages' structs."""
    assert context.FixedLds.__dsl_size_of__() == 8192
    assert route_lds.RouteLds.__dsl_size_of__() == 3904
    assert sum(spec.ROUTE_LDS.values()) * 4 == 3904
    assert context.lds_total_bytes() == 8192 + 3904
    sizes = context.stage_bytes(LDS)
    assert sizes == {"one": THREADS * 4, "two": 64 * 4 + 64 * 4}
    assert context.lds_total_bytes(LDS) == 8192 + 3904 + THREADS * 4
    with pytest.raises(TypeError):
        context.stage_union({"bad": int})
    with pytest.raises(ValueError):
        context.stage_union({"empty": {}})


def test_probe_argument_checks():
    with pytest.raises(ValueError, match="fixed probe arguments"):
        context.probe_kernel("p_bad0", 1, _timeout, None, ("scratch",))
    with pytest.raises(ValueError, match="repeat"):
        context.probe_kernel("p_bad1", 1, _timeout, None, ("out", "out"))
    with pytest.raises(ValueError, match="grid"):
        context.probe_kernel("p_bad2", 1, _timeout, None, ("out",), blocks=257)
    with pytest.raises(ValueError, match="rows"):
        context.probe_kernel("p_bad3", 9, _timeout, None, ("out",))

    @fx.struct
    class Huge:
        z: fx.Array[fx.Int32, 40 * 1024, 16]

    with pytest.raises(ValueError, match="LDS"):
        context.probe_kernel("p_bad4", 1, _timeout, {"huge": Huge}, ("out",))
    # one build a key; another S is another build
    p = _probes()
    assert _probes()["roundtrip"] is p["roundtrip"]
    assert _probes(3)["roundtrip"] is not p["roundtrip"]
    assert context.PROBE_ABI.names == context.PROBE_ARGS


def _closure_body(v):
    @traced
    def body(c):
        if c["tid"] == 0:
            _st(c["args"]["out"], 0, v)

    return body


def test_probe_key_covers_captured_values():
    """Bodies of one factory that differ only in a captured value are two
    builds (one binary otherwise: the second probe would run the first's
    constant); equal captures share one."""
    a = context.probe_kernel("p_closure", 1, _closure_body(1), None, ("out",))
    b = context.probe_kernel("p_closure", 1, _closure_body(2), None, ("out",))
    a2 = context.probe_kernel("p_closure", 1, _closure_body(1), None, ("out",))
    assert a is not b and a is a2
    # module-level scalars a body reads are keyed too
    d0 = context._body_digest(_timeout)
    assert d0 == context._body_digest(_timeout)
    assert context._body_digest(_roundtrip) != context._body_digest(_grid)
    g = globals()
    old = g["PEER_VAL"]
    d = context._body_digest(_roundtrip)
    try:
        g["PEER_VAL"] = old + 1
        assert context._body_digest(_roundtrip) != d
    finally:
        g["PEER_VAL"] = old
    assert context._body_digest(_roundtrip) == d


def test_plan_helpers():
    """K3's plan plus DSv4.1's arithmetic / shards and #61482's nested key."""
    import dataclasses

    from vllm.models.muse_spark.amd.specialized_decode.mono import abi
    from vllm.models.muse_spark.amd.specialized_decode.mono.common import plan

    @dataclasses.dataclass(frozen=True)
    class Inner:
        a: int
        b: bool = False

    @dataclasses.dataclass(frozen=True)
    class Outer:
        tokens: int
        inner: Inner

    @dataclasses.dataclass(frozen=True)
    class Bad:
        rows: list

    assert plan.key_tuple(Outer(2, Inner(3)), "src") == (
        ("tokens", 2),
        ("inner.a", 3),
        ("inner.b", False),
        ("sources", "src"),
    )
    with pytest.raises(TypeError, match="scalar"):
        plan.key_tuple(Bad([1]), "src")
    key = spec.K2Build(tokens=4, debug_dump=True)
    assert plan.key_tuple(key, "x")[:2] == (("tokens", 4), ("debug_dump", True))
    assert plan.sel(True, 1, 2) == 1 and plan.imin(3, 2) == 2
    assert plan.cdiv(17, 8) == 3 and plan.padded(17, 8) == 24
    with pytest.raises(AssertionError):
        plan.i32(2**31)
    assert plan.Shard(8).split(4096, "pooled width") == 512
    with pytest.raises(plan.ShardError):
        plan.Shard(8).split(100, "rows")
    with pytest.raises(plan.ShardError):
        plan.Shard(0)
    assert plan.HEALTH_WORDS == abi.HEALTH_WORDS == 2056
    assert (plan.BLOCKS, plan.THREADS, plan.WAVES) == (256, 512, 8)
    assert len(plan.source_digest(".")) == 16 and context.SOURCES


def test_tags_and_debug_on_host():
    """``sync.step_tag`` is ``spec.step_tag`` (also on a wrapped epoch); the
    wait records and pair reads decode on the host."""
    from vllm.models.muse_spark.amd.specialized_decode.mono.common import debug, sync

    for e in (0, 1, 5, (1 << 22) - 1, 1 << 22, 2**31 - 1, -1, -(2**31)):
        for slot in (0, 128, 189, 255):
            assert sync.step_tag(e, slot) == spec.step_tag(e, slot) > 0
    assert sync.TICKS_PER_SECOND * 10 == sync.WAIT_TICKS
    health = torch.zeros(context.HEALTH_WORDS, dtype=torch.int32)
    rid = debug.region_id("rsx")
    health[(5 + 1) * RECORD_WORDS : (6 + 1) * RECORD_WORDS] = torch.tensor(
        [1, rid, 77, 9000, 8744, 5, 300, 0]
    )
    health[(2 + 1) * RECORD_WORDS : (3 + 1) * RECORD_WORDS] = torch.tensor(
        [1, debug.region_id("moe"), 3, 9000, 0, 2, 100, 0]
    )
    waits = debug.given_up_waits(health, spec.REGIONS)
    assert waits == [
        "+0 us CTA 2: moe pair 3 waited for tag 9000, saw 0",
        "+2 us CTA 5: rsx pair 77 waited for tag 9000, saw 8744",
    ]
    buf = torch.zeros(64, dtype=torch.uint8)
    buf[8:24].view(torch.int32)[:] = torch.tensor([11, 22, 33, 44], dtype=torch.int32)
    vals, tags = debug.read_pairs(buf, 8, 2)
    assert vals.tolist() == [11, 33] and tags.tolist() == [22, 44]
    with pytest.raises(ValueError):
        debug.read_pairs(buf, 60, 1)
    with pytest.raises(TypeError):
        debug.read_pairs(buf.view(torch.int32), 0, 1)


@pytest.mark.parametrize("epoch", [6, 7])
def test_probe_buffers_on_host(epoch):
    """``ProbeBuffers`` (on CPU memory here): the parity half, tags, host
    writes of tagged pairs, the argument table."""
    bufs = context.ProbeBuffers("cpu", epoch=epoch, budget=3, rank=5, slot=130)
    assert bufs.health[1].item() == 3  # WAIT_BUDGET
    assert bufs.tag() == spec.step_tag(epoch, 130)
    assert bufs.alarm == bufs.peer.data_ptr() + 20
    assert (bufs.peers == bufs.peer.data_ptr() + PEER_HEAD).all()
    buf, off = bufs.where("moe")
    half = spec.peer_half_bytes()
    assert buf is bufs.peer
    assert off == PEER_HEAD + (epoch & 1) * half + spec.peer_layout()["moe"][0]
    assert bufs.where("route") == (bufs.scratch, spec.scratch_layout()["route"][0])
    bufs.write_pairs("rsx", 4, torch.tensor([1.0, -2.0]))
    bufs.write_pairs("moe", 0, torch.tensor([1.0, 2.0], dtype=torch.bfloat16), tag=9)
    vals, tags = bufs.read_pairs("rsx", 4, 2)
    assert vals.view(torch.float32).tolist() == [1.0, -2.0]
    assert tags.tolist() == [bufs.tag()] * 2
    vals, tags = bufs.read_pairs("moe", 0, 1)
    assert tags.item() == 9
    words = torch.tensor(vals.tolist(), dtype=torch.int32)
    assert words.view(torch.bfloat16).tolist() == [1.0, 2.0]
    # pair ranges are checked against the region (the next region's pairs
    # are never touched)
    n = spec.peer_layout()["moe"][1] // 8
    bufs.read_pairs("moe", n - 1, 1)
    with pytest.raises(IndexError):
        bufs.read_pairs("moe", n - 1, 2)
    with pytest.raises(IndexError):
        bufs.write_pairs("xrdy", spec.NA, torch.tensor([1], dtype=torch.int32))
    with pytest.raises(IndexError):
        bufs.write_pairs("rsx", -1, torch.tensor([1], dtype=torch.int32))
    w = torch.zeros(3, dtype=torch.int32)
    argv = bufs.argv(("w", "n"), {"w": w, "n": 17})
    assert argv.tolist() == [w.data_ptr(), 17]
    assert not bufs.failed() and bufs.given_up_waits() == []


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


@compiles
@pytest.mark.parametrize("which", ["roundtrip", "grid", "timeout", "helpers"])
def test_probe_compiles(which, tmp_path, monkeypatch):
    """The GPU-less compile gate (README, Residency) on each harness probe."""
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    probe = _probes()[which]
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
    if which == "roundtrip":
        # static LDS: a symbol a struct field (unused ones dropped), one a
        # union; here the union (2048 B) and RouteLds.route (512 B)
        assert res["group_segment_fixed_size"] == THREADS * 4 + 512, res
        assert res["group_segment_fixed_size"] <= context.lds_total_bytes(LDS)
        # ieee_sigmoid: the IEEE divide's fixup; hw_sigmoid_approx: v_rcp
        assert "v_div_fixup_f32" in txt
        assert "v_rcp_f32" in txt
        assert "v_exp_f32" in txt
    if which == "helpers":
        assert "v_mfma_f32_16x16x32_bf16" in txt
        assert "v_mfma_scale_f32_16x16x128_f8f6f4" in txt
        assert "v_cvt_scalef32_pk_fp4_f32" in txt
        assert "v_cvt_scalef32_pk_fp4_bf16" in txt


def test_context_without_peers_or_route_lds():
    """K1's context (DESIGN_BC 5.2): no ``peers`` argument and no route table
    in LDS; K2's context unchanged."""
    assert set(context.CONTEXT_ARGS) <= set(spec.K2_ARGS)
    assert set(context.LOCAL_CONTEXT_ARGS) <= set(spec.K1_ARGS)
    assert "peers" not in spec.K1_ARGS
    local = tuple(n for n in context.CONTEXT_ARGS if n != "peers")
    assert local == context.LOCAL_CONTEXT_ARGS
    assert context.lds_total_bytes(route_lds=False) == 8192
    assert context.lds_total_bytes(LDS, route_lds=False) == 8192 + THREADS * 4
    assert context.lds_total_bytes(LDS) == 8192 + 3904 + THREADS * 4
    # the argument checks run before anything is traced
    args = {n: 0 for n in context.LOCAL_CONTEXT_ARGS}
    with pytest.raises(KeyError, match="peers"):
        context.make_context(1, args)
    with pytest.raises(KeyError, match="rank"):
        context.make_context(1, {n: 0 for n in args if n != "rank"}, peers=False)
    with pytest.raises(ValueError, match="peers_alias"):
        context.make_context(1, args, peers=False, peers_alias=True)

    # LDS that fits only without RouteLds
    @fx.struct
    class Big:
        z: fx.Array[fx.Int32, (spec.LDS_BYTES - 8192) // 4, 16]

    assert context.lds_total_bytes({"big": Big}, route_lds=False) == spec.LDS_BYTES
    with pytest.raises(ValueError, match="LDS"):
        context.make_context(1, args, {"big": Big}, peers=False)
    with pytest.raises(ValueError, match="LDS"):
        context.probe_kernel("p_big", 1, _timeout, {"big": Big}, ("out",))
    context.probe_kernel(
        "p_big", 1, _timeout, {"big": Big}, ("out",), peers=False, route_lds=False
    )


def test_probe_key_covers_the_context():
    """``peers``, ``route_lds`` and the build key are part of a probe's key;
    without peers ``peers_alias`` does not apply."""
    p = _b1_probes()
    assert _b1_probes()["k1ctx"] is p["k1ctx"]
    a = context.probe_kernel("p_ctx", 1, _timeout, None, ("out",))
    b = context.probe_kernel("p_ctx", 1, _timeout, None, ("out",), peers=False)
    c = context.probe_kernel("p_ctx", 1, _timeout, None, ("out",), route_lds=False)
    d = context.probe_kernel("p_ctx", 1, _timeout, None, ("out",), build=K1_KEY)
    assert len({id(x) for x in (a, b, c, d)}) == 4
    e = context.probe_kernel(
        "p_ctx", 1, _timeout, None, ("out",), peers=False, peers_alias=False
    )
    assert e is b


# readfirstlane waterfall around a load whose base is not wave-uniform
_WATERFALL = re.compile(r"v_cmp_eq_u64_e\d+\s+\S+,\s*s\[\d+:\d+\],\s*v\[")


def _fma_ops(txt: str) -> list[str]:
    """The fused multiply-adds of a final ISA (every encoding)."""
    return re.findall(r"^\s*(v_(?:pk_)?fma\w*|v_fmac\w*|v_mad\w*|v_mac\w*)", txt, re.M)


@compiles
@pytest.mark.parametrize(
    "which", ["k1ctx", "k1ctx_fault", "sqrt", "seq8", "qknorm", "tl64", "gate"]
)
def test_b1_probe_compiles(which, tmp_path, monkeypatch):
    """The compile gate on the B1 probes, and the ISA the stock orders need:
    ``seq8_sum``'s serial chain and tree contract nothing into an fma (the
    .cu's adds are separate, 1.3 step 2); ``tl_sum64`` has no fma (the
    served leaf is a multiply and an add) and the xor 1, 32, 16, 8, 4, 2
    butterfly (permlane swaps and DPP); ``hw_sqrt`` is a bare
    ``v_sqrt_f32`` (no 2^32 range scaling, no correction)."""
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    probe = _b1_probes()[which]
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
    loops = _WATERFALL.findall(txt)
    assert not loops, f"{len(loops)} waterfall compares, e.g. {loops[0]}"
    fmas = _fma_ops(txt)
    shr = {k: len(re.findall(rf"row_shr:{k}\b", txt)) for k in (1, 2, 4, 8)}
    if which.startswith("k1ctx"):
        # the stage union only: no RouteLds field, no fixed red (unused)
        assert res["group_segment_fixed_size"] == THREADS * 4, res
        # the fault prologue (sync.stall's s_sleep 127) only in the fault build
        assert ("s_sleep 0x7f" in txt or "s_sleep 127" in txt) == (
            which == "k1ctx_fault"
        )
    if which == "sqrt":
        assert txt.count("v_sqrt_f32") == 1, txt.count("v_sqrt_f32")
        assert "0x4f800000" not in txt and "0xf800000" not in txt
        assert "v_rsq_f32" not in txt and not fmas, fmas
    if which in ("seq8", "qknorm"):
        assert shr[1] >= 7, shr
        assert len(re.findall(r"^\s*v_add_f32", txt, re.M)) >= 10
    if which == "seq8":
        assert not fmas, fmas
        assert len(re.findall(r"^\s*v_mul_f32", txt, re.M)) >= 1
    if which == "qknorm":
        # the one fma is fma(tot, 2^-6, eps) (v_fmamk / v_fmaak: the
        # constants inline); the rsq is the hardware one
        assert len(fmas) == 1 and "v_rsq_f32" in txt, fmas
    if which in ("tl64", "gate"):
        # the served gate's butterfly: lanes ^ 1 (the leaf), 32, 16, 8, 4, 2
        assert "v_permlane32_swap" in txt and "v_permlane16_swap" in txt
        assert shr[8] >= 1 and shr[4] >= 1, shr
    if which == "tl64":
        # the leaf is RN(x^2) + RN(x'^2): no fma (stock: v_pk_mul + v_add)
        assert not fmas, fmas
    if which == "gate":
        # a_j and the sqrt argument, plus the IEEE divides' fma sequences
        assert "v_sqrt_f32" in txt and "v_div_fixup_f32" in txt
        assert "v_rsq_f32" not in txt and "0x4f800000" not in txt


# ---------------------------------------------------------------- [1GPU]


@needs_gfx950
@pytest.mark.parametrize("epoch", [1, 2, (1 << 22) + 3, -5])
def test_roundtrip_probe(epoch):
    probe = _probes()["roundtrip"]
    bufs = context.ProbeBuffers(epoch=epoch, rank=OWN_SRC, slot=spec.k2_slot(7))
    tag = bufs.tag()
    assert tag == spec.step_tag(epoch, 135) and tag > 0
    # the other ranks' rlog pushes, in this epoch's parity half
    for src in range(spec.TP):
        if src != OWN_SRC:
            bufs.write_pairs(
                "rlog",
                src * spec.MAX_TOKENS * 32 + 5,
                torch.tensor([1000 + src], dtype=torch.int32),
            )
    pad = torch.tensor([0, 1, 0, 0, 1, 1, 0, 1], dtype=torch.bool, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(0)
    xin = torch.randn(THREADS, generator=g, device="cuda") * 8
    xin[:4] = torch.tensor([0.0, -0.0, 88.0, -88.0])
    out = torch.zeros(OUT_WORDS, dtype=torch.int32, device="cuda")
    probe(bufs, out=out, pad=pad, xin=xin)
    torch.accelerator.synchronize()
    o = out.cpu()
    assert not bufs.failed(), bufs.given_up_waits()
    t = torch.arange(64)
    assert torch.equal(o[:64], (t + 1) % 64 * 3 + 1)
    want = torch.tensor([PEER_VAL if s == OWN_SRC else 1000 + s for s in range(8)])
    assert torch.equal(o[64:72], want)
    bf = torch.tensor([1.5, -2.25], dtype=torch.bfloat16).view(torch.int32)
    assert o[72].item() == bf.item()
    assert o[74].item() == 0x7FC00001 and o[75].item() == 0x12345678
    # the pairs carry this launch's tag, in the parity half of this epoch
    vals, tags = bufs.read_pairs("rlog", OWN_SRC * spec.MAX_TOKENS * 32 + 5, 1)
    assert vals.item() == PEER_VAL and tags.item() == tag
    par = (epoch & spec.EPOCH_MASK) & 1
    other = PEER_HEAD + (1 - par) * spec.peer_half_bytes()
    assert not bufs.peer[other : other + spec.peer_half_bytes()].any()
    vals, tags = bufs.read_pairs("xrdy", 63, 1)
    assert vals.item() == 1 and tags.item() == tag
    xrow = bufs.where("xrow")[1]
    assert bufs.scratch[xrow : xrow + 16].view(torch.int32).tolist() == [40, 41, 42, 43]
    # LDS
    tt = torch.arange(THREADS)
    assert torch.equal(o[128 : 128 + THREADS], (tt + 1) % THREADS * 7)
    assert torch.equal(o[640:704], 100 + 63 - t)
    assert torch.equal(o[704:768].view(torch.float32), (63 - t).float() * 0.5)
    assert torch.equal(o[2048:2112], 100 + t)  # the composite's flat view
    # route table
    k = 7 - torch.arange(8)
    ids = k * 100 - 1
    assert torch.equal(o[768:776], ids)
    assert torch.equal(o[776:784].view(torch.float32), k.float() * 0.125)
    assert torch.equal(o[784:792], ((ids >= 0) & (ids < 256)).int())
    assert torch.equal(o[792:800], ids.clamp(0, 255))
    assert torch.equal(o[808:816], pad.cpu().int())
    e = epoch & spec.EPOCH_MASK
    assert o[800:805].tolist() == [e, e & 1, spec.k2_slot(7), OWN_SRC, 2]
    # sigmoids, against the exact sigmoid of the kernel's own f32 exponent
    # x * f32(-LOG2E) (its rounding moves the result by up to |x| ulps, which
    # neither sigmoid can undo): v_exp_f32 1 ulp, the add and the IEEE divide
    # half an ulp each (v_rcp 1 ulp); results below the normal range
    # (x = -88) may flush in the approximate one
    f32 = torch.finfo(torch.float32)
    arg = xin.cpu() * torch.tensor(-spec.LOG2E, dtype=torch.float32)
    ref = 1.0 / (1.0 + torch.exp2(arg.double()))
    ieee = o[1024:1536].view(torch.float32).double()
    approx = o[1536:2048].view(torch.float32).double()
    assert ((ieee - ref).abs() <= 3 * f32.eps * ref + f32.tiny).all()
    assert ((approx - ref).abs() <= 4 * f32.eps * ref + f32.tiny).all()
    assert ieee[0].item() == 0.5 and ieee[1].item() == 0.5 and ieee[2].item() == 1.0


@needs_gfx950
def test_grid_probe():
    """Every CTA resident (a spin on a neighbour CTA would hang otherwise)."""
    probe = _probes()["grid"]
    bufs = context.ProbeBuffers(epoch=11)
    out = torch.zeros(OUT_WORDS, dtype=torch.int32, device="cuda")
    for step in range(3):  # the epoch moves: last step's pairs are stale
        out.zero_()
        probe(bufs, out=out)
        torch.accelerator.synchronize()
        assert not bufs.failed(), bufs.given_up_waits()
        b = torch.arange(spec.BLOCKS)
        assert torch.equal(out[: spec.BLOCKS].cpu(), (b + 1) % spec.BLOCKS * 5 + 3)
        assert torch.equal(
            out[spec.BLOCKS : 2 * spec.BLOCKS].cpu(), (b + 7) % 256 + 1000
        )
        bufs.epoch += 1


@needs_gfx950
def test_timeout_probe():
    """A wait gives up after the WAIT_BUDGET seconds, records itself, raises
    this rank's alarm word and fails the rank; a failed rank's waits exit at
    once (sticky FAILED)."""
    probe = _probes()["timeout"]
    out = torch.zeros(4, dtype=torch.int32, device="cuda")
    # a warm-up whose pair is there (the JIT compile stays out of the timing;
    # the wait's satisfied path): no record, no alarm
    warm = context.ProbeBuffers(budget=1, rank=2)
    warm.write_pairs("pss", 5, torch.tensor([4242], dtype=torch.int32))
    probe(warm, out=out)
    torch.accelerator.synchronize()
    assert not warm.failed() and warm.given_up_waits() == []
    assert out[0].item() == 4242 and warm.flags().sum().item() == 0
    bufs = context.ProbeBuffers(budget=1, rank=2)
    out.zero_()
    t0 = time.monotonic()
    probe(bufs, out=out)
    torch.accelerator.synchronize()
    took = time.monotonic() - t0
    assert 0.9 < took < 5, took
    assert bufs.failed()
    assert bufs.flags()[2].item() == 1 and bufs.flags()[:2].sum().item() == 0
    rec = bufs.health[RECORD_WORDS : 2 * RECORD_WORDS].cpu().tolist()
    assert rec[0] == 1 and rec[2] == 5 and rec[3] == bufs.tag() and rec[5] == 0
    waits = bufs.given_up_waits()
    assert len(waits) == 1 and "pss pair 5" in waits[0], waits
    assert out[1].item() == 1
    t0 = time.monotonic()
    probe(bufs, out=out)
    torch.accelerator.synchronize()
    assert time.monotonic() - t0 < 0.5
    assert bufs.health[FAILED].item() == 1


def _code_roundup(amax: torch.Tensor) -> torch.Tensor:
    """The aiter HIP RoundUp without the floor: ceil code of amax *
    f32(1/6), at most 254 (``e8m0_roundup``'s clamp)."""
    u = (amax.float() * torch.tensor(1.0 / 6.0)).view(torch.int32)
    e = (u >> 23) & 0xFF
    e = e + (((u & 0x7FFFFF) != 0) & (e < 0xFF)).int()
    return e.clamp(max=254)


@needs_gfx950
def test_helpers_probe():
    from aiter.utility.fp4_utils import e8m0_shuffle

    probe = _probes()["helpers"]
    bufs = context.ProbeBuffers()
    g = torch.Generator(device="cuda").manual_seed(1)
    w = torch.randn(16, GK, generator=g, device="cuda").to(torch.bfloat16)
    x = torch.randn(HS, GK, generator=g, device="cuda").to(torch.bfloat16)
    amax = (torch.rand(THREADS, generator=g, device="cuda") + 0.01) * torch.exp2(
        torch.randint(-30, 30, (THREADS,), generator=g, device="cuda")
    )
    amax[:3] = torch.tensor([6.0, 0.75, 6.0 * 2.0**-100])
    codes = torch.randint(0, 256, (2 * 1024, SC_COLS), dtype=torch.uint8)
    src = (
        codes.view(torch.float8_e8m0fnu) if hasattr(torch, "float8_e8m0fnu") else codes
    )
    w13s = e8m0_shuffle(src).view(torch.uint8).reshape(-1).cuda()
    rows = torch.randint(0, 2 * 1024, (THREADS,), generator=g, device="cuda")
    cols = torch.randint(0, SC_COLS, (THREADS,), generator=g, device="cuda")
    rc = torch.stack([rows, cols], 1).to(torch.int32)
    out = torch.zeros(4096, dtype=torch.int32, device="cuda")
    probe(bufs, out=out, w=w, x=x, amax=amax, rc=rc, w13s=w13s)
    torch.accelerator.synchronize()
    assert not bufs.failed()
    o = out.cpu()
    ref = (x.double() @ w.double().T).cpu()  # [HS, 16], exact products
    got = o[: HS * 16].view(torch.float32).view(HS, 16).double()
    # fp32 accumulation of 1024 terms (|partial sums| ~ 32); a layout bug is O(1)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-3)
    tt = torch.arange(THREADS)
    wave = tt // 64
    assert torch.equal(o[64:576].view(torch.float32), (2016 + 64 * wave).float())
    v = (tt * 7) % 5
    assert torch.equal(o[576:1088], torch.cumsum(v, 0) - v)
    assert o[1088].item() == v.sum().item()
    assert torch.equal(o[1100:1612], _code_roundup(amax.cpu()))
    assert o[1100:1103].tolist() == [127, 124, 27]
    r, cc = rows.cpu(), cols.cpu()
    assert torch.equal(o[1612:2124], codes[r, cc].int())
    # the w13 scale word holds (row, col) at byte (st % 2) 2 + rg % 2
    st, rg = cc // 4, (r % 1024) // 16
    byte = (o[2124:2636] >> ((st % 2) * 16 + (rg % 2) * 8)) & 0xFF
    assert torch.equal(byte, codes[r, cc].int())


# ---------------------------------------------------------------- B / C stock orders

N_ROWS = 100_000  # DESIGN_BC 10.1: the B1 helpers on 1e5 rows


def _f32(v: float) -> torch.Tensor:
    return torch.tensor(v, dtype=torch.float32)


def _tl_sum64_ref(x: torch.Tensor) -> torch.Tensor:
    """``tl_sum64``'s order in fp32 (correctly rounded on the CPU): the
    served gfx950 ``rms_norm_gated`` tree, ``reference.gated_sumsq`` (leaf
    RN(x_2c^2) + RN(x_2c+1^2), then halving 16, 8, 4, 2, 1)."""
    return reference.gated_sumsq(x)


def _seq8_sum_ref(x: torch.Tensor) -> torch.Tensor:
    """``seq8_sum``'s order in fp32: per 8-dim chunk the serial sum of the
    rounded squares, the chunks as ((s0 + s4) + (s2 + s6)) + ((s1 + s5) +
    (s3 + s7))."""
    sq = x.float().reshape(-1, 8, 8) ** 2
    s = sq[..., 0]
    for k in range(1, 8):
        s = s + sq[..., k]
    return ((s[:, 0] + s[:, 4]) + (s[:, 2] + s[:, 6])) + (
        (s[:, 1] + s[:, 5]) + (s[:, 3] + s[:, 7])
    )


def _rows(n: int, seed: int) -> torch.Tensor:
    """N bf16 rows of 64 (CPU): normal rows over five decades of scale, and
    rows that make the edges bite: eps-scale (|x| ~ 3e-3: var ~ eps), all
    zero, one huge element, alternating signs, exactly representable sums."""
    g = torch.Generator().manual_seed(seed)
    scale = torch.exp2(torch.randint(-8, 9, (n, 1), generator=g).float())
    x = torch.randn(n, 64, generator=g) * scale
    x[0::97] *= 3e-3 / scale[0::97]
    x[1::997] = 0
    x[2::991, 5] = 3e4
    x[3::983] = torch.tensor([1.0, -1.0] * 32)
    return x.to(torch.bfloat16)


def _run_rows(probe, bufs, n: int, **arrays):
    """Launch a row probe over n rows, 2048 a launch: every array is [n_pad,
    ...] and is sliced per launch (the probe's row r is the slice's)."""
    for r0 in range(0, n, ROW_PROBE_ROWS):
        probe(bufs, **{k: v[r0 : r0 + ROW_PROBE_ROWS] for k, v in arrays.items()})
    torch.accelerator.synchronize()
    assert not bufs.failed(), bufs.given_up_waits()


def _padded(x: torch.Tensor) -> torch.Tensor:
    n = -(-x.shape[0] // ROW_PROBE_ROWS) * ROW_PROBE_ROWS
    out = torch.zeros((n, *x.shape[1:]), dtype=x.dtype)
    out[: x.shape[0]] = x
    return out.cuda()


def test_stock_order_emulations_match_the_references():
    """[CPU] The probes' host emulations of the two stock orders are B0's
    reviewed references, bit for bit: ``_seq8_sum_ref`` -> the NoPE QK-norm
    of ``reference.qk_norm_rope``, ``_tl_sum64_ref`` -> ``reference.gated_norm``."""
    x = _rows(8192, 23)
    og = (torch.randn(8192, 64, generator=torch.Generator().manual_seed(24)) * 30).to(
        torch.bfloat16
    )
    eps, m = _f32(spec.RMS_EPS), _f32(2.0**-6)
    r = torch.rsqrt(_seq8_sum_ref(x) * m + eps)
    norm = (x.float() * r[:, None]).to(torch.bfloat16)
    assert torch.equal(norm, reference.qk_norm_rope(x, None))
    rstd = _f32(1.0) / torch.sqrt(_tl_sum64_ref(x) * m + eps)
    y = (x.float() * rstd[:, None]) * reference.sigmoid(og)
    assert torch.equal(y.to(torch.bfloat16), reference.gated_norm(x, og))


@needs_gfx950
def test_k1_context_probe():
    """K1's context runs: the fss hand-off, the arow row behind its ardy flag,
    every Phase B / C region where ``spec.scratch_layout`` puts it, tags of a
    K1 slot."""
    probe = _b1_probes()["k1ctx"]
    bufs = context.ProbeBuffers(epoch=9, rank=5, slot=spec.k1_slot(3))
    tag = bufs.tag()
    out = torch.zeros(OUT_WORDS, dtype=torch.int32, device="cuda")
    probe(bufs, out=out)
    torch.accelerator.synchronize()
    assert not bufs.failed(), bufs.given_up_waits()
    o = out.cpu()
    t = torch.arange(64)
    assert torch.equal(o[:64], (t + 1) % 64 * 3 + 1)
    assert torch.equal(o[64:128], 1 + (t + 1) % 64 + 40)
    assert o[128:133].tolist() == [9, 1, spec.k1_slot(3), 5, 2]
    vals, tags = bufs.read_pairs("fss", 0, 64)
    assert torch.equal(vals.cpu(), t * 3 + 1) and (tags == tag).all()
    for i, r in enumerate(spec.SCRATCH_REGIONS_BC):
        vals, tags = bufs.read_pairs(r, BC_LAST_PAIR[i], 1)
        assert (vals.item(), tags.item()) == (1000 + i, tag), r
    for i, r in enumerate(BC_PLAIN):
        off, nbytes = LAY[r]
        word = bufs.scratch[off + nbytes - 4 : off + nbytes].view(torch.int32)
        assert word.item() == 2000 + i, r
    arow = LAY[spec.AROW][0]
    got = bufs.scratch[arow : arow + 256].view(torch.int32).cpu()
    assert torch.equal(got, t + 40)
    tt = torch.arange(THREADS)
    assert torch.equal(o[256 : 256 + THREADS], (tt + 1) % THREADS * 7)
    # nothing outside the scratch's regions moved: the peer buffer is untouched
    assert not bufs.peer.any()


@needs_gfx950
def test_hw_sqrt_probe():
    """``hw_sqrt`` is v_sqrt_f32: within 1 ulp of the exact root over the
    normal range, exact on perfect squares, sqrt(+-0) = +-0, sqrt(inf) = inf,
    NaN below 0."""
    probe = _b1_probes()["sqrt"]
    bufs = context.ProbeBuffers()
    g = torch.Generator().manual_seed(3)
    x = torch.exp2(torch.empty(THREADS).uniform_(-120, 120, generator=g))
    x[:8] = torch.tensor([0.0, -0.0, float("inf"), -1.0, 4.0, 2.25, 1e-30, 9.0])
    y = torch.zeros(THREADS, dtype=torch.float32, device="cuda")
    probe(bufs, x=x.cuda(), y=y)
    torch.accelerator.synchronize()
    y = y.cpu()
    assert y[0].item() == 0.0 and y[1].item() == 0.0 and torch.signbit(y[1])
    assert y[2].item() == float("inf") and y[3].isnan()
    assert y[4].item() == 2.0 and y[5].item() == 1.5 and y[7].item() == 3.0
    exact = x[6:].double().sqrt()
    ulp = torch.finfo(torch.float32).eps * exact
    assert ((y[6:].double() - exact).abs() <= ulp).all()


@needs_gfx950
def test_seq8_sum_matches_the_fused_insert():
    """``seq8_sum`` (and the QK-norm built on it) on 1e5 heads: the sum equals
    the fp32 emulation of the .cu's order bit for bit, and the normalized
    head equals the stock ``fused_muse_spark_qknorm_rope_kv_insert`` (NoPE,
    HEADS_PER_GROUP 1) bit for bit."""
    from vllm import _custom_ops as ops

    probe = _b1_probes()["qknorm"]
    bufs = context.ProbeBuffers()
    x = _rows(N_ROWS, 20)
    xs = _padded(x)
    y = torch.zeros(xs.shape, dtype=torch.float32, device="cuda")
    tot = torch.zeros(xs.shape[0], dtype=torch.float32, device="cuda")
    _run_rows(probe, bufs, N_ROWS, x=xs, y=y, tot=tot)
    assert torch.equal(tot[:N_ROWS].cpu(), _seq8_sum_ref(x))
    got = y[:N_ROWS].to(torch.bfloat16).cpu()
    # the stock op on the same heads: 18 normalized heads a token (q and k),
    # 1024 tokens a call (HEADS_PER_GROUP 1 at NoPE below 2048), no insert
    nh = spec.HEADS + spec.KV_HEADS
    S = 1024
    tokens = -(-N_ROWS // nh)
    heads = torch.zeros(tokens * nh, 64, dtype=torch.bfloat16)
    heads[:N_ROWS] = x
    kv = torch.zeros(
        4, spec.KV_HEADS, spec.KV_BLOCK, spec.KV_C, dtype=torch.bfloat16, device="cuda"
    )
    stock = []
    for t0 in range(0, tokens, S):
        n = min(S, tokens - t0)
        qkv = torch.zeros(n, spec.QKV_ROWS, dtype=torch.bfloat16)
        qkv[:, : spec.ROW_V] = heads[t0 * nh : (t0 + n) * nh].reshape(n, -1)
        qkv = qkv.cuda()
        ops.fused_muse_spark_qknorm_rope_kv_insert(
            qkv,
            torch.zeros(n, dtype=torch.int64, device="cuda"),
            spec.HEADS,
            spec.KV_HEADS,
            spec.HEAD_DIM,
            spec.RMS_EPS,
            None,
            0,
            torch.full((n,), -1, dtype=torch.int64, device="cuda"),
            kv,
            spec.KV_BLOCK,
        )
        stock.append(qkv[:, : spec.ROW_V].reshape(-1, 64).cpu())
    stock = torch.cat(stock)[:N_ROWS]
    assert not kv.any()  # slot -1: no insert
    same = (got == stock).all(1)
    assert same.all(), (
        f"{(~same).sum().item()} heads differ, e.g. {(~same).nonzero()[:4]}"
    )


@needs_gfx950
def test_tl_sum64_matches_the_gated_norm():
    """``tl_sum64`` (and the gate built on it with ``hw_sqrt``) on 1e5 rows:
    the sum equals the fp32 emulation of Triton's gfx950 tree bit for bit, and
    the gated row equals the runtime-compiled FLA ``rms_norm_gated`` (o_gate
    read at the served stride 2304) ``torch.equal``; gate values cover
    sigmoid's saturation (+-88, +-100)."""
    from vllm.third_party.flash_linear_attention.ops.kda import (
        rms_norm_gated,
    )

    probe = _b1_probes()["gate"]
    bufs = context.ProbeBuffers()
    x = _rows(N_ROWS, 21)
    tokens = -(-N_ROWS // spec.HEADS)
    g = torch.Generator().manual_seed(22)
    qkvo = (torch.randn(tokens, spec.QKV_ROWS, generator=g) * 4).to(torch.bfloat16)
    # og rows (token, head) as a copy: the og columns have the row stride
    # 2304, so no [rows, 64] view of them exists; the edits go back below
    og = qkvo[:, spec.ROW_G :].reshape(-1, 64)
    og[0::89] = torch.tensor([-100.0, -88.0, 0.0, 88.0, 100.0] * 12 + [0.0] * 4).to(
        torch.bfloat16
    )
    qkvo[:, spec.ROW_G :] = og.view(tokens, -1)
    attn = torch.zeros(tokens * spec.HEADS, 64, dtype=torch.bfloat16)
    attn[:N_ROWS] = x
    xs, gs = _padded(attn[:N_ROWS]), _padded(og[:N_ROWS])
    y = torch.zeros(xs.shape, dtype=torch.float32, device="cuda")
    tot = torch.zeros(xs.shape[0], dtype=torch.float32, device="cuda")
    _run_rows(probe, bufs, N_ROWS, x=xs, g=gs, y=y, tot=tot)
    assert torch.equal(tot[:N_ROWS].cpu(), _tl_sum64_ref(x))
    got = y[:N_ROWS].to(torch.bfloat16).cpu()
    qkvo_d = qkvo.cuda()
    stock = rms_norm_gated(
        attn.cuda().view(tokens, spec.HEADS, 64),
        qkvo_d[:, spec.ROW_G :].reshape(tokens, spec.HEADS, 64),
        None,
        None,
        "sigmoid",
        eps=spec.RMS_EPS,
    )
    stock = stock.reshape(-1, 64)[:N_ROWS].cpu()
    same = (got == stock).all(1)
    assert same.all(), (
        f"{(~same).sum().item()} rows differ, e.g. {(~same).nonzero()[:4]}"
    )
