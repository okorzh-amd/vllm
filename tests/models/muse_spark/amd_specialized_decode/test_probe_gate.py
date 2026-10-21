# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2(gated)'s core load (``mono/attention/gate.py``, DESIGN_BC 6, WP-B6):
the attention output's norm and gate, in a probe kernel and in K2.

[CPU] The probe compiles GPU-less for gfx950 (the ISA gate, no waterfall);
the test inputs make the edges bite (the eps rows, sigmoid's saturation, NaN
pad rows), checked on B0's reference.
[1GPU] ``back.core_loads`` + ``back.load_core`` ("gated") on every CTA of the
grid, each on its own S rows: the LDS ``xl`` rows ``torch.equal`` the
runtime-compiled FLA ``rms_norm_gated`` (one served-shape call first, so
Triton's cache holds the specialization decode runs; o_gate read in place at
the qkv row stride 2304, as served); pad rows with NaN / inf inputs are exact
zeros. 9.2's acceptance.
[TP8] K2(gated) on (attn, qkvo) equals K2(tensor) on vLLM's gated core
``rms_norm_gated(attn, og)`` bit for bit (``stream_out``, ``branch_out``),
at widths 1, 3, 8, with the gate's and the seam's eps rows biting; NaN pad
rows; an all-pad step; a CUDA graph of gated steps replayed 300 times; no
wait given up.
"""

import glob
import os

import pytest
import ray
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
from tests.models.muse_spark.amd_specialized_decode import (  # noqa: E402
    test_numerics as tn,
)
from tests.utils import multi_gpu_test, multi_process_parallel  # noqa: E402
from vllm.models.muse_spark.amd.specialized_decode.mono import spec  # noqa: E402
from vllm.models.muse_spark.amd.specialized_decode.mono.abi import (  # noqa: E402
    STARTUP_WAIT_SECONDS,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.attention import (  # noqa: E402
    back,
    gate,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common import (  # noqa: E402
    context,
)
from vllm.models.muse_spark.amd.specialized_decode.mono.common.ops import (  # noqa: E402
    rsrc,
)

bf = torch.bfloat16
CORE, QKV, ROW_G = spec.CORE, spec.QKV_ROWS, spec.ROW_G
XL_WORDS = back.XL_ROW // 2  # i32 words a padded LDS core row
ROW_WORDS = CORE // 2  # i32 words a core row
PAD_BYTES = 16  # a CTA's pad flags in the probe (whole words: ld_pad's)
WIDTHS = (1, 3, 8)
G_EDGES = (-100.0, -88.0, 0.0, 88.0, 100.0)  # sigmoid's saturation and centre


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


def _gate_core(c):
    """CTA b's gated core load on rows b S .. b S + S - 1 of ``attn`` / ``qkvo``
    (its own pad word block), as K2(gated) runs it, then its LDS ``xl`` rows
    -> ``out`` rows b S .. (bf16 [BLOCKS S, CORE])."""
    s, tid = c["S"], c["tid"]
    a = c["args"]
    b = fx.Int64(fx.Int32(c["bid"]))
    mine = dict(c)
    mine["args"] = {
        **a,
        "attn": a["attn"] + b * (s * CORE * 2),
        "qkvo": a["qkvo"] + b * (s * QKV * 2),
        "pad": a["pad"] + b * PAD_BYTES,
    }
    pre = back.core_loads(mine, "gated")
    back.load_core(mine, "gated", pre)
    xl = c["lds"]["back"]["xl"]
    out = rsrc(a["out"] + b * (s * CORE * 2))
    for i in range_constexpr(s):  # s rows of 512 words, a word a thread
        v = fx.ptr_load(xl + (i * XL_WORDS + tid))
        bo.buffer_store(v, out, i * ROW_WORDS + tid)


def _probe(s: int):
    return context.probe_kernel(
        f"muse_probe_gate_core_s{s}",
        s,
        _gate_core,
        {"back": back.lds_struct(s)},
        ("attn", "qkvo", "pad", "out"),
        blocks=spec.BLOCKS,
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


# ---------------------------------------------------------------- inputs


def _cases(s: int, seed: int):
    """(attn bf16 [BLOCKS s, CORE], qkvo bf16 [BLOCKS s, QKV], pad bool
    [BLOCKS, PAD_BYTES], live [BLOCKS s]) on the CPU. Heads over five decades
    of scale; eps-scale heads (|x| ~ 3e-3: mean(x^2) ~ eps); all-zero heads;
    one huge element; o_gate rows of +-100, +-88, 0 and large random gates;
    pad rows in most CTAs (CTA 7 all pad) with NaN / inf inputs."""
    g = torch.Generator().manual_seed(seed)
    n = spec.BLOCKS * s
    heads = n * spec.HEADS
    scale = torch.exp2(torch.randint(-8, 9, (heads, 1), generator=g).float())
    x = torch.randn(heads, 64, generator=g) * scale
    x[0::37] = torch.randn(x[0::37].shape, generator=g) * 3e-3
    x[5::101] = 0
    x[9::113, 17] = 3e4
    qkvo = torch.randn(n, QKV, generator=g) * 4
    og = qkvo[:, ROW_G:].reshape(heads, 64)
    og[2::29] = torch.tensor(G_EDGES * 12 + (0.0,) * 4)
    og[3::31] *= 30
    qkvo[:, ROW_G:] = og.reshape(n, CORE)
    attn = x.reshape(n, CORE)
    pad = torch.zeros(spec.BLOCKS, PAD_BYTES, dtype=torch.bool)
    for b in range(spec.BLOCKS):
        for t in range(s):
            pad[b, t] = (b + 3 * t) % 5 == 0 or b == 7
    live = ~pad[:, :s].reshape(-1)
    attn[~live] = float("nan")
    attn[~live, 3] = float("inf")
    qkvo[~live, ROW_G:] = float("nan")
    return attn.to(bf), qkvo.to(bf), pad, live


def test_cases_make_the_edges_bite():
    """[CPU] On B0's reference of the stock order: some live heads' rstd moves
    by > 5 % with eps (the eps rows), some gates saturate to 0 and 1, the pad
    rows hold NaN, every CTA width has live and pad rows."""
    for s in WIDTHS:
        attn, qkvo, pad, live = _cases(s, 40 + s)
        a = attn[live].float().reshape(-1, 64)
        ms = (a * a).mean(1)
        eps_rows = (ms > 0) & (ms < 20 * spec.RMS_EPS)
        assert eps_rows.sum() > 100, eps_rows.sum()
        og = qkvo[live, ROW_G:].float().reshape(-1, 64)
        for v in G_EDGES:
            assert (og == v).any(), v
        y = reference.gated_norm(attn[live], qkvo[live, ROW_G:])
        assert torch.isfinite(y.float()).all()
        assert (y.float() == 0).any()  # sigmoid(-100) x ... rounds to 0
        assert attn[~live].float().isnan().any() and live.any() and (~live).any()
        assert pad[7, :s].all()


@compiles
@pytest.mark.parametrize("s", [1, 8])
def test_gate_probe_compiles(s, tmp_path, monkeypatch):
    """The compile gate on the probe (K2's context and LDS union member)."""
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(tmp_path))
    probe = _probe(s)
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
    waterfall = re.compile(r"v_cmp_eq_u64_e\d+\s+\S+,\s*s\[\d+:\d+\],\s*v\[")
    assert not waterfall.findall(txt)
    assert len(re.findall(r"^\s*ds_write_b16\b", txt, re.M)) == gate.passes(s)
    assert "v_sqrt_f32" in txt and "v_div_fixup_f32" in txt
    assert "v_rsq_f32" not in txt


# ---------------------------------------------------------------- [1GPU]


@needs_gfx950
@pytest.mark.parametrize("s", WIDTHS)
def test_load_core_gated_matches_rms_norm_gated(s):
    """Every CTA's xl rows equal the runtime-compiled ``rms_norm_gated`` on
    the same (attn, og) bit for bit (and B0's reference of its order); pad
    rows are exact zeros."""
    from vllm.third_party.flash_linear_attention.ops.kda import (
        rms_norm_gated,
    )

    probe = _probe(s)
    bufs = context.ProbeBuffers()
    n = spec.BLOCKS * s
    for seed in (1, 2):
        attn, qkvo, pad, live = _cases(s, 100 * s + seed)
        attn_d, qkvo_d, pad_d = attn.cuda(), qkvo.cuda(), pad.cuda()

        def stock(a, q):
            t = a.shape[0]
            # rms_norm_gated writes y over x (kda.py: ``y = x`` when
            # out_dtype is None): on a copy, the inputs are reused below
            return rms_norm_gated(
                a.clone().view(t, spec.HEADS, 64),
                q[:, ROW_G:].view(t, spec.HEADS, 64),
                None,
                None,
                "sigmoid",
                eps=spec.RMS_EPS,
            ).reshape(t, CORE)

        # the served shape first (Triton's cache: the specialization decode
        # runs, o_gate at stride 2304), then every row at once (same one)
        first = stock(attn_d[:s], qkvo_d[:s])
        want = stock(attn_d, qkvo_d)
        assert torch.equal(first.view(torch.int16), want[:s].view(torch.int16))
        out = torch.full((n, CORE), float("nan"), dtype=bf, device="cuda")
        probe(bufs, attn=attn_d, qkvo=qkvo_d, pad=pad_d, out=out)
        torch.accelerator.synchronize()
        assert not bufs.failed(), bufs.given_up_waits()
        got = out.cpu()
        want = want.cpu()
        # bit for bit (a -0 is not a +0)
        same = (got[live].view(torch.int16) == want[live].view(torch.int16)).all(1)
        assert same.all(), (
            f"S={s}: {(~same).sum().item()} of {same.numel()} rows differ, e.g. "
            f"{live.nonzero()[(~same).nonzero()[:4, 0]].flatten().tolist()}"
        )
        ref = reference.gated_norm(attn[live], qkvo[live, ROW_G:])
        frac = (got[live] == ref).float().mean().item()
        print(f"S={s} seed={seed}: equal to the reference on {frac:.6f}")
        assert (got[~live].view(torch.int16) == 0).all()


# ---------------------------------------------------------------- [TP8]


def _gated_inputs(rank: int, device, S: int, case: str = "base"):
    """(attn [S, CORE] bf16, qkvo [S, QKV] bf16 of this rank, stream [S, H]
    fp32 replicated). "eps": row 0's heads at |x| ~ 3e-3 (the gate's eps
    bites) and its o_gate at -10 (sigmoid ~ 4.5e-5: the gated core is small,
    the seam's post-attention eps bites, as test_numerics' 1e-4 core row);
    the last row's stream x 3e-3 (the pre-FFN eps, as test_numerics)."""
    gr = torch.Generator(device=device).manual_seed(300 + 8 * S + rank)
    gc = torch.Generator(device=device).manual_seed(11 + S)
    attn = torch.randn(S, CORE, generator=gr, device=device)
    qkvo = torch.randn(S, QKV, generator=gr, device=device) * 2
    stream = torch.randn(S, spec.HIDDEN, generator=gc, device=device)
    if case == "eps":
        attn[0] *= 3e-3
        qkvo[0, ROW_G:] = -10.0 + 0.1 * qkvo[0, ROW_G:]
        stream[S - 1] *= 3e-3
    return attn.to(bf), qkvo.to(bf), stream


def _stock_core(attn, qkvo):
    """VLLM's gated core (o_proj's input) from the same rows."""
    from vllm.third_party.flash_linear_attention.ops.kda import (
        rms_norm_gated,
    )

    S = attn.shape[0]
    # rms_norm_gated writes y over x (kda.py: ``y = x`` when
    # out_dtype is None): on a copy, the inputs are reused below
    return rms_norm_gated(
        attn.clone().view(S, spec.HEADS, 64),
        qkvo[:, ROW_G:].view(S, spec.HEADS, 64),
        None,
        None,
        "sigmoid",
        eps=spec.RMS_EPS,
    ).reshape(S, CORE)


def _launch(runner, w, core_args: dict, stream, pad, key, *, begin=True):
    """One K2 step of build ``key`` (``core_args``: its core's tensors)."""
    from vllm.models.muse_spark.amd.specialized_decode.mono import layer

    S = key.tokens
    stream_out = torch.empty_like(stream)
    branch_out = torch.empty(S, spec.HIDDEN, dtype=bf, device=stream.device)
    if begin:
        runner.step_begin()
    layer.k2_launch(
        key,
        {
            **{n: w[n] for n in tn.KERNEL_WEIGHTS},
            **core_args,
            "stream": stream,
            "pad": pad,
            "stream_out": stream_out,
            "branch_out": branch_out,
            "slot": tn.SLOT,
            **runner.launch_args(),
            **runner.peer_args(),
        },
    )
    return stream_out, branch_out


def _pair(runner, w, attn, qkvo, stream, pad):
    """(gated outputs, tensor outputs on the stock gated core) of one step."""
    S = attn.shape[0]
    core = _stock_core(attn, qkvo)
    want = _launch(runner, w, {"core": core}, stream, pad, spec.K2Build(tokens=S))
    got = _launch(
        runner,
        w,
        {"attn": attn, "qkvo": qkvo},
        stream,
        pad,
        spec.K2Build(tokens=S, core="gated"),
    )
    torch.accelerator.synchronize()
    tn._assert_healthy(runner)
    return got, want


@ray.remote(num_gpus=1, max_calls=1)
def _gated_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    device = tn._init_worker(monkeypatch, tp_size, pp_size, rank, distributed_init_port)

    from vllm.models.muse_spark.amd.specialized_decode.runtime.runner import MonoRunner

    runner = MonoRunner(device)
    # the first launch of each build compiles it: the startup budget
    runner.set_wait_budget(STARTUP_WAIT_SECONDS)
    for case in ("base", "eps"):
        w = tn._weights(rank, device, case)
        for S in WIDTHS:
            attn, qkvo, stream = _gated_inputs(rank, device, S, case)
            got, want = _pair(runner, w, attn, qkvo, stream, tn._pad(device))
            with tn._tp_checks(f"gated == tensor ({case}, S={S})"):
                assert torch.equal(got[0], want[0]), (case, S, "stream_out")
                assert torch.equal(got[1], want[1]), (case, S, "branch_out")
            tn._all_ranks_equal(got[0], f"stream_out ({case}, S={S})")

    # pad rows with NaN / inf in attn and og: the live rows as the clean step's,
    # the pad rows exact zeros
    w = tn._weights(rank, device)
    attn, qkvo, stream = _gated_inputs(rank, device, 8)
    pad = tn._pad(device, (5, 6, 7))
    key = spec.K2Build(tokens=8, core="gated")
    clean = _launch(runner, w, {"attn": attn, "qkvo": qkvo}, stream, pad, key)
    dirty_a, dirty_q = attn.clone(), qkvo.clone()
    dirty_a[5:] = float("nan")
    dirty_a[6, 3] = float("inf")
    dirty_q[5:] = float("nan")
    got = _launch(runner, w, {"attn": dirty_a, "qkvo": dirty_q}, stream, pad, key)
    torch.accelerator.synchronize()
    tn._assert_healthy(runner)
    with tn._tp_checks("the NaN pad rows"):
        assert all(torch.equal(g[:5], c[:5]) for g, c in zip(got, clean))
        assert (got[0][5:] == 0).all() and (got[1][5:].float() == 0).all()
    # every row a pad row: zero outputs
    for S in (1, 8):
        attn, qkvo, stream = _gated_inputs(rank, device, S)
        all_pad = torch.ones(16, dtype=torch.bool, device=device)
        s_out, b_out = _launch(
            runner,
            w,
            {"attn": attn, "qkvo": qkvo},
            stream,
            all_pad,
            spec.K2Build(tokens=S, core="gated"),
        )
        torch.accelerator.synchronize()
        tn._assert_healthy(runner)
        with tn._tp_checks(f"the all-pad gated step S={S}"):
            assert (s_out == 0).all() and (b_out.float() == 0).all(), S

    # one CUDA graph of two gated steps at two widths, replayed: every replay
    # matches the first
    ins = {S: _gated_inputs(rank, device, S) for S in (1, 8)}
    pad = tn._pad(device)

    def step(S):
        attn, qkvo, stream = ins[S]
        return _launch(
            runner,
            w,
            {"attn": attn, "qkvo": qkvo},
            stream,
            pad,
            spec.K2Build(tokens=S, core="gated"),
        )

    step(1)  # warm-up (code objects loaded)
    step(8)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    outs = {}
    with torch.cuda.graph(graph):
        outs[1] = step(1)
        outs[8] = step(8)
    graph.replay()
    torch.accelerator.synchronize()
    first = {S: [t.clone() for t in o] for S, o in outs.items()}
    for _ in range(300):
        graph.replay()
    torch.accelerator.synchronize()
    with tn._tp_checks("the gated graph replays"):
        for S, o in outs.items():
            assert all(torch.equal(a, b) for a, b in zip(o, first[S])), S
    tn._assert_healthy(runner)


@needs_gfx950
@multi_gpu_test(num_gpus=8)
def test_k2_gated_equals_tensor_on_the_stock_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    multi_process_parallel(monkeypatch, 8, 1, _gated_worker)
