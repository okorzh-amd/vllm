# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C1AB tuned4: the TP4 builds that VLLM_C1AB_TUNED_FRONT, VLLM_C1AB_TUNED_FFN
and VLLM_C1AB_X8R switch on, against the generic TP4 builds of the same layer
(a ``DSV41MonoLayer(generic=True)`` runner), every K1 and K2 output bit for bit
on the same inputs. Adapted from vllm-project/vllm#63027's
``test_mono_tuned_builds_match_generic`` (be9f259d42), which covers TP2 ratio 2.

Place at tests/models/test_dsv41_mono_c1ab_tuned4.py of the c1ab/tuned4 tree
(or the image's) and run on 4 MI355X GPUs:

    pytest -sv tests/models/test_dsv41_mono_c1ab_tuned4.py

Knobs (environment):
  C1AB_TEST_WIDTHS   step widths, default "1,7,14,21,28" (c1 is 7, c4 7..28)
  C1AB_TEST_RATIOS   compress ratios, default "0,1,2"
  C1AB_TEST_COMBOS   flag sets, ';'-separated, each ','-separated of
                     front / ffn / x8r; default "front;ffn;x8r;front,x8r;front,ffn,x8r"
  C1AB_TEST_KLEN_FIXED=1  also compare an index source's K2 at widths <= 36:
                     without the klen-race fix (#61482 port) its combine can
                     read klen before the key table publishes it, generic
                     builds included, so it is not reproducible run to run.
"""

import os

import pytest
import ray
import torch

from vllm.platforms import current_platform

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

TP = 4

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
HIDDEN, Q_RANK, HEAD_DIM, NOPE, ROPE = 5120, 1280, 512, 448, 64
N_HEADS, O_GROUPS, O_RANK = 64, 8, 1024
WINDOW, TOPK, SWA_BLOCK, CACHE_BLOCK = 128, 512, 32, 128
RECORD, DATA, ALIGN = 584, 576, 576
REPLAYS = 300
HC = 4


def _on_cdna4() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import get_cdna_version

    return get_cdna_version() == 4


def _on_gfx942() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx942

    return on_gfx942()


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative L2 error of ``a`` against ``b``: unlike the cosine, it sees a
    wrong scale."""
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return ((a - b).norm() / b.norm()).item()


# ---------------------------------------------------------------- caches


def _paged_cache(num_blocks: int, block: int, device) -> torch.Tensor:
    """[blocks, block, 584] u8 over pages of round_up(block * 584, 576) bytes,
    the way vLLM binds a packed V4 cache."""
    page = -(-block * RECORD // ALIGN) * ALIGN
    raw = torch.zeros(num_blocks * page, dtype=torch.uint8, device=device)
    return torch.as_strided(raw, (num_blocks, block, RECORD), (page, RECORD, 1))


def _rope(kv: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor):
    """GPT-J RoPE of the last 64 dims (fp32 math, bf16 out)."""
    cs = cos_sin[positions.long()]
    cos, sin = cs[:, :32], cs[:, 32:]
    x = kv.float()
    e, o = x[:, NOPE::2].clone(), x[:, NOPE + 1 :: 2].clone()
    x[:, NOPE::2] = e * cos - o * sin
    x[:, NOPE + 1 :: 2] = e * sin + o * cos
    return x.to(torch.bfloat16)


def _write_records(cache, block: int, slots, kv, fnuz: bool = False):
    """fp8_ds_mla records of roped bf16 kv rows at ``slots``: the 448 nope dims
    in e4m3 with a UE8M0 per 64, the rope dims in bf16, as vLLM's insert.
    ``fnuz``: the e4m3 bytes in FNUZ with a maximum of 224, as vLLM's gfx942
    window cache holds them. In FNUZ the OCP -0 byte would be NaN."""
    fp8, fp8_max = torch.float8_e4m3fn, 448.0
    if fnuz:
        fp8, fp8_max = torch.float8_e4m3fnuz, 224.0
    nope = kv.float()[:, :NOPE].view(-1, 7, 64)
    exp = torch.ceil(torch.log2(nope.abs().amax(-1).clamp_min(1e-4) / fp8_max))
    q = nope * torch.exp2(-exp)[..., None]
    q = q.clamp(-fp8_max, fp8_max).reshape(-1, NOPE)
    data = torch.cat(
        [
            q.to(fp8).view(torch.uint8),
            kv[:, NOPE:].contiguous().view(torch.uint8),
        ],
        1,
    )
    scales = torch.zeros(kv.shape[0], 8, dtype=torch.uint8, device=kv.device)
    scales[:, :7] = (exp + 127).clamp(0, 255).to(torch.uint8)
    n = cache.untyped_storage().nbytes() - cache.storage_offset()
    flat = torch.as_strided(cache, (n,), (1,))
    slots = slots.to(kv.device).long()
    b, o = slots // block, slots % block
    d0 = b * cache.stride(0) + o * DATA
    s0 = b * cache.stride(0) + block * DATA + o * 8
    ar = torch.arange(DATA, device=kv.device)
    flat[(d0[:, None] + ar).reshape(-1)] = data.reshape(-1)
    ar = torch.arange(8, device=kv.device)
    flat[(s0[:, None] + ar).reshape(-1)] = scales.reshape(-1)


def _normed_kv(n: int, kv_norm, g, device):
    x = torch.randn(n, HEAD_DIM, generator=g).to(device)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)) * kv_norm.float()
    return x.to(torch.bfloat16)


def _decode_step(
    ratio: int, R: int, T: int, kv_norm, cos_sin, seed: int, fnuz: bool = False
):
    """R requests x T tokens past random contexts: the step's metadata as
    vLLM's builders lay it out, its window rows and (ratio > 0) the compressed
    cache filled, a random top-512 a token. ``fnuz``: the window records in
    FNUZ, as on gfx942. The compressed records stay OCP, as vLLM's Triton
    compressor writes them on gfx942 too."""
    g = torch.Generator().manual_seed(seed)
    dev = kv_norm.device
    ctx = [int(c) for c in torch.randint(2000, 40000, (R,), generator=g)]
    M = R * T
    spans = [
        (max(0, c - WINDOW - T) // SWA_BLOCK, (c + T - 1) // SWA_BLOCK) for c in ctx
    ]
    n_swa = 1 + sum(hi - lo + 1 for lo, hi in spans)
    perm = (torch.randperm(n_swa - 1, generator=g) + 1).tolist()
    bt = torch.zeros(R, max(hi for _, hi in spans) + 1, dtype=torch.int64)
    for r, (lo, hi) in enumerate(spans):
        for lb in range(lo, hi + 1):
            bt[r, lb] = perm.pop()
    swa = _paged_cache(n_swa, SWA_BLOCK, dev)

    def slot(r, p):
        return int(bt[r, p // SWA_BLOCK]) * SWA_BLOCK + p % SWA_BLOCK

    pos = [ctx[r] + t for r in range(R) for t in range(T)]
    t2r = [r for r in range(R) for _ in range(T)]
    swa_idx = torch.full((M, WINDOW), -1, dtype=torch.int32)
    swa_len = torch.zeros(M, dtype=torch.int32)
    for i in range(M):
        ps = range(max(pos[i] - WINDOW + 1, 0), pos[i] + 1)
        swa_idx[i, : len(ps)] = torch.tensor([slot(t2r[i], p) for p in ps])
        swa_len[i] = len(ps)
    hist = [
        (r, p) for r in range(R) for p in range(max(0, ctx[r] - WINDOW - T), ctx[r])
    ]
    hp = torch.tensor([p for _, p in hist], device=dev)
    kv = _rope(_normed_kv(len(hist), kv_norm, g, dev), hp, cos_sin)
    _write_records(
        swa, SWA_BLOCK, torch.tensor([slot(r, p) for r, p in hist]), kv, fnuz
    )

    step = dict(
        M=M,
        positions=torch.tensor(pos, dtype=torch.int64, device=dev),
        slot_mapping=torch.tensor(
            [slot(t2r[i], pos[i]) for i in range(M)], dtype=torch.int64, device=dev
        ),
        token_to_req=torch.tensor(t2r, dtype=torch.int32, device=dev),
        swa_indices=swa_idx.to(dev),
        swa_lens=swa_len.to(dev),
        swa_cache=swa,
        topk=None,
        comp_cache=None,
        comp_block_table=None,
        comp_entries=0,
    )
    if ratio:
        entries = CACHE_BLOCK // ratio
        lens = [(c + T) // ratio for c in ctx]
        nblk = [-(-n // entries) for n in lens]
        perm = (torch.randperm(sum(nblk), generator=g) + 1).tolist()
        cbt = torch.zeros(R, max(nblk), dtype=torch.int32)
        for r in range(R):
            for b in range(nblk[r]):
                cbt[r, b] = perm.pop()
        comp = _paged_cache(1 + sum(nblk), entries, dev)
        for r in range(R):
            idx = torch.arange(lens[r])
            slots = cbt[r, idx // entries].long() * entries + idx % entries
            kv = _rope(
                _normed_kv(lens[r], kv_norm, g, dev), (idx * ratio).to(dev), cos_sin
            )
            _write_records(comp, entries, slots, kv)
        topk = torch.full((M, TOPK), -1, dtype=torch.int32)
        for i in range(M):
            cand = (pos[i] + 1) // ratio
            n = min(cand, TOPK)
            topk[i, :n] = torch.randperm(cand, generator=g)[:n].to(torch.int32)
        step.update(
            topk=topk.to(dev),
            comp_cache=comp,
            comp_block_table=cbt.to(dev),
            comp_entries=entries,
        )
    return step


def _rows(s: dict, a: int, b: int) -> dict:
    """The decode step ``s`` cut to its rows a .. b - 1, as a step of its own
    on the same caches. The rows keep their request numbers, so the block
    tables of ``s`` stay valid."""
    cut = dict(s, M=b - a)
    for k in ("positions", "slot_mapping", "token_to_req", "swa_indices", "swa_lens"):
        cut[k] = s[k][a:b]
    if s["topk"] is not None:
        cut["topk"] = s["topk"][a:b]
    return cut


# ---------------------------------------------------------------- weights


def _mxfp8(n: int, k: int, g, device):
    """A random MXFP8 weight [n, k] e4m3 and its E8M0 32 x 32 block scales."""
    w = torch.randn(n, k, generator=g).clamp(-6, 6).to(torch.float8_e4m3fn)
    s = torch.randint(118, 123, (n // 32, k // 32), generator=g, dtype=torch.uint8)
    return w.to(device), s.to(device)


def _random_moe(moe, gd, gr, inter: int) -> None:
    """Random parameters in the loaded layout, before post-load processing: this
    rank's shards (``gd``, a device generator) of the MXFP4 routed experts
    (random nibbles, E8M0 scales; the intermediate past ``inter`` zero, as the
    loader pads it) and the MXFP8 shared expert (one E8M0 a 32 x 32 block); the
    replicated router from ``gr``, seeded alike on every rank."""
    u8 = torch.uint8
    for lin in (moe.shared_experts.gate_up_proj, moe.shared_experts.down_proj):
        n = lin.weight.shape[0]
        w = torch.randn(lin.weight.shape, generator=gd, device=gd.device)
        lin.weight.data.copy_(w.clamp(-6, 6).to(lin.weight.dtype))
        sc = lin.weight_scale.data.view(u8)
        rows = -(-n // 32) if sc.shape[0] == n else sc.shape[0]
        blocks = torch.randint(
            118, 123, (rows, sc.shape[1]), generator=gd, device=gd.device, dtype=u8
        )
        sc.copy_(blocks.repeat_interleave(32, 0)[:n] if sc.shape[0] == n else blocks)
    e = moe.experts.routed_experts
    for name, lo, hi in (("weight", 0, 256), ("weight_scale", 118, 122)):
        for w in ("w13", "w2"):
            t = getattr(e, f"{w}_{name}").data.view(u8)
            t.copy_(
                torch.randint(lo, hi, t.shape, generator=gd, device=gd.device, dtype=u8)
            )
    padded = e.w13_weight.shape[1] // 2
    if padded > inter:
        w13, w2 = e.w13_weight.data.view(u8), e.w2_weight.data.view(u8)
        for half in (0, padded):
            w13[:, half + inter : half + padded] = 0
        w2[..., inter // 2 :] = 0
    for t in (moe.gate.weight, moe.gate.e_score_correction_bias):
        t.data.copy_(torch.randn(t.shape, generator=gr, device=gr.device) * 0.02)


# ---------------------------------------------------------------- tuned4


def _env_list(name, default, conv=int):
    raw = os.environ.get(name, default)
    return [conv(v) for v in raw.split(",") if v.strip()]


def _combos():
    raw = os.environ.get("C1AB_TEST_COMBOS", "front;ffn;x8r;front,x8r;front,ffn,x8r")
    out = []
    for combo in raw.split(";"):
        names = {n.strip() for n in combo.split(",") if n.strip()}
        assert names <= {"front", "ffn", "x8r"}, names
        out.append({k: k in names for k in ("front", "ffn", "x8r")})
    return out


@ray.remote(num_gpus=1, max_calls=1)
def _mono_tuned4(monkeypatch, tp_size, pp_size, rank, distributed_init_port):
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.v1.worker.workspace import init_workspace_manager

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.setenv("VLLM_ROCM_USE_AITER", "1")
        m.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
        # the runners take their switches from the test, never the environment
        for v in ("VLLM_C1AB_TUNED_FRONT", "VLLM_C1AB_TUNED_FFN", "VLLM_C1AB_X8R"):
            m.delenv(v, raising=False)
        rocm_aiter_ops.refresh_env_variables()
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        init_workspace_manager(device)
        _check_tuned4(rank, tp_size, device)


def _check_tuned4(rank: int, tp: int, device) -> None:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import get_tp_group
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.models.deepseek_v41.amd.model import DeepseekV4MoE
    from vllm.models.deepseek_v41.amd.mono.runner import (
        AttnWeights,
        DSV41MonoLayer,
        MonoLayerWeights,
    )
    from vllm.models.deepseek_v41.common.rope import build_deepseek_v4_rope
    from vllm.utils.torch_utils import set_default_torch_dtype

    widths = _env_list("C1AB_TEST_WIDTHS", "1,7,14,21,28")
    ratios = _env_list("C1AB_TEST_RATIOS", "0,1,2")
    klen_fixed = os.environ.get("C1AB_TEST_KLEN_FIXED", "0") == "1"
    combos = _combos()

    vc = EngineArgs(
        model=MODEL,
        tensor_parallel_size=tp,
        moe_backend="aiter",
        load_format="dummy",
        max_model_len=65536,
        language_model_only=True,
    ).create_engine_config()
    cfg = vc.model_config.hf_config
    cfg = getattr(cfg, "text_config", cfg)
    g = torch.Generator().manual_seed(1000)
    gs = torch.Generator().manual_seed(1001 + rank)
    H, G = N_HEADS // tp, O_GROUPS // tp
    bf = torch.bfloat16
    with set_current_vllm_config(vc), set_default_torch_dtype(bf), device:
        moe = DeepseekV4MoE(vc, prefix="model.layers.21.ffn")
    _random_moe(
        moe,
        torch.Generator(device=device).manual_seed(2001 + rank),
        torch.Generator(device=device).manual_seed(2000),
        inter=2304 // tp,
    )
    process_weights_after_loading(moe, vc.model_config, device)

    def f32(*shape, s=1.0, gen=g):
        return (torch.randn(*shape, generator=gen) * s).to(device)

    # the rope table's values do not matter to a tuned-vs-generic comparison:
    # the ratio-2 table for every ratio (as #63027's test builds it)
    with set_current_vllm_config(VllmConfig()):
        rope = build_deepseek_v4_rope(
            cfg, head_dim=HEAD_DIM, rope_head_dim=ROPE,
            max_position_embeddings=cfg.max_position_embeddings, compress_ratio=2,
        )  # fmt: skip
    cos_sin = rope.cos_sin_cache.to(device=device, dtype=torch.float32)
    wqkv, wqkv_s = _mxfp8(1792, HIDDEN, g, device)
    wq_b, wq_b_s = _mxfp8(H * HEAD_DIM, Q_RANK, gs, device)
    wo_a, wo_a_s = _mxfp8(G * O_RANK, H // G * HEAD_DIM, gs, device)
    wo_b, wo_b_s = _mxfp8(HIDDEN, G * O_RANK, gs, device)
    q_norm, kv_norm = (1 + f32(Q_RANK, s=0.1)).to(bf), (1 + f32(HEAD_DIM, s=0.1)).to(bf)
    sink = f32(H, gen=gs)
    mix = (2 + HC) * HC
    hc = [
        (f32(mix, HC * HIDDEN, s=0.01), f32(3, s=0.5).abs() + 0.5, f32(mix, s=0.1),
         (1 + f32(HIDDEN, s=0.1)).to(bf))
        for _ in range(2)
    ]  # fmt: skip
    e, sh = moe.experts.routed_experts, moe.shared_experts

    def weights(ratio):
        return MonoLayerWeights(
            attn=AttnWeights(
                layer_id=21, wqkv=wqkv, wqkv_scale=wqkv_s, q_norm=q_norm,
                kv_norm=kv_norm, wq_b=wq_b, wq_b_scale=wq_b_s, wo_a=wo_a,
                wo_a_scale=wo_a_s, wo_b=wo_b, wo_b_scale=wo_b_s,
                attn_sink=sink, cos_sin=cos_sin, ratio=ratio,
            ),
            hc_attn_fn=hc[0][0], hc_attn_scale=hc[0][1], hc_attn_base=hc[0][2],
            attn_norm=hc[0][3], hc_ffn_fn=hc[1][0], hc_ffn_scale=hc[1][1],
            hc_ffn_base=hc[1][2], ffn_norm=hc[1][3],
            gate_w=moe.gate.weight, bias=moe.gate.e_score_correction_bias,
            w13=e.w13_weight, w13_s=e.w13_weight_scale, w2=e.w2_weight,
            w2_s=e.w2_weight_scale, sgu=sh.gate_up_proj.weight,
            sgu_s=sh.gate_up_proj.weight_scale.view(torch.uint8),
            sw2=sh.down_proj.weight,
            sw2_s=sh.down_proj.weight_scale.view(torch.uint8),
        )  # fmt: skip

    group = get_tp_group().cpu_group
    generic = DSV41MonoLayer(tp, rank, group, device, generic=True)
    runners = [
        (combo, DSV41MonoLayer(tp, rank, group, device, c1ab=combo)) for combo in combos
    ]

    def run(runner, mw, s, ins, fold: bool, index: bool) -> dict:
        ratio = mw.attn.ratio
        swa, topk = s["swa_cache"].clone(), s["topk"]
        select = None
        if index:
            topk = torch.full_like(s["topk"], -1)
            select = lambda normed, qr: topk.copy_(s["topk"])  # noqa: E731
        outs = runner.forward(
            mw, *ins, s["positions"], s["slot_mapping"], swa, s["swa_indices"],
            s["swa_lens"], s["token_to_req"], topk_indices=topk,
            comp_cache=s["comp_cache"], comp_block_table=s["comp_block_table"],
            fold=fold, select=select,
        )  # fmt: skip
        torch.accelerator.synchronize()
        M = s["M"]
        scratch, layout = runner.step_scratch(M, ratio)
        off = layout["normed"][0]
        normed = scratch[off : off + M * HIDDEN * 2]
        k1 = dict(
            res_mid=runner.res_mid[:M], post_a=runner.post_a[:M],
            comb_a=runner.comb_a[:M], pre_a=runner.pre_a[:M], swa_cache=swa,
        )  # fmt: skip
        if index:
            k1.update(normed=runner.normed[:M], qr=runner.qr[:M])
        k2 = dict(enumerate(outs), ffn_in=normed)
        return {
            "k1": {k: t.clone() for k, t in k1.items()},
            "k2": {k: t.clone() for k, t in k2.items()},
        }

    fails = []
    for ratio in ratios:
        mw = weights(ratio)
        for M in widths:
            # DSpark k=6: a request's 7 rows a step
            R, T = (M // 7, 7) if M % 7 == 0 else (M, 1)
            s = _decode_step(ratio, R, T, kv_norm, cos_sin, seed=100 + M + 7 * ratio)
            gi = torch.Generator().manual_seed(11 + M)
            res0 = torch.randn(M, HC, HIDDEN, generator=gi)
            res0 = (res0 * torch.rsqrt(res0.pow(2).mean(-1, keepdim=True))).to(bf)
            comb0 = torch.rand(M, HC, HC, generator=gi) + 0.1
            comb0 = comb0 / comb0.sum(-1, keepdim=True)
            ins = [
                (torch.randn(M, HIDDEN, generator=gi) * 0.5).to(bf),
                res0,
                torch.sigmoid(torch.randn(M, HC, 1, generator=gi)) * 2,
                comb0,
                torch.sigmoid(torch.randn(M, HC, generator=gi)) + 1e-6,
            ]
            ins = [t.to(device) for t in ins]
            for fold in (True, False):
                for index in (False, True) if ratio else (False,):
                    want = run(generic, mw, s, ins, fold, index)
                    parts = ["k1", "k2"]
                    if index and M <= 36 and not klen_fixed:
                        parts = ["k1"]
                    for combo, runner in runners:
                        b = runner.build_key(M, ratio, fold, index)
                        assert b.tuned_front == combo["front"], (combo, b)
                        assert b.tuned_k2 == combo["ffn"], (combo, b)
                        assert b.x8r == (combo["x8r"] and not combo["ffn"]), (combo, b)
                        got = run(runner, mw, s, ins, fold, index)
                        tag = f"{combo} ratio={ratio} M={M} fold={fold} index={index}"
                        for part in parts:
                            bad = [
                                k
                                for k, t in got[part].items()
                                if not torch.equal(t, want[part][k])
                            ]
                            if bad:
                                fails.append(f"{tag}: {part} outputs {bad} differ")
                        if rank == 0:
                            print("checked", tag, parts, flush=True)
    assert not fails, "\n".join(fails)


@pytest.mark.skipif(not _on_cdna4(), reason="the mono kernels target CDNA4")
def test_mono_c1ab_tuned4_builds_match_generic(monkeypatch: pytest.MonkeyPatch):
    if torch.accelerator.device_count() < TP:
        pytest.skip(f"needs {TP} GPUs")
    pytest.importorskip("flydsl")
    multi_process_parallel(monkeypatch, TP, 1, _mono_tuned4)
