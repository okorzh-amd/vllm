# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Phase B / C references (``reference.py``) every probe and chain test
compares with, checked on their own.

[CPU] ``fma32`` is the correctly rounded fp32 fma (exact rationals, midpoint
ties); the rounding-order references (``qk_norm_rope``, ``gated_norm``,
``ffn_boundary``) are within 1 bf16 ulp of plain fp64 math; ``gqa_decode``
equals dense attention over the same keys, ignores every byte
``poison_unread`` marks and reads the cache through any strides; ``FakeMrv2``
keeps its buffers' addresses and writes block tables, slots, positions and pad
flags as MRV2 does.
[1GPU] On the GPU the front and gate references against the stock kernels
(``fused_muse_spark_qknorm_rope_kv_insert``, ``rms_norm_gated``): within 1 bf16
ulp everywhere and equal on >= 99.9 %, the cache bytes the insert writes too.
"""

import math
import random
from fractions import Fraction

import pytest
import torch

from tests.models.muse_spark.amd_specialized_decode import reference as ref
from vllm.models.muse_spark.amd.specialized_decode.mono import spec

BF, F32, F64 = torch.bfloat16, torch.float32, torch.float64


def _on_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


needs_gfx950 = pytest.mark.skipif(
    not _on_gfx950(), reason="the stock Muse kernels run on gfx950 (MI355X)"
)

# ---------------------------------------------------------------- [CPU] fma32


def _exact_fma(a: float, b: float, c: float) -> float:
    """fp32 RNE of the exact ``a b + c`` (fp32 operands): the nearest fp32 by
    exact rationals, ties to the even significand."""
    x = Fraction(a) * Fraction(b) + Fraction(c)
    r = torch.tensor(float(x), dtype=F64).float()  # within an ulp of x
    cands = [
        torch.nextafter(r, torch.tensor(-math.inf)),
        r,
        torch.nextafter(r, torch.tensor(math.inf)),
    ]
    best = min(
        cands,
        key=lambda v: (
            abs(Fraction(v.item()) - x),
            int(v.view(torch.int32).item()) & 1,  # a tie: the even significand
        ),
    )
    return best.item()


def test_fma32_is_correctly_rounded():
    rnd = random.Random(0)
    cases = []
    for _ in range(3000):
        a, b, c = (rnd.uniform(-4, 4) * 2.0 ** rnd.randint(-30, 30) for _ in "abc")
        cases.append((a, b, c))
    # ties after an fp64 sum: a b + c = an fp32 midpoint + 2^-30 of an fp64
    # ulp's worth, so fp64-then-fp32 double-rounds to the even neighbour
    ties = []
    for _ in range(500):
        k, e = rnd.randint(15, 20), rnd.randint(-60, 60)
        sign = rnd.choice([1.0, -1.0])
        odd = (1 + (2 * rnd.randint(0, 2**22 - 1) + 1) * 2.0**-23) * 2.0**e
        ties.append((-sign * 2.0 ** (e - 24) * (1 - 2.0**-k), 1 + 2.0**-k, sign * odd))
    cases += ties
    # fp32 operands (the generators draw doubles)
    cases = [tuple(torch.tensor(v, dtype=F32).item() for v in abc) for abc in cases]
    a, b, c = (torch.tensor(col, dtype=F32) for col in zip(*cases))
    want = torch.tensor([_exact_fma(*abc) for abc in cases], dtype=F32)
    assert torch.equal(ref.fma32(a, b, c), want)
    # the ties are real: an fp64 sum rounded to fp32 is wrong on every one
    naive = (a.double() * b.double() + c.double()).float()
    assert (naive[-len(ties) :] != want[-len(ties) :]).all()
    # broadcasting, non-finite
    assert ref.fma32(torch.tensor([math.inf, 1.0]), 2.0, 1.0).tolist() == [
        math.inf,
        3.0,
    ]
    assert math.isnan(ref.fma32(math.nan, 1.0, 0.0).item())


# ---------------------------------------------------------------- [CPU] front


def _heads(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 64, generator=g) * torch.exp(torch.randn(n, 1, generator=g))
    return x.to(BF)


def _plain_norm_rope(heads, cs):
    """fp64 QK-norm and GPT-J RoPE (pairs 2i, 2i + 1 of the head)."""
    x = heads.double()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + spec.RMS_EPS)
    if cs is None:
        return x
    c, s = cs[:, :32].double(), cs[:, 32:].double()
    xe, xo = x[:, 0::2], x[:, 1::2]
    out = torch.empty_like(x)
    out[:, 0::2] = xe * c - xo * s
    out[:, 1::2] = xo * c + xe * s
    return out


@pytest.mark.parametrize("rope", [False, True])
def test_qk_norm_rope_is_the_norm_and_rotation(rope):
    n = 20000
    heads = _heads(n, 1)
    table = ref.rope_table(4096)
    pos = torch.randint(0, 4096, (n,), generator=torch.Generator().manual_seed(2))
    cs = table[pos] if rope else None
    got = ref.qk_norm_rope(heads, cs)
    assert got.dtype == BF and got.shape == heads.shape
    ref.check_ulps(got, _plain_norm_rope(heads, cs), frac=0.999, max_ulps=1)
    # position 0 rotates nothing: cos 1, sin 0
    zero = ref.qk_norm_rope(heads[:64], table[torch.zeros(64, dtype=torch.long)])
    assert torch.equal(zero, ref.qk_norm_rope(heads[:64], None))


def test_rope_table_is_the_rotary_embeddings(default_vllm_config):
    """``rope_table`` equals vLLM's ``RotaryEmbedding`` cache for Muse's
    sliding layers (head 64, rotary 64, GPT-J, theta 5e5, fp32)."""
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

    rot = RotaryEmbedding(64, 64, 4096, spec.ROPE_THETA, False, torch.float32)
    torch.testing.assert_close(
        rot.cos_sin_cache.float(), ref.rope_table(4096), rtol=0, atol=2e-6
    )


def test_front_and_kv_insert():
    S = 5
    g = torch.Generator().manual_seed(3)
    qkv = torch.randn(S, spec.QKV_ROWS, generator=g).to(BF)
    qkv[3] = math.nan
    pad = torch.tensor([False, False, False, True, False])
    positions = torch.tensor([7, 2**40, -5, 9, 0])  # out of range: clamped
    table = ref.rope_table(64)
    out = ref.front(qkv, positions, table, pad)
    assert (out[3] == 0).all()
    live = ~pad
    # v and o_gate verbatim; q and k heads through qk_norm_rope
    assert torch.equal(out[live, spec.ROW_V :], qkv[live, spec.ROW_V :])
    t = 1  # position 2^40 -> the table's last row
    heads = qkv[t, : spec.ROW_V].view(-1, 64)
    cs = table[63].expand(heads.shape[0], 64)
    assert torch.equal(out[t, : spec.ROW_V], ref.qk_norm_rope(heads, cs).view(-1))
    # a pad row's own K / V (K1 inserts them when the slot is >= 0): position 0
    pad1 = torch.tensor([False, True, False])
    keep = ref.front(qkv[:3], positions[:3], table, pad1, zero_pad=False)
    at0 = ref.qk_norm_rope(heads, table[0].expand(heads.shape[0], 64))
    assert torch.equal(keep[1, : spec.ROW_V], at0.view(-1))
    assert torch.equal(keep[1, spec.ROW_V :], qkv[1, spec.ROW_V :])
    assert torch.equal(
        keep[~pad1], ref.front(qkv[:3], positions[:3], table, pad1)[~pad1]
    )
    nope = ref.front(qkv, positions, None, pad)
    assert torch.equal(nope[live, spec.ROW_V :], qkv[live, spec.ROW_V :])
    # the insert: slots >= 0 inside the cache only
    kv = torch.zeros(4, spec.KV_HEADS, spec.KV_BLOCK, spec.KV_C, dtype=BF)
    slots = torch.tensor([17, -1, 63, -1, 64])  # 64: block 4 of 4 blocks
    ref.kv_insert(kv, out, slots)
    assert torch.equal(kv[1, 1, 1, :64], out[0, spec.ROW_K + 64 : spec.ROW_V])
    assert torch.equal(kv[1, 0, 1, 64:], out[0, spec.ROW_V : spec.ROW_V + 64])
    assert torch.equal(kv[3, 0, 15, :64], out[2, spec.ROW_K : spec.ROW_K + 64])
    written = (kv != 0).any(-1)
    assert int(written.sum()) == 2 * spec.KV_HEADS


def test_ffn_boundary():
    S, H = 4, spec.HIDDEN
    g = torch.Generator().manual_seed(4)
    branch = torch.randn(S, H, generator=g).to(BF)
    stream = torch.randn(S, H, generator=g)
    w_post = (torch.randn(H, generator=g) * 0.1).to(BF)
    w_pre = (torch.randn(H, generator=g) * 0.1).to(BF)
    alpha, beta = torch.rand(H, generator=g) + 0.5, torch.rand(H, generator=g)
    branch[2] = math.nan
    pad = torch.tensor([False, False, True, False])
    stream_out, attn_in = ref.ffn_boundary(
        branch, stream, w_post, alpha, beta, w_pre, pad
    )
    assert stream_out.dtype == F32 and attn_in.dtype == BF
    assert (stream_out[2] == 0).all() and (attn_in[2] == 0).all()
    live = ~pad
    x = branch.double() * (w_post.double() + 1)
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + spec.POST_FFN_EPS)
    n = alpha.double() * stream.double() + beta.double() * y
    z = n * torch.rsqrt(n.square().mean(-1, keepdim=True) + spec.RMS_EPS)
    z = z * (w_pre.double() + 1)
    tol = ref.stream_tolerance(branch, stream, w_post, alpha, beta)
    assert ((stream_out.double() - n).abs()[live] <= tol.double()[live]).all()
    ref.check_ulps(attn_in[live], z[live], max_ulps=1)


# ---------------------------------------------------------------- [CPU] gate


def test_gated_norm():
    S = 64
    g = torch.Generator().manual_seed(5)
    attn = (torch.randn(S, spec.CORE, generator=g) * 3).to(BF)
    attn[1] *= 3e-3  # mean x^2 ~ eps: the epsilon bites
    og = (torch.randn(S, spec.CORE, generator=g) * 4).to(BF)
    og[2, :5] = torch.tensor([-100.0, -88.0, 0.0, 88.0, 100.0])
    attn[7] = math.nan
    pad = torch.zeros(S, dtype=torch.bool)
    pad[7] = True
    got = ref.gated_norm(attn, og, pad)
    assert got.shape == attn.shape and got.dtype == BF and (got[7] == 0).all()
    x = attn.double().view(S, 16, 64)
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + spec.RMS_EPS)
    y = (y * torch.sigmoid(og.double().view(S, 16, 64))).view(S, -1)
    ref.check_ulps(got[~pad], y[~pad], frac=0.999, max_ulps=1)
    assert torch.isfinite(got).all()


# ---------------------------------------------------------------- [CPU] core


def _dense(q, keys, vals, scale):
    """fp64 attention of q [16, 64] over keys / values [n, 2, 64]."""
    out = torch.empty(16, 64, dtype=F64)
    for h in range(16):
        k, v = keys[:, h // 8].double(), vals[:, h // 8].double()
        p = torch.softmax(k @ q[h].double() * scale, 0)
        out[h] = p @ v
    return out


@pytest.mark.parametrize("layout", ["LBHNC", "LHBNC"])
@pytest.mark.parametrize("window", [False, True])
def test_gqa_decode_reads_only_its_keys(layout, window):
    """Dense attention over the row's keys; NaN in every byte the row must not
    read (the null block, pages below the window, the window's first page head
    and the last page's tail, stale table entries) changes nothing."""
    gen = torch.Generator().manual_seed(6)
    lens = [1, 16, 17, 2047, 2049, 2065, 4103, 0]
    pad = torch.tensor([False] * 7 + [True])
    # blocks 1 .. 1198: 1199 is only ever a stale entry
    pages = ref.assign_pages(lens, 1199, extra=1, generator=gen)
    kv = ref.kv_cache(1200, layout=layout, generator=gen)
    if layout == "LHBNC":
        assert kv.stride(1) > kv.stride(0)
    bt = torch.full((len(lens), 300), 1199, dtype=torch.int32)  # stale tail
    for r, p in enumerate(pages):
        bt[r, : len(p)] = torch.tensor(p)
    q = torch.randn(len(lens), spec.HEADS * 64, generator=gen).to(BF)
    seq = torch.tensor(lens, dtype=torch.int32)
    clean = ref.gqa_decode(q, kv, bt, seq, window=window, pad=pad)
    assert (clean[7] == 0).all() and torch.isfinite(clean).all()
    for t, L in enumerate(lens[:7]):
        lo = max(0, L - spec.WINDOW) if window else 0
        k = torch.arange(lo, L)
        rows = kv[bt[t].long()[k // 16], :, k % 16]
        want = _dense(q[t].view(16, 64), rows[..., :64], rows[..., 64:], spec.QK_SCALE)
        torch.testing.assert_close(clean[t], want, rtol=1e-12, atol=1e-12)
    kv[1199] = math.nan  # the stale entries' block
    ref.poison_unread(kv, bt, seq, window=window, rows=range(7))
    assert torch.equal(ref.gqa_decode(q, kv, bt, seq, window=window, pad=pad), clean)


def test_kv_cache_layouts_hold_the_same_rows():
    a = ref.kv_cache(5, generator=torch.Generator().manual_seed(7))
    b = ref.kv_cache(5, layout="LHBNC", generator=torch.Generator().manual_seed(7))
    assert a.is_contiguous() and not b.is_contiguous()
    assert a.shape == b.shape == (5, spec.KV_HEADS, spec.KV_BLOCK, spec.KV_C)
    assert a[0].isnan().all() and b[0].isnan().all()
    with pytest.raises(ValueError):
        ref.kv_cache(5, layout="NHD")
    pages = ref.assign_pages([1, 33, 16], 10, extra=1)
    flat = [p for ps in pages for p in ps]
    assert [len(p) for p in pages] == [2, 4, 2] and 0 not in flat
    assert len(set(flat)) == len(flat)


# ---------------------------------------------------------------- [CPU] MRV2


def test_fake_mrv2_writes_in_place():
    groups = {"full": ["l1.attn", "l5.attn"], "sw0": ["l2.attn"]}
    m = ref.FakeMrv2(groups, {"sw0"}, max_blocks=200)
    ptrs = {g: bt.data_ptr() for g, bt in m.block_tables.items()} | {
        "slots": m.slot_mappings.data_ptr(),
        "seq": m.seq_lens.data_ptr(),
    }
    cap = m.capture(4)
    assert (cap.pad == 1).all() and (cap.slot_mapping["l2.attn"] == -1).all()
    assert (cap.attn_metadata["l1.attn"].seq_lens == 1).all()
    lens = [2100, 5]
    pages = {
        "full": ref.assign_pages(
            lens, 1000, generator=torch.Generator().manual_seed(8)
        ),
        "sw0": ref.assign_pages(lens, 1000, generator=torch.Generator().manual_seed(9)),
    }
    st = m.step(lens, pages, 4, stale=999)
    for g, bt in m.block_tables.items():
        assert bt.data_ptr() == ptrs[g]
    assert m.slot_mappings.data_ptr() == ptrs["slots"]
    md = st.attn_metadata["l2.attn"]
    assert md.block_table.data_ptr() == ptrs["sw0"] and md.block_table.shape[0] == 4
    assert md.seq_lens.tolist() == [2100, 5, 0, 0]
    assert st.attn_metadata["l5.attn"].block_table is not md.block_table
    # the sliding group's entries wholly below the window are nulled
    null = (2100 - 1 - 2047) // 16
    assert null == 3 and (m.block_tables["sw0"][0, :null] == 0).all()
    assert m.block_tables["sw0"][0, null].item() == pages["sw0"][0][null]
    assert m.block_tables["full"][0, :132].tolist() == pages["full"][0]
    assert (m.block_tables["full"][0, 132:] == 999).all()  # stale
    assert (m.block_tables["full"][2:] == 0).all()  # padded rows
    # slots, positions, pad flags
    assert st.positions.tolist() == [2099, 4, 0, 0]
    assert st.pad.tolist() == [False, False, True, True]
    want = pages["sw0"][0][2099 // 16] * 16 + 2099 % 16
    assert st.slot_mapping["l2.attn"].tolist() == [
        want,
        pages["sw0"][1][0] * 16 + 4,
        -1,
        -1,
    ]
    # what the split reads is never nulled
    _, b0, _, _, _ = spec.split_plan(2100, True, 16)
    assert (m.block_tables["sw0"][0, b0:132] != 0).all()
    # a DSpark verify step: 8 rows of one request
    st = m.step(
        [40],
        pages={"full": [list(range(1, 4))], "sw0": [list(range(5, 8))]},
        width=8,
        q=8,
    )
    assert st.positions.tolist() == list(range(32, 40)) and not st.pad.any()
    assert st.slot_mapping["l1.attn"].tolist() == [3 * 16 + i for i in range(8)]
    assert st.attn_metadata["l1.attn"].num_decodes == 1


# ---------------------------------------------------------------- [1GPU]


@needs_gfx950
@pytest.mark.parametrize("rope", [False, True])
def test_front_reference_matches_the_fused_insert(rope):
    """``front`` + ``kv_insert`` on the GPU vs the stock fused insert on the
    same bf16 rows (HEADS_PER_GROUP 1 at 8 rows): <= 1 bf16 ulp on every q / k
    element and equal on >= 99.9 % (the reference's rsqrt is torch's), cache
    rows likewise, v and o_gate bytes equal."""
    from vllm import _custom_ops as ops

    dev = torch.device("cuda")
    S = 8
    g = torch.Generator().manual_seed(10)
    qkv = (torch.randn(S, spec.QKV_ROWS, generator=g) * 2).to(BF).to(dev)
    positions = torch.randint(0, 4096, (S,), generator=g).to(dev)
    table = ref.rope_table(4096, device=dev) if rope else None
    kv = torch.zeros(64, spec.KV_HEADS, spec.KV_BLOCK, spec.KV_C, dtype=BF, device=dev)
    slots = torch.randperm(64 * 16, generator=g)[:S].to(dev)
    stock = qkv.clone()
    ops.fused_muse_spark_qknorm_rope_kv_insert(
        stock,
        positions,
        spec.HEADS,
        spec.KV_HEADS,
        spec.HEAD_DIM,
        spec.RMS_EPS,
        table,
        64 if rope else 0,
        slots,
        kv,
        spec.KV_BLOCK,
    )
    got = ref.front(qkv, positions, table)
    ref.check_ulps(got, stock, max_ulps=1)
    assert torch.equal(got[:, spec.ROW_V :], stock[:, spec.ROW_V :])
    mine = ref.kv_insert(torch.zeros_like(kv), got, slots.cpu())
    ref.check_ulps(mine, kv, max_ulps=1)
    # every rounding point is the stock one's (only an rsqrt may differ)
    same = (got == stock).float().mean().item()
    print(f"rope={rope}: front == stock on {same:.6f}")
    assert same >= 0.999 and (mine == kv).float().mean().item() >= 0.999


@needs_gfx950
def test_gate_reference_matches_rms_norm_gated():
    from vllm.third_party.flash_linear_attention.ops.kda import (
        rms_norm_gated,
    )

    dev = torch.device("cuda")
    S = 8
    g = torch.Generator().manual_seed(11)
    qkvo = (torch.randn(S, spec.QKV_ROWS, generator=g) * 3).to(BF).to(dev)
    attn = (torch.randn(S, spec.CORE, generator=g) * 2).to(BF).to(dev)
    attn[1] *= 3e-3
    og = qkvo[:, spec.ROW_G :]
    # rms_norm_gated writes y over x (kda.py: ``y = x`` when
    # out_dtype is None): on a copy, the inputs are reused below
    stock = rms_norm_gated(
        attn.clone().view(S, 16, 64),
        og.reshape(S, 16, 64),
        None,
        None,
        "sigmoid",
        eps=spec.RMS_EPS,
    ).view(S, -1)
    got = ref.gated_norm(attn, og)
    ref.check_ulps(got, stock, max_ulps=1)
    # every rounding point is the stock one's (only a sqrt may differ)
    same = (got == stock).float().mean().item()
    print(f"gate == stock on {same:.6f}")
    assert same >= 0.999
