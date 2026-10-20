# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MXFP4 expert layouts the mono kernels read, against aiter's own
shufflers on CPU, at Muse Spark's per-rank shapes (w13 [256, 1024, 4096 fp4],
w2 [256, 4096, 512 fp4], both scales ``e8m0_shuffle``d, weights
``shuffle_weight(layout=(16, 16))``), and the RoundUp E8M0 rule of the
activation quant against aiter's reference.

The address helpers are plain integer arithmetic (``mx.scale_index``,
``mx.w13_scale_word``), evaluated here on tensors; ``fp4_tile_load``'s dword
address is its documented formula. No GPU.
"""

import pytest
import torch

pytest.importorskip("aiter")

from aiter.ops.shuffle import shuffle_weight  # noqa: E402
from aiter.utility.fp4_utils import e8m0_shuffle  # noqa: E402

from vllm.models.muse_spark.amd.specialized_decode.mono import spec  # noqa: E402
from vllm.models.muse_spark.amd.specialized_decode.mono.common import mx  # noqa: E402

W13_ROWS = 2 * spec.INTER  # gate 512 | up 512
W13_COLS = spec.W13_K // 32  # 128 scale codes a row
W2_ROWS = spec.MOE_HIDDEN  # 4096
W2_COLS = spec.INTER // 32  # 16


def _shuffled_scales(experts: int, rows: int, cols: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(
        0, 256, (experts * rows, cols), dtype=torch.uint8, generator=g
    )
    src = codes
    if hasattr(torch, "float8_e8m0fnu"):
        src = codes.view(torch.float8_e8m0fnu)
    sh = e8m0_shuffle(src).view(torch.uint8).reshape(-1)
    assert sh.numel() == codes.numel()
    return codes, sh


@pytest.mark.parametrize(
    "rows,cols", [(W13_ROWS, W13_COLS), (W2_ROWS, W2_COLS)], ids=["w13s", "w2s"]
)
def test_scale_index_is_e8m0_shuffle(rows, cols):
    """Every scale byte of two experts (a 32-row block never straddles two)."""
    experts = 2
    codes, sh = _shuffled_scales(experts, rows, cols, seed=rows + cols)
    r = torch.arange(experts * rows).view(-1, 1).expand(-1, cols)
    c = torch.arange(cols).view(1, -1).expand(experts * rows, -1)
    idx = mx.scale_index(r, c, cols)
    assert torch.equal(sh[idx], codes)
    assert torch.equal(torch.sort(idx.reshape(-1)).values, torch.arange(codes.numel()))
    assert rows % 32 == 0


def _k3_w13_scale_word(e, rg, st, lane, ri=spec.INTER, k=spec.W13_K):
    """K3's ``w13_scale_word`` (pr60683 stages/moe.py) at Muse's shapes."""
    n1, k1 = rg // 2, st // 2
    return (
        e * (2 * ri * (k // 32) // 4)
        + ((n1 * (k // 128 // 2) + k1) * 4 + lane // 16) * 16
        + lane % 16
    )


def test_w13_scale_word():
    """``w13_scale_word`` names the dword of the code that a ug lane's MFMA
    takes for (expert, row group, step), at byte (st % 2) 2 + rg % 2, and is
    K3's formula at Muse's shapes."""
    experts = 3
    codes, sh = _shuffled_scales(experts, W13_ROWS, W13_COLS, seed=7)
    words = sh.view(torch.int32)
    e, rg, st, lane = torch.meshgrid(
        torch.arange(experts),
        torch.arange(W13_ROWS // 16),
        torch.arange(spec.W13_STEPS),
        torch.arange(64),
        indexing="ij",
    )
    word = mx.w13_scale_word(e, rg, st, lane)
    assert torch.equal(word, _k3_w13_scale_word(e, rg, st, lane))
    byte = (words[word] >> ((st % 2) * 16 + (rg % 2) * 8)) & 0xFF
    row = e * W13_ROWS + rg * 16 + lane % 16
    col = 4 * st + lane // 16
    assert torch.equal(byte.to(torch.uint8), codes[row, col])
    # gate (row groups 0..31) and up (32..63) never share a 32-row scale block
    assert spec.INTER // 16 % 2 == 0


def _fp4_tile_dword(e, rg, k, st, lane, rows):
    """``mx.fp4_tile_load``'s dword: base + rg (2 k) + st 256 + lane 4, with an
    expert's base e rows k / 8 dwords."""
    return e * rows * (k // 8) + rg * (2 * k) + st * 256 + lane * 4


@pytest.mark.parametrize(
    "rows,k,base",
    [(W13_ROWS, spec.W13_K, 524288), (W2_ROWS, spec.INTER, 262144)],
    ids=["w13", "w2"],
)
def test_fp4_tile_addressing(rows, k, base):
    """A lane's 16 B at a step: row rg 16 + lane % 16, K bytes 64 st + 16
    (lane / 16) of the unshuffled [rows, k / 2] bytes (2 experts, every row
    group, step and lane)."""
    assert rows * k // 8 == base  # the expert stride experts.py uses
    experts = 2
    g = torch.Generator().manual_seed(k)
    w = torch.randint(0, 256, (experts, rows, k // 2), dtype=torch.uint8, generator=g)
    ws = shuffle_weight(w, layout=(16, 16)).contiguous().view(-1)
    e, rg, st, lane = torch.meshgrid(
        torch.arange(experts),
        torch.arange(rows // 16),
        torch.arange(k // 128),
        torch.arange(64),
        indexing="ij",
    )
    dw = _fp4_tile_dword(e, rg, k, st, lane, rows).reshape(-1)
    got = ws.view(-1, 4)[dw.view(-1, 1) + torch.arange(4)].reshape(-1, 16)
    row = (rg * 16 + lane % 16).reshape(-1)
    b0 = (st * 64 + (lane // 16) * 16).reshape(-1)
    want = w[
        e.reshape(-1).view(-1, 1), row.view(-1, 1), b0.view(-1, 1) + torch.arange(16)
    ]
    assert torch.equal(got, want)


def code_roundup(amax: torch.Tensor) -> torch.Tensor:
    """The RoundUp E8M0 code of aiter's HIP ``fused_mx_quant_moe_sort``
    (``mx_quant_utils.h``): amax floored at 1e-10, times f32(1/6), the biased
    exponent rounded up. The torch reference the quant probes use."""
    amax = torch.clamp(amax.float(), min=spec.AMAX_FLOOR)
    u = (amax * torch.tensor(1.0 / 6.0, dtype=torch.float32)).view(torch.int32)
    e = (u >> 23) & 0xFF
    return e + (((u & 0x7FFFFF) != 0) & (e < 0xFF)).int()


def k3_e8m0_roundup(amax: torch.Tensor) -> torch.Tensor:
    """``mx.e8m0_roundup``'s integer recipe on the host: ((bits(amax f32(1/6))
    + 0x7FFFFF) >> 23) & 0xFF, at most 254 (amax already floored)."""
    wi = (amax.float() * torch.tensor(1.0 / 6.0, dtype=torch.float32)).view(torch.int32)
    b = ((wi.long() & 0xFFFFFFFF) + 0x7FFFFF) >> 23
    return (b & 0xFF).clamp(max=254).int()


def test_roundup_rule():
    """RoundUp, not even: aiter's reference, the HIP rule and the kernel's
    integer recipe agree on finite inputs (incl. 0 and the floor)."""
    from aiter.utility.fp4_utils import f32_to_mx_e8m0_scale
    from aiter.utility.mx_types import MxScaleRoundModeInt

    g = torch.Generator().manual_seed(0)
    x = torch.randn(1_000_000, generator=g) * torch.exp2(
        torch.randint(-20, 20, (1_000_000,), generator=g).float()
    )
    a = torch.cat(
        [
            x.abs(),
            torch.tensor([0.0, 1e-12, 1e-10, 6.0, 0.75, 3.0, 6.0 * 2.0**100]),
            torch.exp2(torch.arange(-60, 60).float()) * 6,
        ]
    )
    floored = torch.clamp(a, min=spec.AMAX_FLOOR)
    ref = f32_to_mx_e8m0_scale(floored, mode=MxScaleRoundModeInt.RoundUp)
    ref = ref.view(torch.uint8).int()
    hip = code_roundup(a)
    assert torch.equal(ref, hip)
    assert torch.equal(k3_e8m0_roundup(floored), hip)
    # the floor matters for the code only: an all-zero group gets 92, not 0
    assert hip[x.numel()].item() == 92 and k3_e8m0_roundup(a[x.numel() :])[0] == 0
    # exact powers of two times 6 keep their exponent (no bump) above the floor
    p = torch.exp2(torch.arange(-60, 60).float())
    above = p * 6 >= spec.AMAX_FLOOR
    want = (p.view(torch.int32) >> 23) & 0xFF
    assert torch.equal(hip[-120:][above], want[above])
    assert (hip[-120:][~above] == 92).all()
