# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The contract every Muse Spark MonoKernel package shares: shapes, numerics,
rows, grid, the K2 placement, mailbox slots, the build key, the argument list
and the scratch / peer layouts.

Pure Python (``dataclasses`` only): the runtime sizes its buffers and the CPU
tests check the layouts without torch.cuda or FlyDSL. ``pair_layout`` is the
K3 helper (``mono.common.plan.pair_layout``) re-implemented here for that
reason; ``test_spec.py`` keeps the two identical.

Every count is per TP rank of Muse Spark Open at TP8. Every scratch and peer
offset is sized by ``MAX_TOKENS`` rows (``MAX_PICKS`` picks), never by a
launch's width, so every width of a step shares one layout and one parity
half.
"""

from dataclasses import dataclass

# --------------------------------------------------------------------------
# The model (Muse Spark Open at TP8)

TP = 8
HIDDEN = 8192  # H: the residual stream
MOE_HIDDEN = 4096  # the pooled width the routed experts see
INTER = 512  # an expert's intermediate a rank (moe_intermediate_size / TP)
EXPERTS = 256
TOPK = 8
EXPERTS_RANK = EXPERTS // TP  # 32 router rows a rank
HEADS = 16  # query heads a rank (128 / TP)
KV_HEADS = 2  # KV heads a rank (16 / TP)
HEAD_DIM = 64
CORE = HEADS * HEAD_DIM  # 1024: o_proj's K a rank

# runtime.text_shapes(text_config) must equal MUSE_SHAPES, field by field.
MUSE_SHAPE_FIELDS = (
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "num_local_experts",
    "num_experts_per_tok",
    "moe_hidden_size",
    "moe_intermediate_size",
    "num_shared_experts",
    "sliding_window",
)
MUSE_SHAPES = (8192, 128, 16, 64, 256, 8, 4096, 4096, 0, 2048)

# --------------------------------------------------------------------------
# Numerics (refuse_model asserts the loaded modules carry these)

POST_ATTN_EPS = 1e-8  # weightless post_attn_norm
PRE_FFN_EPS = 1e-5
PRE_EXP_EPS = 1e-5
POST_EXP_EPS = 1e-8
GAIN = 1.0  # fused_gain_center of pre_ffn / pre_expert / post_expert norms
ROUTE_SCALE = 1.0
ROUTE_EPS = 1e-15  # router_score_division_eps
AMAX_FLOOR = 1e-10  # MXFP4 RoundUp: amax = max(amax, AMAX_FLOOR)
LOG2E = 1.4426950408889634

# --------------------------------------------------------------------------
# Rows

MAX_ROWS = 8  # rows a served step has at most (the K3 thread maps)
MAX_TOKENS = 8  # rows scratch and peer memory are sized for
MAX_PICKS = MAX_TOKENS * TOPK  # 64 (row, k) picks
MAX_U = MAX_PICKS  # experts in a step's union
SERVED_Q = (1, MAX_ROWS)  # rows a request decodes: plain decode, DSpark K = 7

# --------------------------------------------------------------------------
# Grid: one CTA per CU, all resident

BLOCKS = 256
THREADS = 512
WAVES = THREADS // 64  # 8
LDS_BYTES = 160 * 1024  # a CU's LDS: one CTA's whole
VGPR_MAX = 256  # arch + acc VGPRs a lane at 2 waves a SIMD
ROWS = 16  # output rows of a GEMV task (one MFMA tile)
COLS = 128  # columns of a seam / gather CTA

# --------------------------------------------------------------------------
# K2 placement (static; section 5.2 of the design)

O_TASKS = HIDDEN // ROWS  # 512 o_proj tasks: CTA b runs b and b + BLOCKS
NA = HIDDEN // COLS  # 64 seam CTAs [0, NA): 128 columns each
R0 = 64  # router CTAs [R0, R0 + R_TASKS): one expert row each
R_TASKS = EXPERTS_RANK  # 32
P_GROUPS = MOE_HIDDEN // TP // ROWS  # 32 pooler row groups a rank
P_PARTS = 4  # K quarters of a pooler row group
P_K = HIDDEN // P_PARTS  # 2048
P0 = 96  # pooler CTAs [P0, P0 + P_TASKS): g = t // P_PARTS, q = t % P_PARTS
P_TASKS = P_GROUPS * P_PARTS  # 128
UG_GROUPS = INTER // 32  # 16 ug tasks a union slot (32 columns each)
W13_K = MOE_HIDDEN  # 4096
W13_STEPS = W13_K // 128  # 32 (16 a K half)
W2_STEPS = INTER // 128  # 4
DOWN_TASKS = MOE_HIDDEN // ROWS  # 256 == BLOCKS: CTA b owns rows 16b .. 16b + 16
DOWN_BATCH = 4  # slots of w2 in flight a wave
OWN_COLS = MOE_HIDDEN // TP  # 512 pooled columns a rank owns (reduce-scatter)
RED_TASKS = OWN_COLS // ROWS  # 32 owner CTAs [32 rank, 32 rank + 32)
U_GROUPS = HIDDEN // TP // ROWS  # 64 unpooler row groups a rank
U_PARTS = 4  # K quarters of an unpooler row group
U_K = MOE_HIDDEN // U_PARTS  # 1024
U_TASKS = U_GROUPS * U_PARTS  # 256: CTA b: g = b // U_PARTS, q = b % U_PARTS
G_TASKS = HIDDEN // COLS  # 64 gather CTAs [0, G_TASKS)

# MXFP4 activation rows (a1): words a pooled row
XQ_WORDS = MOE_HIDDEN // 8  # 512 int32 words of packed fp4
XSC_WORDS = MOE_HIDDEN // 32  # 128 e8m0 codes, one a word in LDS / xqd

# LDS route table (common/route_lds.RouteLds): 32-bit words a field
ROUTE_LDS = dict(
    route=MAX_TOKENS * 16,  # t*16 + k: id; t*16 + 8 + k: fp32 bits of the weight
    flag=EXPERTS,
    expert=MAX_U,
    kof=MAX_U * 8,
    scan=WAVES,
    tot=MAX_TOKENS,
)

# --------------------------------------------------------------------------
# Mailbox tags: tag = (epoch & EPOCH_MASK) * TAG_SLOTS + slot + 1

EPOCH_MASK = (1 << 22) - 1  # kernels use e & EPOCH_MASK: the int32 epoch may wrap
TAG_SLOTS = 256
K1_SLOT0 = 0  # Phase B's K1 launches
K2_SLOT0 = 128  # K2 launches


def k1_slot(layer_idx: int) -> int:
    """The tag slot of a layer's K1 launch (Phase B)."""
    if not 0 <= layer_idx < K2_SLOT0 - K1_SLOT0:
        raise ValueError(f"layer {layer_idx} has no K1 slot")
    return K1_SLOT0 + layer_idx


def k2_slot(layer_idx: int) -> int:
    """The tag slot of a layer's K2 launch."""
    if not 0 <= layer_idx < TAG_SLOTS - K2_SLOT0:
        raise ValueError(f"layer {layer_idx} has no K2 slot")
    return K2_SLOT0 + layer_idx


def step_tag(epoch: int, slot: int) -> int:
    """The host mirror of ``sync.step_tag``: the tag every pair a launch of
    ``slot`` writes at ``epoch`` carries (two's complement ``&``, so a wrapped,
    negative int32 epoch gives the kernel's tag too). Never 0: a zeroed pair is
    never current."""
    return (epoch & EPOCH_MASK) * TAG_SLOTS + slot + 1


assert P0 + P_TASKS <= BLOCKS and R0 + R_TASKS <= P0 and NA <= R0
assert DOWN_TASKS == BLOCKS and U_TASKS == BLOCKS and O_TASKS == 2 * BLOCKS
assert RED_TASKS * TP == DOWN_TASKS and MAX_ROWS <= MAX_TOKENS <= 8
assert G_TASKS <= BLOCKS and MAX_U <= EXPERTS and TOPK <= 8
assert step_tag(EPOCH_MASK, TAG_SLOTS - 1) < 2**31  # a tag is a positive int32

# --------------------------------------------------------------------------
# The build key and the argument list


@dataclass(frozen=True)
class K2Build:
    """Every K2 build parameter, scalars only (``plan.key_tuple``): two builds
    differing in any field never share a binary. ``rank`` is a runtime argument,
    never a field (every rank runs one binary)."""

    tokens: int  # S, the launch's rows: 1 .. MAX_ROWS
    debug_dump: bool = False  # tests: CTA 0 also stores its xq / xsc to "xqd"
    # The fault hook (sync.stall, tests only): rank fault_rank stalls fault_ticks
    # once, in the launch of slot fault_slot of the step at epoch fault_epoch.
    fault_rank: int = -1
    fault_epoch: int = 0
    fault_slot: int = 0
    fault_ticks: int = 0

    def __post_init__(self):
        for f in ("tokens", "fault_rank", "fault_epoch", "fault_slot", "fault_ticks"):
            v = getattr(self, f)
            if type(v) is not int:  # bool too: it would name the symbol "sTrue"
                raise TypeError(f"K2Build.{f} = {v!r}: an int")
        if type(self.debug_dump) is not bool:
            raise TypeError(f"K2Build.debug_dump = {self.debug_dump!r}: a bool")
        if not 1 <= self.tokens <= MAX_ROWS:
            raise ValueError(f"K2Build.tokens {self.tokens} (1 .. {MAX_ROWS})")
        if not -1 <= self.fault_rank < TP:
            raise ValueError(f"K2Build.fault_rank {self.fault_rank} (-1 .. {TP - 1})")
        if not self.faulty:
            # One build, one key: the fault fields of a build without the hook
            # would key a second binary under the same kernel name.
            if (self.fault_epoch, self.fault_slot, self.fault_ticks) != (0, 0, 0):
                raise ValueError("K2Build.fault_* set without fault_rank")
            return
        # sync.stall compares these with the int32 epoch word and the K2 slot
        # argument: an epoch past int32 or a slot no K2 launch has never fires.
        if not 0 <= self.fault_epoch < 2**31:
            raise ValueError(f"K2Build.fault_epoch {self.fault_epoch} (0 .. 2^31 - 1)")
        if not K2_SLOT0 <= self.fault_slot < TAG_SLOTS:
            raise ValueError(
                f"K2Build.fault_slot {self.fault_slot}: a K2 slot "
                f"({K2_SLOT0} .. {TAG_SLOTS - 1}, spec.k2_slot)"
            )
        if not 0 < self.fault_ticks < 2**63:
            raise ValueError(f"K2Build.fault_ticks {self.fault_ticks} (> 0)")

    @property
    def faulty(self) -> bool:
        """The build compiles ``sync.stall`` into its prologue."""
        return self.fault_rank >= 0

    def kernel_name(self) -> str:
        """The kernel symbol: one per build."""
        name = f"muse_mono_k2_s{self.tokens}"
        if self.faulty:
            name += (
                f"_f{self.fault_rank}_{self.fault_epoch}_{self.fault_slot}"
                f"_{self.fault_ticks}"
            )
        if self.debug_dump:
            name += "_dbg"
        return name


K2_ARGS = (
    "core",  # bf16 [S, CORE]: o_proj's input
    "stream",  # fp32 [S, HIDDEN]: the residual stream
    "pad",  # bool [>= S]: pad rows
    "w_o",
    "attn_alpha",
    "attn_beta",
    "w_preffn",
    "w_router",
    "bias",
    "w_pool",
    "w_preexp",
    "w13",
    "w13s",
    "w2",
    "w2s",
    "w_postexp",
    "w_unpool",
    "stream_out",  # fp32 [S, HIDDEN]
    "branch_out",  # bf16 [S, HIDDEN]
    "scratch",
    "peers",  # int64 [TP]: every rank's peer buffer past PEER_HEAD
    "rank",
    "epoch",
    "slot",
    "health",
    "alarm",
)
K2_INT32_ARGS = ("rank", "slot")  # every other argument is an Int64 device address

# --------------------------------------------------------------------------
# Scratch and peer layouts (section 5.10 of the design)

PAIR_BYTES = 8  # a (value, tag) int32 pair

# Mailbox regions, in layout order; then debug.region_ids order (wait records).
SCRATCH_REGIONS = (
    "xrdy",
    "ass",
    "bss",
    "ppart",
    "route",
    "inter",
    "isc",
    "pss",
    "upart",
)
PEER_REGIONS = ("ar", "rlog", "pool", "rsx", "sqx", "moe", "branch")
REGIONS = SCRATCH_REGIONS + PEER_REGIONS
# Plain (untagged) scratch regions, after the mailboxes.
XQD = "xqd"  # debug_dump builds only: CTA 0's xq / xsc words
XROW = "xrow"  # bf16 [MAX_TOKENS, HIDDEN]: ffn_in rows (seam -> router, pooler)
XROW_ALIGN = 16  # dwordx4 loads

_MT, _MP = MAX_TOKENS, MAX_PICKS
SCRATCH_PAIRS = (
    ("xrdy", NA),  # j: seam slice j published (flag)
    ("ass", NA * _MT),  # j*MT + t: fp32 sum of attn_out^2 over slice j
    ("bss", NA * _MT),  # j*MT + t: fp32 sum of new_stream^2 over slice j
    ("ppart", P_GROUPS * (P_PARTS - 1) * _MT * ROWS),  # ((g*3 + q-1)*MT + t)*16 + r
    ("route", _MT * 16),  # t*16 + k: id; t*16 + 8 + k: fp32 weight
    ("inter", _MP * 64),  # pick*64 + 4g + w: a2 fp4 words
    ("isc", _MP * UG_GROUPS),  # pick*16 + g: a2 e8m0 code
    ("pss", RED_TASKS * _MP),  # j*MP + pick: fp32 sum x^2 of 16 owner columns
    ("upart", U_GROUPS * (U_PARTS - 1) * _MT * ROWS),  # ((g*3 + q-1)*MT + t)*16 + r
)
# xqd is sized in pairs as the design's table has it (twice the words it holds).
XQD_PAIRS = _MT * (XQ_WORDS + XSC_WORDS)
XQD_ROW_WORDS = XQ_WORDS + XSC_WORDS  # a row's plain words: xq, then xsc
XROW_BYTES = _MT * HIDDEN * 2

# One parity half; pair indexes are in the reader's buffer.
PEER_PAIRS = (
    ("ar", TP * _MT * HIDDEN // 2),  # ((src*MT + t)*8192 + col)/2: bf16 partials
    ("rlog", TP * _MT * EXPERTS_RANK),  # (src*MT + t)*32 + i: fp32 logits
    ("pool", _MT * MOE_HIDDEN // 2),  # (t*4096 + col)/2: bf16 pooled
    ("rsx", TP * _MP * OWN_COLS // 2),  # ((src*MP + pick)*512 + cc)/2: bf16 rows
    ("sqx", TP * _MP),  # src*MP + pick: fp32 owner sum of squares
    ("moe", _MT * MOE_HIDDEN // 2),  # (t*4096 + col)/2: bf16 MoE output
    ("branch", _MT * HIDDEN // 2),  # (t*8192 + col)/2: bf16 unpooler output
)


def pair_layout(items, align: int = PAIR_BYTES, start: int = 0) -> dict:
    """``items`` (name, pairs) laid out in order: name -> (byte offset, bytes),
    each region from an ``align``-byte boundary at or past ``start``."""
    out, off = {}, start
    for name, pairs in items:
        off = (off + align - 1) // align * align
        out[name] = (off, pairs * PAIR_BYTES)
        off += pairs * PAIR_BYTES
    return out


def _end(layout: dict) -> int:
    return max(off + size for off, size in layout.values())


def scratch_layout() -> dict[str, tuple[int, int]]:
    """A rank's scratch (one buffer shared by every launch of a step; the tags
    tell launches apart): the mailboxes, then ``xqd``, then ``xrow``
    (16-byte aligned). name -> (byte offset, bytes)."""
    out = pair_layout(SCRATCH_PAIRS)
    out.update(pair_layout(((XQD, XQD_PAIRS),), start=_end(out)))
    xrow = (_end(out) + XROW_ALIGN - 1) // XROW_ALIGN * XROW_ALIGN
    out[XROW] = (xrow, XROW_BYTES)
    return out


def scratch_bytes() -> int:
    return _end(scratch_layout())


def peer_layout() -> dict[str, tuple[int, int]]:
    """One parity half of a rank's peer buffer: name -> (byte offset, bytes).
    The kernels see the buffer past ``abi.PEER_HEAD``; parity ``e & 1`` adds
    ``e & 1`` halves."""
    return pair_layout(PEER_PAIRS)


def peer_half_bytes() -> int:
    return _end(peer_layout())


def peer_bytes() -> int:
    """Both parity halves; ``abi.PEER_HEAD`` is not included."""
    return 2 * peer_half_bytes()


# --------------------------------------------------------------------------
# Pair indexes (section 5.10): pair units for put / put_words / poll; put_bf
# takes the element index 2 * pair. Plain integer arithmetic, so the host tests
# and the kernels' traced Int32 math read the same formulas.


def ar_pair(src, t, col):
    return ((src * _MT + t) * HIDDEN + col) // 2


def rlog_pair(src, t, i):
    return (src * _MT + t) * EXPERTS_RANK + i


def pool_pair(t, col):
    return (t * MOE_HIDDEN + col) // 2


def rsx_pair(src, pick, cc):
    return ((src * _MP + pick) * OWN_COLS + cc) // 2


def sqx_pair(src, pick):
    return src * _MP + pick


def moe_pair(t, col):
    return (t * MOE_HIDDEN + col) // 2


def branch_pair(t, col):
    return (t * HIDDEN + col) // 2


def xrdy_pair(j):
    return j


def ss_pair(j, t):
    """``ass`` and ``bss``."""
    return j * _MT + t


def ppart_pair(g, q, t, r):
    return ((g * (P_PARTS - 1) + q - 1) * _MT + t) * ROWS + r


def route_id_pair(t, k):
    return t * 16 + k


def route_weight_pair(t, k):
    return t * 16 + 8 + k


def inter_pair(pick, g, w):
    return pick * 64 + 4 * g + w


def isc_pair(pick, g):
    return pick * UG_GROUPS + g


def pss_pair(j, pick):
    return j * _MP + pick


def upart_pair(g, q, t, r):
    return ((g * (U_PARTS - 1) + q - 1) * _MT + t) * ROWS + r


# Plain (untagged) regions: word / element indexes from the region's base.


def xqd_xq_word(t, w):
    """``xqd`` int32 word of row t's packed fp4 word w (0 .. XQ_WORDS): CTA 0's
    LDS ``xq[t*512 + w]`` (debug_dump builds)."""
    return t * XQD_ROW_WORDS + w


def xqd_xsc_word(t, kb):
    """``xqd`` int32 word of row t's e8m0 code kb (0 .. XSC_WORDS), the code in
    the low byte: CTA 0's LDS ``xsc[t*128 + kb]``."""
    return t * XQD_ROW_WORDS + XQ_WORDS + kb


def xrow_elem(t, col):
    """``xrow`` bf16 element of row t, column col."""
    return t * HIDDEN + col
