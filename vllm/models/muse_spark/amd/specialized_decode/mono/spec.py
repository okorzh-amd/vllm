# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The contract every Muse Spark MonoKernel package shares: shapes, numerics,
rows, grid, the K1 / K2 placements, mailbox slots, the build keys, the argument
lists and the scratch / peer layouts.

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

# Phases B and C (DESIGN_BC section 1.1): the attention front and core.
RMS_EPS = 1e-5  # pre_qkv_proj_norm, the weightless qk_norm and o_norm
POST_FFN_EPS = 1e-8  # the previous layer's post_ffn_norm (weight first)
QK_SCALE_FACTOR = 5.4730064863875  # config.qk_scale_factor
QK_SCALE = HEAD_DIM**-0.5 * QK_SCALE_FACTOR  # == float(impl.scale)
QK_SCALE_LOG2E = QK_SCALE * LOG2E  # the split's base-2 logit scale (f32 in-kernel)
ROPE_THETA = 5e5  # sliding layers' GPT-J RoPE over all 64 dims; full layers: NoPE
WINDOW = 2048  # sliding layers attend keys L - WINDOW <= k < L
KV_BLOCK = 16  # tokens a KV page (the kernel block)
KV_C = 2 * HEAD_DIM  # a (block, head, token) row: K at [0, 64), V at [64, 128)
V_OFF = HEAD_DIM  # V's element offset in the row
Q_PER_KV = HEADS // KV_HEADS  # 8 query heads a KV head

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
# K1 placement (Phase B1; DESIGN_BC 5.1): weight-free boundary CTAs [0, NA),
# the epilogue on CTAs [0, E_UNITS * S), the qkv GEMV on CTAs [NA, BLOCKS)

QKV_ROWS = 2 * CORE + 2 * KV_HEADS * HEAD_DIM  # 2304 a rank: q | k | v | og
QKV_TILES = QKV_ROWS // ROWS  # 144
KPARTS = 16  # K parts of a qkv tile
KPART = HIDDEN // KPARTS  # 512 columns: 4 chunks of 128, one a wave
K1_TASKS = QKV_TILES * KPARTS  # 2304 (tile, K part) tasks: tau = 16 T + kp
K1_GEMV0 = NA  # 64: the first GEMV CTA
K1_GEMV_CTAS = BLOCKS - K1_GEMV0  # 192
K1_TPC = K1_TASKS // K1_GEMV_CTAS  # 12 tasks a GEMV CTA: tau = 12 g + i
K1_ROUNDS = K1_TPC // 2  # 6 rounds: waves 0-3 take task 2r, waves 4-7 2r + 1
K1_MAXC = 2  # GEMV CTAs contributing to a tile (test_spec proves it)
ROW_Q = 0  # the qkv row blocks: q heads, k heads, v heads, o_gate heads
ROW_K = CORE
ROW_V = ROW_K + KV_HEADS * HEAD_DIM
ROW_G = ROW_V + KV_HEADS * HEAD_DIM
E_UNITS = 5  # epilogue units a token: q 0-7, q 8-15, k + v, og 0-7, og 8-15


def k1_contributor0(tile: int) -> int:
    """The first of the ``K1_MAXC`` GEMV CTAs (g = b - K1_GEMV0) whose tasks
    cover ``tile``: contributor c = g - k1_contributor0(tile) holds a run of the
    tile's K parts, c = 0 the lower ones."""
    return (KPARTS * tile) // K1_TPC


def epilogue_row(unit: int, wave: int, lane: int) -> int | None:
    """The qkv row epilogue CTA ``unit`` (token unit // E_UNITS) handles in
    wave ``wave``, lane ``lane`` (= the head dim d), or None (an idle wave):
    host mirror of DESIGN_BC 5.6's table."""
    g5, d = unit % E_UNITS, lane
    if g5 < 2:  # q heads 8 g5 + wave
        return ROW_Q + 64 * (8 * g5 + wave) + d
    if g5 == 2:  # k heads 0, 1 (waves 0, 1), v heads 0, 1 (waves 2, 3)
        if wave >= 2 * KV_HEADS:
            return None
        base = ROW_K if wave < KV_HEADS else ROW_V
        return base + 64 * (wave % KV_HEADS) + d
    return ROW_G + 64 * (8 * (g5 - 3) + wave) + d  # og heads 8 (g5 - 3) + wave


# K1's GEMV task layouts (``K1Build.layout``): each a tuple of roles (first
# CTA, end CTA, tasks a CTA, first task); CTA b of a role runs tasks tau0 +
# n (b - lo) + i, i < n, consecutive (M2's ``test_probe_m2.W_LAYOUTS``).
#   g192  E3 (5.3): 12 tasks on each GEMV CTA [NA, BLOCKS), none on the
#         boundary CTAs; K1_MAXC = 2 contributors a tile (``qpart``)
#   r256  the 10/6 rebalance (11.3 rung 3, open question 1): 6 tasks on each
#         boundary CTA [0, NA), issued after the slice's ``ardy`` publish
#         (rule 8), 10 on each GEMV CTA; 2 to 4 contributors a tile
#         (``qpart4``). No tile straddles the two roles (16 x 24 = 6 x 64).
# Measured (STATUS_BC 11): r256's front is 1.3 us faster at width 1 and ~0.5
# us slower at width 8 (c1 TPOT -1.5 %, c8 +0.5 %), so g192 stays the default.
K1_LAYOUTS = ("g192", "r256")
K1_ROLES = {
    "g192": ((K1_GEMV0, BLOCKS, K1_TPC, 0),),
    "r256": ((0, NA, 6, 0), (NA, BLOCKS, 10, 6 * NA)),
}
K1_MAXC_BY_LAYOUT = {"g192": K1_MAXC, "r256": 4}
K1_QPART = {"g192": "qpart", "r256": "qpart4"}  # the layout's partials region


def k1_role(layout: str, tau: int) -> tuple[int, int, int, int]:
    """The role of ``layout`` whose CTAs run task ``tau``."""
    for lo, hi, n, tau0 in K1_ROLES[layout]:
        if tau0 <= tau < tau0 + n * (hi - lo):
            return lo, hi, n, tau0
    raise ValueError(f"task {tau} is past layout {layout!r}")


def k1_cta(layout: str, tau: int) -> int:
    """The CTA of ``layout`` that runs task ``tau``."""
    lo, _, n, tau0 = k1_role(layout, tau)
    return lo + (tau - tau0) // n


def k1_contributors(layout: str, tile: int) -> tuple[int, int]:
    """(the first CTA holding one of ``tile``'s K parts, how many CTAs do):
    contributor c = b - first holds a run of the tile's K parts, c = 0 the
    lower ones."""
    first = k1_cta(layout, KPARTS * tile)
    return first, k1_cta(layout, KPARTS * tile + KPARTS - 1) - first + 1


assert K1_TASKS == K1_TPC * K1_GEMV_CTAS and ROWS * QKV_TILES == QKV_ROWS
assert ROW_G + CORE == QKV_ROWS and KPART * KPARTS == HIDDEN and KPART == 4 * COLS
assert E_UNITS * MAX_ROWS <= NA and K1_ROUNDS * 2 == K1_TPC

# --------------------------------------------------------------------------
# K2 split-KV core placement (Phase C, q = 1 widths; DESIGN_BC 7.1). Split
# unit u = (t * 2 + kvh) * N + p runs on CTA u; combine (t, h) on CTA
# C0(S) + 16 t + h. A CTA holding both runs its unit first.

NSPLIT_SW = 16  # sliding: <= 129 pages a window -> <= 9 a unit
MIN_PER = 8  # pages a live unit at least: one a wave


def nsplit(S: int, window: bool) -> int:
    """N: units a (token, KV head), the design's (DESIGN_BC 7.1; a build's
    own is ``K2Build.nsplit``)."""
    return NSPLIT_SW if window else BLOCKS // (2 * S)


def max_nsplit(S: int) -> int:
    """The most units a (token, KV head) can have: one unit a CTA."""
    return BLOCKS // (2 * S)


def n_units(S: int, window: bool, n: int = 0) -> int:
    """Split units of a launch (CTAs [0, n_units)); ``n``: N, 0 the design's."""
    return 2 * S * (n or nsplit(S, window))


def n_comb(S: int) -> int:
    """Combine CTAs of a launch: one a (token, q head), 16 S."""
    return HEADS * S


def C0(S: int) -> int:
    """The first combine CTA: [C0, BLOCKS) runs one (token, q head) each."""
    return BLOCKS - n_comb(S)


def split_unit(u: int, n: int) -> tuple[int, int, int]:
    """Split unit u -> (token t, KV head kvh, part p)."""
    return u // (2 * n), (u // n) % 2, u % n


def split_plan(
    L: int, window: bool, n: int, min_per: int = MIN_PER
) -> tuple[int, int, int, int, int]:
    """A (token, KV head)'s pages at context length L (already 0 for a pad
    row and clamped to the table): (lo, b0, nb, per, live). Keys lo <= k < L;
    pages b0 .. b0 + nb; unit p < live covers pages [b0 + p per, min(b0 + (p
    + 1) per, b0 + nb)); units p >= live are empty."""
    lo = max(0, L - WINDOW) if window else 0
    b0 = lo // KV_BLOCK
    nb = (L - 1) // KV_BLOCK - b0 + 1 if L > 0 else 0
    per = max(-(-nb // n), min_per)
    return lo, b0, nb, per, -(-nb // per)


assert all(
    n_units(s, w) <= BLOCKS and C0(s) >= BLOCKS // 2
    for s in range(1, MAX_ROWS + 1)
    for w in (False, True)
)

# --------------------------------------------------------------------------
# K2(split) split / combine retune (TUNE lever L3, DESIGN_BC 12 open question
# 4): a build's N (``K2Build.split_n``, 0 = ``nsplit``) and its least pages
# a live unit (``K2Build.split_min_per``, MIN_PER). A preset names a set of
# them by width and kind (``split_fields``); "+" composes atoms (later atoms
# win a field); the runtime knob ``mono_split_preset`` picks one per width
# (default ``DEFAULT_SPLIT_PRESET``). Every value stays within 7.1's
# invariants: 2 S N <= 256 (a unit a CTA), live = cdiv(nb, per) <= N. A field
# equal to the design's value is stored as the design default (one build,
# one key). Measured (MI355X TP8, STATUS_BC 11): "coarse" is never slower
# than "base" (c8 TPOT -0.4 %, c1 / c4 / 31k context equal), so it is the
# runtime default; "short" / "narrow" win at <= 2k context (c1 -0.8 %) but
# cost a full layer ~7.5 us at 31k (c1 +2.2 %).
#
#   base    the design: N 16 sliding / 256 // 2S full, MIN_PER 8
#   short   full N = min(16, 256 // 2S): up to 2k context the same 16 live
#           units as base (per 8), but only 32 S unit CTAs, so CTAs [32 S,
#           C0) issue their o_proj weights at entry; past 2k a unit grows
#           (16 pages a wave at 32k)
#   wide    sliding N = min(17, 256 // 2S): per = 8 for a whole window (129
#           pages), so no wave runs a second page (base: per 9, wave 0 two)
#   coarse  MIN_PER 16 and sliding N 9: two pages a wave (the unit loop holds
#           two in flight), half the live units -> half the combine's polls
#           and partials, and 7 fewer unit CTAs a (t, kvh) on sliding layers
#   narrow  S <= 4: short+wide; S >= 5: coarse (short-context deployments)
SPLIT_ATOMS = ("base", "short", "wide", "coarse")
SPLIT_PRESETS = (*SPLIT_ATOMS, "narrow")
DEFAULT_SPLIT_PRESET = "coarse"  # mono_split_preset's default (measured)
NARROW_MAX = 4  # "narrow": the widths <= this run short+wide, the rest coarse


def _split_atom(atom: str, S: int, window: bool, f: dict) -> None:
    cap = max_nsplit(S)
    if atom == "short" and not window:
        f["split_n"] = min(16, cap)
    elif atom == "wide" and window:
        f["split_n"] = min(17, cap)
    elif atom == "coarse":
        f["split_min_per"] = 2 * MIN_PER
        if window:
            f["split_n"] = min(9, cap)


def check_split_preset(preset: str) -> tuple[str, ...]:
    """``preset``'s atoms ("+"-joined names of SPLIT_PRESETS); ValueError
    for an unknown one."""
    if not isinstance(preset, str) or not preset:
        raise ValueError(f"split preset {preset!r}: a str")
    atoms = tuple(preset.split("+"))
    bad = [a for a in atoms if a not in SPLIT_PRESETS]
    if bad:
        raise ValueError(f"split preset {preset!r}: {bad} (atoms of {SPLIT_PRESETS})")
    return atoms


def split_fields(preset: str, S: int, window: bool) -> dict:
    """The K2Build split fields of ``preset`` at width S and kind ``window``
    (a design value stored as its default: base -> {})."""
    f: dict = {}
    for atom in check_split_preset(preset):
        if atom == "narrow":
            for a in ("short", "wide") if S <= NARROW_MAX else ("coarse",):
                _split_atom(a, S, window, f)
        else:
            _split_atom(atom, S, window, f)
    if f.get("split_n") == nsplit(S, window):
        del f["split_n"]
    if f.get("split_min_per") == MIN_PER:
        del f["split_min_per"]
    return f


# --------------------------------------------------------------------------
# K2(split) o_proj task placement (``K2Build.oproj_issue``; TUNE lever L2,
# DESIGN_BC 7.2's deferred D4). "late" (7.2): CTA b runs tasks b and b +
# BLOCKS; a combine CTA issues their weights after its ``crdy`` publish (rule
# 8), so its 64 KiB stream starts when the core ends (M2's otail: ~2.0 / 2.7 us
# of tail at S = 1 / 8). "early": the 16 S combine CTAs run no o_proj task.
# Their 32 S tasks are extra tasks of the CTAs below C0, issued with the CTA's
# own two at its own issue point (entry for a CTA with no role, after its
# ``srdy`` publish for a split unit): CTA b < C0 takes moved task e = (C0 - 1
# - b) + j C0 for slot j < oproj_slots(S) while e < 32 S (the CTAs next to C0
# first: a sliding build's no-role CTAs, which issue at entry). No CTA polls or
# publishes with another stage's weights in flight (rule 8 as written), and
# every task's arithmetic is unchanged (the outputs are bit for bit "late"'s).
# Measured (STATUS_BC 11): early gains 0.2-0.5 us at width 1 and loses 0.6-0.9
# us at width 8 (c1 TPOT -0.6 %, c8 +0.5 %), so "late" stays the default.
OPROJ_ISSUES = ("late", "early")  # K2Build.oproj_issue
W_O_BYTES = HIDDEN * CORE * 2  # o_proj's bf16 weight a rank (16 MiB)


def oproj_moved(S: int) -> int:
    """The combine CTAs' o_proj tasks an "early" build moves (2 a CTA)."""
    return 2 * n_comb(S)


def oproj_slots(S: int) -> int:
    """Extra task slots a CTA below C0 holds in an "early" build (1 for S <=
    5, 2 above)."""
    return -(-oproj_moved(S) // C0(S))


def oproj_moved_task(S: int, e: int) -> int:
    """Moved task e (0 .. 2 n_comb - 1): combine CTA C0 + e's task C0 + e,
    then combine CTA C0 + e - n_comb's task BLOCKS + C0 + e - n_comb."""
    c0, half = C0(S), n_comb(S)
    return c0 + e if e < half else BLOCKS + c0 + e - half


def oproj_tasks(S: int, b: int, issue: str) -> list[int]:
    """The o_proj tasks CTA b runs, in order, under ``issue``."""
    if issue not in OPROJ_ISSUES:
        raise ValueError(f"oproj_issue {issue!r} (one of {OPROJ_ISSUES})")
    if issue == "late":
        return [b, b + BLOCKS]
    c0 = C0(S)
    if b >= c0:
        return []
    out = [b, b + BLOCKS]
    for j in range(oproj_slots(S)):
        e = c0 - 1 - b + j * c0
        if e < oproj_moved(S):
            out.append(oproj_moved_task(S, e))
    return out


assert W_O_BYTES < 2**31 and all(
    sorted(t for b in range(BLOCKS) for t in oproj_tasks(s, b, "early"))
    == list(range(O_TASKS))
    for s in range(1, MAX_ROWS + 1)
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
# The build keys and the argument lists

K2_CORES = ("tensor", "gated", "split")  # K2Build.core
_CORE_SUFFIX = {"tensor": "", "gated": "_g"}  # split: "_sw" / "_full"


def _check_key(key, ints: tuple[str, ...], bools: tuple[str, ...], slots: range):
    """The field checks K1Build and K2Build share: ``ints`` / ``bools`` typed,
    the width in range, and the fault hook complete and able to fire in a
    launch of a slot in ``slots``."""
    cls = type(key).__name__
    for f in ints:
        v = getattr(key, f)
        if type(v) is not int:  # bool too: it would name the symbol "sTrue"
            raise TypeError(f"{cls}.{f} = {v!r}: an int")
    for f in bools:
        v = getattr(key, f)
        if type(v) is not bool:
            raise TypeError(f"{cls}.{f} = {v!r}: a bool")
    if not 1 <= key.tokens <= MAX_ROWS:
        raise ValueError(f"{cls}.tokens {key.tokens} (1 .. {MAX_ROWS})")
    if not -1 <= key.fault_rank < TP:
        raise ValueError(f"{cls}.fault_rank {key.fault_rank} (-1 .. {TP - 1})")
    if not key.faulty:
        # One build, one key: the fault fields of a build without the hook
        # would key a second binary under the same kernel name.
        if (key.fault_epoch, key.fault_slot, key.fault_ticks) != (0, 0, 0):
            raise ValueError(f"{cls}.fault_* set without fault_rank")
        return
    # sync.stall compares these with the int32 epoch word and the slot
    # argument: an epoch past int32 or a slot this kernel never runs never fires.
    if not 0 <= key.fault_epoch < 2**31:
        raise ValueError(f"{cls}.fault_epoch {key.fault_epoch} (0 .. 2^31 - 1)")
    if key.fault_slot not in slots:
        what = "K1" if slots.start == K1_SLOT0 else "K2"
        raise ValueError(
            f"{cls}.fault_slot {key.fault_slot}: a {what} slot "
            f"({slots.start} .. {slots.stop - 1}, spec.{what.lower()}_slot)"
        )
    if not 0 < key.fault_ticks < 2**63:
        raise ValueError(f"{cls}.fault_ticks {key.fault_ticks} (> 0)")


def _fault_suffix(key) -> str:
    if not key.faulty:
        return ""
    return f"_f{key.fault_rank}_{key.fault_epoch}_{key.fault_slot}_{key.fault_ticks}"


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
    # Phases B and C, past Phase A's fields (a Phase A key's prefix stays).
    # The attention core K2 reads (K2_CORES): "tensor" o_proj's input from
    # vLLM (Phase A), "gated" vLLM's attention output and the qkv rows, the
    # gated norm in the core load (B2), "split" the split-KV core in-kernel
    # (C, q = 1 widths).
    core: str = "tensor"
    window: bool = False  # split only: a sliding layer's key window
    # Opt-in tuning variant (TUNE L2), split only: where the combine CTAs'
    # o_proj tasks run (OPROJ_ISSUES; "late" is DESIGN_BC 7.2's build)
    oproj_issue: str = "late"
    # Tuning variants (TUNE L3), split only (``split_fields``): N, units a
    # (token, KV head) (0: the design's ``nsplit``), and the least pages a
    # live unit (the runtime picks them by ``mono_split_preset``)
    split_n: int = 0
    split_min_per: int = MIN_PER

    def __post_init__(self):
        _check_key(
            self,
            (
                "tokens",
                "fault_rank",
                "fault_epoch",
                "fault_slot",
                "fault_ticks",
                "split_n",
                "split_min_per",
            ),
            ("window", "debug_dump"),
            range(K2_SLOT0, TAG_SLOTS),
        )
        if type(self.core) is not str:
            raise TypeError(f"K2Build.core = {self.core!r}: a str")
        if self.core not in K2_CORES:
            raise ValueError(f"K2Build.core {self.core!r} (one of {K2_CORES})")
        if self.window and self.core != "split":
            raise ValueError(f"K2Build.window with core {self.core!r} (split only)")
        if type(self.oproj_issue) is not str:
            raise TypeError(f"K2Build.oproj_issue = {self.oproj_issue!r}: a str")
        if self.oproj_issue not in OPROJ_ISSUES:
            raise ValueError(
                f"K2Build.oproj_issue {self.oproj_issue!r} (one of {OPROJ_ISSUES})"
            )
        if self.oproj_issue != "late" and self.core != "split":
            raise ValueError(
                f"K2Build.oproj_issue {self.oproj_issue!r} with core "
                f"{self.core!r} (split only)"
            )
        tuned = (self.split_n, self.split_min_per)
        if tuned != (0, MIN_PER) and self.core != "split":
            raise ValueError(
                f"K2Build split_n / split_min_per {tuned} with core "
                f"{self.core!r} (split only)"
            )
        if self.split_n == nsplit(self.tokens, self.window) or not (
            0 <= self.split_n <= max_nsplit(self.tokens)
        ):
            raise ValueError(
                f"K2Build.split_n {self.split_n} (0: the design's "
                f"{nsplit(self.tokens, self.window)}, else 1 .. "
                f"{max_nsplit(self.tokens)} and not the design's)"
            )
        if not 1 <= self.split_min_per <= 4096:
            raise ValueError(f"K2Build.split_min_per {self.split_min_per} (1 .. 4096)")

    @property
    def faulty(self) -> bool:
        """The build compiles ``sync.stall`` into its prologue."""
        return self.fault_rank >= 0

    def nsplit(self) -> int:
        """A split build's N: units a (token, KV head)."""
        return self.split_n or nsplit(self.tokens, self.window)

    def n_units(self) -> int:
        """A split build's units (CTAs [0, n_units))."""
        return 2 * self.tokens * self.nsplit()

    def c0(self) -> int:
        """A split build's first combine CTA."""
        return C0(self.tokens)

    def oproj_slots(self) -> int:
        """Extra o_proj task slots of a CTA below C0 (0 but "early")."""
        if self.oproj_issue != "early":
            return 0
        return oproj_slots(self.tokens)

    def split_suffix(self) -> str:
        """The symbol's TUNE L3 part: empty for the design's values."""
        out = f"_n{self.split_n}" if self.split_n else ""
        return out + (
            f"_m{self.split_min_per}" if self.split_min_per != MIN_PER else ""
        )

    def kernel_name(self) -> str:
        """The kernel symbol: one per build. A Phase A build ("tensor") keeps
        its Phase A name."""
        if self.core == "split":
            core = "_sw" if self.window else "_full"
            core += self.split_suffix()
            core += "" if self.oproj_issue == "late" else "_oe"
        else:
            core = _CORE_SUFFIX[self.core]
        name = f"muse_mono_k2_s{self.tokens}{core}" + _fault_suffix(self)
        return name + ("_dbg" if self.debug_dump else "")


@dataclass(frozen=True)
class K1Build:
    """Every K1 build parameter (Phase B1: the residual boundary, the qkv GEMV,
    QK-norm / RoPE and the KV insert of one layer), scalars only. One template;
    ``rope`` is the only difference between the two attention kinds."""

    tokens: int  # S, the launch's rows: 1 .. MAX_ROWS
    rope: bool  # a sliding layer's GPT-J RoPE (full layers: NoPE)
    debug_dump: bool = False  # tests: the GEMV rows before the epilogue -> "qkvd"
    # The fault hook as K2Build's, for a K1 slot (spec.k1_slot).
    fault_rank: int = -1
    fault_epoch: int = 0
    fault_slot: int = 0
    fault_ticks: int = 0
    # Opt-in tuning variant (default: the E3 build): the GEMV task layout
    # (K1_LAYOUTS)
    layout: str = "g192"

    def __post_init__(self):
        _check_key(
            self,
            ("tokens", "fault_rank", "fault_epoch", "fault_slot", "fault_ticks"),
            ("rope", "debug_dump"),
            range(K1_SLOT0, K2_SLOT0),
        )
        if type(self.layout) is not str:
            raise TypeError(f"K1Build.layout = {self.layout!r}: a str")
        if self.layout not in K1_LAYOUTS:
            raise ValueError(f"K1Build.layout {self.layout!r} (one of {K1_LAYOUTS})")

    @property
    def faulty(self) -> bool:
        """The build compiles ``sync.stall`` into its prologue."""
        return self.fault_rank >= 0

    def kernel_name(self) -> str:
        """The kernel symbol: one per build."""
        kind = "rope" if self.rope else "nope"
        variant = "" if self.layout == "g192" else f"_{self.layout}"
        name = f"muse_mono_k1_s{self.tokens}_{kind}{variant}" + _fault_suffix(self)
        return name + ("_dbg" if self.debug_dump else "")


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

# K2's arguments by K2Build.core: "gated" reads vLLM's attention output and the
# qkv rows (og at ROW_G) in place of ``core``; "split" reads the qkv rows, this
# layer's KV cache and its KV group's block table and sequence lengths. The
# launch state (the last 7) is every core's.
K2_KV_ARGS = (
    "kv",  # bf16 [blocks, KV_HEADS, KV_BLOCK, KV_C]: the layer's cache
    "kv_sb",  # byte strides of a block, a head, a token (Int64)
    "kv_sh",
    "kv_st",
    "kv_blocks",  # Int32: the cache's blocks (block ids are clamped below it)
)
K2_ARGS_BY_CORE = {
    "tensor": K2_ARGS,
    "gated": (
        "attn",  # bf16 [S, CORE]: the stock core's output, contiguous
        "qkvo",  # bf16 [S, QKV_ROWS]: the qkv rows (og at ROW_G)
        *K2_ARGS[1:],
    ),
    "split": (
        "qkvo",  # bf16 [S, QKV_ROWS]: q normalized and roped, og raw
        *K2_KV_ARGS,
        "block_table",  # int32 [bt_rows, bt_stride]: the KV group's (persistent)
        "bt_rows",  # Int32: the table's rows (its buffer's size)
        "bt_stride",  # Int32: the table's row stride in entries
        "seq_lens",  # int32 [>= S]: context lengths, this step's token included
        *K2_ARGS[1:],
    ),
}
K2_INT32_ARGS_BY_CORE = {
    "tensor": K2_INT32_ARGS,
    "gated": K2_INT32_ARGS,
    "split": ("kv_blocks", "bt_rows", "bt_stride", *K2_INT32_ARGS),
}

K1_ARGS = (
    "branch",  # bf16 [S, HIDDEN]: the previous layer's deferred branch
    "stream",  # fp32 [S, HIDDEN]: the residual stream
    "pad",  # bool [>= S]: pad rows
    "positions",  # int64 [>= S]
    "slots",  # int64 [>= S]: this layer's slot mapping (< 0: no insert)
    "w_postffn",  # layer i - 1: post_ffn_norm.weight bf16
    "ffn_alpha",  # layer i - 1: add_ffn alpha / beta fp32
    "ffn_beta",
    "w_preqkv",  # layer i: pre_qkv_proj_norm.weight bf16
    "w_qkv",  # layer i: qkv_proj.weight bf16 [QKV_ROWS, HIDDEN]
    "cos_sin",  # layer i: rotary cos_sin_cache fp32 [max_pos, 64] (0 if NoPE)
    *K2_KV_ARGS,
    "max_pos",  # Int32: cos_sin's rows (positions are clamped below it)
    "stream_out",  # fp32 [S, HIDDEN]: K2's stream
    "qkvo",  # bf16 [S, QKV_ROWS]: the qkv rows after QK-norm / RoPE
    "scratch",
    "rank",  # no peers: K1 is rank-local; stall() compares the rank
    "epoch",
    "slot",
    "health",
    "alarm",
)
K1_INT32_ARGS = ("kv_blocks", "max_pos", "slot", "rank")

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
K2_REGIONS = SCRATCH_REGIONS + PEER_REGIONS  # every K2 core puts and polls these
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

# Phases B and C (DESIGN_BC 5.7): mailboxes, then plain regions, laid out past
# ``xrow`` so that every Phase A offset (and every Phase A K2 binary) stays.
SCRATCH_PAIRS_BC = (
    ("fss", NA * _MT),  # j*MT + t: K1 boundary, fp32 sum of x^2 over slice j
    ("nss", NA * _MT),  # j*MT + t: fp32 sum of new_stream^2 over slice j
    ("ardy", NA),  # j: K1 boundary slice j's arow published (flag)
    ("qpart", QKV_TILES * K1_MAXC * _MT * ROWS),  # ((T*2 + c)*MT + t)*16 + r
    ("srdy", BLOCKS),  # u: split unit u's opart / pstat published (flag)
    ("crdy", 16 * _MT),  # 16 t + h: combine (t, h)'s xo row published (flag)
)
SCRATCH_REGIONS_BC = tuple(n for n, _ in SCRATCH_PAIRS_BC)
# Tuning variants' mailboxes, laid out past every plain region so that no
# earlier offset (and no earlier build's binary) moves.
SCRATCH_PAIRS_TAIL = (
    # K1 r256: ((T*4 + c)*MT + t)*16 + r, up to 4 contributors a tile
    ("qpart4", QKV_TILES * K1_MAXC_BY_LAYOUT["r256"] * _MT * ROWS),
)
SCRATCH_REGIONS_TAIL = tuple(n for n, _ in SCRATCH_PAIRS_TAIL)
SCRATCH_REGIONS_ALL = SCRATCH_REGIONS + SCRATCH_REGIONS_BC + SCRATCH_REGIONS_TAIL
K1_REGIONS = ("fss", "nss", "ardy", "qpart")  # K1 puts and polls these
# ... by K1Build.layout
K1_REGIONS_BY_LAYOUT = {
    lay: ("fss", "nss", "ardy", K1_QPART[lay]) for lay in K1_LAYOUTS
}
SPLIT_REGIONS = ("srdy", "crdy")  # ... and K2(split) these, past K2_REGIONS
# Every region a wait record may name (debug.region_ids; ids are name CRCs, so
# the order is free).
REGIONS = SCRATCH_REGIONS_ALL + PEER_REGIONS

# Plain (untagged) regions past the B/C mailboxes: (name, bytes, alignment).
AROW = "arow"  # bf16 [MT, HIDDEN]: K1's attn_in rows (boundary -> GEMV)
QKVD = "qkvd"  # bf16 [MT, QKV_ROWS]: debug K1 builds' GEMV rows (epilogue -> tests)
XO = "xo"  # bf16 [MT, CORE]: the split core's gated rows (combine -> load_core)
PSTAT = "pstat"  # fp32 [BLOCKS, 8, 2]: a split unit's (m, l) per q head
OPART = "opart"  # fp32 [BLOCKS, 8, HEAD_DIM]: a split unit's unnormalized rows
SCRATCH_PLAIN_BC = (
    (AROW, _MT * HIDDEN * 2, 16),
    (QKVD, _MT * QKV_ROWS * 2, 16),
    (XO, _MT * CORE * 2, 16),
    (PSTAT, BLOCKS * Q_PER_KV * 2 * 4, 16),
    (OPART, BLOCKS * Q_PER_KV * HEAD_DIM * 4, 16),
)
# Plain regions read inside the launch that wrote them: stored and loaded with
# CM_DEV, never through SMEM (each XCD has its own L2; DESIGN_BC rule 9).
IN_LAUNCH_PLAIN = (XROW, AROW, XO, PSTAT, OPART)

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


def _align(off: int, align: int) -> int:
    return (off + align - 1) // align * align


def scratch_layout() -> dict[str, tuple[int, int]]:
    """A rank's scratch (one buffer shared by every launch of a step; the tags
    tell launches apart): the mailboxes, then ``xqd``, then ``xrow``
    (16-byte aligned); then Phases B / C's mailboxes and plain regions.
    name -> (byte offset, bytes)."""
    out = pair_layout(SCRATCH_PAIRS)
    out.update(pair_layout(((XQD, XQD_PAIRS),), start=_end(out)))
    out[XROW] = (_align(_end(out), XROW_ALIGN), XROW_BYTES)
    out.update(pair_layout(SCRATCH_PAIRS_BC, start=_end(out)))
    for name, nbytes, align in SCRATCH_PLAIN_BC:
        out[name] = (_align(_end(out), align), nbytes)
    out.update(pair_layout(SCRATCH_PAIRS_TAIL, start=_end(out)))
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
    """``ass`` and ``bss`` (K2's seam); ``fss`` and ``nss`` (K1's boundary)."""
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


def ardy_pair(j):
    return j


def qpart_pair(T, c, t, r):
    """K1: contributor c's partial of qkv tile T's row r for token t."""
    return ((T * K1_MAXC + c) * _MT + t) * ROWS + r


def k1_qpart_pair(layout, T, c, t, r):
    """K1 ``layout``'s partials region (``K1_QPART``) pair of contributor c's
    partial of tile T's row r for token t (``qpart_pair`` under g192)."""
    return ((T * K1_MAXC_BY_LAYOUT[layout] + c) * _MT + t) * ROWS + r


def srdy_pair(u):
    return u


def crdy_pair(t, h):
    return 16 * t + h


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


def arow_elem(t, col):
    """``arow`` bf16 element of row t, column col."""
    return t * HIDDEN + col


def qkvd_elem(t, row):
    """``qkvd`` bf16 element of token t's qkv row ``row``."""
    return t * QKV_ROWS + row


def xo_elem(t, h, d):
    """``xo`` bf16 element of token t, q head h, dim d."""
    return t * CORE + HEAD_DIM * h + d


def pstat_word(u, j, k):
    """``pstat`` fp32 word of split unit u, its q head j (of 8), k = 0: the
    running max m (base 2), k = 1: the sum l."""
    return (u * Q_PER_KV + j) * 2 + k


def opart_word(u, j, d):
    """``opart`` fp32 word of split unit u, its q head j, dim d."""
    return (u * Q_PER_KV + j) * HEAD_DIM + d
