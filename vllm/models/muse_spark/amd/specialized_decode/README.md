# Muse Spark specialized decode (MI355X)

An opt-in decode path for Muse Spark Open (`MuseSpark1xForCausalLM`,
`MuseSparkForConditionalGeneration`) at TP8 on MI355X (gfx950), following
RFC #60904. On a pure decode step (or a pure DSpark verify step) of at most 8 rows
every decoder layer whose routed experts run AITER's MXFP4 x MXFP4 MoE becomes one
persistent FlyDSL MonoKernel launch (K2: 256 CTAs, one a CU, all resident), in
"layer mode", inside the stock layer loop. The TP all-reduces, all-gathers and the
expert reduce-scatter run in-kernel over peer memory.

This is **Phase A**: vLLM still runs each kernel layer's attention up to o_proj's
input (the residual boundary, the QKV GEMV, QK-norm/RoPE, the KV insert, the
attention and the gated norm). K2 then runs, in one launch:

- o_proj and its all-reduce (one hop, rank-order fp32 sum, bf16);
- the attention boundary: the weightless post-attention norm, the gated
  norm-preserving residual blend into the fp32 stream, the pre-FFN norm;
- the router (fp32, this rank's 32 experts) and the top-8 routing (sigmoid plus
  selection bias, lowest index on ties, serial renormalization);
- the pooler (8192 -> 4096), the pre-expert norm and the MXFP4 activation quant;
- the routed experts (gate/up, SiLU, MXFP4 requant, down; this rank's 512
  intermediate columns of every expert in the step's union);
- the column-owner reduce of the 8 ranks' expert rows (fp32, rank order), the
  post-expert norm, the route weights and the top-8 sum;
- the unpooler (4096 -> 8192) and its all-gather.

It returns the same deferred FFN boundary as the stock layer (stream, branch,
the layer's `post_ffn_norm` / `add_ffn`), so kernel and stock layers interleave
freely. The kernels read vLLM's loaded weights in place: nothing is copied or
re-quantized.

## Enabling

```bash
VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ROCM_USE_AITER=1 VLLM_MOE_SKIP_PADDING=1 \
vllm serve amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4 -tp 8 \
  --attention-backend ROCM_AITER_FA --block-size 16 \
  --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --monokernel
```

`--monokernel` and `--additional-config '{"decode_backend": "latency_specialized"}'`
are the same switch. `decode_backend=auto` (the default) without `--monokernel`
runs vLLM's decode; any other `decode_backend` value is an error.

That switch alone serves Phase A. Phases B and C (see Phases) are off by default
and are selected in the same `additional_config`, each on its own:

```bash
--additional-config '{"decode_backend": "latency_specialized",
  "mono_front": "k1", "mono_core": "split"}'
```

`mono_front` is `stock` (the default) or `k1`, `mono_core` is `tensor` (the
default), `gated` or `split`.

`mono_core: "gated"` is B2, `mono_front: "k1"` is B1 and `mono_core: "split"` is
C; `split` serves `q = 1` widths, and DSpark verify widths keep `gated`. The
tuning options below were measured on MI355X TP8 (see Measured):

| Key | Values (default first) | Selects |
| --- | --- | --- |
| `mono_k1_min_width` | 1..8, with `k1` | narrower widths keep vLLM's front |
| `mono_k1_layout` | `g192` \| `r256`, with `k1` | K1's GEMV task layout: `r256` also gives the 64 boundary CTAs 6 GEMV tasks each (issued after their boundary publish), 10 on each GEMV CTA. Faster at width 1, slower at width 8 |
| `mono_split_preset` | `coarse` \| `base` \| `short` \| `wide` \| `narrow`, with `split` | the split-KV core's units: `base` is the design's (16 units a (token, KV head) on a sliding layer, 256 / 2S on a full one, at least 8 pages a unit); `coarse` asks for 16 pages a unit (and 9 sliding units: half the combine's polls and partials); `short` caps a full layer at 16 units, `wide` gives a sliding layer 17; `narrow` is `short+wide` up to width 4 and `coarse` above. Presets compose with `+`; a dict picks one by launch width (`{"1-4": "short+wide", "5-8": "coarse"}`, widths it leaves out `base`) |
| `mono_core_oproj` | `late` \| `early`, with `split` | `early` moves the combine CTAs' o_proj tasks onto the CTAs below them, which issue those weights with their own, so no weight stream starts at the end of the core (bit for bit `late`'s outputs) |

A split preset moves outputs by fp32 rounding only (a different split of the
same keys). `short` and `narrow` are for short contexts only: a full layer's
unit grows with the context (+7.5 us a full layer at 31k). An unknown value fails
at startup, before anything collective runs. A tree without a kernel (no
`mono/k1.py`, a core missing from `mono.layer.ABI_BY_CORE`) refuses the
deployment at startup.

The fastest measured configuration for single-stream latency (c1) is

```bash
--additional-config '{"decode_backend": "latency_specialized",
  "mono_front": "k1", "mono_core": "split", "mono_k1_layout": "r256"}'
```

and with the default `g192` layout the same front and core are best at c8.

`VLLM_MOE_SKIP_PADDING=1` is required: the kernels read the step's pad-row mask
(`forward_context.is_padding`), which Model Runner V2 refreshes every step only
with that setting. Without it the mask of a padded FULL-graph replay could be
stale, and pad rows would be served as live ones.

## Supported deployments

TP8 on MI355X with all 256 CUs, Model Runner V2, the Quark OCP-MXFP4 checkpoint,
bf16 model and KV cache (`ROCM_AITER_FA`, unshuffled layout, block size 16), no
speculative decoding or DSpark with 7 speculative tokens (static K; its EAGLE3
auxiliary hidden-state capture works unchanged), FULL decode CUDA graphs or eager
decode. Breakable CUDA graphs are fine. Multimodal inputs are not refused (the
embedding happens before the layer loop); `--language-model-only` is the tested
configuration.

Everything else is refused (default-deny). If you request the specialized decode
for anything unsupported, startup fails with one `ValueError` that lists every
reason, in three stages:

- **Config, in the model's `__init__`:** pipeline, data, expert, decode-context
  or prefill-context parallelism, EPLB, DBO, sequence-parallel MoE, LoRA, KV
  transfer / offloading connectors, routed-expert capture, sleep mode, CPU weight
  offload, weight updates, prompt embeds, a speculative method other than DSpark,
  `num_speculative_tokens` other than 0 or 7, adaptive verification, dynamic
  speculative decoding, the V1 model runner, `STOCK_TORCH_COMPILE`, PIECEWISE
  decode graphs, a model dtype other than bf16, an fp8 KV cache, a KV block size
  other than 16, an attention backend other than `ROCM_AITER_FA`, the shuffled
  AITER KV layout, `VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE` other than fp32 / bf16,
  another quantization, another model's shapes,
  gated delta-net or dense MLP layers, an unknown layer type, the wrong RoPE
  pattern, `VLLM_MOE_SKIP_PADDING=0`, batch-sharded sampling, and the platform
  (not ROCm, not gfx950, FlyDSL or AITER's FlyDSL helpers missing, fewer than 256
  CUs).
- **Loaded weights, when the model state builds the executor:** each kernel
  layer's weight shapes, dtypes and contiguity, the norm epsilons and gain
  centres, the folded router multiplier, the route scale, the AITER-shuffled
  expert weights and scales, the expert activation, an expert map.
- **KV caches, at every bind:** dtype, `[blocks, 2, 16, 128]` layout, a dense
  last dimension (Phase A does not read the KV cache; these hold the Phase B/C
  contract), and the resolved decode CUDA-graph mode (PIECEWISE).

Phases B and C add weight and cache rows (the qkv and norm weights K1 reads,
the attention kind's window and RoPE, the attention options, the KV block size,
16-byte rows and the 32-bit page window). They apply only when a kernel that
reads them is on, so a Phase A deployment is refused for exactly what it was
before. Every TP rank agrees on each verdict before anything collective runs.

## Steps, widths and padding

`plan()` runs once a forward. A request decodes `q` rows: 1 without speculative
decoding, 8 with DSpark (K = 7). The built widths are every multiple of `q` up to
8 rows that the scheduler can reach (`w <= max_num_seqs * q`). `plan()` picks the
kernels for a pure decode / uniform verify step of whole requests whose width is
a built width; every other step runs vLLM's layers: prefill, mixed, wider and
PIECEWISE steps.

Under FULL graphs the plan is baked in at capture; vLLM's capture sizes are not
changed. A replay with fewer requests than its graph pads rows; pad rows produce
exact zeros (selected, never multiplied by a mask, so NaN/Inf in a pad row cannot
leak) and their expert picks stay out of the step's expert union. The startup log
lists every width without a FULL graph; to capture every width exactly, pass them
in `--cudagraph-capture-sizes`. Eager steps run the kernels too.

## K2 program order

Module docstrings and tests name K2's phases by number. Every CTA runs them in
this order; each poll waits only on an earlier phase, and every CTA is
resident, so the order is deadlock-free:

| Phase | Module | Work |
| --- | --- | --- |
| 1 | `mono/attention/back.py` | o_proj and the push half of its all-reduce (every CTA) |
| 2a | `mono/stages/seam.py` | the all-reduce's reduce half and the attention boundary |
| 2b | `mono/stages/router.py` | the fp32 router and the top-8 routing |
| 2c | `mono/stages/pool.py` | the pooler and its all-gather |
| 3a | `pool.load_xq` | the pre-expert norm and MXFP4 a1 (every CTA, into LDS) |
| 3b | `mono/stages/experts.py` | the expert union, gate/up (`ug`), down, pushed to the column owner (`rsx`) |
| 4a | `mono/stages/reduce.py` | the column-owner reduce and the post-expert finalize |
| 4b | `mono/stages/unpool.py` | the unpooler and the branch all-gather |

The stage structs share one LDS union; a stage that touches its struct or the
shared `red` buffer ends with a barrier on all 512 threads (the hand-over rule),
and `test_compile` checks the order and the barriers on every build.

## Numerics

- o_proj's all-reduce is an fp32 sum of the 8 ranks' bf16 partials in rank
  order, rounded once to bf16 (the stock all-reduce's output dtype). Every
  cross-rank sum (the all-reduce, the expert reduce, the sums of squares) runs
  in one fixed order on every rank, so every rank's outputs are bit-identical.
- The router is an fp32 dot (no bf16 split of the weight); only the summation
  order differs from the stock fp32 router GEMM. The routing sigmoid is
  `1 / (1 + exp2(-x * log2(e)))` with an IEEE divide (`tl.sigmoid`'s math), and
  the selection, tie rule and renormalization are the stock kernel's, so ids and
  weights equal stock's on the same logits.
- MXFP4 activations use the RoundUp scale with the `1e-10` amax floor, as AITER's
  HIP quant, for both the pooled rows and the bf16 SiLU output.
- **Deviation:** the expert rows are reduced over the 8 ranks in fp32 and never
  rounded before the post-expert norm. `VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE` (fp32 or
  bf16) is accepted but changes only the stock path; with `bf16` the stock layers
  round the reduced rows to bf16 and the kernel layers do not.

**Tolerances** (the tests check each stage on its own inputs as the kernel read
them, so one stage's error cannot hide behind upstream noise): the all-reduce
is bit-exact vs the rank-order fp32 sum; the router logits are within
`rtol=1e-5, atol=1e-6` of `rocm_fp32_router_gemm`; the routing ids and weights
are `torch.equal` to `fused_musespark1x_routing`; the pooled rows, the
pre-expert norm, the finalize and the unpooler are within 1 bf16 ulp on at
least 99.9 % of elements (max 2 ulp); the MXFP4 a1/a2 bytes are bit-exact vs
the RoundUp reference; the outputs are `torch.equal` across the 8 ranks. Each
norm is also tested at its epsilon and with an all-zero weight, so an eps swap
or a missing gain centre fails.

## Fail-stop

Every in-kernel wait gives up after a budget kept in the health header word
`WAIT_BUDGET`: 120 s from startup and every KV-cache bind until the CUDA-graph
capture ends (the eager warm-ups interleave the kernels with stock kernels that
may compile for the first time, and ranks can skew by more than 10 s), then 10 s.
Under `--enforce-eager` there is no capture and the budget stays at 120 s. A rank
whose wait gives up records it in its health words (sticky: its later waits exit
at once, with addresses kept in bounds) and stops waiting, so a stalled peer fails
every rank instead of hanging them.

Startup failures are voted before the engine reports ready: a self-test launch of
every build at every KV bind, and a vote after the CUDA-graph capture. While
serving, each step `compute_logits` queues a non-blocking copy of the rank's
`FAILED` word; every 16 steps the TP ranks agree on what has landed over the TP
CPU group, and every rank raises at the same step, so the engine stops. Detection
takes up to about 16 steps, so tokens of the failed steps can reach the client
before the engine stops. A zero-delay check needs a core hook and is left to a
follow-up.

Mailbox tags are `(epoch & (2^22 - 1)) * 256 + slot + 1`, with a device step
epoch the runner bumps once a kernel step (inside the graph); nothing is ever
zeroed. **Aliasing window:** a pair last written a multiple of 2^22 epochs
earlier by the same launch slot carries the current tag. Only pairs a launch
does not write keep an older tag: the rows past a narrow step's width and, once
Phase C's split core runs, the split flags past a launch's unit count (a split
flag's unit depends on the width and the layer kind). So a launch exactly 2^22
kernel steps (about 35 h at 30 ms a step) after the last one that wrote such a
pair could accept stale data in a pair polled before its producer wrote it.
This is accepted.

`tests/models/muse_spark/amd_specialized_decode/test_engine_fault.py` stalls a
rank in a served model and checks that the engine stops.

## Residency

K2 is a plain grid launch, not a cooperative one: its spin waits need all 256
CTAs resident at once. That rests on one CTA a CU with 256 CUs (refused
otherwise), on the resource gate every build passes (at most 256 VGPRs, no
scratch spills, LDS within 160 KiB) and on `known_block_size=[512, 1, 1]` in the
kernel decorator (without it the compiler assumes 1024 threads a block, caps
VGPRs at 128 and spills). The LDS is a fixed 8 KiB `red` buffer, the ~4 KiB route
table and one union of the stage structs (the pooler's 32.9 KiB at 8 rows is
the largest): about 17-45 KiB per build (29.5 KiB at 1 row for a split build;
K1, with its residency pin, 80-106 KiB). Model Runner V2 side streams can occupy
CUs briefly; that costs latency, not progress. Do not co-schedule other GPU work
on the devices.

## Layout

Everything lives in this folder. The model files only call in:
`amd/model.py` (the config check in `__init__`, `plan_specialized_decode` in
`forward`, `step.layer(i)` in the layer loop, `after_specialized_decode_step` in
`compute_logits`, `skip_o_proj` in the attention and its `front` / `attend`
split for Phases B and C, `get_model_state_cls`) and
`amd/model_state.py` (builds the executor after load, binds KV caches, votes
after capture).

```text
hooks.py           SpecializedDecodeMixin: the model-side hooks
runtime/
    __init__.py    the switch, DecodeConfig, specialized_decode_config(),
                   refuse_config() (= unsupported())
    decode.py      SpecializedDecode: plan() -> Step, Step.layer(i), prepare_kv(),
                   self-test, after_capture(), after_step()
    guards.py      kernel layers, weight and KV cache refusals, the fail-stop watch
    layers.py      KernelLayer: vLLM tensors -> K2 launches
    runner.py      per-rank launch state: scratch, step epoch, health, peers
    peer.py        peer memory and TP agreement (adapted from ROCm/ATOM, MIT)
mono/              kernel source
    spec.py        the shared contract: shapes, placements, build keys,
                   argument lists, layouts (Phases A, B and C)
    abi.py         the error-word layout
    layer.py       K2 (core variants: tensor, gated, split)
    k1.py          K1 (Phase B1)
    front/         K1's stages: boundary, qkv GEMV, epilogue (QK-norm / RoPE /
                   KV insert)
    attention/     back.py (o_proj + all-reduce push), gate.py (the gated
                   norm), split.py and combine.py (the split-KV core)
    stages/        seam, router, pool, experts, reduce, unpool, gemv
    common/        device ops, mailbox sync, MX helpers, plan / build keys,
                   the route table and probe context (adapted from ROCm/ATOM
                   atom/mono, MIT, via the Kimi-K3 specialized decode)
```

## Phases

- **A (this):** stock attention up to o_proj's input, then K2.
- **B2:** K2 reads the stock attention output and the qkv rows and applies the
  gated norm in its core load (`K2Build.core = "gated"`); the stock front stays.
- **B1:** a K1 launch per kernel layer replaces the stock front (the residual
  boundary, the QKV GEMV, QK-norm/RoPE, the KV insert) and stock attention reads
  its output. K1 is rank-local; one template, RoPE on for sliding layers
  (`K1Build`, `muse_mono_k1_s{S}_{rope,nope}`).
- **C:** the split-KV GQA attention core moves into K2 for `q = 1` widths
  (`core = "split"`, a window for sliding layers: `_sw` / `_full` builds);
  DSpark verify widths keep the stock core with the gated core load, since the
  split does not mask causally per query row. DSpark stays supported.

The phases ship independently (Enabling); each is CPU-tested and GPU-validated
on MI355X TP8 (probes against fp64 and the stock kernels, the TP8 layer chain,
CUDA graph replay, stall / fail-stop, GSM8K parity). `mono/spec.py` holds their
whole contract: the build
keys (a `core = "tensor"` key is a Phase A build, with its Phase A name and
binary), the argument lists, the placements and the scratch layout (B / C
regions past Phase A's, so every Phase A offset stays; 1.5 MB a rank in all).
The tag slots are K1: the layer index, K2: 128 + the layer index, so a step's K1
and K2 pairs never alias.

- **K2(gated)** (B2) takes `attn` and `qkvo` in place of `core`. Its core load
  (`attention/gate.py`) issues the attention output and o_gate loads first, then
  the o_proj weights, and computes FLA's `rms_norm_gated` bit for bit into `xl`;
  `test_compile_gated.py` pins the order (no wait between the first weight load
  and the last `xl` write retires a weight load).
- **K1** (`mono/k1.py`) is one launch a kernel layer and rank on the K2 grid,
  its roles split by CTA so that no CTA polls or publishes with another task's
  loads in flight. It builds its context with `make_context(..., peers=False,
  route_lds=False)`: no `peers` argument, no peer regions, no route table, and
  only `alarm` leaves the rank.
    - CTAs 0-63 run the boundary (`front/boundary.py`): K2's seam with the
    deferred branch in place of the all-reduce and the previous layer's
    `post_ffn_norm` weight applied first; past its input both run
    `stages/seam.py`'s `blend_norm`. They exchange the per-slice sums of
    squares over `fss` / `nss` and publish each slice's `arow` (device scope)
    behind its `ardy` flag.
    - CTAs 64-255 run the qkv GEMV (`front/qkv.py`): 12 (tile, 512-column K part)
    tasks each, all their weights in flight from entry, the `arow` rows staged
    into LDS in one pass behind the `ardy` flags. Every tile has two
    contributors. `mono_k1_layout: "r256"` (`K1Build.layout`) spreads the 2304
    tasks over all 256 CTAs: 6 on each boundary CTA, issued only after its
    slice's `ardy` publish so that no boundary poll waits on weights, and 10 on
    each GEMV CTA; a tile then has 2 to 4 contributors, summed in a fixed
    order from `qpart4`.
    - CTAs 0 .. 5S - 1 then run the epilogue (`front/epilogue.py`, 5 CTAs a token,
    a head a wave): it adds the two fp32 partials, rounds to bf16 once and
    reproduces the stock fused QK-norm / RoPE / KV insert's gfx950 arithmetic bit
    for bit (including the uneven RoPE pairs of dims 0 and 7), writing `qkvo`
    and, for slots >= 0, the K / V cache. A rank that has FAILED writes no K / V.
    - Every build holds more than half a CU's LDS (a residency pin in its stage
    union), so the dispatcher places one CTA a CU at every width.
- **K2(split)** (C, `q = 1` widths) runs the attention core in the launch. The
  first `n_units(S, kind)` CTAs each attend one split-KV unit
  (`attention/split.py`). The last 16 S CTAs each merge one (token, q head)
  (`attention/combine.py`): a combine waits only for the units that hold keys,
  merges their base-2 partials at device scope, rounds the attention output to
  bf16 as the stock core does and applies the gate (`gate.o_gate`), so `xo`
  equals `rms_norm_gated` of that output bit for bit. A CTA issues its o_proj
  weights only after its last publish (at entry if it has no role), so no flag
  waits on a weight stream; the core load (`split.load_core`) then waits for
  the combine's flags with only those weights in flight. `test_compile_split.py`
  pins that order, the split loop's two pages in flight, device-scope merge
  loads and no scalar reads of the in-launch regions. A build's units
  (`K2Build.split_n`, `split_min_per`) come from `mono_split_preset`;
  `K2Build.oproj_issue = "early"` (`mono_core_oproj`) moves the combine CTAs'
  o_proj tasks to the CTAs below C0 (`spec.oproj_tasks`).

K1 and the split core read MRV2's persistent metadata buffers (slot mappings,
block tables, sequence lengths, positions, the pad mask). The KV bind pins each
kernel layer's table and slot mapping to its group's persistent buffers, and
every `plan()` checks their addresses (the first plan of each FULL graph is its
capture), so a builder that broke persistence stops the startup.

Two guards keep the phases honest without a GPU. `test_compile.py` compares
every Phase A build's final ISA with `k2_isa_baseline.json`, recorded from the
Phase A sources: B and C must leave Phase A launches bit-identical.
`test_stock_isa_pin.py` pins the gfx950 code of the two stock kernels Phase B
reproduces bit for bit, the fused QK-norm / RoPE / KV insert (read from vLLM's
own code object) and FLA's `rms_norm_gated` (compiled for gfx950), so a
compiler, Triton or kernel-source change that would alter their rounding fails
on CPU first.

## Measured

MI355X, TP8, `amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4`, FULL decode CUDA
graphs. Per-layer serving spans (vLLM profiler, slowest rank, median, attention
part plus K2, weighted over the 15 full and 45 sliding layers):

| configuration | width 1 | width 8 |
| --- | --- | --- |
| A (`--monokernel`) | 80.1 us | 131.9 us |
| K1 + split, `base` preset | 67.7 us | 119.3 us |
| K1 + split (default `coarse` preset) + `r256` | 66.4 us | 119.1 us |

Decode TPOT, ms (median of 3 runs; 2k-token prompts, 1k output tokens, or 31k
prompts; concurrency c), and GSM8K at c8:

| configuration | c1 2k | c4 2k | c8 2k | c1 31k | GSM8K |
| --- | --- | --- | --- | --- | --- |
| A | 5.094 | 6.471 | 8.579 | 5.154 | 95.5 |
| K1 + split, `base` preset | 4.787 | 6.134 | 8.133 | 4.815 | 95.6 |
| K1 + split (`coarse`) | 4.782 | 6.127 | **8.103** | - | 94.5 |
| K1 + split (`coarse`) + `r256` | 4.712 | 6.156 | 8.148 | **4.730** | 94.8 |
| ... + `mono_core_oproj: "early"` | 4.683 | 6.146 | 8.188 | - | 95.3 |
| K1 + split, `narrow` + `r256` | **4.675** | 6.143 | 8.139 | 4.834 | 95.3 |

GSM8K stays within 1.5 points of Phase A on every configuration (Phase A itself
measured 94.9 and 95.5 on two runs). `r256` and `early` win at width 1 and lose at
width 8, so they stay opt-in; `narrow` wins at 2k and loses at 31k. The design's
per-kernel gates are not all met: K1's front is 13.8 us at width 1 with `r256`
(stock 18.7 us), and the split core adds 7.1-8.6 us over the gated core at width
1 (5.8-6.1 us at width 8).
