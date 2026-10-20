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
are the same switch. There are no other options. `decode_backend=auto` (the
default) without `--monokernel` runs vLLM's decode; any other `decode_backend`
value is an error.

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

Every TP rank agrees on each verdict before anything collective runs.

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
earlier by the same launch slot carries the current tag. Only the rows past a
narrow step's width go unwritten, so a wide step exactly 2^22 kernel steps (about
35 h at 30 ms a step) after the previous wide step could accept stale data in a
pair polled before its producer wrote it. This is accepted.

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
the largest): about 17-45 KiB per build. Model Runner V2 side streams can occupy CUs briefly;
that costs latency, not progress. Do not co-schedule other GPU work on the
devices.

## Layout

Everything lives in this folder. The model files only call in:
`amd/model.py` (the config check in `__init__`, `plan_specialized_decode` in
`forward`, `step.layer(i)` in the layer loop, `after_specialized_decode_step` in
`compute_logits`, `skip_o_proj` in the attention, `get_model_state_cls`) and
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
    spec.py        the shared contract: shapes, placement, build key, layouts
    abi.py         the error-word layout
    layer.py       K2
    attention/     back.py (o_proj + all-reduce push)
    stages/        seam, router, pool, experts, reduce, unpool, gemv
    common/        device ops, mailbox sync, MX helpers, plan / build keys,
                   the route table and probe context (adapted from ROCm/ATOM
                   atom/mono, MIT, via the Kimi-K3 specialized decode)
```

## Phases

- **A (this):** stock attention up to o_proj's input, then K2.
- **B:** a K1 launch per kernel layer replaces the stock front (the residual
  boundary, the QKV GEMV, QK-norm/RoPE, the KV insert) and stock attention reads
  its output; K2 is unchanged. The tag slots (K1: the layer index, K2: 128 + the
  layer index) and the scratch tail are reserved for it.
- **C:** the split-KV GQA attention core moves into K2 for `q = 1` widths; DSpark
  verify widths keep the stock core until the split masks causally per query
  row. DSpark stays supported.
