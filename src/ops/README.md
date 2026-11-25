# src/ops

Every numeric operator in athanor. Each op is a pair of files plus tests, the portable C at the root of the folder and the CUDA C in `cuda/` (`src/backend/README.md`, Source layout):

```
src/ops/<op>.c        <op>_ref(): scalar, fp32 accumulation, obviously correct, never optimized
                      <op>_fast(): optional SIMD/multithreaded CPU path, only for ops used on CPU at runtime
src/ops/cuda/<op>.cu  GPU kernel(s), CUDA C compiled for CUDA and HIP
tests/test_ops.c      ref vs golden data; GPU vs ref; fast vs ref
```

`ops.h` declares the functions with their shapes, dtypes and math. Step 3 landed the CPU references; step 4 the first fast CPU path, `ath_matmul_fast`; step 7 a GPU kernel for every op of the forward pass, `ath_<op>_gpu`; step 8 one entry point per op, `ath_<op>(d, ...)` in `dispatch.c`, which a family's forward pass calls for both devices.

## Entry points (step 8)

`ath_<op>(const ath_device *d, ...)` runs the op on device d when d is set (`ath_<op>_gpu`) and on the CPU otherwise (`ath_<op>_ref`; `ath_matmul(d, pool, ...)` runs `ath_matmul_fast` on the pool). The caller chooses, from where its forward pass runs; each implementation checks that the tensors are on its side, so a tensor left behind is an error and never a silent copy. `src/models/qwen35/forward.c` is written once against these.

## The references (step 3)

| File | Functions |
| --- | --- |
| `ops.h` | every signature, with shapes, dtypes and math; the conventions below |
| `check.h` | private: argument checks shared by the `_ref` functions, element access (`ld`, `st`) |
| `embed.c` | `ath_embed_ref` |
| `matmul.c` | `ath_matmul_ref`: `x · wᵀ` against a weight in checkpoint layout `[M, K]`; `ath_matmul_fast` (step 4, below) |
| `norm.c` | `ath_rmsnorm_ref` (plain or zero-centered: `w_offset` 0 or 1), `ath_rmsnorm_gated_ref` (SiLU gate), `ath_l2norm_ref` |
| `rope.c` | `ath_rope_ref`: rotate-half on the first `rope_dim` dimensions of each head |
| `act.c` | `ath_swiglu_ref`, `ath_sigmoid_mul_ref`, `ath_add_ref` |
| `attn.c` | `ath_attn_ref`: causal softmax attention with grouped-query heads, the queries being the last T of S positions |
| `conv.c` | `ath_conv_ref`: causal depthwise conv1d over time with a `[K − 1, C]` running state, optional SiLU |
| `gdn.c` | `ath_gdn_gates_ref` (decay and write strength), `ath_gdn_ref` (gated delta rule, recurrent form, state `[nv, dk, dv]` in f32; from step 10 it L2-normalizes q and k itself, with `ath_l2norm_ref`'s expression, and takes q, k, v as rows any stride apart) |
| `sample.c` (step 5) | `ath_sample_ref`: the next token from the logits, below |
| `dispatch.c` (step 8) | `ath_<op>(d, ...)`: the op on device d, or on the CPU without one |

Conventions, stated once in `ops.h`:

- Tensors are `ath_tensor` descriptors: contiguous, row-major, in host memory. Activations and weights are f32 or bf16 in any mix; the dtype of an output is the dtype the result is rounded to.
- Arithmetic and accumulation in fp32, one rounding at the end. An intermediate rounding happens only where the reference does it and the contract records it (the normalized value in `rmsnorm_gated`, the cos/sin tables in `rope`).
- `_ref` functions allocate nothing: the state of `conv` and `gdn` is the caller's tensor, read and updated in place, which is also how decode works.
- Ops know shapes, never models: the zero-centered RMSNorm is `w_offset = 1`, the Gated DeltaNet query scale and L2-norm epsilon are arguments, the L2 norm is a separate op that the family composes before the recurrence.

Every function is tested on hand-computed cases in `tests/test_ops.c` and against the reference implementation on real weights in `tests/test_golden.c` (`tests/README.md`).

## Fast CPU paths (step 4)

`ath_matmul_fast` (`matmul.c`, task 4.7) computes the same sums as `ath_matmul_ref`, in the same order, so its results are the same bits, with any number of threads. The reference is slow because each output is one chain of fp32 additions, and every addition waits for the previous one. The fast path computes 8 outputs at once on each thread. Each one is still its own chain in order of k, with a rounded product and a rounded sum per step, and no fused multiply-add. The chains overlap in the core's pipelines, and blocks of 8 outputs are spread over the threads of an `ath_pool` (`src/core/threads.h`).

- Every row of x passes over a block before the next block starts, so the block's weights stay in cache across the tokens of a prompt.
- The weights keep the checkpoint layout: no copy, no repacking.
- Weights in f32 (only in tests) and the last partial block run one output at a time, as the reference.
- A matmul under 2^18 multiply-adds stays on the calling thread.
- The inner loop reads bf16 weights directly instead of through `ld`. A branch on the dtype inside it stops the compiler from keeping the 8 accumulators in registers and costs half the speed.

| Variant (one token, Ryzen 7 9800X3D, measured once; not a performance claim) | One thread | 8 threads |
| --- | --- | --- |
| `ath_matmul_fast` against `ath_matmul_ref`, 3584 × 1024 and 248320 × 1024 | 3.4× to 3.5× | 12× to 16× |

`tests/test_ops.c` (`matmul_fast_same_bits`) and every matmul record of `tests/test_golden.c` check the bits against the reference. The f32 inputs there would show a fused multiply-add, whose single rounding differs from the reference's two. The faster variant that was measured and not chosen, AVX-512 across outputs on repacked weights, is in `docs/proposals/cpu-matmul-avx512.md`.

## GPU kernels (step 7)

`ath_<op>_gpu(const ath_device *d, ...)` takes the device, then the arguments of `ath_<op>_ref`: the same shapes, dtypes and math, with every tensor in that device's memory (tier `VRAM`). The work is queued on the device's compute stream and the function returns; ops queued one after the other run in that order, and the host synchronizes the stream (or copies on it) before it reads a result.

| File | What it holds |
| --- | --- |
| `<op>.c` | `ath_<op>_gpu` next to `ath_<op>_ref`: the same argument checks (one `args` function per op, called with `OPS_HOST` or `OPS_GPU`), then the launcher |
| `cuda/<op>.cu` | the kernels and the launcher `ath_<op>_launch` (C linkage), which makes the device current and queues the kernels |
| `launch.h` | private: the launchers' declarations, the boundary between the C and the kernels |
| `cuda/kernel.cuh` | private, device code: `k_ld`/`k_st` (f32 and bf16, the host's rounding bit for bit), `k_ld4`/`k_ld8` (4 or 8 elements in one load), `k_warp_sum`, `k_block_sum`, `k_silu`, grid-stride loops |
| `cuda/mma.cuh` (step 10) | private, device code: NVIDIA's warp-level matrix instructions in PTX (`mma.sync` bf16, `ldmatrix`, `cp.async`), their fragment layouts, and bf16 packing; `ATH_HAVE_MMA` is unset under HIP, where the kernels that use them are not compiled |
| `gpu_none.c` | the launchers in a build without a GPU (`make`): `ATH_ERR_UNSUPPORTED`, after the `_gpu` function has checked its arguments |
| `check.h` | `ops_place` (`OPS_PLACE`): every tensor on the host for `_ref` and `_fast`, on the device for `_gpu` |

The kernels, simple and correct first (steps 9 and 10 make them fast):

| Op | Kernel |
| --- | --- |
| `embed` | one thread per output element. The ids are on the device, so the host cannot check them: an id outside the table gives a row of NaN, never a read outside it (`ath_model_forward` checks token ids before the forward pass) |
| `matmul` | cuBLASLt (hipBLASLt on AMD), the "TN" product on the checkpoint layout, fp32 accumulation, never TF32; the alignment of each operand is passed to its heuristic. cuBLASLt takes bf16·bf16 into bf16 or f32, and f32 throughout; the other combinations (f32 activations against bf16 weights, as in the exact forward pass) run on athanor's own kernel, one warp per output |
| `rmsnorm`, `rmsnorm_gated`, `l2norm` | one kernel, one block per row: a block-wide sum of squares, then every thread scales its elements, in the reference's order of operations (and with `rmsnorm_gated`'s rounding of the normalized value) |
| `rope` | one thread per rotated pair or copied element. The frequencies are computed on the host by the reference's own expression and passed in the kernel's arguments: the angle is p · f_i, so a one-ulp difference in the device's `powf` would become p ulps in the angle at long positions |
| `swiglu`, `sigmoid_mul`, `add` | one kernel, one thread per element; `out` may alias an input |
| `attn` | one block per query and head; the keys in tiles of 32 with a running maximum, sum and weighted sum (online softmax, the recurrence of FlashAttention for one query), fp32 throughout; d and dv up to `ATH_ATTN_MAX_DV`. Split-K for decode is step 9, tiled prefill step 10 |
| `conv` | one thread per output, reading `[state; x]`; then a second kernel, one thread per channel, writes the new state (in place, ascending, as the reference) |
| `gdn_gates` | one thread per token and value head |
| `gdn` | one warp per column of a head's state: the column stays in registers (lane l holds rows l, l + 32, ...) for all the tokens of the call, two warp sums per token; dk up to `ATH_GDN_MAX_DK` (256). The chunked form for long prompts is step 10 |

Each kernel is compared with its `_ref` on random inputs (`tests/test_gpu_ops.c`) and on the real inputs of every golden record (`tests/test_golden.c`); `tests/README.md`, GPU kernels, has the tolerances and the measured errors. The results are not the reference's bits: other summation orders, fused multiply-adds, and the device's `expf`, `log1pf`, `cosf`, `sinf`. Copies, the conv state and the residual add are exact, and every kernel sums in an order fixed by the launch shape, so it gives the same bits every time and feeding tokens in pieces gives the bits of feeding them at once (conv, gdn, attention; cuBLASLt's matmul chooses its algorithm by shape, so not there).

Dtypes reach kernels as an argument, and `k_ld`/`k_st` branch on it. Every thread of a launch takes the same branch, so it costs no divergence; kernels specialized per dtype and per shape come with the fast paths of steps 9 and 10, instantiated by macros in one place per op (Rules).

## Decode kernels (step 9)

Decode reads every weight once per token, so its time is the time to read them plus a fixed cost per kernel. Step 9 made the reads stream and cut the kernels. Each change was measured on the RTX 5080 with `athanor bench` and kept only if it paid; the numbers are in `docs/results/2026-11-25-step9-decode/`.

| Op | Kernel |
| --- | --- |
| `matmul`, a few rows | athanor's GEMV for up to 8 rows of x against bf16 weights: a warp per output column (2 or 4 for rows of K >= 1536, by K alone), 16-byte pieces, four in flight per lane, loaded past L1 (`ld.global.nc.L1::no_allocate`, about 6% on decode: x stays in L1), fp32 sums whose order depends on K only. A one-row kernel (72 registers) beside the few-rows one. cuBLASLt keeps the other shapes |
| `ath_matmul_n` | several weights sharing x in one GEMV launch: the projections of one input (q/k/v, qkv/z/a/b, gate/up) read in one pass, with the bits of one call each |
| `ath_matmul_add_gpu` | out += x · wᵀ, the product rounded first: an output projection added to the residual stream without the add kernel, with its bits |
| `ath_matmul_swiglu_gpu` | gate, up and SwiGLU for one row: a warp per column of both, silu(g)·u with g and u rounded first, with the bits of the three ops |
| `attn` | queries by position (`pos`) over a whole cache; keys in pieces of `ATH_ATTN_SPLIT` (64), each reduced to a partial softmax and combined in order; with scratch, up to 8 queries split-K (a block per piece and key head, each key read once for the query heads that share it, then a merge kernel). Heads up to 256: warp w takes 8 keys of a tile and loads them, then their values, at once; the tile's softmax runs head g on warp g, key j on lane j. Rows and split-K share the functions and their rounding intrinsics: the same bits, and decode still gives the bits of the prompt's last row |
| `scatter_rows` | new rows of the cache at their positions, from device memory |
| `sample` | `ath_sample_gpu`: greedy or top-k up to 64 with temperature, top-p, min-p; each block takes its k best in the reference's order, one block merges and runs the reference's double arithmetic |
| `rmsnorm`, `l2norm`, gated | rows up to 1024 in 16-byte pieces take a warp each, with x, weights and gate loaded at once |
| `conv` | one token: output and state in one kernel |
| `gdn` | a lane per column of the state, the rows split over the block's warps: every state row is one 128-byte access; the parts added in warp order |

Tried and dropped, measured: programmatic dependent launch for the GEMV (no gain inside the graph), double-buffered weight loads and eight pieces in flight (no gain), normalizing x inside the GEMV (each block's prologue cost more than the norm kernel it saved), 256-key attention pieces (too few blocks for one query).

## Prompt kernels (step 10)

A prompt is matrix products: cuBLASLt's run near the GPU's bf16 rate, and step 10 made the rest follow. Before it, a prompt of 4096 tokens spent 60% of its time in attention (each query walking its keys alone) and 28% in the Gated DeltaNet (the recurrence, token by token); after it, 53% in cuBLASLt. The numbers are in `docs/results/2026-11-25-step10-prefill/`.

| File, op | Kernel |
| --- | --- |
| `cuda/attn_mma.cu`, `attn` from `ATH_ATTN_MMA_MIN_T` (16) queries in bf16, d = dv up to 256 | FlashAttention's forward pass on the tensor cores: a block of 8 warps takes 128 queries of one head (16 per warp) and walks tiles of 32 keys; scores and the weighted sum with `mma.sync` (bf16 products, fp32 sums), the online softmax in fp32 and base 2, the probabilities rounded to bf16 before they multiply the values (as FlashAttention), and their sum taken over the rounded values, so the weights the output applies add up to one. The queries stay in shared memory (64 KiB), keys and values load by `cp.async` while the other half of the tile is computed, the 16-byte chunks of every row permuted by the row (swizzle) so that `ldmatrix` reads 8 rows from 8 banks. Key tiles start at multiples of 32 whatever the query rows, so a query's output depends on its own position only: pieces of a prompt give the bits of the prompt at once. Two warps per scheduler let one's softmax overlap the other's products |
| `cuda/gdn_chunk.cu`, `gdn` from `ATH_GDN_CHUNK_MIN_T` (64) tokens with scratch, dk up to 128 | the chunked form of the reference (`torch_chunk_gated_delta_rule`, chunks of 64), fp32 throughout. `prep`, a block per chunk and value head: q and k normalized (L2) as they arrive, the in-chunk decays, A = β (k_i · k_j) Γ_ij and M = q_scale (q_i · k_j) Γ_ij, T = (I + A)⁻¹ by blocks of 16, W = T (β e^γ K), U = T (βV), Q'' = q_scale e^γ Q − M W, O_U = M U; matrix products of 8 × 8 tiles per lane, so that a load from shared memory feeds 16 multiply-adds. `scan`, a block per value head and 32 state columns, walks the chunks: O = Q'' S + O_U, Δ = U − W S, S ← e^γ_last S + Kᵀ(e^(γ_last − γ) Δ / \|k\|). The tokens go in rounds that fill the GPU four times with prep blocks and stay in L2; each launch runs the scan of one round (64 blocks) and the prep of the next on the multiprocessors the scan leaves free. A prompt fed in pieces of whole chunks gives the bits of the prompt at once |
| `gdn` (both forms) | q and k are normalized inside the op (step 10): the normalized values are never written. q, k and v may be rows any stride apart, so the forward pass passes column ranges of the convolution's output without copying them |
| `cuda/conv.cu`, a prompt | 8 tokens of 256 channels per block, a thread's inputs loaded at once, no index divided: the same bits |
| `cuda/norm.cu`, `ath_add_rmsnorm_gpu` | the residual add fused into the next RMSNorm (x += y rounded and stored as `ath_add`, then the norm of what was stored), the bits of the two ops; rows of at most 16 pieces take 16 or 8 lanes, two or four rows to a warp |

Every kernel of the table matches its reference within the tolerances of `tests/README.md`, Prompt kernels. The attention's bf16 probabilities move each weight by a relative 2^-8 at most; the chunked recurrence is fp32 with another order of operations; the rest keeps the bits.

Measured and dropped: adding a prompt's output projection inside cuBLASLt (beta = 1, 1.2 ms of pp4096) rounds once where the reference rounds twice and moved one greedy token above the noise in step 8's gate; choosing cuBLASLt's algorithm by timing its candidates (0 to 9% per shape, about 1.3 ms of pp4096) would make the bits depend on the timings; a plan cache for cuBLASLt (prompts are GPU-bound: the host queues 10.2 ms of kernels in less time than they run); not inlining the triangular product (its 8 × 8 tile went through local memory, 8 times slower). Kernels are kept small on purpose: the chunked prep was 9.7k instructions, more than the instruction cache, and code that runs once per block (stores of 64 values, IEEE divisions) ran at the speed of fetching it from L2; written as loops it went from 145 to 130 µs.

## Q8_0 (step 12)

Q8_0 weights (`core/tensor.h`, `ATH_DTYPE_Q8_0`: blocks of 32 elements, an f16 scale and 32 int8) are summed in one canonical order in every implementation, so that the CPU and the GPU give the same bits (AGENTS.md invariant 7, option (b) decided at step 12): inside a block a chain of fused multiply-adds over its 32 elements, 16 partial sums taking the blocks in turn (block b into partial b mod 16) with one fused multiply-add of the block's scale, and a fixed tree over the 16 partials (`q8_0.h`). The same header holds the exact f16 conversion and `ath_q8_expf`, an exponential written with IEEE operations only (fma, floor, bit assembly), within 0.89 ulp of exp on [-87, 88] and identical on both sides, for SiLU inside the experts.

| File | Content |
| --- | --- |
| `q8_0.h` | private, C and CUDA: `ath_f16_to_f32`, `ath_q8_0_scale`, the canonical tree, `ath_q8_expf`, `ath_q8_silu` |
| `matmul.c` | `ath_matmul_ref` and `ath_matmul_fast` take a Q8_0 weight: the canonical order, the fast path computing 8 outputs at a time with each block of x widened once |
| `embed.c` | a Q8_0 table: d · q, exact in f32, then the output's rounding |
| `cuda/q8_0.cuh` | private, device code: a half-warp per dot product (lane j takes the blocks b with b mod 16 = j, the xor butterfly is the canonical tree); 16 blocks (544 bytes) staged in shared memory with coalesced 16-byte loads, each lane reading its block as words shifted into place |
| `cuda/matmul_q8_0.cu` | `q8_gemv` for up to 8 rows (decode: the output head, 509 MB in Q8_0, at 897 GB/s, the bf16 GEMV's 907), `q8_tile` for prompts (32 columns × 16 rows per block, the activations transposed in shared memory so that 4 loads feed 16 fused multiply-adds: about 9 TFLOPS, a twelfth of cuBLASLt's bf16, enough for step 12, where prefill has no gate), `q8_any` for K not a multiple of 512 or unaligned weights (one lane reads its own block) |
| `cuda/embed.cu` | the Q8_0 lookup |

All of them give `ath_matmul_ref`'s bits for any number of rows (`tests/test_ops.c`, `tests/test_gpu_ops.c`).

## MoE (step 12)

The mixture-of-experts layer (docs/contract/qwen35-35b-a3b.md, MoE op contract) as three ops and the machinery that splits the experts between the GPU and the CPU.

| File | Content |
| --- | --- |
| `moe.c` | `ath_moe_route_ref` / `_gpu`: softmax in fp32 over the experts, the top k (ties: the lower index), renormalized, sorted by expert index (the reference's order of addition); `ath_moe_expert_ref`: one expert's SwiGLU MLP, its matmuls in the canonical Q8_0 order, SiLU by `ath_q8_silu`; `ath_moe_combine_ref` / `_gpu`: the weighted sum as a chain of fused multiply-adds by expert index, plus the gated shared expert, one rounding; `ath_moe_experts_gpu` (the experts in device memory); `ath_moe_ws` and `ath_moe_experts_split`: a layer's experts on the GPU and the CPU at once |
| `moe_cpu.c` | `ath_q8_0_repack` (Q8_0 → Q8_0_R16) and `ath_moe_experts_cpu`: the experts away from the device, the items sorted by expert so that a row of weights is read once for all the tokens that chose it, rows spread over the pool; AVX-512 (16 partial sums in one register, a group's values in one 16-byte load, the tree by shuffles) or the same operations one lane at a time |
| `cuda/moe.cu` | `route_kernel` (a warp per token), `combine_kernel`, `expert_up` and `expert_down` (a half-warp per row, the canonical dot of `q8_0.cuh` for Q8_0 or Q8_0_R16 weights; an expert absent from the table is skipped, a block whose rows all belong to such experts leaves at once), `merge_kernel` (the CPU's outputs, read from pinned memory over the bus, where the expert is not on the device) |

How a layer runs on a GPU (`ath_moe_experts_split`): the compute stream records an event; the device-to-host stream waits for it, copies the layer's normalized input and the router's choices to pinned memory and queues the CPU's share (`ath_host_call`: `ath_moe_experts_cpu` on the experts whose flag says they are not on the device); meanwhile the compute stream runs the device's experts, then waits for the CPU and merges its outputs. Nothing waits on the host, so a decode step that contains it is captured in a graph and replayed; a replay runs the CPU's share again, for wherever the experts are then. The CPU's jobs of captured passes are kept for the graph's life; the others come from a ring. Tokens go in chunks of the working state's size.

The expert path has one canonical order everywhere, so the bits do not depend on placement: `tests/test_gpu_ops.c` (`gpu_moe_split`) checks the split against the reference, bit for bit, directly and in a graph replayed after experts moved; `tests/test_engine.c` (`engine_moe_placement`) checks whole passes of Qwen3.6-35B-A3B.

## Op list

| Op | Content | Step |
| --- | --- | --- |
| `embed` | token embedding lookup; Q8_0 tables (step 12) | 3, 7 (GPU), 12 |
| `norm` | RMSNorm (parameterization per contract), gated RMSNorm, L2 norm; the residual add fused into RMSNorm on the GPU | 3, 7 (GPU), 10 |
| `rope` | rotary embedding, full or partial | 3, 7 (GPU) |
| `act` | SwiGLU, sigmoid gates, residual add | 3, 7 (GPU) |
| `matmul` | dense GEMM/GEMV for bf16 and f32 (cuBLASLt on the GPU); Q8_0 weights in the canonical order (step 12); the other formats from phase 2 | 3 (ref), 4 (fast CPU), 7 (GPU), 9–10 (fast GPU), 12 (Q8_0) |
| `attn` | causal attention with GQA: naive reference; split-K decode and tensor-core prefill on GPU | 3, 7, 9, 10 |
| `conv` | short depthwise causal conv1d with state | 3, 7 (GPU) |
| `scatter` | rows written at given rows (the KV cache by position) | 9 |
| `gdn` | Gated DeltaNet: gates; recurrent reference (CPU and GPU decode), chunkwise-parallel GPU prefill; L2 norm of q and k inside (step 10) | 3, 7, 10 |
| `moe` | router top-k, routed experts (canonical order: the same bits on CPU and GPU), weighted combination with the gated shared expert; the experts split between GPU and CPU | 12 |
| `sample` | greedy, temperature, top-k, top-p, min-p (step 5); GPU sampling, greedy and top-k up to 64 (step 9); presence and repetition penalties later | 5, 9 |
| `quant` | (de)quantization per format | Q1 |

Backward functions for each op are added in phase 4 (T1) in the same files, with gradient checks.

## Sampling (step 5)

`ath_sample_ref` chooses the next token from one row of logits. Greedy (temperature 0, or top-k 1) takes the largest logit, the lowest id among equals. Otherwise the candidates are ordered from the most likely and filtered in the order of transformers' logits warpers: temperature, top-k (a sorted list of the k best kept while scanning, about one comparison per token for small k), top-p (the shortest prefix whose probability reaches p, at least one token), min-p (drop tokens below min-p times the best). The token is where `u` falls in the cumulative distribution of the kept candidates. `u` is a uniform number in [0, 1) that the caller draws from its own generator (`ath_rng`, `src/core/rng.h`), so the op is deterministic and its tests are hand-computed: `tests/test_text.c` (`sample_rules`, and `sample_distribution`, 40,000 draws with a fixed seed against the softmax). The reference implementation records no sampling, so there is no golden data; the GPU kernel comes with GPU sampling in step 9.

## Rules

- Kernels are parameterized by shape; specializations for hot shapes (head dimension, group size) are instantiated by macros in one place per op.
- Fused or model-specific kernels do not belong here: they go in `src/models/<family>/cuda/fused.cu`.
- The `_ref` functions are the truth. Optimizations go in `_fast` or the kernels, never in `_ref`.
