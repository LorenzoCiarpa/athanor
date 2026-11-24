# src/ops

Every numeric operator in athanor. Each op is a pair of files plus tests:

```
src/ops/<op>.c    <op>_ref(): scalar, fp32 accumulation, obviously correct, never optimized
                  <op>_fast(): optional SIMD/multithreaded CPU path, only for ops used on CPU at runtime
src/ops/<op>.cu   GPU kernel(s), CUDA C compiled for CUDA and HIP
tests/test_ops.c  ref vs golden data; GPU vs ref; fast vs ref
```

`ops.h` declares the functions with their shapes, dtypes and math. Step 3 landed the CPU references; step 4 the first fast CPU path, `ath_matmul_fast`. The entry point per op that dispatches on the device of its arguments joins `ops.h` with the GPU kernels (step 7). Until then callers use the `_ref` functions, and the forward pass uses `ath_matmul_fast` for its matmuls.

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
| `gdn.c` | `ath_gdn_gates_ref` (decay and write strength), `ath_gdn_ref` (gated delta rule, recurrent form, state `[nv, dk, dv]` in f32) |
| `sample.c` (step 5) | `ath_sample_ref`: the next token from the logits, below |

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

## Op list

| Op | Content | Step |
| --- | --- | --- |
| `embed` | token embedding lookup | 3 |
| `norm` | RMSNorm (parameterization per contract), gated RMSNorm, L2 norm | 3 |
| `rope` | rotary embedding, full or partial | 3 |
| `act` | SwiGLU, sigmoid gates, residual add | 3 |
| `matmul` | dense GEMM/GEMV for bf16 and f32; per-format GEMV/GEMM from phase 2 | 3 (ref), 4 (fast CPU), 7 (GPU), 9–10 (fast GPU) |
| `attn` | causal attention with GQA: naive reference; tiled prefill and split-K decode on GPU | 3, 7, 9, 10 |
| `conv` | short depthwise causal conv1d with state | 3 |
| `gdn` | Gated DeltaNet: gates; recurrent reference (CPU and GPU decode), chunkwise-parallel GPU prefill | 3, 7, 10 |
| `moe` | router top-k, expert dispatch, grouped GEMM, shared expert | 12 |
| `sample` | greedy, temperature, top-k, top-p, min-p (step 5); presence and repetition penalties later; GPU sampling | 5, 9 |
| `quant` | (de)quantization per format | Q1 |

Backward functions for each op are added in phase 4 (T1) in the same files, with gradient checks.

## Sampling (step 5)

`ath_sample_ref` chooses the next token from one row of logits. Greedy (temperature 0, or top-k 1) takes the largest logit, the lowest id among equals. Otherwise the candidates are ordered from the most likely and filtered in the order of transformers' logits warpers: temperature, top-k (a sorted list of the k best kept while scanning, about one comparison per token for small k), top-p (the shortest prefix whose probability reaches p, at least one token), min-p (drop tokens below min-p times the best). The token is where `u` falls in the cumulative distribution of the kept candidates. `u` is a uniform number in [0, 1) that the caller draws from its own generator (`ath_rng`, `src/core/rng.h`), so the op is deterministic and its tests are hand-computed: `tests/test_text.c` (`sample_rules`, and `sample_distribution`, 40,000 draws with a fixed seed against the softmax). The reference implementation records no sampling, so there is no golden data; the GPU kernel comes with GPU sampling in step 9.

## Rules

- Kernels are parameterized by shape; specializations for hot shapes (head dimension, group size) are instantiated by macros in one place per op.
- Fused or model-specific kernels do not belong here: they go in `src/models/<family>/fused.cu`.
- The `_ref` functions are the truth. Optimizations go in `_fast` or `.cu`, never in `_ref`.
