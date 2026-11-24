# src/ops

Every numeric operator in athanor. Each op is a pair of files plus tests:

```
src/ops/<op>.c    <op>_ref(): scalar, fp32 accumulation, obviously correct, never optimized
                  <op>_fast(): optional SIMD/multithreaded CPU path, only for ops used on CPU at runtime
src/ops/<op>.cu   GPU kernel(s), CUDA C compiled for CUDA and HIP
tests/test_ops.c  ref vs golden data; GPU vs ref; fast vs ref
```

`ops.h` declares the functions with their shapes, dtypes and math. Step 3 landed the CPU references; the entry point per op that dispatches on the device of its arguments joins `ops.h` with the second implementation of an op (the GPU kernels, step 7). Until then callers use the `_ref` functions.

## The references (step 3)

| File | Functions |
| --- | --- |
| `ops.h` | every signature, with shapes, dtypes and math; the conventions below |
| `check.h` | private: argument checks shared by the `_ref` functions, element access (`ld`, `st`) |
| `embed.c` | `ath_embed_ref` |
| `matmul.c` | `ath_matmul_ref`: `x · wᵀ` against a weight in checkpoint layout `[M, K]` |
| `norm.c` | `ath_rmsnorm_ref` (plain or zero-centered: `w_offset` 0 or 1), `ath_rmsnorm_gated_ref` (SiLU gate), `ath_l2norm_ref` |
| `rope.c` | `ath_rope_ref`: rotate-half on the first `rope_dim` dimensions of each head |
| `act.c` | `ath_swiglu_ref`, `ath_sigmoid_mul_ref`, `ath_add_ref` |
| `attn.c` | `ath_attn_ref`: causal softmax attention with grouped-query heads, the queries being the last T of S positions |
| `conv.c` | `ath_conv_ref`: causal depthwise conv1d over time with a `[K − 1, C]` running state, optional SiLU |
| `gdn.c` | `ath_gdn_gates_ref` (decay and write strength), `ath_gdn_ref` (gated delta rule, recurrent form, state `[nv, dk, dv]` in f32) |

Conventions, stated once in `ops.h`:

- Tensors are `ath_tensor` descriptors: contiguous, row-major, in host memory. Activations and weights are f32 or bf16 in any mix; the dtype of an output is the dtype the result is rounded to.
- Arithmetic and accumulation in fp32, one rounding at the end. An intermediate rounding happens only where the reference does it and the contract records it (the normalized value in `rmsnorm_gated`, the cos/sin tables in `rope`).
- `_ref` functions allocate nothing: the state of `conv` and `gdn` is the caller's tensor, read and updated in place, which is also how decode works.
- Ops know shapes, never models: the zero-centered RMSNorm is `w_offset = 1`, the Gated DeltaNet query scale and L2-norm epsilon are arguments, the L2 norm is a separate op that the family composes before the recurrence.

Every function is tested on hand-computed cases in `tests/test_ops.c` and against the reference implementation on real weights in `tests/test_golden.c` (`tests/README.md`).

## Op list

| Op | Content | Step |
| --- | --- | --- |
| `embed` | token embedding lookup | 3 |
| `norm` | RMSNorm (parameterization per contract), gated RMSNorm, L2 norm | 3 |
| `rope` | rotary embedding, full or partial | 3 |
| `act` | SwiGLU, sigmoid gates, residual add | 3 |
| `matmul` | dense GEMM/GEMV for bf16 and f32; per-format GEMV/GEMM from phase 2 | 3 (ref), 7 (GPU), 9–10 (fast) |
| `attn` | causal attention with GQA: naive reference; tiled prefill and split-K decode on GPU | 3, 7, 9, 10 |
| `conv` | short depthwise causal conv1d with state | 3 |
| `gdn` | Gated DeltaNet: gates; recurrent reference (CPU and GPU decode), chunkwise-parallel GPU prefill | 3, 7, 10 |
| `moe` | router top-k, expert dispatch, grouped GEMM, shared expert | 12 |
| `sample` | temperature, top-k, top-p, min-p, repetition penalty | 5, 9 |
| `quant` | (de)quantization per format | Q1 |

Backward functions for each op are added in phase 4 (T1) in the same files, with gradient checks.

## Rules

- Kernels are parameterized by shape; specializations for hot shapes (head dimension, group size) are instantiated by macros in one place per op.
- Fused or model-specific kernels do not belong here: they go in `src/models/<family>/fused.cu`.
- The `_ref` functions are the truth. Optimizations go in `_fast` or `.cu`, never in `_ref`.
