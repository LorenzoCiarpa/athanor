# src/ops

Every numeric operator in athanor. Each op is a pair of files plus tests:

```
src/ops/<op>.c    <op>_ref(): scalar, fp32 accumulation, obviously correct, never optimized
                  <op>_fast(): optional SIMD/multithreaded CPU path, only for ops used on CPU at runtime
src/ops/<op>.cu   GPU kernel(s), CUDA C compiled for CUDA and HIP
tests/test_ops.c  ref vs golden data; GPU vs ref; fast vs ref
```

`ops.h` declares one entry point per op that dispatches on the device of its arguments.

## Op list

| Op | Content | Step |
| --- | --- | --- |
| `embed` | token embedding lookup | 3 |
| `norm` | RMSNorm (parameterization per contract), L2 norm | 3 |
| `rope` | rotary embedding, full or partial | 3 |
| `act` | SiLU, SwiGLU, sigmoid gates | 3 |
| `matmul` | dense GEMM/GEMV for bf16 and f32; per-format GEMV/GEMM from phase 2 | 3 (ref), 7 (GPU), 9–10 (fast) |
| `attn` | causal attention with GQA: naive reference; tiled prefill and split-K decode on GPU | 3, 7, 9, 10 |
| `conv` | short depthwise causal conv1d with state | 3 |
| `gdn` | Gated DeltaNet: recurrent reference (CPU and GPU decode), chunkwise-parallel GPU prefill | 3, 7, 10 |
| `moe` | router top-k, expert dispatch, grouped GEMM, shared expert | 12 |
| `sample` | temperature, top-k, top-p, min-p, repetition penalty | 5, 9 |
| `quant` | (de)quantization per format | Q1 |

Backward functions for each op are added in phase 4 (T1) in the same files, with gradient checks.

## Rules

- Kernels are parameterized by shape; specializations for hot shapes (head dimension, group size) are instantiated by macros in one place per op.
- Fused or model-specific kernels do not belong here: they go in `src/models/<family>/fused.cu`.
- The `_ref` functions are the truth. Optimizations go in `_fast` or `.cu`, never in `_ref`.
