# src/quant

Weight formats, the algorithms that produce them, the container that stores them, and the tools that measure their cost in quality. Phase 2. Kernels that read the formats live in `src/ops/cuda/matmul.cu` and `src/ops/cuda/quant.cu`.

| File | Content |
| --- | --- |
| `formats.c/.h` | format definitions, packing and unpacking |
| `rtn.c` | round-to-nearest, plain and importance-weighted |
| `gptq.c` | GPTQ-class quantization with Hessian-based error compensation, one layer at a time |
| `awq.c` | activation-aware per-channel scaling |
| `recipe.c` | per-tensor format choice by role and flags; size-budgeted search |
| `container.c` | `.ath` files: JSON header (model revision, recipe, per-tensor format, offsets, hashes) + 4 KiB-aligned data, mmap-able |
| `gguf.c` | GGUF import and export |
| `eval.c` | KL divergence vs bf16, top-1 agreement, perplexity |

## Formats

| Format | Layout | Use |
| --- | --- | --- |
| `bf16` | reference | golden path, training masters |
| `q8` | 8-bit, blocks of 32, fp16 scale | safe default |
| `q4`, `q3`, `q2` | super-blocks with sub-block scales | consumer inference, cold training weights |
| `fp8_e4m3` | block-scaled | prefill on Blackwell, KV cache |
| `nvfp4` | FP4 E2M1, 16-element blocks, FP8 E4M3 block scale, fp32 tensor scale | Blackwell tensor cores |
| `mxfp4` | FP4 E2M1, 32-element blocks, E8M0 scale | cross-vendor FP4 |

## Generic by construction

The recipe chooses formats from roles and flags (`ATH_P_SENSITIVE` defaults to higher precision: norms, router, gates, short-conv weights; embeddings, lm_head and shared experts are configurable). Calibration statistics come from the `on_matmul_in` hook. Nothing in this module knows which family it is quantizing.

Choosing a format per tensor under a size budget is a knapsack problem: each tensor has a byte cost and a measured damage (its KL contribution) per candidate format. `athanor quant plan` solves it; `athanor quant apply` writes the model; `athanor quant eval` writes the report.
