# src/core

The bottom layer: memory, tensors, files, parsing, threads. No GPU code and no knowledge of models. Introduced in step 1.

| File | Content |
| --- | --- |
| `base.h` | fixed-width types, `ath_status`, `ATH_CHECK`, logging, assertions |
| `arena.c/.h` | lifetime-scoped linear arenas (weights, per-request, per-step scratch) |
| `tensor.h` | `ath_dtype`, `ath_tensor` descriptor, views |
| `file.c/.h` | read-only `mmap`, `pread`; async reads later (io_uring via raw syscalls) |
| `json.c/.h` | read-only JSON parser into an arena |
| `safetensors.c/.h` | single and multi-shard readers, zero-copy views; writer added in phase 4 for checkpoint export |
| `threads.c/.h` | fixed pthread pool for CPU work (experts, dequantization, I/O completion) |
| `rng.c/.h` | small deterministic PRNG for sampling, tests and training |

## Memory tiers

Every tensor lives in one tier: `ATH_TIER_VRAM`, `ATH_TIER_PINNED`, `ATH_TIER_RAM`, `ATH_TIER_DISK`. A placement map records where each tensor is; kernels never branch on tier, the engine moves data before launching them. Tier allocators and budgets arrive with step 8 (VRAM) and step 12 (pinned RAM ring, disk).

## Rules

- No `malloc` in hot paths; arenas only.
- Weights stay where `mmap` put them unless the engine decides to place them elsewhere; no hidden copies.
