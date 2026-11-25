# src/core

The bottom layer: memory, tensors, files, parsing, threads. No GPU code and no knowledge of models. Introduced in step 1.

| File | Content | Step |
| --- | --- | --- |
| `base.h`, `base.c` | fixed-width types, `ath_status` and `ath_status_str` (`ATH_ERR_DEVICE` since step 6), `ATH_CHECK`, logging (`ath_log`, `ATH_LOG` environment variable), `ATH_ASSERT` | 1, 6 |
| `arena.h`, `arena.c` | lifetime-scoped linear arenas (weights, per-request, per-step scratch): chained blocks, alignment, mark/reset | 1 |
| `tensor.h`, `tensor.c` | `ath_dtype`, `ath_tier`, `ath_tensor` descriptor, slice and reshape views; bf16 conversion (round to nearest even, as torch) and element access for f32 and bf16 (step 3) | 1, 3 |
| `file.h`, `file.c` | read-only `mmap`, `pread`, path join; async reads later (io_uring via raw syscalls) | 1 |
| `json.h`, `json.c` | read-only JSON parser into an arena; lookup by key, index and dotted path; a writer with the formatting of Python's `json.dumps(ensure_ascii=False)` (step 5), because chat templates serialize tool definitions and arguments with it: `", "` and `": "`, Python's escapes and its shortest float repr (`1e-05`, `1e+16`) | 1, 5 |
| `buf.h`, `buf.c` | `ath_buf`: a growable byte buffer in an arena, always NUL-terminated (rendered prompts, decoded text, JSON) | 5 |
| `safetensors.h`, `safetensors.c` | single and multi-shard readers, zero-copy views; writer added in phase 4 for checkpoint export | 1 |
| `threads.c/.h` | fixed pthread pool for CPU work: `ath_pool_init`, `ath_pool_run` (fork and join, tasks assigned statically, runs from several threads take turns), `ath_pool_free`; first user `ath_matmul_fast`; later experts, dequantization, I/O completion | 4 |
| `sha256.h`, `sha256.c` | SHA-256 (FIPS 180-4), streaming and of a whole file through its mapping; the weight files of benchmark reports (step 9), later the keys of on-disk prefix snapshots (step 14) | 9 |
| `rng.h`, `rng.c` | `ath_rng`: xoshiro256** seeded by splitmix64, the same sequence for the same seed on every machine; the state belongs to the caller. First user: sampling in `athanor chat` | 5 |

## Conventions

- Headers that GPU code includes (`base.h`, `arena.h`; `tensor.h` and `threads.h` since step 7, through `ops.h`) wrap their declarations in `extern "C"`, because nvcc compiles `.cu` files as C++ (`src/backend/README.md`).

- Every fallible function returns `ath_status`; `ATH_OK` is 0. Propagate with `ATH_CHECK(expr)`; with `ATH_LOG=debug` each hop is logged, which reads like a backtrace.
- Allocation functions (`ath_arena_alloc`, `ath_arena_zalloc`, `ath_arena_strndup`) return `NULL` on failure; callers map it to `ATH_ERR_NOMEM`.
- Library code logs the context of an error (file, tensor, offsets) at `error` level and returns a status; it never exits. Default verbosity is `warn`; `ATH_LOG=none|error|warn|info|debug|trace` (or `0`–`5`) changes it.
- `ATH_ASSERT` checks invariants only in `make DEBUG=1` builds, never input.

## Loading a checkpoint

```c
ath_arena a;
ath_arena_init(&a, 0);
ath_safetensors st;
ATH_CHECK(ath_safetensors_open(&st, &a, "models/Qwen3.5-0.8B"));   /* index + shards, or model.safetensors */
const ath_safetensors_entry *e = ath_safetensors_find(&st, "model.language_model.embed_tokens.weight");
/* e->t is an ath_tensor whose data points into the read-only mapping of the shard */
ath_safetensors_close(&st);
ath_arena_free(&a);
```

Opening validates the format the way the reference implementation does: known dtypes, shapes consistent with byte ranges, ranges inside the data section, tensors covering each data section exactly (no gaps, no overlaps), no duplicate names, and for sharded checkpoints a one-to-one match between the index and the shard headers. Tensors are sorted by name with digit runs compared as numbers (`layers.2` before `layers.10`); lookups are binary searches.

## Memory tiers

Every tensor lives in one tier: `ATH_TIER_VRAM`, `ATH_TIER_PINNED`, `ATH_TIER_RAM`, `ATH_TIER_DISK`. Memory-mapped checkpoint data is `ATH_TIER_RAM`: host-addressable and paged in from the file on first touch. A placement map records where each tensor is; kernels never branch on tier, the engine moves data before launching them. Tier allocators and budgets arrive with step 8 (VRAM) and step 12 (pinned RAM ring, disk).

## Rules

- No `malloc` in hot paths; arenas only. `arena.c` is the only host file that calls `malloc`; `make check-alloc` verifies that `athanor inspect` allocates nothing else.
- No `qsort`: glibc's may allocate a merge buffer. `safetensors.c` carries a small heapsort.
- Weights stay where `mmap` put them unless the engine decides to place them elsewhere; no hidden copies.
