# Status

Updated by whoever completes a task (human or agent). Never delete lines: mark them done, superseded or dropped.

## Current step

**Step 2: contract and family loading** (see `docs/PLAN.md` §4). Started 2026-11-24 on branch `step2-contract`.

Goal: give every tensor of the checkpoint a meaning that pipelines can use without knowing the family: a role, a layer, an expert index and flags, plus the layer schedule. No model math yet.

### Tasks

- [x] **2.1 Reference and op contract.** Pin the reference implementation (transformers 5.24.0, file hashes) and write the op contract of Qwen3.5-0.8B in `docs/contract/qwen35-0.8b.md`: dtype boundaries, norms, RoPE, output gate, Gated DeltaNet, state layout, every **verify** item of `docs/MODELS.md` that applies. Record the MoE naming and router contract from the real Qwen3.5-35B-A3B headers in `docs/contract/qwen35-35b-a3b.md`.
- [x] **2.2 `src/model/model.h`.** Roles, param flags, `ath_param`, the layer schedule (`ath_layer`), ignored groups, `ath_model`, `ath_family` (`name`, `match`, `load`); `docs/ARCHITECTURE.md` updated to the real header.
- [x] **2.3 `src/model/registry.c`.** Families compiled in from the Makefile `FAMILIES` list (no family names in the file); `ath_model_open(dir)` reads `config.json`, picks the family whose `match` accepts it, opens the checkpoint, calls `load`, then checks that every checkpoint tensor is bound to a param or belongs to a declared ignored group, and that every param views memory of the tensor it names.
- [x] **2.4 `src/models/qwen35/config.c`.** `config.json` → dimensions, layer schedule (`layer_types`, or `full_attention_interval`), MoE parameters; reject values the reference does not support (bias, non-default RoPE, `mlp_only_layers`, activations other than SiLU).
- [x] **2.5 `src/models/qwen35/weights.c`.** Bind every language-model tensor with role, layer, expert and flags; check dtype and shape against the config; fused expert tensors become one view per expert; vision encoder and MTP head are declared ignored groups.
- [x] **2.6 `athanor inspect --roles`.** Family, dimensions, schedule, per-layer totals, one row per param, totals per role, ignored groups, coverage check; `--json` and `--filter` work with it.
- [x] **2.7 `tests/test_model.c`.** Synthetic dense, MoE and text-only checkpoints; every failure mode (missing, unexpected, misshaped, wrong dtype, tied with `lm_head`, unsupported config); the real checkpoint.

### Acceptance criteria

1. `athanor inspect models/Qwen3.5-0.8B --roles` binds all 320 language-model tensors (role, layer, flags), lists the 168 vision and MTP tensors as ignored, prints the schedule (18 Gated DeltaNet, 6 full attention, matching `layer_types`) and exits 0.
2. A missing, unexpected or misshaped tensor makes loading fail with a message naming it (tests).
3. `make test` green with zero warnings, also with `DEBUG=1`.
4. `make check-alloc` also covers `inspect --roles`: no heap allocation outside arenas.
5. No model names outside `src/models/` (the Makefile `FAMILIES` list excepted).
6. Contract files, module READMEs, `docs/ARCHITECTURE.md` and this file updated.

## Completed steps

### Step 1

**Step 1: core and `athanor inspect`** (see `docs/PLAN.md` §4). **Done 2026-11-23** on branch `step1-core-inspect` (pushed, to be merged into `main`); step 2 is next. Merged into `main` (fast-forward) on 2026-11-24.

Goal: read a Qwen3.5-0.8B checkpoint from disk without copying it and describe every tensor. No GPU, no model math yet.

#### Tasks

- [x] **1.1 Build skeleton.** `Makefile` with targets `cpu` (default), `cuda`, `hip`, `test`, `clean`; output in `build/`; flags `-std=c11 -O2 -Wall -Wextra -Werror -D_GNU_SOURCE`; `FAMILIES` variable (default `qwen35`). `cuda` and `hip` may be stubs that fail with a clear message until step 6.
- [x] **1.2 `src/core/base.h`.** fixed-width types, `ath_status` enum with `ath_status_str()`, `ATH_CHECK(expr)`, `ath_log(level, fmt, ...)` with levels and an environment variable for verbosity, `ATH_ASSERT` for debug builds.
- [x] **1.3 `src/core/arena.{h,c}`.** linear arena with alignment, `ath_arena_alloc`, `ath_arena_mark/reset`, growth by chained blocks; tests.
- [x] **1.4 `src/core/file.{h,c}`.** read-only `mmap` of a file, size, `pread` helper, close; errors mapped to `ath_status`.
- [x] **1.5 `src/core/json.{h,c}`.** read-only JSON parser: objects, arrays, strings with all escapes including `\uXXXX` and surrogate pairs, numbers (int64 and double), booleans, null; nodes allocated in an arena; lookup by key and by dotted path; tests on real `config.json`, `tokenizer_config.json` and safetensors headers.
- [x] **1.6 `src/core/tensor.h`.** `ath_dtype` (f32, f16, bf16, i32, i64, u8, f8_e4m3, f8_e5m2; quantized formats reserved), element size, `ath_tensor` descriptor (dtype, ndim, shape, stride, tier, data), view helpers.
- [x] **1.7 `src/core/safetensors.{h,c}`.** header = 8-byte little-endian length + JSON; dtype mapping; shapes; data offsets validated against file size; multi-shard models through `model.safetensors.index.json`; zero-copy tensor views over the mmap.
- [x] **1.8 `tools/athanor.c`.** subcommand dispatch table; `athanor inspect <model_dir> [--json] [--filter <substring>]` printing a config summary (model type, layers, hidden size, vocab), a table of tensors (name, dtype, shape, bytes) and totals per dtype.
- [x] **1.9 `tests/test_core.c`** with a minimal runner (`ATH_TEST(name)`, assertions, summary); synthetic safetensors file generated by the test itself.

#### Acceptance criteria

1. `make && ./build/athanor inspect <path>/Qwen3.5-0.8B` lists every tensor; the sum of tensor bytes equals the sum of the data sections of the shard files.
2. `make test` is green with zero warnings.
3. `ldd build/athanor` shows only the C library, libm and pthreads.
4. No heap allocation in `inspect` outside arenas.
5. Module READMEs and this file updated.

Result (2026-11-23, Qwen3.5-0.8B revision `2fc0636`, gcc 15.2, glibc 2.43):

| # | Criterion | Result |
| --- | --- | --- |
| 1 | every tensor listed; tensor bytes == data sections | 488 tensors listed; 1,746,882,752 == 1,746,882,752 (also == index `total_size`); names, dtypes, shapes and offsets identical to a Python read of the header |
| 2 | `make test` green, zero warnings | 22 unit tests pass (release and `DEBUG=1` with ASan/UBSan); no warnings under `-Wall -Wextra -Werror` |
| 3 | `ldd build/athanor` | `linux-vdso`, `libc.so.6`, `ld-linux` only (libm and libpthread are not needed; pthreads live in libc since glibc 2.34) |
| 4 | no heap allocation outside arenas | `make check-alloc`: one allocation during `inspect`, the arena's first block, from `ath_arena_alloc` (LD_PRELOAD trace, `tools/dev/check_alloc.sh`) |
| 5 | READMEs and STATUS updated | yes; contract started in `docs/contract/qwen35-0.8b.md` |

#### Notes for this step

- Download the checkpoint into `models/` (ignored by git). Downloading is a development task and may use any tool.
- Record in `docs/contract/qwen35-0.8b.md` the exact repository revision used, and list the tensor names grouped by layer type as `inspect` reports them. This file becomes the op contract in step 2.
- Done: checkpoint downloaded with `hf download` (huggingface_hub 2.4.0, installed with `uv tool install huggingface_hub`); revision and tensor groups recorded in `docs/contract/qwen35-0.8b.md`.

## Next step (preview)

**Step 3: CPU reference ops and golden harness.** `src/ops/*.c` references (scalar, fp32 accumulation) for every op in the contract, `src/backend/cpu.c`, `tests/golden/dump.py` against the pinned reference, `tests/test_ops.c`. Gate: every op within tolerance of golden tensors on real weights.

Superseded preview (written in step 1): **Step 2: contract and family loading.** Write `src/model/model.h` as in `docs/ARCHITECTURE.md`, `registry.c`, and `src/models/qwen35/{config.c,weights.c}`: parse `config.json` into the layer schedule (which layers are Gated DeltaNet, which are full attention, MLP or MoE), bind every tensor to an `ath_param` with role, layer, expert and flags, and fail loudly on any unbound or missing tensor. `athanor inspect --roles` prints the result.

## Open questions

- (none yet) (superseded 2026-11-23 by the entries below)
- **CUDA toolkit too old for `sm_120` (blocks step 6).** The development machine has nvcc 12.4 (driver 580, CUDA 13.0 runtime). Blackwell consumer GPUs (`sm_120`, RTX 5080) need CUDA 12.8 or newer. Options: install CUDA 12.8+/13.x before step 6 (recommended), or build for `compute_90` PTX and rely on JIT (slower start, no FP4/FP8 `sm_120` paths). Nothing in steps 1–5 depends on it.
- **Hardware assumptions in `docs/PLAN.md` §2 can be confirmed:** CPU is an AMD Ryzen 7 9800X3D (Zen 5; `avx512f`, `avx512bw`, `avx512vl` and `avx512_bf16` present); RAM shows 59 GiB usable, consistent with 64 GB installed.
- **Step 2, weight parameterizations.** The checkpoint suggests zero-centered RMSNorm weights for the decoder-layer norms and `q_norm`/`k_norm`, and plain weights for the Gated DeltaNet output norm (`docs/contract/qwen35-0.8b.md`, observations). To be confirmed against the pinned reference implementation, as the **verify** rule requires. **Resolved 2026-11-24:** confirmed against transformers 5.24.0 (`Qwen3_5RMSNorm` uses `1 + w`, `Qwen3_5RMSNormGated` a plain `w`); see the contract, sections RMSNorm and Gated RMSNorm.
- **Golden data and f32 checkpoint tensors (step 3).** `A_log` and `linear_attn.norm.weight` are f32 in the checkpoint, and the reference has no `_keep_in_fp32_modules`. If the golden dump loads the model in bf16, the reference rounds them to bf16 and its outputs differ from athanor's (which uses the stored f32 values). Options: load the reference in fp32 for op-level goldens, or check what transformers 5.24.0 does with mixed-dtype checkpoints and record it. To settle before writing `tests/golden/dump.py`.

## Log

- 2026-11-23: repository documentation created; design in `docs/ARCHITECTURE.md`; plan in `docs/PLAN.md`.
- 2026-11-23: step 1 implemented on branch `step1-core-inspect`: Makefile (`cpu`, `test`, `check-alloc`, `clean`, `DEBUG=1`; `cuda`/`hip` stubs), `src/core/{base,arena,file,json,tensor,safetensors}`, `tools/athanor.c` with `inspect`, `tests/test.h` runner and `tests/test_core.c`. All acceptance criteria met (table above).
- 2026-11-23: decisions taken within step 1 (no `PLAN.md` impact): logging threshold from the `ATH_LOG` environment variable, read on every call (no global state); memory-mapped weights are tier `ATH_TIER_RAM`; the safetensors reader rejects gaps and overlaps in the data section like the reference implementation; sorting uses an in-file heapsort because glibc `qsort` may call `malloc`; `inspect` reads config keys at the top level or under `text_config` (Hugging Face convention for multimodal checkpoints) and leaves model-specific interpretation to step 2.
- 2026-11-23: fixed `.gitignore`: `models/` also matched `src/models/`; now `/models/`.
- 2026-11-24: `step1-core-inspect` merged into `main` (fast-forward); step 2 started on `step2-contract`. Reference implementation pinned (transformers 5.24.0); op contract of Qwen3.5-0.8B written; MoE naming, shapes and router contract recorded from the Qwen3.5-35B-A3B headers (no weights downloaded).
