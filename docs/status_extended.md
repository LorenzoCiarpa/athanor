# Status, extended

The companion of [`STATUS.md`](STATUS.md). `STATUS.md` is the overview: the current step, its task list, its acceptance criteria and results, the open questions and the log. This file goes one level down. For every task of every step it lists the files created or changed, says what each file is for, and describes what was implemented in it and why.

How to read it:

- Steps are grouped by phase, as in `docs/PLAN.md`. Each task (1.1, 1.2, …) has a table of files and a list of what was implemented. **New** marks a file created by the task; **Changed** marks an edit to an existing file.
- The commit that landed the task is named at the end of each task, so `git show <commit>` gives the exact diff.
- Updated with every task, like `STATUS.md` (AGENTS.md, workflow step 4). History is never rewritten: a file that is later replaced or removed is marked superseded here, not deleted from the text.

## Contents

- [Phase 0–1: foundations and inference](#phase-01-foundations-and-inference)
  - [Before step 1: design documents](#before-step-1-design-documents)
  - [Step 1: core and `athanor inspect`](#step-1-core-and-athanor-inspect)
  - [Step 2: contract and family loading](#step-2-contract-and-family-loading)
  - [Step 3: CPU reference ops and golden harness](#step-3-cpu-reference-ops-and-golden-harness)
  - [Step 4: CPU forward](#step-4-cpu-forward)
- [Files by step](#files-by-step)

---

## Phase 0–1: foundations and inference

Steps 1 to 5 run on the CPU only. They build a slow but obviously correct implementation that every GPU kernel is checked against later.

### Before step 1: design documents

Commit `6a3e13c` (2026-11-23), before any code. It fixed the rules and the shape of the project.

| File | Kind | What it is for |
| --- | --- | --- |
| `AGENTS.md` | New | The rules for anyone who changes the code: reading order, the nine invariants (dependencies point down, no model names outside `src/models/`, no external libraries, every op with reference + kernel + test, never guess model math, numerics are protected, placement never changes results, no performance claims without a report, minimal by default), the workflow per task, checklists, code style, definition of done |
| `CLAUDE.md` | New | Points coding agents to `AGENTS.md` and `docs/STATUS.md` |
| `README.md` | New | Public description of the project |
| `LICENSE` | New | MIT |
| `.gitignore` | New | Build output, downloaded models (`/models/`, anchored so that it does not also match `src/models/`), golden data, benchmark raw data, Python caches and environments |
| `docs/ARCHITECTURE.md` | New | The layers (`tools → pipelines → contract → families → ops → backend → core`), the model contract as first designed, common versus family-specific code, the directory tree, what athanor writes itself instead of using libraries |
| `docs/PLAN.md` | New | Decisions, assumptions, the model ladder, the steps of every phase with their gates, performance references, risks |
| `docs/STATUS.md` | New | The current step and its tasks (then step 1), open questions, log |
| `docs/MODELS.md` | New | Why the Qwen hybrid family, target sizes per phase, the op contract with the items to **verify** against the reference |
| `docs/BENCHMARKS.md` | New | How a performance report must be written (AGENTS.md invariant 8) |
| `docs/contract/README.md` | New | What a per-model contract file contains |
| `docs/program/PROGRAM.md` | New | Map from each article of the course to the code that implements it |
| `include/README.md` and the `README.md` of `src/core`, `src/backend`, `src/ops`, `src/text`, `src/model`, `src/models/qwen35`, `src/infer`, `src/quant`, `src/ablit`, `src/train`, `src/net`, `tests`, `tools` | New | One README per module, written before the module: its responsibility, planned files and rules. Each is updated when code lands in the module |

---

### Step 1: core and `athanor inspect`

Done 2026-11-23, merged into `main` 2026-11-24. Goal: read a Qwen3.5-0.8B checkpoint without copying it and describe every tensor; no GPU, no model math. Everything here lives in `src/core/` (the bottom layer, which knows nothing about models), `tools/` and `tests/`.

#### 1.1 Build skeleton

| File | Kind | What it is for |
| --- | --- | --- |
| `Makefile` | New | The whole build: no configure step, no build system beyond make |

What was implemented:

- Targets `cpu` (default) building `build/athanor`, `test`, `clean`; `cuda` and `hip` exist as stubs that fail with a message naming the step that brings them (6 and 11).
- Flags `-std=c11 -O2 -Wall -Wextra -Werror -D_GNU_SOURCE`, so any warning breaks the build; `-MMD -MP` for header dependencies.
- `DEBUG=1`: `-O0 -g`, `ATH_ASSERT` enabled, AddressSanitizer and UndefinedBehaviorSanitizer, output in `build/debug/` so it never mixes with the release objects.
- The `FAMILIES` variable (default `qwen35`) that selects the model families compiled in (rule 1 of `docs/ARCHITECTURE.md`). It became functional in step 2.
- `ATH_MODEL_DIR` (default `models/Qwen3.5-0.8B`), exported to the tests that need real weights.

Commit `4de0f35`.

#### 1.2 `src/core/base.h`

| File | Kind | What it is for |
| --- | --- | --- |
| `src/core/base.h` | New | The definitions every other file includes |
| `src/core/base.c` | New | Status strings and logging |

What was implemented:

- `ath_status`: the return type of every fallible function. `ATH_OK` is 0, so `if (s)` reads as "if failed". Codes: invalid argument, out of memory, I/O, not found, parse error, format error (well-formed but inconsistent), type, range, unsupported. `ath_status_str()` gives a short English description.
- `ATH_CHECK(expr)`: evaluates a call and returns its status from the current function if it fails. With `ATH_LOG=debug` each hop is logged with file and line, which reads like a backtrace.
- `ath_log(level, fmt, ...)` to stderr. The threshold comes from the `ATH_LOG` environment variable (`none`, `error`, `warn` default, `info`, `debug`, `trace`, or `0`–`5`) and is read on every call, so there is no global state.
- `ATH_ASSERT`: checks programmer errors (never input errors) in debug builds only, logging the location and trapping; in release builds the condition is not even evaluated.
- No `exit()` or `abort()` in library code: errors always go back to the caller.

Commit `4de0f35`.

#### 1.3 `src/core/arena.{h,c}`

| File | Kind | What it is for |
| --- | --- | --- |
| `src/core/arena.h` | New | Interface of the linear allocator |
| `src/core/arena.c` | New | Implementation; the only host file that calls `malloc` |

What was implemented:

- An arena hands out memory by bumping a pointer inside a block; nothing is freed individually, everything dies together. This is how athanor manages memory everywhere (AGENTS.md: no `malloc` in hot paths).
- Blocks are chained: when a block is full, a new one (default 1 MiB, or the request size if larger) is linked after it.
- `ath_arena_mark` / `ath_arena_reset`: save a position and roll back to it later, keeping the blocks for reuse (scratch memory per step or per request).
- `ath_arena_zalloc` (zeroed), `ath_arena_strndup`, `ath_arena_used`, `ath_arena_reserved`; alignment on every allocation.

Commit `4de0f35`.

#### 1.4 `src/core/file.{h,c}`

| File | Kind | What it is for |
| --- | --- | --- |
| `src/core/file.h` | New | Read-only file access |
| `src/core/file.c` | New | POSIX implementation |

What was implemented:

- `ath_file_open`: opens a regular file and maps it read-only with `mmap`. A 1.7 GB checkpoint costs nothing until its bytes are touched, and the page cache is shared with other processes. An empty file gives a NULL mapping.
- `ath_file_pread`: reads a byte range into a caller buffer, retrying short reads and `EINTR` (the base of the disk tier in step 12).
- `errno` mapped to `ath_status` (not found, invalid argument for non-regular files, I/O, out of memory).
- `ath_file_exists`, `ath_path_join` (with a range error if the path does not fit).

Commit `4de0f35`; `file.h` touched again in `5f851df`.

#### 1.5 `src/core/json.{h,c}`

| File | Kind | What it is for |
| --- | --- | --- |
| `src/core/json.h` | New | Read-only JSON tree and lookups |
| `src/core/json.c` | New | Recursive-descent parser |

What was implemented:

- A strict RFC 8259 parser for `config.json`, tokenizer files and safetensors headers. The input need not be NUL-terminated, so it parses straight out of an mmap.
- Each grammar production is one function; recursion depth is capped at 256 so hostile input cannot exhaust the stack.
- Strings: every escape including `\uXXXX` and surrogate pairs (a lone surrogate is rejected), UTF-8 validated, no raw control characters.
- Numbers follow the JSON grammar exactly (no leading zeros, digits required after `.` and in exponents) and are kept as int64 when integral, double otherwise.
- Nodes and decoded strings live in an arena; children are a linked list in document order; duplicate keys are kept and lookups return the first.
- Navigation: `ath_json_get` (key), `ath_json_at` (index), `ath_json_path` (dotted path such as `text_config.layer_types.3`), `ath_json_first`, `ath_json_count`; typed access `to_int`, `to_double`, `to_bool`, `to_str`.
- Errors report byte offset, line and column.

Commit `a699d34`.

#### 1.6 `src/core/tensor.h`

| File | Kind | What it is for |
| --- | --- | --- |
| `src/core/tensor.h` | New | Element types, memory tiers, the tensor descriptor |
| `src/core/tensor.c` | New | Dtype tables, init, views |

What was implemented:

- `ath_dtype`: f32, f16, bf16, i32, i64, u8, f8 e4m3, f8 e5m2; values 64–127 reserved for athanor's quantized block formats (phase 2), which have no per-element size.
- `ath_tier`: RAM, PINNED, VRAM, DISK. Kernels never branch on it; memory-mapped checkpoint data is RAM.
- `ath_tensor`: a descriptor that owns nothing (dtype, tier, up to 8 dimensions, shape, strides in elements, data pointer). Views are new descriptors over the same bytes.
- `ath_tensor_init` checks the dtype, the number of dimensions, negative sizes and byte-size overflow; `numel`, `nbytes`, `is_contiguous`; `ath_tensor_slice` (a range along one dimension) and `ath_tensor_reshape` (contiguous only, same element count).

Commit `7d264f6`. Extended in step 3 (task 3.2).

#### 1.7 `src/core/safetensors.{h,c}`

| File | Kind | What it is for |
| --- | --- | --- |
| `src/core/safetensors.h` | New | Zero-copy checkpoint reader |
| `src/core/safetensors.c` | New | Single-file and sharded readers, validation |

What was implemented:

- The format: 8-byte little-endian header length, a JSON header (`__metadata__` plus one entry per tensor with dtype, shape and data offsets), then the data section. Each file is mmapped and every tensor becomes an `ath_tensor` pointing into the mapping: opening copies no weights.
- Sharded checkpoints through `model.safetensors.index.json`; `ath_safetensors_open(dir)` handles both layouts, `ath_safetensors_open_file(path)` opens one file (used again in step 3 for the golden data).
- Validation as strict as the reference implementation: known dtypes, at most 8 dimensions, shape and dtype consistent with the byte range, ranges inside the data section, tensors covering each data section exactly (no gaps, no overlaps), no duplicate names. For sharded checkpoints the index and the shard headers must agree tensor by tensor; a mismatching `metadata.total_size` and misaligned tensors give warnings.
- Tensors sorted by name with digit runs compared as numbers (`layers.2` before `layers.10`); `ath_safetensors_find` is a binary search. Sorting uses a small heapsort in the file because glibc's `qsort` may call `malloc`.

Commit `5f851df`.

#### 1.8 `tools/athanor.c` and `athanor inspect`

| File | Kind | What it is for |
| --- | --- | --- |
| `tools/athanor.c` | New | The binary: dispatch table and process setup only |
| `tools/commands.h` | New | Signatures of the subcommands (each gets its arguments and an arena, returns 0, 1 or 2) |
| `tools/inspect.c` | New | `athanor inspect <model_dir> [--json] [--filter <substring>]` |
| `tools/dev/alloc_trace.c` | New | Development only: an `LD_PRELOAD` shim that reports every `malloc`, `calloc`, `realloc` with its caller |
| `tools/dev/check_alloc.sh` | New | Runs a command under the shim and fails if any allocation comes from somewhere other than `ath_arena_alloc` |
| `Makefile` | Changed | Links the binary; `make check-alloc` |

What was implemented:

- `athanor.c` gives stdout a buffer taken from the arena (libc would otherwise `malloc` one on first use) and closes it before freeing the arena.
- `inspect` prints a summary of `config.json` (model type, layers, hidden size, vocabulary; keys looked up at the top level and then under `text_config`, the Hugging Face convention for multimodal checkpoints), the shard files with header and data sizes, one row per tensor (name, dtype, shape, bytes), totals per dtype, and the check that tensor bytes equal the data sections and the index `total_size`. `--json` gives the same as JSON, `--filter` keeps the tensors whose name contains a substring. Only headers are read.
- `check_alloc.sh` resolves each caller with `addr2line`; `make check-alloc` proves acceptance criterion 4 (no heap allocation outside arenas: one allocation, the arena's first block).

Commit `e1a3e88`.

#### 1.9 `tests/test_core.c` and the test runner

| File | Kind | What it is for |
| --- | --- | --- |
| `tests/test.h` | New | The minimal header-only test runner |
| `tests/test_core.c` | New | Tests of every core module |

What was implemented:

- The runner: `ATH_TEST(name)`, `ATH_EXPECT` (record and continue), `ATH_REQUIRE` (record and end the test), status, integer, string and float comparisons, `ATH_SKIP`, a summary and a non-zero exit code on failure. `./build/test_core json` runs only the tests whose name contains `json`. Library logging is silenced unless `ATH_LOG` is set, since many tests feed broken input on purpose.
- 22 tests: status strings, `ATH_CHECK` propagation, log levels; arena alignment, growth, mark/reset, zeroed allocation and overflow; mmap and pread, path join; JSON scalars and containers, escapes, numbers, invalid inputs, error positions, real `config.json`, `tokenizer_config.json`, `tokenizer.json`; dtype table, tensor init and views; safetensors name order, single file, 20 malformed headers, sharded checkpoints with inconsistent indexes, the real checkpoint.
- Synthetic safetensors files are written by the tests into a fresh temporary directory and removed afterwards.

Commits `4de0f35` to `5f851df` (the tests grew with each module).

#### Closing step 1

| File | Kind | What it is for |
| --- | --- | --- |
| `docs/STATUS.md` | Changed | Tasks ticked, results table, log, open questions (CUDA 12.4 too old for `sm_120`, hardware confirmed, the RMSNorm parameterizations suggested by the weights, to verify in step 2) |
| `docs/contract/qwen35-0.8b.md` | New | The checkpoint as `inspect` reports it: revision, file hashes, tensor groups by layer type, shape arithmetic against `config.json`, weight statistics. It became the op contract in step 2 |
| `README.md`, `docs/ARCHITECTURE.md`, `docs/program/PROGRAM.md`, `src/core/README.md`, `tests/README.md`, `tools/README.md` | Changed | Brought up to date with the code |

Commit `c802bbc`.

---

### Step 2: contract and family loading

Done 2026-11-24, merged into `main` 2026-11-24. Goal: give every tensor of the checkpoint a meaning that pipelines can use without knowing the family (a role, a layer, an expert index, flags) plus the layer schedule. No model math yet.

#### 2.1 Reference implementation and op contract

| File | Kind | What it is for |
| --- | --- | --- |
| `docs/contract/qwen35-0.8b.md` | Changed | Became the op contract of Qwen3.5-0.8B |
| `docs/contract/qwen35-35b-a3b.md` | New | The MoE sibling: what differs from the dense model |
| `docs/MODELS.md` | Changed | Status line: which **verify** items are resolved and where |
| `docs/STATUS.md` | Changed | Step 2 opened |

What was implemented:

- The reference implementation pinned: transformers 5.24.0, with the sha256 of the wheel and of `modeling_qwen3_5.py`, `configuration_qwen3_5.py`, `conversion_mapping.py`. Read from the wheel, never imported by athanor.
- The op contract, confirmed line by line against the reference: the decoder layer and residual stream; zero-centered RMSNorm `x̂·(1+w)` in fp32 for the layer norms, the final norm and `q_norm`/`k_norm`; the gated RMSNorm of the Gated DeltaNet with a plain weight (normalize, round to bf16, multiply by `w`, multiply by `silu(z)`); full attention (query and output gate interleaved per head in `q_proj`, QK-norm, partial rotate-half RoPE on 64 of 256 dimensions with theta 1e7, M-RoPE reducing to plain RoPE for text, grouped-query attention, `attn·sigmoid(gate)` before `o_proj`); the Gated DeltaNet (projections and channel layout, causal conv + SiLU, L2 norm with a fixed eps of 1e-6, query scale `dk^-0.5`, gates indexed by value head, the recurrence, the state size); the MLP; checkpoint dtypes; the MTP head, which transformers does not implement.
- The parameter binding table: role and flags of every tensor.
- For Qwen3.5-35B-A3B, from the real config, index and shard headers (no weights downloaded): fused expert tensors `gate_up_proj` [E, 2I, H] and `down_proj` [E, H, I], softmax router in fp32 then top-k and renormalization, shared expert with a scalar sigmoid gate, untied `lm_head`.

Commit `654d4be`.

#### 2.2 `src/model/model.h`

| File | Kind | What it is for |
| --- | --- | --- |
| `src/model/model.h` | New | The contract between pipelines and model families: the only way pipelines see a model |
| `docs/ARCHITECTURE.md` | Changed | The contract section shows the real header, the members still to come, the coverage rule |
| `src/model/README.md` | Changed | How to load a model, the coverage rule, what roles and flags are for |

What was implemented:

- `ath_role`: embed, lm_head, norm, mix_in, mix_out (token mixer, attention or linear attention), mlp_in, mlp_out, router, expert_in, expert_out, conv, gate, other; `ath_role_name()`.
- Param flags: `ATH_P_WRITES_RESIDUAL` (abliteration projects the refusal direction out of these), `ATH_P_SENSITIVE` (quantization keeps these at higher precision by default), `ATH_P_SHARED_EXPERT`.
- `ath_param`: a contiguous view of checkpoint memory plus role, layer (−1 for global), expert (−1 if none), flags and the original checkpoint name.
- `ath_layer` (mixer and FFN labels for people, number of experts), `ath_ignored` (a prefix of tensors left unbound on purpose, with a reason and counts), `ath_model` (family, config, checkpoint, params in forward order, layers, ignored groups, main dimensions, tied embeddings, expert counts, family-private data), `ath_family` (`name`, `match`, `load`).
- State specs, hooks, `forward`, `pretokenize`, `chat` and `backward` were deliberately left out until their first user (AGENTS.md rule 9); `docs/ARCHITECTURE.md` keeps their design.

Commit `8e66f78`.

#### 2.3 `src/model/registry.c`

| File | Kind | What it is for |
| --- | --- | --- |
| `src/model/registry.c` | New | The compiled-in families, `ath_model_open`, the coverage check |
| `Makefile` | Changed | Generates `build/gen/families.h` from `FAMILIES`; fails on an unknown family |
| `AGENTS.md` | Changed | "Add a model family" checklist: add it to the Makefile list, no registry edit |

What was implemented:

- The family table comes from the Makefile: `FAMILIES=qwen35` writes the line `ATH_FAMILY(qwen35)` into a generated header, rewritten only when the list changes. `registry.c` therefore names no family (invariant 2).
- `ath_model_open(dir)`: reads `config.json`, asks each family whether it `match`es, opens the checkpoint, calls the family's `load`, then runs the coverage check. On any failure the checkpoint is closed and the model zeroed.
- The coverage rule, enforced by the registry rather than trusted to the family: every checkpoint tensor is viewed by at least one param or falls in exactly one ignored group, never both, and every param is a contiguous view inside the tensor it names. A tensor nobody accounts for fails the load with its name.

Commit `8e66f78`; a comment fixed in `232897f`.

#### 2.4 `src/models/qwen35/config.c`

| File | Kind | What it is for |
| --- | --- | --- |
| `src/models/qwen35/qwen35.h` | New | Private header of the family: `ath_qwen35_config` and the family's functions |
| `src/models/qwen35/config.c` | New | `config.json` → `ath_qwen35_config` |

What was implemented:

- `match` accepts the model types `qwen3_5`, `qwen3_5_moe` (vision-language: keys under `text_config`, tensors under `model.language_model.`) and `qwen3_5_text`, `qwen3_5_moe_text` (text only: keys at the top level, tensors under `model.`).
- Every dimension the op contract needs: layers, hidden size, vocabulary, RMSNorm eps, attention heads, KV heads, head dimension, RoPE dimension and theta, Gated DeltaNet key/value heads and dimensions, conv kernel, MLP width, MoE experts, active experts, expert and shared-expert widths, tied embeddings.
- The layer schedule from `layer_types`, or from `full_attention_interval` (default 4) when absent, as the reference does.
- Values the reference supports but athanor cannot honor are rejected instead of ignored: activations other than SiLU, attention bias, a missing output gate, non-default RoPE, `mlp_only_layers`, inconsistent head counts, a RoPE dimension that is odd or larger than the head.

Commit `8e66f78`.

#### 2.5 `src/models/qwen35/weights.c`

| File | Kind | What it is for |
| --- | --- | --- |
| `src/models/qwen35/weights.c` | New | Checkpoint tensors → `ath_param` with role, layer, expert, flags |
| `src/models/qwen35/family.c` | New | The family's registry entry: `load` parses the config, fills the model's dimensions and schedule, binds the tensors |
| `src/models/qwen35/README.md` | Changed | File table and checkpoint layouts |

What was implemented:

- Binding is driven by the config: for every layer the family knows which tensors must exist and their shapes, finds them by name, and fails on a missing tensor (not found), a non-float dtype or a shape that does not follow from `config.json` (format error, both shapes in the message).
- Params in forward order: embedding, then per layer input norm, token mixer, post-attention norm, MLP or MoE, then final norm and output head. Roles and flags as in the contract's binding table (for example `out_proj`, `o_proj` and `down_proj` write the residual; the router and the small gate matmuls are sensitive).
- Fused MoE expert tensors become one view per expert (slice and reshape, no copy). Tied embeddings give no `lm_head` param and make the embedding sensitive; a tied config with an `lm_head` tensor is an error.
- Two ignored groups: `model.visual.` (vision encoder, text path only) and `mtp.` (multi-token prediction head, step 13), the same prefixes the reference text model drops.

Commit `8e66f78`.

#### 2.6 `athanor inspect --roles`

| File | Kind | What it is for |
| --- | --- | --- |
| `tools/inspect.c` | Changed | The `--roles` view |
| `tools/README.md` | Changed | Its documentation |
| `tools/dev/check_alloc.sh` | Changed | Now fails if the traced command itself fails |
| `Makefile` | Changed | `make test` also runs `inspect --roles`; `check-alloc` covers it |

What was implemented:

- `--roles` loads the checkpoint through the contract and prints what pipelines see: family and dimensions, the layer schedule, per-layer totals, one row per param (role, layer, expert, flags, dtype, shape, bytes), totals per role, the ignored groups, the coverage check. `--json` and `--filter` work with it. Nothing in it knows a family.
- A bug found on the way: `check_alloc.sh` ignored the exit status of the traced command, so a command that stopped early (an unknown option) passed for the wrong reason.

Commit `d79df7a`.

#### 2.7 `tests/test_model.c`

| File | Kind | What it is for |
| --- | --- | --- |
| `tests/test_model.c` | New | Tests of the contract, the registry and the qwen35 family |
| `tests/util.h` | New | Helpers shared by the suites (temporary directory, location of the real checkpoint), moved out of `test_core.c` |
| `tests/test_core.c` | Changed | Uses `util.h` |
| `tests/README.md` | Changed | The new suite |

What was implemented:

- 11 tests: role names; the registry lists the families; config parsing of a dense model, the schedule fallback and 22 rejected configs; binding of tiny synthetic checkpoints (dense, MoE with fused experts, text-only) and every failure mode (missing, unexpected, misshaped, integer dtype, tie mismatches); `ath_model_open` errors; the real checkpoint (coverage, schedule against `layer_types`, residual writers).
- Synthetic checkpoints use tiny dimensions and zero data, so the tests cost nothing.

Commit `8e66f78`.

#### Closing step 2

| File | Kind | What it is for |
| --- | --- | --- |
| `docs/STATUS.md` | Changed | Results table (320 tensors bound, 168 ignored, 0 unbound; schedule 18 + 6), extra MoE check on a sparse 35B-A3B copy, decisions, open question on f32 tensors |
| `docs/program/PROGRAM.md` | Changed | Row 2: roles landed |

Commit `232897f`.

---

### Step 3: CPU reference ops and golden harness

Done 2026-11-24 on branch `step3-ops-golden`. Goal: every op of the Qwen3.5-0.8B forward pass as a scalar CPU reference, and evidence on the real weights that each one computes what the pinned reference implementation computes.

#### 3.1 Golden environment and the f32 question

| File | Kind | What it is for |
| --- | --- | --- |
| `tests/golden/requirements.in` | New | The two pinned packages: transformers 5.24.0 and torch 2.15.0 (CPU build), plus the PyTorch CPU index |
| `tests/golden/requirements.txt` | New | The full lock compiled with uv (every transitive dependency at an exact version) |
| `tests/golden/.venv/` | not committed | The environment itself, Python 3.14 |

What was implemented:

- A reproducible Python environment for the golden data only. Python stays out of the build and of the runtime (invariant 3).
- The open question of step 2 settled by experiment: transformers 5.24.0 rounds the two f32 tensors of the checkpoint (`linear_attn.A_log`, `linear_attn.norm.weight`) to bf16 on load, with `dtype="auto"` as well as bf16, because the model has no `_keep_in_fp32_modules`. Decision: the golden data puts the stored f32 values back after loading, so both sides compute with the same weights. Measured on the golden prompt against an fp32 run of the whole model: max KL 1.4e-3 with the stored values, 2.9e-3 with the default rounding. Recorded in the contract (Checkpoint dtypes), `tests/golden/README.md` and `STATUS.md`.

Commits `3cbc234` (files), `c508b76` (STATUS).

#### 3.2 `src/core/tensor.h`: bf16 and element access

| File | Kind | What it is for |
| --- | --- | --- |
| `src/core/tensor.h` | Changed | bf16 conversion and element access, as inline functions |
| `src/core/tensor.c` | Changed | `ath_tensor_same_shape` |
| `src/core/README.md` | Changed | The new content of `tensor.h` |

What was implemented:

- `ath_bf16_to_f32`: exact (bf16 is the upper half of an f32).
- `ath_f32_to_bf16`: round to nearest, ties to even, every NaN becomes `0x7fc0`, exactly like torch, so a value rounded by athanor and by the reference is bit-identical when the f32 values are. The test checks ties, values just above and below a tie, signs, infinities, overflow to infinity and subnormals.
- `ath_load_f32` / `ath_store_f32`: read or write element `i` of an f32 or bf16 buffer as a float; `ath_round_to(dtype, v)`: the value `v` takes once stored in `dtype`. The reference ops read and write all activations and weights through these.
- `ath_tensor_same_shape`.

Commit `5ffe0a7`.

#### 3.3 `src/ops/`: the reference ops

| File | Kind | What it is for |
| --- | --- | --- |
| `src/ops/ops.h` | New | Public interface: every signature with shapes, dtypes and math; the conventions shared by all ops |
| `src/ops/check.h` | New | Private to `src/ops/`: argument checks (present, f32 or bf16, host memory, contiguous) with a logged message, `ld`/`st` element access, row helpers |
| `src/ops/embed.c` | New | `ath_embed_ref`: rows of the embedding table for a list of token ids; ids out of range give a range error |
| `src/ops/matmul.c` | New | `ath_matmul_ref`: `out = x · wᵀ` with the weight in checkpoint layout `[M, K]`; one fp32 dot product per output in order of `k` (for bf16 inputs every product is exact in fp32) |
| `src/ops/norm.c` | New | `ath_rmsnorm_ref` (`w_offset` 0 for a plain weight, 1 for the zero-centered `1 + w`), `ath_rmsnorm_gated_ref` (normalize, round to the input dtype as the reference does, then `w · n · silu(z)`), `ath_l2norm_ref` (`x / sqrt(Σx² + eps)`) |
| `src/ops/rope.c` | New | `ath_rope_ref`: rotate-half on the first `rope_dim` dimensions of each head, dimension `i` paired with `i + rope_dim/2`; frequencies, angles, cos and sin computed in fp32 the way the reference does, then rounded to the activation dtype like its tables |
| `src/ops/act.c` | New | `ath_swiglu_ref` (`silu(gate) · up`), `ath_sigmoid_mul_ref` (`x · sigmoid(gate)`, the attention output gate), `ath_add_ref` (residual addition, may run in place) |
| `src/ops/attn.c` | New | `ath_attn_ref`: causal softmax attention with grouped-query heads; the T queries are the last T of S positions, so the same function serves prefill and decode with a KV cache; scores and probabilities in fp32, no scratch memory (two passes over the keys) |
| `src/ops/conv.c` | New | `ath_conv_ref`: causal depthwise conv1d over time with a `[K − 1, C]` running state read and updated in place, optional SiLU; feeding a sequence in pieces gives the same bits as feeding it at once |
| `src/ops/gdn.c` | New | `ath_gdn_gates_ref` (decay `g = −exp(A_log)·softplus(a + dt_bias)` with torch's softplus threshold, write strength `beta = sigmoid(b)`); `ath_gdn_ref` (gated delta rule, recurrent form: decay the state, read `u = Sᵀk`, write `beta(v − u)` along `k`, read the output with the scaled query; value heads grouped over query/key heads; state `[nv, dk, dv]` in f32 updated in place, column by column, no scratch memory) |
| `src/ops/README.md` | Changed | File table and conventions |
| `Makefile` | Changed | `src/ops/*.c` compiled into the library |

What was implemented, across all ops:

- Scalar loops, fp32 arithmetic and accumulation, one rounding to the output dtype at the end: obviously correct and never optimized (AGENTS.md, "Add an op"). The dtype of each output tensor is the dtype the result is rounded to.
- An intermediate rounding happens only where the reference does it and the contract records it: the normalized value in `rmsnorm_gated` and the cos/sin tables in `rope`. Everywhere else athanor is at least as precise as the reference.
- No op allocates memory and none knows a model: the zero-centered norm is a parameter, the Gated DeltaNet query scale and L2-norm epsilon are arguments, the L2 norm is a separate op that the family composes before the recurrence.
- Errors: invalid argument for missing tensors and inconsistent shapes, type error for unsupported dtypes, unsupported for tensors outside host memory, each with a message naming the op and the argument.
- Not written, on purpose: the device-dispatching entry point per op (it joins `ops.h` with the second implementation of an op, the GPU kernels of step 7), and `src/backend/cpu.c` (task 3.7).

Commit `366a857`.

#### 3.4 `tests/golden/dump.py`: the golden data

| File | Kind | What it is for |
| --- | --- | --- |
| `tests/golden/dump.py` | New | Runs the reference and writes the golden data |
| `tests/golden/README.md` | New | Environment, how to run it, what is recorded, the two deliberate choices |
| `Makefile` | Changed | `make golden`; `ATH_GOLDEN_DIR` (default `tests/golden/data/qwen35-0.8b`) and `GOLDEN_PYTHON` |
| `tests/golden/data/qwen35-0.8b/` | not committed | `manifest.json` and `golden.safetensors` (434 MiB, regenerated in about 4 s) |

What was implemented:

- Safety checks before anything runs: transformers 5.24.0, torch 2.15.0+cpu, the sha256 of `modeling_qwen3_5.py` from the contract, and none of `fla`, `causal_conv1d`, `kernels` installed (any of them would replace the torch functions with fused kernels).
- The reference runs on the CPU in bf16 with eager attention and one thread (reproducible sums) over a fixed 75-token prompt, longer than one chunk (64 tokens) of the reference's chunked Gated DeltaNet, then one decode step with the cache.
- Capture without changing the reference: forward hooks on modules (embedding, every RMSNorm, every linear layer, the gated norm, each decoder layer) and wrappers around the module-level functions the forward code calls (`apply_rotary_pos_emb`, `eager_attention_forward`, `causal_conv1d_fn`, `causal_conv1d_update`, `l2norm`, the chunked and recurrent Gated DeltaNet functions). Each wrapper calls the original.
- One record per op call, 906 in total, for all 24 layers in prefill and decode: the inputs, `out` (the reference's own result in its dtype) and `out32` (the same function rerun in fp32 from the same inputs, with fp32 RoPE tables; for the Gated DeltaNet the recurrent form, athanor's algorithm). Matmuls keep rows 0 and T − 1 only: an output row depends only on its input row.
- Checks that turn assumptions into facts at every regeneration: the attention mask is causal; the bf16 RoPE tables are the rounding of the fp32 ones at the token positions (M-RoPE is the identity for text); the oldest of the K conv state columns the reference keeps never reaches an output (so athanor's state of K − 1 is enough); with fp32 attention scores the eager output falls within 2 bf16 ulps of the fp32 rerun (so the score rounding explains the whole native difference of `attn`).
- `manifest.json`: versions, CPU, the sha256 of `config.json` and of the data file, the f32 tensors restored, the prompt and its tokens, the decode token, the records with the checkpoint names of their weights and their parameters (eps, RoPE theta and dimension, attention scale, query scale, L2-norm epsilon).

Commit `3cbc234`.

#### 3.5 `tests/test_ops.c`: hand-computed cases

| File | Kind | What it is for |
| --- | --- | --- |
| `tests/test_ops.c` | New | Tests of every `_ref` function without the reference implementation |
| `Makefile` | Changed | `test_ops` added to the suites |

What was implemented (16 tests):

- bf16 rounding (ties to even, NaN, infinities, overflow, subnormals); argument checks shared by the ops (missing tensor, wrong shape, integer dtype, tensor in VRAM, non-contiguous view).
- Hand-computed values: embedding rows and out-of-range ids; a small integer matmul and its bf16 rounding; RMSNorm on (3, 4) for both parameterizations and for eps; the gated norm, including the bf16 rounding of the normalized value; L2 norm; RoPE at position 0 (identity), the rotate-half pairing, pass-through dimensions, and scores that depend only on the distance between positions; SwiGLU, sigmoid gate and a bf16 add with a tie; attention averaging the visible values (causality), grouped heads reading the right KV head, a dominant score selecting its value; conv with identity and delay kernels and SiLU; gates around the softplus threshold; one delta-rule step, a second write along the same key, decay and `beta = 0`.
- Exact properties, bit for bit: the last query of a prefill equals the same query alone against the cache; changing a later key leaves earlier outputs unchanged; conv fed in pieces of 2, 1 and 3 tokens equals the whole sequence, state included; the Gated DeltaNet with grouped heads equals the same run with the heads repeated explicitly, and token by token equals all at once.

Commit `366a857`.

#### 3.6 `tests/test_golden.c`: replay on the real weights

| File | Kind | What it is for |
| --- | --- | --- |
| `tests/test_golden.c` | New | Replays every golden record through the `_ref` functions |
| `tests/README.md` | Changed | Golden data, the error measure, the tolerance table with measured values and reasons |
| `Makefile` | Changed | `test_golden` added to the suites |

What was implemented:

- Reads `manifest.json` with athanor's JSON parser and `golden.safetensors` with athanor's own safetensors reader, and loads the weights through `ath_model_open`, by the checkpoint names the records carry: the real weights reach the ops through the step 2 contract.
- One handler per op. Each record runs twice: **native** (inputs as recorded, output in the recorded dtype, compared with `out`) and **exact** (inputs widened to f32, output in f32, compared with `out32`). The Gated DeltaNet is replayed as athanor composes it: two `l2norm` calls, then `gdn`.
- One error measure, `max |got − want| / (|want| + rms(want))`: a relative error for elements of typical size, an error relative to the tensor's RMS near zero.
- A tolerance per op and output, justified rather than tuned: native 1 bf16 ulp where the reference rounds once, 2 where it rounds two or three times, 0 for copies and the residual add (bit-exact); exact 1e-5, raised to 2e-5 to 1e-4 only where a long reduction or the recurrent state brings the measured error within a factor of 4. The one loose tolerance, native `attn` at 0.2, is explained by the eager path rounding `q·kᵀ` to bf16, which `dump.py` verifies.
- A summary table per op, output and phase (records, worst score, worst ulp distance, tolerance); `ATH_GOLDEN_WORST=1` names the worst record of each row. The suite fails on any score above tolerance, on an op that was never compared, or on a record it cannot replay. Skipped when the golden data or the checkpoint is missing.

Commit `3cbc234`.

#### 3.7 Documentation and closing step 3

| File | Kind | What it is for |
| --- | --- | --- |
| `docs/contract/qwen35-0.8b.md` | Changed | The reference is now run by the golden harness; the eager attention rounds the scores to bf16 while athanor keeps them in fp32 (measured difference); the f32 tensors resolved with numbers |
| `docs/MODELS.md` | Changed | Status: every dense op checked against the reference |
| `AGENTS.md` | Changed | "Add an op" checklist: a golden record in `dump.py`, a handler and a justified tolerance in `test_golden.c` |
| `src/backend/README.md` | Changed | Why `cpu.c` is not written yet: the reference ops need no runtime; it arrives with its first user (the GPU runtime of step 6 or the thread pool of the first fast CPU path) |
| `docs/ARCHITECTURE.md` | Changed | Directory tree: the op files, `check.h`, the new tests and `tests/golden/` |
| `docs/program/PROGRAM.md` | Changed | Rows 3 and 4: the CPU reference ops and the recurrent Gated DeltaNet landed |
| `docs/STATUS.md` | Changed | Step 3 tasks, criteria and results; step 2 moved to completed steps; the f32 question resolved; new open question on the end-to-end metric of step 4; decisions in the log |

Commits `3cbc234`, `c508b76`. This file, `docs/status_extended.md`, was added after step 3 was closed.

### Step 4: CPU forward

Done 2026-11-24 on branch `step4-cpu-forward`. Goal: the whole Qwen3.5-0.8B forward pass on the CPU, composed from the step 3 ops, with the per-sequence state that prefill and decode need, reached by pipelines only through the contract; and evidence on fixed prompts that its logits are as close to the exact model as the reference implementation's.

#### 4.1 End-to-end metric

| File | Kind | What it is for |
| --- | --- | --- |
| `docs/STATUS.md` | Changed | Step 4 plan (tasks, acceptance criteria); step 3 moved to completed steps; the open question on the end-to-end metric resolved |

What was decided (before any measurement):

- The gate is option (b) of the open question: the mean KL against an fp32 run of the reference over every position of the fixed prompts, prefill and decode, no larger than the bf16 reference's mean KL against the same run. The reference computes logits in bf16 and rounds attention scores to bf16, athanor does neither, so the logits cannot match bit for bit; the question is whether athanor is at least as close to the exact model.
- Greedy agreement where it is meaningful: athanor's greedy token must equal the fp32 reference's wherever the fp32 top-2 margin is larger than the bf16 noise, the largest change of that margin between the fp32 and the bf16 reference.
- Option (a), the residual stream entering every layer, is a diagnostic, not gated.
- In addition, athanor runs with f32 activations against the fp32 reference within a tolerance justified like the op tolerances: a tight check of the composition that the bf16 noise would hide.

Commit `08ca309`.

#### 4.2 The contract: state, batches, hooks, forward

| File | Kind | What it is for |
| --- | --- | --- |
| `src/model/model.h` | Changed | `ath_state_kind` (KV, conv, recurrent) and `ath_state_kind_name`; `ath_state_spec` (kind, layer, `fixed` and `per_token` element counts); `states` / `n_states` in `ath_model`; `ath_state` (activation dtype, capacity, `n_past`, one buffer per spec); `ath_batch` (tokens, optional f32 logits for every token or the last); `ath_hooks` with `on_residual`; `forward` in `ath_family`; the functions of `state.c` |
| `src/model/state.c` | New | The state of one sequence and the only way into a family's forward pass |

What was implemented:

- **Specs in elements.** The element type follows from the kind and the activation dtype the state is created with: KV and conv state hold activations (bf16 or f32), the recurrent state is always fp32. The first design counted bytes; with two activation dtypes a byte count would have to be declared twice.
- `ath_state_bytes`: the bytes of one sequence for a capacity, every buffer rounded up to 64 bytes, overflow checked.
- `ath_state_init`: one 64-byte aligned host buffer per spec from an arena, then a reset. Accepts bf16 or f32 activations and a capacity of at least one token.
- `ath_state_reset`: `n_past = 0`, conv and recurrent state zeroed; the KV cache is left as it is, since positions at or past `n_past` are never read.
- `ath_model_forward`: checks everything a family would otherwise check (model, batch, state and scratch present; the family has a forward pass; the state initialized; at least one token; room in the state; every token id inside the vocabulary; logits f32, host memory, contiguous, `[n_tokens, vocab]` or `[1, vocab]`), marks the scratch arena, calls the family, releases the scratch memory, and advances `n_past` on success. A batch that does not fit leaves the state untouched; a failure inside the family leaves it undefined until a reset.
- `on_residual(u, layer, x)`: the residual stream entering layer `layer`, and with `layer == n_layers` the stream entering the final norm. Its first users are the forward tests; abliteration will capture it in phase 3. The matmul hook of the design (`on_matmul_in`) waits for its first user in phase 2.
- `forward` takes the model as `const` and a scratch arena, two departures from the first design recorded in `docs/ARCHITECTURE.md`.

Commit `b8a0a56`.

#### 4.3 `src/models/qwen35/forward.c`: the forward pass

| File | Kind | What it is for |
| --- | --- | --- |
| `src/models/qwen35/forward.c` | New | The forward pass composed from the CPU reference ops |
| `src/models/qwen35/qwen35.h` | Changed | `ath_qwen35_layer` (the params of a layer by what the forward pass does with them, and its state indices); `ath_qwen35_model` (the family data: config, layer table, embedding, final norm, output head); `ath_qwen35_forward` |
| `src/models/qwen35/weights.c` | Changed | `bind` records every bound param in the layer table (`add` returns the param, whose address is stable because `m->params` is allocated once with its capacity); the output head is the embedding when tied |
| `src/models/qwen35/family.c` | Changed | `load` allocates the family data and the layer table, then declares the state specs (a KV spec per attention layer, a conv and a recurrent spec per Gated DeltaNet layer, in layer order); `.forward` |

What was implemented:

- **One path for every call.** T tokens at positions `n_past …`: a whole prompt, a piece of one and a single decode token run the same code. Per layer, as in the contract: input RMSNorm (zero-centered), the mixer, the residual addition, post-attention RMSNorm, the SwiGLU MLP, the residual addition. Then the final norm and the logits in fp32, for every token or only the last.
- **Attention.** `q_proj` gives query and output gate interleaved per head; both are copied out into contiguous tensors (the ops take contiguous tensors only; the copies are exact). Values go from `v_proj` straight into the KV cache; keys after QK-norm and RoPE. `ath_attn_ref` runs over every position so far; output gate, `o_proj`.
- **Gated DeltaNet.** The four projections; the causal convolution with SiLU over all q, k, v channels, continuing from the conv state; q, k, v copied out of the convolution output; L2 norm of q and k into fp32 (the reference normalizes after its fp32 cast) with the fixed epsilon 1e-6; the gates (`beta` rounded to the activation dtype as the reference computes it on its bf16 projection, `g` in fp32); the delta rule on the fp32 recurrent state with the query scale `1/sqrt(dk)`; the gated RMSNorm per value head with `z`; `out_proj`.
- **Activation dtype.** Every activation is stored in the state's dtype. bf16 rounds each op's output once; f32 keeps everything in fp32, an exact forward pass. What the reference keeps in fp32 in any case stays fp32.
- **Memory.** Working tensors come from the scratch arena and are released after each layer: one layer's worth for the tokens of the call (10 MiB for the 75-token prompt). No heap allocation.
- MoE layers fail with `ATH_ERR_UNSUPPORTED` until step 12.
- First run on the real checkpoint: the 75-token golden prompt gives the reference's greedy token, at 0.29 s per token (the scalar `matmul_ref`, about 3.6 GFLOP/s, is almost all of it).

Commit `1344b30`.

#### 4.4 `tests/golden/dump.py`: forward records

| File | Kind | What it is for |
| --- | --- | --- |
| `tests/golden/dump.py` | Changed | Forward records after the op records |
| `tests/golden/README.md` | Changed | Forward records section, new size and time |

What was implemented:

- Two prompts: the op prompt (75 tokens) and a chat-formatted one (28 tokens: the checkpoint's template with the generation prompt, written out as text so that the data does not depend on the template). Each is followed by 8 greedy decode steps of the bf16 reference with its cache.
- `run_forward` captures, with pre-hooks on every decoder layer and on the final norm, the residual stream entering each, plus the logits, for the prefill and every decode step.
- The same tokens through the reference loaded in fp32 (every weight widened exactly; the f32 tensors checked to equal the restored ones of the bf16 model).
- Per prompt: tokens, residual stream and logits of both runs, and in the manifest the bf16 reference's KL against the fp32 run (mean, max) and the number of positions where their greedy choices differ. On the data of 2026-11-24: mean KL 5.3e-4 (prose) and 4.4e-4 (chat).
- The golden data grows to 620 MiB and about 9 s.

Commit `ed38a7c`.

#### 4.5 `tests/test_forward.c` and `tests/tiny.h`

| File | Kind | What it is for |
| --- | --- | --- |
| `tests/tiny.h` | New | The tiny synthetic Qwen3.5 checkpoints, moved out of `test_model.c` and shared; new: pseudo-random contents from a seed, and `real_dtypes` (`A_log` and `linear_attn.norm.weight` in F32, as published) |
| `tests/test_model.c` | Changed | Uses `tests/tiny.h` |
| `tests/test_forward.c` | New | The forward pass through the contract only |
| `Makefile` | Changed | `test_forward` in `TESTS` |

What was implemented:

- `state_specs`: the specs of a tiny model (kinds, layers, element counts), `ath_state_bytes` in bf16 and f32, kind names, `ath_state_init` errors, `ath_state_reset`.
- `forward_rejects`: every argument check of `ath_model_forward`, the state filling up, the scratch arena released after every call, MoE unsupported.
- `forward_pieces_bf16`, `forward_pieces_f32`: 12 tokens all at once, as 5 + 1 + 6 and one at a time give bit-identical logits and states; logits of the last token alone equal the last row; a reset state runs the sequence again bit for bit.
- `forward_hooks`: `on_residual` called once per layer and for the final norm, in order, with the call's rows and the activation dtype; the stream entering layer 0 is the embedding of the tokens.
- `real_state_specs`: 6 KV, 18 conv and 18 recurrent buffers on Qwen3.5-0.8B; 18.6 MiB fixed plus 12 KiB per token in bf16.
- `forward_golden`: four runs in parallel threads (each prompt in bf16 and in f32, the prompt in one call and then one token at a time), with `on_residual` comparing the residual stream with both reference runs as it goes. Results: the bf16 mean KL over all 119 positions 3.0e-4 against the reference's 5.0e-4; greedy agreement at all 104 positions above the bf16 noise (0.21); with f32 activations, mean KL 4e-11, the residual stream within 4.7e-5 and the logits within 2.7e-5 of the fp32 reference (tolerances 2e-4 and 1e-4). A DEBUG build runs the first 4 positions only.
- What the measurements showed along the way, recorded in `tests/README.md` and the contract: per prompt, athanor beats the reference on prose (1.7e-4 against 5.3e-4) but not on chat (6.0e-4 against 4.4e-4), because the chat prompt's first position is ill-conditioned (a near-cancellation in layer 15 that turns one bf16 rounding into a 10% error; the reference's own KL there goes from 2.7e-4 to 5.5e-3 when two small tensors are rounded as transformers does by default). The gate pools all positions, as the resolution of the open question wrote it; the per-prompt test that was implemented first failed on chat, and the reference with its default sdpa attention would fail it too.
- A tiny-model check of bf16 against f32 activations was dropped: with uniform random weights the network is ill-conditioned (bf16 and f32 logits differ by 2% to 30% depending on the seed), so no fixed bound would mean anything; the real checkpoint does that check.

Commits `ed38a7c`, `370a452` (the pooled KL line names the positions it covers).

#### 4.6 Documentation and closing step 4

| File | Kind | What it is for |
| --- | --- | --- |
| `docs/ARCHITECTURE.md` | Changed | The contract as of step 4 (state specs, `ath_state`, `ath_batch`, `ath_hooks`, `forward`, the functions of `state.c`); the departures from the first design; a section on state and the forward pass; tree (`state.c`, `tiny.h`, `test_forward.c`) |
| `src/model/README.md` | Changed | `state.c`; running a model; state specs |
| `src/models/qwen35/README.md` | Changed | What `qwen35.h`, `family.c`, `weights.c` and `forward.c` hold now; the forward pass and the layout of the state buffers |
| `docs/contract/qwen35-0.8b.md` | Changed | The logits difference measured; End-to-end section (results, the ill-conditioned position) |
| `tests/README.md` | Changed | Forward pass section (checks, measured values, why the gate pools the prompts, environment variables), `tiny.h`, `test_forward` in the suite table |
| `docs/MODELS.md` | Changed | Status: the forward pass checked end to end |
| `docs/program/PROGRAM.md` | Changed | Rows 3, 4 and 9: the forward pass, the composed Gated DeltaNet layer, the CPU state of one sequence |
| `docs/STATUS.md` | Changed | Step 4 tasks, criteria and results; step 5 preview; decisions in the log |

Commit `d1e93fe`.

---

## Files by step

Every file of the repository and the steps that created or changed it (documentation-only edits included).

| File | Step 1 | Step 2 | Step 3 | Step 4 |
| --- | --- | --- | --- | --- |
| `Makefile` | 1.1, 1.8 | 2.3, 2.6 | 3.3, 3.4, 3.5, 3.6 | 4.5 |
| `src/core/base.{h,c}` | 1.2 | | | |
| `src/core/arena.{h,c}` | 1.3 | | | |
| `src/core/file.{h,c}` | 1.4, 1.7 | | | |
| `src/core/json.{h,c}` | 1.5 | | | |
| `src/core/tensor.{h,c}` | 1.6 | | 3.2 | |
| `src/core/safetensors.{h,c}` | 1.7 | | | |
| `src/model/model.h` | | 2.2 | | 4.2 |
| `src/model/registry.c` | | 2.3 | | |
| `src/model/state.c` | | | | 4.2 |
| `src/models/qwen35/qwen35.h`, `config.c` | | 2.4 | | 4.3 (`qwen35.h`) |
| `src/models/qwen35/weights.c`, `family.c` | | 2.5 | | 4.3 |
| `src/models/qwen35/forward.c` | | | | 4.3 |
| `src/ops/ops.h`, `check.h`, `embed.c`, `matmul.c`, `norm.c`, `rope.c`, `act.c`, `attn.c`, `conv.c`, `gdn.c` | | | 3.3 | |
| `tools/athanor.c`, `commands.h` | 1.8 | | | |
| `tools/inspect.c` | 1.8 | 2.6 | | |
| `tools/dev/alloc_trace.c` | 1.8 | | | |
| `tools/dev/check_alloc.sh` | 1.8 | 2.6 | | |
| `tests/test.h` | 1.9 | | | |
| `tests/test_core.c` | 1.9 | 2.7 | | |
| `tests/util.h`, `tests/test_model.c` | | 2.7 | | 4.5 (`test_model.c`) |
| `tests/test_ops.c` | | | 3.5 | |
| `tests/test_golden.c` | | | 3.6 | |
| `tests/tiny.h`, `tests/test_forward.c` | | | | 4.5 |
| `tests/golden/dump.py`, `README.md` | | | 3.4 | 4.4 |
| `tests/golden/requirements.in`, `requirements.txt` | | | 3.1 | |
| `docs/contract/qwen35-0.8b.md` | closing | 2.1 | 3.1, 3.7 | 4.6 |
| `docs/contract/qwen35-35b-a3b.md` | | 2.1 | | |
| `docs/MODELS.md` | | 2.1 | 3.7 | 4.6 |
| `docs/ARCHITECTURE.md` | closing | 2.2 | 3.7 | 4.6 |
| `docs/STATUS.md` | closing | 2.1 to 2.7, closing | 3.1, closing | 4.1, closing |
| `docs/program/PROGRAM.md` | closing | closing | 3.7 | 4.6 |
| `docs/status_extended.md` | | | closing | 4.6 |
| `AGENTS.md` | | 2.3 | 3.7 | |
| module READMEs | closing (`src/core`, `tests`, `tools`) | 2.2, 2.5, 2.6, 2.7 (`src/model`, `src/models/qwen35`, `tools`, `tests`) | 3.2, 3.3, 3.6, 3.7 (`src/core`, `src/ops`, `tests`, `tests/golden`, `src/backend`) | 4.6 (`src/model`, `src/models/qwen35`, `tests`, `tests/golden`) |
