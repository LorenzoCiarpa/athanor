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
  - [Step 5: tokenizer, sampling, CPU chat](#step-5-tokenizer-sampling-cpu-chat)
  - [Step 6: GPU backend](#step-6-gpu-backend)
  - [Step 7: GPU ops, one by one](#step-7-gpu-ops-one-by-one)
  - [Step 8: sequence state and engine](#step-8-sequence-state-and-engine)
  - [Before step 9: source layout](#before-step-9-source-layout)
  - [Step 9: fast decode and bench](#step-9-fast-decode-and-bench)
  - [Step 10: fast prefill](#step-10-fast-prefill)
  - [Step 12: MoE and tiering](#step-12-moe-and-tiering)
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

#### 4.7 Fast CPU matmul (added after the step closed)

Added at the maintainer's request on 2026-11-24. With `ath_matmul_ref` the CPU chat of step 5 would decode about 2 tokens per second. Three designs were measured with a microbenchmark outside the repository:

- B: scalar, several outputs per thread;
- C: AVX-512 across outputs on repacked weights;
- along k, as BLAS libraries and `vdpbf16ps` do.

B was chosen, and C was written down for later.

| File | Kind | What it is for |
| --- | --- | --- |
| `src/core/threads.h`, `threads.c` | New | A fixed pthread pool: `ath_pool_init` (n − 1 workers, their descriptors from an arena), `ath_pool_run` (fork and join), `ath_pool_free`, `ath_pool_threads`, `ath_pool_default_threads` (online CPUs) |
| `src/ops/matmul.c` | Changed | The argument checks shared by both versions (`check`); `ath_matmul_fast` |
| `src/ops/ops.h` | Changed | Declaration of `ath_matmul_fast`; includes `core/threads.h` |
| `src/model/model.h`, `src/model/state.c` | Changed | `ath_pool *pool` in the family's `forward` and in `ath_model_forward`, passed through |
| `src/models/qwen35/qwen35.h`, `forward.c` | Changed | The pool in the state of a forward call; every matmul, the lm_head included, through `ath_matmul_fast` |
| `tests/test_core.c` | Changed | `pool_runs_every_task_once`, `pool_shared_by_threads` |
| `tests/test_ops.c` | Changed | `matmul_fast_same_bits`; a three-thread pool for the suite |
| `tests/test_golden.c` | Changed | Every matmul record also through `ath_matmul_fast` on a pool of all CPUs, required bit-identical to `ath_matmul_ref` |
| `tests/test_forward.c` | Changed | The pool argument everywhere; the four golden runs share one pool, started after the setup and freed before anything can return early |
| `docs/proposals/cpu-matmul-avx512.md` | New | Variant C, not scheduled: idea, layout, code checked bit for bit against the reference, measurements, costs, when it would pay, the decisions it would need |
| `src/ops/README.md`, `src/core/README.md`, `src/backend/README.md`, `src/model/README.md`, `src/models/qwen35/README.md`, `tests/README.md`, `docs/ARCHITECTURE.md`, `docs/program/PROGRAM.md` (rows 3 and 8), `docs/STATUS.md` | Changed | The fast path, the pool, the contract change, the test times, task 4.7 with its criterion, results and decisions |

What was implemented:

- **The pool.**
  - A run publishes the function, its context and the task count under a mutex, bumps a generation counter and wakes the workers. Each worker runs its share and decrements a pending count; the last one wakes the caller.
  - Worker i runs tasks i, i + n, …, and the caller is thread 0. A worker that wakes without a new generation goes back to sleep.
  - Runs from several threads take turns on a second mutex. A NULL pool, or a pool of one thread, runs the tasks in order on the caller.
  - The pool is passed explicitly, because AGENTS.md allows no globals. That is why `forward` takes it, and why `src/backend/cpu.c` is still not written.
- **`ath_matmul_fast`.**
  - Each thread takes a contiguous range of blocks of 8 outputs. For every row of x, the 8 outputs of a block are 8 fp32 chains in order of k, a rounded product and a rounded sum per step, which is exactly what the reference computes. The same bits therefore come out with any number of threads.
  - The 8 chains overlap in the pipeline: that is the whole gain on one thread. Rows pass over a block before the next block starts, so its weights stay in cache during prefill.
  - The inner loop reads the bf16 weights directly. The first version used the generic `ld`, which branches on the dtype; the compiler then kept the 8 accumulators in memory, and the speedup on one thread was 1.6× instead of 3.5×.
  - Weights in f32 (only in tests) and the last partial block run one output at a time. Under 2^18 multiply-adds the call stays on the calling thread.
- **Verification.**
  - Random shapes and dtypes in `test_ops`, and every golden matmul record in `test_golden`, bit for bit.
  - Every number `test_forward` prints is unchanged.
  - A scratch program hashed every logit of 32 prefill tokens and 8 decode steps, in bf16 and f32. The hashes were the same for the tree before 4.7 (`ath_matmul_ref`) and for 4.7 with 1 and 8 threads.
- **Speed, for the record (not a performance claim).** On the Ryzen 7 9800X3D with 8 threads:
  - a decode step with logits went from 0.43 s to 41 ms, and 32 prefill tokens with all their logits from 13.7 s to 0.84 s;
  - `make test` went from about 45 s to about 13 s (`test_forward` 37 s to 11 s).

Commits `715c7f7` (the pool), `cdc40aa` (`ath_matmul_fast` and its tests), `25c4bed` (the pool in the contract, the forward pass on the fast matmul), and the documentation commit that follows them.

#### 4.8 CHANGE-001: scope boundary, tool calling, prefix cache, vision tensors (added after the step closed)

A specification change from the maintainer, `docs/CHANGE-001-scope-tools-cache.md`, applied on 2026-11-24 before step 4 was merged. The STATUS log lists each of its edits and what was deferred.

| File | Kind | What it is for |
| --- | --- | --- |
| `docs/CHANGE-001-scope-tools-cache.md` | New | The change itself, as the maintainer wrote it |
| `src/model/model.h` | Changed | `ATH_P_UNUSED`; the header comment, `ath_ignored`, `params` and `ath_model_open` describe the new coverage rule |
| `src/model/registry.c` | Changed | Coverage: every tensor bound; a group's tensors bound only by unused params and in one group; every unused param in a group |
| `src/models/qwen35/weights.c`, `qwen35.h` | Changed | The tensors of the two ignored groups bound last, one whole-tensor param each, role other, layer −1, flagged `ATH_P_UNUSED`; capacity counted from the checkpoint |
| `tools/inspect.c` | Changed | Flag name `unused`; per-layer and per-role totals over the used params, the unused ones on their own line (`unused` in JSON); coverage line `B bound to P params (I in ignored groups, unused), U unbound` |
| `tests/test_model.c` | Changed | `count_role` counts used params; the tiny checkpoints check the 4 unused params (flag, role, layer, zero-copy, after the used ones) and the MoE count; the real checkpoint: all tensors bound, unused params = ignored tensors |
| `tests/golden/dump.py`, `README.md` | Changed | `check_text_path`: stop unless the loaded class is `Qwen3_5ForCausalLM` with no vision parameters; `reference.class` in the manifest |
| `AGENTS.md` | Changed | Invariant 10, no agent features |
| `README.md` | Changed | The agent-harness row (out of scope by design); the status line, still at step 1, brought to step 4 |
| `docs/ARCHITECTURE.md` | Changed | Section "Scope: engine versus harness"; `ATH_P_UNUSED` and the text members (`ath_tool_def`, `ath_msg_role`, `ath_out_kind`, `ath_out_event`, `chat`, `parse`) as the contract target; the coverage rule; how quant, ablit, train and net use them; tool rendering and output parsing family-specific; the tree |
| `docs/PLAN.md` | Changed | Scope decision; gates of steps 2, 5, 13 and 14; the ds4 row; the multimodal risk |
| `docs/MODELS.md`, `docs/contract/qwen35-0.8b.md`, `docs/contract/qwen35-35b-a3b.md` | Changed | The vision encoder and MTP head bound as unused |
| `src/model/README.md`, `src/models/qwen35/README.md`, `src/net/README.md`, `src/infer/README.md`, `src/text/README.md`, `tools/README.md` | Changed | Coverage rule; `text.c` (tool definitions, thinking switch, parser; markup from the reference template, recorded in `docs/contract/`); event conversion, `ATH_MSG_TOOL`, session id; prefix snapshots on disk; `parse`; the `inspect` output |
| `docs/program/PROGRAM.md` | Changed | Rows 6b (tool calling) and 9b (prefix caching on disk) |
| `docs/STATUS.md` | Changed | Task 4.8, criterion 9 and its result, notes, the amended step 5 preview, the log |

What was implemented:

- **Unused params.**
  - Before, the vision encoder and the MTP head were ignored groups: declared with a reason and counted, but unbound.
  - Now each of their tensors is a param flagged `ATH_P_UNUSED`, so pipelines see them. An export can copy them unchanged, and the text path skips them.
  - The groups stay as the family's declaration of why the tensors are there. The registry checks that each group and the unused params coincide, so a tensor can be neither used and unused at once, nor unused without a reason.
- **Unchanged results.** The 320 used params, their order and every forward and golden number are the same; the regenerated golden data only adds the reference class to the manifest.
- **Deferred to step 5.** The text members of the contract, the template fixtures and `tests/test_text.c` arrive with `text.c`, their first user.

---

### Step 5: tokenizer, sampling, CPU chat

Branch `step5-tokenizer-chat`, from `main` after step 4 and CHANGE-001 were merged. The gate of `docs/PLAN.md`, as CHANGE-001 amended it: exact token round trip against the reference tokenizer, prompts byte-identical to the reference chat template with and without tools, parser fixtures, a first chat.

#### 5.1 Text golden data

| File | Kind | What it is for |
| --- | --- | --- |
| `tests/golden/text.py` | New | The reference's tokenizer, template and generation, recorded for `test_text` |
| `Makefile` | Changed | `make golden` runs `text.py` after `dump.py` |

What was implemented:

- **Corpus.**
  - 43 hand-written cases: languages and scripts, emoji sequences, numbers, contractions in every case (U+017F included), white space of every kind, code, JSON, HTML, URLs, punctuation runs, marks, text that needs NFC, added tokens inside text and their prefixes, tool markup, long runs, control and private-use characters, unassigned code points.
  - Five repository files and the model card.
  - Every code point but the surrogates, in five contexts, in chunks of 8192.
  - Every canonical decomposition, decomposed, with up to three extra marks shuffled by a seeded generator, and Hangul syllables as jamo.
  - For each text: the text, the ids from `tokenizer.encode`, and the text from `tokenizer.decode`.
- **Templates.** 20 conversations, each rendered with `apply_chat_template` (and tokenized) or recorded as an error. The C view of each conversation stores schemas and arguments as compact JSON text, so the test also checks that athanor re-serializes them as `json.dumps` does.
- **Generation.** The prompt of a one-message chat, 32 greedy steps of the fp32 reference with the top-2 margin of each, and the bf16 noise of the margin: the bf16 reference run on the same tokens, the largest change of the margin.

#### 5.2 Unicode

| File | Kind | What it is for |
| --- | --- | --- |
| `tools/dev/gen_unicode.py` | New | Generates the tables from Python's `unicodedata`, the normalization limited to what the reference tokenizer applies |
| `src/text/unicode_tables.h` | New (generated) | Class ranges (1,152), combining-class ranges (333), full canonical decompositions (2,060), primary composites (940) |
| `src/text/unicode.h`, `unicode.c` | New | UTF-8 decoding (invalid bytes kept as themselves) and encoding, `ath_unicode_class`, `ath_unicode_ccc`, `ath_nfc` |

What was implemented:

- **Classes** for the regex: letter, mark, number, and white space as Oniguruma's `\s`, from Unicode 16.0.
- **NFC** after UAX #15: full decomposition (Hangul computed), a stable sort of each run of non-starters, composition with the last starter when not blocked. Text below U+0300 is returned as it is (a quick check that holds for every such code point).
- **Two Unicode versions.** The first corpus run failed on the decompositions case: the reference did not compose a Todhri letter added in Unicode 16. Its NFC uses Unicode 9.0 data. The generator now runs the reference's NFC and NFD on every entry and keeps only what it applies; 21 compositions and 29 combining classes are left out.

#### 5.3 `src/text/bpe.c`

| File | Kind | What it is for |
| --- | --- | --- |
| `src/text/bpe.h`, `bpe.c` | New | `ath_tokenizer`: load, encode, decode; `ath_spans`, `ath_tokens` |
| `Makefile` | Changed | `src/text/*.c` in the library |

What was implemented:

- **Loading.** `tokenizer.json` is parsed into a temporary arena. Token bytes (mapped back from the GPT-2 alphabet), the added tokens and a merge table (open addressing, pair of ids → rank and result) stay. A temporary byte-string → id map resolves the merges. Unsupported features fail with `ATH_ERR_UNSUPPORTED`, and the split regex must equal the family's.
- **Encoding.** Added tokens are found first (a first-byte filter, then the longest match). The text between them goes through NFC and the family's split; each piece is merged from bytes, always taking the lowest-rank pair, leftmost among equals. This is the order Hugging Face's priority queue gives.
- **Speed**, for the record: load 0.06 s, 28 million tokens encoded in 0.6 s.

#### 5.4 Contract and core helpers

| File | Kind | What it is for |
| --- | --- | --- |
| `src/model/model.h` | Changed | Text types, the family's text members, `ath_model_tokenizer`, `ath_model_chat`, `ath_model_parse`, `ath_state_copy` |
| `src/model/text.c` | New | The three text functions: argument checks, then the family |
| `src/model/state.c` | Changed | `ath_state_copy` |
| `src/core/buf.h`, `buf.c` | New | `ath_buf` |
| `src/core/json.h`, `json.c` | Changed | `ath_json_write`, `ath_json_write_str`, `ath_json_write_double` with Python's formatting |
| `src/core/rng.h`, `rng.c` | New | `ath_rng` |

What was implemented:

- **The text contract** of CHANGE-001, with the details the step settled (STATUS log): what a message carries, the chat options, the parser state with the family's state behind a pointer, the end of output, the event lifetime, and a bound on the number of events.
- **`pretokenize_regex`**, a member the CHANGE did not have, so that the tokenizer can refuse a file with another regex.
- **`ath_state_copy`** copies every buffer and `n_past` between two states of equal capacity and dtype.
- **The JSON writer** reproduces `json.dumps(ensure_ascii=False)`: separators, escapes (`\u00xx` in lower case for control characters), floats in Python's shortest repr, found by trying increasing precision until the value reads back the same.

#### 5.5 `src/models/qwen35/text.c`

| File | Kind | What it is for |
| --- | --- | --- |
| `src/models/qwen35/text.c` | New | `ath_qwen35_pretokenize` (and its regex), `ath_qwen35_chat`, `ath_qwen35_parse` |
| `src/models/qwen35/qwen35.h`, `family.c` | Changed | Declarations; the text members of `ath_family_qwen35` |

What was implemented:

- **Pre-tokenization**: at each position the first alternative of the regex that matches, each with its own backtracking written out. The optional character of the letter alternative is taken first and given back if no letter follows. The three white-space alternatives share one scan of the run: up to its last CR or LF, all of it at the end of the text, all but its last character, or all of it.
- **The template**, line by line from `chat_template.jinja`:
  - the tools block with the fixed instructions;
  - the system message, trimmed with Python's `str.strip`;
  - the last real user query, which decides which assistant turns show their thinking;
  - reasoning taken from the content around `<think>` and `</think>` when not given apart;
  - tool calls with arguments printed as Python's `str()` or `tojson`;
  - runs of tool results in one user turn;
  - the generation prompt with or without thinking;
  - the three errors the template raises.
- **The parser**, a state machine over a buffer of unconsumed bytes, in six modes: text, thinking, after `<tool_call>`, inside a function, inside a parameter value, after `</function>`.
  - A tag split across pieces is held back until it is complete, and so are newlines that may turn out to be formatting.
  - A call begins only once `<function=NAME>` is complete, so a malformed header can still be given back as text.
  - Each value becomes JSON by its schema type. Events of one kind merge, and their bytes are copied into one buffer per call.

#### 5.6 `src/ops/sample.c`

| File | Kind | What it is for |
| --- | --- | --- |
| `src/ops/sample.c` | New | `ath_sample_ref` |
| `src/ops/ops.h` | Changed | `ath_sampler`, `ath_sample_ref` |

What was implemented: greedy, then temperature, top-k (an insertion-sorted list of the k best), or a heapsort of the whole vocabulary without top-k; top-p and min-p cuts; inverse-transform draw with the caller's uniform number. Probabilities are relative weights in double, and scratch memory is released.

#### 5.7 `athanor chat`

| File | Kind | What it is for |
| --- | --- | --- |
| `tools/chat.c` | New | The subcommand |
| `tools/athanor.c`, `commands.h` | Changed | Its entry in the dispatch table |

What was implemented:

- **Options and modes.** Option parsing with the model card's defaults; one-shot `-p` or messages from stdin with `/reset`, `/tool` and `/exit`; tool definitions from a JSON file.
- **Each turn**:
  - render the whole conversation;
  - tokenize it, and tokenize it again without the generation header to find the stable prefix;
  - continue from the state, or restore its copy, or reset;
  - run the prompt with a copy taken at the stable prefix (none in one-shot mode);
  - sample until `<|im_end|>` or `<|endoftext|>`, parse each token's bytes and show the events;
  - store the answer (text, thinking, calls) in the history.
- **Memory.** One arena for the model, the tokenizer and the states; one for the history; one per turn; one for the forward pass's scratch memory.

#### 5.8 `tests/test_text.c`

| File | Kind | What it is for |
| --- | --- | --- |
| `tests/test_text.c` | New | 12 tests (README of `tests/`) |
| `Makefile` | Changed | `test_text` in the suites |

What was implemented: unit tests that need no data (UTF-8, classes, NFC, the JSON writer, pre-tokenization pieces, 17 parser fixtures under every split, template errors, sampling rules and distribution), and the three golden comparisons. The parser fixtures caught one difference on the first run: newlines after a tool call were kept as text. The first DEBUG run caught one `memcmp` with a NULL pointer in the test itself.

#### 5.9 Documentation and closing step 5

| File | Kind | What it is for |
| --- | --- | --- |
| `docs/STATUS.md` | Changed | Step 5 current and done, step 4 completed and merged, step 6 preview, open question (presence penalty), log |
| `docs/ARCHITECTURE.md` | Changed | Contract excerpt as of step 5, section Text, `athanor chat` in the pipeline table, the tree, the template engine in "What athanor writes itself" |
| `docs/contract/qwen35-0.8b.md` | Changed | Sections Tokenizer, Chat template, Output markup |
| `src/core/README.md`, `src/text/README.md`, `src/ops/README.md`, `src/model/README.md`, `src/models/qwen35/README.md`, `tools/README.md`, `tests/README.md`, `tests/golden/README.md` | Changed | New files and interfaces, `athanor chat`, the text tests and data, times |
| `docs/program/PROGRAM.md` | Changed | Rows 5, 6 and 6b |
| `README.md` | Changed | Status line |

---

### Step 6: GPU backend

Branch `step6-gpu-backend`, from `main` after step 5 was merged and tagged `v0.1-cpu-chat`. The gate of `docs/PLAN.md`: allocation, copies, streams, events and timing on CUDA. Before the step the toolkit was upgraded (STATUS, open questions): CUDA 13.5 from NVIDIA's redist archives, selected with `CUDA_HOME`, on the existing 580 driver.

#### 6.1 Build

| File | Kind | What it is for |
| --- | --- | --- |
| `Makefile` | Changed | `GPU=cuda` when `cuda` is among the goals; `build/cuda`, `build/cuda-debug`; `CUDA_HOME`, `CUDA_ARCH`, `NVCC`, `NVCCFLAGS`; a pattern rule for `.cu`; static CUDA runtime in `LDLIBS`; the backend sources (`gpu.cu` or `gpu_none.c`); `test_backend` among the suites; `ASAN_OPTIONS` for the CUDA debug tests |
| `tools/dev/check_nvcc.sh` | New | Fails with a clear message when `CUDA_HOME` has no nvcc, or one too old for `CUDA_ARCH` (12.8 for `sm_120`) |

#### 6.2 to 6.5 The backend

| File | Kind | What it is for |
| --- | --- | --- |
| `src/backend/backend.h` | New | The interface: `ath_device` (properties, three streams), device and pinned memory, `ath_copy`, `ath_memset`, stream and device synchronization, `ath_event` (ordering, timing), `ath_dev_arena` |
| `src/backend/gpu.cu` | New | The CUDA implementation: errors mapped and the sticky error cleared, the device selected on every call, non-blocking streams, events with or without timing |
| `src/backend/gpu_none.c` | New | The interface in a build without a GPU backend |
| `src/backend/compat.h` | New | CUDA names to HIP for every runtime call in use; `ATH_WARP`; `ATH_GPU_BACKEND` |
| `src/backend/devarena.c` | New | The device arena: blocks from `ath_dev_alloc`, records in a host arena, bumping, mark and reset, blocks kept and reused in order |
| `src/core/base.h`, `base.c` | Changed | `ATH_ERR_DEVICE`; `extern "C"` guards |
| `src/core/arena.h` | Changed | `extern "C"` guards |

What was implemented:

- **Opening a device.** It reads the properties: name, compute capability, multiprocessors, warp size (refusing anything but 32), memory, L2, the opt-in shared memory per block, driver and runtime versions. It then creates the three streams, non-blocking so that they never synchronize with the legacy default stream.
- **Copies** use `cudaMemcpyAsync` with `cudaMemcpyDefault`: unified virtual addressing gives the direction from the pointers, so one function serves every pair of memory kinds.
- **The device arena** follows the core arena's semantics. The differences are in the bookkeeping: records on the host, and a request that does not fit the next kept block gets a new block inserted before it, so blocks stay in allocation order and marks remain meaningful.

#### 6.6 `athanor devices`

| File | Kind | What it is for |
| --- | --- | --- |
| `tools/devices.c` | New | The subcommand: devices and properties; `--bench` times 256 MiB copies with events, best of 5 |
| `tools/athanor.c`, `commands.h` | Changed | Its entry in the dispatch table |

#### 6.7 `tests/test_backend.c`

| File | Kind | What it is for |
| --- | --- | --- |
| `tests/test_backend.c` | New | 7 tests: the stub (CPU build only); on the GPU, properties, copies and memset, streams ordered by events, timing, an impossible allocation, the device arena |

What was found while testing:
- Under AddressSanitizer the CUDA driver did not start: ASan reserves the address range the driver maps into. The CUDA debug tests run with `ASAN_OPTIONS=protect_shadow_gap=0`.
- compute-sanitizer reports the two API errors of the deliberate impossible allocation, and nothing else.

#### 6.8 Documentation

| File | Kind | What it is for |
| --- | --- | --- |
| `docs/STATUS.md` | Changed | Step 6 current and done, step 5 completed, merged and tagged, the CUDA question resolved, the `output_gate_type` question, step 7 preview, log |
| `docs/proposals/chat-state-at-last-query.md`, `chat-shadow-sequence.md` | New | Two ways to reuse the state across chat turns, written down at the maintainer's request, not scheduled |
| `src/backend/README.md` | Changed | Rewritten for what exists: files, building, the interface, streams, tests |
| `src/core/README.md`, `tools/README.md`, `tests/README.md` | Changed | `ATH_ERR_DEVICE` and the C linkage; `athanor devices` and `check_nvcc.sh`; `test_backend` and the CUDA test commands |
| `docs/ARCHITECTURE.md`, `CLAUDE.md`, `docs/program/PROGRAM.md` | Changed | Tree and Makefile targets; `make cuda test`; row 7 |

---

### Step 7: GPU ops, one by one

Branch `step7-gpu-ops`, from `main` after step 6 was merged. The gate of `docs/PLAN.md`: every kernel matches its CPU reference. Correct and simple kernels; the fast ones are steps 9 and 10.

#### 7.1 The matmul library in the backend

| File | Kind | What it is for |
| --- | --- | --- |
| `src/backend/backend.h` | Changed | `ath_device.blas`, `blas_ws`, `blas_ws_bytes`; `ATH_BLAS_WORKSPACE` (32 MiB); the compute stream named as the stream of every op |
| `src/backend/gpu.cu` | Changed | Creates the cuBLASLt handle and the workspace in `ath_device_open`, frees them in `ath_device_close`; uses `gpu.h` |
| `src/backend/gpu.h` | New | Private to GPU code: `ath_gpu_fail`, `ATH_GPU_CHECK`, `ath_gpu_use`, `ath_gpu_stream`, `ath_gpu_launched` |
| `src/backend/compat.h` | Changed | `cublasLt.h`; the hipBLASLt names; `ath_shfl_xor` |
| `Makefile` | Changed | `-lcublasLt` with `-rpath` to the toolkit; `-fno-threadsafe-statics` for device code; `src/ops/*.cu` built as `.cu.o`; `src/ops/gpu_none.c` in the CPU build; `test_gpu_ops` among the suites |
| `tools/dev/check_nvcc.sh` | Changed | Fails when the toolkit has no `libcublasLt.so` |

What was implemented:

- **Linking.** cuBLASLt is linked as a shared library. Statically, a program that makes one bf16 matmul is 585 MB (measured with CUDA 13.5): the static archive carries the kernels for every architecture.
- **Launches without libstdc++.** The launch stubs that nvcc generates keep the kernel's address in a function-local static. C++ guards its initialization with `__cxa_guard_acquire`, which lives in libstdc++, and the binaries are linked as C. `-fno-threadsafe-statics` drops the guard; the value is a constant.

Commit `0e12354` (the build flags for kernels landed with them, in `71ccfff`).

#### 7.2 The GPU entry points

| File | Kind | What it is for |
| --- | --- | --- |
| `src/ops/ops.h` | Changed | `ath_<op>_gpu` for the 13 ops of the forward pass, the GPU conventions, `ATH_ROPE_MAX_DIM`, `ATH_GDN_MAX_DK`; includes `backend/backend.h`; `extern "C"` guards |
| `src/ops/check.h` | Changed | `ops_tensor` no longer checks the tier; `ops_place` / `OPS_PLACE` do, for the host (`_ref`, `_fast`) or the device (`_gpu`) |
| `src/ops/{embed,matmul,norm,rope,act,attn,conv,gdn}.c` | Changed | One argument check per op shared by `_ref` and `_gpu`; `ath_<op>_gpu`, which checks and calls the launcher; `rope.c` computes the frequencies for both |
| `src/ops/sample.c` | Changed | `OPS_PLACE` (the tier check moved out of `ops_tensor`) |
| `src/ops/launch.h` | New | Private: the launchers, one per op (one for the three norms, one for the three elementwise ops) |
| `src/ops/gpu_none.c` | New | The launchers in a build without a GPU: `ATH_ERR_UNSUPPORTED` |
| `src/core/tensor.h`, `src/core/threads.h` | Changed | `extern "C"` guards (GPU code includes them through `ops.h`) |

Commit `71ccfff`, with 7.3.

#### 7.3 The kernels

| File | Kind | What it is for |
| --- | --- | --- |
| `src/ops/kernel.cuh` | New | Device code shared by the kernels: `k_ld`, `k_st`, `k_round` (bf16 rounded as the host does), `k_warp_sum`, `k_block_sum`, `k_silu`, `K_FOR`, `k_blocks`, `k_stream` |
| `src/ops/embed.cu` | New | One thread per element; an id outside the table gives NaN |
| `src/ops/matmul.cu` | New | cuBLASLt ("TN" on the checkpoint layout, fp32 accumulation, alignment preferences from the operands' addresses and pitches); `matmul_kernel`, one warp per output, for the other dtype combinations |
| `src/ops/norm.cu` | New | One kernel for RMSNorm, gated RMSNorm and the L2 norm: a block per row |
| `src/ops/rope.cu` | New | One thread per pair or copied element; the frequencies in the kernel's arguments |
| `src/ops/act.cu` | New | One kernel for SwiGLU, the sigmoid gate and the add |
| `src/ops/attn.cu` | New | A block per query and head; tiles of 32 keys with an online softmax |
| `src/ops/conv.cu` | New | Outputs, then the state (a thread per channel), two kernels on one stream |
| `src/ops/gdn.cu` | New | The gates; the recurrence, a warp per column of the state with the column in registers |

What was implemented:

- **The same expressions as the references**, in the same order where the order is not a reduction: the RMSNorm product, the gated norm's rounding, SiLU and the sigmoid, softplus with torch's threshold.
- **Reductions are deterministic.** The order of a sum depends on the launch shape only: a warp's butterfly (every lane gets the same bits), then the warps in order.
- **RoPE.** The frequencies come from the host, so that long positions do not amplify a difference between the device's `powf` and libm's.

Commit `71ccfff`.

#### 7.4 `tests/test_gpu_ops.c`

| File | Kind | What it is for |
| --- | --- | --- |
| `tests/test_gpu_ops.c` | New | 10 tests: argument checks and limits (both builds); embed, matmul, the norms, rope, the elementwise ops, attention, conv, the gates and the recurrence against `_ref` on random inputs, plus the bit-for-bit properties (pieces equal all at once) |

What was found while testing:

- **Comparing bf16 outputs in ulps** fails near zero, where a distance of several ulps is a tiny error. The comparison uses the golden score, against n ulps (n · 2^-7), and prints the ulp distance as information.

Commit `4dcf563`, with 7.5.

#### 7.5 GPU replay of the golden records

| File | Kind | What it is for |
| --- | --- | --- |
| `tests/test_golden.c` | Changed | With a GPU, every record also runs through the GPU kernel on device copies of the same inputs, outputs pre-filled with NaN; `gpu_tolerances`; a second table of statistics, printed and checked |

What was found while testing:

- **Three tolerances were too tight** on real inputs: the attention output (1.1e-5) and the state of the recurrence after 75 tokens (1.2e-5) went over the first guess of 1e-5. The tolerances now follow the rule of the step 3 tolerances, at least four times the worst error measured.

Commit `4dcf563`.

#### 7.6 Documentation

| File | Kind | What it is for |
| --- | --- | --- |
| `docs/STATUS.md` | Changed | Step 7 current and done, step 6 completed and merged, step 8 preview, the open question on invariant 7 and the two devices, log |
| `src/ops/README.md` | Changed | GPU kernels: files, kernels, how they are checked |
| `src/backend/README.md` | Changed | `gpu.h`, the matmul library, linking cuBLASLt, launches without libstdc++, `.cu.o` |
| `tests/README.md` | Changed | GPU kernels: the measure, the tolerances, the measured errors, the properties; `test_gpu_ops` |
| `src/core/README.md`, `docs/ARCHITECTURE.md`, `docs/program/PROGRAM.md` | Changed | The `extern "C"` guards; the tree; rows 7, 8 and 10 |

---

### Step 8: sequence state and engine

Branch `step8-engine`, from `main` after step 7 was merged. The gate of `docs/PLAN.md`: a GPU chat in bf16, and greedy output identical to the reference for 256 tokens on 50 prompts, read as the step 4 and 5 gates read it (where the reference's top-2 margin exceeds the bf16 noise).

#### 8.1 Strided copies

| File | Kind | What it is for |
| --- | --- | --- |
| `src/backend/backend.h`, `gpu.cu`, `gpu_none.c`, `compat.h` | Changed | `ath_copy_2d` (`cudaMemcpy2DAsync`; `hipMemcpy2DAsync`) |
| `tests/test_backend.c` | Changed | A gather of columns on the device and a scatter into wider host rows, byte for byte; a pitch smaller than the width refused |

Commit `ee6fff1`.

#### 8.2 Op entry points

| File | Kind | What it is for |
| --- | --- | --- |
| `src/ops/ops.h` | Changed | `ath_<op>(d, ...)` for the 13 ops |
| `src/ops/dispatch.c` | New | The op on device d (`_gpu`) or on the CPU (`_ref`, `ath_matmul_fast`) |

Commit `0b1013a`.

#### 8.3 The contract on two devices

| File | Kind | What it is for |
| --- | --- | --- |
| `src/model/model.h` | Changed | `ath_exec` (pool, device, the params' device copies, device scratch); `ath_state.dev`; `ath_state_init_gpu`; `ath_state_reset` returns a status; `forward` and `ath_model_forward` take `const ath_exec *` instead of `ath_pool *`; hooks may see device tensors; includes `backend/backend.h` |
| `src/model/state.c` | Changed | States on a device (buffers from a device arena), reset and copy queued on its compute stream; `ath_model_forward` checks that the pass runs where the state is and releases the device scratch memory |

What was implemented:

- **One composition, two devices.** The pool became a member of `ath_exec`, which says where a pass runs. A family reads a weight through its param, so the copies on the device are indexed like `m->params`: the forward pass finds a weight's copy from the param's offset in that array, without a second layer table.

#### 8.4 The forward pass on both devices

| File | Kind | What it is for |
| --- | --- | --- |
| `src/models/qwen35/forward.c` | Changed | The ops' entry points instead of `_ref` and `ath_matmul_fast`; working tensors from the host or the device scratch arena; `W()` for a param's tensor where the pass reads it; `copy_cols` with `ath_copy_2d` on a GPU; token ids and positions copied to the device; logits copied back and the stream synchronized at the end |
| `src/models/qwen35/qwen35.h` | Changed | The new `forward` signature |

The CPU pass is unchanged: the entry points call the same functions with the same arguments, and every CPU number of `test_forward` and `test_text` is the same.

#### 8.5 The engine

| File | Kind | What it is for |
| --- | --- | --- |
| `src/infer/engine.h`, `engine.c` | New | `ath_engine_open` (the device, every used param copied once, the exec), `ath_engine_state`, `ath_engine_close` |
| `Makefile` | Changed | `src/infer/*.c` in the library; `test_engine`; `golden-generate` |

#### 8.6 `athanor chat` on the GPU

| File | Kind | What it is for |
| --- | --- | --- |
| `tools/chat.c` | Changed | `--device cpu\|gpu` (the GPU by default when there is one, the CPU as fallback); the engine's exec and states; `ath_state_reset` checked |

8.3 to 8.6, and the GPU runs of `test_forward` (8.8): commit `faddfcc`.

#### 8.7 Golden generations

| File | Kind | What it is for |
| --- | --- | --- |
| `tests/golden/generate.py` | New | 50 chat prompts (topics, languages, code, system prompts, multi-turn, thinking), 256 greedy tokens each by the fp32 reference with its cache, the top-2 margins, and the bf16 noise of each prompt from one bf16 pass over the same tokens |

What was found while making it:

- **The reference decodes at 11 tokens/s** on 8 CPU threads (0.8 while the CPU was busy with the debug test suites): its Gated DeltaNet and convolution run torch fallbacks. The 12,800 steps take about 20 minutes, hence a separate make target.

#### 8.8 Tests

| File | Kind | What it is for |
| --- | --- | --- |
| `tests/test_engine.c` | New | The engine without a GPU; the weights on the device byte for byte; states on the device (reset, copy, the side a pass runs on); a sequence at once, in pieces and token by token against itself and the CPU; the gate on 50 prompts |
| `tests/test_forward.c` | Changed | `forward_golden` runs its four runs on the GPU too, through the engine, held to the same gate and tolerances; hooks copy device tensors back |
| `tests/test_text.c` | Changed | The new `ath_model_forward` signature |

What was found while testing:

- **The gate's noise.** The bf16 noise of the 50 prompts ranges from 0.17 to 0.62, with one prompt at 1.24. The largest over all prompts, the rule of the forward records, would check 8,352 of the 12,800 steps; each prompt's own noise checks 11,445. The gate uses the prompt's own, and reports the other: 0 differences either way.
- **On the tiny model the GPU gives the same bits** all at once, in pieces and token by token (cuBLASLt chose the same algorithm for every row count there), and differs from the CPU by 9.8e-6 in f32.

8.7 and `test_engine`: commit `fc151ee`; the documentation: `98d16f7` and the commit that names them here.

#### 8.9 Documentation

| File | Kind | What it is for |
| --- | --- | --- |
| `docs/STATUS.md` | Changed | Step 8 current and done, step 7 completed and merged, step 9 preview, log |
| `docs/ARCHITECTURE.md` | Changed | `ath_exec`, states on a device, the new signatures, why; the tree |
| `src/model/README.md`, `src/models/qwen35/README.md`, `src/infer/README.md`, `src/ops/README.md`, `src/backend/README.md`, `tools/README.md`, `tests/README.md`, `tests/golden/README.md` | Changed | Running on a GPU; the forward pass on both devices; the engine; the entry points; `ath_copy_2d`; `chat --device`; the Engine section; `generate.py` |
| `docs/program/PROGRAM.md`, `README.md` | Changed | Row 9; the status paragraph and the build commands |

---

### Before step 9: source layout

Branch `step9-fast-decode`, before the first task of step 9, at the maintainer's request (2026-11-25; the options and the reasons are in the log of `STATUS.md`). Code is grouped by layer, then by language: portable C at the root of a module, CUDA C in its `cuda/` folder.

| File | Kind | What it is for |
| --- | --- | --- |
| `src/backend/cuda/gpu.cu`, `gpu.h`, `compat.h` | Moved | From `src/backend/`; include paths and comments only |
| `src/ops/cuda/kernel.cuh`, `embed.cu`, `matmul.cu`, `norm.cu`, `rope.cu`, `act.cu`, `attn.cu`, `conv.cu`, `gdn.cu` | Moved | From `src/ops/`; include paths and comments only |
| `Makefile` | Changed | GPU sources from `src/backend/cuda/` and `src/ops/cuda/`; `.cu` objects named `.o` again, since a kernel and its op's reference no longer share a directory |
| `src/core/base.h`, `src/backend/backend.h`, `src/ops/ops.h`, `src/ops/launch.h`, `src/ops/matmul.c` | Changed | Comments that name the moved files; `launch.h` says why it stays at the root (the boundary between the C and the kernels) |
| `src/backend/README.md` | Changed | Source layout (new section): the rule, the three places that hold GPU code, `hip/` and `metal/` if ever needed; the file table, object names, launchers, one source for two vendors |
| `src/ops/README.md`, `src/models/qwen35/README.md`, `src/quant/README.md` | Changed | The kernels' paths; fused kernels in `src/models/<family>/cuda/fused.cu` |
| `AGENTS.md` | Changed | Invariant 4 and the op checklist name `src/ops/cuda/<op>.cu` and `src/backend/cuda/compat.h`; Code style gets a Source layout row |
| `docs/ARCHITECTURE.md`, `docs/PLAN.md`, `docs/program/PROGRAM.md` | Changed | The tree and rule 6; the Source layout decision, and the paths of steps 6 and 7 marked as moved; rows 7, 8 and 10 |

The move: commit `7e8aaad`; the documentation: the commit that names it here.

Then, the same day, two rules written down at the maintainer's request:

| File | Kind | What it is for |
| --- | --- | --- |
| `AGENTS.md` | Changed | Invariant 11, pipelines contain no GPU code; the op checklist covers operators only a pipeline uses |
| `docs/ARCHITECTURE.md` | Changed | Three names: `docs/contract/` (op contract, a document), `src/model/` (model contract, the C interface), `src/models/` (the families), what each is and when it changes; rule 6 says "nowhere else" |
| `docs/PLAN.md`, `src/backend/README.md` | Changed | The Source layout decision and section name the rule |
| `src/model/README.md`, `docs/contract/README.md` | Changed | Pointers to Three names |
| `src/train/README.md`, `src/quant/README.md` | Changed | Optimizer steps and the GPU parts of quantization algorithms are ops |

---

### Step 9: fast decode and bench

Branch `step9-fast-decode`, from `main` after step 8, opened with the source layout change (above). The gate of `docs/PLAN.md`: decode tokens/s at least llama.cpp's at batch 1, same model, precision and GPU. Every change below was measured with `athanor bench` on the RTX 5080 and kept only if it paid; the report is `docs/results/2026-11-25-step9-decode/`.

#### 9.1 `athanor bench` and the baselines

| File | Kind | What it is for |
| --- | --- | --- |
| `tools/bench.c` | New | ppN and tgN@dD as llama-bench names them; warm-up and measured runs from a reset state, median with min and max; the JSON report of `docs/BENCHMARKS.md` (hardware from the device and sysfs, software with the commit, model files with SHA-256 and Hugging Face revision, workload, method, every run, memory from the driver and the kernel); `--direct` for step 8's decode |
| `src/core/sha256.h`, `sha256.c` | New | SHA-256 (FIPS 180-4), streaming and of a file through its mapping |
| `tests/test_core.c` | Changed | The NIST examples whole and in pieces, a million 'a', a file |
| `src/backend/backend.h`, `cuda/gpu.cu`, `cuda/compat.h` | Changed | `ath_device.pci_bus_id`, `blas_version` |
| `Makefile` | Changed | `build/gen/version.h` with the commit, written only when it changes (no `rm` in the rule) |
| `tools/athanor.c`, `commands.h` | Changed | The `bench` entry |

Found while measuring: a tg test timed the end of its prompt too (a pass without logits returns once queued); the clock now starts after the prompt, and step 8's numbers at 4k and 32k were measured again with that fix. llama.cpp (commit `5220a2d`) was built for `sm_120` with CUDA 13.5 outside the repository (static, the toolkit's path given to the linker), the checkpoint converted to GGUF bf16 with its `convert_hf_to_gguf.py` in a separate environment. Nsight Systems 2026.5 from NVIDIA's redist archives (the system's 2023.4 could not import traces). Commit `57f95dd`.

#### 9.2 GEMV

| File | Kind | What it is for |
| --- | --- | --- |
| `src/ops/cuda/matmul.cu` | Changed | The GEMV kernel: up to 8 rows of x, bf16 weights, a warp per output column (2 or 4 by K), 16-byte pieces loaded past L1, fp32 sums in an order fixed by K; a set of up to 4 weights per launch; `gemv_one` for one row; `gemv_swiglu`; the add epilogue |
| `src/ops/ops.h`, `matmul.c`, `dispatch.c`, `launch.h`, `gpu_none.c` | Changed | `ath_matmul_n` (and `_gpu`), `ath_matmul_add_gpu`, `ath_matmul_swiglu_gpu`, their `fits` queries and launchers |
| `src/ops/cuda/kernel.cuh` | Changed | `k_unpack_bf16x8`, `k_ld8` shared by the GEMV and the norms |

The cuBLASLt plans of other shapes are still built per call: with the GEMV kernel decode does not call cuBLASLt, so their cache moves to step 10 with prefill (log). Commits `e56d2f9`, `d4c80ed`, `758f3a3`, `7f0f3a6`.

#### 9.3 Positions on the device

| File | Kind | What it is for |
| --- | --- | --- |
| `src/ops/attn.c`, `ops.h`, `dispatch.c` | Changed | `ath_attn` takes the queries' positions (NULL: the last T of S) and, on the GPU, the device scratch arena |
| `src/ops/scatter.c`, `src/ops/cuda/scatter.cu` | New | `ath_scatter_rows`: rows written at given rows, byte for byte |
| `src/model/model.h`, `state.c` | Changed | `ath_batch.pos` and `flags`, `ATH_BATCH_DEVICE`; checks |
| `src/models/qwen35/forward.c` | Changed | The new keys and values into the cache by position, attention over the whole cache, positions from a device batch, no copy or wait for a device batch |
| callers of `ath_batch` | Changed | Designated initializers |

The CPU forward pass gives the same bits (test_forward: KL 3.018e-4, ratio 0.60). Commits `e56d2f9`, `f22046b`, `4421aae`.

#### 9.4 Split-K attention

`src/ops/cuda/attn.cu`, rewritten: pieces of 64 keys, partial softmaxes combined in order; the rows kernel and the split-K pair (split, merge) share the device functions and their rounding intrinsics; the fast path for heads up to 256 (8 keys per warp loaded at once, the tile's softmax one head per warp). Commits `e56d2f9`, `d4c80ed`, `758f3a3`.

#### 9.5 GPU sampling

`src/ops/cuda/sample.cu` (new) and `ath_sample_gpu` in `src/ops/sample.c`: select (each block's k best in the reference's order) and merge (one block, then the reference's double arithmetic on thread 0). Commit `7d680de`.

#### 9.6 Graph capture and the decoder

| File | Kind | What it is for |
| --- | --- | --- |
| `src/backend/backend.h`, `cuda/gpu.cu`, `gpu_none.c`, `cuda/compat.h` | Changed | `ath_graph_begin`, `end`, `launch`, `destroy` |
| `src/infer/engine.h`, `engine.c` | Changed | `ath_decoder`: the first step direct, the second recorded, the rest replayed; pinned inputs and the 4-byte result; host sampling when the sampler does not fit the GPU's |

Commit `7d680de`.

#### 9.7 `athanor chat` and `athanor bench` on the decoder

`tools/chat.c` decodes through the decoder on a GPU (the first token from the prompt's logits); `tools/bench.c` tg tests too. Commits `7d680de`, `7f0f3a6`.

#### Other kernels of the decode step

`src/ops/cuda/norm.cu` (a warp per row up to 1024 elements, 16-byte pieces), `src/ops/cuda/conv.cu` (one kernel for one token), `src/ops/cuda/gdn.cu` (a lane per column of the state, rows over the warps), and in `src/models/qwen35/forward.c` the one-token forms (views of q, k, v and one L2 norm; q and k rotated in one call; gate, up and SwiGLU fused). Commits `d4c80ed`, `758f3a3`, `7f0f3a6`.

#### 9.8 Tests

`tests/test_gpu_ops.c` (`gpu_gemv`, `gpu_matmul_n`, `gpu_matmul_swiglu`, `gpu_scatter_rows`, `gpu_sample`, positions and split-K in `gpu_attn`), `tests/test_ops.c` (positions in `attn`, `scatter_rows`), `tests/test_golden.c` (decode records split-K), `tests/test_engine.c` (`engine_decoder`; the pieces tolerance on the tiny model). Results in `tests/README.md` and `STATUS.md`.

#### 9.9 Report and documentation

The report, `docs/results/2026-11-25-step9-decode/`; READMEs of `src/ops`, `src/backend`, `src/infer`, `src/model`, `src/models/qwen35`, `src/core`, `tools`, `tests`; `docs/ARCHITECTURE.md` (the device batch, positions); `docs/program/PROGRAM.md` rows 6 and 8; `STATUS.md` and this file.

### Step 10: fast prefill

Branch `step10-fast-prefill`, from `step9-fast-decode`. The gate of `docs/PLAN.md`: prefill tokens/s within 15% of vLLM's for a single request. Every change below was measured with `athanor bench` on the RTX 5080 and profiled with Nsight Systems; the report is `docs/results/2026-11-25-step10-prefill/`.

#### 10.1 The vLLM baseline

No file of the repository: vLLM 0.34.0 (torch 2.14.0 for CUDA 13.0) in a virtual environment in `~/opt/vllm-venv`, outside athanor (invariant 3 is about the engine, not the tools it is measured against), with the FlashInfer kernels it asks for (`vllm download-kernels`) and `CUDA_HOME` set to the CUDA 13.5 toolkit, as its own warnings recommend. Single requests with `vllm bench latency` (one output token, prefix caching off, the default for that tool) and the forward pass alone with `vllm bench forward`.

#### 10.2 Tiled attention on the tensor cores

| File | Kind | What it is for |
| --- | --- | --- |
| `src/ops/cuda/mma.cuh` | New | NVIDIA's warp-level matrix instructions in PTX: `mma.sync` (bf16, fp32 sums), `ldmatrix` (plain and transposed), `cp.async` with zero fill, bf16 packing (`cvt.rn.bf16x2`); the fragment layouts in the header; `ATH_HAVE_MMA` unset under HIP |
| `src/ops/cuda/attn_mma.cu` | New | FlashAttention's forward pass: 8 warps and 128 queries of one head per block, tiles of 32 keys at absolute positions, the online softmax in base 2, bf16 probabilities whose rounded values the sum adds up, swizzled shared memory, the next tile loading during the current one; rescaling skipped when no row's maximum moved; one reciprocal per output row; `ath_attn_mma_fits` |
| `src/ops/attn.c`, `ops.h`, `launch.h`, `gpu_none.c` | Changed | `ath_attn_gpu` takes the tensor cores from `ATH_ATTN_MMA_MIN_T` (16) queries where the kernel fits; the declarations and stubs |

#### 10.3 The chunked Gated DeltaNet

| File | Kind | What it is for |
| --- | --- | --- |
| `src/ops/cuda/gdn_chunk.cu` | New | The chunked form (`docs/program/gdn-gpu-kernel.md`): `prep` (normalize q and k, decays, A and M, T = (I + A)⁻¹ in blocks of 16, W, U, Q'', O_U, as 8 × 8 tiles per lane) and `scan` (the chunks in order with the state, three products per chunk), run as one launch per round (`gdn_round`: a round's scan and the next round's prep); rounds sized from the device's multiprocessors, two rounds' scratch |
| `src/ops/gdn.c`, `ops.h`, `dispatch.c` | Changed | `ath_gdn` takes an L2 epsilon and normalizes q and k itself (the reference with `ath_l2norm_ref`'s expression: the CPU's bits as before), q, k, v as rows any stride apart, and a scratch arena; the chunked form from `ATH_GDN_CHUNK_MIN_T` (64) tokens |
| `src/ops/cuda/gdn.cu` | Changed | The recurrence normalizes q and k per token (sums of squares over the warps, in warp order) and reads strided rows |
| `src/ops/check.h` | Changed | `ops_float_rows`: rows any stride apart, each contiguous |
| `src/ops/cuda/kernel.cuh` | Changed | `k_ld4` |

#### 10.4 The rest of a prompt

| File | Kind | What it is for |
| --- | --- | --- |
| `src/ops/cuda/conv.cu` | Changed | A prompt in tiles of 8 tokens per thread, every input loaded before the sums, no index divided: the same bits |
| `src/ops/cuda/norm.cu`, `norm.c`, `ops.h`, `launch.h`, `gpu_none.c` | Changed | Rows of up to 16 pieces take 16 or 8 lanes (two or four rows per warp), the same bits; `ath_add_rmsnorm_gpu`, the residual add in the norm's launch with the bits of the two ops |
| `src/models/qwen35/forward.c` | Changed | q, k, v of the Gated DeltaNet as views of the convolution's output, normalized inside `ath_gdn`; a prompt's output projection waits for the next norm (`pend`), not when a hook watches the residual stream |
| `src/ops/cuda/matmul.cu` | Unchanged | Adding a prompt's projection inside cuBLASLt (beta = 1) was written, measured and removed (log) |

#### 10.5 Tests

`tests/test_gpu_ops.c` (`gpu_attn_mma`: the error bound of bf16 probabilities on five shapes, pieces bit for bit, NaN rows; `gpu_gdn_chunk`: five shapes in f32 and bf16, pieces of whole chunks bit for bit; `gpu_add_rmsnorm`: bit for bit against the two ops; q and k of the Gated DeltaNet tests given not normalized; decode against the prompt's last row now with a prompt short enough to take the other attention kernels), `tests/test_golden.c` (the Gated DeltaNet record with q and k as given, through the chunked form on the GPU; the attention tolerance on the tensor cores), `tests/test_ops.c` (the new `ath_gdn_ref` argument: 0 for the hand-computed cases, whose unit vectors stay exact). `tests/README.md`, Prompt kernels.

#### 10.6 Report and documentation

The report, `docs/results/2026-11-25-step10-prefill/`; READMEs of `src/ops` (Prompt kernels), `src/models/qwen35`, `tests`; `docs/program/gdn-gpu-kernel.md` (the chunked form), `docs/program/PROGRAM.md` (rows 10 and 10b), `docs/PLAN.md` (Dense GEMM decided), `README.md`, this file and `STATUS.md`.

### Step 12: MoE and tiering

Branch `step12-moe-tiering`, from `main` (steps 1–10; step 11 is postponed). The gate of `docs/PLAN.md`: Qwen3.6-35B-A3B decode at least Strata's on the same machine and quantization. Decisions at the start (`STATUS.md`, step 12): Q8_0 weights read from Unsloth's GGUF, the file Strata uses; invariant 7 as option (b), the CPU and GPU expert paths give the same bits.

#### 12.1 The Strata baseline

(in progress)

#### 12.2 The GGUF reader and the checkpoint abstraction

| File | Kind | What it is for |
| --- | --- | --- |
| `src/core/gguf.h`, `gguf.c` | New | The GGUF reader: header (versions 2 and 3), metadata (every value type; keys and strings copied with a NUL, arrays left in the mapping with their type and count), the tensor table (shapes reversed from `ne`, types F32, F16, BF16, Q8_0), zero-copy views; validation: bounded reads, unique keys and names, a power-of-two `general.alignment`, aligned offsets, no overlap, padding-only gaps, nothing past the data |
| `src/core/ckpt.h`, `ckpt.c` | New | `ath_ckpt`: safetensors (index and shards, or one file) or the one `.gguf` of a directory (`mmproj-*.gguf` passed over, safetensors first) as one name-sorted table of `ath_ckpt_tensor` (name, view, bytes, shard, file offset) |
| `src/core/tensor.h`, `tensor.c` | Changed | `ATH_DTYPE_Q8_0` (64, the first block format: 32 elements, an f16 scale and 32 int8, 34 bytes); `ath_dtype_block`, `ath_dtype_bytes`, `ath_dtype_is_quant`; `ath_tensor_init` requires rows of whole blocks, `ath_tensor_slice` block boundaries, `ath_tensor_nbytes` counts blocks |
| `src/core/base.h`, `base.c`, `safetensors.c` | Changed | `ath_sort`, the heapsort moved out of `safetensors.c` now that the GGUF reader uses it too |
| `src/model/model.h`, `registry.c` | Changed | `m->ckpt` is an `ath_ckpt`; `ath_model_open` opens it, the coverage check walks its table |
| `src/models/qwen35/weights.c` | Changed | Binds from the `ath_ckpt` table (the GGUF names arrive with 12.4) |
| `tools/inspect.c` | Changed | Lists a GGUF checkpoint too: the file line (version, keys, architecture, data section), its tensors with their dtypes (`q8_0`), totals per dtype including block formats, the padding check |
| `tests/test_core.c` | Changed | `q8_0_blocks` (sizes, views and slices on block boundaries); `gguf_reads` (a file built in memory: metadata of five types, three tensors, the checkpoint directory with a vision projector beside it, two files refused, safetensors first); `gguf_rejects_bad_files` (16 malformed variants); `gguf_real_model` (the Qwen3.6-35B-A3B GGUF when present: 753 tensors in natural order) |
| `src/core/README.md`, `docs/ARCHITECTURE.md` | Changed | The two new files, Q8_0, loading through `ath_ckpt`; the weight-source abstraction arrived with its second user |

---

## Files by step

Every file of the repository and the steps that created or changed it (documentation-only edits included).

| File | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 | Step 6 | Step 7 | Step 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Makefile` | 1.1, 1.8 | 2.3, 2.6 | 3.3, 3.4, 3.5, 3.6 | 4.5 | 5.1, 5.3, 5.8 | 6.1 | 7.1 | 8.5 |
| `src/core/base.{h,c}` | 1.2 | | | | | 6.2 |  |  |
| `src/core/arena.{h,c}` | 1.3 | | | | | 6.2 (`arena.h`) |  |  |
| `src/core/file.{h,c}` | 1.4, 1.7 | | | | | |  |  |
| `src/core/json.{h,c}` | 1.5 | | | | 5.4 | |  |  |
| `src/core/tensor.{h,c}` | 1.6 | | 3.2 | | | | 7.2 (`tensor.h`) |  |
| `src/core/safetensors.{h,c}` | 1.7 | | | | | |  |  |
| `src/core/threads.{h,c}` | | | | 4.7 | | | 7.2 (`threads.h`) |  |
| `src/model/model.h` | | 2.2 | | 4.2, 4.7, 4.8 | 5.4 | |  | 8.3 |
| `src/model/registry.c` | | 2.3 | | 4.8 | | |  |  |
| `src/model/state.c` | | | | 4.2, 4.7 | 5.4 | |  | 8.3 |
| `src/models/qwen35/qwen35.h`, `config.c` | | 2.4 | | 4.3, 4.7, 4.8 (`qwen35.h`) | 5.5 (`qwen35.h`) | |  | 8.4 (`qwen35.h`) |
| `src/models/qwen35/weights.c`, `family.c` | | 2.5 | | 4.3, 4.8 (`weights.c`) | 5.5 (`family.c`) | |  |  |
| `src/models/qwen35/forward.c` | | | | 4.3, 4.7 | | |  | 8.4 |
| `src/ops/ops.h`, `check.h`, `embed.c`, `matmul.c`, `norm.c`, `rope.c`, `act.c`, `attn.c`, `conv.c`, `gdn.c` | | | 3.3 | 4.7 (`ops.h`, `matmul.c`) | 5.6 (`ops.h`) | | 7.2 | 8.2 (`ops.h`) |
| `tools/athanor.c`, `commands.h` | 1.8 | | | | 5.7 | 6.6 |  |  |
| `tools/inspect.c` | 1.8 | 2.6 | | 4.8 | | |  |  |
| `tools/dev/alloc_trace.c` | 1.8 | | | | | |  |  |
| `tools/dev/check_alloc.sh` | 1.8 | 2.6 | | | | |  |  |
| `tests/test.h` | 1.9 | | | | | |  |  |
| `tests/test_core.c` | 1.9 | 2.7 | | 4.7 | | |  |  |
| `tests/util.h`, `tests/test_model.c` | | 2.7 | | 4.5, 4.8 (`test_model.c`) | | |  |  |
| `tests/test_ops.c` | | | 3.5 | 4.7 | | |  |  |
| `tests/test_golden.c` | | | 3.6 | 4.7 | | | 7.5 |  |
| `tests/tiny.h`, `tests/test_forward.c` | | | | 4.5, 4.7 (`test_forward.c`) | | |  | 8.8 (`test_forward.c`) |
| `tests/golden/dump.py`, `README.md` | | | 3.4 | 4.4, 4.8 | 5.9 (`README.md`) | |  | 8.9 (`README.md`) |
| `tests/golden/requirements.in`, `requirements.txt` | | | 3.1 | | | |  |  |
| `src/core/buf.{h,c}`, `src/core/rng.{h,c}` | | | | | 5.4 | |  |  |
| `src/text/unicode.{h,c}`, `unicode_tables.h` | | | | | 5.2 | |  |  |
| `src/text/bpe.{h,c}` | | | | | 5.3 | |  |  |
| `src/model/text.c` | | | | | 5.4 | |  |  |
| `src/models/qwen35/text.c` | | | | | 5.5 | |  |  |
| `src/ops/sample.c` | | | | | 5.6 | | 7.2 |  |
| `tools/chat.c` | | | | | 5.7 | |  | 8.6 |
| `tools/dev/gen_unicode.py` | | | | | 5.2 | |  |  |
| `tests/test_text.c` | | | | | 5.8 | |  | 8.8 |
| `tests/golden/text.py` | | | | | 5.1 | |  |  |
| `src/backend/backend.h`, `gpu.cu`, `gpu_none.c`, `compat.h`, `devarena.c` | | | | | | 6.2 to 6.5 | 7.1 (`backend.h`, `gpu.cu`, `compat.h`) | 8.1 (`backend.h`, `gpu.cu`, `gpu_none.c`, `compat.h`) |
| `tools/devices.c` | | | | | | 6.6 |  |  |
| `tools/dev/check_nvcc.sh` | | | | | | 6.1 | 7.1 |  |
| `tests/test_backend.c` | | | | | | 6.7 |  | 8.1 |
| `src/backend/gpu.h` | | | | | | | 7.1 |  |
| `src/ops/launch.h`, `gpu_none.c` | | | | | | | 7.2 |  |
| `src/ops/kernel.cuh`, `embed.cu`, `matmul.cu`, `norm.cu`, `rope.cu`, `act.cu`, `attn.cu`, `conv.cu`, `gdn.cu` | | | | | | | 7.3 |  |
| `tests/test_gpu_ops.c` | | | | | | | 7.4 |  |
| `src/ops/dispatch.c` | | | | | | | | 8.2 |
| `src/infer/engine.h`, `engine.c` | | | | | | | | 8.5 |
| `tests/golden/generate.py` | | | | | | | | 8.7 |
| `tests/test_engine.c` | | | | | | | | 8.8 |
| `docs/proposals/chat-state-at-last-query.md`, `chat-shadow-sequence.md` | | | | | | 6.8 |  |  |
| `CLAUDE.md` | | | | | | 6.8 |  |  |
| `docs/contract/qwen35-0.8b.md` | closing | 2.1 | 3.1, 3.7 | 4.6, 4.8 | 5.9 | |  |  |
| `docs/contract/qwen35-35b-a3b.md` | | 2.1 | | 4.8 | | |  |  |
| `docs/MODELS.md` | | 2.1 | 3.7 | 4.6, 4.8 | | |  |  |
| `docs/ARCHITECTURE.md` | closing | 2.2 | 3.7 | 4.6, 4.7, 4.8 | 5.9 | 6.8 | 7.6 | 8.9 |
| `docs/PLAN.md` | | | | 4.8 | | |  |  |
| `docs/STATUS.md` | closing | 2.1 to 2.7, closing | 3.1, closing | 4.1, closing, 4.7, 4.8 | 5.1 to 5.9, closing | 6.1 to 6.8 | 7.1 to 7.6 | 8.1 to 8.9 |
| `docs/program/PROGRAM.md` | closing | closing | 3.7 | 4.6, 4.7, 4.8 | 5.9 | 6.8 | 7.6 | 8.9 |
| `docs/proposals/cpu-matmul-avx512.md` | | | | 4.7 | | |  |  |
| `docs/CHANGE-001-scope-tools-cache.md` | | | | 4.8 | | |  |  |
| `docs/status_extended.md` | | | closing | 4.6, 4.7, 4.8 | 5.9 | 6.8 | 7.6 | 8.9 |
| `AGENTS.md` | | 2.3 | 3.7 | 4.8 | | |  |  |
| `README.md` | closing | | | 4.8 | 5.9 | |  | 8.9 |
| module READMEs | closing (`src/core`, `tests`, `tools`) | 2.2, 2.5, 2.6, 2.7 (`src/model`, `src/models/qwen35`, `tools`, `tests`) | 3.2, 3.3, 3.6, 3.7 (`src/core`, `src/ops`, `tests`, `tests/golden`, `src/backend`) | 4.6 (`src/model`, `src/models/qwen35`, `tests`, `tests/golden`); 4.7 (`src/core`, `src/ops`, `src/backend`, `src/model`, `src/models/qwen35`, `tests`); 4.8 (`src/model`, `src/models/qwen35`, `src/net`, `src/infer`, `src/text`, `tools`) | 5.9 (`src/core`, `src/text`, `src/ops`, `src/model`, `src/models/qwen35`, `tools`, `tests`) | 6.8 (`src/backend`, `src/core`, `tools`, `tests`) | 7.6 (`src/ops`, `src/backend`, `tests`, `src/core`) | 8.9 (`src/model`, `src/models/qwen35`, `src/infer`, `src/ops`, `src/backend`, `tools`, `tests`, `tests/golden`) |

Before step 9 (source layout), `src/backend/{gpu.cu,gpu.h,compat.h}` moved to `src/backend/cuda/` and `src/ops/{kernel.cuh,*.cu}` to `src/ops/cuda/`. The rows above keep the paths the steps used.

Steps 9 and 10 list their files in their own sections above. New in step 10: `src/ops/cuda/mma.cuh`, `attn_mma.cu`, `gdn_chunk.cu`.
