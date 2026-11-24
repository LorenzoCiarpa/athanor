# Status

Updated by whoever completes a task (human or agent). Never delete lines: mark them done, superseded or dropped.

## Current step

**Step 3: CPU reference ops and golden harness** (see `docs/PLAN.md` §4). Started 2026-11-24 on branch `step3-ops-golden`. **Done 2026-11-24** (pushed, to be merged into `main`); step 4 is next.

Goal: every op of the Qwen3.5-0.8B forward pass as a scalar CPU reference in `src/ops/`, and evidence on the real weights that each one computes what the pinned reference implementation computes.

### Tasks

- [x] **3.1 Golden environment, f32 question.** Pin the Python environment of the golden harness (`tests/golden/requirements.in` and the uv lock `requirements.txt`: transformers 5.24.0, torch 2.15.0 CPU build, Python 3.14) and settle the open question on the f32 checkpoint tensors.
- [x] **3.2 `src/core/tensor.h`.** bf16 conversion (round to nearest, ties to even, NaN as torch does), element load and store for f32 and bf16, `ath_tensor_same_shape`.
- [x] **3.3 `src/ops/`.** `ops.h` and a `_ref` function for every op of the contract: `embed`, `matmul`, `rmsnorm` (plain and zero-centered), `rmsnorm_gated`, `l2norm`, `rope` (partial, rotate-half), `swiglu`, `sigmoid_mul`, `add`, `attn` (causal, grouped-query), `conv` (causal depthwise with state), `gdn_gates`, `gdn` (gated delta rule, recurrent).
- [x] **3.4 `tests/golden/dump.py`.** Run the reference (CPU, bf16, eager attention, torch fallbacks) over a fixed prompt longer than one Gated DeltaNet chunk and one decode step; record every op of every layer with its inputs, its output and the same function rerun in fp32; check versions and hashes; write a manifest and a safetensors file. `make golden` runs it.
- [x] **3.5 `tests/test_ops.c`.** Hand-computed cases and exact properties: streaming equals all at once (conv, Gated DeltaNet), decode equals the last prefill row (attention), causality, head grouping, argument checks.
- [x] **3.6 `tests/test_golden.c`.** Replay every record through the `_ref` functions on the real weights, native and exact, with a tolerance per op and output justified in `tests/README.md`.
- [x] **3.7 Documentation.** Contract, READMEs of `src/ops`, `src/core`, `src/backend`, `tests`, `tests/golden`, `docs/ARCHITECTURE.md` tree, AGENTS op checklist, `docs/program/PROGRAM.md`.

### Acceptance criteria

1. Every op of the Qwen3.5-0.8B contract has a `_ref` function in `src/ops/`, documented in `ops.h` (shapes, dtypes, math).
2. `make golden` regenerates the golden data from the pinned reference and refuses any other version.
3. `test_golden`: every op the reference computes on Qwen3.5-0.8B (all 24 layers, prefill and decode) is within tolerance of the reference's own output and of its fp32 rerun, on the real weights.
4. `make test` green with zero warnings, also with `DEBUG=1`.
5. No model names in `src/` outside `src/models/`.
6. Contract, module READMEs, `docs/ARCHITECTURE.md` and this file updated.

Result (2026-11-24, Qwen3.5-0.8B revision `2fc0636`, transformers 5.24.0, torch 2.15.0+cpu, gcc 15.2, glibc 2.43):

| # | Criterion | Result |
| --- | --- | --- |
| 1 | a `_ref` per op | 13 functions in 8 files (`embed`, `matmul`, `norm`, `rope`, `act`, `attn`, `conv`, `gdn`), none allocating; the decode state of `conv` and `gdn` is the caller's tensor |
| 2 | golden data from the pinned reference | `dump.py` checks transformers, torch and the sha256 of `modeling_qwen3_5.py`, and that no fused-kernel package is installed; 75-token prompt (two chunks of the chunked Gated DeltaNet) plus one decode step; 906 records, 3,244 tensors, 434 MiB, about 4 s, identical across runs |
| 3 | every op within tolerance | 906 of 906 records replayed. Exact (fp32) worst errors from 1.1e-7 to 9.7e-6, `add` bit-exact; the Gated DeltaNet state 1.0e-5 after 75 tokens. Native: within 1 bf16 ulp where the reference rounds once, 2 where it rounds two or three times; `embed`, `add` and the conv state bit-exact; `attn` 9.8e-2, all of it from the bf16 rounding of `q·kᵀ` in the eager reference (`dump.py` checks that fp32 scores bring it within 2 ulps). Full table in `tests/README.md` |
| 4 | `make test`, zero warnings, also `DEBUG=1` | 22 + 11 + 16 + 1 tests pass in release (about 5 s with `check-alloc`) and with ASan/UBSan (`test_golden` about 2.5 minutes there); no warnings |
| 5 | no model names outside `src/models/` | `grep -rni qwen src` outside `src/models/` finds nothing |
| 6 | documentation | contract (attention scores, checkpoint dtypes, reference now run by the golden harness), `docs/MODELS.md` status, READMEs of `src/ops`, `src/core`, `src/backend`, `tests`, `tests/golden`, `docs/ARCHITECTURE.md` tree, AGENTS op checklist, `docs/program/PROGRAM.md` rows 3 and 4 |

`ldd build/athanor` now lists `libm` too (the ops call `expf`, `sqrtf`, `powf`), which the step 1 criterion allows.

## Completed steps

### Step 2

**Step 2: contract and family loading** (see `docs/PLAN.md` §4). Started 2026-11-24 on branch `step2-contract`. **Done 2026-11-24** (pushed, to be merged into `main`); step 3 is next. Merged into `main` (fast-forward) on 2026-11-24.

Goal: give every tensor of the checkpoint a meaning that pipelines can use without knowing the family: a role, a layer, an expert index and flags, plus the layer schedule. No model math yet.

#### Tasks

- [x] **2.1 Reference and op contract.** Pin the reference implementation (transformers 5.24.0, file hashes) and write the op contract of Qwen3.5-0.8B in `docs/contract/qwen35-0.8b.md`: dtype boundaries, norms, RoPE, output gate, Gated DeltaNet, state layout, every **verify** item of `docs/MODELS.md` that applies. Record the MoE naming and router contract from the real Qwen3.5-35B-A3B headers in `docs/contract/qwen35-35b-a3b.md`.
- [x] **2.2 `src/model/model.h`.** Roles, param flags, `ath_param`, the layer schedule (`ath_layer`), ignored groups, `ath_model`, `ath_family` (`name`, `match`, `load`); `docs/ARCHITECTURE.md` updated to the real header.
- [x] **2.3 `src/model/registry.c`.** Families compiled in from the Makefile `FAMILIES` list (no family names in the file); `ath_model_open(dir)` reads `config.json`, picks the family whose `match` accepts it, opens the checkpoint, calls `load`, then checks that every checkpoint tensor is bound to a param or belongs to a declared ignored group, and that every param views memory of the tensor it names.
- [x] **2.4 `src/models/qwen35/config.c`.** `config.json` → dimensions, layer schedule (`layer_types`, or `full_attention_interval`), MoE parameters; reject values the reference does not support (bias, non-default RoPE, `mlp_only_layers`, activations other than SiLU).
- [x] **2.5 `src/models/qwen35/weights.c`.** Bind every language-model tensor with role, layer, expert and flags; check dtype and shape against the config; fused expert tensors become one view per expert; vision encoder and MTP head are declared ignored groups.
- [x] **2.6 `athanor inspect --roles`.** Family, dimensions, schedule, per-layer totals, one row per param, totals per role, ignored groups, coverage check; `--json` and `--filter` work with it.
- [x] **2.7 `tests/test_model.c`.** Synthetic dense, MoE and text-only checkpoints; every failure mode (missing, unexpected, misshaped, wrong dtype, tied with `lm_head`, unsupported config); the real checkpoint.

#### Acceptance criteria

1. `athanor inspect models/Qwen3.5-0.8B --roles` binds all 320 language-model tensors (role, layer, flags), lists the 168 vision and MTP tensors as ignored, prints the schedule (18 Gated DeltaNet, 6 full attention, matching `layer_types`) and exits 0.
2. A missing, unexpected or misshaped tensor makes loading fail with a message naming it (tests).
3. `make test` green with zero warnings, also with `DEBUG=1`.
4. `make check-alloc` also covers `inspect --roles`: no heap allocation outside arenas.
5. No model names outside `src/models/` (the Makefile `FAMILIES` list excepted).
6. Contract files, module READMEs, `docs/ARCHITECTURE.md` and this file updated.

Result (2026-11-24, Qwen3.5-0.8B revision `2fc0636`, transformers 5.24.0 as reference, gcc 15.2, glibc 2.43):

| # | Criterion | Result |
| --- | --- | --- |
| 1 | `inspect --roles` on Qwen3.5-0.8B | 320 tensors bound to 320 params (1,504,791,232 bytes, equal to the language-model total recorded in step 1), 168 ignored (153 `model.visual.`, 15 `mtp.`), 0 unbound; schedule 18 gated_deltanet + 6 attention at layers 3, 7, 11, 15, 19, 23, matching `layer_types`; 49 residual writers (embedding + 24 mixer outputs + 24 MLP downs); exit 0 |
| 2 | failures name the tensor | `test_model`: missing, unexpected, misshaped, integer dtype, tied with `lm_head`, untied without it, plus 22 rejected configs; each message names the tensor or key |
| 3 | `make test`, zero warnings, also `DEBUG=1` | 22 + 11 tests pass in release and with ASan/UBSan; no warnings |
| 4 | `check-alloc` covers `--roles` | one allocation (the arena's first block) for `inspect` and for `inspect --roles`; the script now also fails if the traced command fails |
| 5 | no model names outside `src/models/` | `grep -rni qwen src tools` outside `src/models/` finds nothing; the registry is generated from the Makefile `FAMILIES` list |
| 6 | documentation | contracts, `docs/MODELS.md` status line, `docs/ARCHITECTURE.md` (contract as of step 2, coverage rule, tree), READMEs of `src/model`, `src/models/qwen35`, `tests`, `tools`, `AGENTS.md` family checklist, `docs/program/PROGRAM.md` row 2 |

Extra check, MoE: a sparse copy of Qwen3.5-35B-A3B (the 14 real shard headers, zero data; 448 KB on disk) loads with `inspect --roles` in release and ASan builds: 1,811 tensors, 693 bound to 21,093 params (two fused expert tensors per layer give 256 views each), 1,118 ignored (333 vision, 785 MTP), 0 unbound; 30 gated_deltanet + 10 attention, 40 moe.

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

**Step 4: CPU forward.** `src/models/qwen35/forward.c` composes the step 3 ops into the decoder layers and the logits; `forward` and the state specs (KV cache, recurrent and conv state) join `model.h` with this first user; a minimal CPU state for one sequence; `tests/golden/dump.py` extends to per-layer hidden states and logits on fixed prompts. Gate: logits match the reference implementation on fixed prompts (metric to settle first: see Open questions).

Superseded preview (written in step 2): **Step 3: CPU reference ops and golden harness.** `src/ops/*.c` references (scalar, fp32 accumulation) for every op in the contract, `src/backend/cpu.c`, `tests/golden/dump.py` against the pinned reference, `tests/test_ops.c`. Gate: every op within tolerance of golden tensors on real weights.

Superseded preview (written in step 1): **Step 2: contract and family loading.** Write `src/model/model.h` as in `docs/ARCHITECTURE.md`, `registry.c`, and `src/models/qwen35/{config.c,weights.c}`: parse `config.json` into the layer schedule (which layers are Gated DeltaNet, which are full attention, MLP or MoE), bind every tensor to an `ath_param` with role, layer, expert and flags, and fail loudly on any unbound or missing tensor. `athanor inspect --roles` prints the result.

## Open questions

- (none yet) (superseded 2026-11-23 by the entries below)
- **CUDA toolkit too old for `sm_120` (blocks step 6).** The development machine has nvcc 12.4 (driver 580, CUDA 13.0 runtime). Blackwell consumer GPUs (`sm_120`, RTX 5080) need CUDA 12.8 or newer. Options: install CUDA 12.8+/13.x before step 6 (recommended), or build for `compute_90` PTX and rely on JIT (slower start, no FP4/FP8 `sm_120` paths). Nothing in steps 1–5 depends on it.
- **Hardware assumptions in `docs/PLAN.md` §2 can be confirmed:** CPU is an AMD Ryzen 7 9800X3D (Zen 5; `avx512f`, `avx512bw`, `avx512vl` and `avx512_bf16` present); RAM shows 59 GiB usable, consistent with 64 GB installed.
- **Step 2, weight parameterizations.** The checkpoint suggests zero-centered RMSNorm weights for the decoder-layer norms and `q_norm`/`k_norm`, and plain weights for the Gated DeltaNet output norm (`docs/contract/qwen35-0.8b.md`, observations). To be confirmed against the pinned reference implementation, as the **verify** rule requires. **Resolved 2026-11-24:** confirmed against transformers 5.24.0 (`Qwen3_5RMSNorm` uses `1 + w`, `Qwen3_5RMSNormGated` a plain `w`); see the contract, sections RMSNorm and Gated RMSNorm.
- **Golden data and f32 checkpoint tensors (step 3).** `A_log` and `linear_attn.norm.weight` are f32 in the checkpoint, and the reference has no `_keep_in_fp32_modules`. If the golden dump loads the model in bf16, the reference rounds them to bf16 and its outputs differ from athanor's (which uses the stored f32 values). Options: load the reference in fp32 for op-level goldens, or check what transformers 5.24.0 does with mixed-dtype checkpoints and record it. To settle before writing `tests/golden/dump.py`. **Resolved 2026-11-24:** transformers 5.24.0 rounds both tensors to bf16 on load (`dtype="auto"` and `dtype=torch.bfloat16` alike). `dump.py` loads in bf16 and puts the stored f32 values back, which the forward code handles unchanged, so the reference and athanor use the same weights; on the golden prompt this brings the bf16 reference closer to an fp32 run of the model (max KL 1.4e-3 instead of 2.9e-3). Recorded in the contract (Checkpoint dtypes) and `tests/golden/README.md`.
- **End-to-end reference and metric for step 4.** The reference differs from athanor by design in a few roundings: the eager attention rounds `q·kᵀ` to bf16 (up to 9.8% of an attention output element of Qwen3.5-0.8B, relative to its magnitude plus the tensor RMS, `tests/README.md`), the reference computes the logits in bf16 where `docs/MODELS.md` asks for fp32, and every elementwise op rounds more often in the reference than in athanor. Logits will therefore not match to the bit. Options for the step 4 gate: (a) per-layer hidden states and logits within a tolerance derived like the op tolerances; (b) KL of athanor against an fp32 run of the reference, no larger than the KL of the bf16 reference against the same fp32 run (athanor at least as close to the exact model as transformers in bf16), plus greedy tokens identical where the reference's top-2 margin is above the bf16 noise. Recommended: (b), with (a) as a diagnostic. To settle at the start of step 4.

## Log

- 2026-11-23: repository documentation created; design in `docs/ARCHITECTURE.md`; plan in `docs/PLAN.md`.
- 2026-11-23: step 1 implemented on branch `step1-core-inspect`: Makefile (`cpu`, `test`, `check-alloc`, `clean`, `DEBUG=1`; `cuda`/`hip` stubs), `src/core/{base,arena,file,json,tensor,safetensors}`, `tools/athanor.c` with `inspect`, `tests/test.h` runner and `tests/test_core.c`. All acceptance criteria met (table above).
- 2026-11-23: decisions taken within step 1 (no `PLAN.md` impact): logging threshold from the `ATH_LOG` environment variable, read on every call (no global state); memory-mapped weights are tier `ATH_TIER_RAM`; the safetensors reader rejects gaps and overlaps in the data section like the reference implementation; sorting uses an in-file heapsort because glibc `qsort` may call `malloc`; `inspect` reads config keys at the top level or under `text_config` (Hugging Face convention for multimodal checkpoints) and leaves model-specific interpretation to step 2.
- 2026-11-23: fixed `.gitignore`: `models/` also matched `src/models/`; now `/models/`.
- 2026-11-24: `step1-core-inspect` merged into `main` (fast-forward); step 2 started on `step2-contract`. Reference implementation pinned (transformers 5.24.0); op contract of Qwen3.5-0.8B written; MoE naming, shapes and router contract recorded from the Qwen3.5-35B-A3B headers (no weights downloaded).
- 2026-11-24: step 2 implemented on branch `step2-contract`: `src/model/{model.h,registry.c}`, `src/models/qwen35/{qwen35.h,family.c,config.c,weights.c}`, `athanor inspect --roles`, `tests/test_model.c` and `tests/util.h`. All acceptance criteria met (table above).
- 2026-11-24: decisions taken within step 2 (no `PLAN.md` impact): state specs, hooks, `forward`, `pretokenize`, `chat` and `backward` join `model.h` with their first user (step 4 onward), as AGENTS.md rule 9 asks; `load` takes the open `ath_safetensors` instead of an `ath_weights` abstraction until the `.ath` container gives it a second user (phase 2); the family list is generated from the Makefile `FAMILIES` into `build/gen/families.h`, so `registry.c` names no family (AGENTS.md checklist updated); the registry, not the family, enforces coverage (every tensor bound or in a declared ignored group, never both; every param a contiguous view inside its tensor); tied embeddings give no `lm_head` param and make the embedding `sensitive`; fused MoE expert tensors become one view per expert sharing the checkpoint name; the vision encoder and the MTP head are ignored groups, the same prefixes the reference text model drops; `ath_layer` labels are for display only.
- 2026-11-24: `step2-contract` merged into `main` (fast-forward); step 3 started on `step3-ops-golden`.
- 2026-11-24: step 3 implemented on branch `step3-ops-golden`: `src/ops/{ops.h,check.h}` and `src/ops/{embed,matmul,norm,rope,act,attn,conv,gdn}.c`, bf16 helpers in `src/core/tensor.h`, `tests/golden/{dump.py,requirements.in,requirements.txt,README.md}`, `tests/test_ops.c`, `tests/test_golden.c`, `make golden`. All acceptance criteria met (table above).
- 2026-11-24: decisions taken within step 3 (`PLAN.md` unchanged; the first one departs from its file list for step 3): `src/backend/cpu.c` is not written yet: the reference ops need no runtime (no allocation, one thread, host memory from arenas), so it arrives with its first user (the GPU runtime of step 6 or the thread pool of the first `_fast` path); the device-dispatching entry points of `ops.h` join with the second implementation of an op (step 7); ops take `ath_tensor` descriptors, f32 or bf16, contiguous, and round once to the output dtype, with intermediate roundings only where the contract records them (`rmsnorm_gated`, `rope` tables); attention keeps scores and probabilities in fp32 (the eager reference rounds both); the L2 norm is a separate op composed before the recurrence (the reference fuses it into its Gated DeltaNet function); the conv state is the last K − 1 inputs (the reference keeps K, the oldest never used: `dump.py` checks it); golden data covers every op of all 24 layers in prefill and decode, matmuls on rows 0 and T − 1 only (each output row depends on its input row only), with eager attention, one CPU thread and the f32 tensors restored; it is not committed (434 MiB, regenerated in seconds) and athanor reads it with its own safetensors reader; each comparison is one score, max |got − want| / (|want| + rms(want)), against the reference's own output (native) and its fp32 rerun (exact).
