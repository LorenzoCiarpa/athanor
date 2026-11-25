# Status

Updated by whoever completes a task (human or agent). Never delete lines: mark them done, superseded or dropped.

This file is the overview. [`status_extended.md`](status_extended.md) goes one level down: for every task of every step, the files created or changed, what each is for and what was implemented in it.

## Current step

**Step 7: GPU ops, one by one** (see `docs/PLAN.md` §4). Started 2026-11-25 on branch `step7-gpu-ops`. **Done 2026-11-25** (pushed, to be merged into `main`); step 8 is next.

Goal: a GPU kernel for every op of the forward pass, each checked against its CPU reference on random and on real inputs, as AGENTS.md's op checklist asks. Correct and simple first: fast decode is step 9, fast prefill step 10.

### Tasks

- [x] **7.1 The matmul library in the backend.** A cuBLASLt handle and a 32 MiB workspace per device, created by `ath_device_open` (`ath_device.blas`, `blas_ws`); cuBLASLt linked as a shared library through the toolkit's path, `tools/dev/check_nvcc.sh` checks it is there; hipBLASLt names in `compat.h`. `src/backend/gpu.h`: what GPU code shares (errors, device, streams, the check after a launch).
- [x] **7.2 The GPU entry points.** `ath_<op>_gpu(const ath_device *d, ...)` in `src/ops/ops.h` for every op of the forward pass, asynchronous on the compute stream. The argument checks are the reference's (one `args` function per op; `ops_place` says on which side the tensors must be); the launchers (`src/ops/launch.h`) are in the `.cu` files, stubbed by `src/ops/gpu_none.c` without a GPU. `make cuda` builds `src/ops/*.cu`.
- [x] **7.3 The kernels.** `embed`, `matmul` (cuBLASLt; athanor's own kernel for the dtype combinations cuBLASLt lacks), `rmsnorm`, `rmsnorm_gated`, `l2norm` (one kernel), `rope` (frequencies from the host), `swiglu`, `sigmoid_mul`, `add` (one kernel), `attn` (online softmax per query), `conv` (outputs, then the state), `gdn_gates`, `gdn` (a warp per column of the state); shared device code in `src/ops/kernel.cuh`.
- [x] **7.4 `tests/test_gpu_ops.c`.** Every kernel against its `_ref` on random inputs: three dtype configurations, odd shapes, a misaligned matmul operand, both matmul paths, pieces equal all at once bit for bit (conv, gdn, attention), an embedding id outside the table, argument checks and the kernels' limits.
- [x] **7.5 GPU replay of the golden records.** `tests/test_golden.c` runs every record through the GPU kernel too, when the build has a GPU, and compares it with `_ref` on the same inputs, native and exact.
- [x] **7.6 Documentation.**

### Acceptance criteria

1. Every op of the forward pass has a GPU kernel: `embed`, `matmul`, `rmsnorm`, `rmsnorm_gated`, `l2norm`, `rope`, `swiglu`, `sigmoid_mul`, `add`, `attn`, `conv`, `gdn_gates`, `gdn`.
2. Every kernel matches its CPU reference within the tolerances of `tests/README.md` (GPU kernels), on random inputs and on the real inputs of every golden record; copies, the residual add and the conv state bit for bit.
3. On the GPU, the convolution and the recurrence fed in pieces, and attention for the last token alone, give the bits of the same tokens at once.
4. `make cuda test` green, also with `DEBUG=1`; `make test` green (the GPU tests skipped, the argument checks run); compute-sanitizer (memcheck with leak check, racecheck, initcheck) clean on `test_gpu_ops`.
5. No warnings; `ldd`: the CPU build unchanged, the CUDA build adds `libcublasLt.so.13` only (invariant 3 allows cuBLASLt).
6. Documentation updated.

Result (2026-11-25, RTX 5080 `sm_120`, driver 580.178.04 (CUDA 13.0), CUDA toolkit 13.5.0, cuBLASLt 13.10, gcc 15.2):

| # | Criterion | Result |
| --- | --- | --- |
| 1 | kernels | 13 ops in `src/ops/{embed,matmul,norm,rope,act,attn,conv,gdn}.cu`, 11 kernels plus cuBLASLt |
| 2 | against the references | random inputs (`test_gpu_ops`, 10 tests): worst f32 score 1.9e-6 (matmul), 1.0e-6 (attention), under 1e-6 for the rest; bf16 outputs within one ulp. Real inputs (`test_golden`, 906 records, prefill and decode): worst exact score 4.1e-6 (matmul), 1.1e-5 (attention), 5.9e-6 (gdn out), 1.2e-5 (gdn state after 75 tokens), under 3e-6 for the rest; native within one ulp (two for rope and rmsnorm_gated, measured: two). `embed`, `add` and the conv state: 0 |
| 3 | pieces | conv 4 + 1 + 5 tokens, gdn 5 + 1 + 6, attention decode against the last prefill row: identical bits |
| 4 | tests | `make cuda test`: 24 + 11 + 17 + 1 + 7 + 12 + 6 (1 skipped) + 10 pass; `make cuda DEBUG=1 test` the same; `make test` and `make DEBUG=1 test`: the same suites with the GPU tests skipped (`test_gpu_ops` 1 pass, 9 skipped); compute-sanitizer on `test_gpu_ops`: memcheck 0 errors and 0 bytes leaked, racecheck 0 hazards, initcheck 0 errors |
| 5 | build | no warnings (host and device); `ldd build/athanor`: libm, libc; `ldd build/cuda/athanor`: libm, libc, `libcublasLt.so.13` (with the librt, libpthread and libdl it loads) |
| 6 | documentation | `src/ops/README.md` (GPU kernels), `src/backend/README.md` (cuBLASLt, `gpu.h`, linking), `tests/README.md` (GPU kernels), `src/core/README.md`, `docs/ARCHITECTURE.md`, `docs/program/PROGRAM.md` rows 7, 8 and 10, this file and `status_extended.md` |

## Completed steps

### Step 6

**Step 6: GPU backend** (see `docs/PLAN.md` §4). Started 2026-11-24 on branch `step6-gpu-backend`. **Done 2026-11-24** (pushed, to be merged into `main`); step 7 is next. Merged into `main` (fast-forward) on 2026-11-25, at the maintainer's request.

Goal: the GPU runtime athanor's kernels will run on, behind one C interface: devices, device and pinned host memory, copies, streams, events and timing, on CUDA, written once for CUDA and HIP. No kernel yet (step 7).

#### Tasks

- [x] **6.1 Build.** `make cuda`: device code compiled by nvcc (CUDA 12.8 or newer for `sm_120`, checked by `tools/dev/check_nvcc.sh`; `CUDA_HOME`, `CUDA_ARCH`), output in `build/cuda` (`build/cuda-debug` with `DEBUG=1`), CUDA runtime linked statically; `make cuda test` runs every suite of that build. `make` stays CPU-only and needs no toolkit.
- [x] **6.2 `src/backend/backend.h`.** Devices and their properties, device and pinned memory, copies (any direction, asynchronous), memset, three streams per device (compute, host to device, device to host), events for ordering and timing; runtime errors mapped to `ATH_ERR_DEVICE` (new) and `ATH_ERR_NOMEM`.
- [x] **6.3 `src/backend/gpu.cu`, `gpu_none.c`.** The CUDA implementation, in the C subset of CUDA C++; the stub for builds without a GPU backend (no devices, every call `ATH_ERR_UNSUPPORTED`).
- [x] **6.4 `src/backend/compat.h`.** The CUDA runtime names used by GPU code mapped to HIP, and `ATH_WARP`; compiled only by nvcc until the HIP build (step 11).
- [x] **6.5 Device arena.** `src/backend/devarena.c`: linear allocation in device memory with mark and reset, like the core arenas, block records on the host.
- [x] **6.6 `athanor devices`.** List the devices; `--bench` times copies with events.
- [x] **6.7 `tests/test_backend.c`.** The stub in the CPU build; on the GPU: properties, copies in every direction at odd sizes and offsets from pinned and ordinary memory, memset, three streams ordered by events, timing, an impossible allocation, the device arena.
- [x] **6.8 Documentation.**

#### Acceptance criteria

1. `make cuda` builds with CUDA 12.8 or newer for `sm_120` with zero warnings (host and device code), and refuses an older nvcc with a clear message; `make` is unchanged and needs no CUDA toolkit.
2. On the RTX 5080: device and pinned allocation, copies in every direction (synchronous to the host and asynchronous), memset, three streams ordered only by events, and event timing, all checked byte for byte by `test_backend`.
3. Errors: an impossible allocation gives `ATH_ERR_NOMEM` and the device keeps working; a bad device index or stream gives `ATH_ERR_INVALID_ARG`; no `exit` or `abort`.
4. `make cuda test` green, also with `DEBUG=1`; `make test` green, where `test_backend` checks the stub and skips the GPU tests; compute-sanitizer finds no memory errors, leaks or races in `test_backend`.
5. `ldd` shows no new library in either build.
6. Documentation updated.

Result (2026-11-24, RTX 5080 `sm_120`, driver 580.178.04 (CUDA 13.0), CUDA toolkit 13.5.0 (nvcc 13.5.37), gcc 15.2, glibc 2.43):

| # | Criterion | Result |
| --- | --- | --- |
| 1 | build | `make cuda CUDA_HOME=<CUDA 13.5>` builds `build/cuda/athanor` and every test with zero warnings; with the system nvcc 12.4 it stops: "CUDA 12.4; sm_120 needs CUDA 12.8 or newer (set CUDA_HOME)". `make` builds as before, without a toolkit |
| 2 | runtime on the GPU | `test_backend` on the RTX 5080 (84 SMs, 15.4 GiB, L2 64 MiB, 99 KiB shared memory per block): copies at 7 sizes from 1 byte to 24 MiB and unaligned offsets, pinned and ordinary memory, both directions and device to device; memset; the three streams ordered only by events over 8 rounds; timing (256 MiB device to device 0.6 ms, host to device 5 ms); every byte checked |
| 3 | errors | an allocation of twice the device memory gives `ATH_ERR_NOMEM` and the device keeps working; device index out of range, bad stream, null pointers give `ATH_ERR_INVALID_ARG` |
| 4 | tests | `make cuda test`: 24 + 11 + 17 + 1 + 7 + 12 + 6 (1 skipped: the stub test) pass; with `DEBUG=1` too (about 4 minutes); `make test` and `make DEBUG=1 test`: the same suites, `test_backend` 1 pass and 6 skipped; compute-sanitizer on `test_backend`: memcheck with leak check finds no memory error and 0 bytes leaked (its only reports are the two API errors of the deliberate impossible allocation), racecheck 0 hazards |
| 5 | `ldd` | both builds: `libm`, `libc` only (the CUDA runtime is static and loads the driver itself) |
| 6 | documentation | `src/backend/README.md` rewritten; READMEs of `src/core`, `tools`, `tests`; `docs/ARCHITECTURE.md` tree; `CLAUDE.md` quick reference; `docs/program/PROGRAM.md` row 7; this file and `status_extended.md` |

For the record (no performance claim), `athanor devices --bench`, 256 MiB copies, best of 5: pinned host to device 55.8 GB/s, device to pinned host 58.1 GB/s, pageable 22 to 24 GB/s both ways, device to device 821 GB/s (read + write).

### Step 5

**Step 5: tokenizer, sampling, CPU chat** (see `docs/PLAN.md` §4, gate amended by CHANGE-001). Started 2026-11-24 on branch `step5-tokenizer-chat`. **Done 2026-11-24** (pushed, to be merged into `main`); step 6 is next. Merged into `main` (fast-forward) on 2026-11-24, at the maintainer's request, and tagged `v0.1-cpu-chat`: the first working chat, CPU only.

Goal: text in and out of the model on the CPU. A byte-level BPE tokenizer that gives the reference tokenizer's ids, the family's chat template (with tool definitions and the thinking switch) rendered byte for byte as the reference renders it, a parser that turns the model's output into text, thinking and tool calls, sampling, and a first `athanor chat` on the CPU forward pass of step 4.

#### Tasks

- [x] **5.1 Text golden data.** `tests/golden/text.py`: with the pinned reference (transformers 5.24.0, tokenizers 0.23.3), the token ids and decoded text of a corpus (hand-written cases in many languages, emoji, code, white space, contractions, added tokens; repository files; every Unicode code point in five contexts; every canonical decomposition with shuffled marks), conversations rendered by the checkpoint's chat template with and without tools and thinking (and the ones it rejects), and a greedy chat generation of the fp32 reference with its top-2 margins and the bf16 noise. `make golden` runs it.
- [x] **5.2 Unicode.** `tools/dev/gen_unicode.py` generates `src/text/unicode_tables.h` (committed): character classes (letter, mark, number, white space) and NFC data. `src/text/unicode.c`: UTF-8, classes, NFC.
- [x] **5.3 `src/text/bpe.c`.** Byte-level BPE from `tokenizer.json`: added tokens, NFC, the family's pre-tokenization, merges by rank; decoding. The loader refuses any tokenizer feature it does not implement and checks that the file's split regex is the one the family implements.
- [x] **5.4 Contract.** `model.h`: `ath_tool_def`, `ath_tool_call`, `ath_msg`, `ATH_CHAT_*`, `ath_out_event`, `ath_parse_state`; the family members `pretokenize_regex`, `pretokenize`, `chat`, `parse`; `ath_model_tokenizer`, `ath_model_chat`, `ath_model_parse` (`src/model/text.c`) and `ath_state_copy`. Core helpers: `ath_buf` (`src/core/buf.c`), a JSON writer with Python's `json.dumps` formatting (`src/core/json.c`), `ath_rng` (`src/core/rng.c`).
- [x] **5.5 `src/models/qwen35/text.c`.** The pre-tokenization regex by hand, the chat template by hand (tools, thinking, tool calls and results, history), the incremental output parser (thinking, tool calls with arguments typed by the tool's schema).
- [x] **5.6 `src/ops/sample.c`.** `ath_sample_ref`: greedy, temperature, top-k, top-p, min-p; the uniform number comes from the caller's generator.
- [x] **5.7 `athanor chat`.** `tools/chat.c`: one prompt (`-p`) or a conversation on stdin, thinking on request, tool definitions from a file (calls shown, never run; `/tool` answers one), state reused across turns.
- [x] **5.8 `tests/test_text.c`.** Unicode, NFC, the JSON writer, pre-tokenization, parser fixtures (whole, split at every byte, byte by byte), template errors, sampling; against the golden data: the corpus, the templates, the greedy generation.
- [x] **5.9 Documentation.** Contract docs, the chat template and tokenizer in `docs/contract/qwen35-0.8b.md`, module READMEs, `tests/README.md`, `tests/golden/README.md`, `docs/program/PROGRAM.md`, this file and `status_extended.md`.

#### Acceptance criteria

1. Exact round trip on a corpus against the reference tokenizer: the same token ids for every text, and the same text decoded from them.
2. Rendered prompts byte-identical to the reference chat template for a fixture set with and without tools, thinking on and off; where the template raises an error, `chat` fails (CHANGE-001).
3. Parser fixtures pass: thinking, several tool calls in one turn, typed arguments, malformed and truncated markup, every fixture also split at every byte and fed byte by byte (CHANGE-001).
4. A first chat, slow and correct: `athanor chat` answers on the CPU; athanor's greedy generation (bf16) on a fixed chat prompt equals the fp32 reference's at every step whose top-2 margin is above the bf16 noise.
5. Pipelines and tools reach tokenizer, template and parser only through the contract; no model names in `src/` outside `src/models/`.
6. `make test` green with zero warnings, also with `DEBUG=1`; `ldd build/athanor` shows no new library.
7. Documentation listed in 5.9 updated.

Result (2026-11-24, Qwen3.5-0.8B revision `2fc0636`, transformers 5.24.0, tokenizers 0.23.3, Python 3.14.4, gcc 15.2, glibc 2.43):

| # | Criterion | Result |
| --- | --- | --- |
| 1 | tokenizer | 188 texts, 28.9 MB, 28,185,794 tokens: every id identical to the reference's, every decoded text identical (`golden_tokenizer_corpus`). It includes every code point but the surrogates in five contexts and every canonical decomposition with shuffled marks. Load 0.06 s, encode 0.6 s for the whole corpus |
| 2 | templates | 17 conversations byte-identical (system or not, multi-turn, thinking in content or apart, tools with and without system, a round trip with two calls and two results, typed arguments including nested objects, floats and nulls, tool responses as user messages, trimming, Unicode), and their token ids identical; the 3 conversations the template rejects fail with `ATH_ERR_INVALID_ARG` (`golden_chat_template`) |
| 3 | parser | 17 fixtures, each whole, split at every byte and byte by byte, give the same events (`parse_fixtures`) |
| 4 | chat | `athanor chat -p "What is the capital of Italy? Answer in one sentence." --temp 0` answers "The capital of Italy is Rome." in 0.9 s. Greedy: 27 of the 32 tokens of the fixed prompt are above the bf16 noise (0.19) and all equal the fp32 reference's; after step 27 (margin 0.01) the comparison stops, and the 32 tokens are still the reference's (`golden_chat_generate`). Multi-turn and tool calls checked by hand (log) |
| 5 | contract | `tools/chat.c` and `tests/test_text.c` include no family header; `grep -rni qwen src tools --include=*.[ch]` outside `src/models/` finds nothing |
| 6 | `make test`, zero warnings, also `DEBUG=1` | 24 + 11 + 17 + 1 + 7 + 12 tests pass in release (about 17 s) and with ASan/UBSan; `ldd build/athanor`: `libm`, `libc` only |
| 7 | documentation | `docs/ARCHITECTURE.md`; `docs/contract/qwen35-0.8b.md` (Tokenizer, Chat template, Output markup); READMEs of `src/core`, `src/text`, `src/ops`, `src/model`, `src/models/qwen35`, `tools`, `tests`, `tests/golden`; `docs/program/PROGRAM.md` rows 5, 6, 6b; README; this file and `status_extended.md` |

Speed, for the record (no performance claim; Ryzen 7 9800X3D, 8 threads, bf16 activations): the prompt runs at about 40 to 45 tokens per second and the answer decodes at about 22 to 25 tokens per second, sampling and parsing included.


### Step 4

**Step 4: CPU forward** (see `docs/PLAN.md` §4). Started 2026-11-24 on branch `step4-cpu-forward`. **Done 2026-11-24** (pushed, to be merged into `main`); step 5 is next. Task 4.7 was added after the step closed, at the maintainer's request, and task 4.8 applies the specification change CHANGE-001 (log, 2026-11-24). Merged into `main` (fast-forward) on 2026-11-24, at the maintainer's request.

Goal: the whole Qwen3.5-0.8B forward pass on the CPU, composed from the step 3 ops, with the per-sequence state (KV cache, conv and recurrent state) that prefill and decode need, reached by pipelines only through the contract; and evidence on fixed prompts that its logits are as close to the exact model as the reference implementation's.

#### Tasks

- [x] **4.1 End-to-end metric.** Settle the open question (below): KL against an fp32 run of the reference, gated on the bf16 reference's own KL; greedy agreement; the residual stream per layer as a diagnostic; an fp32 forward of athanor against the fp32 reference, tight.
- [x] **4.2 Contract.** `src/model/model.h`: state kinds and specs (`ath_state_spec`, listed by the model), `ath_state` (one sequence: activation dtype, capacity, tokens so far, one buffer per spec), `ath_batch`, `ath_hooks` with `on_residual`, and `forward` in `ath_family`. `src/model/state.c`: `ath_state_init`, `ath_state_reset`, `ath_state_bytes`, `ath_model_forward` (argument checks, positions, scratch memory, `n_past`).
- [x] **4.3 `src/models/qwen35/forward.c`.** Embedding, Gated DeltaNet and attention mixers, MLP, final norm and fp32 logits (every token or the last), composed from the step 3 ops in the order of the contract; state specs declared at load; activations in the state's dtype (bf16 as the reference, or f32); MoE layers rejected until step 12.
- [x] **4.4 `tests/golden/dump.py`, forward records.** Two fixed prompts (the op prompt and a chat-formatted one), each followed by 8 greedy decode steps of the bf16 reference; the same tokens through the reference loaded in fp32; per position, the residual stream entering every layer and the final norm, and the logits, of both runs.
- [x] **4.5 `tests/test_forward.c`.** On a tiny random checkpoint (builder shared with `test_model.c` in `tests/tiny.h`): argument checks, state specs and sizes, hooks, and prefill all at once = in pieces = token by token, bit for bit, in bf16 and f32. On the real checkpoint, against the forward records: the KL gate, greedy agreement, the fp32 forward against the fp32 reference.
- [x] **4.6 Documentation.** Contract docs (`docs/ARCHITECTURE.md`, `src/model/README.md`), family README, `tests/README.md`, `tests/golden/README.md`, `docs/contract/qwen35-0.8b.md` (logits measured), `docs/program/PROGRAM.md`, this file and `status_extended.md`.
- [x] **4.7 Fast CPU matmul** (added after the step closed). `src/core/threads.c`: a fixed pthread pool. `ath_matmul_fast` in `src/ops/matmul.c`: several outputs per thread, each summed exactly as the reference sums it, so the same bits with any number of threads. The pool joins the contract (`forward`, `ath_model_forward`) and the forward pass uses `ath_matmul_fast`. Tests: the pool; fast against reference bit for bit, on random shapes and dtypes and on every golden matmul record. The AVX-512 variant that was not chosen is written down in `docs/proposals/cpu-matmul-avx512.md`.
- [x] **4.8 CHANGE-001** (`docs/CHANGE-001-scope-tools-cache.md`, from the maintainer). Scope boundary (engine plus server, no agent harness): AGENTS.md invariant 10, `docs/ARCHITECTURE.md` (Scope: engine versus harness), README, PLAN decisions. Vision-encoder tensors (and the MTP head until step 13) bound as params flagged `ATH_P_UNUSED`, in the family's ignored groups; the coverage rule leaves no tensor unbound; `inspect --roles` reports their count and size. `tests/golden/dump.py` checks that it runs the text path only and records the reference class. Tool calling (the text members `chat` with tool definitions and `parse`) and on-disk prefix snapshots written into the plan, the contract target and the module READMEs, for steps 5, 13 and 14.

#### Acceptance criteria

1. `forward`, the state specs, `ath_state`, `ath_batch` and `ath_hooks` are in `model.h`, documented, and the tests reach the forward pass only through the contract (no family header).
2. Over every position of both fixed prompts, prefill and decode, athanor with bf16 activations has a mean KL against the fp32 reference no larger than the bf16 reference's mean KL against the same run (the gate); the maximum and the per-prompt means are reported. (Wording made explicit on 2026-11-24 after the measurement: the resolution of the open question already said "over every position of the fixed prompts"; see the log.)
3. athanor's greedy token equals the fp32 reference's at every position where the fp32 reference's top-2 margin is larger than the bf16 noise: the largest change of that margin between the fp32 and the bf16 reference over the same positions.
4. athanor with f32 activations: the residual stream entering every layer and the logits are within a justified tolerance of the fp32 reference (`tests/README.md`).
5. Prefill all at once, in pieces and token by token give bit-identical logits and states.
6. `make test` green with zero warnings, also with `DEBUG=1`; no model names in `src/` outside `src/models/`.
7. Documentation listed in 4.6 updated.
8. (4.7) `ath_matmul_fast` gives the same bits as `ath_matmul_ref` for every dtype combination, with and without threads, on random shapes and on every golden matmul record; the forward pass gives the same logits as before 4.7, bit for bit.
9. (4.8) CHANGE-001 acceptance: contract changes compile; existing tests stay green; every document it lists is updated consistently (no file describes an integrated agent or omits the scope boundary); a log line records it with what was deferred and why.

Result (2026-11-24, Qwen3.5-0.8B revision `2fc0636`, transformers 5.24.0, torch 2.15.0+cpu, gcc 15.2, glibc 2.43):

| # | Criterion | Result |
| --- | --- | --- |
| 1 | contract | `ath_state_spec`, `ath_state`, `ath_batch`, `ath_hooks` (`on_residual`) and `forward` in `model.h`; `ath_state_bytes`, `ath_state_init`, `ath_state_reset`, `ath_model_forward` in `src/model/state.c`; `tests/test_forward.c` includes no family header |
| 2 | KL gate | mean KL against the fp32 reference over all 119 positions: athanor 3.0e-4, bf16 reference 5.0e-4 (ratio 0.60). Per prompt: prose (75 + 8) 1.7e-4 against 5.3e-4; chat (28 + 8) 6.0e-4 against 4.4e-4, all of the excess from one ill-conditioned position (position 0, KL 7.3e-3 against 2.7e-4; without it 4.1e-4 against 4.5e-4). Maximum 7.3e-3 against 3.2e-3, at that position |
| 3 | greedy | athanor equals the fp32 reference at all 104 positions whose fp32 top-2 margin exceeds the bf16 noise (0.21); below it, athanor differs at 2 positions of chat and the bf16 reference at 2 of prose |
| 4 | f32 forward | mean KL 4e-11 (prose) and 5e-11 (chat); residual stream within 4.7e-5 at every layer (tolerance 2e-4), logits within 2.7e-5 (tolerance 1e-4) |
| 5 | pieces | on tiny random checkpoints, 12 tokens all at once, as 5 + 1 + 6 and one at a time give bit-identical logits and states, in bf16 and f32; the real-checkpoint runs use one prefill call and then single tokens |
| 6 | `make test`, zero warnings, also `DEBUG=1` | 22 + 11 + 16 + 1 + 7 tests pass in release (about 45 s, `test_forward` 37 s with four threads) and with ASan/UBSan (about 4 minutes; `test_forward` runs the first 4 positions there); `grep -rni qwen src tools --include=*.[ch]` outside `src/models/` finds nothing; `ldd build/athanor`: `libm`, `libc` only. After 4.7: 24 + 11 + 17 + 1 + 7 tests, about 13 s in release (`test_forward` 11 s) and about 3 minutes with ASan/UBSan (`test_forward` 34 s there instead of 100 s) |
| 7 | documentation | `docs/ARCHITECTURE.md`, READMEs of `src/model`, `src/models/qwen35`, `tests`, `tests/golden`; contract (logits, End-to-end); `docs/MODELS.md`; `docs/program/PROGRAM.md` rows 3, 4, 9; this file and `status_extended.md` |
| 8 | fast matmul (4.7) | `matmul_fast_same_bits` (7 shapes, including partial blocks and work above the threading threshold, × 8 dtype combinations, without a pool and with 3 threads) and every matmul record of `test_golden` (all projections and the lm_head, rows 0 and T − 1, 8 threads) are bit-identical to `ath_matmul_ref`. The forward pass: a hash of every logit of 32 prefill tokens and 8 decode steps, in bf16 and f32, is the same for the tree before 4.7 and for 4.7 with 1 and 8 threads; every number `test_forward` prints is unchanged |
| 9 | CHANGE-001 (4.8) | `ATH_P_UNUSED` in `model.h`; `inspect --roles` on Qwen3.5-0.8B: 488 checkpoint tensors bound to 488 params, 320 used by the text path (the same 1,504,791,232 bytes as before) and 168 unused in the ignored groups `model.visual.` (153 tensors, 191.87 MiB) and `mtp.` (15 tensors, 39.01 MiB), 0 unbound; `test_model` checks the unused params (flag, role, layer, zero-copy, order) on tiny checkpoints and on the real one. Golden data regenerated: the manifest records `Qwen3_5ForCausalLM` (transformers 5.24.0), and `dump.py` stops if it loads another class or a vision encoder; the reference's KL numbers are unchanged (5.255e-4 and 4.427e-4). `make test` green (24 + 11 + 17 + 1 + 7), also with `DEBUG=1`. Documents: AGENTS.md, README, `docs/ARCHITECTURE.md`, `docs/PLAN.md`, `docs/MODELS.md`, contracts, READMEs of `src/model`, `src/models/qwen35`, `src/net`, `src/infer`, `src/text`, `tools`, `docs/program/PROGRAM.md`. Deferred to step 5, with their first user: the text members of the contract and the template and parser tests (log) |

Speed, for the record (no performance claim): 0.29 s per token per thread for prefill and decode, almost all of it in the scalar `ath_matmul_ref` (about 3.6 GFLOP/s, bound by the latency of one fp32 addition chain per output). Logits for one token add 0.14 s.

After 4.7, for the record (no performance claim; Ryzen 7 9800X3D, 8 threads): a decode step with its logits takes 41 ms (0.43 s before), and 32 prefill tokens with the logits of all 32 take 0.84 s (13.7 s before). On one thread: 138 ms and 4.2 s. `ath_matmul_fast` alone, one token: 3.4× the reference on one thread, 12× to 16× on 8. On the lm_head that is about 58 GB/s of weights, close to what the memory delivers.

#### Notes (CHANGE-001)

- Qwen3.5-0.8B is a vision-language checkpoint (architecture `Qwen3_5ForConditionalGeneration`; transformers loads it with `AutoModelForMultimodalLM`, or its text path with `AutoModelForCausalLM`). `weights.c` recognizes the vision-encoder tensors by name prefix (`model.visual.`), binds them with `ATH_P_UNUSED`, and `athanor inspect --roles` reports their count and size. The MTP head (`mtp.`) is handled the same way until step 13 gives it real roles.
- `tests/golden/dump.py` runs the text path only: it stops unless the loaded class is `Qwen3_5ForCausalLM` without vision parameters, and the manifest records the class (`reference.class`) next to the transformers version.


### Step 3

**Step 3: CPU reference ops and golden harness** (see `docs/PLAN.md` §4). Started 2026-11-24 on branch `step3-ops-golden`. **Done 2026-11-24** (pushed, to be merged into `main`); step 4 is next. Merged into `main` (fast-forward) on 2026-11-24.

Goal: every op of the Qwen3.5-0.8B forward pass as a scalar CPU reference in `src/ops/`, and evidence on the real weights that each one computes what the pinned reference implementation computes.

#### Tasks

- [x] **3.1 Golden environment, f32 question.** Pin the Python environment of the golden harness (`tests/golden/requirements.in` and the uv lock `requirements.txt`: transformers 5.24.0, torch 2.15.0 CPU build, Python 3.14) and settle the open question on the f32 checkpoint tensors.
- [x] **3.2 `src/core/tensor.h`.** bf16 conversion (round to nearest, ties to even, NaN as torch does), element load and store for f32 and bf16, `ath_tensor_same_shape`.
- [x] **3.3 `src/ops/`.** `ops.h` and a `_ref` function for every op of the contract: `embed`, `matmul`, `rmsnorm` (plain and zero-centered), `rmsnorm_gated`, `l2norm`, `rope` (partial, rotate-half), `swiglu`, `sigmoid_mul`, `add`, `attn` (causal, grouped-query), `conv` (causal depthwise with state), `gdn_gates`, `gdn` (gated delta rule, recurrent).
- [x] **3.4 `tests/golden/dump.py`.** Run the reference (CPU, bf16, eager attention, torch fallbacks) over a fixed prompt longer than one Gated DeltaNet chunk and one decode step; record every op of every layer with its inputs, its output and the same function rerun in fp32; check versions and hashes; write a manifest and a safetensors file. `make golden` runs it.
- [x] **3.5 `tests/test_ops.c`.** Hand-computed cases and exact properties: streaming equals all at once (conv, Gated DeltaNet), decode equals the last prefill row (attention), causality, head grouping, argument checks.
- [x] **3.6 `tests/test_golden.c`.** Replay every record through the `_ref` functions on the real weights, native and exact, with a tolerance per op and output justified in `tests/README.md`.
- [x] **3.7 Documentation.** Contract, READMEs of `src/ops`, `src/core`, `src/backend`, `tests`, `tests/golden`, `docs/ARCHITECTURE.md` tree, AGENTS op checklist, `docs/program/PROGRAM.md`.

#### Acceptance criteria

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

**Step 8: sequence state and engine.** The forward pass on the GPU: the weights uploaded once, the state of a sequence (KV cache, conv and recurrent state) in device memory, the ops chosen by where the tensors live (the dispatching entry point per op joins `ops.h` with this first caller), and an engine in `src/infer/` that runs prompts and decodes with the GPU forward pass; `athanor chat` on the GPU in bf16. Gate (`docs/PLAN.md`): GPU chat in bf16; greedy output identical to the reference for 256 tokens on 50 prompts.

Superseded preview (written in step 6): **Step 7: GPU ops, one by one.** A CUDA kernel in `src/ops/<op>.cu` for every op of the forward pass (embed, matmul through cuBLASLt with its handle in the backend, the norms, RoPE, activations, attention, the short convolution, the Gated DeltaNet gates and recurrence), each compared with its CPU reference on random and real inputs, as AGENTS.md's op checklist asks. Gate: every kernel matches its CPU reference.

Superseded preview (written in step 5): **Step 6: GPU backend.** `src/backend/gpu.cu` and `compat.h`: device memory, copies, streams, events and timing on CUDA, written once for CUDA and HIP. Gate: allocation, copies, streams, events, timing on CUDA. Blocked until the CUDA toolkit is upgraded (open questions: nvcc 12.4 cannot target `sm_120`).

Superseded preview (written in step 4): **Step 5: tokenizer, sampling, CPU chat.** `src/text/bpe.c` and the generated `unicode.h` (byte-level BPE from `tokenizer.json`), the family's pre-tokenization split and chat template in `src/models/qwen35/text.c` (`pretokenize` and `chat` join the contract), `src/ops/sample.c`, and a first `athanor chat` on the CPU forward of step 4. Gate: exact token round-trip on a corpus against the reference tokenizer; a first chat, slow and correct.

Amended by CHANGE-001 (2026-11-24): `chat` takes tool definitions and a thinking switch, and `parse` (incremental, streaming-safe: thinking blocks, tool calls, markup split across pieces, malformed markup as text, end-of-stream flush) joins the contract with them. A fixture generator in `tests/golden/` renders reference prompts with the upstream chat template (with and without tools, thinking on and off) and stores them byte-exact; `tests/test_text.c` compares the rendering with the fixtures and runs parser fixtures (split pieces, malformed markup, several tool calls in one turn). Gate additions: rendered prompts byte-identical to the reference chat template for the fixture set; parser fixtures pass.

Superseded preview (written in step 3): **Step 4: CPU forward.** `src/models/qwen35/forward.c` composes the step 3 ops into the decoder layers and the logits; `forward` and the state specs (KV cache, recurrent and conv state) join `model.h` with this first user; a minimal CPU state for one sequence; `tests/golden/dump.py` extends to per-layer hidden states and logits on fixed prompts. Gate: logits match the reference implementation on fixed prompts (metric to settle first: see Open questions).

Superseded preview (written in step 2): **Step 3: CPU reference ops and golden harness.** `src/ops/*.c` references (scalar, fp32 accumulation) for every op in the contract, `src/backend/cpu.c`, `tests/golden/dump.py` against the pinned reference, `tests/test_ops.c`. Gate: every op within tolerance of golden tensors on real weights.

Superseded preview (written in step 1): **Step 2: contract and family loading.** Write `src/model/model.h` as in `docs/ARCHITECTURE.md`, `registry.c`, and `src/models/qwen35/{config.c,weights.c}`: parse `config.json` into the layer schedule (which layers are Gated DeltaNet, which are full attention, MLP or MoE), bind every tensor to an `ath_param` with role, layer, expert and flags, and fail loudly on any unbound or missing tensor. `athanor inspect --roles` prints the result.

## Open questions

- **Invariant 7 and the two devices (step 7).** "Placement never changes results." The GPU kernels match the CPU references within tolerances, not to the bit: other summation orders, fused multiply-adds, the device's `expf`. On one device, where a tensor is stored never changes a result (the engine moves data before launching kernels). The question comes back at step 12, where an expert may run on the CPU (`_fast`) or on the GPU depending on where its weights are: then placement would change the last bits. Options: (a) read the invariant as "storage never changes results on a given compute device" and hold the CPU and GPU expert paths to a tolerance; (b) require the CPU expert path to give the GPU's bits, which ties both to one summation order without fused multiply-adds and costs speed on both. Recommended: (a), decided at step 12. Nothing in steps 7 to 11 depends on it.

- **Qwen3.6-27B and Qwen3.8-27B declare `output_gate_type: "swish"`** (their `config.json`, read 2026-11-24). Qwen3.5 has no such key and gates the attention output with a sigmoid (contract, Full-attention layer). `config.c` does not read the key, so athanor would load those checkpoints with the sigmoid gate and no error. Before they are supported: check the reference for what "swish" means there, record it in a contract file, and reject any value the code does not implement. Until then nothing depends on it (only Qwen3.5-0.8B runs).

- **Presence penalty (step 5).** Qwen3.5's model card recommends a presence penalty (1.5 without thinking, 0 to 1.5 with it) next to temperature, top-k and top-p. `ath_sampler` has temperature, top-k, top-p and min-p only; a penalty needs the counts of the tokens generated so far. Options: add it to the sampler now (CPU), or with GPU sampling (step 9) or the server (step 13), where request parameters ask for it. Recommended: with the server, step 13. Nothing in step 5 depends on it; `athanor chat` uses the card's other settings.

- (none yet) (superseded 2026-11-23 by the entries below)
- **CUDA toolkit too old for `sm_120` (blocks step 6).** The development machine has nvcc 12.4 (driver 580, CUDA 13.0 runtime). Blackwell consumer GPUs (`sm_120`, RTX 5080) need CUDA 12.8 or newer. Options: install CUDA 12.8+/13.x before step 6 (recommended), or build for `compute_90` PTX and rely on JIT (slower start, no FP4/FP8 `sm_120` paths). Nothing in steps 1–5 depends on it. **Resolved 2026-11-24:** CUDA 13.5 installed at user level from NVIDIA's redist archives (no root needed; the system nvcc 12.4 stays), selected with `CUDA_HOME`. The 13.5 runtime runs on driver 580 (CUDA 13.0) through CUDA's minor-version compatibility: checked with `sm_120` code and a cuBLASLt bf16 matmul, then by `test_backend`. The driver stays 580 for now; a newer one is worth fixing before the benchmarks of step 9, together with GPU performance-counter access for Nsight Compute (needs root).
- **Hardware assumptions in `docs/PLAN.md` §2 can be confirmed:** CPU is an AMD Ryzen 7 9800X3D (Zen 5; `avx512f`, `avx512bw`, `avx512vl` and `avx512_bf16` present); RAM shows 59 GiB usable, consistent with 64 GB installed.
- **Step 2, weight parameterizations.** The checkpoint suggests zero-centered RMSNorm weights for the decoder-layer norms and `q_norm`/`k_norm`, and plain weights for the Gated DeltaNet output norm (`docs/contract/qwen35-0.8b.md`, observations). To be confirmed against the pinned reference implementation, as the **verify** rule requires. **Resolved 2026-11-24:** confirmed against transformers 5.24.0 (`Qwen3_5RMSNorm` uses `1 + w`, `Qwen3_5RMSNormGated` a plain `w`); see the contract, sections RMSNorm and Gated RMSNorm.
- **Golden data and f32 checkpoint tensors (step 3).** `A_log` and `linear_attn.norm.weight` are f32 in the checkpoint, and the reference has no `_keep_in_fp32_modules`. If the golden dump loads the model in bf16, the reference rounds them to bf16 and its outputs differ from athanor's (which uses the stored f32 values). Options: load the reference in fp32 for op-level goldens, or check what transformers 5.24.0 does with mixed-dtype checkpoints and record it. To settle before writing `tests/golden/dump.py`. **Resolved 2026-11-24:** transformers 5.24.0 rounds both tensors to bf16 on load (`dtype="auto"` and `dtype=torch.bfloat16` alike). `dump.py` loads in bf16 and puts the stored f32 values back, which the forward code handles unchanged, so the reference and athanor use the same weights; on the golden prompt this brings the bf16 reference closer to an fp32 run of the model (max KL 1.4e-3 instead of 2.9e-3). Recorded in the contract (Checkpoint dtypes) and `tests/golden/README.md`.
- **End-to-end reference and metric for step 4.** The reference differs from athanor by design in a few roundings: the eager attention rounds `q·kᵀ` to bf16 (up to 9.8% of an attention output element of Qwen3.5-0.8B, relative to its magnitude plus the tensor RMS, `tests/README.md`), the reference computes the logits in bf16 where `docs/MODELS.md` asks for fp32, and every elementwise op rounds more often in the reference than in athanor. Logits will therefore not match to the bit. Options for the step 4 gate: (a) per-layer hidden states and logits within a tolerance derived like the op tolerances; (b) KL of athanor against an fp32 run of the reference, no larger than the KL of the bf16 reference against the same fp32 run (athanor at least as close to the exact model as transformers in bf16), plus greedy tokens identical where the reference's top-2 margin is above the bf16 noise. Recommended: (b), with (a) as a diagnostic. To settle at the start of step 4. **Resolved 2026-11-24 (step 4, task 4.1):** (b) is the gate: mean KL over every position of the fixed prompts (prefill and decode) against an fp32 run of the reference, no larger than the bf16 reference's mean KL against the same run; the maximum is reported, not gated. Greedy agreement is required where the fp32 reference's top-2 margin exceeds the bf16 noise, measured as the largest change of that margin between the fp32 and the bf16 reference. (a) is reported as a diagnostic (the residual stream entering every layer, against both runs). In addition athanor runs its forward pass with f32 activations against the fp32 reference, within a tolerance justified like the op tolerances: this checks the composition tightly, without the bf16 noise that the KL gate tolerates.

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
- 2026-11-24: `docs/status_extended.md` added: for every task of steps 1 to 3, the files created or changed, what each is for and what was implemented; kept up to date with every task from now on (AGENTS.md workflow, step 4).
- 2026-11-24: `step3-ops-golden` merged into `main` (fast-forward); step 4 started on `step4-cpu-forward`. End-to-end metric settled (open questions): KL gate against an fp32 run of the reference, greedy agreement above the bf16 noise, residual-stream diagnostic, and an fp32 forward of athanor checked tightly.
- 2026-11-24: step 4 implemented on branch `step4-cpu-forward`: state specs, `ath_state`, `ath_batch`, `ath_hooks` and `forward` in `src/model/model.h`, `src/model/state.c`, `src/models/qwen35/forward.c` (with the layer table in `qwen35.h`, `weights.c`, `family.c`), forward records in `tests/golden/dump.py`, `tests/tiny.h` (moved out of `test_model.c`), `tests/test_forward.c`. All acceptance criteria met (table above).
- 2026-11-24: decisions taken within step 4 (`PLAN.md` unchanged): state specs count elements, not bytes, and the element type follows from the kind and the state's activation dtype (KV and conv state hold activations, the recurrent state is fp32), because a state can run with bf16 or f32 activations; `forward` takes the model as `const` and a scratch arena released by `ath_model_forward` after every call; a batch is one sequence until the engine (step 8); `ath_hooks` has `on_residual` only (its first users are the forward tests; `on_matmul_in` waits for phase 2); token ids, capacity and the logits tensor are checked once in the contract, not by each family; the family keeps a table of each layer's params, filled while binding, instead of finding params by name in the forward pass; strided slices (query and gate interleaved in `q_proj`, q/k/v in the convolution output) are copied into contiguous tensors for the ops; scratch memory is released after every layer; the minimal CPU state of one sequence is in `src/model/state.c` (host memory) and the engine's state on the GPU stays in `src/infer/state.c` (step 8); `forward_golden` runs its four independent runs in parallel threads (test only; the ops stay single-threaded) and a DEBUG build runs only the first 4 positions.
- 2026-11-24: the KL gate as first implemented compared each prompt separately and failed on the chat prompt (athanor 6.0e-4, reference 4.4e-4) while athanor beat the reference on prose (1.7e-4 against 5.3e-4). Analysis (contract, End-to-end; `tests/README.md`, Forward pass): the excess is one ill-conditioned position (`<|im_start|>`, a near-cancellation in layer 15), whose KL varies twentyfold within the reference itself (2.7e-4 with the f32 tensors as stored, 5.5e-3 with them rounded to bf16 as transformers does by default), and the reference with its default sdpa attention would fail the per-prompt comparison against the eager reference too (7.5e-4 against 5.1e-4 on the chat prefill). The gate now pools every position of both prompts, as the resolution of the open question worded it; criterion 2 states it explicitly. Per-prompt means stay in the report. A more robust per-prompt gate would need many more positions (more prompts) than a CPU test can afford in seconds.
- 2026-11-24: task 4.7 added after step 4 closed, at the maintainer's request: on `ath_matmul_ref` the CPU chat of step 5 would decode about 2 tokens per second. Three designs were measured with a microbenchmark:
  - B: scalar, several outputs per thread, the same bits;
  - C: AVX-512 across outputs on repacked weights, the same bits;
  - along k, as BLAS libraries and `vdpbf16ps` do: other bits.

  B was chosen. It reaches the memory bandwidth in decode, like C, without a copy of the weights or a CPU requirement. C is written down in `docs/proposals/cpu-matmul-avx512.md`, not scheduled. AGENTS.md admits a CPU `_fast` path for an op that runs on the CPU at runtime, which the step 5 chat does; `PLAN.md` placed the first one at step 12 (MoE experts) and stays unchanged.
- 2026-11-24: decisions taken within task 4.7:
  - The pool is passed explicitly, because AGENTS.md allows no globals. `forward` and `ath_model_forward` take `ath_pool *pool`; NULL means the calling thread only. This replaces "the ops stay single-threaded" of the step 4 decisions, for matmul only.
  - The pool is in `src/core/threads.c`, as `ARCHITECTURE.md` planned. It assigns tasks statically, and runs from several threads take turns.
  - `src/backend/cpu.c` is still not written: the pool needs no backend. It arrives with CPU feature detection or the GPU runtime.
  - `ath_matmul_fast` computes 8 outputs per thread with bf16 weights. Weights in f32 (only in tests) and the last partial block run one output at a time, as the reference. A matmul under 2^18 multiply-adds runs on the calling thread.
  - The bit identity relies on GCC not contracting a·b + c in ISO C mode (`-std=c11`). The tests with f32 inputs would catch a build that did.
  - The four runs of `forward_golden` share one pool.
- 2026-11-24: a tiny-model check of bf16 against f32 activations was dropped from `test_forward`: random uniform weights make an ill-conditioned network (bf16 and f32 logits differ by 2% to 30% of their RMS depending on the seed), so no fixed bound means anything; the real checkpoint does that comparison.
- 2026-11-24: CHANGE-001 applied (task 4.8; `docs/CHANGE-001-scope-tools-cache.md`, committed with it). What was done, by its edit number:
  - 1, contract: `ATH_P_UNUSED` added to `model.h` and documented. The text types (`ath_tool_def`, `ath_msg_role`, `ath_out_kind`, `ath_out_event`) and the `chat` and `parse` members are written into `docs/ARCHITECTURE.md` as the contract target. They join `model.h` in step 5 with their first user, `text.c`, because AGENTS.md rule 9 and the step 2 decision keep members without a user out of the header, and `ath_msg`, `ath_buf` and `ath_parse_state` still need a design that step 5 settles.
  - 2 to 5, 7 to 9, 11: done as written. README has no table titled "scope", so the row went into "What it does". PROGRAM.md takes the two rows as 6b and 9b, so that the existing article numbers do not change.
  - 6: the notes went into the current step (above). The vision encoder was already explicit since step 2: an ignored group with a reason, its tensors counted by `inspect`, and any unknown tensor a load error. It was not bound, though. Now its tensors are params flagged `ATH_P_UNUSED`, and the ignored group remains as the declaration that explains them. The MTP head gets the same treatment, so that the coverage rule has no exception ("any other unbound tensor is an error"). The registry checks that each group and the unused params coincide. Results unchanged: the 320 used params and every forward number.
  - 10: the fixture generator and `tests/test_text.c` test `text.c`, so they are step 5 work, now in its preview and gate.
  - Decision 5 (ds4): the ds4 README (commit `5379c1c`, 2026-10-19) lists Qwen3.8 Flash Next, on Metal and CUDA, with its own Q2 and Q4K weights. `PLAN.md` records it next to the requested "possibly yes (verify)". It will be confirmed by running ds4 when that model enters scope (open decision 3).
- 2026-11-24: `step4-cpu-forward` merged into `main` (fast-forward), at the maintainer's request after CHANGE-001; step 5 started on `step5-tokenizer-chat`.
- 2026-11-24: step 5 implemented on branch `step5-tokenizer-chat`: `src/text/{unicode.h,unicode.c,unicode_tables.h,bpe.h,bpe.c}`, `tools/dev/gen_unicode.py`, `src/core/{buf,rng}.{h,c}` and the JSON writer in `src/core/json.c`, the text members of `src/model/model.h` with `src/model/text.c` and `ath_state_copy`, `src/models/qwen35/text.c`, `src/ops/sample.c`, `tools/chat.c`, `tests/golden/text.py`, `tests/test_text.c`. All acceptance criteria met (table above). Checked by hand as well: a three-turn conversation reuses the state (the second and third prompts run 43 of 56 and 60 of 109 tokens), thinking mode, and a tool round trip with `--tools` (the model calls `get_weather {"city": "Paris", "days": 3}`, `/tool` gives the result, the model answers from it).
- 2026-11-24: decisions taken within step 5 (`PLAN.md` unchanged):
  - Unicode data comes from two versions, as the reference does. Character classes follow Unicode 16.0 (Python 3.14's unicodedata), which the reference's regex engine (Oniguruma) also uses. NFC follows what Hugging Face tokenizers 0.23.3 applies, which is Unicode 9.0 data: `gen_unicode.py` probes the reference and leaves out 21 compositions and 29 combining classes added since. The corpus test found the difference (Todhri letters added in 16.0) and covers every code point.
  - The generated tables are `src/text/unicode_tables.h`, included by `src/text/unicode.c`; the plan named a single `unicode.h`. `unicode.h` is now the interface.
  - The tokenizer loader accepts only what it implements (BPE without unknown token, byte fallback, dropout or affixes; NFC or no normalizer; Split + ByteLevel pre-tokenizer; ByteLevel decoder; added tokens without strip or normalization) and compares the file's split regex with `pretokenize_regex`, a contract member the CHANGE-001 signature did not have, so that a checkpoint with another regex is refused rather than split wrongly.
  - The text contract as implemented: `ath_msg` has role, content, reasoning (thinking given apart from content), tool calls (name, arguments as raw JSON, an optional id) and `tool_call_id`; `chat` takes `ATH_CHAT_GENERATE` and `ATH_CHAT_THINKING` and appends to an `ath_buf`; `parse` keeps the family's state behind a pointer in `ath_parse_state`, which carries the prompt's options and tools and an arena; `piece == NULL` ends the output; event bytes live until the next call; `cap >= n + 8` events always suffice.
  - The parser drops the newlines that are only template formatting (around `</think>`, before and after a tool call, around a parameter value). Arguments become one JSON object per call, typed by the tool's schema: a string parameter stays a string; integer, number, boolean, object and array values are written as such when they parse as one, and as strings otherwise; `null` is null for non-string types. Argument bytes come out per parameter, when the parameter closes. A call whose `<function=...>` header is malformed is text; a call cut off by the end of the output is closed with what it has.
  - `ath_state_copy` (contract, host memory) lets `athanor chat` keep a copy of the state at the end of the conversation's last message: the template renders a past assistant turn differently from the prompt that generated it (Qwen3.5 drops the empty thinking block), and a recurrent state cannot be rolled back. It is the minimal form of the prefix snapshots of step 14.
  - Sampling: the filters apply in the order of transformers' logits warpers (temperature, top-k, top-p, min-p); the op takes a uniform number from the caller's generator (`ath_rng`, xoshiro256**), so it is deterministic and testable. `athanor chat` defaults to the model card's settings without the presence penalty (open questions): 0.7 / 0.8 / 20, and 0.6 / 0.95 / 20 with thinking.
  - `ath_buf`, `ath_rng` and the JSON writer are in `src/core/`; the JSON writer reproduces Python's `json.dumps(ensure_ascii=False)`, because chat templates serialize tool definitions and arguments with it.
  - `athanor chat` never runs a tool (AGENTS.md, invariant 10): it prints the call, and `/tool <text>` lets the user answer as a harness would.
  - The sampling op has a CPU reference and tests; its GPU kernel comes with GPU sampling (step 9). The reference implementation has no sampling records to compare against, so the tests are hand-computed cases and a distribution check with a fixed seed.
- 2026-11-24: `step5-tokenizer-chat` merged into `main` (fast-forward) at the maintainer's request and tagged `v0.1-cpu-chat` (the first working chat, CPU only). Two ideas for reusing the state across chat turns written down, not scheduled, at the maintainer's request: `docs/proposals/chat-state-at-last-query.md` and `docs/proposals/chat-shadow-sequence.md`. Step 6 started on `step6-gpu-backend`.
- 2026-11-24: step 6 implemented on branch `step6-gpu-backend`: `src/backend/{backend.h,gpu.cu,gpu_none.c,compat.h,devarena.c}`, `ATH_ERR_DEVICE` and C linkage in `src/core/{base,arena}.h`, `make cuda` and `tools/dev/check_nvcc.sh`, `athanor devices` (`tools/devices.c`), `tests/test_backend.c`. All acceptance criteria met (table above).
- 2026-11-24: decisions taken within step 6 (`PLAN.md` unchanged):
  - The CUDA build is chosen by naming the goal: `make cuda` (and `make cuda test`) sets `GPU=cuda` for the whole run and builds in `build/cuda`, so the CPU and CUDA builds live side by side. `CUDA_HOME` selects the toolkit (default `/usr/local/cuda`); `tools/dev/check_nvcc.sh` refuses one too old for `CUDA_ARCH`.
  - The CUDA runtime is linked statically and device code is machine code for one architecture (`sm_120`), with no PTX: the binary needs only the driver, and nothing is compiled at run time.
  - The interface is C with opaque handles; `gpu.cu` is written in the C subset of CUDA C++ with C linkage, compiled with `-fno-exceptions -fno-rtti` and every warning an error. Core headers that GPU code includes carry `extern "C"` guards.
  - A build without a GPU links `gpu_none.c` (no devices, `ATH_ERR_UNSUPPORTED`) instead of `#ifdef`s in the callers.
  - A new status, `ATH_ERR_DEVICE`, for errors of the GPU runtime; the runtime's sticky last error is cleared after each failure.
  - Every call selects the device its handle names, because the runtime keeps one current device per host thread.
  - Device memory comes in blocks through `ath_dev_arena` (records in a host arena, since the host cannot write device memory).
  - ASan and the NVIDIA driver need `ASAN_OPTIONS=protect_shadow_gap=0`; the Makefile sets it for the CUDA debug tests.
  - Not in the plan's file list: `athanor devices`, the visible part of the step and a check of the timing path. Deferred to their first users: graph capture (step 9), the cuBLASLt handle (step 7), `src/backend/cpu.c` (CPU feature detection).
  - The HIP half of `compat.h` is written but not compiled until step 11.
- 2026-11-25: `step6-gpu-backend` merged into `main` (fast-forward) at the maintainer's request; step 7 started on `step7-gpu-ops`.
- 2026-11-25: step 7 implemented on branch `step7-gpu-ops`: the cuBLASLt handle and workspace in `ath_device` and `src/backend/gpu.h`; `ath_<op>_gpu` in `src/ops/ops.h`, the shared argument checks (`ops_place` in `check.h`), `src/ops/{launch.h,kernel.cuh,gpu_none.c}` and the kernels `src/ops/{embed,matmul,norm,rope,act,attn,conv,gdn}.cu`; `tests/test_gpu_ops.c` and the GPU replay in `tests/test_golden.c`. All acceptance criteria met (table above).
- 2026-11-25: decisions taken within step 7 (`PLAN.md` unchanged):
  - The GPU entry points are explicit, `ath_<op>_gpu(const ath_device *d, ...)`, beside `_ref` and `_fast`. The entry point that chooses among them from where the tensors live waits for its first caller, the GPU forward pass of step 8 (AGENTS.md rule 9), instead of landing with the kernels as the step 3 notes planned.
  - A `_gpu` function checks its arguments with the same code as `_ref`, then calls a launcher in the `.cu` file. Without a GPU, `src/ops/gpu_none.c` stubs the launchers, so the checks run and are tested in both builds.
  - Ops run on the device's compute stream, asynchronously; one cuBLASLt workspace per device follows from that.
  - cuBLASLt is linked as a shared library through the toolkit's path (`-rpath`): statically it adds about 560 MB to every binary (measured). The CUDA build's `ldd` gains `libcublasLt.so.13`, which invariant 3 allows; the CPU build is unchanged.
  - cuBLASLt serves bf16·bf16 (into bf16 or f32) and f32 throughout, with fp32 accumulation and never TF32; athanor's own kernel serves the other dtype combinations, which only the exact f32-activation forward pass uses.
  - Kernels take dtypes as arguments and branch on them (uniform across a launch); per-dtype and per-shape specializations come with the fast paths of steps 9 and 10. No templates: the C subset of CUDA C++, as `gpu.cu`.
  - RoPE frequencies are computed on the host by the reference's expression and passed to the kernel, so long positions do not amplify a difference between the device's `powf` and libm's.
  - The embedding kernel cannot check ids on the device: an id outside the table writes NaN instead of reading outside it, and `ath_model_forward` keeps checking token ids on the host.
  - Kernels are compiled with `-fno-threadsafe-statics`, because the launch stubs nvcc writes would otherwise need libstdc++; `.cu` files compile to `.cu.o`, since a kernel file shares its op's name.
  - The GPU kernels are compared with `_ref` on the same inputs, in `test_gpu_ops` (random) and in `test_golden` (every record), with tolerances at least four times the worst error measured; bf16 outputs within one ulp (two where the op rounds an intermediate value). The step 3 tolerances against the reference implementation are unchanged.
  - Core headers that GPU code includes through `ops.h` (`tensor.h`, `threads.h`) carry `extern "C"` guards, like `base.h` and `arena.h`.
