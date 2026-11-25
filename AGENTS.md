# AGENTS.md

Instructions for coding agents and human contributors. Read this file completely before changing anything.

## The project in one paragraph

athanor is a single C engine that runs, quantizes, abliterates and trains one family of language models (the Qwen hybrid line) on consumer NVIDIA and AMD GPUs. It is deliberately minimal: C11 host code, CUDA/HIP C kernels, no dependencies beyond the CUDA and ROCm toolkits and libc. Correctness comes before speed, and the code is also teaching material for a public blog and video course.

## Reading order

1. `AGENTS.md` (this file)
2. `docs/STATUS.md`: the current step and its task list. **Work only on the current step.**
3. `docs/ARCHITECTURE.md`: layers, the model contract, where code goes
4. `docs/PLAN.md`: decisions, steps, gates
5. The `README.md` of every module you touch, and `docs/MODELS.md` when touching model math

## Invariants (never break these)

1. **Dependencies point down only**: `tools → pipelines (infer, quant, ablit, train, net) → model contract → models/<family> → ops → backend → core`. Pipelines never include anything from `src/models/`. Families never include pipeline headers.
2. **No model names outside `src/models/`.** No `if (qwen)` anywhere else. If a pipeline needs model-specific information, extend the contract in `src/model/model.h` (a new role, flag, state kind or hook) and document it in `docs/ARCHITECTURE.md`.
3. **No external libraries.** Allowed: libc, libm, pthreads, POSIX, the CUDA toolkit (runtime, cuBLASLt) and ROCm (HIP runtime, hipBLASLt). Anything else requires an explicit decision recorded in `docs/PLAN.md`. Python is allowed only under `tests/golden/` and `tools/dev/` and never at runtime.
4. **Every op has three parts**: a CPU reference in `src/ops/<op>.c`, a GPU kernel in `src/ops/cuda/<op>.cu`, and a test in `tests/` that compares them, plus golden data where available. No op lands without all three (the GPU kernel may arrive in a later step, but the reference and test come first).
5. **Never guess model math.** Dimensions, eps, rope parameters, head counts and layer types come from `config.json` and from `docs/contract/<model>.md`. If the contract says **verify**, check the reference implementation, record what you found in the contract file, then write code.
6. **Numerics are protected.** A change that alters outputs must state why, keep golden tests green or update tolerances with a written justification, and report KL vs bf16 before and after.
7. **Placement never changes results.** Where a tensor lives (VRAM, RAM, disk) must not change outputs.
8. **No performance claims without a report** under `docs/results/` following `docs/BENCHMARKS.md`.
9. **Minimal by default.** No abstraction until it has two real users. No speculative features outside the current step. Prefer one clear function over a framework.
10. **No agent features.** athanor never executes tools, runs agent loops, stores conversation memory or compacts context. Requests in that direction are out of scope; note them under "Open questions" in `docs/STATUS.md` instead of implementing them. Rendering tool definitions into the prompt and parsing tool calls out of the output are in scope (`docs/ARCHITECTURE.md`, Scope: engine versus harness).
11. **Pipelines contain no GPU code.** A pipeline (`src/infer`, `src/quant`, `src/ablit`, `src/train`, `src/net`) uses the GPU only through `backend.h` (devices, memory, copies, streams, graphs) and `ops.h` (computation): no `.cu` file, no kernel, no vendor call. A kernel a pipeline needs (an optimizer step, a quantization algorithm) is an op, with its CPU reference and test (invariant 4); a runtime service (graph capture) belongs to the backend. GPU code lives in `src/backend/cuda/`, `src/ops/cuda/` and `src/models/<family>/cuda/` only, so that a port to new hardware is the content of those folders.

## Workflow for every task

1. Read the task in `docs/STATUS.md`, including its acceptance criteria.
2. Implement the smallest complete slice. Keep the build warning-free (`-Wall -Wextra -Werror`).
3. Add or update tests; run `make test` (and `make cuda test` when GPU code changed).
4. Update documentation in the same change:
   - `docs/STATUS.md`: tick the task, add a dated line to the log, note open questions.
   - `docs/status_extended.md`: under the task, the files created or changed, what each is for and what was implemented in it.
   - The module `README.md` if a file or public interface changed.
   - `docs/ARCHITECTURE.md` if the contract or the tree changed.
   - `docs/contract/<model>.md` if you confirmed a **verify** item.
   - `docs/program/PROGRAM.md` when a component that an article explains has landed: link the article slot to the code paths.
5. Commit with a message `<module>: <what>` (e.g. `core: safetensors multi-shard loader`).

Never rewrite history in `STATUS.md`, `PLAN.md` or `PROGRAM.md`: mark items done, superseded or dropped.

## Checklists

### Add an op

- [ ] Signature in `src/ops/ops.h`, documented (shapes, dtypes, layout)
- [ ] `<op>_ref` in `src/ops/<op>.c`: scalar, fp32 accumulation, obviously correct, never optimized
- [ ] Tests vs hand-computed values in `tests/test_ops.c`, and vs golden data: a record in `tests/golden/dump.py`, a handler and a justified tolerance in `tests/test_golden.c` (`tests/README.md`)
- [ ] GPU kernel in `src/ops/cuda/<op>.cu`, compiled for CUDA and HIP through `src/backend/cuda/compat.h`
- [ ] An operator that only a pipeline uses (an optimizer step, a quantization update) follows the same checklist (invariant 11)
- [ ] Test GPU vs CPU reference on random and real inputs
- [ ] Optional `<op>_fast` CPU path only if the op runs on CPU at runtime (MoE experts), tested vs `_ref`

### Add a model size of an existing family

- [ ] Add it to the golden dump list and run `tests/golden/dump.py`
- [ ] End-to-end test passes; no code changes expected. If code changes are needed, the family config parsing is incomplete: fix it there.

### Add a model family

- [ ] `src/models/<family>/` with `config.c`, `weights.c`, `forward.c`, `text.c` (and later `backward.c`)
- [ ] Define `const ath_family ath_family_<family>` and add the family to the Makefile `FAMILIES` list (the registry is generated from it; `src/model/registry.c` needs no edit)
- [ ] `docs/contract/<model>.md` with the op contract, sourced from the reference implementation
- [ ] Missing ops added to `src/ops/` following "Add an op"
- [ ] Zero changes under `src/infer`, `src/quant`, `src/ablit`, `src/train`. If one is needed, extend the contract instead.

### Add a subcommand

- [ ] Entry in `tools/athanor.c` dispatch table, `--help` text, README in `tools/`

## Code style

| Topic | Rule |
| --- | --- |
| Language | C11 for host; CUDA C (C-style subset of CUDA C++) for kernels |
| Naming | public functions `ath_<module>_<verb>`; types `ath_<name>`; macros `ATH_<NAME>`; `snake_case` everywhere |
| Errors | functions return `ath_status`; check with `ATH_CHECK(expr)`; no `exit()` or `abort()` in library code |
| Memory | allocate through `src/core` arenas and tier allocators; no `malloc` in hot paths |
| Globals | none, except the static family registry and constant tables |
| Headers | one header per module; include what you use; no circular includes |
| Files | one concept per file; split a file that passes about 800 lines |
| Source layout | by layer, then by language: portable C at the root of a module, GPU code in a folder named after its language (`cuda/`: CUDA C, also compiled by hipcc; `hip/` or `metal/` only if ever needed). `src/backend/README.md`, Source layout |
| Comments | explain *why* and the math; reference the contract section for model ops |
| Docs and code language | English |

## Definition of done

The task's acceptance criteria pass, `make test` is green on every backend touched, the build has no warnings, `ldd build/athanor` shows no unexpected libraries, and the documentation listed in the workflow is updated.

## When unsure

Do not guess. Write the question under "Open questions" in `docs/STATUS.md`, explain the options, and stop or continue with a task that does not depend on the answer.
