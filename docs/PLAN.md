# Plan

Order of work: inference first, because every later stage reuses the forward pass; quantization second, because abliteration and training consume and produce quantized weights; abliteration third, a forward-only pipeline plus weight editing; training last, because it needs a backward pass for every kernel.

Within inference, the first five steps run on the CPU only. They produce a slow but obviously correct reference implementation that every GPU kernel is later checked against, and they are the first chapter of the course.

## 1. Decisions

| Decision | Choice | Status |
| --- | --- | --- |
| Name | athanor; binary `athanor`; prefix `ath_`; container extension `.ath` | decided |
| Languages | C11 host code; CUDA C kernels compiled by nvcc and, through `compat.h`, by hipcc | decided |
| Source layout | by layer, then by language: portable C at the root of a module, CUDA C in its `cuda/` folder (one source for nvcc and hipcc); `hip/` for kernels whose AMD form cannot share the source and `metal/` for a Metal port, only if needed (`src/backend/README.md`, Source layout). Pipelines hold no GPU code: a kernel they need is an op (AGENTS.md invariant 11) | decided 2026-11-25, before step 9, at the maintainer's request |
| Dependencies | libc, libm, pthreads, POSIX, CUDA toolkit, ROCm; Python only for golden data | decided |
| License | MIT | decided |
| Model family | Qwen hybrid line (Qwen3.5 / 3.6 / 3.8) | decided |
| First model | Qwen3.5-0.8B, directly (no classic Qwen3 bring-up) | default, revisit if blocked |
| Dense GEMM | cuBLASLt / hipBLASLt behind `ops/matmul`; own kernels for GEMV, quantized and fused paths; replace vendor GEMM only where a benchmark shows a gain | decided at step 10 (2026-11-25): cuBLASLt kept, with its heuristic's first algorithm. Measured: its prompt GEMMs run at 106 to 145 TFLOPS on the RTX 5080; timing its candidates would gain 0 to 9% per shape but make the bits depend on the timings; adding the residual in its epilogue changes the rounding (`src/ops/README.md`, Prompt kernels) |
| Build system | plain Makefile | default |
| Binary | one `athanor` binary with subcommands | decided |
| Docs language | English | decided |
| Scope | engine + server; no agent harness (no tool execution, agent loop, conversation memory or UI; tool rendering and parsing are in scope). CHANGE-001, 2026-11-24 | decided |
| Step 12 weights | Q8_0, read from a GGUF file (Unsloth's `Qwen3.6-35B-A3B-Q8_0.gguf` with the MTP layer, the file Strata uses): a GGUF reader for F32, F16, BF16 and Q8_0 in `src/core`, and Q8_0 as the first quantized block format. The other formats (Q1) and GGUF import and export (Q6) stay in phase 2 | decided 2026-11-25, at the start of step 12, at the maintainer's request (`docs/STATUS.md`, step 12 decisions) |
| Invariant 7 across devices | the CPU and GPU expert paths give the same bits: one canonical arithmetic order for routed experts (AGENTS.md invariant 7, option (b) of the step 7 open question) | decided 2026-11-25, at the start of step 12, at the maintainer's request |
| Step 14 weights and KV | Qwen3.5-9B from Unsloth's Q8_0 GGUF (its bf16 weights do not fit 16 GB); an int8 KV cache with a scale per token and head, beside bf16 (`docs/STATUS.md`, step 14 decisions); refined after measuring it to a scale per 32 values of a row (a scale per row cost 2.4 times the bf16 noise) | decided 2026-11-26, at the start of step 14, at the maintainer's request; refined the same day |
| Order after phase 1 | after step 14, before phase 2: a phase of analysis and optimization of everything built, measured against the other engines on this machine (vLLM, Strata, Bonsai and others) until athanor is as fast or faster; then larger models; then streaming weights from disk, with the larger models. Phase 2 (quantization) and the later phases follow | decided 2026-11-26, at the maintainer's request |

## 2. Assumptions

| Item | Assumption | Status |
| --- | --- | --- |
| Development GPU | NVIDIA RTX 5080, 16 GB, Blackwell (`sm_120`) | confirmed |
| AMD GPU | none yet; needed from step 11 | open: a second card with more VRAM (24–32 GB Radeon) would cover AMD support and larger models |
| System RAM | 64 GB DDR5 | to confirm |
| CPU | AM5 platform (Zen 4 / Zen 5, AVX-512) | to confirm the exact model |
| OS | Linux first | confirmed |

## 3. Model ladder

Weight memory, weights only (estimates; about 4.5 effective bits for 4-bit formats):

| Model | Type | bf16 | ~4-bit | Role |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B | dense hybrid | 1.6 GB | 0.5 GB | CPU reference, first GPU model; full FT, pre-training configs, abliteration and quant experiments |
| Qwen3.5-2B | dense hybrid | 4 GB | 1.1 GB | main teaching model; full FT with 8-bit optimizer |
| Qwen3.5-4B | dense hybrid | 8 GB | 2.3 GB | QLoRA; full FT with hot/cold layout |
| Qwen3.5-9B | dense hybrid | 18 GB | 5 GB | headline: 4-bit inference in VRAM; full FT on 16 GB with hot/cold |
| Qwen3.6-27B / Qwen3.8-27B | dense hybrid | 54 GB | 15 GB | inference with partial offload |
| Qwen3.6-35B-A3B | MoE hybrid, 3B active | 70 GB | 20 GB | MoE step; direct comparison with Strata |
| Qwen3.5-122B-A10B | MoE hybrid | 244 GB | 69 GB | streaming from RAM and disk (stretch) |
| Qwen3.8-Flash-Next | MoE 125B, 6B active, extra blocks | — | — | separate milestone |

Training memory on 16 GB, weights plus optimizer, before activations:

| Model | Standard mixed-precision AdamW (16 B/param) | athanor plan |
| --- | --- | --- |
| 0.8B | 12.8 GB, tight | 8-bit AdamW or bf16 master with stochastic rounding: about 5 GB |
| 2B | 32 GB, does not fit | 8-bit states + bf16 master: about 12 GB, or hot/cold |
| 4B / 9B | 64 / 144 GB | hot/cold: cold blocks in 4-bit in VRAM, bf16 master in RAM, one hot block with full AdamW |

## 4. Phase 0–1: foundations and inference

| Step | Deliverable | Main files | Gate |
| --- | --- | --- | --- |
| 1 | Core and `athanor inspect` | `src/core/*`, `tools/athanor.c`, `Makefile` | lists every tensor of Qwen3.5-0.8B with dtype, shape and size; `make test` green |
| 2 | Contract and family loading | `src/model/model.h`, `registry.c`, `src/models/qwen35/config.c`, `weights.c` | every tensor bound to a param with role, layer and flags; layer schedule printed; vision-encoder tensors bound with `ATH_P_UNUSED`; any other unbound tensor is an error (added by CHANGE-001, 2026-11-24) |
| 3 | CPU reference ops and golden harness | `src/ops/*.c`, `src/backend/cpu.c`, `tests/golden/dump.py`, `tests/test_ops.c` | every op within tolerance of golden tensors on real weights |
| 4 | CPU forward | `src/models/qwen35/forward.c` | logits match the reference implementation on fixed prompts |
| 5 | Tokenizer, sampling, CPU chat | `src/text/bpe.c`, `unicode.h`, `src/models/qwen35/text.c` (`pretokenize`, `chat` with tool definitions, `parse`), `src/ops/sample.c` | exact token round-trip on a corpus; first chat, slow and correct; rendered prompts byte-identical to the reference chat template for a fixture set with and without tools; parser fixtures pass (added by CHANGE-001) |
| 6 | GPU backend | `src/backend/gpu.cu`, `compat.h` (in `src/backend/cuda/` since 2026-11-25) | allocation, copies, streams, events, timing on CUDA |
| 7 | GPU ops, one by one | `src/ops/*.cu` (in `src/ops/cuda/` since 2026-11-25) | every kernel matches its CPU reference |
| 8 | Sequence state and engine | `src/infer/state.c`, `engine.c` | GPU chat in bf16; greedy output identical to reference for 256 tokens on 50 prompts |
| 9 | Fast decode and bench | fused GEMV, graph capture, GPU sampling, `athanor bench` | decode tok/s ≥ llama.cpp at batch 1, same model, precision and GPU |
| 10 | Fast prefill | tiled attention, chunked Gated DeltaNet, GEMM path | prefill within 15% of vLLM for a single request |
| 11 | HIP build | Makefile `hip` target, `compat.h` coverage | all tests green on AMD; performance report published. Postponed 2026-11-25 at the maintainer's request until AMD hardware is available (§2); step 12 goes first |
| 12 | MoE and tiering | `src/ops/moe.*`, `src/infer/tier.c`, CPU `_fast` expert path | Qwen3.6-35B-A3B decode ≥ Strata on the same machine and quantization |
| 13 | Server and MTP | `src/net/*`, `src/infer/spec.c` | works with Claude Code, Codex CLI and OpenCode; speculative-decoding speedup reported; tool-calling conformance: multi-turn sessions with tool calls succeed with Claude Code, OpenCode and Codex CLI on a fixed task set (added by CHANGE-001) |
| 14 | Long context | KV quantization, state snapshots for prefix reuse; persistent prefix snapshots on disk keyed by a hash of the token prefix (added by CHANGE-001) | 128k context on Qwen3.5-9B in 16 GB; resume after restart; snapshot size reported per context length |

Steps 1–5 need no GPU. Steps 1–9 use bf16 weights and one simple 8-bit format at most; quantization proper is phase 2.

## 5. Phase 2: quantization

| Milestone | Deliverable | Gate |
| --- | --- | --- |
| Q1 Formats | block formats 2–8 bits with scales, FP8 e4m3 block, NVFP4, MXFP4; `.ath` container | dequant round-trip tests; GEMV and GEMM per format on CUDA and HIP |
| Q2 Importance RTN | rounding weighted by activation statistics from calibration | KL vs bf16 ≤ llama.cpp imatrix quant of equal size |
| Q3 GPTQ / AWQ class | layer-by-layer PTQ on 16 GB with streamed calibration | KL and perplexity ≤ GGUF of equal size |
| Q4 Mixed precision | per-tensor format choice under a size budget, solved as a knapsack over measured damage | Pareto curve beats fixed recipes |
| Q5 Blackwell FP4 | NVFP4 tensor-core GEMM on `sm_120` | prefill speedup with KL reported |
| Q6 Interop | GGUF import and export | round-trip with llama.cpp |

## 6. Phase 3: abliteration

| Milestone | Deliverable | Gate |
| --- | --- | --- |
| A1 Capture | residual capture through `on_residual`, streamed to disk | per-layer means for two prompt sets on Qwen3.5-2B |
| A2 Directions | difference of means, projected variant, multi-direction (SVD), layer selection | reproduces published directional-ablation results |
| A3 Editing | orthogonalize every `ATH_P_WRITES_RESIDUAL` param, including every expert; norm-preserving option | edited model passes structural checks |
| A4 Search | multi-objective search (refusal rate, KL) over layer range, strength, direction | Pareto front; ≤ Heretic on both metrics on the same model |
| A5 Eval and export | refusal metric, KL, capability suite (math first); export bf16, `.ath`, GGUF; model card section | report from `athanor ablit eval` |

## 7. Phase 4: training

| Milestone | Deliverable | Gate |
| --- | --- | --- |
| T1 Backward | backward for every op, chunked cross-entropy fused with lm_head | gradient checks vs reference on every op |
| T2 LoRA | LoRA on bf16 base, 0.8B and 2B | loss curve matches a reference trainer |
| T3 QLoRA | frozen base in athanor 4-bit formats | VRAM and step time ≤ Unsloth on Qwen3.5-4B |
| T4 Full FT small | AdamW, 8-bit AdamW, Muon; bf16 master with stochastic rounding | full FT of Qwen3.5-2B on 16 GB |
| T5 Hot/cold full FT | cold blocks quantized in VRAM, master in RAM, one hot block with full AdamW | full FT of Qwen3.5-9B on 16 GB, quality reported |
| T6 Memory planner | per-layer precision, recomputation, offload and grouping as a small mixed-integer program | never OOMs; within 10% of best manual config |
| T7 QAT | quantization-aware fine-tuning, NVFP4 matmuls in training | beats PTQ of the same format on KL |
| T8 Pre-training | 100M–1B configs of the same architecture from scratch | reproducible loss curve on a public dataset |
| T9 MoE fine-tuning | LoRA on attention, block-wise expert updates with tiering | Qwen3.6-35B-A3B fine-tune on one GPU + RAM |
| T10 RL | GRPO with verifiable rewards, rollouts from the engine | reward improves on a math task with a 2B model |

## 8. Performance references

| Reference | Meaning for athanor | Same model? |
| --- | --- | --- |
| Strata | MoE inference with VRAM/RAM/SSD placement | yes: Qwen3.6-35B-A3B, Qwen3.8-Flash-Next |
| DwarfStar (ds4) | design quality: specialization, per-component quantization, small C code | possibly yes on Qwen3.8-Flash-Next (verify); otherwise compared through llama.cpp. Checked 2026-11-24: the ds4 README (commit `5379c1c`, 2026-10-19) lists Qwen3.8 Flash Next on Metal and CUDA, with its own Q2 and Q4K weights; still to verify by running it when that model enters scope |
| Unsloth | training memory and speed | yes: Qwen3.5 sizes |
| llama.cpp, vLLM | decode and prefill baselines | yes |
| Heretic | abliteration quality and time | yes |

Gates are measured, not promised; every comparison records the competitor's version.

## 9. Risks

| Risk | Mitigation |
| --- | --- |
| Gated DeltaNet kernels (chunked forward; backward for training) are the hardest code | the recurrent CPU reference from step 4 is the golden path; port the chunked form from flash-linear-attention and llama.cpp as references |
| The family moves fast | block library per layer type; new releases add ops, not rewrites |
| No AMD hardware | second-GPU decision; cloud runner as fallback |
| One-person bandwidth | narrow scope, published gates, contributor-friendly issues per step |
| Qwen3.5 models are multimodal | text path only; vision encoder is a later milestone; its tensors are bound with `ATH_P_UNUSED`, never silently ignored (CHANGE-001) |
| Name shared with unrelated projects (AI dev tools, MU* frameworks) | clear tagline and repository description |

## 10. Open decisions

1. Second GPU: AMD with more VRAM, or a second NVIDIA.
2. Primary on-disk format: `.ath` only, or GGUF as an equal citizen.
3. When Qwen3.8-Flash-Next enters scope.
