# Course program

Maps each article and lesson of the course to the athanor code that implements it. Updated when a component lands (see AGENTS.md). Articles are written in English on the blog; this file is the map, not the explanation.

| # | Article / lesson | Step | Code | Status |
| --- | --- | --- | --- | --- |
| 1 | Reading a model from disk: safetensors, mmap and zero-copy | 1 | `src/core/file.c`, `safetensors.c`, `json.c`, `arena.c` | code landed (step 1) |
| 2 | What is inside a Qwen checkpoint | 1–2 | `tools/inspect.c` (`--roles`), `src/model/model.h`, `src/model/registry.c`, `src/models/qwen35/config.c`, `weights.c`; tensor groups, op contract and parameter binding in `docs/contract/qwen35-0.8b.md` and `qwen35-35b-a3b.md` | inspect landed (step 1); roles landed (step 2) |
| 3 | The forward pass in plain C: embeddings, norms, attention, MLP | 3–4 | `src/ops/*.c` (`embed`, `matmul`, `norm`, `rope`, `act`, `attn`), `src/models/qwen35/forward.c`, `ath_model_forward` in `src/model/state.c`; checking each op against the reference: `tests/golden/dump.py`, `tests/test_golden.c`, tolerances in `tests/README.md`; checking the whole pass (KL against an fp32 run, greedy agreement, an exact f32 run): `tests/test_forward.c`, `tests/README.md` (Forward pass), `docs/contract/qwen35-0.8b.md` (End-to-end); making it usable without changing a bit: `ath_matmul_fast` in `src/ops/matmul.c` (independent chains, `src/ops/README.md`, Fast CPU paths), the pool in `src/core/threads.c`, the variant not taken in `docs/proposals/cpu-matmul-avx512.md` | CPU reference ops landed (step 3); forward landed (step 4); fast CPU matmul landed (step 4, task 4.7) |
| 4 | Linear attention and Gated DeltaNet, from the recurrence | 3–4 | `src/ops/gdn.c`, `src/ops/conv.c`, `ath_l2norm_ref` in `src/ops/norm.c`; the layer as composed: `gated_deltanet()` in `src/models/qwen35/forward.c` | recurrent reference landed (step 3); composed in the forward pass (step 4) |
| 5 | Byte-level BPE from scratch | 5 | `src/text/bpe.c` (loading `tokenizer.json`, added tokens, merges by rank); `src/text/unicode.c` and `tools/dev/gen_unicode.py` (classes, NFC, and why the reference uses two Unicode versions: `src/text/README.md`); the regex as a hand-written split: `ath_qwen35_pretokenize` in `src/models/qwen35/text.c`; checked on 28 million tokens: `tests/golden/text.py`, `tests/test_text.c` | landed (step 5) |
| 6 | Sampling: temperature, top-k, top-p | 5 | `src/ops/sample.c` (`src/ops/README.md`, Sampling), `src/core/rng.c`; the loop that uses it: `tools/chat.c` | CPU landed (step 5); GPU planned (step 9) |
| 6b | Tool calling: from chat template to API | 5, 13 | `src/models/qwen35/text.c` (`chat` with tool definitions, `parse`), `src/net/api.c` | template and parser landed (step 5): the markup in `docs/contract/qwen35-0.8b.md` (Chat template, Output markup), the contract in `docs/ARCHITECTURE.md` (Text), a round trip by hand in `athanor chat --tools`; API planned (step 13) |
| 7 | How a GPU runs a kernel; one source for CUDA and HIP | 6–7 | `src/backend/` (`backend.h`: devices, memory, copies, streams, events; `gpu.cu`; `compat.h` for HIP; `README.md`), `athanor devices --bench`; kernels in `src/ops/*.cu` (`src/ops/README.md`, GPU kernels): threads, blocks and grid-stride loops (`act.cu`, `embed.cu`), a block per row and its reduction (`norm.cu`, `k_block_sum` in `kernel.cuh`), warps and shuffles (`gdn.cu`), shared memory and the online softmax (`attn.cu`), a vendor library on the checkpoint's layout (`matmul.cu`); checking a kernel against its reference: `tests/test_gpu_ops.c`, the GPU replay in `tests/test_golden.c`, `tests/README.md` (GPU kernels) | runtime landed (step 6); kernels landed (step 7) |
| 8 | Why decode is memory-bound | 4, 9 | `src/ops/matmul.cu` (cuBLASLt since step 7; own GEMV in step 9); on the CPU first: `ath_matmul_fast` reaches the RAM bandwidth on the lm_head with 8 threads, where every faster instruction stops helping (`src/ops/README.md`, Fast CPU paths; `docs/proposals/cpu-matmul-avx512.md`, Measurements) | CPU illustration landed (step 4); GPU planned (step 9) |
| 9 | KV cache vs recurrent state | 4, 8 | state specs in `src/model/model.h`, `src/model/state.c`; the buffers of Qwen3.5 in `src/models/qwen35/forward.c` (layouts in its README); `src/infer/state.c` | CPU state of one sequence landed (step 4); GPU engine planned (step 8) |
| 9b | Prefix caching on disk with hybrid models | 14 | `src/infer/state.c` | planned (added by CHANGE-001, 2026-11-24) |
| 10 | Chunkwise Gated DeltaNet on the GPU | 10 | `src/ops/gdn.cu` | recurrent kernel landed (step 7: a warp per column of the state); chunkwise form planned (step 10) |
| 11 | Mixture of experts and placing experts across VRAM, RAM and disk | 12 | `src/ops/moe.*`, `src/infer/tier.c` | planned |
| 12 | Speculative decoding with an MTP head | 13 | `src/infer/spec.c` | planned |
| 13 | Number formats from bf16 to FP4 | Q1 | `src/quant/formats.c` | planned |
| 14 | RTN, GPTQ, AWQ and measuring quantization error | Q2–Q3 | `src/quant/` | planned |
| 15 | Mixed precision as a knapsack problem | Q4 | `src/quant/recipe.c` | planned |
| 16 | The refusal direction and abliteration | A1–A5 | `src/ablit/` | planned |
| 17 | Backpropagation by hand | T1 | `src/models/qwen35/backward.c` | planned |
| 18 | Optimizers and their memory | T4 | `src/train/optim.c` | planned |
| 19 | LoRA and QLoRA | T2–T3 | `src/train/lora.c` | planned |
| 20 | Training beyond VRAM: the hot/cold layout | T5 | `src/train/hotcold.c` | planned |
| 21 | Planning memory with mixed-integer programming | T6 | `src/train/planner.c` | planned |
| 22 | RL with verifiable rewards | T10 | `src/train/` | planned |
