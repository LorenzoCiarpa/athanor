# Course program

Maps each article and lesson of the course to the athanor code that implements it. Updated when a component lands (see AGENTS.md). Articles are written in English on the blog; this file is the map, not the explanation.

| # | Article / lesson | Step | Code | Status |
| --- | --- | --- | --- | --- |
| 1 | Reading a model from disk: safetensors, mmap and zero-copy | 1 | `src/core/file.c`, `safetensors.c`, `json.c`, `arena.c` | code landed (step 1) |
| 2 | What is inside a Qwen checkpoint | 1–2 | `tools/inspect.c`, `src/models/qwen35/weights.c`; tensor groups in `docs/contract/qwen35-0.8b.md` | inspect landed (step 1); roles in step 2 |
| 3 | The forward pass in plain C: embeddings, norms, attention, MLP | 3–4 | `src/ops/*.c`, `src/models/qwen35/forward.c` | planned |
| 4 | Linear attention and Gated DeltaNet, from the recurrence | 3–4 | `src/ops/gdn.c` | planned |
| 5 | Byte-level BPE from scratch | 5 | `src/text/bpe.c` | planned |
| 6 | Sampling: temperature, top-k, top-p | 5 | `src/ops/sample.c` | planned |
| 7 | How a GPU runs a kernel; one source for CUDA and HIP | 6–7 | `src/backend/`, `src/ops/*.cu` | planned |
| 8 | Why decode is memory-bound | 9 | `src/ops/matmul.cu` | planned |
| 9 | KV cache vs recurrent state | 8 | `src/infer/state.c` | planned |
| 10 | Chunkwise Gated DeltaNet on the GPU | 10 | `src/ops/gdn.cu` | planned |
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
