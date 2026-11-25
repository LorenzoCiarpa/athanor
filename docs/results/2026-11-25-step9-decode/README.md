# Decode at batch 1: athanor step 9 against llama.cpp

The report of step 9's gate (`docs/PLAN.md`): decode tokens per second at batch 1, at least llama.cpp's, on the same model, precision and GPU. Measured on 2026-11-25, one tool at a time on an otherwise idle machine, in this order: athanor step 9, llama.cpp, athanor step 8.

## Result

| test | athanor step 9: median (min–max) | llama.cpp: mean ± sd | athanor / llama.cpp | athanor step 8: median |
| --- | --- | --- | --- | --- |
| **tg128 @ depth 0** | **441.0** (439.4–443.4) | 438.5 ± 0.4 | **1.006** | 287.2 |
| tg128 @ depth 4096 | 393.0 (392.2–393.3) | 428.7 ± 0.8 | 0.917 | 59.2 |
| tg128 @ depth 32768 | 266.5 (266.2–266.6) | 360.4 ± 0.6 | 0.740 | 8.6 |
| pp512 | 17,604 (17,297–17,609) | 45,279 ± 1,900 | 0.389 | 19,840 |
| pp4096 | 8,553 (8,550–8,559) | 45,373 ± 1,936 | 0.188 | 10,585 |

Tokens per second. tgN @ depth D: N tokens decoded one at a time after a prompt of D tokens. ppN: a prompt of N tokens.

**The gate passes**, narrowly: at depth 0 athanor decodes 441.0 tokens/s against llama.cpp's 438.5, and athanor's slowest run (439.4) is above llama.cpp's mean plus one standard deviation. On a long cache athanor is behind (0.92 at 4k, 0.74 at 32k), and its prefill is far behind (step 10's subject). Step 8's decode was 287 tokens/s at depth 0 and fell to 8.6 at 32k; step 9 made it 1.5, 6.6 and 31 times faster.

## Setup

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5080, 16 GB, `sm_120`, PCIe 5.0 x16 (32 GT/s) |
| Driver | 580.178.04 (CUDA 13.0) |
| CPU, RAM | AMD Ryzen 7 9800X3D (8 cores), 59 GiB usable; RAM speed not recorded (it needs root) |
| Storage | Crucial P310 500 GB (the weights are read once, then held in VRAM) |
| athanor | commit `7f0f3a6` (`git describe`: `v0.1-cpu-chat-27-g7f0f3a66e445`), `make cuda`, CUDA toolkit 13.5 (runtime 13050), cuBLASLt 13.10, gcc 15.2 |
| llama.cpp | commit `5220a2d` (2026-11-25), built with `-DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release`, the same CUDA 13.5 toolkit |
| Model | Qwen/Qwen3.5-0.8B, revision `2fc06364715b967f1860aea9cf38778875588b17`, `model.safetensors-00001-of-00001.safetensors` sha256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| llama.cpp's model | the same checkpoint converted with llama.cpp's `convert_hf_to_gguf.py --outtype bf16` (1.44 GiB, 772.85 M parameters: the text model) |
| Precision | bf16 weights as stored and bf16 activations for both; KV cache bf16 for athanor, f16 for llama.cpp (its default) |

## Method

- **athanor**: `athanor bench models/Qwen3.5-0.8B -p 512,4096 -n 128 -d 0,4096,32768 -r 5 --json athanor.json`. One unmeasured run, then 5 measured runs per test, each from a reset state; the median, with the minimum and maximum. Token ids pseudo-random (seed 1). Decode runs through the decoder (`src/infer/README.md`): one recorded graph per token, the greedy choice made on the GPU and the token copied back, so each measured token includes its sampling and the 4-byte copy. The prompt before a tg test is finished before the clock starts. Prompts are fed in pieces of 2048 tokens.
- **llama.cpp**: `llama-bench -m Qwen3.5-0.8B-bf16.gguf -ngl 99 -fa 1 -p 512,4096 -n 0 -r 5` and `-p 0 -n 128 -d 0,4096,32768 -r 5`, with `-o json -oe md`: every layer on the GPU and flash attention, as its CUDA documentation recommends for this GPU, other settings at their defaults (`n_batch` 2048, `n_ubatch` 512, 8 threads). llama-bench reports the mean and the standard deviation of its 5 runs; its tg runs decode without sampling.
- **athanor step 8**: the same `athanor bench` command on commit `57f95dd` (step 8's engine with the first version of `athanor bench`), with the fix below, and `--no-hash`.

## Files

| File | Content |
| --- | --- |
| `athanor.json`, `athanor.log` | athanor step 9: the report of `docs/BENCHMARKS.md` (hardware, software, model files and hashes, workload, method, every run, memory) and the console output |
| `llama-bench-pp.json`, `llama-bench-pp.log`, `llama-bench-tg.json`, `llama-bench-tg.log` | llama.cpp: llama-bench's JSON (stdout) and its markdown table with the device lines (stderr) |
| `athanor-step8.json`, `athanor-step8.log` | athanor step 8 |

The raw files are as the tools wrote them, except for paths: the absolute paths of the model files and of the report, which named the machine's home directory, are written relative to the repository.

## Notes

- **Memory.** athanor's device memory, from the driver: 1.74 GB in use before the model (the desktop and this process's context), 4.13 GB at the peak of the run, with a context of 32,897 tokens: 1.40 GiB of weights, the KV cache, the recurrent and conv state, and scratch memory. Peak RSS 3.8 GB (the memory-mapped checkpoint).
- **An earlier measurement was wrong.** The first baseline of step 8, taken before this report, timed each tg test's prompt as part of decode (a forward call without logits returns as soon as its work is queued): 49.7 and 6.9 tokens/s at 4k and 32k. `athanor bench` now waits for the prompt before starting the clock, and the step 8 numbers above were measured again with that fix: 59.2 and 8.6.
- **Prefill regressed** from step 8 (pp512 19,840 to 17,604; pp4096 10,585 to 8,553). The Gated DeltaNet kernel now gives a lane to each column of the state, so that its rows are contiguous accesses in decode; for a prompt it walks the tokens one at a time with 64 blocks for the whole model and four block barriers per token. The same kernel serves decode and prefill so that a prompt fed in pieces keeps giving the bits of the whole prompt. The chunked Gated DeltaNet of step 10 replaces it for prompts; prefill is step 10's gate (within 15% of vLLM).
- **Where decode stands.** At depth 0 a token takes 2.27 ms; reading the 1.5 GB of weights once takes 1.57 ms at the GPU's 960 GB/s. Most of the difference is the GEMV kernels' distance from the bandwidth, and the rest is the fixed cost of the other small kernels of a step (some 300 kernels in all) and the gaps between them inside the graph. At depth 32k the attention over the cache (6 layers, 2 key heads of 256) adds 1.5 ms per token against llama.cpp's 0.5.
- **Numerics.** The kernels of this step pass the gates of step 8 (`STATUS.md`, step 9): 0 greedy differences from the fp32 reference above the bf16 noise in 11,445 steps; on the forward records a pooled KL of 4.295e-4 against the bf16 reference's 5.005e-4.
