# Prefill: athanor step 10 against vLLM and llama.cpp

The report of step 10's gate (`docs/PLAN.md`): prefill tokens per second for a single request, within 15% of vLLM's, on the same model, precision and GPU. Measured on 2026-11-25, one tool at a time on an otherwise idle machine, a minute's pause before each, in this order: athanor, llama.cpp, vLLM (17:07 to 17:16). An earlier session, with a build whose decode had regressed, is kept in `first-session/` (Notes).

## Result

| test | athanor step 10: median (min–max) | vLLM, single request: median (min–max) | athanor / vLLM | vLLM, forward pass alone | llama.cpp: mean ± sd | athanor step 9 |
| --- | --- | --- | --- | --- | --- | --- |
| **pp512** | **49,744** (49,589–49,828) | 49,624 (46,142–51,221) | **1.002** | 69,668 | 40,260 ± 10,877 | 17,604 |
| **pp4096** | **58,223** (57,061–58,770) | 67,652 (66,781–68,349) | **0.861** | 72,275 | 45,660 ± 1,554 | 8,553 |
| **pp32768** | **41,781** (41,615–41,917) | 47,229 (46,801–47,456) | **0.885** | 47,366 | 35,787 ± 345 | not measured |

Prompt tokens per second. ppN: a prompt of N tokens, its forward pass and the last token's logits (athanor), or a request of N prompt tokens and one output token (vLLM). vLLM's single request is the median of 20 latencies after 5 warm-up requests; its forward pass alone is the median of 5 samples of 20 replays.

**The gate passes**: at 512, 4096 and 32768 tokens athanor's prefill is 100%, 86% and 89% of vLLM's for a single request, above the 85% of `docs/PLAN.md`. The margin at 4096 is thin (1.1 points; athanor's slowest run is at 0.843 of vLLM's median, its fastest at 0.869). Against vLLM's forward pass alone (no scheduler, no sampler, the pass captured as a CUDA graph) athanor is at 0.71, 0.81 and 0.88. Against llama.cpp it is 1.24, 1.28 and 1.17 times as fast. Step 9's prefill (the recurrence token by token, attention one query per block) was 3 to 7 times slower.

Decode, step 9's gate, measured in the same session:

| test | athanor step 10 | llama.cpp | athanor / llama.cpp | athanor step 9 (its report) |
| --- | --- | --- | --- | --- |
| tg128 @ depth 0 | 445.6 (439.6–447.7) | 437.5 ± 0.6 | 1.018 | 441.0 |
| tg128 @ depth 4096 | 390.3 (383.1–395.7) | 426.5 ± 1.6 | 0.915 | 393.0 |
| tg128 @ depth 32768 | 267.6 (266.0–268.4) | 359.7 ± 0.8 | 0.744 | 266.5 |

## Setup

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5080, 16 GB, `sm_120`, PCIe 5.0 x16 |
| Driver | 580.178.04 (CUDA 13.0) |
| CPU, RAM | AMD Ryzen 7 9800X3D (8 cores), 59 GiB usable |
| athanor | commit `5323f02` (`git describe`: `v0.1-cpu-chat-40-g5323f02ef020`), `make cuda`, CUDA toolkit 13.5 (runtime 13050), cuBLASLt 13.10, gcc 15.2 |
| vLLM | 0.34.0 with torch 2.14.0 (CUDA 13.0 build), FlashInfer 0.7.3 with its kernels downloaded (`vllm download-kernels`), Triton 3.8.0, transformers 5.22.2; Python 3.12 in a virtual environment outside the repository; `CUDA_HOME` set to the CUDA 13.5 toolkit. Its log: attention backend `FLASH_ATTN`, "FlashInfer GDN prefill kernel" for the Gated DeltaNet, CUDA graphs `FULL_AND_PIECEWISE` |
| llama.cpp | commit `5220a2d`, built as for step 9 (`-DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_CUDA_ARCHITECTURES=120`, CUDA 13.5) |
| Model | Qwen/Qwen3.5-0.8B, revision `2fc06364715b967f1860aea9cf38778875588b17`, `model.safetensors-00001-of-00001.safetensors` sha256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`; the same files for vLLM; for llama.cpp the GGUF of step 9 (`convert_hf_to_gguf.py --outtype bf16`) |
| Precision | bf16 weights and activations for all three; KV cache bf16 (athanor, vLLM), f16 (llama.cpp) |

## Method

- **athanor**: `athanor bench models/Qwen3.5-0.8B -p 512,4096,32768 -n 128 -d 0,4096,32768 -r 5 --chunk 32768 --json athanor.json`. One unmeasured run, then 5 measured runs per test from a reset state; median, minimum and maximum. Token ids pseudo-random (seed 1). Every prompt goes in one forward pass (`--chunk 32768`); a pp test is the time of that pass, from the call with the token ids in host memory to the last token's logits in host memory.
- **vLLM**: `vllm bench latency --model <dir> --input-len N --output-len 1 --batch-size 1 --num-iters-warmup 5 --num-iters 20 --max-model-len 33024 --gpu-memory-utilization 0.85 --max-num-batched-tokens 32768` for N = 512, 4096, 32768: whole requests through vLLM's engine (scheduler, input preparation, forward pass, sampling of one token), random token ids, prefix caching off (the tool's default). `--max-num-batched-tokens 32768` lets a prompt of 32k run in one pass, as athanor's; with the default (8192) vLLM measured 716 against 704 ms at 32k in the first runs of the step. `vllm bench forward --case p512=512 --case p4096=4096 --case p32768=32768 --kv-len-jitter 0` (same engine options): the forward pass of one prompt as a CUDA graph, without the scheduler and the sampler. vLLM runs as its own warnings recommend: without its FlashInfer kernels and with only the system's nvcc 12.4 it falls back to slower paths and says so; these runs print no such warning.
- **llama.cpp**: `llama-bench -m Qwen3.5-0.8B-bf16.gguf -ngl 99 -fa 1 -p 512,4096,32768 -n 0 -r 5` and `-p 0 -n 128 -d 0,4096,32768 -r 5`, with `-o json -oe md`, as in step 9 (`n_ubatch` 512, its default).
- **The gate's measure** is vLLM's single request (`docs/STATUS.md`, step 10 decisions): what a user of either engine waits for, a prompt in and its first token out. The forward pass alone is vLLM's GPU time without its host overhead, a stricter comparison reported beside it.

## Files

| File | Content |
| --- | --- |
| `athanor.json`, `athanor.log` | athanor step 10: the report of `docs/BENCHMARKS.md` (hardware, software, model files and hashes, workload, method, every run, memory) and the console output |
| `vllm-latency-{512,4096,32768}.json`, `.log` | vLLM: every latency of each request size, and its console output (configuration, backends, warnings) |
| `vllm-forward.json`, `.log` | vLLM's forward pass alone |
| `llama-bench-pp.json`, `.log`, `llama-bench-tg.json`, `.log` | llama.cpp: llama-bench's JSON (stdout) and its table with the device lines (stderr) |
| `first-session/` | the session of 16:48 (Notes): the same files for every tool |

The raw files are as the tools wrote them, except for paths under the home directory, written relative to the repository or to `~`.

## Notes

- **Where a prompt's time goes.** At 4096 tokens (Nsight Systems on the same build, 72 ms of kernels in a span of 73): 38.5 ms of matrix products (cuBLASLt at 106 to 145 TFLOPS depending on the shape, and the last token's 0.6 ms GEMV over the output head), 16.9 ms of the Gated DeltaNet (the chunked form, fp32 on the CUDA cores), 6.2 ms of attention (on the tensor cores), 3.9 ms of norms (the residual adds inside them), 3.1 ms of the convolution, 2.9 ms of SwiGLU and the attention gate, 0.4 ms of RoPE, about 1 ms between kernels. At 32k attention takes most of the rest: about 320 ms of the 784 (81 TFLOPS), the products about 300 ms. Step 9's prompt of 4096 spent 289 ms in attention and 135 ms in the Gated DeltaNet.
- **What differs between the engines.** vLLM's Gated DeltaNet prefill (FlashInfer, flash-linear-attention's algorithm) computes on the tensor cores in bf16; athanor's is fp32, as the reference's chunked function, and stays within 6e-6 of the reference on the golden records; that costs athanor most of its distance at 4096 tokens. Attention is FlashAttention in both (vLLM's `FLASH_ATTN`, athanor's own kernel, both rounding probabilities to bf16). vLLM fuses norms and activations through its compiled graph; athanor fuses the residual add into the next norm and keeps the other elementwise ops as kernels.
- **Numerics with these kernels** (`docs/STATUS.md`, step 10): the gate of step 8 still passes (0 of 11,445 teacher-forced steps above each prompt's bf16 noise differ from the fp32 reference); the GPU forward records' pooled KL is 4.399e-4, under the bf16 reference's 5.005e-4 (step 9: 4.295e-4); the f32 run stays at KL 3e-12.
- **The first session.** The session of 16:48 measured a build (`da4967e`) whose decode had become 1% slower than step 9's (437.7 against llama.cpp's 440.3 tokens/s): the norm kernel had grown to 7,700 instructions with step 10's changes, more than the instruction cache holds, and decode launches it some 80 times per token between the GEMV kernels. Specialized per kind, with the Gated DeltaNet's recurrence specialized for heads of up to 128 (`5323f02`), decode is back above step 9 and llama.cpp. The prefill numbers of that session agree with these within the noise: athanor 49,836, 58,332 and 42,022 tokens/s, vLLM 49,718, 67,337 and 47,836 (ratios 1.002, 0.866, 0.878). A partial run between the two sessions found all three tools 2 to 3% slower; the GPU's counters show software power capping under sustained load, which is why this session pauses a minute before each tool.
- **Memory.** athanor's device memory, from the driver: 1.63 GB in use before the model (the desktop and this process's context), 6.72 GB at the peak (1.40 GiB of weights, the KV cache and state for 32,897 tokens, the scratch memory of a 32k prompt in one pass: activations and the Gated DeltaNet's rounds). vLLM reserves 85% of the GPU by its configuration.
