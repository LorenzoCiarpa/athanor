# Long context: athanor step 14

The report of step 14's gate (`docs/PLAN.md`): a 128k context on Qwen3.5-9B in 16 GB; resume after a restart; the snapshot's size per context length.

Measured on 2026-11-26, from 10:00 to 11:30. One program ran at a time on an otherwise idle machine, with a desktop session open as in steps 9 to 13. The desktop's own use of the GPU (1.5 GiB, varying by a few hundred MiB with what is on screen) counts against the 16 GB.

## Result

**128k on Qwen3.5-9B in 16 GB, with the KV cache in bf16 and in int8** (`tests/test_long.c`, `long/test_long-9b.log`). Q8_0 weights (8.86 GiB), a context of 131,072 tokens, and one prompt that fills it: 130,524 tokens of short notes, one of which, a tenth of the way in, gives a passphrase; at the end the question.

| | KV bf16 | KV int8 |
| --- | --- | --- |
| prefill of 130,524 tokens | 197.5 s (661 tokens/s) | 199.0 s (656 tokens/s) |
| decode at that depth | 36.3 tokens/s | 32.8 tokens/s |
| device memory in use (all processes, the driver's count) | 14.85 of 15.45 GiB | 14.29 of 15.45 GiB |
| the passphrase in the greedy answer | found | found |
| the prompt in pieces of 640 against 1024 | the same state and logits, bit for bit | the same, bit for bit |

The int8 cache against the bf16 one at that depth (the bf16 answer's 24 tokens fed to both): mean KL 1.1e-5 (max 9.9e-5), the same greedy token at 24 of 24. The bound is the bf16 noise, the bf16 reference's own KL against fp32 on the forward records (3.8e-4); the cache moves the output 35 times less than that.

**Benchmarks** (`athanor bench`, `--chunk 1024`, `bench/`), tokens/s, median of 3 runs (2 at depth 130,000):

| KV | pp4096 | pp32768 | tg64 at depth 0 | at 32,768 | at 130,000 |
| --- | --- | --- | --- | --- | --- |
| bf16 | 734 | 709 | 85.3 | 63.2 | 37.1 |
| int8 | 720 | 710 | 83.8 | 60.9 | 32.6 |

Decode at depth 0 reads the 8.86 GiB of weights at 807 GB/s for each token, near the RTX 5080's bandwidth. With depth, the attention's share grows: at 130,000 tokens the bf16 cache is 4 GiB, read once per token. The int8 cache reads half of it, yet decodes slower: its kernels convert each value as they read it, and on this GPU that costs more than the bytes it saves (an item for the optimization phase). The prefill runs at about 700 tokens/s at every length: its matrix products with Q8_0 weights do not use the tensor cores yet (step 12's kernel; the same item).

**Resume after a restart, at 128k** (`serve/`). `athanor serve models/Qwen3.5-9B --ctx 131072 --chunk 1024 --no-think --snapshots <dir>`. A document of about 123,800 tokens (5,500 notes) is the system message, and the user asks for the passphrase. The server answers ("amber-falcon-4127", 187.7 s for the request) and is stopped (SIGINT). The snapshots go to disk at exit: the shared part (the system message), 4.1 GB, and the conversation's stable part. A new server on the same folder is asked about note 7 instead. It starts from the shared part on disk: 123,827 of 123,861 prompt tokens cached, the prompt in 0.96 s, the request in 2.4 s. The answer is right: "The brown letter in the chapel yard was moved on day 81." `athanor chat --ctx 131072 --chunk 1024 --snapshots <dir>` likewise (`chat/`), with the 126,122-token prompt as one message read from stdin, run twice. The first run prefills it in 190.8 s and answers "amber-falcon-4127", 3 min 14 s in all. The second loads 126,115 tokens from the snapshot on disk (3,990 MiB in 0.99 s), runs 7 tokens and gives the same answer, 2.4 s in all.

**Snapshot size per context length** (`long/test_long-9b.log`): a snapshot holds the KV rows of its tokens and the fixed state (the Gated DeltaNet's 48 MiB, the convolution's 1.1 MiB). Copied out of the GPU and written to disk:

| tokens | bf16 KV | copied out | int8 KV | copied out |
| --- | --- | --- | --- | --- |
| 8,192 | 305 MiB | 0.06 s | 193 MiB | 0.04 s |
| 32,768 | 1,073 MiB | 0.17 s | 625 MiB | 0.10 s |
| 65,536 | 2,097 MiB | 0.25 s | 1,201 MiB | 0.15 s |
| 130,524 | 4,128 MiB | 0.48 s | 2,344 MiB | 0.27 s |
| 130,524, to disk and back | written in 0.82 s, read and loaded in 1.06 s | | written in 0.45 s, read and loaded in 0.62 s | |

So 32 KiB per token in bf16 and 18 KiB in int8, plus 49 MiB. The copies run at about 8.6 GB/s from pageable host memory. The disk times are those of a file the system had just written (the page cache); a file read cold from the NVMe drive takes longer.

**Numerics** (criterion 2):

| Check | Result |
| --- | --- |
| Qwen3.5-9B against its reference (the GGUF dequantized; `tests/golden/gguf.py`), bf16 | pooled KL ratio 0.33 (CPU), 0.49 (GPU); 0 greedy differences above the noise at 206 positions |
| the same, f32 | logits within 1.6e-5 (CPU) and 1.1e-4 (GPU), residual stream within 2.0e-5 and 8.5e-5: within the 9B's tolerances, 5e-4 and 4e-4, set at four times the worst measured. Step 4's 1e-4 did not hold on the GPU: two ill-conditioned positions of the chat prompt amplify every device's rounding tenfold (`tests/README.md`) |
| int8 KV against the fp32 reference, short prompts | KL ratio 0.68 and 0.99 on the 0.8B (CPU, GPU), 0.46 and 0.47 on the 9B, 0.62 on Qwen3.6-35B-A3B (GPU); 0 greedy differences above the noise; bound 1.25, the bf16 reference with the same cache (`tests/golden/kv_int8.py`) |
| int8 KV against bf16 KV, at 130,524 tokens | KL 1.1e-5, 24 of 24 greedy tokens (above) |
| long caches, the kernels (`gpu_attn_long`) | decode and split-K at positions to 131,199 within the short caches' tolerance; tensor cores at 0.02 of their bound; the int8 cache bit for bit |

**Nothing else changes** (criterion 5). The numerics of the 0.8B and the 35B are those of before: the forward records, the decoder and the engine tests give the same results and bits (`make cuda test`). Speed, with step 13's commands (`bench/athanor-*.json`):

| test | step 13's report | step 14 |
| --- | --- | --- |
| Qwen3.5-0.8B pp512 | 49,722 | 49,238 (48,110–49,512) |
| Qwen3.5-0.8B pp4096 | 57,004 | 56,286 (55,982–56,845) |
| Qwen3.5-0.8B tg128 at depth 0 | 446.4 | 434.5 (431.0–434.9) |
| Qwen3.6-35B-A3B tg128 after 33 tokens | 92.7 | 82.6 (73.4–88.0) |
| Qwen3.6-35B-A3B tg128 after 4096 tokens | 77.7 | 74.1 (72.4–75.8) |
| Qwen3.6-35B-A3B prefill of 4096 tokens | 194.8 | 202.8 |

The decodes came out under step 13's report, the 35B's with a wide range. So step 13's build (`main`, `f840d33`, in a worktree) and step 14's ran alternately right after (`bench/ab/`):

| test | step 13's build | step 14's build |
| --- | --- | --- |
| 35B tg128 after 33 tokens, two rounds | 83.3, 89.8 | 84.1, 88.3 |
| 0.8B tg128, four rounds of 10 runs | 446.3, 431.9, 445.8, 434.5 | 440.2, 440.4, 440.2, 440.2 |
| the same, step 14 with the weights in 256 MiB blocks again | | 441.0, 439.5, 440.0, 440.3 |

The same speed: step 13's own build is as far from its report today, as in step 13's own A/B. The weights' single allocation changes nothing for decode. The 35B keeps one more expert per layer in device memory (74 instead of 73), from the memory it frees.

**Build and tests** (criterion 6). `make test` and `make cuda test` are green, and so is `DEBUG=1` (ASan and UBSan; the 35B's tests skipped with `ATH_MOE_DIR` unset, as in steps 12 and 13) green too, after one fix: UBSan caught `qsort` called on the store's empty list when a snapshot folder is new (undefined, harmless in practice; now skipped below two entries). compute-sanitizer: memcheck (with leak check), racecheck and initcheck on `gpu_kv_q8`; memcheck on `gpu_attn_long`, `engine_snapshots`, `engine_decoder`, the forward records on the GPU and `serve_snapshots_resume`. 0 errors, 0 hazards, 0 bytes leaked. No warnings.

## What the step changed on the way

- **The int8 format.** It was decided as one scale per row (a token and a KV head: 256 values). Measured, that cost 2.4 times the bf16 noise in KL, almost all of it in the values (`tests/golden/kv_int8.py`), and changed a greedy token above the noise. With a scale per 32 values it costs 1.2 times and passes the gate, at 18 KiB per token instead of 16.25.
- **The weights' memory.** After the 9B's weights only 3.66 GiB were free, too little for the bf16 cache of 131,072 tokens (4.05 GiB). The engine copied the weights into arena blocks of 256 MiB, and the end of each block went unused: 1.15 GiB in all. In one allocation of their size, 4.89 GiB are free.
- **Prompt pieces.** A pass's working memory grows with the tokens it runs: on the 9B about 0.24, 0.51 and 0.99 GiB for pieces of 1024, 2048 and 4096 tokens, at the same speed (715 tokens/s). At 128k in bf16 only pieces of 1024 fit. `chat` now runs prompts in pieces (`--chunk`), and `chat` and `serve` warn when little device memory is left for them.
- **The shared part.** The first resume at 128k found nothing to resume. The snapshot was of the stable part, which includes the last user message, and the new question diverged inside it. `serve` and `chat` now also keep the shared part (the system messages and tools, 1024 tokens or more), which never counts as superseded.

## Setup

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5080, 16 GB, `sm_120`, PCIe 5.0 x16 |
| Driver | 580.178.04 (CUDA 13.0) |
| CPU, RAM, disk | AMD Ryzen 7 9800X3D (8 cores, AVX-512), 59 GiB usable; Crucial P310 NVMe (500 GB) |
| Model | Qwen3.5-9B, Unsloth's Q8_0 GGUF (`docs/contract/qwen35-9b.md`) |
| athanor | `make cuda`, CUDA toolkit 13.5, gcc 15.2. `test_long` and the benchmarks ran on the tree of commit `0ed1c91` (the benchmark reports name `a7a2015-dirty`: built from that tree before it was committed); `serve` and `chat` on commit `2a5b862` |

## Method

- **128k** (`ATH_LONG=1 ATH_LONG_CHUNK=1024 ATH_MODEL_DIR=models/Qwen3.5-9B ./build/cuda/test_long`): the prompt is chat-formatted (one user message, thinking off) and sized to the context less room for the answer. For each KV dtype: a state of 131,072 tokens; the prompt in pieces of 1024, with the snapshot copied out at 8k, 32k and 64k and at the end (each copy timed after the pass it follows is done); 24 greedy tokens with the decoder (one recorded graph per token); the state reset, the prompt in pieces of 640, the snapshot and the last logits compared byte for byte; the bf16 answer's tokens fed from the prompt's end, logits kept; the snapshot through the store to a folder and back. Device memory in use is the driver's count (`cudaMemGetInfo`), all processes, after the run.
- **Benchmarks**: `athanor bench models/Qwen3.5-9B -p 4096,32768 -n 64 -d 0,32768 -r 3 --warmup 1 --chunk 1024 --kv bf16|q8` and `-p 0 -n 64 -d 130000 -r 2 --warmup 0`; pseudo-random token ids; reports in `bench/`.
- **Resume**: `serve/run.sh`, `requests.py` and `haystack.py` (the notes of `test_long`); the server's lines, the responses and the folder's files after each run are in `serve/`. `chat/run.sh` likewise.
