# MoE decode: athanor step 12 against Strata and llama.cpp

The report of step 12's gate (`docs/PLAN.md`): Qwen3.6-35B-A3B decode at least Strata's, on the same machine and quantization.

Measured on 2026-11-25 in one session, from 19:42 to 20:20. One tool ran at a time on an otherwise idle machine; a desktop session with a browser was open, as in steps 9 and 10. Each tool got a minute's pause first, in this order: athanor, Strata, llama.cpp, then Qwen3.5-0.8B for step 10's criterion. An A/B of the 0.8B against step 10's build followed, from 20:21 to 20:24 (Notes).

## Result

Decode tokens per second at batch 1. All three engines read one file, Unsloth's Q8_0 GGUF of Qwen3.6-35B-A3B (Setup).

| test | athanor step 12: median (min–max) | Strata, MTP off: each run | Strata, MTP on: each run | llama.cpp: mean ± sd | athanor / Strata MTP off (its best run) |
| --- | --- | --- | --- | --- | --- |
| **tg128 after a 33-token chat prompt** | **94.1** (88.1–96.4) | 29.7, 24.9, 30.1 (context 8192); 23.0, 27.0, 27.7 (32768, its default) | 38.8, 33.4 | | **3.13** |
| **tg128 after a 4096-token prompt** | **80.3** (78.7–80.4) | 22.3, 25.9 | 33.9 | | **3.10** |
| tg128, depth 0, pseudo-random ids | 102.6 (97.3–104.7) | | | 59.5 ± 0.4 | |
| tg128, depth 4096, pseudo-random ids | 69.6 (68.9–70.5) | | | 58.8 ± 0.9 | |

Prefill, reported beside (no gate):

| test | athanor | Strata | llama.cpp |
| --- | --- | --- | --- |
| the 33-token prompt | 109.3 tok/s (106.4–113.0) | 60.5 to 65.8 tok/s | |
| the 4096-token prompt | 209.4 (209.0–209.6) | 3,140 and 3,198 | |
| pp512, pseudo-random ids | 157.8 (155.6–158.7) | | 650.3 ± 14.0 |

**The gate passes.** After the same prompt, the same token ids for both engines, athanor decodes 128 tokens 3.1 times as fast as Strata with MTP off. Its best run is the comparison: 94.1 against 30.1 after the short prompt, 80.3 against 25.9 after the long one. athanor has no speculative decoding until step 13, so MTP off compares the same algorithm. With MTP on (its drafts accepted 62 to 72%), Strata reaches 38.8 and 33.9: athanor is still 2.4 times as fast. Against llama.cpp, with its experts of 27 layers on the CPU (the most that fit), athanor is 1.72 times as fast at depth 0 and 1.18 times at 4096.

Prefill is where Strata leads: it streams the experts a prompt needs to the GPU in big chunks, 3,140 to 3,198 tokens/s at 4096 tokens. athanor computes the experts that are not in VRAM on the CPU, as in decode: 209 tokens/s. This is in `docs/STATUS.md`, Possible later work.

## Setup

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5080, 16 GB, `sm_120`, PCIe 5.0 x16 |
| Driver | 580.178.04 (CUDA 13.0) |
| CPU, RAM | AMD Ryzen 7 9800X3D (8 cores, AVX-512), 59 GiB usable (62,267,304 kB); RAM speed not recorded (needs root) |
| Model | Qwen3.6-35B-A3B. Weights: `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` revision `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`, `Qwen3.6-35B-A3B-Q8_0.gguf`, 37,801,097,504 bytes, sha256 `c1283d8b80c3e38b2735ddbc9766d3b3126f44d6c484be419d4e101d09a76131`. The same file for the three engines; athanor reads `config.json` and the tokenizer from `Qwen/Qwen3.6-35B-A3B` revision `345d467b117d1fd5eda9b879a5988de9ae5f47a7` |
| Precision | Q8_0 weights for every matmul and the embedding (the file's), f32 norms and router; activations bf16 (athanor), the engines' own for the others |
| athanor | built from the tree of commit `b10b307` (`make cuda`, CUDA toolkit 13.5, runtime 13050, cuBLASLt 13.10, gcc 15.2). The reports say `bc80935-dirty` because `tools/bench.c` (`--prompt-ids`) and documents were not yet committed when it was built; the code is `b10b307`'s |
| Strata | commit `2af30cc` (2026-11-24; version 0.1.68 in its CMake project), built from source with plain cmake for `sm_120` with CUDA 13.5 (`-DSTRATA_ENABLE_CUDA=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF`, the flags of its `setup.py`; `~/opt/strata-build.log`), `strata-q36` |
| llama.cpp | commit `5220a2d`, built as for steps 9 and 10 (`-DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_CUDA_ARCHITECTURES=120`, CUDA 13.5) |

## Method

- **The same prompts.** `prompt-short.ids` (33 tokens) and `prompt-4096.ids` (4096 tokens) are chat-formatted prompts with thinking off, tokenized by Strata's own tokenizer from the GGUF (`q36tok.py`, written while preparing Strata). The short one asks how a hash table finds a value; the long one pads a request with text. Both engines run exactly these ids, then decode 128 tokens greedily, each choosing from its own logits. Both decode past the end of the answer, so 128 tokens each.
- **athanor**: `athanor bench models/Qwen3.6-35B-A3B --prompt-ids prompt-short.ids -n 128 -r 5 --warmup 1 --no-hash` (and `prompt-4096.ids` with `-r 3`). With `--prompt-ids`, ppN runs the prompt and tg128 decodes after it: the last prompt token's step gives the first token. Then `-p 512 -n 128 -d 0,4096 -r 3 --warmup 1` with pseudo-random ids, the method of steps 9 and 10. Medians with min and max. The model hash was not recomputed (`--no-hash`; the sha256 above was taken when the file was downloaded).
  - Expert placement: 75 of 256 experts per layer in VRAM (9.34 GiB), every expert in pinned RAM (31.9 GiB). The slots follow use: every 4 decoder steps, up to 2 moves per layer.
  - The placement persists across a run's tests, so the warm-up run also adapts the slots. 5,222 to 9,246 moves over a whole run.
- **Strata**: `strata-q36 --model <the GGUF> --tokens <the ids> --n 128 [--max-context N] [--mtp 2]`, one process per measurement, each printing its prompt and decode speeds. Everything else is at its defaults.
  - At its default context of 32,768 (an fp32 KV cache) it keeps 26 of 256 experts per layer in VRAM; at `--max-context 8192`, 35. The second is faster, and the gate uses it.
  - MTP on is `--mtp 2`, as in its documentation's Qwen3.6 examples. MTP off is its default for `strata-q36`.
  - Three runs at each context with MTP off after the short prompt; two after the long one; MTP on twice and once.
- **llama.cpp**: `llama-bench -m <the GGUF> -ngl 99 -ncmoe 27 -fa 1 -p 512 -n 128 -d 0,4096 -r 3 -o json`. `--n-cpu-moe 27` keeps the experts of 27 of the 40 layers on the CPU, the fewest that leave the rest in 16 GB. llama-bench takes pseudo-random ids, so it is compared with athanor's pseudo-random tests.
- **Qwen3.5-0.8B** (step 10's criterion 5): `athanor bench models/Qwen3.5-0.8B -p 512,4096 -n 128 -d 0 -r 5` and `llama-bench -m Qwen3.5-0.8B-bf16.gguf -ngl 99 -fa 1 -p 512,4096 -n 128 -r 5`, as in step 10.

## Files

| File | Content |
| --- | --- |
| `athanor-short.json`, `.log`; `athanor-4096.json`, `.log`; `athanor-random.json`, `.log` | athanor's reports (`docs/BENCHMARKS.md`: hardware, software, workload, the experts' placement, every run, memory) and console output. Their `"weights"` field reads "as stored in the checkpoint (bf16)": a fixed string in `tools/bench.c`, corrected after the session to the format and dtypes the params have (Q8_0 here) |
| `strata-*.log` | each Strata run: its configuration (experts in VRAM, context, KV cache), prompt and decode speed, the share of expert work on the GPU, the generated ids, and a 2-second monitor of VRAM, RSS and free RAM (`MON` lines) |
| `llama-bench.json`, `.log` | llama.cpp on the 35B |
| `athanor-0.8b.json`, `.log`, `llama-bench-0.8b.json`, `.log` | Qwen3.5-0.8B |
| `ab-0.8b/` | the A/B of the 0.8B between step 10's build and this one (Notes) |
| `prompt-short.ids`, `prompt-4096.ids` | the prompts' token ids |
| `session.log` | the order and times of the session |

Paths under the home directory in the raw files are written as they were.

## Notes

- **Where the time goes in a decode step.** athanor reads 2.5 GB of attention, Gated DeltaNet, shared experts and output head on the GPU per token. It runs 320 routed experts, 3.34 MB each, about 1.07 GB: the ones in VRAM on the GPU, the others on the CPU from pinned RAM, both at once in every layer.
  - With every expert on the CPU (`--experts 0`) it decodes at 34.9 tokens/s, so the CPU reads about 43 GB/s.
  - With 75 experts per layer in VRAM the slots catch most choices once they have adapted: 94 to 103 tokens/s.
  - Strata keeps 26 to 35 experts per layer in VRAM, and runs 43 to 44% of the expert work on the GPU by its own count.
- **Placement never changes a result** (AGENTS.md invariant 7, option (b)): the CPU and GPU expert paths follow one arithmetic order. With 73, 0 and 24 experts per layer in VRAM, and the slots changing between decoder steps, the logits of a 40-token prompt and 12 decoded tokens are the same, bit for bit (`tests/test_engine.c`, `engine_moe_placement`). The speeds above therefore come from runs whose outputs do not depend on what the cache held.
- **Numerics** (`docs/STATUS.md`, step 12): against transformers 5.24.0 run one layer at a time over the same GGUF (`tests/golden/moe.py`, 234 positions).
  - bf16: pooled KL 1.811e-3 against the bf16 reference's 2.970e-3; the same greedy token at every position above the noise.
  - f32: KL 1e-12, logits within 9.6e-6.
- **Memory.** athanor: 1.66 GB of VRAM in use before the model (the desktop and the process's context), 14.0 to 14.3 GB at the peak (2.48 GiB of other weights, 9.34 GiB of expert slots, states and working memory). Peak RSS 50 to 53 GB: 31.9 GiB of pinned expert copies, plus the mapped GGUF pages read when loading. Strata pins 32.7 GiB of experts and uses 4.40 GiB of VRAM for the other weights, 4.47 GiB for its 35 slots at context 8192.
- **Qwen3.5-0.8B** (step 10's criterion 5), in this session:
  - prefill pp512 49,665 tokens/s, pp4096 56,997 (step 10's report: 49,744 and 58,223);
  - decode tg128 448.1 against llama.cpp's 438.8 ± 0.2 (step 10: 445.6 against 437.5).
  
  pp4096 is 2.1% under step 10's median, so step 10's build (commit `5323f02`) was rebuilt in a separate worktree and run alternately with this one: three rounds each, 30 s apart, `-p 512,4096 -n 128 -d 0 -r 5` (`ab-0.8b/`).

| build | pp512 (each round) | pp4096 | tg128 |
| --- | --- | --- | --- |
| step 10 (`5323f02`) | 49,578, 49,447, 49,615 | 56,660, 56,841, 56,981 | 448.6, 448.1, 450.1 |
| step 12 | 49,612, 49,541, 49,633 | 56,683, 56,975, 56,985 | 447.3, 449.0, 449.4 |

  The two builds are the same within the noise. The 2% under step 10's report is this evening's machine, not the code: power capping under sustained load, which step 10's report measured at 2 to 3% between sessions.
