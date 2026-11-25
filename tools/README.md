# tools

One binary, `athanor`, with subcommands. `athanor.c` holds only the dispatch table and process setup; each subcommand lives in its own file, parses its own options and calls library code.

| File | Content |
| --- | --- |
| `athanor.c` | dispatch table, global `--help`, arena-backed stdout buffer, exit codes |
| `commands.h` | subcommand entry points (`ath_cmd_<name>`) and exit codes: 0 success, 1 failure, 2 usage |
| `inspect.c` | `athanor inspect` |
| `chat.c` | `athanor chat` (step 5; on the GPU since step 8, through the decoder since step 9) |
| `devices.c` | `athanor devices` (step 6) |
| `bench.c` | `athanor bench` (step 9): ppN and tgN@dD as llama-bench names them, median of measured runs, the JSON report of `docs/BENCHMARKS.md`; `--mtp` decodes with the draft head (step 13) |
| `serve.c` | `athanor serve` (step 13): the HTTP APIs of `src/net` |
| `dev/` | development-only helpers, never built into `athanor` |

| Subcommand | Purpose | Step |
| --- | --- | --- |
| `inspect` | tensors, dtypes, shapes, roles, placement map, expert statistics | 1, 2 |
| `chat` | interactive chat and one-shot generation | 5 (CPU), 8 (GPU) |
| `devices` | the GPUs this build can use; `--bench` times copies | 6 |
| `bench` | benchmarks per `docs/BENCHMARKS.md`, JSON output | 9 |
| `serve` | OpenAI- and Anthropic-compatible API | 13 |
| `quant` | `plan`, `apply`, `eval`, `convert` (GGUF) | phase 2 |
| `ablit` | `capture`, `search`, `apply`, `eval`, `export` | phase 3 |
| `train` | `plan`, `run`, `resume`, `export` | phase 4 |

Planned usage (flags are illustrative until each step lands):

```
athanor inspect models/Qwen3.5-0.8B --roles
athanor chat    models/Qwen3.5-9B --format q4
athanor serve   models/Qwen3.6-35B-A3B --ctx 65536 --port 8080
athanor bench   decode models/Qwen3.5-9B --ctx 0,4k,32k --batch 1,4
athanor quant   plan models/Qwen3.5-9B --budget 5.5G --calib data/calib.txt
athanor ablit   search models/Qwen3.5-2B --trials 200
athanor train   run models/Qwen3.5-9B --regime hotcold --data data/sft.jsonl
```

## `athanor inspect` (steps 1 and 2)

```
athanor inspect <model_dir> [--roles] [--json] [--filter <substring>]
```

Reads `config.json` and the safetensors headers (never the tensor data) and prints: a config summary (architectures, model type, layers, hidden size, vocabulary; keys are read at the top level or under `text_config`), the shard files, one row per tensor (name, dtype, shape, bytes) in natural name order, totals per dtype, and the check that the tensors cover the shards' data sections exactly. `--filter` restricts the table and the totals to names containing the substring; `--json` prints the same information as one JSON object (tensors also carry `shard` and absolute file `offset`). The exit code is 1 if the checkpoint cannot be opened or fails validation.

With `--roles` the checkpoint is loaded through the model contract (`src/model/model.h`) by the family that recognizes its `config.json`, and the output is what pipelines see: family and dimensions, the layer schedule with per-layer totals, one row per param (name, role, layer, expert, flags, dtype, shape, bytes; a fused MoE tensor gives one row per expert; the tensors the text path does not use, such as the vision encoder, are params flagged `unused`), totals per role over the params the text path uses, the unused params apart, the ignored groups with their reason, count and size, and the coverage check (`N checkpoint tensors: B bound to P params (I in ignored groups, unused), 0 unbound`). Loading fails, with the offending tensor named on stderr, if any tensor is missing, unexpected, or has a dtype or shape that does not follow from the config. `--filter` restricts the param rows and totals; `--json` prints one JSON object (`layers`, `params` with `flags` as a list, `roles`, `unused`, `ignored`, counts).

```
$ athanor inspect models/Qwen3.5-0.8B --roles
family   qwen35: 24 layers, hidden size 1024, vocab 248320, tied embeddings
schedule 18 gated_deltanet + 6 attention; 24 mlp
...
unused           168       242091520  230.88 MiB  (not in the totals above)

ignored  model.visual.      153 tensors  191.87 MiB  vision encoder: text path only (docs/MODELS.md)
         mtp.                15 tensors   39.01 MiB  multi-token prediction head: speculative decoding, step 13
check    488 checkpoint tensors (1.63 GiB): 488 bound to 488 params (168 in ignored groups, unused), 0 unbound
```

## `athanor chat` (step 5, GPU step 8)

```
athanor chat <model_dir> [-p <text>] [--system <text>] [--think] [--tools <file.json>]
             [--temp t] [--top-k k] [--top-p p] [--min-p p] [--seed n] [--max-tokens n]
             [--ctx n] [--device cpu|gpu] [--threads n] [--f32] [--stats]
```

A conversation, through the contract only: the family's chat template renders the whole conversation, the tokenizer encodes it, the forward pass runs on the GPU (the engine, `src/infer/engine.h`) or on a thread pool on the CPU, `ath_sample_ref` picks each token on the host, and the family's parser splits the output into text, thinking and tool calls as it streams. With `-p` it answers one message and exits; otherwise it reads one message per line from stdin (a line ending in `\` continues), with `/reset`, `/tool <text>` and `/exit`.

- `--think` lets the model think first; the thinking is shown dimmed on a terminal and goes to stderr otherwise, so stdout holds the answer.
- `--tools` takes a JSON array of tool definitions in OpenAI format. A call is printed as `[tool call] name {arguments}`; athanor never runs it (AGENTS.md, invariant 10). `/tool <result>` sends a result back as the harness would.
- Sampling defaults are the model card's for Qwen3.5 without the presence penalty: temperature 0.7, top-p 0.8, top-k 20, and 0.6, 0.95, 20 with `--think`. `--temp 0` is greedy. The seed defaults to the clock.
- The state is reused across turns: only new tokens run when the prompt extends what the state has seen; otherwise a copy of the state taken at the end of the previous prompt's last message is restored (the template rewrites a past assistant turn), and only when that does not match either does the conversation run again.
- `--device`: the GPU by default when the build has a GPU backend (`make cuda`) and the machine a device; otherwise, or with `--device cpu`, the CPU. Without a usable GPU the default falls back to the CPU with a note; `--device gpu` then fails instead. On the GPU the weights are copied to the device at start (1.40 GiB for Qwen3.5-0.8B) and the state and its copy live there.
- `--ctx` bounds the conversation (default 8192 tokens); the KV cache and the copy are allocated for it. `--f32` runs with f32 activations. `--stats` prints the device at start and, after each answer, how many prompt tokens ran and the speeds.

```
$ athanor chat models/Qwen3.5-0.8B -p "What is the capital of Italy? Answer in one sentence." --temp 0
The capital of Italy is Rome.
```

For the record (not a benchmark, `docs/BENCHMARKS.md`): with `--stats`, greedy, Qwen3.5-0.8B in bf16, a short answer decodes at about 290 tokens/s on the RTX 5080 and 8.5 tokens/s on the CPU (8 threads, Ryzen 7 9800X3D). Step 9 makes the GPU decode fast.

## `athanor devices` (step 6)

```
athanor devices [--bench]
```

Lists the GPUs the build can use, through the backend only (`src/backend/backend.h`): name, compute capability, multiprocessors, warp size, memory (total and free), L2, the largest shared memory a block can use, driver and runtime versions. A build without a GPU backend (`make`) lists none and says so. With `--bench` it also times 256 MiB copies with GPU events, best of 5: host to device and back from pinned and from ordinary (pageable) memory, and device to device. The numbers describe the machine for the record; they are not a benchmark report (`docs/BENCHMARKS.md`).

## `tools/dev/`

Development-only helpers (Python allowed): Unicode table generation, golden data utilities.

| File | Content |
| --- | --- |
| `check_nvcc.sh` | `make cuda`'s check that the nvcc in `CUDA_HOME` exists and is recent enough for `CUDA_ARCH` (12.8 for `sm_120`) |
| `gen_unicode.py` | generates `src/text/unicode_tables.h` (character classes and NFC data) from Python's `unicodedata`, with the normalization limited to what the reference tokenizer applies; run with the golden environment (`src/text/README.md`) |
| `check_alloc.sh`, `alloc_trace.c` | run a command under an `LD_PRELOAD` shim that reports every `malloc`/`calloc`/`realloc`, and fail if any caller is not `ath_arena_alloc` or if the command itself fails. `make check-alloc` runs it on `athanor inspect` with and without `--roles`. Needs a C compiler and `addr2line` |

## `athanor bench` (step 9)

```
athanor bench models/Qwen3.5-0.8B -p 512,4096 -n 128 -d 0,4096,32768 -r 5 --json report.json
```

- `ppN`: a prompt of N tokens into an empty state, in pieces of `--chunk` (2048); prefill tokens/s and the time to the first token (the prompt and the choice of the first token).
- `tgN@dD`: N tokens decoded one at a time, greedy, after a prompt of D tokens; decode tokens/s. On a GPU through the decoder (`src/infer/README.md`); `--direct` runs step 8's way, one forward call per token with the logits on the host. The prompt is finished before the clock starts.
- Token ids are pseudo-random (`--seed`), as llama-bench's. Every test runs `--warmup` times unmeasured, then `--reps` times measured from a reset state: the median with min and max.
- `--mtp` (step 13): the tg tests decode speculatively with the model's draft head (`src/infer/README.md`, Speculative decoding); the context and one more token are the prompt, so that the head runs over it, and the share of passes whose guess was kept is printed and reported (`guesses_kept`).
- `--json` writes the report `docs/BENCHMARKS.md` asks for: the hardware (GPU, PCIe link, driver, CPU, RAM, storage), the software (the commit, from `build/gen/version.h`, the compiler, runtime and cuBLASLt versions), the model files with their SHA-256 and Hugging Face revision (`--no-hash` skips the hashes), the workload, the method, every run, and memory from the driver and the kernel.

## `athanor serve` (step 13)

```
athanor serve <model_dir> [--host 127.0.0.1] [--port 8080] [--ctx 65536] [--experts n] [--no-think]
              [--max-tokens n] [--server-sampling] [--mtp] [--dump dir] [--quiet]
```

The model behind the HTTP APIs of coding agents: Anthropic Messages (`/v1/messages`, Claude Code), OpenAI Responses (`/v1/responses`, Codex CLI) and OpenAI Chat Completions (`/v1/chat/completions`, OpenCode and most clients), streamed or whole, with tool calls; `/v1/models`. One request at a time; the server never runs a tool. `src/net/README.md` has the endpoints, how requests become messages, the prompt's snapshots and the clients' configuration; `tests/conformance/` runs a task set through the three harnesses.

```
$ athanor serve models/Qwen3.6-35B-A3B --ctx 65536
athanor serve: loading models/Qwen3.6-35B-A3B
athanor serve: Qwen3.6-35B-A3B on NVIDIA GeForce RTX 5080, context 65536; listening on http://127.0.0.1:8080 (Ctrl-C stops)
serve: messages stream think: prompt 13025 (0 cached) in 56.72 s, 114 out in 2.74 s (41.5/s), tool_calls
serve: messages stream think: prompt 13267 (13138 cached) in 0.95 s, 129 out in 2.67 s (48.2/s), tool_calls
```

- `--ctx`: the context of the server's two states (the conversation and its snapshot), on the GPU before the experts take the rest.
- `--no-think`: requests that do not say get no thinking. `--server-sampling`: the model card's sampling for every request.
- `--mtp`: decode with the draft head: the same tokens; on a GPU slower than the default decoder until its passes are recorded too (`src/infer/README.md`).
- `--dump <dir>`: every request body saved there, to see what a client sends.
