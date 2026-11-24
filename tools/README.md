# tools

One binary, `athanor`, with subcommands. `athanor.c` holds only the dispatch table and process setup; each subcommand lives in its own file, parses its own options and calls library code.

| File | Content |
| --- | --- |
| `athanor.c` | dispatch table, global `--help`, arena-backed stdout buffer, exit codes |
| `commands.h` | subcommand entry points (`ath_cmd_<name>`) and exit codes: 0 success, 1 failure, 2 usage |
| `inspect.c` | `athanor inspect` |
| `chat.c` | `athanor chat` (step 5) |
| `devices.c` | `athanor devices` (step 6) |
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
athanor serve   models/Qwen3.6-35B-A3B --vram 14G --ram 48G --port 8080
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

## `athanor chat` (step 5)

```
athanor chat <model_dir> [-p <text>] [--system <text>] [--think] [--tools <file.json>]
             [--temp t] [--top-k k] [--top-p p] [--min-p p] [--seed n] [--max-tokens n]
             [--ctx n] [--threads n] [--f32] [--stats]
```

A conversation on the CPU, through the contract only: the family's chat template renders the whole conversation, the tokenizer encodes it, the forward pass runs on a thread pool, `ath_sample_ref` picks each token, and the family's parser splits the output into text, thinking and tool calls as it streams. With `-p` it answers one message and exits; otherwise it reads one message per line from stdin (a line ending in `\` continues), with `/reset`, `/tool <text>` and `/exit`.

- `--think` lets the model think first; the thinking is shown dimmed on a terminal and goes to stderr otherwise, so stdout holds the answer.
- `--tools` takes a JSON array of tool definitions in OpenAI format. A call is printed as `[tool call] name {arguments}`; athanor never runs it (AGENTS.md, invariant 10). `/tool <result>` sends a result back as the harness would.
- Sampling defaults are the model card's for Qwen3.5 without the presence penalty: temperature 0.7, top-p 0.8, top-k 20, and 0.6, 0.95, 20 with `--think`. `--temp 0` is greedy. The seed defaults to the clock.
- The state is reused across turns: only new tokens run when the prompt extends what the state has seen; otherwise a copy of the state taken at the end of the previous prompt's last message is restored (the template rewrites a past assistant turn), and only when that does not match either does the conversation run again.
- `--ctx` bounds the conversation (default 8192 tokens); the KV cache and the copy are allocated for it. `--f32` runs with f32 activations. `--stats` prints, after each answer, how many prompt tokens ran and the speeds.

```
$ athanor chat models/Qwen3.5-0.8B -p "What is the capital of Italy? Answer in one sentence." --temp 0
The capital of Italy is Rome.
```

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
