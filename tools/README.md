# tools

One binary, `athanor`, with subcommands. `athanor.c` holds only the dispatch table and process setup; each subcommand lives in its own file, parses its own options and calls library code.

| File | Content |
| --- | --- |
| `athanor.c` | dispatch table, global `--help`, arena-backed stdout buffer, exit codes |
| `commands.h` | subcommand entry points (`ath_cmd_<name>`) and exit codes: 0 success, 1 failure, 2 usage |
| `inspect.c` | `athanor inspect` |
| `dev/` | development-only helpers, never built into `athanor` |

| Subcommand | Purpose | Step |
| --- | --- | --- |
| `inspect` | tensors, dtypes, shapes, roles, placement map, expert statistics | 1, 2 |
| `chat` | interactive chat and one-shot generation | 5 (CPU), 8 (GPU) |
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

## `athanor inspect` (step 1)

```
athanor inspect <model_dir> [--json] [--filter <substring>]
```

Reads `config.json` and the safetensors headers (never the tensor data) and prints: a config summary (architectures, model type, layers, hidden size, vocabulary; keys are read at the top level or under `text_config`), the shard files, one row per tensor (name, dtype, shape, bytes) in natural name order, totals per dtype, and the check that the tensors cover the shards' data sections exactly. `--filter` restricts the table and the totals to names containing the substring; `--json` prints the same information as one JSON object (tensors also carry `shard` and absolute file `offset`). The exit code is 1 if the checkpoint cannot be opened or fails validation. `--roles` arrives in step 2.

## `tools/dev/`

Development-only helpers (Python allowed): Unicode table generation, golden data utilities.

| File | Content |
| --- | --- |
| `check_alloc.sh`, `alloc_trace.c` | run a command under an `LD_PRELOAD` shim that reports every `malloc`/`calloc`/`realloc`, and fail if any caller is not `ath_arena_alloc`. `make check-alloc` runs it on `athanor inspect`. Needs a C compiler and `addr2line` |
