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

## `tools/dev/`

Development-only helpers (Python allowed): Unicode table generation, golden data utilities.

| File | Content |
| --- | --- |
| `check_alloc.sh`, `alloc_trace.c` | run a command under an `LD_PRELOAD` shim that reports every `malloc`/`calloc`/`realloc`, and fail if any caller is not `ath_arena_alloc` or if the command itself fails. `make check-alloc` runs it on `athanor inspect` with and without `--roles`. Needs a C compiler and `addr2line` |
