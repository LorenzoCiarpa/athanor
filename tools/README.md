# tools

One binary, `athanor`, with subcommands. `athanor.c` holds only argument parsing and the dispatch table; every subcommand calls library code.

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

`tools/dev/` holds development-only helpers (Python allowed): Unicode table generation, golden data utilities.
