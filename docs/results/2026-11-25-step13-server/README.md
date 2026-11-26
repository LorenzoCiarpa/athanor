# Server and MTP: athanor step 13

The report of step 13's gate (`docs/PLAN.md`): `athanor serve` works with Claude Code, Codex CLI and OpenCode; multi-turn sessions with tool calls succeed on a fixed task set; the speculative-decoding speedup is reported.

Measured on 2026-11-25: the conformance runs from 20:57 to 21:15, the benchmarks from 22:38 to 22:51 and the A/B against step 12's build after them. One tool ran at a time on an otherwise idle machine, with a desktop session and a browser open, as in steps 9 to 12.

## Result

**Conformance: 15 of 15.** Five tasks, each through the three harnesses, against `athanor serve` with Qwen3.6-35B-A3B (Q8_0) on the RTX 5080:

| task | what the agent must do | Claude Code 2.1.332 | Codex CLI 0.167.0 | OpenCode 1.19.6 |
| --- | --- | --- | --- | --- |
| `fix-bug` | find the bug behind a failing test, fix it, run the test | pass, 73 s | pass, 38 s | pass, 77 s |
| `new-file` | write `fizzbuzz.py` to a specification, run it | pass, 76 s | pass, 47 s | pass, 88 s |
| `rename` | rename a function in its definition and every call site across three files, run the tests | pass, 83 s | pass, 47 s | pass, 99 s |
| `find-constant` | find which file defines a constant and its value, write the answer to a file | pass, 68 s | pass, 52 s | pass, 99 s |
| `csv-stats` | write a script computing a column's mean, run it, write its output to a file | pass, 95 s | pass, 47 s | pass, 115 s |

Each harness used its own API: Anthropic Messages (Claude Code), OpenAI Responses (Codex CLI), OpenAI Chat Completions (OpenCode), streamed, with thinking on and tool calls in both directions. A task's outcome is decided by its `check.sh` in the folder the agent worked in, not by what the agent says. Most of a task's time is the first request's prompt (13k, 7k and 11k tokens of system prompt and tool definitions, at 180 to 225 tokens/s: 30 to 73 s); every later request of a session reused the state and ran only what it added (`conformance/serve.log`, the server's line per request: for example 13,291 tokens with 13,150 cached, in 0.91 s).

**Speculative decoding** with the model's MTP head (step 13's draft head), greedy, tokens/s at batch 1 (median of 5 runs for the 0.8B, 3 for the 35B):

| model, prompt | guesses kept | speculative (host-driven) | plain, host-driven (`--direct`) | speedup, same kind | plain, recorded decoder |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-0.8B, coding (63 tokens), 256 generated | 86% | 430.6 | 358.4 | **1.20** | 441.1 |
| Qwen3.5-0.8B, story (33 tokens), 256 generated | 63% | 371.2 | 357.1 | 1.04 | 441.1 |
| Qwen3.6-35B-A3B, coding, 128 generated | 98% | 55.5 | 44.3 | **1.25** | 86.8 |
| Qwen3.6-35B-A3B, story, 128 generated | 76% | 48.5 | 45.2 | 1.07 | 105.9 |

The tokens are plain decoding's, bit for bit, in every case (`tests/test_spec.c`: 96 of 96 tokens on the CPU and the GPU, greedy, sampled and with a presence penalty; the 35B too). Against plain decoding driven the same way (a forward call per pass from the host), speculative decoding is 1.04 to 1.25 times as fast, more where the text is predictable. Against the recorded decoder of steps 9 and 12 it is slower: each of its passes pays the kernel launches the decoder's graph saves (on the 35B also the CPU's experts for two tokens), so its passes must be recorded too before it pays (`docs/STATUS.md`, Possible later work). Until then `--mtp` is off by default.

**Nothing else changed** (criterion 5), the same commands as step 12's report:

| test | step 12's report | step 13 |
| --- | --- | --- |
| Qwen3.5-0.8B pp512 | 49,665 | 49,722 (49,515–49,764) |
| Qwen3.5-0.8B pp4096 | 56,997 | 57,004 (56,246–57,757) |
| Qwen3.5-0.8B tg128 at depth 0 | 448.1 | 446.4 (443.3–447.0) |
| Qwen3.6-35B-A3B tg128 after 33 tokens | 94.1 (88.1–96.4) | 92.7 (89.7–93.3) |
| Qwen3.6-35B-A3B tg128 after 4096 tokens | 80.3 (78.7–80.4) | 77.7 (76.4–78.1); A/B below |
| Qwen3.6-35B-A3B prefill of 4096 tokens | 209.4 | 194.8 (193.3–208.0); A/B below |

The 35B after 4096 tokens came out 3% under step 12's report (and its prefill 7%). Step 12's build (`39c31d7`, built in a worktree) and step 13's, run alternately right after, two rounds each (`bench/ab/`; the model's absolute path in those reports made relative to the repository):

| test | step 12's build | step 13's build |
| --- | --- | --- |
| tg128 after 4096 tokens | 77.5, 77.6 | 76.9, 78.0 |
| prefill of 4096 tokens | 202.9, 209.7 | 207.7, 209.1 |
| tg128 after 33 tokens | 91.1, 91.7 | 92.1, 92.2 |

The same: step 12's own build is as far from its report as step 13's. The machine that evening, not the code (as with the 0.8B's pp4096 in step 12's report). Without `--mtp` the draft head is neither copied to the device nor placed (`ath_place_opts.draft`), so the experts' slots are those of step 12; the states grow by the head's cache (2 KiB per token) and its input.

## Setup

| | |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5080, 16 GB, `sm_120`, PCIe 5.0 x16 |
| Driver | 580.178.04 (CUDA 13.0) |
| CPU, RAM | AMD Ryzen 7 9800X3D (8 cores, AVX-512), 59 GiB usable |
| Models | Qwen3.5-0.8B (bf16 safetensors, as in steps 4 to 12) and Qwen3.6-35B-A3B (Unsloth's Q8_0 GGUF with the MTP block, as in step 12: `docs/contract/qwen36-35b-a3b.md`) |
| athanor | commit `049d497` (`make cuda`, CUDA toolkit 13.5, gcc 15.2); the reports name it `v0.1-cpu-chat-61-g049d497c6aec` |
| Harnesses | Claude Code 2.1.332, Codex CLI 0.167.0, OpenCode 1.19.6 (`npm install -g opencode-ai@1.19.6`) |

## Method

- **Conformance** (`tests/conformance/run.sh <harness> all <outdir>`): `athanor serve models/Qwen3.6-35B-A3B --ctx 65536` (thinking on, the request's sampling). Each task runs in a fresh git repository under `$TMPDIR` made from `tasks/<task>/files` (Claude Code refuses to edit files under `~/.claude`), with the task's prompt, in one non-interactive session: `claude -p` (stream-json), `codex exec --json`, `opencode run --format json`. Each harness has a configuration folder of its own (`CLAUDE_CONFIG_DIR`, `CODEX_HOME`, `XDG_*`), so the user's settings, hooks and MCP servers are never read. The commands an agent may run are a list for Claude Code and OpenCode (python3, ls, cat, head, grep, find, wc, git diff and status; file tools); Codex runs in its workspace-write sandbox. Then `check.sh`. Transcripts (`<harness>-<task>.jsonl`), stderr, diffs and checks are in `conformance/`; `results.txt` is the run's summary.
- **Speculative decoding** (`athanor bench --prompt-ids <file> -p <n> -d <n-1> -n 256|128 [--direct|--mtp] --no-hash`): the same prompt ids for the three ways, greedy. With `--mtp` the whole prompt runs through the main model and the draft head, the first token comes from its logits, and the clock runs over the generated tokens; plain decoding starts its timed steps from the prompt's last token, so it counts one more step: the difference is under 1%. `bench/*.json` are the reports, `bench/*.log` the output. `prompt-code.ids` asks for a Python function on CSV orders, `prompt-story.ids` for a short story; both chat-formatted with thinking off, tokenized with the model's tokenizer (the 0.8B's and the 35B's give the same ids).
- **Unchanged**: `athanor bench models/Qwen3.5-0.8B -p 512,4096 -n 128 -d 0 -r 5` and `athanor bench models/Qwen3.6-35B-A3B --prompt-ids prompt-short.ids|prompt-4096.ids -n 128 -r 5|3 --warmup 1 --no-hash`, step 12's commands and prompts.

## Notes

- **What the draft head guesses.** On the coding prompt the 35B keeps 98% of its guesses: code is predictable, and the head was trained on the model's own outputs. The 0.8B keeps 86%. On a story, 63% and 76%. Two tokens per pass at most (one guess); the passes of two or three tokens give each row the bits of a pass of one, which is why the output cannot change.
- **The first request of a session** runs its whole prompt: one snapshot is kept, and each harness's system prompt and tools are 7k to 13k tokens. On the 35B that is a minute before the first token; afterwards a turn runs only its new tokens. Several snapshots, or one on disk, would let a new session start from the shared prefix (Possible later work).
- **OpenCode's title request** (a short prompt with the small model, here the same model) comes first in each session and thinks for about a thousand tokens with the server's thinking default: 25 s per session (`serve.log`, the requests without tools).
- **Errors seen in the transcripts**, none affecting an outcome: Claude Code warns that the model name is unknown to it; Codex warns that it has no metadata for the model and once reports a `write_stdin` to a process that had ended.
