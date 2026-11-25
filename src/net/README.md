# src/net

`athanor serve`: the model behind the HTTP APIs of coding agents, on POSIX sockets. Step 13.

| File | Content |
| --- | --- |
| `http.c/.h` | HTTP/1.1: requests with a `Content-Length` or chunked body, `Expect: 100-continue`, keep-alive; whole responses or server-sent events in chunks; one `poll` loop over the listening socket and up to 32 connections |
| `api.c/.h` | the three APIs: each request parsed into one `ath_api_request` (`ath_api_parse`, tested on fixtures), the generation's events written back in the request's format, streamed or whole; each API's error shapes |
| `gen.c/.h` | generation for the server, whatever the API: the prompt (from a snapshot when it extends one), decoding (the GPU decoder, or the draft head's speculative decoder with `--mtp`), stop strings, whole UTF-8 characters, cancellation when the client goes away |

## Endpoints

| Endpoint | API | Used by |
| --- | --- | --- |
| `POST /v1/messages` | Anthropic Messages: `text`, `thinking` and `tool_use` content blocks, streamed as `message_start`, `content_block_*`, `message_delta`, `message_stop` (and `ping` while a long prompt runs) | Claude Code 2.1.332 |
| `POST /v1/messages/count_tokens` | its token count | Claude Code |
| `POST /v1/responses` | OpenAI Responses: `reasoning`, `message`, `function_call` and `custom_tool_call` output items, streamed as `response.*` events with sequence numbers | Codex CLI 0.167.0 (`wire_api = "responses"`) |
| `POST /v1/chat/completions` | OpenAI Chat Completions: `content`, `reasoning_content` and `tool_calls` deltas, `[DONE]`, usage with `stream_options.include_usage` | OpenCode 1.19.6 and most other clients |
| `GET /v1/models`, `/v1/models/<id>` | the one model, in a shape both families of clients read | all |
| `GET /health` | `{"status": "ok"}` | |

Paths work with and without the `/v1` prefix; a query string is ignored (Claude Code appends `?beta=true`).

## Requests into messages

- **System messages.** The chat template takes one system message, first. System and developer messages before any other (Anthropic's `system` blocks, OpenAI's `system` and `developer` messages, Responses' `instructions` and leading `developer` items) are joined into it; one later in the conversation becomes a user message where it stands (Claude Code 2.1.332 sends its environment as a `system` message after the first user message). Claude Code's `x-anthropic-billing-header` line, which Anthropic's API takes out of the prompt, is dropped.
- **Content.** The texts of a content array are joined as the template joins them (directly); images and documents become a one-line note (the server runs the model on text).
- **Tool calls and results.** Anthropic `tool_use` and `tool_result` blocks, OpenAI `tool_calls` and `tool` messages, Responses `function_call` / `function_call_output` and `custom_tool_call` / `custom_tool_call_output` items become `ath_tool_call`s and `ATH_MSG_TOOL` messages. Responses' reasoning items come back as the assistant's thinking (their `content`, else their `summary`). Arguments that are not a JSON object become `{}`, with a warning.
- **Tools.** Function tools are passed to the family's chat template. A Responses `custom` (freeform) tool is declared to the model with one string parameter `input`, and the model's call comes back as a `custom_tool_call` with that string. Server tools (web search, `namespace` groups) are not passed. `tool_choice: none` passes no tools; forcing a tool is not supported (no constrained decoding).
- **Thinking.** Anthropic `thinking.type` (`enabled`, `adaptive`, `disabled`), OpenAI `chat_template_kwargs.enable_thinking` then `reasoning_effort` (`none` turns it off), Responses `reasoning.effort` (`none` and `minimal` turn it off); otherwise the server's default (`--no-think`).
- **Sampling.** `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, and what a request leaves out from the model card: with thinking 0.6, 0.95, 20, presence 0; without 0.7, 0.8, 20, presence 1.5. `--server-sampling` ignores the requests' values. Limits: `max_tokens`, `max_completion_tokens`, `max_output_tokens`; stop strings: `stop_sequences`, `stop` (8 at most).

## The prompt's state

A recurrent state cannot be rolled back, so `gen.c` keeps a snapshot after each prompt's stable part (the conversation up to its last message, as `athanor chat` does) and a request goes on from:

1. the state itself, when it holds a prefix of the new prompt (an agent's loop of tool calls usually does: the model's turn renders back as it was generated);
2. else the snapshot, when its tokens are a prefix of the new prompt;
3. else the start.

A prompt under a quarter of the snapshot's length (an agent's side request: OpenCode's title, a summary) does not replace it, unless the snapshot went unused twice in a row (another conversation). With Claude Code, Codex CLI and OpenCode the first request of a session runs its whole prompt (13k, 7k and 11k tokens with Qwen3.6-35B-A3B: about a minute at the 35B's prefill speed), and every later one runs only what it adds. Several snapshots, and prefix snapshots on disk, are later work (`docs/STATUS.md`).

## Running it

```sh
athanor serve models/Qwen3.6-35B-A3B --ctx 65536            # http://127.0.0.1:8080
```

The server keeps two states of `--ctx` tokens on the GPU (state and snapshot: 1.31 GiB each at 64k for the 35B) and gives the experts the rest. Clients, with their configuration kept apart from the user's (as `tests/conformance/run.sh` does):

```sh
# Claude Code
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=local ANTHROPIC_MODEL=qwen3.6-35b-a3b \
ANTHROPIC_SMALL_FAST_MODEL=qwen3.6-35b-a3b claude

# Codex CLI: in $CODEX_HOME/config.toml
#   model = "qwen3.6-35b-a3b"
#   model_provider = "athanor"
#   [model_providers.athanor]
#   base_url = "http://127.0.0.1:8080/v1"
#   wire_api = "responses"
#   env_key = "ATHANOR_KEY"

# OpenCode: a provider in opencode.json
#   "athanor": {"npm": "@ai-sdk/openai-compatible",
#               "options": {"baseURL": "http://127.0.0.1:8080/v1", "apiKey": "local"},
#               "models": {"qwen3.6-35b-a3b": {"reasoning": true, "tool_call": true}}}
```

`--dump <dir>` saves every request body, to see what a client sends. A line per request on stderr gives the prompt's tokens (and how many came from the state), the prefill time, the answer's tokens and speed, and how it ended.

## Conformance (13.7)

`tests/conformance/`: five tasks (fix a failing test, write a file and run it, rename a function across files, find a constant and write the answer, compute a mean from a CSV), each with a script that checks the outcome, run through each harness by `run.sh`. With Qwen3.6-35B-A3B on an RTX 5080: 15 of 15 pass (`docs/results/2026-11-25-step13-server/`).

## Limits

One sequence at a time: requests that arrive while one runs wait in their sockets. No TLS and no keys: it listens on 127.0.0.1 unless `--host` says otherwise; put a reverse proxy in front for remote access. The server never runs a tool (AGENTS.md, invariant 10): the harness does, and sends the results back.
