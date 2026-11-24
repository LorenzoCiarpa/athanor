# src/net

A minimal HTTP server for `athanor serve`, written on POSIX sockets. Introduced in step 13.

| File | Content |
| --- | --- |
| `http.c/.h` | HTTP/1.1 request parsing, keep-alive, chunked responses, server-sent events |
| `api.c/.h` | OpenAI-compatible `/v1/chat/completions` and `/v1/models`; Anthropic-compatible `/v1/messages`; streaming; tool-call passthrough as the clients expect it |

Tool calls: the family's `parse` turns the model output into `ath_out_event`s (text, thinking, tool-call begin, arguments, end); `api.c` converts that stream into OpenAI `tool_calls` deltas and Anthropic `tool_use` content blocks. Tool results in a request (OpenAI `tool` messages, Anthropic `tool_result` blocks) become messages with role `ATH_MSG_TOOL`, and the tools a request declares are passed to the family's `chat`. An optional session id lets the engine keep a conversation's state and improves prefix-cache hits. The server never runs a tool: that is the harness's job (`docs/ARCHITECTURE.md`, Scope: engine versus harness).

Scope: localhost and trusted networks. No TLS: put a reverse proxy in front for remote access. Targets for compatibility testing: Claude Code, Codex CLI, OpenCode.
