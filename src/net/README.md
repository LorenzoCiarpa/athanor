# src/net

A minimal HTTP server for `athanor serve`, written on POSIX sockets. Introduced in step 13.

| File | Content |
| --- | --- |
| `http.c/.h` | HTTP/1.1 request parsing, keep-alive, chunked responses, server-sent events |
| `api.c/.h` | OpenAI-compatible `/v1/chat/completions` and `/v1/models`; Anthropic-compatible `/v1/messages`; streaming; tool-call passthrough as the clients expect it |

Scope: localhost and trusted networks. No TLS: put a reverse proxy in front for remote access. Targets for compatibility testing: Claude Code, Codex CLI, OpenCode.
