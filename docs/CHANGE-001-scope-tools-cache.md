# CHANGE-001: scope boundary, tool calling, persistent prefix cache, vision tensors

Date: 2026-11-24. Apply on top of the current repository state. Files may already differ from the original design docs: adapt each edit to what is there now, keep existing content unless this document says to replace it, and record what you did in `docs/STATUS.md` (log line + any open question).

## Context and decisions

1. **athanor is an engine plus a server, not an agent harness.** It owns everything that depends on the model or on the computation: weights, memory placement (VRAM/RAM/disk), KV and recurrent state (including on-disk persistence), prompt rendering, tool-call formatting and parsing, OpenAI/Anthropic-compatible APIs. It does **not** execute tools, run an agent loop, keep long-term agent memory, compact conversations or ship a UI. Any harness (Claude Code, OpenCode, Codex CLI, a custom one) connects through the API.
2. **Tool calling is model-specific**, so rendering tool definitions into the prompt and parsing tool calls and thinking out of the model output belong to the family (`src/models/<family>/text.c`). Converting parsed events into API formats belongs to `src/net/`.
3. **Prefix reuse persists to disk.** Snapshots of sequence state for a token prefix (KV pages for attention layers, recurrent and conv state for Gated DeltaNet layers) can be saved and resumed by hash. Hybrid models make these snapshots small.
4. **Qwen3.5 checkpoints are multimodal.** They contain vision-encoder tensors that the text path does not use; they must be bound and ignored explicitly, never silently.
5. **DwarfStar (ds4) reportedly supports Qwen3.8-Flash-Next** (source: a LinkedIn post, Nov 2026). Verify in the ds4 README before relying on it; if confirmed, ds4 becomes directly comparable on that model.

## Edits

### 1. `src/model/model.h` and `docs/ARCHITECTURE.md` (contract)

Add, and document in the contract section of `docs/ARCHITECTURE.md`:

```c
#define ATH_P_UNUSED 8u   /* present in the checkpoint, unused by the text path (e.g. vision encoder) */

typedef struct {
    const char *name;
    const char *description;
    const char *json_schema;      /* parameters schema, raw JSON */
} ath_tool_def;

typedef enum { ATH_MSG_SYSTEM, ATH_MSG_USER, ATH_MSG_ASSISTANT, ATH_MSG_TOOL } ath_msg_role;
/* ath_msg gains: role, content, optional tool calls (assistant), tool_call_id (tool results) */

typedef enum {
    ATH_OUT_TEXT, ATH_OUT_THINKING,
    ATH_OUT_TOOL_CALL_BEGIN, ATH_OUT_TOOL_CALL_ARGS, ATH_OUT_TOOL_CALL_END
} ath_out_kind;

typedef struct {
    ath_out_kind kind;
    const char  *data;  size_t len;   /* text, thinking or argument bytes */
    const char  *tool_name;           /* set on TOOL_CALL_BEGIN */
} ath_out_event;
```

Change the family interface:

```c
ath_status (*chat) (const ath_msg *msgs, int32_t n_msgs,
                    const ath_tool_def *tools, int32_t n_tools,
                    uint32_t opts,                 /* e.g. ATH_CHAT_THINKING */
                    ath_buf *out);
ath_status (*parse)(ath_parse_state *st,           /* incremental, streaming-safe */
                    const char *piece, size_t n,
                    ath_out_event *events, int32_t cap, int32_t *n_events);
```

`parse` must handle tokens split across pieces, nested or malformed markup (emit text rather than crash), and an end-of-stream flush.

### 2. `docs/ARCHITECTURE.md` (scope)

Add a section **"Scope: engine versus harness"** with this table:

| Function | Layer | In athanor |
| --- | --- | --- |
| VRAM/RAM/disk placement; KV and recurrent state; on-disk prefix snapshots | engine (`src/infer`) | yes |
| Prompt rendering, tool definitions in the template | family (`text.c`) | yes |
| Parsing text, thinking and tool calls from output | family (`text.c`) | yes |
| OpenAI/Anthropic APIs, streaming, tool-call conversion | server (`src/net`) | yes |
| Tool execution, agent loop, permissions | harness | no |
| Long-term memory, knowledge base, context compaction | harness | no |
| Desktop app, GUI | product | no |

Add to the "Common versus family-specific" table: tool rendering and output parsing are family-specific.

### 3. `AGENTS.md`

Add invariant 10: **No agent features.** athanor never executes tools, runs agent loops, stores conversation memory or compacts context. Requests in that direction are out of scope; note them under "Open questions" instead of implementing them.

### 4. `README.md`

In the scope table add the row: "Agent harness (tool execution, agent memory, UI)" → out of scope by design; use any harness through the API.

### 5. `docs/PLAN.md`

- Decisions table: add "Scope: engine + server; no agent harness — decided".
- Step 2 gate: add "vision-encoder tensors bound with `ATH_P_UNUSED`; any other unbound tensor is an error".
- Step 5: deliverable includes `chat` with tool definitions and the `parse` function; gate adds "rendered prompts byte-identical to the reference chat template for a fixture set with and without tools; parser fixtures pass".
- Step 13 gate: add "tool-calling conformance: multi-turn sessions with tool calls succeed with Claude Code, OpenCode and Codex CLI on a fixed task set".
- Step 14: add "persistent prefix snapshots on disk keyed by a hash of the token prefix; resume after restart; snapshot size reported per context length".
- Performance references: ds4 row → "possibly yes on Qwen3.8-Flash-Next (verify)".

### 6. `docs/STATUS.md`

In the step 2 notes (or the current step if already past it):

- Qwen3.5-0.8B is a vision-language checkpoint (loaded upstream with `AutoModelForMultimodalLM`). `weights.c` must recognize vision-encoder tensors by name prefix, bind them with `ATH_P_UNUSED` and report their count and size in `athanor inspect --roles`.
- `tests/golden/dump.py` must run the text path only and record which reference class and version it used.

### 7. `src/models/qwen35/README.md`

`text.c` row: "pre-tokenization, chat template with tool definitions and thinking switch, incremental output parser (thinking blocks, tool calls)". Add a note that the exact markup must be taken from the reference chat template and recorded in `docs/contract/`.

### 8. `src/net/README.md`

Add: conversion of `ath_out_event` streams to OpenAI `tool_calls` deltas and Anthropic `tool_use` content blocks; tool results from requests mapped to `ATH_MSG_TOOL`; optional session id to improve prefix-cache hits.

### 9. `src/infer/README.md`

`state.c` row: add "on-disk persistence of prefix snapshots (hash of token prefix → KV pages + recurrent and conv state), with a size budget and LRU eviction" (step 14).

### 10. `tests/`

- `tests/golden/`: a fixture generator that renders reference prompts with the upstream chat template (with and without tools, thinking on and off) and stores them byte-exact.
- `tests/test_text.c`: template rendering vs fixtures; parser fixtures including split pieces, malformed markup and multiple tool calls in one turn.

### 11. `docs/program/PROGRAM.md`

Add two rows: "Tool calling: from chat template to API" (step 5 and 13, `src/models/qwen35/text.c`, `src/net/api.c`) and "Prefix caching on disk with hybrid models" (step 14, `src/infer/state.c`).

## Acceptance

- Contract changes compile; existing tests stay green.
- All documents above updated consistently (no file still describes an integrated agent or omits the scope boundary).
- `docs/STATUS.md` log line: `2026-11-24: CHANGE-001 applied` plus anything deferred and why.
