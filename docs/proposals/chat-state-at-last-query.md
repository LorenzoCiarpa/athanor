# Proposal: keep a state copy at the end of the last user query

Status: **written down, not scheduled.** On 2026-11-24 the maintainer asked to keep this idea on file and to leave `athanor chat` as it is for now. It may never land. If it does, it needs its own task and the decisions listed at the end. A larger variant is [chat-shadow-sequence.md](chat-shadow-sequence.md).

## The problem

`athanor chat` (`tools/chat.c`) reuses the state of the model across turns. When the new prompt starts with every token the state has seen, only the new tokens run. When it does not, the chat restores a copy of the state taken at the end of the previous prompt's last message; if that copy is not a prefix of the new prompt either, the conversation runs again from the start.

The reason the prompt does not simply extend is the chat template of Qwen3.5 (`docs/contract/qwen35-0.8b.md`, Chat template). An assistant turn shows its thinking block only if it comes after the last user message that is not just a `<tool_response>`. A turn is generated with the block, empty when thinking is off, and rendered without it once a new user query follows. The tokens change from that turn on. A recurrent state cannot be rolled back, so the only way back to an earlier position is a copy taken there.

Without tools the rewritten turn is the last one, right after the saved copy, and the copy works. With tools, the turns of a tool loop sit between the last query and the next one. When a new query arrives, the change begins at the first turn of the loop, before the saved copy, and everything runs again.

Example: a conversation with one tool, rebuilt with the reference template and tokenizer. The counts are identical to what `athanor chat --stats` measured.

| Turn | Prompt tokens | Run today | Run with this proposal |
| --- | --- | --- | --- |
| 1. "Che tempo fa a Parigi?" → tool call | 300 | 300 | 300 |
| 2. tool result → answer | 357 | 31 | 31 |
| 3. "E a Roma?" → tool call | 386 | **386** | **93** |
| 4. tool result → answer | 443 | 32 | 32 |

At turn 3 the tool definitions (about 280 tokens) and the first question run again, although they never changed.

## The idea

Keep a second copy, taken at the end of the last real user query: a user message that is not only a tool response. In the example, the copy taken at turn 1 after "Che tempo fa a Parigi?" (293 tokens) is kept when turn 2 takes its own. At turn 3 it is a prefix of the new prompt and is restored. Only the rewritten loop and the new question run: 93 tokens instead of 386.

Nothing changes in what the model sees: the prompt is still the template's, byte for byte. Only the choice of copies to keep changes.

## Doing it generically

`tools/chat.c` must not know Qwen's rule, which is family-specific: AGENTS.md, invariant 2. Two ways keep it generic:

1. **Copies at the end of every user message.** Keep a copy at the end of each user message and keep the most recent few (two are enough for the case above). At each turn, restore the latest copy whose tokens are a prefix of the new prompt. The prefix check already in `prefill` decides validity, so no template knowledge is needed.
2. **A contract hook.** The family tells where the stable prefix of a conversation ends, and the chat keeps one copy there. This is more precise, but adds a member to `ath_family` for one user.

The first is enough and needs no contract change.

## Making copies cheap

`ath_state_copy` copies every buffer, KV cache included: about 115 MiB per copy for Qwen3.5-0.8B with `--ctx 8192`. The KV entries of positions before the copy point are never overwritten while the sequence continues past that point, so a copy needs only the fixed-size buffers (the conv and recurrent state, `ATH_STATE_CONV` and `ATH_STATE_RECURRENT`) and `n_past`. That is 18.6 MiB for the 0.8B and 147 MiB for the 27B. The KV entries stay valid in the live state.

The condition is stack discipline. Restoring a copy at position p and running from there overwrites the KV entries from p on, which invalidates every copy taken at a later position. Those copies must be dropped. The state kinds are generic contract information, so a function such as `ath_state_copy_fixed` (or a flag on `ath_state_copy`) can do this without knowing the family.

## Costs

- One or two more copies per conversation: 18.6 MiB each for the 0.8B with fixed-size copies, or 115 MiB at 8k context with full copies.
- A few dozen lines in `tools/chat.c`, plus the fixed-size copy in `src/model/state.c` if adopted.
- No change to the numerics: the restored state is bit-identical to a run from the start, because prefill all at once, in pieces and token by token give the same bits (`tests/test_forward.c`).

## Decisions it would need

1. Copies at every user message (generic) or a family hook for the stable prefix.
2. Full copies, or fixed-size copies with stack discipline (a new contract function).
3. How many copies to keep.

## Relation to the plan

This is a small, host-memory case of the prefix snapshots of step 14 (`src/infer/state.c`, snapshots on disk keyed by a hash of the token prefix). Whatever lands here should use the same rule for what a snapshot contains.
