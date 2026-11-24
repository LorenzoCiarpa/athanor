# Proposal: a shadow sequence that follows the template's rewrite during decode

Status: **written down, not scheduled.** Proposed by the maintainer on 2026-11-24 and kept on file. It depends on multi-sequence batches (the engine of step 8) and is worth measuring before it lands. The smaller alternative is [chat-state-at-last-query.md](chat-state-at-last-query.md), which explains the problem both proposals address.

## The idea

While the model generates an answer, run a second sequence in the same decode steps: the **shadow**. It sees the conversation as the chat template will render it once the turn is history: without the thinking block, trimmed, with tool calls rendered from their parsed arguments.

When the next user message arrives, the shadow's state is exactly the state of the new prompt up to that message. Only the new message and the generation header need a prefill. The previous answer, which today runs again because the template rewrote it, needs nothing.

## Why it costs little

Decode is bound by memory bandwidth: every step reads all the weights (about 1.5 GB for Qwen3.5-0.8B), and the arithmetic waits for them. Two sequences in one step read the weights once, and the second row of every matmul is nearly free.

- The shadow does not sample, so it skips the output projection, the largest matrix: 508 MB of the 0.8B's weights, tied with the embedding.
- What it adds per step is its own state traffic: the recurrent and conv state of 18 layers (18.6 MiB for the 0.8B, about 1% of the bytes of a step) and its KV entries.
- On a GPU this should be close to free. On the CPU the arithmetic weighs more, and the cost must be measured.

Memory is modest in a hybrid model:
- **Recurrent and conv state:** one more fixed-size copy (18.6 MiB for the 0.8B, 147 MiB for the 27B).
- **KV cache:** the prefix before the turn is identical for both sequences and can be shared with paged KV (step 8); only the turn's own entries differ.

## Conditions for correctness

1. **The shadow must see the tokens of the next prompt, not the generated tokens.** The template trims the content, drops the thinking, and re-renders tool calls; the next prompt is then tokenized from text. The model does not always emit the canonical tokens: it may write `Hel` + `lo` where the tokenizer gives `Hello`. So the shadow lags one or two pre-tokenization pieces behind the generation, and feeds the canonical tokens of the text that can no longer change. It holds back trailing white space the template may trim, as the output parser already does. At the end of the answer it runs the last pieces and `<|im_end|>\n`, a prefill of a few tokens.
2. **It is checked, not trusted.** At the next turn the tokens the shadow has seen are compared with the new prompt, as `athanor chat` already does for its state. On a mismatch the chat falls back to a state copy, so the result is always the same bits as a run from the start: the forward pass gives the same bits all at once, in pieces or token by token (`tests/test_forward.c`).
3. **With tools there are two futures.** During a tool loop the turns keep their thinking block until the next real user query, and nobody knows in advance whether the next message is a tool result or a query. So two sequences are kept for the whole loop: the conversation as it is now, and as it will be after the next query. Tool results are appended to both. When a query arrives, the chat switches to the shadow.
4. **Thinking mode.** The shadow idles while the model thinks (the reasoning is dropped from history) and follows only the content after `</think>`.

## When it pays

The gain is the prefill of the previous answer at every turn:
- **CPU**, prefill at about 40 tokens per second for the 0.8B: a 500-token answer costs about 12 s at the next turn, which the shadow removes.
- **GPU**: prefill runs at thousands of tokens per second; the same answer costs tens of milliseconds, a small gain for the complexity.

So it matters most where it costs most, on the CPU during decode. The step 8 engine should measure both sides before adopting it.

## What it needs

- Multi-sequence batches in the forward pass: today `ath_model_forward` runs one sequence (`docs/ARCHITECTURE.md`); the engine of step 8 introduces batches of several sequences.
- Logits for a subset of the batch, so that the shadow skips the output projection.
- Paged KV with shared prefixes (step 8), so that the two sequences share the KV entries before the turn.
- In the chat: incremental canonical tokenization of the content (the pre-tokenization pieces that can no longer change), and the fork at the assistant header (one state copy per turn).

## Decisions it would need

1. Whether to adopt it at all, after measuring decode cost with and without the shadow on CPU and GPU.
2. Where it lives: in `athanor chat` only, or in the server's session handling (step 13), where harnesses resend whole conversations.
3. How it combines with the prefix snapshots of step 14.
