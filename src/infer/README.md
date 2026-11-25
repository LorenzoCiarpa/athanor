# src/infer

The inference engine. Decode at batch 1 is limited by memory bandwidth, so the goal is to read every byte of weights once per token and nothing else. Introduced in step 8.

| File | Content | Step |
| --- | --- | --- |
| `engine.h`, `engine.c` | a model on a GPU: the device opened, every param the forward pass reads copied there once, the `ath_exec` a GPU pass needs (device, weights, scratch), sequence states on the device; the decoder, one token at a time in a recorded graph with the choice on the device (step 9); the routed experts of a MoE model handed to a placement (`ath_engine_open_tier`, step 12), and the placement updated between decoder steps | 8 (load, states, exec), 9 (decoder), 12 |
| `state.c` | paged KV cache, snapshots for prefix reuse; on-disk persistence of prefix snapshots (hash of token prefix → KV pages + recurrent and conv state), with a size budget and LRU eviction. The plain state of one sequence on a device is `ath_state_init_gpu` in `src/model/state.c` (step 8), next to the host one | 13, 14 |
| `sched.c` | request queue, small continuous batching for the server | 13 |
| `tier.h`, `tier.c` | MoE expert placement (`ath_placement`, step 12): every routed expert in pinned RAM, regrouped for the CPU; slots in VRAM for as many as fit; the tables a pass reads (`ath_exec.experts`); the slots follow use between passes. Disk and a copy-or-compute cost model: not yet (below) | 12 |
| `spec.c` | speculative decoding with the MTP head | 13 |

## Sequence state

| Layer type | State | Size vs context |
| --- | --- | --- |
| Full attention (one in four) | paged KV cache, quantizable | linear |
| Gated DeltaNet (three in four) | recurrent matrix per head + conv state | constant |

Hybrid models make prefix snapshots small: three layers in four keep a fixed-size recurrent and conv state, and only the attention layers keep KV pages that grow with the prefix. A snapshot saved on disk survives a restart, so a harness that resends a long system prompt and tool list resumes it instead of recomputing it (step 14).

## MoE placement (step 12)

A MoE model larger than the GPU (Qwen3.6-35B-A3B in Q8_0: 2.6 GB of attention, Gated DeltaNet, shared experts, embeddings and head, 34.2 GB of routed experts) keeps its routed experts in RAM and a share of them in VRAM. `tier.c` (`ath_placement`):

| Where | Holds |
| --- | --- |
| VRAM, the engine's copies | every param but the routed experts |
| VRAM, slots | as many experts per layer as fit after a reserve (2 GiB by default, for states and working memory), the same number in every layer: 73 to 75 of 256 on the RTX 5080 |
| Pinned RAM | every routed expert, in Q8_0_R16 (core/tensor.h: the CPU's layout, which the GPU reads too, so that moving an expert into a slot is one asynchronous copy) |

A pass computes the experts in slots on the GPU and the others on the CPU from pinned RAM, in parallel (`ath_moe_experts_split`, `src/ops/moe.c`), with the same bits whichever computes an expert (AGENTS.md invariant 7, option (b)): the logits do not depend on the number of slots or on what they hold (`tests/test_engine.c`, `engine_moe_placement`). The CPU's share counts how often each expert is chosen; every `period` passes (`ath_place_update`, called by the decoder between steps) the most chosen experts of a layer move into the slots of the least chosen, `moves` at a time, and the counts age by a quarter. A move is a copy on the host-to-device stream and a change of two tables (the device's, which the kernels read, and the host's flags, which the CPU reads); no pass is running then, so a captured decode step simply sees the new places at its next replay.

Not done in step 12: a disk tier (a model whose experts do not fit in RAM), a cost model choosing between copying an expert and computing it on the CPU (prompts compute the experts outside VRAM on the CPU too, which is slow for long prompts), prefetching the next layer's experts from the router's choice.

## Interfaces

C API in `include/athanor.h` (load, session, prefill, decode, sample, free); `athanor chat`; `athanor serve` through `src/net`.

## The engine (step 8)

```c
ath_engine e;                                          /* must stay where it is while open */
ATH_CHECK(ath_engine_open(&e, &model, 0, &arena));     /* device 0; 1.40 GiB for Qwen3.5-0.8B */
ath_state s;
ATH_CHECK(ath_engine_state(&e, &s, ATH_DTYPE_BF16, 8192, &arena));
ATH_CHECK(ath_model_forward(&model, &batch, &s, NULL, &scratch, &e.exec));
ath_engine_close(&e);
```

- **Weights.** Every param not flagged `ATH_P_UNUSED` is copied from the memory-mapped checkpoint to the device, once, with its dtype and bytes unchanged (bf16). The copies are indexed like `m->params`, which is how a family's forward pass finds them (`ath_exec.weights`). The vision encoder and the MTP head stay on the host.
- **Memory.** Three device arenas: the weights for the engine's life, the states, and the scratch memory of one pass, released after every call.
- **One sequence per batch.** Batches of several sequences, paged KV memory, graph capture and sampling on the GPU come with fast decode and the server (steps 9 and 13).
- **Checked** by `tests/test_engine.c` (the copies byte for byte, states on the device, sequences in pieces, and the gate: greedy generation against the reference on 50 prompts of 256 tokens) and by the GPU runs of `tests/test_forward.c` (`tests/README.md`, Engine).

## The decoder (step 9)

`ath_decoder` runs one token at a time on the engine's device. A step is the forward pass of the token as a device batch (`ATH_BATCH_DEVICE`: token, position and logits in device memory, so nothing in the pass depends on how many tokens the state holds) followed by the choice of the next token on the GPU (`ath_sample_gpu`, greedy or top-k up to 64; otherwise the logits are copied to pinned memory and `ath_sample_ref` chooses on the host). The first step runs that work directly, which also sizes the scratch memory; the second records it in a graph (`ath_graph`, `src/backend/README.md`); every later step writes the token, its position and the uniform number into pinned memory, launches the graph and reads 4 bytes back. A replayed step advances `n_past` as `ath_model_forward` would.

- **One launch per token.** Some 300 kernels, recorded once. The step 8 decode launched them one by one from the host and waited for the logits at every token.
- **The same bits.** A replayed step gives the logits of the same step run directly (`tests/test_engine.c`, `engine_decoder`), and the tokens `ath_sample_ref` chooses with the same uniform numbers.
- **What may change between steps.** The state may be reset or copied into, and other forward calls on it may come between steps: its buffers and the engine's scratch memory stay where they are, which is what the recording relies on. Another state or another sampler needs another decoder.
