# src/infer

The inference engine. Decode at batch 1 is limited by memory bandwidth, so the goal is to read every byte of weights once per token and nothing else. Introduced in step 8.

| File | Content | Step |
| --- | --- | --- |
| `engine.h`, `engine.c` | a model on a GPU: the device opened, every param the forward pass reads copied there once, the `ath_exec` a GPU pass needs (device, weights, scratch), sequence states on the device; later placement, warm-up (graph capture) and the prefill/decode loop | 8 (load, states, exec), 9–10 |
| `state.c` | paged KV cache, snapshots for prefix reuse; on-disk persistence of prefix snapshots (hash of token prefix → KV pages + recurrent and conv state), with a size budget and LRU eviction. The plain state of one sequence on a device is `ath_state_init_gpu` in `src/model/state.c` (step 8), next to the host one | 13, 14 |
| `sched.c` | request queue, small continuous batching for the server | 13 |
| `tier.c` | MoE expert placement across VRAM, pinned RAM, RAM and disk; usage statistics; copy-or-compute-on-CPU cost model; router-ahead prefetch | 12 |
| `spec.c` | speculative decoding with the MTP head | 13 |

## Sequence state

| Layer type | State | Size vs context |
| --- | --- | --- |
| Full attention (one in four) | paged KV cache, quantizable | linear |
| Gated DeltaNet (three in four) | recurrent matrix per head + conv state | constant |

Hybrid models make prefix snapshots small: three layers in four keep a fixed-size recurrent and conv state, and only the attention layers keep KV pages that grow with the prefix. A snapshot saved on disk survives a restart, so a harness that resends a long system prompt and tool list resumes it instead of recomputing it (step 14).

## MoE placement

| Tier | Holds |
| --- | --- |
| VRAM | dense part, shared experts, hottest routed experts (budgeted cache: learned pin set + LRU) |
| Pinned RAM | staging ring for asynchronous copies |
| RAM / page cache | remaining experts; may be computed on CPU instead of copied |
| Disk | everything else, read asynchronously |

When an expert is not in VRAM, `tier.c` chooses between copying it and computing it on the CPU using measured PCIe bandwidth, RAM bandwidth and CPU throughput: in decode the CPU often wins, in prefill the GPU does. Placement never changes results.

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
