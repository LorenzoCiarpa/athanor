# src/infer

The inference engine. Decode at batch 1 is limited by memory bandwidth, so the goal is to read every byte of weights once per token and nothing else. Introduced in step 8.

| File | Content | Step |
| --- | --- | --- |
| `engine.c` | load → placement → warm-up (graph capture) → prefill/decode loop | 8–10 |
| `state.c` | per-sequence state allocated from the model's `state_specs`: paged KV cache, recurrent and conv states; snapshots for prefix reuse; on-disk persistence of prefix snapshots (hash of token prefix → KV pages + recurrent and conv state), with a size budget and LRU eviction | 8, 14 |
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
