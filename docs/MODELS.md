# Model family

athanor supports one family: the **Qwen hybrid line** (Qwen3.5, Qwen3.6, Qwen3.8). This file records why, which sizes each phase targets, and the op contract every kernel must match.

## Why this family

| Criterion | Qwen hybrid line |
| --- | --- |
| One architecture from small to huge | dense 0.8B, 2B, 4B, 9B, 27B and MoE 35B-A3B, 122B-A10B, 397B-A17B in Qwen3.5; 27B and 35B-A3B in Qwen3.6; 27B and Flash-Next in Qwen3.8 |
| Dense and MoE siblings with the same blocks | yes: MoE replaces the MLP, everything else is shared |
| Small enough to train on 16 GB | 0.8B and 2B full fine-tuning; 4B and 9B with QLoRA or the hot/cold layout |
| Large enough to need placement | 35B-A3B and up require VRAM/RAM/disk tiering on consumer hardware |
| License | Qwen3.5 and Qwen3.6 open models: Apache 2.0. Qwen3.8-Flash-Next: Qwen community license |
| Comparable projects | Strata (Qwen3.6-35B-A3B, Qwen3.8-Flash-Next), Unsloth (Qwen3.5 sizes), llama.cpp, vLLM |
| Long context on small memory | three of every four layers keep a fixed-size recurrent state instead of a growing KV cache |

The alternative inside the same vendor is **Qwen3** (April 2025): classic transformer (GQA, QK-norm, RoPE, SwiGLU), dense 0.6B–32B and MoE 30B-A3B / 235B-A22B. It is simpler and is used here only as an optional bring-up model, because its full-attention layer is a subset of what the hybrid line needs.

## Target sizes per phase

| Phase | Primary | Secondary | Stretch |
| --- | --- | --- | --- |
| 1 Inference | Qwen3.5-2B, Qwen3.5-9B | Qwen3.6-35B-A3B | Qwen3.5-122B-A10B, Qwen3.8-Flash-Next |
| 2 Quantization | Qwen3.5-9B | Qwen3.6-35B-A3B | Qwen3.6-27B |
| 3 Abliteration | Qwen3.5-2B, Qwen3.5-9B | Qwen3.6-35B-A3B | — |
| 4 Training | Qwen3.5-0.8B, Qwen3.5-2B | Qwen3.5-4B, Qwen3.5-9B | Qwen3.6-35B-A3B |

## Op contract

Source of truth: the reference implementation in Hugging Face `transformers` (pinned version, file sha256 recorded) plus each model's `config.json`. Values marked *config* are read at load time and never hard-coded. Items marked **verify** must be confirmed against the reference before the corresponding kernel is written; the confirmed contract goes in `docs/contract/<model>.md`.

Status (step 2, transformers 5.24.0): every **verify** item below is resolved in [`docs/contract/qwen35-0.8b.md`](contract/qwen35-0.8b.md) (RMSNorm, RoPE, output gate, L2 norm, Gated DeltaNet output) and [`docs/contract/qwen35-35b-a3b.md`](contract/qwen35-35b-a3b.md) (router, shared expert), except the MTP head, which transformers does not implement (step 13).

### Shared

| Op | Notes |
| --- | --- |
| Token embedding | tied or untied to lm_head: *config* |
| RMSNorm | eps *config*; weight parameterization (plain or zero-centered) **verify** |
| SwiGLU MLP | gate, up, down projections; SiLU on gate |
| Final norm + lm_head | logits in fp32; chunked over vocabulary in training |
| Residual stream | bf16 storage, fp32 accumulation inside fused kernels |

### Full-attention layer (one in four)

| Op | Notes |
| --- | --- |
| Q, K, V projections | grouped-query attention, head counts *config* |
| QK-norm | per-head RMSNorm on q and k |
| RoPE | rotate-half; partial rotary dimension and theta *config*, **verify** |
| Softmax | fp32 island |
| Output gate | sigmoid gate on the attention output before o_proj (Qwen3.6 "attention output gate") **verify** shape and placement |
| KV cache | only these layers keep one; quantizable in phase 2 |

### Gated DeltaNet layer (three in four)

| Op | Notes |
| --- | --- |
| Projections | q, k, v, plus gate inputs; number of QK heads and V heads differ, head dims *config* |
| Short causal convolution | depthwise conv1d over time on q, k, v with SiLU; kernel size *config*; its state is kept at full precision |
| Normalization | L2 norm on q and k **verify** |
| Gates | decay α_t and write strength β_t per head and token |
| State update | delta rule with gating: S_t = α_t · S_{t−1} · (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ; output o_t = S_t q_t |
| Prefill / training | chunkwise-parallel form (WY / UT representation); reference kernels in flash-linear-attention |
| Decode | recurrent form, one state of size d_k × d_v per head, fixed in sequence length |
| Output | gated normalization with a learned gate, then output projection **verify** |

### MoE (MoE variants)

| Op | Notes |
| --- | --- |
| Router | linear → softmax → top-k; k and weight renormalization *config*, **verify** |
| Routed experts | SwiGLU per expert, small inner width; grouped GEMM |
| Shared expert(s) | always active, with a sigmoid gate **verify** |
| Load-balance loss | training only |

### MTP head

Multi-token prediction head used for speculative decoding in step 13; structure **verify**.

### Qwen3.8-Flash-Next extras (separate milestone)

| Block | Notes |
| --- | --- |
| Qwen Sparse Attention | block-sparse attention replacing full attention in one of every four layers; retrieval budget of 2048 tokens |
| Gated residual | residual stream widened into four branches with gated reads and writes (hyper-connections) |
| N-gram embedding | 51B-parameter lookup table, designed to live in host memory |
| Context | 262,144 tokens natively |

### Not in scope until later

Vision encoder (Qwen3.5 models are vision-language models): text path first.
