# src/train

Training, the last and hardest phase: every op needs a backward, memory is tighter than in inference, and correctness is measured over thousands of steps. Hand-written backward in the style of llm.c, no autograd. Phase 4.

| File | Content |
| --- | --- |
| `loop.c` | step loop, gradient accumulation, schedules, logging, evaluation hooks |
| `optim.c` | AdamW, 8-bit AdamW (block-wise states), Muon; bf16 master with stochastic rounding. The update steps that run on the GPU are ops in `src/ops/` (CPU reference, kernel, test), called from here: this module holds no GPU code (AGENTS.md invariant 11) |
| `lora.c` | LoRA and QLoRA adapters, targets chosen by role |
| `hotcold.c` | full fine-tuning beyond VRAM (below) |
| `data.c` | tokenized binary shards via mmap, sequence packing, SFT loss masking, JSONL input |
| `planner.c` | memory planner: per-layer precision, recomputation, offload and hot/cold grouping as a small mixed-integer program |
| `ckpt.c` | resumable checkpoints; export to safetensors |

## Regimes

| Regime | Models on 16 GB | Milestone |
| --- | --- | --- |
| LoRA | 0.8B–9B | T2 |
| QLoRA | 4B–27B | T3 |
| Full fine-tuning, small | 0.8B, 2B | T4 |
| Full fine-tuning, hot/cold | 4B, 9B | T5 |
| QAT, NVFP4 matmuls | any of the above | T7 |
| Pre-training | 100M–1B configs of the same architecture | T8 |
| MoE fine-tuning | 35B-A3B with RAM | T9 |
| RL with verifiable rewards (GRPO) | 0.8B–4B | T10 |

## Hot/cold layout

| Where | What |
| --- | --- |
| VRAM, cold | every block except the active one, quantized; used for the forward pass and to propagate gradients backwards |
| VRAM, hot | the active block in bf16 with gradients and full AdamW states |
| RAM | bf16 master copy of all weights |
| Block switch | write the hot block to its master, re-quantize its cold copy from the master, load the next block |

The cold copy is a cache of the master, never the truth, so quantization error does not accumulate across rounds. PCIe traffic happens per block switch, not per step. Block order, steps per block and optimizer-state persistence are configuration and research questions.

## Numerics

bf16 compute with fp32 accumulation; chunked cross-entropy fused with lm_head so logits are never materialized; activation recomputation per layer by default.
