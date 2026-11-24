# athanor

> The alchemist's furnace: slow, steady heat that transforms matter.
> A single C engine to **run, quantize, abliterate and train** one family of language models on the hardware you own.

athanor runs, compresses, edits and trains large language models on one or two consumer GPUs (NVIDIA or AMD), a desktop's worth of RAM and an NVMe drive. It supports one model family at a time, on purpose: an engine that serves few models can be small, readable and faster than a generic one, because every kernel, memory layout and file format is chosen for those models only.

It follows the thesis of [DwarfStar](https://github.com/antirez/ds4), [Strata](https://github.com/Niko1221/Strata) and [colibrì](https://github.com/JustVugg/colibri) and applies it to the whole model lifecycle, not only to inference. It is developed in public, alongside a blog series and video lessons that explain each component as it is built.

## Status

Pre-alpha. Steps 1 to 4 done: the core library and `athanor inspect` read a checkpoint without copying it, the model contract binds every tensor, and the whole Qwen3.5-0.8B forward pass runs on the CPU, checked against the reference implementation. Live status: [docs/STATUS.md](docs/STATUS.md).

## What it does

| Stage | Capability |
| --- | --- |
| Inference | fast local chat and serving; dense and MoE; experts placed across VRAM, RAM and disk; OpenAI- and Anthropic-compatible API |
| Quantization | own weight formats from 2 to 8 bits, FP8 and FP4 (NVFP4, MXFP4); RTN, importance-weighted, GPTQ- and AWQ-class algorithms; GGUF interop |
| Abliteration | refusal-direction removal with automatic multi-objective search, MoE-aware |
| Training | LoRA, QLoRA, full fine-tuning beyond VRAM, quantization-aware training, small-scale pre-training, RL with verifiable rewards |
| Agent harness (tool execution, agent memory, UI) | out of scope by design: athanor renders tool definitions and parses tool calls, and any harness (Claude Code, OpenCode, Codex CLI, your own) runs the tools through the API |

## Supported models

The Qwen hybrid line (Qwen3.5, Qwen3.6, Qwen3.8): Gated DeltaNet plus gated attention, dense from 0.8B to 27B and MoE from 35B-A3B upward. Why this family and which sizes each stage targets: [docs/MODELS.md](docs/MODELS.md). Other families can be added later without touching the engine: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#adding-models).

## Principles

1. **Specialize.** Hand-written forward and backward per block type. No generic graph, no autograd.
2. **Correct before fast.** Every kernel matches a CPU reference and golden tensors from the reference implementation before it is optimized.
3. **Placement, not fitting.** A model does not need to fit in VRAM; it needs to be placed. Placement decides speed, never numerics.
4. **Minimal.** C11 and CUDA/HIP C. No dependencies beyond the CUDA and ROCm toolkits and the C library.
5. **Honest numbers.** Every claim is a reproducible `athanor bench` report ([docs/BENCHMARKS.md](docs/BENCHMARKS.md)).
6. **Permissive.** MIT.

## Architecture in one picture

```
athanor (one binary)    inspect · chat · serve · bench · quant · ablit · train
pipelines               infer · quant · ablit · train            ← never see a model family
contract                src/model/model.h: params with roles, state specs, forward, backward, hooks
families                src/models/qwen35/                        ← the only model-specific code
ops                     src/ops/: CPU reference + GPU kernel per op
backend                 cpu · gpu (CUDA and HIP from one source)
core                    arenas, tensors, mmap, json, safetensors, threads
```

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Build (from step 1)

```
make            # CPU build
make cuda       # NVIDIA, default arch sm_120
make hip        # AMD, default arch gfx1201
make test
```

Linux first. Requirements: a C11 compiler; CUDA toolkit or ROCm for GPU builds. Python is used only by contributors to generate golden test data, never at runtime.

## Documentation map

| File | Content |
| --- | --- |
| [AGENTS.md](AGENTS.md) | rules for contributors and coding agents |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | layers, contract, extension rules, directory tree |
| [docs/PLAN.md](docs/PLAN.md) | decisions, model ladder, steps and milestones, risks |
| [docs/STATUS.md](docs/STATUS.md) | current step, task list, log |
| [docs/MODELS.md](docs/MODELS.md) | model family and op contract |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | benchmark protocol |
| [docs/program/PROGRAM.md](docs/program/PROGRAM.md) | course program: articles and lessons mapped to code |

## License

MIT. Model weights keep their own licenses: Qwen3.5 and Qwen3.6 open models are Apache 2.0; Qwen3.8-Flash-Next uses a Qwen community license.

## Acknowledgements

Design references: llama.cpp/ggml, llama2.c and llm.c, DwarfStar, Strata, colibrì, KTransformers, Unsloth, Heretic, flash-linear-attention.
