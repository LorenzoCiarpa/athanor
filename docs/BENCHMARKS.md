# Benchmark protocol

Every performance or quality claim in athanor, in an issue, a pull request, an article or a video, is a `athanor bench` JSON report committed under `docs/results/`. No report, no claim.

## What every report records

| Field | Content |
| --- | --- |
| Hardware | GPU model and VRAM, driver, CPU, RAM size and speed, PCIe generation and lanes, storage model |
| Software | athanor commit, compiler and CUDA/ROCm versions, competitor name, version and flags |
| Model | repository, revision hash, weight file hashes, quantization and format |
| Workload | prompt set id, prompt length, generation length, batch size, context length, sampling settings |
| Method | warm-up runs, measured runs (at least 5), statistic reported (median, with min and max) |

## Inference metrics

| Metric | Definition |
| --- | --- |
| TTFT | time from request to first token |
| Prefill tok/s | prompt tokens / prefill time, at 512, 4k and 32k prompt length |
| Decode tok/s | generated tokens / decode time, at batch 1 and 4, at context 0, 4k and 32k |
| Peak VRAM, peak RAM | from the driver and the process, not from athanor's own accounting |
| Disk read | bytes read per token for MoE streaming |
| Speculative decoding | decode tok/s with and without the draft, on the same prompt ids, and the share of passes whose guess was kept (`athanor bench --mtp`, `guesses_kept`); the prompt is named, since predictable text keeps more guesses (step 13) |

## Quality metrics

| Metric | Definition |
| --- | --- |
| KL divergence | mean KL between bf16 reference and tested logits on a fixed corpus |
| Top-1 agreement | share of positions where argmax matches the bf16 reference |
| Perplexity | on a fixed held-out corpus |
| Task suite | small fixed set: math reasoning, coding, instruction following |

## Training metrics

| Metric | Definition |
| --- | --- |
| Peak VRAM, peak RAM | as above, at stated sequence length and micro-batch |
| Step time, tokens/s | median over steps after warm-up |
| Loss equivalence | loss curve vs a reference trainer on the same data, seed and hyperparameters |
| Final quality | task suite after training |

## Abliteration metrics

Refusal rate on a held-out prompt set, KL divergence on harmless prompts, task suite delta, wall-clock time.

## Comparison rules

1. Same machine, same model files, same quantization where both tools support it; otherwise the closest size, stated explicitly.
2. Each competitor runs with the configuration its own documentation recommends for that hardware.
3. Report both tools' best settings, not athanor's best against a default.
4. Publish raw logs next to the JSON.
5. A regression found later is reported in the same file, not deleted.
