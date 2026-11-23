# tests

Correctness is what lets everything else be fast.

| Level | Checks | When |
| --- | --- | --- |
| Unit | core modules; each op's `_ref` on hand-computed cases | every commit |
| Golden | each op and each layer vs tensors dumped from the reference implementation on real weights | every commit (requires `tests/golden/data`) |
| Cross-backend | GPU kernels and `_fast` paths vs `_ref` | every commit on GPU machines |
| End-to-end | greedy generation identical to reference for N tokens; KL on a fixed corpus | nightly |
| Gradient | each backward vs finite differences and reference autograd | phase 4 |
| Performance | bench subset; fails on regressions beyond a threshold | nightly, self-hosted |

## Golden data

`tests/golden/dump.py` (development only) loads a model with a pinned version of the reference implementation, runs fixed prompts, and writes per-op inputs and outputs with hashes into `tests/golden/data/` (ignored by git; regenerated locally). The pinned version and file hashes are recorded in `docs/contract/`.

## Tolerances

| Dtype | Default |
| --- | --- |
| fp32 | rtol 1e-5 |
| bf16 | rtol 1e-2 plus a max-ulp check |
| quantized | through dequantized values, and KL at model level |

## Runner

A minimal C runner: `ATH_TEST(name) { ... }`, `ATH_EXPECT_*` assertions, a summary line, non-zero exit on failure. No external test framework.
