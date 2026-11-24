# tests

Correctness is what lets everything else be fast.

| Level | Checks | When |
| --- | --- | --- |
| Unit | core modules; each op's `_ref` on hand-computed cases | every commit |
| Golden | each op of each layer vs tensors dumped from the reference implementation on real weights (layers and logits from step 4) | every commit (requires `tests/golden/data`) |
| Cross-backend | GPU kernels and `_fast` paths vs `_ref` | every commit on GPU machines |
| End-to-end | greedy generation identical to reference for N tokens; KL on a fixed corpus | nightly |
| Gradient | each backward vs finite differences and reference autograd | phase 4 |
| Performance | bench subset; fails on regressions beyond a threshold | nightly, self-hosted |

## Golden data

`tests/golden/dump.py` (development only) runs the pinned reference implementation over a fixed prompt and one decode step and records every op it computes: inputs, its output in its own dtype (`out`) and the same function recomputed in fp32 (`out32`), with the checkpoint names of the weights and the op parameters. The data lives in `tests/golden/data/` (ignored by git; `make golden` regenerates it in seconds). `tests/golden/README.md` has the environment, the record list and the two deliberate choices (eager attention, f32 tensors kept as stored).

`test_golden` replays each record through the `_ref` function twice:

- **native**: inputs as recorded (bf16), output in the recorded dtype, compared with `out`. It measures how far athanor is from what the reference computes, roundings included.
- **exact**: inputs widened to f32, output in f32, compared with `out32`. No bf16 rounding is left on either side: this is the check of the math.

## Tolerances

Every comparison reduces to one number, `score = max_i |got_i − want_i| / (|want_i| + rms(want))`: a relative error for elements of typical size or larger and an error relative to the tensor's RMS near zero, where a relative error means nothing. One bf16 ulp is at most 2^-7 (0.0078) relative.

athanor computes in fp32 and rounds each output once. A native tolerance follows from how many times the reference rounds to bf16 along the way: once (1 ulp: the two fp32 values may straddle a rounding boundary), two or three times (2 ulps: half an ulp per extra rounding). Exact tolerances are 1e-5, raised to keep at least a factor of 4 above the worst error measured where a long reduction or the recurrent state brings it close: `matmul` and `gdn` out 2e-5, `attn` 5e-5, the `gdn` state 1e-4. Measured values: Qwen3.5-0.8B, golden data of 2026-11-24 (transformers 5.24.0, torch 2.15.0, 906 records), worst over prefill and decode.

| Op | Output | Native tolerance | Measured | Exact tolerance | Measured | Why |
| --- | --- | --- | --- | --- | --- | --- |
| `embed` | out | 0 | 0 | — | — | a copy |
| `add` | out | 0 | 0 | 0 | 0 | one rounding of the same fp32 sum |
| `conv` | state_out | 0 | 0 | — | — | the state holds inputs |
| `matmul` | out | 1 ulp | 5.2e-3 | 2e-5 | 4.2e-6 | fp32 sum in another order, one rounding |
| `rmsnorm` | out | 1 ulp | 5.5e-3 | 1e-5 | 2.6e-6 | all fp32, one rounding |
| `rmsnorm_gated` | out | 1 ulp | 5.1e-3 | 1e-5 | 7.8e-7 | same intermediate rounding as the reference, then one |
| `gdn_gates` | beta | 1 ulp | 0 | 1e-5 | 1.1e-7 | sigmoid in fp32, one rounding |
| `gdn` | out | 1 ulp | 4.6e-3 | 2e-5 | 4.7e-6 | prefill: the reference uses the chunked form, same sums in another order |
| `swiglu` | out | 2 ulps | 7.6e-3 | 1e-5 | 2.0e-7 | the reference rounds silu, then the product |
| `sigmoid_mul` | out | 2 ulps | 7.3e-3 | 1e-5 | 2.1e-7 | the reference rounds the sigmoid, then the product |
| `conv` | out | 2 ulps | 8.4e-3 | 1e-5 | 1.6e-7 | the reference rounds the convolution, then silu |
| `rope` | out | 2 ulps | 9.4e-3 | 1e-5 | 2.8e-7 | the reference rounds both products and their sum |
| `l2norm` | out | 1e-5 | 6.7e-7 | 1e-5 | 6.7e-7 | f32 on both sides |
| `gdn_gates` | g | 1e-5 | 1.1e-7 | 1e-5 | 1.1e-7 | f32 on both sides |
| `gdn` | state_out | 1e-4 | 1.4e-5 | 1e-4 | 1.0e-5 | f32 state accumulating 75 tokens of updates |
| `attn` | out | 0.2 | 9.8e-2 | 5e-5 | 9.7e-6 | the eager reference rounds `q·kᵀ` to bf16 before the softmax: up to 2^-8 of a score, 0.06 at the largest score of the prompt (16.3), up to 6% on a probability; athanor keeps scores in fp32 |

A score above its tolerance fails the test and names the record and the element. `ATH_GOLDEN_WORST=1 ./build/test_golden` also prints the worst record of every row. Quantized formats (phase 2) will be compared through dequantized values, and with KL at model level.

## Runner

A minimal C runner: `ATH_TEST(name) { ... }`, `ATH_EXPECT_*` assertions, a summary line, non-zero exit on failure. No external test framework.

`tests/test.h` holds the runner (header-only); `tests/util.h` the helpers the suites share (scratch directory, location of the real checkpoint). Each suite is one `tests/test_<area>.c` built into `build/test_<area>` with a table of `ATH_TEST_ENTRY(name)` at the bottom. Assertions: `ATH_EXPECT` (record and continue), `ATH_REQUIRE` (record and end the test), `ATH_EXPECT_OK`, `ATH_EXPECT_STATUS`, `ATH_EXPECT_EQ_INT`, `ATH_EXPECT_EQ_U64`, `ATH_EXPECT_STR_EQ`, `ATH_EXPECT_NEAR`, and `ATH_SKIP(why)`.

```
make test                          # all suites, smoke runs of athanor inspect (with and without --roles), and make check-alloc
make DEBUG=1 test                  # same suites with ASan and UBSan, in build/debug (test_golden takes about 2.5 minutes there, 4 s in release)
./build/test_core json             # only tests whose name contains "json"
ATH_LOG=error ./build/test_core    # show the library's error messages (silenced by default)
```

Synthetic files (safetensors shards, indexes, broken headers) are written by the tests into a fresh directory under `$TMPDIR` and removed afterwards. Tests that need real weights read `$ATH_MODEL_DIR` (default `models/Qwen3.5-0.8B`, set in the Makefile) and are skipped when it is absent; except `test_golden`, they check format-level invariants, so any checkpoint directory works. `test_golden` reads `$ATH_GOLDEN_DIR` (default `tests/golden/data/qwen35-0.8b`) and needs the checkpoint the data was made from.

| Suite | Covers |
| --- | --- |
| `test_core.c` | status and logging, arenas, files, JSON (synthetic and real `config.json`, `tokenizer_config.json`, `tokenizer.json`), dtypes and tensor views, safetensors (single file, sharded, 20 malformed headers, inconsistent indexes, the real checkpoint) |
| `test_model.c` | role names and the registry; qwen35 config parsing (dense, MoE, text-only, schedule fallback, 20 rejected configs); binding of tiny synthetic checkpoints (dense, MoE with fused experts, text-only) and every failure mode (missing, unexpected, misshaped, integer dtype, tie mismatches); the real checkpoint (coverage, schedule against `layer_types`, residual writers) |
| `test_ops.c` | every `_ref` function of `src/ops/` on hand-computed cases (bf16 rounding, norms, RoPE pairing and relative positions, attention averaging and head grouping, conv taps, the delta rule step by step, gates around the softplus threshold) and on exact properties: streaming equals all at once for conv and Gated DeltaNet, decode equals the last prefill row for attention, causality, argument checks |
| `test_golden.c` | every op the reference computes on Qwen3.5-0.8B (24 layers, prefill and decode, 906 records), replayed through the `_ref` functions on the real weights, native and exact (above) |
