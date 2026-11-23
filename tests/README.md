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

`tests/test.h` holds the runner (header-only). Each suite is one `tests/test_<area>.c` built into `build/test_<area>` with a table of `ATH_TEST_ENTRY(name)` at the bottom. Assertions: `ATH_EXPECT` (record and continue), `ATH_REQUIRE` (record and end the test), `ATH_EXPECT_OK`, `ATH_EXPECT_STATUS`, `ATH_EXPECT_EQ_INT`, `ATH_EXPECT_EQ_U64`, `ATH_EXPECT_STR_EQ`, `ATH_EXPECT_NEAR`, and `ATH_SKIP(why)`.

```
make test                          # all suites, a smoke run of athanor inspect, and make check-alloc
make DEBUG=1 test                  # same suites with ASan and UBSan, in build/debug
./build/test_core json             # only tests whose name contains "json"
ATH_LOG=error ./build/test_core    # show the library's error messages (silenced by default)
```

Synthetic files (safetensors shards, indexes, broken headers) are written by the tests into a fresh directory under `$TMPDIR` and removed afterwards. Tests that need real weights read `$ATH_MODEL_DIR` (default `models/Qwen3.5-0.8B`, set in the Makefile) and are skipped when it is absent; they check format-level invariants, so any checkpoint directory works.

| Suite | Covers |
| --- | --- |
| `test_core.c` | status and logging, arenas, files, JSON (synthetic and real `config.json`, `tokenizer_config.json`, `tokenizer.json`), dtypes and tensor views, safetensors (single file, sharded, 20 malformed headers, inconsistent indexes, the real checkpoint) |
