# tests

Correctness is what lets everything else be fast.

| Level | Checks | When |
| --- | --- | --- |
| Unit | core modules; each op's `_ref` on hand-computed cases | every commit |
| Golden | each op of each layer vs tensors dumped from the reference implementation on real weights; the whole forward pass (residual stream at every layer, logits, KL) on fixed prompts | every commit (requires `tests/golden/data`) |
| Cross-backend | GPU kernels and `_fast` paths vs `_ref` | every commit on GPU machines |
| End-to-end | greedy generation identical to reference for N tokens; KL on a fixed corpus | nightly |
| Gradient | each backward vs finite differences and reference autograd | phase 4 |
| Performance | bench subset; fails on regressions beyond a threshold | nightly, self-hosted |

## Golden data

`tests/golden/dump.py` (development only) runs the pinned reference implementation over a fixed prompt and one decode step and records every op it computes (the forward records of step 4 are described under [Forward pass](#forward-pass)): inputs, its output in its own dtype (`out`) and the same function recomputed in fp32 (`out32`), with the checkpoint names of the weights and the op parameters. The data lives in `tests/golden/data/` (ignored by git; `make golden` regenerates it in seconds). `tests/golden/README.md` has the environment, the record list and the two deliberate choices (eager attention, f32 tensors kept as stored).

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

## Forward pass

`test_forward` reaches the forward pass only through `src/model/model.h`, like a pipeline.

On tiny random checkpoints (`tests/tiny.h`): state specs and their sizes, every argument check of `ath_model_forward`, the hooks, and the property every pipeline relies on: a sequence run all at once, in pieces (5 + 1 + 6 tokens) or one token at a time gives the same logits and the same state, bit for bit, with bf16 and with f32 activations; the logits of the last token alone equal the last row; a reset state runs the sequence again from scratch.

On the real checkpoint, against the forward records of the golden data: two prompts (the op prompt, 75 tokens, and a chat-formatted one, 28 tokens), each followed by 8 greedy decode steps of the bf16 reference, and the same tokens through the reference loaded in fp32. athanor runs each prompt twice, the prompt in one call and then one token at a time, in four parallel threads that share one pool for the matmuls (about 11 s; 36 s before `ath_matmul_fast`, task 4.7, which changed no number below):

| Run | Check | Measured (2026-11-24) |
| --- | --- | --- |
| bf16 activations | **gate**: mean KL against the fp32 reference over every position of every prompt no larger than the bf16 reference's | 3.0e-4 against 5.0e-4 (ratio 0.60) |
| bf16 activations | **gate**: the greedy token equals the fp32 reference's wherever the fp32 top-2 margin exceeds the bf16 noise | 0 differences at 104 such positions; noise 0.21 |
| f32 activations | residual stream entering every layer and the final norm within 2e-4 of the fp32 reference | 4.7e-5 |
| f32 activations | logits within 1e-4 of the fp32 reference | 2.7e-5 |

- **KL.** KL(p₃₂ ‖ p) per position, in double precision, from the logits of the fp32 reference and of the run under test. The bf16 reference is held to the same measure; `test_forward` checks that its KL equals the one `dump.py` computed with torch.
- **Bf16 noise of a greedy decision.** For the two tokens the fp32 reference ranks first, the largest change of their logit difference between the fp32 and the bf16 reference over all positions. Where the fp32 margin is smaller than that, rounding alone can flip the choice, and the reference itself flips it twice on the prose prompt.
- **f32 tolerances.** Both sides compute in fp32 but sum in different orders (athanor's dot products in order of k, torch's blocked ones; the reference's prefill uses the chunked Gated DeltaNet). The errors of single ops are 1e-7 to 1e-5 (Tolerances, exact) and grow through 24 layers; the tolerances are about four times the worst measured value, as for the exact op tolerances.
- **Why the gate pools the prompts.** A mean over the positions of one short prompt is dominated by its worst-conditioned positions. On the chat prompt, position 0 (`<|im_start|>`) goes through a near-cancellation in layer 15 that turns one bf16 rounding into a 10% error, and its KL ranges from 2.7e-4 (the eager reference) to 5.5e-3 (the same reference with `A_log` rounded to bf16, the transformers default) and 7.3e-3 (athanor). Per prompt, athanor's mean KL is 1.7e-4 against 5.3e-4 on prose and 6.0e-4 against 4.4e-4 on chat (4.1e-4 against 4.5e-4 without position 0). The reference with sdpa attention, the transformers default, would fail a per-prompt comparison with the eager reference on chat too: 7.5e-4 against 5.1e-4 on the prefill. `docs/contract/qwen35-0.8b.md`, End-to-end, has the analysis.

The bf16 run also reports the residual stream at every layer against both references, as a diagnostic, not gated: against the fp32 run it is 2e-2 to 3e-2 after the first layer and at most 0.08 (prose) and 0.18 (chat), where the bf16 reference's own distance is at most 0.14 and 0.15.

```
./build/test_forward forward_golden                  # the real-checkpoint comparison only
ATH_FORWARD_POSITIONS=6 ./build/test_forward         # only the first 6 positions of each prompt (no gate)
ATH_FORWARD_LAYERS=1 ./build/test_forward golden     # residual stream per layer, every run
ATH_FORWARD_DETAIL=1 ./build/test_forward golden     # KL, margin and greedy choice per position
```

A `DEBUG=1` build runs the first 4 positions of each prompt (three in one call, one alone) and checks the f32 tolerances; the gate needs every position.

## Text

`tests/golden/text.py` records, with the pinned reference (transformers 5.24.0, tokenizers 0.23.3), what `test_text` compares exactly:

- **Tokenizer.** For 188 texts, 28 million tokens in all: the ids the reference gives and the text it decodes them to. Hand-written cases (languages, scripts, emoji sequences, code, every kind of white space, contractions in every case, added tokens inside text, normalization), files of this repository and the model card, every Unicode code point but the surrogates in five contexts (`a·`, ` ·`, `·1`, `·!`, `·'`), and every canonical decomposition, decomposed, with up to three extra marks in shuffled order. athanor must give the same ids and decode them to the same bytes: no tolerance.
- **Chat template.** 20 conversations rendered by `apply_chat_template`: with and without system, thinking on and off, history with thinking in the content or apart, tools with and without system, tool calls with typed arguments and their results, tool responses written as user messages, trimming, Unicode; 3 of them the template rejects. athanor must render the same bytes (and give the same ids), or fail where the template raises.
- **Generation.** A chat prompt, 32 greedy steps of the fp32 reference with the top-2 margin of each step, and the bf16 noise of that margin (its largest change from fp32 to the bf16 reference on the same tokens). athanor, with bf16 activations and its own template and tokenizer, must choose the reference's token at every step until the first one whose margin is within the noise (step 27 of 32 for this prompt); after it the two may part legitimately, and the test reports how many tokens are equal in all (32). A `DEBUG=1` build runs 4 steps.

## Runner

A minimal C runner: `ATH_TEST(name) { ... }`, `ATH_EXPECT_*` assertions, a summary line, non-zero exit on failure. No external test framework.

`tests/test.h` holds the runner (header-only); `tests/util.h` the helpers the suites share (scratch directory, location of the real checkpoint); `tests/tiny.h` the tiny synthetic Qwen3.5 checkpoints (`config.json` and safetensors, zero-filled or pseudo-random, with mutations for the failure tests) that `test_model` and `test_forward` share. Each suite is one `tests/test_<area>.c` built into `build/test_<area>` with a table of `ATH_TEST_ENTRY(name)` at the bottom. Assertions: `ATH_EXPECT` (record and continue), `ATH_REQUIRE` (record and end the test), `ATH_EXPECT_OK`, `ATH_EXPECT_STATUS`, `ATH_EXPECT_EQ_INT`, `ATH_EXPECT_EQ_U64`, `ATH_EXPECT_STR_EQ`, `ATH_EXPECT_NEAR`, and `ATH_SKIP(why)`.

```
make test                          # all suites, smoke runs of athanor inspect (with and without --roles), and make check-alloc; about 17 s
make DEBUG=1 test                  # same suites with ASan and UBSan, in build/debug; about 4 minutes
                                   # (test_golden 2.6 minutes there, 5 s in release; test_forward 34 s on 4 positions, 11 s on all;
                                   # test_text 3 s in release, its generation 4 steps instead of 32 there)
./build/test_core json             # only tests whose name contains "json"
ATH_LOG=error ./build/test_core    # show the library's error messages (silenced by default)
```

Synthetic files (safetensors shards, indexes, broken headers) are written by the tests into a fresh directory under `$TMPDIR` and removed afterwards. Tests that need real weights read `$ATH_MODEL_DIR` (default `models/Qwen3.5-0.8B`, set in the Makefile) and are skipped when it is absent; except `test_golden`, they check format-level invariants, so any checkpoint directory works. `test_golden` reads `$ATH_GOLDEN_DIR` (default `tests/golden/data/qwen35-0.8b`) and needs the checkpoint the data was made from.

| Suite | Covers |
| --- | --- |
| `test_core.c` | status and logging, arenas, files, JSON (synthetic and real `config.json`, `tokenizer_config.json`, `tokenizer.json`), dtypes and tensor views, safetensors (single file, sharded, 20 malformed headers, inconsistent indexes, the real checkpoint), the thread pool (every task once, with fewer, as many and more tasks than threads; runs from four threads at once on one pool) |
| `test_model.c` | role names and the registry; qwen35 config parsing (dense, MoE, text-only, schedule fallback, 20 rejected configs); binding of tiny synthetic checkpoints (dense, MoE with fused experts, text-only) and every failure mode (missing, unexpected, misshaped, integer dtype, tie mismatches); the real checkpoint (coverage, schedule against `layer_types`, residual writers) |
| `test_ops.c` | every `_ref` function of `src/ops/` on hand-computed cases (bf16 rounding, norms, RoPE pairing and relative positions, attention averaging and head grouping, conv taps, the delta rule step by step, gates around the softplus threshold) and on exact properties: streaming equals all at once for conv and Gated DeltaNet, decode equals the last prefill row for attention, causality, argument checks; `ath_matmul_fast` bit for bit against `ath_matmul_ref` (7 shapes × 8 dtype combinations, without and with threads) |
| `test_golden.c` | every op the reference computes on Qwen3.5-0.8B (24 layers, prefill and decode, 906 records), replayed through the `_ref` functions on the real weights, native and exact (above); every matmul record also through `ath_matmul_fast` on 8 threads, required bit-identical to `ath_matmul_ref` |
| `test_forward.c` | the forward pass through the contract: state specs and sizes, argument checks, hooks, all at once = in pieces = token by token on tiny random checkpoints; the real checkpoint against the forward records (above) |
| `test_text.c` | text, through the contract only: UTF-8, character classes, NFC, the Python-style JSON writer, pre-tokenization pieces, the output parser (17 fixtures, each also split at every byte and fed byte by byte), template errors, sampling (hand-computed cases and a distribution with a fixed seed); against the text golden data (below): the corpus, the templates, the greedy chat generation |
