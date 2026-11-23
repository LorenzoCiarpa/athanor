# src/text

Text handling that is common to every byte-level BPE model. Introduced in step 5.

| File | Content |
| --- | --- |
| `bpe.c/.h` | byte-level BPE: vocabulary and merges loaded from `tokenizer.json`, encode/decode, special tokens |
| `unicode.h` | Unicode category and case tables needed by pre-tokenization, generated once by `tools/dev/gen_unicode.py` and committed |

Pre-tokenization (the regex-like split before BPE) and the chat template are family-specific: each family provides `pretokenize` and `chat` through the contract, implemented as hand-written C in `src/models/<family>/text.c`. No regex engine is used.

Test: exact round-trip equality with the reference tokenizer on a multilingual corpus, including emoji, code and whitespace edge cases.
