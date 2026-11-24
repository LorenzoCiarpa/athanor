/* bpe.h: byte-level BPE, read from a Hugging Face tokenizer.json.
 *
 * Encoding, as Hugging Face tokenizers does it for byte-level BPE models:
 *
 *   1. added tokens (<|im_start|>, <think>, ...) are found in the raw text,
 *      leftmost first and longest at the same place, and become their ids;
 *   2. each stretch of text between them is normalized (NFC, if the
 *      tokenizer asks for it);
 *   3. pre-tokenization splits it into pieces: a function of the model
 *      family (a hand-written version of the family's regex), passed in;
 *   4. every piece is BPE-merged on its bytes: start from one token per
 *      byte, then merge the adjacent pair with the lowest merge rank (the
 *      leftmost of equals) until no pair has a merge.
 *
 * Decoding concatenates the bytes of the tokens (added tokens as their
 * text). The vocabulary in tokenizer.json is written in the byte-to-
 * character alphabet of GPT-2 (a space is "Ġ"); it is mapped back to bytes
 * at load, so tokens here are plain bytes.
 *
 * Supported tokenizer.json: model BPE without unknown token, byte fallback,
 * dropout, subword prefix or suffix; normalizer NFC or none; pre-tokenizer
 * a Split on one regex (behavior Isolated) followed by ByteLevel without
 * prefix space or its own regex; decoder ByteLevel. The loader checks that
 * the Split pattern is exactly the one the family's split implements.
 * Anything else is ATH_ERR_UNSUPPORTED.
 */
#ifndef ATH_BPE_H
#define ATH_BPE_H

#include "core/arena.h"
#include "core/base.h"
#include "core/buf.h"

/* The pieces of a text, as pre-tokenization returns them: piece i is
 * [end[i - 1], end[i]), the first one starts at 0, and the last one ends at
 * the length of the text. */
typedef struct {
    size_t    *end;
    size_t     n, cap;
    ath_arena *a;
} ath_spans;

void       ath_spans_init(ath_spans *s, ath_arena *a);
ath_status ath_spans_push(ath_spans *s, size_t end);

/* Pre-tokenization of a family: split text[0 .. n) into pieces, appended
 * to out (which may already hold pieces of other text: the offsets are
 * relative to this text). */
typedef ath_status (*ath_pretokenize_fn)(const char *text, size_t n, ath_spans *out);

/* A growable list of token ids in an arena. */
typedef struct {
    int32_t   *ids;
    size_t     n, cap;
    ath_arena *a;
} ath_tokens;

void       ath_tokens_init(ath_tokens *t, ath_arena *a);
ath_status ath_tokens_push(ath_tokens *t, int32_t id);

typedef struct {
    const char *text;
    size_t      len;
    int32_t     id;
    bool        special;   /* tokenizer.json "special": control tokens (<|im_end|>) */
} ath_added_token;

typedef struct ath_bpe_merge ath_bpe_merge;

typedef struct {
    int32_t             n_tokens;     /* ids 0 .. n_tokens − 1 */
    const char        **token;        /* bytes of each id, NULL for an unused id */
    uint32_t           *token_len;
    int32_t             byte_token[256];
    ath_added_token    *added;
    int32_t             n_added;
    ath_bpe_merge      *merges;       /* hash table: pair of ids → rank and result */
    uint64_t            merge_mask;   /* capacity − 1                              */
    int32_t             n_merges;
    bool                nfc;
    ath_pretokenize_fn  split;
} ath_tokenizer;

/* Load the tokenizer.json at path. split is the family's pre-tokenization
 * and split_regex the pattern it implements, which must equal the Split
 * pattern in the file. Everything is allocated in a. ATH_ERR_UNSUPPORTED for
 * a tokenizer this engine does not handle, ATH_ERR_FORMAT for an
 * inconsistent one (a merge whose parts or result are not in the
 * vocabulary). */
ath_status ath_tokenizer_load(ath_tokenizer *t, ath_arena *a, const char *path, ath_pretokenize_fn split,
                              const char *split_regex);

/* Append the ids of text[0 .. n) to out. Working memory comes from scratch
 * and is released before returning, so scratch must not be out's arena. */
ath_status ath_tokenizer_encode(const ath_tokenizer *t, const char *text, size_t n, ath_tokens *out,
                                ath_arena *scratch);

/* Append the bytes of ids to out. ATH_ERR_RANGE for an id without a token.
 * The bytes of one token may be part of a UTF-8 sequence: decode a whole
 * sequence to get text. */
ath_status ath_tokenizer_decode(const ath_tokenizer *t, const int32_t *ids, size_t n, ath_buf *out);

/* The bytes of id, NULL if it has none. */
const char *ath_tokenizer_token(const ath_tokenizer *t, int32_t id, size_t *len);

/* The id of an added token by its text ("<|im_end|>"), -1 if none. */
int32_t ath_tokenizer_added(const ath_tokenizer *t, const char *text);

#endif /* ATH_BPE_H */
