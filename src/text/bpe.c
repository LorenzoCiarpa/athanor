/* bpe.c: byte-level BPE, read from a Hugging Face tokenizer.json. See bpe.h.
 *
 * The file (12 MB for Qwen3.5, with 248,044 tokens and 247,587 merges) is
 * parsed into a temporary arena; only the token bytes, the added tokens and
 * the merge table are kept. Merges are found by a hash table keyed on the
 * pair of ids, which gives the merge's rank and the id of the result.
 */
#include "text/bpe.h"

#include "core/file.h"
#include "core/json.h"
#include "text/unicode.h"

#include <limits.h>
#include <string.h>

struct ath_bpe_merge {
    uint64_t key;    /* left id << 32 | right id; EMPTY if unused */
    int32_t  rank;   /* position in the merge list: lower merges first */
    int32_t  id;     /* the merged token */
};

#define EMPTY UINT64_MAX

/* ---- small growable lists ----------------------------------------------------- */

void ath_spans_init(ath_spans *s, ath_arena *a) { *s = (ath_spans){.a = a}; }

ath_status ath_spans_push(ath_spans *s, size_t end)
{
    if (s->n == s->cap) {
        size_t cap = s->cap ? 2 * s->cap : 64;
        size_t *e = ath_arena_alloc(s->a, cap * sizeof *e, _Alignof(size_t));
        if (!e)
            return ATH_ERR_NOMEM;
        if (s->n)
            memcpy(e, s->end, s->n * sizeof *e);
        s->end = e, s->cap = cap;
    }
    s->end[s->n++] = end;
    return ATH_OK;
}

void ath_tokens_init(ath_tokens *t, ath_arena *a) { *t = (ath_tokens){.a = a}; }

ath_status ath_tokens_push(ath_tokens *t, int32_t id)
{
    if (t->n == t->cap) {
        size_t cap = t->cap ? 2 * t->cap : 256;
        int32_t *v = ath_arena_alloc(t->a, cap * sizeof *v, _Alignof(int32_t));
        if (!v)
            return ATH_ERR_NOMEM;
        if (t->n)
            memcpy(v, t->ids, t->n * sizeof *v);
        t->ids = v, t->cap = cap;
    }
    t->ids[t->n++] = id;
    return ATH_OK;
}

/* ---- the GPT-2 byte alphabet -----------------------------------------------------
 *
 * Every byte is written as one printable character: the printable bytes of
 * Latin-1 ('!' .. '~', '¡' .. '¬', '®' .. 'ÿ') stand for themselves, and the
 * other 68 take U+0100 onward in byte order (space, 0x20, becomes U+0120,
 * "Ġ"). */
static bool printable(unsigned b) { return (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || b >= 0xAE; }

/* char_to_byte[c] for c < 0x144: the byte written as c, or -1. */
static void byte_alphabet(int16_t *char_to_byte)
{
    for (int c = 0; c < 0x144; c++)
        char_to_byte[c] = -1;
    int extra = 0;
    for (unsigned b = 0; b < 256; b++)
        char_to_byte[printable(b) ? (int)b : 0x100 + extra++] = (int16_t)b;
}

/* Bytes of a vocabulary string written in the alphabet. */
static ath_status alphabet_to_bytes(const int16_t *char_to_byte, const char *s, size_t n, ath_arena *a,
                                    char **out, uint32_t *out_len)
{
    char *o = ath_arena_alloc(a, n + 1, 1);
    if (!o)
        return ATH_ERR_NOMEM;
    uint32_t k = 0;
    for (size_t i = 0; i < n;) {
        uint32_t c;
        i += ath_utf8_decode(s + i, n - i, &c);
        if (c >= 0x144 || char_to_byte[c] < 0)
            return ATH_ERR_FORMAT;
        o[k++] = (char)char_to_byte[c];
    }
    o[k] = '\0';
    *out = o;
    *out_len = k;
    return ATH_OK;
}

/* ---- hash tables -------------------------------------------------------------- */

static uint64_t mix(uint64_t x)
{
    x ^= x >> 33;
    x *= 0xFF51AFD7ED558CCDull;
    x ^= x >> 33;
    return x;
}

static uint64_t hash_bytes(const char *s, size_t n)
{
    uint64_t h = 0xCBF29CE484222325ull;   /* FNV-1a */
    for (size_t i = 0; i < n; i++)
        h = (h ^ (unsigned char)s[i]) * 0x100000001B3ull;
    return mix(h);
}

/* Temporary map from token bytes to id, for resolving merges at load. */
typedef struct {
    int32_t       *slot;   /* id, or -1 */
    uint64_t       mask;
    const ath_tokenizer *t;
} vocab_map;

static ath_status vocab_map_init(vocab_map *v, const ath_tokenizer *t, ath_arena *tmp)
{
    uint64_t cap = 1;
    while (cap < 2 * (uint64_t)t->n_tokens)
        cap *= 2;
    v->slot = ath_arena_alloc(tmp, cap * sizeof *v->slot, _Alignof(int32_t));
    if (!v->slot)
        return ATH_ERR_NOMEM;
    memset(v->slot, 0xFF, cap * sizeof *v->slot);
    v->mask = cap - 1;
    v->t = t;
    for (int32_t id = 0; id < t->n_tokens; id++) {
        if (!t->token[id])
            continue;
        uint64_t h = hash_bytes(t->token[id], t->token_len[id]) & v->mask;
        while (v->slot[h] >= 0)
            h = (h + 1) & v->mask;
        v->slot[h] = id;
    }
    return ATH_OK;
}

static int32_t vocab_find(const vocab_map *v, const char *s, size_t n)
{
    for (uint64_t h = hash_bytes(s, n) & v->mask;; h = (h + 1) & v->mask) {
        int32_t id = v->slot[h];
        if (id < 0)
            return -1;
        if (v->t->token_len[id] == n && memcmp(v->t->token[id], s, n) == 0)
            return id;
    }
}

static const ath_bpe_merge *merge_find(const ath_tokenizer *t, int32_t a, int32_t b)
{
    uint64_t key = (uint64_t)(uint32_t)a << 32 | (uint32_t)b;
    for (uint64_t h = mix(key) & t->merge_mask;; h = (h + 1) & t->merge_mask) {
        const ath_bpe_merge *m = &t->merges[h];
        if (m->key == key)
            return m;
        if (m->key == EMPTY)
            return NULL;
    }
}

/* ---- loading ------------------------------------------------------------------ */

static bool str_is(const ath_json *n, const char *want)
{
    const char *s;
    return ath_json_to_str(n, &s, NULL) == ATH_OK && strcmp(s, want) == 0;
}

/* Missing, null, or the given string. */
static bool null_or_str(const ath_json *n, const char *want)
{
    return !n || n->type == ATH_JSON_NULL || str_is(n, want);
}

static bool false_or_absent(const ath_json *n)
{
    bool b;
    return !n || (ath_json_to_bool(n, &b) == ATH_OK && !b);
}

static ath_status unsupported(const char *path, const char *what)
{
    ath_log(ATH_LOG_ERROR, "%s: unsupported tokenizer: %s", path, what);
    return ATH_ERR_UNSUPPORTED;
}

/* The parts of tokenizer.json this engine does not implement are refused. */
static ath_status check_pipeline(const ath_json *root, const char *path, const char *split_regex, bool *nfc)
{
    const ath_json *model = ath_json_get(root, "model");
    if (!str_is(ath_json_get(model, "type"), "BPE"))
        return unsupported(path, "model type is not BPE");
    if (!null_or_str(ath_json_get(model, "unk_token"), "") ||
        !null_or_str(ath_json_get(model, "continuing_subword_prefix"), "") ||
        !null_or_str(ath_json_get(model, "end_of_word_suffix"), "") ||
        !false_or_absent(ath_json_get(model, "byte_fallback")) ||
        !false_or_absent(ath_json_get(model, "ignore_merges")))
        return unsupported(path, "BPE with unknown token, subword affixes, byte fallback or ignore_merges");
    const ath_json *dropout = ath_json_get(model, "dropout");
    if (dropout && dropout->type != ATH_JSON_NULL)
        return unsupported(path, "BPE dropout");

    const ath_json *norm = ath_json_get(root, "normalizer");
    if (!norm || norm->type == ATH_JSON_NULL)
        *nfc = false;
    else if (str_is(ath_json_get(norm, "type"), "NFC"))
        *nfc = true;
    else
        return unsupported(path, "normalizer other than NFC");

    const ath_json *pre = ath_json_get(root, "pre_tokenizer");
    const ath_json *split = ath_json_at(ath_json_get(pre, "pretokenizers"), 0);
    const ath_json *bytes = ath_json_at(ath_json_get(pre, "pretokenizers"), 1);
    if (!str_is(ath_json_get(pre, "type"), "Sequence") || ath_json_count(ath_json_get(pre, "pretokenizers")) != 2 ||
        !str_is(ath_json_get(split, "type"), "Split") || !str_is(ath_json_get(split, "behavior"), "Isolated") ||
        !false_or_absent(ath_json_get(split, "invert")) || !str_is(ath_json_get(bytes, "type"), "ByteLevel") ||
        !false_or_absent(ath_json_get(bytes, "add_prefix_space")) ||
        !false_or_absent(ath_json_get(bytes, "use_regex")))
        return unsupported(path, "pre-tokenizer other than Split(regex, Isolated) + ByteLevel");
    if (!str_is(ath_json_path(split, "pattern.Regex"), split_regex)) {
        ath_log(ATH_LOG_ERROR, "%s: the pre-tokenizer regex is not the one the family implements", path);
        return ATH_ERR_UNSUPPORTED;
    }
    if (!str_is(ath_json_path(root, "decoder.type"), "ByteLevel"))
        return unsupported(path, "decoder other than ByteLevel");
    return ATH_OK;
}

/* The two halves of a merge: "a b" (older files) or ["a", "b"]. */
static bool merge_parts(const ath_json *m, const char **a, size_t *na, const char **b, size_t *nb)
{
    if (m->type == ATH_JSON_ARRAY && ath_json_count(m) == 2) {
        return ath_json_to_str(ath_json_at(m, 0), a, na) == ATH_OK &&
               ath_json_to_str(ath_json_at(m, 1), b, nb) == ATH_OK;
    }
    const char *s;
    size_t n;
    if (ath_json_to_str(m, &s, &n) != ATH_OK)
        return false;
    const char *sp = memchr(s, ' ', n);
    if (!sp)
        return false;
    *a = s, *na = (size_t)(sp - s);
    *b = sp + 1, *nb = n - *na - 1;
    return true;
}

static ath_status load_tokens(ath_tokenizer *t, const ath_json *root, ath_arena *a, const char *path)
{
    const ath_json *vocab = ath_json_path(root, "model.vocab");
    const ath_json *added = ath_json_get(root, "added_tokens");
    if (!vocab || vocab->type != ATH_JSON_OBJECT || (added && added->type != ATH_JSON_ARRAY)) {
        ath_log(ATH_LOG_ERROR, "%s: no model.vocab object", path);
        return ATH_ERR_FORMAT;
    }
    int64_t max_id = -1, id;
    for (const ath_json *v = ath_json_first(vocab); v; v = v->next) {
        if (ath_json_to_int(v, &id) != ATH_OK || id < 0 || id >= INT32_MAX)
            return ATH_ERR_FORMAT;
        max_id = id > max_id ? id : max_id;
    }
    for (const ath_json *v = ath_json_first(added); v; v = v->next) {
        if (ath_json_to_int(ath_json_get(v, "id"), &id) != ATH_OK || id < 0 || id >= INT32_MAX)
            return ATH_ERR_FORMAT;
        max_id = id > max_id ? id : max_id;
    }
    t->n_tokens = (int32_t)(max_id + 1);
    t->token = ath_arena_zalloc(a, (size_t)t->n_tokens * sizeof *t->token, _Alignof(const char *));
    t->token_len = ath_arena_zalloc(a, (size_t)t->n_tokens * sizeof *t->token_len, _Alignof(uint32_t));
    t->n_added = (int32_t)ath_json_count(added);
    t->added = ath_arena_zalloc(a, (size_t)(t->n_added ? t->n_added : 1) * sizeof *t->added,
                                _Alignof(ath_added_token));
    if (!t->token || !t->token_len || !t->added)
        return ATH_ERR_NOMEM;

    int16_t char_to_byte[0x144];
    byte_alphabet(char_to_byte);
    for (const ath_json *v = ath_json_first(vocab); v; v = v->next) {
        ath_json_to_int(v, &id);
        char *bytes;
        ath_status s = alphabet_to_bytes(char_to_byte, v->key, v->key_len, a, &bytes, &t->token_len[id]);
        if (s == ATH_ERR_FORMAT)
            ath_log(ATH_LOG_ERROR, "%s: vocabulary entry %lld is not in the byte alphabet", path, (long long)id);
        ATH_CHECK(s);
        t->token[id] = bytes;
    }
    int32_t k = 0;
    for (const ath_json *v = ath_json_first(added); v; v = v->next, k++) {
        ath_added_token *at = &t->added[k];
        const char *content;
        size_t len;
        bool special = false;
        ath_json_to_int(ath_json_get(v, "id"), &id);
        if (ath_json_to_str(ath_json_get(v, "content"), &content, &len) != ATH_OK || len == 0)
            return ATH_ERR_FORMAT;
        ath_json_to_bool(ath_json_get(v, "special"), &special);
        bool flag;
        if ((ath_json_to_bool(ath_json_get(v, "lstrip"), &flag) == ATH_OK && flag) ||
            (ath_json_to_bool(ath_json_get(v, "rstrip"), &flag) == ATH_OK && flag) ||
            (ath_json_to_bool(ath_json_get(v, "single_word"), &flag) == ATH_OK && flag) ||
            (ath_json_to_bool(ath_json_get(v, "normalized"), &flag) == ATH_OK && flag))
            return unsupported(path, "added token with lstrip, rstrip, single_word or normalized");
        *at = (ath_added_token){.text = content, .len = len, .id = (int32_t)id, .special = special};
        at->text = ath_arena_strndup(a, content, len);
        if (!at->text)
            return ATH_ERR_NOMEM;
        t->token[id] = at->text;
        t->token_len[id] = (uint32_t)len;
    }
    return ATH_OK;
}

static ath_status load_merges(ath_tokenizer *t, const ath_json *root, ath_arena *a, ath_arena *tmp,
                              const char *path)
{
    vocab_map v;
    ATH_CHECK(vocab_map_init(&v, t, tmp));
    for (int b = 0; b < 256; b++) {
        char c = (char)b;
        t->byte_token[b] = vocab_find(&v, &c, 1);
        if (t->byte_token[b] < 0) {
            ath_log(ATH_LOG_ERROR, "%s: no token for byte 0x%02x", path, b);
            return ATH_ERR_FORMAT;
        }
    }

    const ath_json *merges = ath_json_path(root, "model.merges");
    size_t n = ath_json_count(merges);
    if (n >= INT32_MAX)
        return ATH_ERR_FORMAT;
    uint64_t cap = 1;
    while (cap < 2 * (uint64_t)n + 2)
        cap *= 2;
    t->merges = ath_arena_alloc(a, cap * sizeof *t->merges, _Alignof(ath_bpe_merge));
    if (!t->merges)
        return ATH_ERR_NOMEM;
    for (uint64_t i = 0; i < cap; i++)
        t->merges[i].key = EMPTY;
    t->merge_mask = cap - 1;
    t->n_merges = (int32_t)n;

    int16_t char_to_byte[0x144];
    byte_alphabet(char_to_byte);
    ath_arena_pos pos = ath_arena_mark(tmp);
    int32_t rank = 0;
    for (const ath_json *m = ath_json_first(merges); m; m = m->next, rank++) {
        const char *sa, *sb;
        size_t na, nb;
        char *ba, *bb;
        uint32_t la, lb;
        if (!merge_parts(m, &sa, &na, &sb, &nb) || alphabet_to_bytes(char_to_byte, sa, na, tmp, &ba, &la) ||
            alphabet_to_bytes(char_to_byte, sb, nb, tmp, &bb, &lb)) {
            ath_log(ATH_LOG_ERROR, "%s: merge %d is malformed", path, rank);
            return ATH_ERR_FORMAT;
        }
        char *ab = ath_arena_alloc(tmp, (size_t)la + lb, 1);
        if (!ab)
            return ATH_ERR_NOMEM;
        memcpy(ab, ba, la);
        memcpy(ab + la, bb, lb);
        int32_t ia = vocab_find(&v, ba, la), ib = vocab_find(&v, bb, lb), iab = vocab_find(&v, ab, (size_t)la + lb);
        if (ia < 0 || ib < 0 || iab < 0) {
            ath_log(ATH_LOG_ERROR, "%s: merge %d: a part or the result is not in the vocabulary", path, rank);
            return ATH_ERR_FORMAT;
        }
        /* As Hugging Face builds its merge map, a repeated pair takes the
         * later rank. */
        uint64_t key = (uint64_t)(uint32_t)ia << 32 | (uint32_t)ib;
        uint64_t h = mix(key) & t->merge_mask;
        while (t->merges[h].key != EMPTY && t->merges[h].key != key)
            h = (h + 1) & t->merge_mask;
        t->merges[h] = (ath_bpe_merge){.key = key, .rank = rank, .id = iab};
        ath_arena_reset(tmp, pos);
    }
    return ATH_OK;
}

ath_status ath_tokenizer_load(ath_tokenizer *t, ath_arena *a, const char *path, ath_pretokenize_fn split,
                              const char *split_regex)
{
    memset(t, 0, sizeof *t);
    if (!split || !split_regex)
        return ATH_ERR_INVALID_ARG;
    ath_file f;
    ATH_CHECK(ath_file_open(&f, path));
    ath_arena tmp;
    ath_arena_init(&tmp, 0);
    const ath_json *root;
    ath_json_error err;
    ath_status s = ath_json_parse(&tmp, f.data, f.size, &root, &err);
    if (s == ATH_ERR_PARSE)
        ath_log(ATH_LOG_ERROR, "%s: line %zu column %zu: %s", path, err.line, err.col, err.msg);
    if (s == ATH_OK)
        s = check_pipeline(root, path, split_regex, &t->nfc);
    if (s == ATH_OK)
        s = load_tokens(t, root, a, path);
    if (s == ATH_OK)
        s = load_merges(t, root, a, &tmp, path);
    ath_arena_free(&tmp);
    ath_file_close(&f);
    if (s != ATH_OK) {
        memset(t, 0, sizeof *t);
        return s;
    }
    t->split = split;
    return ATH_OK;
}

/* ---- encoding ----------------------------------------------------------------- */

/* BPE on the bytes of one piece. sym has room for n ids. */
static ath_status bpe_piece(const ath_tokenizer *t, const char *s, size_t n, int32_t *sym, ath_tokens *out)
{
    for (size_t i = 0; i < n; i++)
        sym[i] = t->byte_token[(unsigned char)s[i]];
    while (n > 1) {
        int32_t best_rank = INT32_MAX, best_id = -1;
        size_t best = 0;
        for (size_t j = 0; j + 1 < n; j++) {
            const ath_bpe_merge *m = merge_find(t, sym[j], sym[j + 1]);
            if (m && m->rank < best_rank)
                best_rank = m->rank, best_id = m->id, best = j;
        }
        if (best_id < 0)
            break;
        sym[best] = best_id;
        memmove(sym + best + 1, sym + best + 2, (n - best - 2) * sizeof *sym);
        n--;
    }
    for (size_t i = 0; i < n; i++)
        ATH_CHECK(ath_tokens_push(out, sym[i]));
    return ATH_OK;
}

/* Text between added tokens: normalize, pre-tokenize, merge. */
static ath_status encode_segment(const ath_tokenizer *t, const char *s, size_t n, ath_tokens *out,
                                 ath_arena *scratch)
{
    if (!n)
        return ATH_OK;
    if (t->nfc)
        ATH_CHECK(ath_nfc(s, n, scratch, &s, &n));
    ath_spans sp;
    ath_spans_init(&sp, scratch);
    ATH_CHECK(t->split(s, n, &sp));
    size_t longest = 0, start = 0;
    for (size_t i = 0; i < sp.n; i++) {
        if (sp.end[i] < start || sp.end[i] > n)
            return ATH_ERR_RANGE;   /* a broken split function */
        longest = sp.end[i] - start > longest ? sp.end[i] - start : longest;
        start = sp.end[i];
    }
    if (start != n)
        return ATH_ERR_RANGE;
    int32_t *sym = ath_arena_alloc(scratch, (longest ? longest : 1) * sizeof *sym, _Alignof(int32_t));
    if (!sym)
        return ATH_ERR_NOMEM;
    start = 0;
    for (size_t i = 0; i < sp.n; i++) {
        ATH_CHECK(bpe_piece(t, s + start, sp.end[i] - start, sym, out));
        start = sp.end[i];
    }
    return ATH_OK;
}

/* The longest added token at s[0 .. n), or NULL. */
static const ath_added_token *added_at(const ath_tokenizer *t, const char *s, size_t n)
{
    const ath_added_token *best = NULL;
    for (int32_t i = 0; i < t->n_added; i++) {
        const ath_added_token *at = &t->added[i];
        if (at->len <= n && at->text[0] == s[0] && memcmp(at->text, s, at->len) == 0 &&
            (!best || at->len > best->len))
            best = at;
    }
    return best;
}

ath_status ath_tokenizer_encode(const ath_tokenizer *t, const char *text, size_t n, ath_tokens *out,
                                ath_arena *scratch)
{
    if (!t->split || (!text && n) || scratch == out->a)
        return ATH_ERR_INVALID_ARG;
    bool first[256] = {false};
    for (int32_t i = 0; i < t->n_added; i++)
        first[(unsigned char)t->added[i].text[0]] = true;

    ath_arena_pos pos = ath_arena_mark(scratch);
    ath_status s = ATH_OK;
    size_t seg = 0;
    for (size_t i = 0; i < n && s == ATH_OK;) {
        const ath_added_token *at = first[(unsigned char)text[i]] ? added_at(t, text + i, n - i) : NULL;
        if (!at) {
            i++;
            continue;
        }
        s = encode_segment(t, text + seg, i - seg, out, scratch);
        if (s == ATH_OK)
            s = ath_tokens_push(out, at->id);
        i += at->len;
        seg = i;
    }
    if (s == ATH_OK)
        s = encode_segment(t, text + seg, n - seg, out, scratch);
    ath_arena_reset(scratch, pos);
    return s;
}

/* ---- decoding ----------------------------------------------------------------- */

const char *ath_tokenizer_token(const ath_tokenizer *t, int32_t id, size_t *len)
{
    if (id < 0 || id >= t->n_tokens || !t->token[id])
        return NULL;
    if (len)
        *len = t->token_len[id];
    return t->token[id];
}

ath_status ath_tokenizer_decode(const ath_tokenizer *t, const int32_t *ids, size_t n, ath_buf *out)
{
    for (size_t i = 0; i < n; i++) {
        size_t len;
        const char *s = ath_tokenizer_token(t, ids[i], &len);
        if (!s)
            return ATH_ERR_RANGE;
        ATH_CHECK(ath_buf_append(out, s, len));
    }
    return ATH_OK;
}

int32_t ath_tokenizer_added(const ath_tokenizer *t, const char *text)
{
    for (int32_t i = 0; i < t->n_added; i++)
        if (strcmp(t->added[i].text, text) == 0)
            return t->added[i].id;
    return -1;
}
