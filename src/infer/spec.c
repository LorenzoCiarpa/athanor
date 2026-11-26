/* spec.c: speculative decoding with the draft head. See spec.h. */
#include "infer/spec.h"

#include <string.h>

static size_t buf_bytes(const ath_model *m, const ath_state *s, uint32_t i)
{
    return ath_state_buf_bytes(m, s, i);
}

static bool fixed_size(ath_state_kind k)
{
    return k == ATH_STATE_CONV || k == ATH_STATE_RECURRENT || k == ATH_STATE_HIDDEN;
}

/* Memory where the passes run. */
static void *mem_alloc(ath_spec *sp, size_t bytes, ath_arena *a)
{
    if (!sp->exec.dev)
        return ath_arena_alloc(a, bytes ? bytes : 1, 64);
    void *p = NULL;
    return ath_dev_alloc(sp->exec.dev, bytes ? bytes : 1, &p) == ATH_OK ? p : NULL;
}

static ath_status copy(const ath_spec *sp, void *dst, const void *src, size_t bytes)
{
    if (sp->exec.dev)
        return ath_copy(sp->exec.dev, dst, src, bytes, ATH_STREAM_COMPUTE);
    memcpy(dst, src, bytes);
    return ATH_OK;
}

static size_t row_bytes(const ath_spec *sp)
{
    return (size_t)sp->m->hidden_size * ath_dtype_size(sp->s->act);
}

ath_status ath_spec_open(ath_spec *sp, const ath_model *m, ath_state *s, const ath_exec *x, const ath_sampler *p,
                         int32_t chunk, ath_arena *a)
{
    memset(sp, 0, sizeof *sp);
    if (!m || !s || !p || !a)
        return ATH_ERR_INVALID_ARG;
    if (m->n_draft_layers < 1)
        return ATH_ERR_UNSUPPORTED;
    sp->m = m, sp->s = s, sp->p = *p;
    sp->exec = x ? *x : (ath_exec){0};
    sp->chunk = chunk > 0 ? chunk : 4096;
    sp->hid_state = -1;
    for (uint32_t i = 0; i < m->n_states; i++)
        if (m->states[i].kind == ATH_STATE_HIDDEN)
            sp->hid_state = (int32_t)i;
    if (sp->hid_state < 0 || (s->dev != NULL) != (sp->exec.dev != NULL))
        return ATH_ERR_INVALID_ARG;
    ath_arena_init(&sp->scratch, 0);
    const int64_t H = m->hidden_size, V = m->vocab_size;
    const int64_t shape[2] = {ATH_SPEC_ROWS, H};
    const ath_tier tier = sp->exec.dev ? ATH_TIER_VRAM : ATH_TIER_RAM;
    ath_status st = ath_tensor_init(&sp->hid, s->act, 2, shape, mem_alloc(sp, ATH_SPEC_ROWS * row_bytes(sp), a), tier);
    if (st == ATH_OK)
        st = ath_tensor_init(&sp->dhid, s->act, 2, shape, mem_alloc(sp, ATH_SPEC_ROWS * row_bytes(sp), a), tier);
    sp->saved = ath_arena_zalloc(a, (m->n_states ? m->n_states : 1) * sizeof *sp->saved, _Alignof(void *));
    sp->logits = ath_arena_alloc(a, (size_t)(ATH_SPEC_ROWS * V) * sizeof(float), 64);
    sp->dlogits = ath_arena_alloc(a, (size_t)V * sizeof(float), 64);
    sp->seen = ath_arena_zalloc(a, (size_t)V, 1);
    if (st == ATH_OK && (!sp->hid.data || !sp->dhid.data || !sp->saved || !sp->logits || !sp->dlogits || !sp->seen))
        st = ATH_ERR_NOMEM;
    for (uint32_t i = 0; st == ATH_OK && i < m->n_states; i++)
        if (fixed_size(m->states[i].kind) && !(sp->saved[i] = mem_alloc(sp, buf_bytes(m, s, i), a)))
            st = ATH_ERR_NOMEM;
    if (st != ATH_OK)
        ath_spec_close(sp);
    return st;
}

void ath_spec_close(ath_spec *sp)
{
    if (!sp || !sp->m)
        return;
    if (sp->exec.dev) {
        ath_stream_sync(sp->exec.dev, ATH_STREAM_COMPUTE);
        if (sp->hid.data)
            ath_dev_free(sp->exec.dev, sp->hid.data);
        if (sp->dhid.data)
            ath_dev_free(sp->exec.dev, sp->dhid.data);
        for (uint32_t i = 0; sp->saved && i < sp->m->n_states; i++)
            if (sp->saved[i])
                ath_dev_free(sp->exec.dev, sp->saved[i]);
    }
    ath_arena_free(&sp->scratch);
    sp->m = NULL;
}

/* The forward pass where the spec runs. */
static ath_status forward(ath_spec *sp, const ath_batch *b)
{
    return ath_model_forward(sp->m, b, sp->s, NULL, &sp->scratch, sp->exec.dev || sp->exec.pool ? &sp->exec : NULL);
}

/* Rows for the draft head: positions pos0 … pos0 + n − 1, the tokens after
 * them, the main model's outputs there (rows of dhid, filled by the
 * caller); with guess, the guess of the last row (greedy). */
static ath_status draft(ath_spec *sp, const int32_t *tokens, int32_t n, int32_t pos0, const ath_tensor *rows,
                        int32_t *guess)
{
    ath_tensor l;
    ATH_CHECK(ath_tensor_init(&l, ATH_DTYPE_F32, 2, (int64_t[]){1, sp->m->vocab_size}, sp->dlogits, ATH_TIER_RAM));
    const ath_draft d = {.tokens = tokens, .hidden = rows, .n_tokens = n, .pos0 = pos0, .logits = guess ? &l : NULL};
    ATH_CHECK(ath_model_draft(sp->m, &d, sp->s, &sp->scratch, sp->exec.dev || sp->exec.pool ? &sp->exec : NULL));
    if (!guess)
        return ATH_OK;
    sp->guesses++;
    static const ath_sampler greedy = {0, 0, 1, 0, 0};
    return ath_sample_ref(&l, &greedy, 0, &sp->scratch, guess);
}

/* The first n rows of t (a [rows, hidden] tensor) as a tensor. */
static ath_tensor first_rows(const ath_tensor *t, int32_t n)
{
    ath_tensor v = *t;
    v.shape[0] = n;
    return v;
}

static void *row(const ath_spec *sp, const ath_tensor *t, int32_t i)
{
    return (char *)t->data + (size_t)i * row_bytes(sp);
}

ath_status ath_spec_feed(ath_spec *sp, const int32_t *ids, int32_t n, float *logits)
{
    ath_state *s = sp->s;
    sp->n_pend = 0;
    const size_t rb = row_bytes(sp);
    for (int32_t a = 0; a < n;) {
        const int32_t T = n - a > sp->chunk ? sp->chunk : n - a;
        const int32_t P = s->n_past;
        /* a row the draft head ran with the token a previous answer chose
         * (and did not run) is run again with the token that comes */
        if (s->n_draft > P - 1)
            s->n_draft = P > 0 ? P - 1 : 0;
        ath_arena_pos mark = ath_arena_mark(&sp->scratch);
        ath_dev_arena_pos dmark = sp->exec.dev ? ath_dev_arena_mark(sp->exec.scratch) : (ath_dev_arena_pos){0};
        /* the outputs of the T tokens, then the draft's rows: the one left
         * waiting at P − 1 (its input in the state) and T − 1 more */
        const int32_t r0 = P > 0 ? 1 : 0, nr = r0 + T - 1;
        const int64_t hs[2] = {T, sp->m->hidden_size}, ds[2] = {nr > 0 ? nr : 1, sp->m->hidden_size};
        const ath_tier tier = sp->exec.dev ? ATH_TIER_VRAM : ATH_TIER_RAM;
        ath_tensor h, dh;
        ATH_CHECK(ath_tensor_init(&h, s->act, 2, hs, NULL, tier));
        ATH_CHECK(ath_tensor_init(&dh, s->act, 2, ds, NULL, tier));
        h.data = sp->exec.dev ? ath_dev_arena_alloc(sp->exec.scratch, (size_t)T * rb, 0)
                              : ath_arena_alloc(&sp->scratch, (size_t)T * rb, 64);
        dh.data = sp->exec.dev ? ath_dev_arena_alloc(sp->exec.scratch, (size_t)ds[0] * rb, 0)
                               : ath_arena_alloc(&sp->scratch, (size_t)ds[0] * rb, 64);
        if (!h.data || !dh.data)
            return ATH_ERR_NOMEM;
        ath_tensor l;
        const bool want = logits && a + T == n;
        ATH_CHECK(ath_tensor_init(&l, ATH_DTYPE_F32, 2, (int64_t[]){1, sp->m->vocab_size}, logits, ATH_TIER_RAM));
        const ath_batch b = {.tokens = ids + a, .n_tokens = T, .logits = want ? &l : NULL, .hidden = &h};
        ath_status st = forward(sp, &b);
        if (st == ATH_OK && nr > 0) {
            if (r0)
                st = copy(sp, dh.data, s->buf[sp->hid_state], rb);
            if (st == ATH_OK && T > 1)
                st = copy(sp, row(sp, &dh, r0), h.data, (size_t)(T - 1) * rb);
            if (st == ATH_OK)
                st = draft(sp, ids + a + (r0 ? 0 : 1), nr, r0 ? P - 1 : 0, &dh, NULL);
        }
        if (st == ATH_OK)
            st = copy(sp, s->buf[sp->hid_state], row(sp, &h, T - 1), rb);
        if (sp->exec.dev && st == ATH_OK)
            st = ath_stream_sync(sp->exec.dev, ATH_STREAM_COMPUTE);   /* before the scratch is reused */
        if (sp->exec.dev)
            ath_dev_arena_reset(sp->exec.scratch, dmark);
        ath_arena_reset(&sp->scratch, mark);
        ATH_CHECK(st);
        s->n_draft = P + T - 1;
        a += T;
    }
    return ATH_OK;
}

ath_status ath_spec_begin(ath_spec *sp, int32_t first)
{
    ath_state *s = sp->s;
    if (s->n_past < 1 || first < 0 || first >= sp->m->vocab_size)
        return ATH_ERR_INVALID_ARG;
    memset(sp->seen, 0, (size_t)sp->m->vocab_size);
    sp->seen[first] = 1;
    const int32_t n = s->n_past;
    ATH_CHECK(copy(sp, sp->dhid.data, s->buf[sp->hid_state], row_bytes(sp)));
    const ath_tensor rows = first_rows(&sp->dhid, 1);
    ATH_CHECK(draft(sp, &first, 1, n - 1, &rows, &sp->guess));
    s->n_draft = n;
    sp->pend[0] = first;
    sp->n_pend = 1;
    return ATH_OK;
}

/* The token chosen from row i of the last pass's logits, as plain decoding
 * would choose it (the presence penalty over the answer's tokens first). */
static ath_status choose(ath_spec *sp, int32_t i, double u, int32_t *tok)
{
    ath_tensor l;
    ATH_CHECK(ath_tensor_init(&l, ATH_DTYPE_F32, 2, (int64_t[]){1, sp->m->vocab_size},
                              sp->logits + (size_t)i * (size_t)sp->m->vocab_size, ATH_TIER_RAM));
    if (sp->p.presence != 0)
        ATH_CHECK(ath_presence_ref(&l, sp->seen, sp->p.presence));
    ATH_CHECK(ath_sample_ref(&l, &sp->p, u, &sp->scratch, tok));
    sp->seen[*tok] = 1;
    return ATH_OK;
}

static ath_status save_fixed(ath_spec *sp, bool back)
{
    for (uint32_t i = 0; i < sp->m->n_states; i++)
        if (sp->saved[i]) {
            const size_t b = buf_bytes(sp->m, sp->s, i);
            ATH_CHECK(back ? copy(sp, sp->s->buf[i], sp->saved[i], b) : copy(sp, sp->saved[i], sp->s->buf[i], b));
        }
    return ATH_OK;
}

ath_status ath_spec_step(ath_spec *sp, const double u[2], int32_t out[2], int32_t *n_out)
{
    ath_state *s = sp->s;
    if (sp->n_pend < 1)
        return ATH_ERR_INVALID_ARG;
    const int32_t n = s->n_past, k = sp->n_pend + 1;
    if (n + k > s->capacity)
        return ATH_ERR_RANGE;
    const size_t rb = row_bytes(sp);
    int32_t toks[ATH_SPEC_ROWS];
    for (int32_t i = 0; i < sp->n_pend; i++)
        toks[i] = sp->pend[i];
    toks[k - 1] = sp->guess;

    /* the pending tokens and the guess in one pass, the fixed-size state
     * copied first */
    ATH_CHECK(save_fixed(sp, false));
    ath_tensor l = {0}, h = first_rows(&sp->hid, k);
    ATH_CHECK(ath_tensor_init(&l, ATH_DTYPE_F32, 2, (int64_t[]){k, sp->m->vocab_size}, sp->logits, ATH_TIER_RAM));
    const ath_batch b = {.tokens = toks, .n_tokens = k, .logits = &l, .hidden = &h};
    ATH_CHECK(forward(sp, &b));
    sp->passes++;

    int32_t x;
    ATH_CHECK(choose(sp, k - 2, u[0], &x));
    out[0] = x;
    *n_out = 1;
    int32_t rows_tok[ATH_SPEC_ROWS];
    if (x == sp->guess) {
        /* the guess holds: the next token from the last row, then the draft
         * head on the k positions this pass ran */
        sp->accepted++;
        int32_t y;
        ATH_CHECK(choose(sp, k - 1, u[1], &y));
        out[1] = y;
        *n_out = 2;
        for (int32_t i = 1; i < k; i++)
            rows_tok[i - 1] = toks[i];
        rows_tok[k - 1] = y;
        ATH_CHECK(copy(sp, sp->dhid.data, sp->hid.data, (size_t)k * rb));
        const ath_tensor rows = first_rows(&sp->dhid, k);
        ATH_CHECK(draft(sp, rows_tok, k, n, &rows, &sp->guess));
        ATH_CHECK(copy(sp, s->buf[sp->hid_state], row(sp, &sp->hid, k - 1), rb));
        s->n_draft = n + k;
        sp->pend[0] = y;
        sp->n_pend = 1;
        return ATH_OK;
    }
    /* the guess fails: the pass undone */
    ATH_CHECK(save_fixed(sp, true));
    s->n_past = n;
    if (sp->n_pend == 1) {
        /* the pending token and x wait for the next pass; the draft head
         * guesses from the output at n, which the undone pass computed */
        ATH_CHECK(copy(sp, sp->dhid.data, sp->hid.data, rb));
        const ath_tensor rows = first_rows(&sp->dhid, 1);
        ATH_CHECK(draft(sp, &x, 1, n, &rows, &sp->guess));
        s->n_draft = n + 1;
        sp->pend[1] = x;
        sp->n_pend = 2;
        return ATH_OK;
    }
    /* two pending tokens and x: run the two now, so that a pass never
     * holds more than three tokens */
    sp->reruns++;
    ath_tensor h2 = first_rows(&sp->hid, 2);
    const ath_batch b2 = {.tokens = sp->pend, .n_tokens = 2, .hidden = &h2};
    ATH_CHECK(forward(sp, &b2));
    rows_tok[0] = sp->pend[1];
    rows_tok[1] = x;
    ATH_CHECK(copy(sp, sp->dhid.data, sp->hid.data, 2 * rb));
    const ath_tensor rows = first_rows(&sp->dhid, 2);
    ATH_CHECK(draft(sp, rows_tok, 2, n, &rows, &sp->guess));
    ATH_CHECK(copy(sp, s->buf[sp->hid_state], row(sp, &sp->hid, 1), rb));
    s->n_draft = n + 2;
    sp->pend[0] = x;
    sp->n_pend = 1;
    return ATH_OK;
}
