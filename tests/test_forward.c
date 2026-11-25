/* test_forward.c: the forward pass, reached only through the contract
 * (src/model/model.h): no family header is included here.
 *
 * Tiny random checkpoints (tests/tiny.h): state specs and sizes, argument
 * checks, hooks, and the property every pipeline relies on: a sequence gives
 * the same logits and the same state, bit for bit, whether it runs all at
 * once, in pieces or one token at a time.
 *
 * The real checkpoint against the forward records of tests/golden/dump.py:
 * fixed prompts and greedy decode steps, run by the reference in bf16 and in
 * fp32 (tests/golden/README.md). athanor runs the same tokens twice:
 *
 *   bf16 activations  the gate (docs/STATUS.md, step 4): the mean KL against
 *                     the fp32 reference over every position of every prompt
 *                     is no larger than the bf16 reference's; the greedy
 *                     token matches the fp32 reference wherever its top-2
 *                     margin exceeds the bf16 noise. Per-prompt means and the
 *                     residual stream entering each layer, against both runs,
 *                     are reported as diagnostics.
 *   f32 activations   the residual stream at every layer and the logits within
 *                     a tolerance of the fp32 reference (tests/README.md).
 *
 * The four runs are independent; they run in parallel threads and share one
 * pool for their matmuls (ath_matmul_fast, the same bits as the reference
 * with any number of threads). With a GPU (make cuda test) the same four
 * runs go through the engine (src/infer/engine.h) afterwards, one after the
 * other, and are held to the same gate and tolerances (step 8). A DEBUG build runs only the first
 * ATH_FORWARD_POSITIONS positions (default 4) and checks the f32 tolerances,
 * not the gate, which needs every position. Needs $ATH_GOLDEN_DIR and
 * $ATH_MODEL_DIR; skipped when either is missing.
 */
#include "test.h"
#include "tiny.h"
#include "util.h"

#include "core/safetensors.h"
#include "infer/engine.h"
#include "model/model.h"

#include <pthread.h>
#include <time.h>

/* ---- helpers ------------------------------------------------------------------- */

typedef struct {
    tmpdir    d;
    ath_arena a;
    ath_model m;
} tiny_model;

static ath_status tiny_open(tiny_model *t, const tiny *c, uint64_t seed)
{
    ath_arena_init(&t->a, 0);
    return open_tiny(&t->a, &t->m, &t->d, c, (ckpt){.seed = seed, .real_dtypes = true}, NULL);
}

static void tiny_close(tiny_model *t)
{
    ath_model_close(&t->m);
    tmpdir_remove(&t->d);
    ath_arena_free(&t->a);
}

/* Bytes of the buffer of one state spec, as model.h defines them. */
static size_t spec_bytes(const ath_state_spec *sp, ath_dtype act, int32_t capacity)
{
    const size_t es = ath_dtype_size(sp->kind == ATH_STATE_RECURRENT ? ATH_DTYPE_F32 : act);
    return (size_t)(sp->fixed + sp->per_token * capacity) * es;
}

/* A state whose buffers are zero everywhere, also in the KV rows a run never
 * writes, so that two states can be compared byte for byte. */
static ath_status zeroed_state(ath_state *s, const ath_model *m, ath_dtype act, int32_t capacity, ath_arena *a)
{
    ATH_CHECK(ath_state_init(s, m, act, capacity, a));
    for (uint32_t i = 0; i < m->n_states; i++)
        memset(s->buf[i], 0, spec_bytes(&m->states[i], act, capacity));
    return ATH_OK;
}

static bool same_state(const ath_state *x, const ath_state *y, const ath_model *m)
{
    if (x->n_past != y->n_past)
        return false;
    for (uint32_t i = 0; i < m->n_states; i++)
        if (memcmp(x->buf[i], y->buf[i], spec_bytes(&m->states[i], x->act, x->capacity)) != 0)
            return false;
    return true;
}

/* f32 [rows, cols] over caller memory. */
static ath_tensor f32_tensor(float *data, int64_t rows, int64_t cols)
{
    ath_tensor t;
    const int64_t shape[2] = {rows, cols};
    ath_tensor_init(&t, ATH_DTYPE_F32, 2, shape, data, ATH_TIER_RAM);
    return t;
}

/* The comparison measure of tests/test_golden.c:
 * max_i |got_i − want_i| / (|want_i| + rms(want)). */
static double rms_of(const void *want, ath_dtype dt, int64_t n)
{
    double s = 0;
    for (int64_t i = 0; i < n; i++) {
        double w = ath_load_f32(want, dt, i);
        s += w * w;
    }
    return n ? sqrt(s / (double)n) : 0;
}

static double score(const void *got, ath_dtype gdt, const void *want, ath_dtype wdt, int64_t n, double rms)
{
    double worst = 0;
    for (int64_t i = 0; i < n; i++) {
        double g = ath_load_f32(got, gdt, i), w = ath_load_f32(want, wdt, i);
        double e = fabs(g - w) / (fabs(w) + rms);
        if (!(e <= worst))  /* NaN counts as the worst */
            worst = isnan(e) ? INFINITY : e;
    }
    return worst;
}

/* ---- tiny checkpoints -------------------------------------------------------------- */

enum { NTOK = 12 };
static const int32_t TOKENS[NTOK] = {3, 15, 0, 7, 7, 2, 11, 9, 4, 14, 1, 8};

ATH_TEST(state_specs)
{
    tiny_model t;
    ATH_REQUIRE_OK(tiny_open(&t, &DENSE, 1));
    const ath_model *m = &t.m;

    /* layers "lllf": conv and recurrent state per Gated DeltaNet layer, KV
     * for the attention layer, in layer order */
    const int conv = 2 * LK * DK + LV * DV;
    const struct { ath_state_kind kind; int32_t layer; int64_t fixed, per_token; } want[] = {
        {ATH_STATE_CONV, 0, (K - 1) * conv, 0}, {ATH_STATE_RECURRENT, 0, LV * DK * DV, 0},
        {ATH_STATE_CONV, 1, (K - 1) * conv, 0}, {ATH_STATE_RECURRENT, 1, LV * DK * DV, 0},
        {ATH_STATE_CONV, 2, (K - 1) * conv, 0}, {ATH_STATE_RECURRENT, 2, LV * DK * DV, 0},
        {ATH_STATE_KV, 3, 0, 2 * NKV * HD},
    };
    ATH_EXPECT(m->n_states == ATH_ARRAY_LEN(want));
    for (uint32_t i = 0; i < m->n_states && i < ATH_ARRAY_LEN(want); i++) {
        ATH_EXPECT(m->states[i].kind == want[i].kind);
        ATH_EXPECT(m->states[i].layer == want[i].layer);
        ATH_EXPECT(m->states[i].fixed == want[i].fixed);
        ATH_EXPECT(m->states[i].per_token == want[i].per_token);
    }

    /* bytes: conv 96 elements, recurrent 64 (fp32), KV 8 per token; every
     * buffer rounded up to 64 bytes */
    size_t bytes = 0;
    ATH_EXPECT_OK(ath_state_bytes(m, ATH_DTYPE_BF16, 10, &bytes));
    ATH_EXPECT(bytes == 3 * (192 + 256) + 192);
    ATH_EXPECT_OK(ath_state_bytes(m, ATH_DTYPE_F32, 10, &bytes));
    ATH_EXPECT(bytes == 3 * (384 + 256) + 320);
    ATH_EXPECT_STATUS(ath_state_bytes(m, ATH_DTYPE_F16, 10, &bytes), ATH_ERR_INVALID_ARG);

    ATH_EXPECT_STR_EQ(ath_state_kind_name(ATH_STATE_KV), "kv");
    ATH_EXPECT_STR_EQ(ath_state_kind_name(ATH_STATE_CONV), "conv");
    ATH_EXPECT_STR_EQ(ath_state_kind_name(ATH_STATE_RECURRENT), "recurrent");
    ATH_EXPECT_STR_EQ(ath_state_kind_name(ATH_STATE_KIND_COUNT), "?");

    ath_state s;
    ATH_EXPECT_STATUS(ath_state_init(&s, m, ATH_DTYPE_F16, 4, &t.a), ATH_ERR_TYPE);
    ATH_EXPECT_STATUS(ath_state_init(&s, m, ATH_DTYPE_BF16, 0, &t.a), ATH_ERR_INVALID_ARG);

    /* reset zeroes the fixed-size state and rewinds */
    ATH_REQUIRE_OK(ath_state_init(&s, m, ATH_DTYPE_BF16, 4, &t.a));
    for (uint32_t i = 0; i < m->n_states; i++)
        memset(s.buf[i], 0x5a, spec_bytes(&m->states[i], s.act, s.capacity));
    s.n_past = 3;
    ATH_EXPECT_OK(ath_state_reset(&s, m));
    ATH_EXPECT(s.n_past == 0);
    for (uint32_t i = 0; i < m->n_states; i++) {
        if (m->states[i].kind == ATH_STATE_KV)
            continue;
        const unsigned char *p = s.buf[i];
        bool zero = true;
        for (size_t j = 0; j < spec_bytes(&m->states[i], s.act, s.capacity); j++)
            zero = zero && p[j] == 0;
        ATH_EXPECT(zero);
    }
    tiny_close(&t);
}

ATH_TEST(forward_rejects)
{
    tiny_model t;
    ATH_REQUIRE_OK(tiny_open(&t, &DENSE, 1));
    const ath_model *m = &t.m;
    ath_arena scratch;
    ath_arena_init(&scratch, 0);
    ath_state s;
    ATH_REQUIRE_OK(ath_state_init(&s, m, ATH_DTYPE_BF16, 4, &t.a));

    float out[5 * V];
    ath_tensor logits = f32_tensor(out, 3, V);
    int32_t toks[5] = {1, 2, 3, 4, 5};
    ath_batch b = {.tokens = toks, .n_tokens = 3, .logits = &logits};

    ATH_EXPECT_STATUS(ath_model_forward(m, &b, &s, NULL, NULL, NULL), ATH_ERR_INVALID_ARG);
    ATH_EXPECT_STATUS(ath_model_forward(m, &(ath_batch){.tokens = toks, .n_tokens = 0, .logits = NULL}, &s, NULL, &scratch, NULL), ATH_ERR_INVALID_ARG);
    ATH_EXPECT_STATUS(ath_model_forward(m, &(ath_batch){.tokens = NULL, .n_tokens = 3, .logits = NULL}, &s, NULL, &scratch, NULL), ATH_ERR_INVALID_ARG);
    ATH_EXPECT_STATUS(ath_model_forward(m, &(ath_batch){.tokens = toks, .n_tokens = 5, .logits = NULL}, &s, NULL, &scratch, NULL), ATH_ERR_RANGE);
    toks[1] = V;
    ATH_EXPECT_STATUS(ath_model_forward(m, &b, &s, NULL, &scratch, NULL), ATH_ERR_RANGE);
    toks[1] = -1;
    ATH_EXPECT_STATUS(ath_model_forward(m, &b, &s, NULL, &scratch, NULL), ATH_ERR_RANGE);
    toks[1] = 2;

    ath_tensor wrong = f32_tensor(out, 2, V);   /* neither 3 rows nor 1 */
    ATH_EXPECT_STATUS(ath_model_forward(m, &(ath_batch){.tokens = toks, .n_tokens = 3, .logits = &wrong}, &s, NULL, &scratch, NULL), ATH_ERR_INVALID_ARG);
    wrong = f32_tensor(out, 3, V - 1);
    ATH_EXPECT_STATUS(ath_model_forward(m, &(ath_batch){.tokens = toks, .n_tokens = 3, .logits = &wrong}, &s, NULL, &scratch, NULL), ATH_ERR_INVALID_ARG);
    wrong = logits;
    wrong.dtype = ATH_DTYPE_BF16;
    ATH_EXPECT_STATUS(ath_model_forward(m, &(ath_batch){.tokens = toks, .n_tokens = 3, .logits = &wrong}, &s, NULL, &scratch, NULL), ATH_ERR_TYPE);
    ath_state none = {.buf = NULL};
    ATH_EXPECT_STATUS(ath_model_forward(m, &b, &none, NULL, &scratch, NULL), ATH_ERR_INVALID_ARG);
    ATH_EXPECT(s.n_past == 0);

    /* the state fills up: 3 tokens, then 2 do not fit, then 1 does */
    ATH_EXPECT_OK(ath_model_forward(m, &b, &s, NULL, &scratch, NULL));
    ATH_EXPECT(s.n_past == 3);
    ATH_EXPECT_STATUS(ath_model_forward(m, &(ath_batch){.tokens = toks, .n_tokens = 2, .logits = NULL}, &s, NULL, &scratch, NULL), ATH_ERR_RANGE);
    ATH_EXPECT(s.n_past == 3);
    ATH_EXPECT_OK(ath_model_forward(m, &(ath_batch){.tokens = toks, .n_tokens = 1, .logits = NULL}, &s, NULL, &scratch, NULL));
    ATH_EXPECT(s.n_past == 4);
    /* scratch memory is released by every call */
    ATH_EXPECT(ath_arena_used(&scratch) == 0);

    /* MoE layers run since step 12 (before, ATH_ERR_UNSUPPORTED) */
    tiny_model moe;
    ATH_REQUIRE_OK(tiny_open(&moe, &MOE, 1));
    ath_state sm;
    ATH_REQUIRE_OK(ath_state_init(&sm, &moe.m, ATH_DTYPE_BF16, 4, &moe.a));
    ATH_EXPECT_OK(ath_model_forward(&moe.m, &(ath_batch){.tokens = toks, .n_tokens = 2, .logits = NULL}, &sm, NULL, &scratch, NULL));
    ATH_EXPECT(sm.n_past == 2);
    tiny_close(&moe);

    ath_arena_free(&scratch);
    tiny_close(&t);
}

/* Run TOKENS through a fresh state in pieces of the given sizes; logits of
 * every token land in out [NTOK, V]. */
static ath_status run_pieces(const ath_model *m, ath_dtype act, const int *sizes, int n_sizes, ath_arena *a,
                             ath_arena *scratch, ath_state *s, float *out)
{
    ATH_CHECK(zeroed_state(s, m, act, NTOK, a));
    int done = 0;
    for (int i = 0; i < n_sizes; i++) {
        ath_tensor logits = f32_tensor(out + (size_t)done * V, sizes[i], V);
        ATH_CHECK(ath_model_forward(m, &(ath_batch){.tokens = TOKENS + done, .n_tokens = sizes[i], .logits = &logits}, s, NULL, scratch, NULL));
        done += sizes[i];
    }
    return done == NTOK ? ATH_OK : ATH_ERR_INVALID_ARG;
}

static void check_pieces(ath_test *t_, ath_dtype act)
{
    tiny_model t;
    ATH_REQUIRE_OK(tiny_open(&t, &DENSE, 7));
    const ath_model *m = &t.m;
    ath_arena scratch;
    ath_arena_init(&scratch, 0);

    static const int whole[] = {NTOK}, split[] = {5, 1, 6}, single[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    float lw[NTOK * V], ls[NTOK * V], l1[NTOK * V], last[V];
    ath_state sw, ss, s1;
    ATH_REQUIRE_OK(run_pieces(m, act, whole, 1, &t.a, &scratch, &sw, lw));
    ATH_REQUIRE_OK(run_pieces(m, act, split, 3, &t.a, &scratch, &ss, ls));
    ATH_REQUIRE_OK(run_pieces(m, act, single, NTOK, &t.a, &scratch, &s1, l1));
    ATH_EXPECT(memcmp(lw, ls, sizeof lw) == 0);
    ATH_EXPECT(memcmp(lw, l1, sizeof lw) == 0);
    ATH_EXPECT(same_state(&sw, &ss, m));
    ATH_EXPECT(same_state(&sw, &s1, m));

    /* logits of the last token only: the same numbers as row NTOK − 1 */
    ath_state sl;
    ath_tensor one = f32_tensor(last, 1, V);
    ATH_REQUIRE_OK(zeroed_state(&sl, m, act, NTOK, &t.a));
    ATH_REQUIRE_OK(ath_model_forward(m, &(ath_batch){.tokens = TOKENS, .n_tokens = NTOK, .logits = &one}, &sl, NULL, &scratch, NULL));
    ATH_EXPECT(memcmp(last, lw + (NTOK - 1) * V, sizeof last) == 0);

    /* after a reset the same state runs the sequence again from scratch */
    float again[NTOK * V];
    ath_tensor all = f32_tensor(again, NTOK, V);
    ath_state_reset(&sw, m);
    ATH_REQUIRE_OK(ath_model_forward(m, &(ath_batch){.tokens = TOKENS, .n_tokens = NTOK, .logits = &all}, &sw, NULL, &scratch, NULL));
    ATH_EXPECT(memcmp(again, lw, sizeof lw) == 0);
    ATH_EXPECT(same_state(&sw, &ss, m));

    bool finite = true;
    for (int i = 0; i < NTOK * V; i++)
        finite = finite && isfinite(lw[i]);
    ATH_EXPECT(finite);
    ath_arena_free(&scratch);
    tiny_close(&t);
}

ATH_TEST(forward_pieces_bf16) { check_pieces(t_, ATH_DTYPE_BF16); }
ATH_TEST(forward_pieces_f32) { check_pieces(t_, ATH_DTYPE_F32); }

/* ---- hooks --------------------------------------------------------------------------- */

typedef struct {
    const ath_model *m;
    const int32_t   *tokens;
    int              calls;
    int32_t          layer[16];
    int64_t          rows[16];
    bool             dtype_ok, embed_ok;
    ath_dtype        act;
} hook_log;

static void on_residual(void *u, int32_t layer, const ath_tensor *x)
{
    hook_log *h = u;
    if (h->calls < 16) {
        h->layer[h->calls] = layer;
        h->rows[h->calls] = x->ndim == 2 && x->shape[1] == h->m->hidden_size ? x->shape[0] : -1;
    }
    h->calls++;
    h->dtype_ok = h->dtype_ok && x->dtype == h->act;
    if (layer != 0)
        return;
    /* the stream entering layer 0 is the embedding of the tokens */
    const ath_param *e = NULL;
    for (uint32_t i = 0; i < h->m->n_params && !e; i++)
        if (h->m->params[i].role == ATH_ROLE_EMBED)
            e = &h->m->params[i];
    const int64_t H = h->m->hidden_size;
    for (int64_t r = 0; e && r < x->shape[0]; r++)
        for (int64_t c = 0; c < H; c++)
            h->embed_ok = h->embed_ok && ath_load_f32(x->data, x->dtype, r * H + c) ==
                                             ath_round_to(x->dtype, ath_load_f32(e->t.data, e->t.dtype,
                                                                                 h->tokens[r] * H + c));
    h->embed_ok = h->embed_ok && e;
}

ATH_TEST(forward_hooks)
{
    tiny_model t;
    ATH_REQUIRE_OK(tiny_open(&t, &DENSE, 3));
    const ath_model *m = &t.m;
    ath_arena scratch;
    ath_arena_init(&scratch, 0);
    ath_state s;
    ATH_REQUIRE_OK(ath_state_init(&s, m, ATH_DTYPE_BF16, NTOK, &t.a));

    hook_log log = {.m = m, .tokens = TOKENS, .dtype_ok = true, .embed_ok = true, .act = ATH_DTYPE_BF16};
    ath_hooks h = {.on_residual = on_residual, .u = &log};
    ATH_REQUIRE_OK(ath_model_forward(m, &(ath_batch){.tokens = TOKENS, .n_tokens = 5, .logits = NULL}, &s, &h, &scratch, NULL));
    ATH_EXPECT(log.calls == m->n_layers + 1);
    for (int i = 0; i < log.calls && i < 16; i++) {
        ATH_EXPECT(log.layer[i] == i);
        ATH_EXPECT(log.rows[i] == 5);
    }
    log.calls = 0;
    log.tokens = TOKENS + 5;
    ATH_REQUIRE_OK(ath_model_forward(m, &(ath_batch){.tokens = TOKENS + 5, .n_tokens = 1, .logits = NULL}, &s, &h, &scratch, NULL));
    ATH_EXPECT(log.calls == m->n_layers + 1);
    ATH_EXPECT(log.rows[0] == 1 && log.rows[m->n_layers] == 1);
    ATH_EXPECT(log.dtype_ok);
    ATH_EXPECT(log.embed_ok);

    /* hooks with no callback, and no hooks at all, are fine */
    ath_hooks empty = {.on_residual = NULL};
    ATH_EXPECT_OK(ath_model_forward(m, &(ath_batch){.tokens = TOKENS + 6, .n_tokens = 1, .logits = NULL}, &s, &empty, &scratch, NULL));
    ath_arena_free(&scratch);
    tiny_close(&t);
}

/* ---- the real checkpoint --------------------------------------------------------------- */

ATH_TEST(real_state_specs)
{
    char path[ATH_PATH_MAX];
    if (!model_file(path, sizeof path, "config.json"))
        ATH_SKIP("no checkpoint in $ATH_MODEL_DIR");
    ath_arena a;
    ath_arena_init(&a, 0);
    ath_model m;
    ATH_REQUIRE_OK(ath_model_open(&m, &a, model_dir()));
    int n[ATH_STATE_KIND_COUNT] = {0};
    for (uint32_t i = 0; i < m.n_states; i++)
        n[m.states[i].kind]++;
    printf("  %u state specs: %d kv, %d conv, %d recurrent\n", m.n_states, n[ATH_STATE_KV], n[ATH_STATE_CONV],
           n[ATH_STATE_RECURRENT]);
    /* one KV per attention layer, conv and recurrent per Gated DeltaNet layer */
    int attention = 0;
    for (int32_t l = 0; l < m.n_layers; l++)
        attention += strcmp(m.layers[l].mixer, "attention") == 0;
    ATH_EXPECT(n[ATH_STATE_KV] == attention);
    ATH_EXPECT(n[ATH_STATE_CONV] == m.n_layers - attention);
    ATH_EXPECT(n[ATH_STATE_RECURRENT] == m.n_layers - attention);
    size_t b1, b2;
    ATH_EXPECT_OK(ath_state_bytes(&m, ATH_DTYPE_BF16, 1, &b1));
    ATH_EXPECT_OK(ath_state_bytes(&m, ATH_DTYPE_BF16, 4097, &b2));
    printf("  state of one sequence, bf16: %.2f MiB fixed, %.1f KiB per token\n", b1 / 1048576.0,
           (double)(b2 - b1) / 4096 / 1024);
    ath_model_close(&m);
    ath_arena_free(&a);
}

/* Tolerances of the f32 forward pass against the fp32 reference: about four
 * times the worst measured error (4.7e-5 for the residual stream, 2.7e-5 for
 * the logits), as for the exact op tolerances; tests/README.md, Tolerances. */
#define TOL_F32_RESIDUAL 2e-4
#define TOL_F32_LOGITS   1e-4

typedef struct {
    /* in */
    const ath_model       *m;
    const ath_safetensors *g;
    ath_pool              *pool;                 /* shared by the CPU runs */
    ath_engine            *eng;                  /* the GPU runs: the engine; NULL on the CPU */
    void                  *res_host;             /* GPU: the residual stream copied back */
    const char            *name;
    ath_dtype              act;
    const int32_t         *tokens;
    int32_t                n_prompt, n_tokens;   /* positions in the record */
    int32_t                n_run;                /* positions this run computes */
    const ath_tensor      *res, *res32;          /* [layers + 1, n_tokens, hidden] */
    double                *rms, *rms32;          /* per layer, over the run's positions */
    /* out */
    ath_status             st;
    float                 *logits;               /* [n_run, vocab] */
    double                *err, *err32;          /* per layer: score against res, res32 */
    int32_t                pos0;                 /* first position of the current call */
    double                 seconds;
} job;

/* Score of the residual stream entering `layer` against both reference runs,
 * for the positions of the current call. */
static void compare_residual(void *u, int32_t layer, const ath_tensor *x_in)
{
    job *j = u;
    ath_tensor xh = *x_in, *x = &xh;
    if (x_in->tier == ATH_TIER_VRAM) {   /* on the GPU: copy it back on the compute stream, after its writers */
        xh.data = j->res_host;
        xh.tier = ATH_TIER_RAM;
        if (ath_copy(&j->eng->dev, xh.data, x_in->data, ath_tensor_nbytes(x_in), ATH_STREAM_COMPUTE) != ATH_OK ||
            ath_stream_sync(&j->eng->dev, ATH_STREAM_COMPUTE) != ATH_OK) {
            j->err[layer] = j->err32[layer] = INFINITY;
            return;
        }
    }
    const int64_t H = j->m->hidden_size, rows = x->shape[0];
    const size_t at = ((size_t)layer * (size_t)j->n_tokens + (size_t)j->pos0) * (size_t)H;
    const size_t es = ath_dtype_size(j->res->dtype), es32 = ath_dtype_size(j->res32->dtype);
    double e = score(x->data, x->dtype, (const char *)j->res->data + at * es, j->res->dtype, rows * H, j->rms[layer]);
    double e32 = score(x->data, x->dtype, (const char *)j->res32->data + at * es32, j->res32->dtype, rows * H,
                       j->rms32[layer]);
    if (!(e <= j->err[layer]))
        j->err[layer] = e;
    if (!(e32 <= j->err32[layer]))
        j->err32[layer] = e32;
}

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* The prompt in one call (all of it, or all but the last position when the
 * run stops inside the prompt, so that the single-token path runs too), then
 * one position at a time, as the reference decoded. */
static void *run_job(void *arg)
{
    job *j = arg;
    const int64_t V = j->m->vocab_size;
    ath_arena a, scratch;
    ath_arena_init(&a, 0);
    ath_arena_init(&scratch, 0);
    double t0 = now();
    ath_state s;
    ath_hooks h = {.on_residual = compare_residual, .u = j};
    const ath_exec x = j->eng ? j->eng->exec : (ath_exec){.pool = j->pool};
    j->st = j->eng ? ath_engine_state(j->eng, &s, j->act, j->n_run, &a) : ath_state_init(&s, j->m, j->act, j->n_run, &a);
    int32_t first = j->n_prompt < j->n_run ? j->n_prompt : j->n_run - 1;
    if (first < 1)
        first = 1;
    for (int32_t p = 0; j->st == ATH_OK && p < j->n_run;) {
        int32_t n = p == 0 ? first : 1;
        ath_tensor logits = f32_tensor(j->logits + (size_t)p * (size_t)V, n, V);
        j->pos0 = p;
        j->st = ath_model_forward(j->m, &(ath_batch){.tokens = j->tokens + p, .n_tokens = n, .logits = &logits}, &s, &h, &scratch, &x);
        p += n;
    }
    j->seconds = now() - t0;
    ath_arena_free(&scratch);
    ath_arena_free(&a);
    return NULL;
}

/* KL(p || q) of two logit rows, in double: Σ p (log p − log q). */
static double kl_row(const void *p_logits, ath_dtype pdt, const float *q_logits, int64_t V)
{
    double pmax = -INFINITY, qmax = -INFINITY;
    for (int64_t v = 0; v < V; v++) {
        pmax = fmax(pmax, ath_load_f32(p_logits, pdt, v));
        qmax = fmax(qmax, q_logits[v]);
    }
    double ps = 0, qs = 0;
    for (int64_t v = 0; v < V; v++) {
        ps += exp(ath_load_f32(p_logits, pdt, v) - pmax);
        qs += exp(q_logits[v] - qmax);
    }
    const double plse = pmax + log(ps), qlse = qmax + log(qs);
    double kl = 0;
    for (int64_t v = 0; v < V; v++) {
        double lp = ath_load_f32(p_logits, pdt, v) - plse, lq = q_logits[v] - qlse;
        kl += exp(lp) * (lp - lq);
    }
    return kl;
}

static void top2(const void *logits, ath_dtype dt, int64_t V, int64_t *a, int64_t *b)
{
    *a = 0;
    *b = 1;
    if (ath_load_f32(logits, dt, 1) > ath_load_f32(logits, dt, 0)) {
        *a = 1;
        *b = 0;
    }
    for (int64_t v = 2; v < V; v++) {
        float x = ath_load_f32(logits, dt, v);
        if (x > ath_load_f32(logits, dt, *a)) {
            *b = *a;
            *a = v;
        } else if (x > ath_load_f32(logits, dt, *b)) {
            *b = v;
        }
    }
}

static int64_t argmax_f32(const float *x, int64_t V)
{
    int64_t a = 0;
    for (int64_t v = 1; v < V; v++)
        if (x[v] > x[a])
            a = v;
    return a;
}

static const char *golden_dir(void)
{
    const char *d = getenv("ATH_GOLDEN_DIR");
    return (d && *d) ? d : "tests/golden/data/qwen35-0.8b";
}

static const ath_tensor *golden(ath_test *t_, const ath_safetensors *g, const char *rid, const char *slot)
{
    char name[128];
    snprintf(name, sizeof name, "%s.%s", rid, slot);
    const ath_safetensors_entry *e = ath_safetensors_find(g, name);
    if (!e) {
        char msg[200];
        snprintf(msg, sizeof msg, "golden tensor %s missing (rerun tests/golden/dump.py)", name);
        ath_test_fail_(t_, __FILE__, __LINE__, msg);
        return NULL;
    }
    return &e->t;
}

enum { MAX_PROMPTS = 4, MAX_LAYERS = 128 };

typedef struct {
    const char       *id, *name;
    int32_t           n_prompt, n_tokens;
    int32_t          *tokens;
    double            kl_mean_py;
    const ath_tensor *res, *res32, *logits, *logits32;
    double            rms[MAX_LAYERS + 1], rms32[MAX_LAYERS + 1];
} prompt;

static bool read_prompt(ath_test *t_, const ath_json *rec, const ath_safetensors *g, ath_arena *a, prompt *p)
{
    int64_t np = 0, nt = 0;
    double kl = 0;
    const ath_json *tokens = ath_json_get(rec, "tokens");
    if (ath_json_to_str(ath_json_get(rec, "id"), &p->id, NULL) != ATH_OK ||
        ath_json_to_str(ath_json_get(rec, "name"), &p->name, NULL) != ATH_OK ||
        ath_json_to_int(ath_json_get(rec, "n_prompt"), &np) != ATH_OK ||
        ath_json_to_double(ath_json_get(rec, "kl_mean"), &kl) != ATH_OK || !tokens) {
        ath_test_fail_(t_, __FILE__, __LINE__, "forward record without id, name, n_prompt, kl_mean or tokens");
        return false;
    }
    nt = (int64_t)ath_json_count(tokens);
    p->n_prompt = (int32_t)np;
    p->n_tokens = (int32_t)nt;
    p->kl_mean_py = kl;
    p->tokens = ath_arena_alloc(a, (size_t)nt * sizeof *p->tokens, _Alignof(int32_t));
    for (int64_t i = 0; p->tokens && i < nt; i++) {
        int64_t v = -1;
        ath_json_to_int(ath_json_at(tokens, (size_t)i), &v);
        p->tokens[i] = (int32_t)v;
    }
    p->res = golden(t_, g, p->id, "residual");
    p->res32 = golden(t_, g, p->id, "residual32");
    p->logits = golden(t_, g, p->id, "logits");
    p->logits32 = golden(t_, g, p->id, "logits32");
    return p->tokens && p->res && p->res32 && p->logits && p->logits32 && np >= 1 && np <= nt;
}

static int32_t positions_to_run(int32_t n_tokens)
{
    const char *e = getenv("ATH_FORWARD_POSITIONS");
    long n = e && *e ? strtol(e, NULL, 10) : 0;
#ifdef ATH_DEBUG
    if (n <= 0)
        n = 4;
#endif
    return n > 0 && n < n_tokens ? (int32_t)n : n_tokens;
}

ATH_TEST(forward_golden)
{
    char path[ATH_PATH_MAX];
    if (ath_path_join(path, sizeof path, golden_dir(), "manifest.json") != ATH_OK || !ath_file_exists(path))
        ATH_SKIP("no golden data (tests/golden/README.md: run tests/golden/dump.py)");
    if (!model_file(path, sizeof path, "config.json"))
        ATH_SKIP("no checkpoint in $ATH_MODEL_DIR");

    ath_arena a;
    ath_arena_init(&a, 0);
    ath_file mf = {.fd = -1};
    ath_safetensors g = {.shards = NULL};
    ath_model m = {.family = NULL};
    const ath_json *manifest = NULL;
    ATH_REQUIRE_OK(ath_path_join(path, sizeof path, golden_dir(), "manifest.json"));
    ATH_REQUIRE_OK(ath_file_open(&mf, path));
    ATH_REQUIRE_OK(ath_json_parse(&a, mf.data, mf.size, &manifest, NULL));
    ATH_REQUIRE_OK(ath_path_join(path, sizeof path, golden_dir(), "golden.safetensors"));
    ATH_REQUIRE_OK(ath_safetensors_open_file(&g, &a, path));
    ATH_REQUIRE_OK(ath_model_open(&m, &a, model_dir()));
    const ath_json *recs = ath_json_get(manifest, "forward");
    const size_t n_prompts = ath_json_count(recs);
    ATH_REQUIRE(n_prompts >= 1 && n_prompts <= MAX_PROMPTS && m.n_layers <= MAX_LAYERS);
    const int64_t V = m.vocab_size, H = m.hidden_size, L1 = m.n_layers + 1;

    prompt P[MAX_PROMPTS];
    for (size_t i = 0; i < n_prompts; i++)
        ATH_REQUIRE(read_prompt(t_, ath_json_at(recs, i), &g, &a, &P[i]));

    /* two runs per prompt, bf16 and f32 activations, on the CPU; then the
     * same on the GPU when there is one */
    ath_pool pool = {.n_threads = 0};   /* started below, once nothing can return early */
    ath_engine *eng = NULL;
    if (ath_gpu_count() > 0) {
        eng = ath_arena_zalloc(&a, sizeof *eng, _Alignof(ath_engine));
        ATH_REQUIRE(eng);
        ATH_REQUIRE_OK(ath_engine_open(eng, &m, 0, &a));
    }
    job J[4 * MAX_PROMPTS];
    pthread_t th[2 * MAX_PROMPTS];
    const int n_cpu = 2 * (int)n_prompts, n_jobs = eng ? 2 * n_cpu : n_cpu;
    for (int k = 0; k < n_jobs; k++) {
        const int kc = k % n_cpu;
        prompt *p = &P[kc / 2];
        job *j = &J[k];
        int32_t n_run = positions_to_run(p->n_tokens);
        *j = (job){
            .m = &m, .g = &g, .pool = &pool, .eng = k >= n_cpu ? eng : NULL, .name = p->name,
            .act = kc % 2 ? ATH_DTYPE_F32 : ATH_DTYPE_BF16,
            .tokens = p->tokens, .n_prompt = p->n_prompt, .n_tokens = p->n_tokens, .n_run = n_run,
            .res = p->res, .res32 = p->res32, .rms = p->rms, .rms32 = p->rms32,
        };
        j->logits = ath_arena_alloc(&a, (size_t)n_run * (size_t)V * sizeof(float), 64);
        j->err = ath_arena_zalloc(&a, (size_t)L1 * sizeof(double), 8);
        j->err32 = ath_arena_zalloc(&a, (size_t)L1 * sizeof(double), 8);
        j->res_host = j->eng ? ath_arena_alloc(&a, (size_t)n_run * (size_t)H * sizeof(float), 64) : NULL;
        ATH_REQUIRE(j->logits && j->err && j->err32 && (!j->eng || j->res_host));
        if (k < n_cpu && kc % 2 == 0)   /* RMS of each layer's stream over the positions run */
            for (int64_t l = 0; l < L1; l++) {
                const size_t at = (size_t)(l * p->n_tokens) * (size_t)H;
                p->rms[l] = rms_of((const char *)p->res->data + at * ath_dtype_size(p->res->dtype), p->res->dtype,
                                   (int64_t)n_run * H);
                p->rms32[l] = rms_of((const char *)p->res32->data + at * ath_dtype_size(p->res32->dtype),
                                     p->res32->dtype, (int64_t)n_run * H);
            }
    }
    ATH_REQUIRE_OK(ath_pool_init(&pool, ath_pool_default_threads(), &a));
    int started = 0;
    while (started < n_cpu && pthread_create(&th[started], NULL, run_job, &J[started]) == 0)
        started++;
    for (int k = 0; k < started; k++)
        pthread_join(th[k], NULL);
    ath_pool_free(&pool);
    ATH_REQUIRE(started == n_cpu);
    for (int k = n_cpu; k < n_jobs; k++)   /* the GPU runs share the engine's scratch: one at a time */
        run_job(&J[k]);
    if (eng)
        ath_engine_close(eng);

    bool full = true;
    for (int k = 0; k < n_jobs; k++) {
        ATH_EXPECT_OK(J[k].st);
        full = full && J[k].n_run == J[k].n_tokens;
    }
    for (int k = 0; k < n_jobs; k++)
        if (J[k].st != ATH_OK)
            goto out;

    /* bf16 noise of the greedy decision: the largest change of the fp32
     * reference's top-2 margin in the bf16 reference, over every position */
    double noise = 0;
    for (size_t i = 0; i < n_prompts; i++)
        for (int32_t t = 0; t < J[2 * i].n_run; t++) {
            const char *l32 = (const char *)P[i].logits32->data + (size_t)t * (size_t)V * 4;
            const char *l16 = (const char *)P[i].logits->data + (size_t)t * (size_t)V * 2;
            int64_t x, y;
            top2(l32, ATH_DTYPE_F32, V, &x, &y);
            double m32 = ath_load_f32(l32, ATH_DTYPE_F32, x) - ath_load_f32(l32, ATH_DTYPE_F32, y);
            double m16 = ath_load_f32(l16, ATH_DTYPE_BF16, x) - ath_load_f32(l16, ATH_DTYPE_BF16, y);
            noise = fmax(noise, fabs(m16 - m32));
        }

    float *row = ath_arena_alloc(&a, (size_t)V * sizeof(float), 64);   /* a bf16 reference row, widened */
    ATH_REQUIRE(row);
    double all_kl[2] = {0}, all_ref[2] = {0};   /* sums over every position of every prompt (bf16 runs), per device */
    int all_n[2] = {0}, all_bad[2] = {0}, all_checked[2] = {0};
    printf("  %-6s %-8s %9s  %5s  %-9s %-9s %-9s  %-9s %-9s  %-7s %-7s %-6s\n", "prompt", "act", "positions", "s",
           "KL mean", "KL max", "ref mean", "ref max", "logits", "greedy", "ref", "res");
    for (int k = 0; k < n_jobs; k++) {
        const job *j = &J[k];
        const prompt *p = &P[(k % n_cpu) / 2];
        double kl_sum = 0, kl_max = 0, ref_sum = 0, ref_max = 0;
        int greedy_bad = 0, greedy_checked = 0, ath_differs = 0, ref_differs = 0;
        for (int32_t t = 0; t < j->n_run; t++) {
            const char *l32 = (const char *)p->logits32->data + (size_t)t * (size_t)V * 4;
            const char *l16 = (const char *)p->logits->data + (size_t)t * (size_t)V * 2;
            const float *got = j->logits + (size_t)t * (size_t)V;
            double kl = kl_row(l32, ATH_DTYPE_F32, got, V);
            kl_sum += kl;
            kl_max = fmax(kl_max, kl);
            /* the bf16 reference against the fp32 one, at the same position */
            for (int64_t v = 0; v < V; v++)
                row[v] = ath_load_f32(l16, ATH_DTYPE_BF16, v);
            double klr = kl_row(l32, ATH_DTYPE_F32, row, V);
            ref_differs += argmax_f32(row, V) != argmax_f32((const float *)l32, V);
            ref_sum += klr;
            ref_max = fmax(ref_max, klr);
            int64_t x, y;
            top2(l32, ATH_DTYPE_F32, V, &x, &y);
            double margin = ath_load_f32(l32, ATH_DTYPE_F32, x) - ath_load_f32(l32, ATH_DTYPE_F32, y);
            bool differs = argmax_f32(got, V) != x;
            ath_differs += differs;
            if (getenv("ATH_FORWARD_DETAIL"))
                printf("  position %-6s %-4s%s %3d  KL %.2e  ref %.2e  top-2 margin %6.3f%s%s\n", p->name,
                       j->act == ATH_DTYPE_F32 ? "f32" : "bf16", j->eng ? " gpu" : "", t, kl, klr, margin,
                       differs ? "  athanor differs" : "", argmax_f32(row, V) != x ? "  reference differs" : "");
            if (margin > noise) {
                greedy_checked++;
                greedy_bad += differs;
            }
        }
        const double n = j->n_run, kl_mean = kl_sum / n, ref_mean = ref_sum / n;
        double lrms = rms_of(p->logits32->data, ATH_DTYPE_F32, (int64_t)j->n_run * V);
        double lerr = score(j->logits, ATH_DTYPE_F32, p->logits32->data, ATH_DTYPE_F32, (int64_t)j->n_run * V, lrms);
        double res_worst = 0;
        int res_layer = 0;
        for (int64_t l = 0; l < L1; l++)
            if (j->err32[l] > res_worst) {
                res_worst = j->err32[l];
                res_layer = (int)l;
            }
        const char *dev = j->eng ? " gpu" : "";
        printf("  %-6s %-4s%-4s %4d of %-3d %5.1f  %-9.2e %-9.2e %-9.2e  %-9.2e %-9.2e  %2d of %-2d %-7d %.2e (layer %d)\n",
               p->name, j->act == ATH_DTYPE_F32 ? "f32" : "bf16", dev, j->n_run, j->n_tokens, j->seconds, kl_mean,
               kl_max, ref_mean, ref_max, lerr, ath_differs, j->n_run, ref_differs, res_worst, res_layer);

        char msg[300];
        if (j->act == ATH_DTYPE_F32) {
            for (int64_t l = 0; l < L1; l++)
                if (!(j->err32[l] <= TOL_F32_RESIDUAL)) {
                    snprintf(msg, sizeof msg, "%s f32%s: residual entering layer %d: %.3g > %.3g", p->name, dev,
                             (int)l, j->err32[l], TOL_F32_RESIDUAL);
                    ath_test_fail_(t_, __FILE__, __LINE__, msg);
                }
            if (!(lerr <= TOL_F32_LOGITS)) {
                snprintf(msg, sizeof msg, "%s f32%s: logits %.3g > %.3g", p->name, dev, lerr, TOL_F32_LOGITS);
                ath_test_fail_(t_, __FILE__, __LINE__, msg);
            }
            continue;
        }
        if (full) {
            /* the KL of the bf16 reference as dump.py computed it: checks kl_row */
            if (!(fabs(ref_mean - p->kl_mean_py) <= 1e-9 + 1e-6 * p->kl_mean_py)) {
                snprintf(msg, sizeof msg, "%s: reference KL %.9g here, %.9g in dump.py", p->name, ref_mean,
                         p->kl_mean_py);
                ath_test_fail_(t_, __FILE__, __LINE__, msg);
            }
        }
        const int d = j->eng != NULL;
        all_kl[d] += kl_sum;
        all_ref[d] += ref_sum;
        all_n[d] += j->n_run;
        all_bad[d] += greedy_bad;
        all_checked[d] += greedy_checked;
    }
    for (int d = 0; d < (eng ? 2 : 1); d++) {
        printf("  bf16 on the %s, all %d positions run: mean KL %.3e, bf16 reference %.3e (ratio %.2f); greedy\n"
               "  token differs from the fp32 reference at %d of %d positions whose top-2 margin exceeds the bf16 noise\n",
               d ? "GPU" : "CPU", all_n[d], all_kl[d] / all_n[d], all_ref[d] / all_n[d], all_kl[d] / all_ref[d],
               all_bad[d], all_checked[d]);
        if (full) {
            char msg[200];
            if (!(all_kl[d] <= all_ref[d])) {
                snprintf(msg, sizeof msg, "bf16 %s: mean KL %.3g > bf16 reference %.3g", d ? "GPU" : "CPU",
                         all_kl[d] / all_n[d], all_ref[d] / all_n[d]);
                ath_test_fail_(t_, __FILE__, __LINE__, msg);
            }
            if (all_bad[d]) {
                snprintf(msg, sizeof msg, "bf16 %s: greedy token differs at %d of %d positions above the noise",
                         d ? "GPU" : "CPU", all_bad[d], all_checked[d]);
                ath_test_fail_(t_, __FILE__, __LINE__, msg);
            }
        }
    }
    printf("  greedy: bf16 noise of the top-2 margin %.3f; columns greedy and ref count the positions where\n"
           "  athanor and the bf16 reference pick another token than the fp32 reference\n",
           noise);
    if (!full)
        printf("  first positions only (ATH_FORWARD_POSITIONS or DEBUG build): KL gate and greedy not checked\n");
    if (getenv("ATH_FORWARD_LAYERS"))
        for (int k = 0; k < n_jobs; k++)
            for (int64_t l = 0; l < L1; l++)
                printf("  layer %-6s %-4s%s %2d  vs bf16 ref %.2e  vs fp32 ref %.2e\n", J[k].name,
                       J[k].act == ATH_DTYPE_F32 ? "f32" : "bf16", J[k].eng ? " gpu" : "", (int)l, J[k].err[l],
                       J[k].err32[l]);

out:
    ath_model_close(&m);
    ath_safetensors_close(&g);
    ath_file_close(&mf);
    ath_arena_free(&a);
}

/* ---- the same weights from a GGUF file (step 12) ------------------------------------ */

/* The logits of tokens for the model in dir, on the CPU (eng NULL) or the GPU. */
static float *gguf_logits(ath_test *t_, const char *dir, ath_dtype act, ath_engine *eng_mem, bool gpu, ath_pool *pool,
                          ath_arena *a, const int32_t *tok, int32_t T)
{
    ath_model *m = ath_arena_zalloc(a, sizeof *m, _Alignof(ath_model));
    if (!m || ath_model_open(m, a, dir) != ATH_OK)
        return NULL;
    ath_state s;
    ath_exec x = {.pool = pool};
    if (gpu) {
        if (ath_engine_open(eng_mem, m, 0, a) != ATH_OK || ath_engine_state(eng_mem, &s, act, 64, a) != ATH_OK)
            return NULL;
        x = eng_mem->exec;
    } else if (ath_state_init(&s, m, act, 64, a) != ATH_OK) {
        return NULL;
    }
    ath_tensor lg;
    ATH_EXPECT_OK(ath_tensor_init(&lg, ATH_DTYPE_F32, 2, (const int64_t[]){T, m->vocab_size}, NULL, ATH_TIER_RAM));
    lg.data = ath_arena_alloc(a, ath_tensor_nbytes(&lg), 64);
    ath_arena scratch;
    ath_arena_init(&scratch, 0);
    const ath_batch b = {.tokens = tok, .n_tokens = T, .logits = &lg};
    const ath_status st = lg.data ? ath_model_forward(m, &b, &s, NULL, &scratch, &x) : ATH_ERR_NOMEM;
    ath_arena_free(&scratch);
    if (gpu)
        ath_engine_close(eng_mem);
    return st == ATH_OK ? lg.data : NULL;
}

/* Qwen3.5-0.8B from llama.cpp's bf16 GGUF (step 9; tests/README.md, GGUF)
 * against its safetensors: the same weights under other names, merged
 * projections bound as slices, norm weights stored as 1 + w. The one value
 * that differs is A = −exp(A_log), which the converter computed with torch's
 * exp where athanor computes it with the device's expf (one or two ulps in
 * some decays). On the CPU, with bf16 activations, that changes no bit of
 * the logits; with f32 the logits stay within 1e-4. On the GPU CUDA's expf
 * moves more decays and bf16 roundings follow: the greedy token is the same
 * wherever the top-2 margin exceeds 0.5, and f32 stays within 1e-4. */
ATH_TEST(gguf_matches_safetensors)
{
    const char *gd = getenv("ATH_GGUF_DIR") ? getenv("ATH_GGUF_DIR") : "models/Qwen3.5-0.8B-gguf";
    char path[1024];
    if (ath_path_join(path, sizeof path, gd, "config.json") != ATH_OK || !ath_file_exists(path))
        ATH_SKIP("no GGUF directory (tests/README.md, GGUF: models/Qwen3.5-0.8B-gguf or $ATH_GGUF_DIR)");
    if (ath_path_join(path, sizeof path, model_dir(), "config.json") != ATH_OK || !ath_file_exists(path))
        ATH_SKIP("no checkpoint in $ATH_MODEL_DIR");
    static const int32_t tok[12] = {9707, 11, 1246, 525, 498, 30, 220, 16, 10, 16, 28, 33};
    ath_arena a;
    ath_arena_init(&a, 0);
    ath_pool pool;
    ATH_REQUIRE_OK(ath_pool_init(&pool, ath_pool_default_threads(), &a));
    ath_engine *eng = ath_arena_zalloc(&a, sizeof *eng, _Alignof(ath_engine));
    ATH_REQUIRE(eng);
    const int64_t n = 12 * 248320;
    for (int dev = 0; dev < (ath_gpu_count() > 0 ? 2 : 1); dev++)
        for (int f32 = 0; f32 < 2; f32++) {
            const ath_dtype act = f32 ? ATH_DTYPE_F32 : ATH_DTYPE_BF16;
            const float *ls = gguf_logits(t_, model_dir(), act, eng, dev, &pool, &a, tok, 12);
            const float *lg = gguf_logits(t_, gd, act, eng, dev, &pool, &a, tok, 12);
            ATH_REQUIRE(ls && lg);
            double worst = 0;
            int64_t differ = 0;
            for (int64_t i = 0; i < n; i++) {
                const double d = fabs((double)ls[i] - (double)lg[i]);
                worst = d > worst ? d : worst;
                differ += ls[i] != lg[i];
            }
            int flips = 0;   /* positions with a clear greedy token that changed */
            for (int t = 0; t < 12; t++) {
                const float *a0 = ls + (int64_t)t * 248320, *b0 = lg + (int64_t)t * 248320;
                int ia = 0, ib = 0;
                for (int v = 1; v < 248320; v++) {
                    ia = a0[v] > a0[ia] ? v : ia;
                    ib = b0[v] > b0[ib] ? v : ib;
                }
                float second = -INFINITY;
                for (int v = 0; v < 248320; v++)
                    if (v != ia && a0[v] > second)
                        second = a0[v];
                flips += ia != ib && a0[ia] - second > 0.5f;
            }
            printf("  %s %s: GGUF against safetensors, %lld of %lld logits differ, worst %.3g, %d greedy flips\n",
                   dev ? "GPU" : "CPU", f32 ? "f32" : "bf16", (long long)differ, (long long)n, worst, flips);
            if (f32)
                ATH_EXPECT(worst <= 1e-4);
            else if (!dev)
                ATH_EXPECT(differ == 0);
            ATH_EXPECT(flips == 0);
        }
    ath_pool_free(&pool);
    ath_arena_free(&a);
}

static const ath_test_case tests[] = {
    ATH_TEST_ENTRY(state_specs),
    ATH_TEST_ENTRY(forward_rejects),
    ATH_TEST_ENTRY(forward_pieces_bf16),
    ATH_TEST_ENTRY(forward_pieces_f32),
    ATH_TEST_ENTRY(forward_hooks),
    ATH_TEST_ENTRY(real_state_specs),
    ATH_TEST_ENTRY(forward_golden),
    ATH_TEST_ENTRY(gguf_matches_safetensors),
};

int main(int argc, char **argv)
{
    return ath_test_main(tests, ATH_ARRAY_LEN(tests), argc, argv);
}
