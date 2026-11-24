/* test_golden.c: every op of the reference implementation, replayed through
 * athanor's _ref functions on the real weights.
 *
 * tests/golden/dump.py runs the pinned reference over a fixed prompt and one
 * decode step and records each op it computes (tests/golden/README.md). For
 * every record this suite runs the matching _ref function twice:
 *
 *   native  inputs as recorded (bf16 activations), outputs in the recorded
 *           dtype, compared with `out`: what the reference computed, with
 *           its own intermediate roundings;
 *   exact   inputs widened to f32, outputs in f32, compared with `out32`:
 *           the same reference function rerun in fp32. No bf16 rounding is
 *           left on either side, so this checks the math tightly.
 *
 * Weights come from the checkpoint through ath_model_open, by the checkpoint
 * names the records carry. The error of a comparison is
 *
 *   score = max_i |got_i − want_i| / (|want_i| + rms(want))
 *
 * a relative error for elements of typical size or larger, and an error
 * relative to the tensor's RMS for elements near zero (where a relative error
 * means nothing). One bf16 rounding is a score of at most 2^-8 (half an ulp,
 * relative); one bf16 ulp is at most 2^-7. The tolerances and why they are
 * what they are: tests/README.md, section Tolerances.
 *
 * Needs $ATH_GOLDEN_DIR (default tests/golden/data/qwen35-0.8b) and
 * $ATH_MODEL_DIR; skipped when either is missing.
 */
#include "test.h"
#include "util.h"

#include "core/safetensors.h"
#include "model/model.h"
#include "ops/ops.h"

#include <stdarg.h>

/* ---- tolerances -------------------------------------------------------------- */

typedef struct {
    const char *op, *slot;
    double      native;   /* vs the reference's own output (bf16, or f32 where it is f32) */
    double      exact;    /* vs the fp32 rerun                                              */
} tolerance;

/* One bf16 ulp, relative: 2^-7. athanor rounds each output once; where the
 * reference also rounds once, the two can differ by one ulp (different
 * summation orders put the fp32 values on either side of a rounding
 * boundary). Where the reference rounds two or three times (an activation,
 * then a product), each rounding adds up to half an ulp: the tolerance is 2.
 * The exact tolerances are 1e-5, raised where a long reduction or the
 * recurrent state brings the measured error within a factor of 4 of it;
 * tests/README.md has the measured values. */
#define ULP (1.0 / 128)

static const tolerance tolerances[] = {
    /* bit-exact: a copy, and one rounding of the same fp32 sum */
    {"embed",         "out",       0,         0},
    {"add",           "out",       0,         0},
    {"conv",          "state_out", 0,         0},
    /* one rounding on both sides */
    {"matmul",        "out",       ULP,       2e-5},
    {"rmsnorm",       "out",       ULP,       1e-5},
    {"rmsnorm_gated", "out",       ULP,       1e-5},
    {"gdn_gates",     "beta",      ULP,       1e-5},
    {"gdn",           "out",       ULP,       2e-5},
    /* the reference rounds an activation, then the product (rope: both
     * products and their sum) */
    {"swiglu",        "out",       2 * ULP,   1e-5},
    {"sigmoid_mul",   "out",       2 * ULP,   1e-5},
    {"conv",          "out",       2 * ULP,   1e-5},
    {"rope",          "out",       2 * ULP,   1e-5},
    /* f32 on both sides */
    {"l2norm",        "out",       1e-5,      1e-5},
    {"gdn_gates",     "g",         1e-5,      1e-5},
    /* the fp32 state accumulates 75 tokens of updates; the prefill reference
     * is the chunked form, a different order of the same sums */
    {"gdn",           "state_out", 1e-4,      1e-4},
    /* eager_attention_forward rounds q·kᵀ to bf16 before the softmax: an
     * error of up to 2^-8 |s| per score, 0.06 at this prompt's largest score
     * (16.3), so up to 6% on a probability. athanor keeps scores in fp32.
     * With fp32 scores the reference lands within 2 ulps of the fp32 rerun
     * (dump.py checks it), so the score rounding is the whole difference. */
    {"attn",          "out",       0.2,       5e-5},
};

static const tolerance *tolerance_of(const char *op, const char *slot)
{
    for (size_t i = 0; i < ATH_ARRAY_LEN(tolerances); i++)
        if (strcmp(tolerances[i].op, op) == 0 && strcmp(tolerances[i].slot, slot) == 0)
            return &tolerances[i];
    return NULL;
}

/* ---- statistics per op, output and phase ------------------------------------- */

typedef struct {
    char   op[24], slot[16];
    char   phase;
    int    n_native, n_exact;
    double native, exact;     /* worst scores */
    int    ulp;               /* worst bf16 ulp distance in native comparisons */
    char   worst_native[96], worst_exact[96];
} stat;

static stat stats[64];
static int  n_stats;

static stat *stat_of(const char *op, const char *slot, char phase)
{
    for (int i = 0; i < n_stats; i++)
        if (strcmp(stats[i].op, op) == 0 && strcmp(stats[i].slot, slot) == 0 && stats[i].phase == phase)
            return &stats[i];
    if (n_stats == (int)ATH_ARRAY_LEN(stats))
        abort();
    stat *s = &stats[n_stats++];
    memset(s, 0, sizeof *s);
    snprintf(s->op, sizeof s->op, "%s", op);
    snprintf(s->slot, sizeof s->slot, "%s", slot);
    s->phase = phase;
    return s;
}

/* ---- one record ---------------------------------------------------------------- */

typedef struct {
    ath_test             *t;
    ath_arena            *a;          /* scratch, reset after each record */
    const ath_safetensors *g;
    const ath_model      *m;
    const ath_json       *rec;
    const char           *id, *op;
    bool                  exact;
    bool                  bad;        /* the record could not be replayed */
} ctx;

static void fail(ctx *c, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void fail(ctx *c, const char *fmt, ...)
{
    char msg[400], line[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    snprintf(line, sizeof line, "%s (%s, %s): %s", c->id, c->op, c->exact ? "exact" : "native", msg);
    ath_test_fail_(c->t, __FILE__, __LINE__, line);
    c->bad = true;
}

/* Golden tensor "<id>.<name>", or NULL. */
static const ath_tensor *slot(ctx *c, const char *name)
{
    char key[256];
    snprintf(key, sizeof key, "%s.%s", c->id, name);
    const ath_safetensors_entry *e = ath_safetensors_find(c->g, key);
    return e ? &e->t : NULL;
}

static ath_tensor *new_tensor(ctx *c, ath_dtype dt, const ath_tensor *like)
{
    ath_tensor *t = ath_arena_alloc(c->a, sizeof *t, _Alignof(ath_tensor));
    if (!t || ath_tensor_init(t, dt, like->ndim, like->shape, NULL, ATH_TIER_RAM) != ATH_OK)
        abort();
    t->data = ath_arena_zalloc(c->a, ath_tensor_nbytes(t) + 1, 64);
    if (!t->data)
        abort();
    return t;
}

static ath_tensor *copy_as(ctx *c, const ath_tensor *src, ath_dtype dt)
{
    ath_tensor *t = new_tensor(c, dt, src);
    for (int64_t i = 0; i < ath_tensor_numel(src); i++)
        ath_store_f32(t->data, dt, i, ath_load_f32(src->data, src->dtype, i));
    return t;
}

/* An input: as recorded (native) or widened to f32 (exact). */
static const ath_tensor *in(ctx *c, const char *name)
{
    const ath_tensor *t = slot(c, name);
    if (!t) {
        fail(c, "golden tensor '%s' missing", name);
        return NULL;
    }
    return c->exact && t->dtype != ATH_DTYPE_F32 ? copy_as(c, t, ATH_DTYPE_F32) : t;
}

/* An output buffer shaped like the golden `name`: its dtype (native) or f32. */
static ath_tensor *out(ctx *c, const char *name)
{
    const ath_tensor *t = slot(c, name);
    if (!t) {
        fail(c, "golden tensor '%s' missing", name);
        return NULL;
    }
    return new_tensor(c, c->exact ? ATH_DTYPE_F32 : t->dtype, t);
}

static const char *str(ctx *c, const char *key)
{
    const char *s = NULL;
    if (ath_json_to_str(ath_json_get(c->rec, key), &s, NULL) != ATH_OK)
        fail(c, "record has no string '%s'", key);
    return s;
}

static double num(ctx *c, const char *key)
{
    double v = 0;
    if (ath_json_to_double(ath_json_get(c->rec, key), &v) != ATH_OK)
        fail(c, "record has no number '%s'", key);
    return v;
}

/* The checkpoint tensor the record names under `key`, through the model's params. */
static const ath_tensor *weight(ctx *c, const char *key)
{
    const char *name = str(c, key);
    for (uint32_t i = 0; name && i < c->m->n_params; i++)
        if (strcmp(c->m->params[i].name, name) == 0)
            return &c->m->params[i].t;
    if (name)
        fail(c, "no param named %s", name);
    return NULL;
}

static int ulp_distance(float a, float b)
{
    /* bf16 patterns as integers ordered like the values they encode */
    int32_t x = ath_f32_to_bf16(a), y = ath_f32_to_bf16(b);
    x = x & 0x8000 ? 0x8000 - x : x;
    y = y & 0x8000 ? 0x8000 - y : y;
    return x > y ? x - y : y - x;
}

static void compare(ctx *c, const char *name, const ath_tensor *got)
{
    char name32[32];
    snprintf(name32, sizeof name32, "%s32", name);
    const ath_tensor *want = slot(c, c->exact ? name32 : name);
    if (c->exact && !want) {                     /* f32 goldens serve both checks */
        want = slot(c, name);
        if (want && want->dtype != ATH_DTYPE_F32)
            return;                              /* nothing to compare in fp32 */
    }
    if (!want || !got) {
        fail(c, "golden tensor '%s' missing", c->exact ? name32 : name);
        return;
    }
    if (!ath_tensor_same_shape(want, got)) {
        fail(c, "'%s': shapes differ", name);
        return;
    }
    const int64_t n = ath_tensor_numel(want);
    double ss = 0;
    for (int64_t i = 0; i < n; i++) {
        double w = ath_load_f32(want->data, want->dtype, i);
        ss += w * w;
    }
    const double rms = n ? sqrt(ss / (double)n) : 0;
    double score = 0;
    int ulp = 0;
    int64_t worst = 0;
    for (int64_t i = 0; i < n; i++) {
        double w = ath_load_f32(want->data, want->dtype, i), g = ath_load_f32(got->data, got->dtype, i);
        double e = fabs(g - w);
        double s = e == 0 ? 0 : e / (fabs(w) + rms);
        if (!(s <= score)) {                     /* also catches NaN */
            score = isnan(s) ? INFINITY : s;
            worst = i;
        }
        if (!c->exact && want->dtype == ATH_DTYPE_BF16) {
            int u = ulp_distance((float)g, (float)w);
            if (u > ulp)
                ulp = u;
        }
    }
    stat *s = stat_of(c->op, name, c->id[0]);
    if (c->exact) {
        s->n_exact++;
        if (score > s->exact || !s->worst_exact[0]) {
            s->exact = score;
            snprintf(s->worst_exact, sizeof s->worst_exact, "%s[%lld]", c->id, (long long)worst);
        }
    } else {
        s->n_native++;
        if (ulp > s->ulp)
            s->ulp = ulp;
        if (score > s->native || !s->worst_native[0]) {
            s->native = score;
            snprintf(s->worst_native, sizeof s->worst_native, "%s[%lld]", c->id, (long long)worst);
        }
    }
}

#define RUN(expr)                                                                \
    do {                                                                         \
        ath_status s_ = (expr);                                                  \
        if (s_ != ATH_OK) {                                                      \
            fail(c, "%s: %s", #expr, ath_status_str(s_));                        \
            return;                                                              \
        }                                                                        \
    } while (0)

#define NEED(...)                                                                \
    do {                                                                         \
        const void *need_[] = {__VA_ARGS__};                                     \
        for (size_t i_ = 0; i_ < ATH_ARRAY_LEN(need_); i_++)                     \
            if (!need_[i_] || c->bad)                                            \
                return;                                                          \
    } while (0)

/* ---- one handler per op --------------------------------------------------------- */

static void run_embed(ctx *c)
{
    const ath_tensor *ids = slot(c, "ids"), *w = weight(c, "weight");
    ath_tensor *y = out(c, "out");
    NEED(ids, w, y);
    RUN(ath_embed_ref(w, ids, y));
    compare(c, "out", y);
}

static void run_matmul(ctx *c)
{
    const ath_tensor *x = in(c, "x"), *w = weight(c, "weight");
    ath_tensor *y = out(c, "out");
    NEED(x, w, y);
    RUN(ath_matmul_ref(x, w, y));
    compare(c, "out", y);
}

static void run_rmsnorm(ctx *c)
{
    const ath_tensor *x = in(c, "x"), *w = weight(c, "weight");
    ath_tensor *y = out(c, "out");
    float offset = (float)num(c, "offset"), eps = (float)num(c, "eps");
    NEED(x, w, y);
    RUN(ath_rmsnorm_ref(x, w, offset, eps, y));
    compare(c, "out", y);
}

static void run_rmsnorm_gated(ctx *c)
{
    const ath_tensor *x = in(c, "x"), *z = in(c, "z"), *w = weight(c, "weight");
    ath_tensor *y = out(c, "out");
    float eps = (float)num(c, "eps");
    const char *act = str(c, "act");
    NEED(x, z, w, y, act);
    if (strcmp(act, "silu") != 0) {
        fail(c, "gate activation %s", act);
        return;
    }
    RUN(ath_rmsnorm_gated_ref(x, z, w, eps, y));
    compare(c, "out", y);
}

static void run_l2norm(ctx *c)
{
    const ath_tensor *x = in(c, "x");
    ath_tensor *y = out(c, "out");
    float eps = (float)num(c, "eps");
    NEED(x, y);
    RUN(ath_l2norm_ref(x, eps, y));
    compare(c, "out", y);
}

static void run_rope(ctx *c)
{
    const ath_tensor *x = in(c, "x"), *pos = slot(c, "pos");
    ath_tensor *y = out(c, "out");
    int32_t rope_dim = (int32_t)num(c, "rope_dim");
    float theta = (float)num(c, "theta");
    NEED(x, pos, y);
    RUN(ath_rope_ref(x, pos, rope_dim, theta, y));
    compare(c, "out", y);
}

static void run_attn(ctx *c)
{
    const ath_tensor *q = in(c, "q"), *k = in(c, "k"), *v = in(c, "v");
    ath_tensor *y = out(c, "out");
    float scale = (float)num(c, "scale");
    NEED(q, k, v, y);
    RUN(ath_attn_ref(q, k, v, scale, y));
    compare(c, "out", y);
}

static void run_swiglu(ctx *c)
{
    const ath_tensor *g = in(c, "gate"), *u = in(c, "up");
    ath_tensor *y = out(c, "out");
    NEED(g, u, y);
    RUN(ath_swiglu_ref(g, u, y));
    compare(c, "out", y);
}

static void run_sigmoid_mul(ctx *c)
{
    const ath_tensor *x = in(c, "x"), *g = in(c, "gate");
    ath_tensor *y = out(c, "out");
    NEED(x, g, y);
    RUN(ath_sigmoid_mul_ref(x, g, y));
    compare(c, "out", y);
}

static void run_add(ctx *c)
{
    const ath_tensor *a = in(c, "a"), *b = in(c, "b");
    ath_tensor *y = out(c, "out");
    NEED(a, b, y);
    RUN(ath_add_ref(a, b, y));
    compare(c, "out", y);
}

static void run_conv(ctx *c)
{
    const ath_tensor *x = in(c, "x"), *w = weight(c, "weight");
    const ath_tensor *s_in = slot(c, "state_in"), *s_out = slot(c, "state_out");
    ath_tensor *y = out(c, "out");
    const char *act = str(c, "act");
    NEED(x, w, y, s_out, act);
    if (strcmp(act, "silu") != 0) {
        fail(c, "activation %s", act);
        return;
    }
    ath_dtype sdt = c->exact ? ATH_DTYPE_F32 : s_out->dtype;
    ath_tensor *state = s_in ? copy_as(c, s_in, sdt) : new_tensor(c, sdt, s_out);   /* prefill: zeros */
    RUN(ath_conv_ref(x, w, state, ATH_ACT_SILU, y));
    compare(c, "out", y);
    compare(c, "state_out", state);
}

static void run_gdn_gates(ctx *c)
{
    const ath_tensor *a = in(c, "a"), *b = in(c, "b");
    const ath_tensor *a_log = weight(c, "a_log"), *dt_bias = weight(c, "dt_bias");
    ath_tensor *g = out(c, "g"), *beta = out(c, "beta");
    NEED(a, b, a_log, dt_bias, g, beta);
    RUN(ath_gdn_gates_ref(a, b, a_log, dt_bias, g, beta));
    compare(c, "g", g);
    compare(c, "beta", beta);
}

/* The reference function normalizes q and k inside (use_qk_l2norm_in_kernel);
 * athanor composes l2norm and the recurrence. */
static void run_gdn(ctx *c)
{
    const ath_tensor *q = in(c, "q"), *k = in(c, "k"), *v = in(c, "v"), *g = in(c, "g"), *beta = in(c, "beta");
    const ath_tensor *s_in = slot(c, "state_in"), *s_out = slot(c, "state_out");
    ath_tensor *y = out(c, "out");
    float eps = (float)num(c, "l2norm_eps"), q_scale = (float)num(c, "q_scale");
    NEED(q, k, v, g, beta, s_out, y);
    ath_tensor *qn = new_tensor(c, ATH_DTYPE_F32, q), *kn = new_tensor(c, ATH_DTYPE_F32, k);
    ath_tensor *state = s_in ? copy_as(c, s_in, ATH_DTYPE_F32) : new_tensor(c, ATH_DTYPE_F32, s_out);
    RUN(ath_l2norm_ref(q, eps, qn));
    RUN(ath_l2norm_ref(k, eps, kn));
    RUN(ath_gdn_ref(qn, kn, v, g, beta, q_scale, state, y));
    compare(c, "out", y);
    compare(c, "state_out", state);
}

static const struct {
    const char *op;
    void      (*run)(ctx *c);
} handlers[] = {
    {"embed", run_embed},         {"matmul", run_matmul},   {"rmsnorm", run_rmsnorm},
    {"rmsnorm_gated", run_rmsnorm_gated}, {"l2norm", run_l2norm}, {"rope", run_rope},
    {"attn", run_attn},           {"swiglu", run_swiglu},   {"sigmoid_mul", run_sigmoid_mul},
    {"add", run_add},             {"conv", run_conv},       {"gdn_gates", run_gdn_gates},
    {"gdn", run_gdn},
};

/* ---- the suite -------------------------------------------------------------------- */

static const char *golden_dir(void)
{
    const char *d = getenv("ATH_GOLDEN_DIR");
    return (d && *d) ? d : "tests/golden/data/qwen35-0.8b";
}

static void print_stats(void)
{
    printf("  %-14s %-10s %-5s %5s  %-9s %-4s %-8s  %5s  %-9s %-8s\n", "op", "output", "phase", "n",
           "native", "ulp", "tol", "n", "exact", "tol");
    for (int i = 0; i < n_stats; i++) {
        const stat *s = &stats[i];
        const tolerance *tol = tolerance_of(s->op, s->slot);
        printf("  %-14s %-10s %-5s %5d  %-9.2e %-4d %-8.1e  %5d  %-9.2e %-8.1e\n", s->op, s->slot,
               s->phase == 'p' ? "pre" : "dec", s->n_native, s->native, s->ulp, tol ? tol->native : -1,
               s->n_exact, s->exact, tol ? tol->exact : -1);
    }
    if (getenv("ATH_GOLDEN_WORST"))
        for (int i = 0; i < n_stats; i++)
            printf("  worst %-14s %-10s native %s, exact %s\n", stats[i].op, stats[i].slot,
                   stats[i].worst_native, stats[i].worst_exact);
}

ATH_TEST(golden_ops)
{
    char path[ATH_PATH_MAX];
    if (ath_path_join(path, sizeof path, golden_dir(), "manifest.json") != ATH_OK || !ath_file_exists(path))
        ATH_SKIP("no golden data (tests/golden/README.md: run tests/golden/dump.py)");
    if (!model_file(path, sizeof path, "config.json"))
        ATH_SKIP("no checkpoint in $ATH_MODEL_DIR");

    ath_arena a, scratch;
    ath_arena_init(&a, 0);
    ath_arena_init(&scratch, 0);
    ath_file mf = {.fd = -1};
    ath_safetensors g = {.shards = NULL};
    ath_model m = {.family = NULL};
    const ath_json *manifest = NULL;

    ATH_REQUIRE_OK(ath_path_join(path, sizeof path, golden_dir(), "manifest.json"));
    ATH_REQUIRE_OK(ath_file_open(&mf, path));
    ath_status s = ath_json_parse(&a, mf.data, mf.size, &manifest, NULL);
    ATH_EXPECT_OK(s);
    if (s == ATH_OK) {
        ATH_REQUIRE_OK(ath_path_join(path, sizeof path, golden_dir(), "golden.safetensors"));
        s = ath_safetensors_open_file(&g, &a, path);
        ATH_EXPECT_OK(s);
    }
    if (s == ATH_OK) {
        s = ath_model_open(&m, &a, model_dir());
        ATH_EXPECT_OK(s);
    }
    const ath_json *records = ath_json_get(manifest, "records");
    const size_t n = ath_json_count(records);
    ATH_EXPECT(s != ATH_OK || n > 0);

    int replayed = 0;
    for (size_t i = 0; s == ATH_OK && i < n; i++) {
        ctx c = {.t = t_, .a = &scratch, .g = &g, .m = &m, .rec = ath_json_at(records, i)};
        if (ath_json_to_str(ath_json_get(c.rec, "id"), &c.id, NULL) != ATH_OK ||
            ath_json_to_str(ath_json_get(c.rec, "op"), &c.op, NULL) != ATH_OK) {
            ath_test_fail_(t_, __FILE__, __LINE__, "record without id or op");
            continue;
        }
        size_t h = 0;
        while (h < ATH_ARRAY_LEN(handlers) && strcmp(handlers[h].op, c.op) != 0)
            h++;
        if (h == ATH_ARRAY_LEN(handlers)) {
            fail(&c, "unknown op");
            continue;
        }
        for (int mode = 0; mode < 2 && !c.bad; mode++) {
            ath_arena_pos p = ath_arena_mark(&scratch);
            c.exact = mode == 1;
            handlers[h].run(&c);
            ath_arena_reset(&scratch, p);
        }
        replayed += !c.bad;
    }
    printf("  %d of %zu records replayed (%s)\n", replayed, n, golden_dir());
    print_stats();

    /* Every op and output must be compared, and within its tolerance. */
    for (int i = 0; i < n_stats; i++) {
        const stat *st = &stats[i];
        const tolerance *tol = tolerance_of(st->op, st->slot);
        char msg[300];
        if (!tol) {
            snprintf(msg, sizeof msg, "%.23s %.15s: no tolerance", st->op, st->slot);
            ath_test_fail_(t_, __FILE__, __LINE__, msg);
            continue;
        }
        if (!(st->native <= tol->native)) {
            snprintf(msg, sizeof msg, "%.23s %.15s native %.3g > %.3g at %.95s", st->op, st->slot, st->native,
                     tol->native, st->worst_native);
            ath_test_fail_(t_, __FILE__, __LINE__, msg);
        }
        if (!(st->exact <= tol->exact)) {
            snprintf(msg, sizeof msg, "%.23s %.15s exact %.3g > %.3g at %.95s", st->op, st->slot, st->exact,
                     tol->exact, st->worst_exact);
            ath_test_fail_(t_, __FILE__, __LINE__, msg);
        }
    }
    for (size_t h = 0; s == ATH_OK && h < ATH_ARRAY_LEN(handlers); h++) {
        bool seen = false;
        for (int i = 0; i < n_stats; i++)
            seen = seen || strcmp(stats[i].op, handlers[h].op) == 0;
        ATH_EXPECT(seen);
    }

    ath_model_close(&m);
    ath_safetensors_close(&g);
    ath_file_close(&mf);
    ath_arena_free(&scratch);
    ath_arena_free(&a);
}

static const ath_test_case tests[] = {
    ATH_TEST_ENTRY(golden_ops),
};

int main(int argc, char **argv)
{
    return ath_test_main(tests, ATH_ARRAY_LEN(tests), argc, argv);
}
