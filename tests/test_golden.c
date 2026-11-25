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
 * With a GPU (make cuda test) every record also runs through the GPU kernel,
 * ath_<op>_gpu, on device copies of the same inputs, and its output is
 * compared with the _ref output, native and exact: the kernels against the
 * references on real inputs (tests/README.md, GPU kernels).
 *
 * Needs $ATH_GOLDEN_DIR (default tests/golden/data/qwen35-0.8b) and
 * $ATH_MODEL_DIR; skipped when either is missing.
 */
#include "test.h"
#include "util.h"

#include "backend/backend.h"
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

/* GPU kernel against the _ref function on the same inputs. native: bf16
 * outputs within one ulp (score at most 2^-7: the two round fp32 values
 * that differ in the last bits), two where the op rounds an intermediate
 * value as well (rope's tables, rmsnorm_gated's normalized value); f32
 * outputs (l2norm, g, the state) within their exact tolerance. exact: the
 * two fp32 computations, at least 4 times the worst error measured
 * (tests/README.md, GPU kernels). */
static const tolerance gpu_tolerances[] = {
    {"embed",         "out",       0,         0},
    {"add",           "out",       0,         0},
    {"conv",          "state_out", 0,         0},
    {"matmul",        "out",       ULP,       2e-5},
    {"rmsnorm",       "out",       ULP,       1.5e-5},
    {"rmsnorm_gated", "out",       2 * ULP,   1e-5},
    {"gdn_gates",     "beta",      ULP,       1e-5},
    {"gdn",           "out",       ULP,       3e-5},
    {"swiglu",        "out",       ULP,       1e-5},
    {"sigmoid_mul",   "out",       ULP,       1e-5},
    {"conv",          "out",       ULP,       1e-5},
    {"rope",          "out",       2 * ULP,   1e-5},
    {"l2norm",        "out",       1e-5,      1e-5},
    {"gdn_gates",     "g",         1e-5,      1e-5},
    /* 75 tokens of updates summed in another order */
    {"gdn",           "state_out", 5e-5,      5e-5},
    /* native: a prompt takes the tensor cores (step 10), whose probabilities
     * are bf16: each weight moves by 2^-8 at most (test_gpu_ops,
     * gpu_attn_mma, checks the bound); 1.03 ulp measured */
    {"attn",          "out",       2 * ULP,   5e-5},
};

static const tolerance *gpu_tolerance_of(const char *op, const char *slot)
{
    for (size_t i = 0; i < ATH_ARRAY_LEN(gpu_tolerances); i++)
        if (strcmp(gpu_tolerances[i].op, op) == 0 && strcmp(gpu_tolerances[i].slot, slot) == 0)
            return &gpu_tolerances[i];
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

static stat stats[64], gpu_stats[64];
static int  n_stats, n_gpu_stats;

static stat *stat_in(stat *table, int *n, const char *op, const char *slot, char phase)
{
    for (int i = 0; i < *n; i++)
        if (strcmp(table[i].op, op) == 0 && strcmp(table[i].slot, slot) == 0 && table[i].phase == phase)
            return &table[i];
    if (*n == 64)
        abort();
    stat *s = &table[(*n)++];
    memset(s, 0, sizeof *s);
    snprintf(s->op, sizeof s->op, "%s", op);
    snprintf(s->slot, sizeof s->slot, "%s", slot);
    s->phase = phase;
    return s;
}

static stat *stat_of(const char *op, const char *slot, char phase) { return stat_in(stats, &n_stats, op, slot, phase); }

/* ---- one record ---------------------------------------------------------------- */

/* The device for the GPU replay, and its memory (reset after each record). */
typedef struct {
    ath_device    d;
    ath_dev_arena mem;
} gpu;

typedef struct {
    ath_test             *t;
    gpu                  *gpu;        /* NULL: no GPU replay */
    ath_arena            *a;          /* scratch, reset after each record */
    const ath_safetensors *g;
    const ath_model      *m;
    ath_pool             *pool;       /* for ath_matmul_fast */
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

/* score and bf16 ulp distance of got against want (same shape). */
static double score_of(const ath_tensor *want, const ath_tensor *got, bool ulps, int *ulp, int64_t *worst)
{
    const int64_t n = ath_tensor_numel(want);
    double ss = 0;
    for (int64_t i = 0; i < n; i++) {
        double w = ath_load_f32(want->data, want->dtype, i);
        ss += w * w;
    }
    const double rms = n ? sqrt(ss / (double)n) : 0;
    double score = 0;
    *ulp = 0;
    *worst = 0;
    for (int64_t i = 0; i < n; i++) {
        double w = ath_load_f32(want->data, want->dtype, i), g = ath_load_f32(got->data, got->dtype, i);
        double e = fabs(g - w);
        double s = e == 0 ? 0 : e / (fabs(w) + rms);
        if (!(s <= score)) {                     /* also catches NaN */
            score = isnan(s) ? INFINITY : s;
            *worst = i;
        }
        if (ulps) {
            int u = ulp_distance((float)g, (float)w);
            if (u > *ulp)
                *ulp = u;
        }
    }
    return score;
}

static void record(stat *s, bool exact, double score, int ulp, const char *id, int64_t worst)
{
    if (exact) {
        s->n_exact++;
        if (score > s->exact || !s->worst_exact[0]) {
            s->exact = score;
            snprintf(s->worst_exact, sizeof s->worst_exact, "%s[%lld]", id, (long long)worst);
        }
    } else {
        s->n_native++;
        if (ulp > s->ulp)
            s->ulp = ulp;
        if (score > s->native || !s->worst_native[0]) {
            s->native = score;
            snprintf(s->worst_native, sizeof s->worst_native, "%s[%lld]", id, (long long)worst);
        }
    }
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
    int ulp;
    int64_t worst;
    double score = score_of(want, got, !c->exact && want->dtype == ATH_DTYPE_BF16, &ulp, &worst);
    record(stat_of(c->op, name, c->id[0]), c->exact, score, ulp, c->id, worst);
}

/* ---- the GPU replay ---------------------------------------------------------------- */

/* A device copy of a host tensor (NULL stays NULL), queued on the compute
 * stream like the kernels. */
static ath_tensor *dev(ctx *c, const ath_tensor *h)
{
    if (!h)
        return NULL;
    ath_tensor *t = ath_arena_alloc(c->a, sizeof *t, _Alignof(ath_tensor));
    if (!t)
        abort();
    *t = *h;
    t->tier = ATH_TIER_VRAM;
    t->data = ath_dev_arena_alloc(&c->gpu->mem, ath_tensor_nbytes(h) + 1, 0);
    if (!t->data || ath_copy(&c->gpu->d, t->data, h->data, ath_tensor_nbytes(h), ATH_STREAM_COMPUTE) != ATH_OK) {
        fail(c, "cannot copy %zu bytes to the GPU", ath_tensor_nbytes(h));
        return NULL;
    }
    return t;
}

/* A device tensor shaped like h for a kernel's output, every byte 0xff (a
 * NaN in f32 and bf16), so that an element the kernel does not write fails. */
static ath_tensor *dev_out(ctx *c, const ath_tensor *h)
{
    ath_tensor *t = ath_arena_alloc(c->a, sizeof *t, _Alignof(ath_tensor));
    if (!t)
        abort();
    *t = *h;
    t->tier = ATH_TIER_VRAM;
    t->data = ath_dev_arena_alloc(&c->gpu->mem, ath_tensor_nbytes(h) + 1, 0);
    if (!t->data || ath_memset(&c->gpu->d, t->data, 0xff, ath_tensor_nbytes(h), ATH_STREAM_COMPUTE) != ATH_OK) {
        fail(c, "cannot allocate %zu bytes on the GPU", ath_tensor_nbytes(h));
        return NULL;
    }
    return t;
}

/* The GPU output `got` (on the device) against the reference op's output
 * `want` on the same inputs. */
static void gpu_compare(ctx *c, const char *name, const ath_tensor *got, const ath_tensor *want)
{
    if (!got || !want || c->bad)
        return;
    ath_tensor *h = new_tensor(c, got->dtype, got);
    if (ath_copy(&c->gpu->d, h->data, got->data, ath_tensor_nbytes(got), ATH_STREAM_COMPUTE) != ATH_OK ||
        ath_stream_sync(&c->gpu->d, ATH_STREAM_COMPUTE) != ATH_OK) {
        fail(c, "GPU '%s': cannot read the result", name);
        return;
    }
    int ulp;
    int64_t worst;
    double score = score_of(want, h, !c->exact && want->dtype == ATH_DTYPE_BF16, &ulp, &worst);
    record(stat_in(gpu_stats, &n_gpu_stats, c->op, name, c->id[0]), c->exact, score, ulp, c->id, worst);
}

#define GPU_RUN(expr)                                                            \
    do {                                                                         \
        ath_status s_ = (expr);                                                  \
        if (s_ != ATH_OK) {                                                      \
            fail(c, "GPU: %s: %s", #expr, ath_status_str(s_));                   \
            return;                                                              \
        }                                                                        \
    } while (0)

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
    if (c->gpu) {
        ath_tensor *yg = dev_out(c, y);
        NEED(yg);
        GPU_RUN(ath_embed_gpu(&c->gpu->d, dev(c, w), dev(c, ids), yg));
        gpu_compare(c, "out", yg, y);
    }
}

static void run_matmul(ctx *c)
{
    const ath_tensor *x = in(c, "x"), *w = weight(c, "weight");
    ath_tensor *y = out(c, "out");
    NEED(x, w, y);
    RUN(ath_matmul_ref(x, w, y));
    compare(c, "out", y);
    /* the fast path, which the forward pass uses: the same bits */
    ath_tensor *yf = new_tensor(c, y->dtype, y);
    RUN(ath_matmul_fast(x, w, yf, c->pool));
    if (memcmp(y->data, yf->data, ath_tensor_nbytes(y)) != 0)
        fail(c, "ath_matmul_fast differs from ath_matmul_ref");
    if (c->gpu) {
        ath_tensor *yg = dev_out(c, y);
        NEED(yg);
        GPU_RUN(ath_matmul_gpu(&c->gpu->d, dev(c, x), dev(c, w), yg));
        gpu_compare(c, "out", yg, y);
    }
}

static void run_rmsnorm(ctx *c)
{
    const ath_tensor *x = in(c, "x"), *w = weight(c, "weight");
    ath_tensor *y = out(c, "out");
    float offset = (float)num(c, "offset"), eps = (float)num(c, "eps");
    NEED(x, w, y);
    RUN(ath_rmsnorm_ref(x, w, offset, eps, y));
    compare(c, "out", y);
    if (c->gpu) {
        ath_tensor *yg = dev_out(c, y);
        NEED(yg);
        GPU_RUN(ath_rmsnorm_gpu(&c->gpu->d, dev(c, x), dev(c, w), offset, eps, yg));
        gpu_compare(c, "out", yg, y);
    }
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
    if (c->gpu) {
        ath_tensor *yg = dev_out(c, y);
        NEED(yg);
        GPU_RUN(ath_rmsnorm_gated_gpu(&c->gpu->d, dev(c, x), dev(c, z), dev(c, w), eps, yg));
        gpu_compare(c, "out", yg, y);
    }
}

static void run_l2norm(ctx *c)
{
    const ath_tensor *x = in(c, "x");
    ath_tensor *y = out(c, "out");
    float eps = (float)num(c, "eps");
    NEED(x, y);
    RUN(ath_l2norm_ref(x, eps, y));
    compare(c, "out", y);
    if (c->gpu) {
        ath_tensor *yg = dev_out(c, y);
        NEED(yg);
        GPU_RUN(ath_l2norm_gpu(&c->gpu->d, dev(c, x), eps, yg));
        gpu_compare(c, "out", yg, y);
    }
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
    if (c->gpu) {
        ath_tensor *yg = dev_out(c, y);
        NEED(yg);
        GPU_RUN(ath_rope_gpu(&c->gpu->d, dev(c, x), dev(c, pos), rope_dim, theta, yg));
        gpu_compare(c, "out", yg, y);
    }
}

static void run_attn(ctx *c)
{
    const ath_tensor *q = in(c, "q"), *k = in(c, "k"), *v = in(c, "v");
    ath_tensor *y = out(c, "out");
    float scale = (float)num(c, "scale");
    NEED(q, k, v, y);
    RUN(ath_attn_ref(q, k, v, NULL, scale, y));
    compare(c, "out", y);
    if (c->gpu) {
        ath_tensor *yg = dev_out(c, y);
        NEED(yg);
        /* with scratch: split-K for the decode record (step 9), rows for the prompt */
        GPU_RUN(ath_attn_gpu(&c->gpu->d, dev(c, q), dev(c, k), dev(c, v), NULL, scale, &c->gpu->mem, yg));
        gpu_compare(c, "out", yg, y);
    }
}

static void run_swiglu(ctx *c)
{
    const ath_tensor *g = in(c, "gate"), *u = in(c, "up");
    ath_tensor *y = out(c, "out");
    NEED(g, u, y);
    RUN(ath_swiglu_ref(g, u, y));
    compare(c, "out", y);
    if (c->gpu) {
        ath_tensor *yg = dev_out(c, y);
        NEED(yg);
        GPU_RUN(ath_swiglu_gpu(&c->gpu->d, dev(c, g), dev(c, u), yg));
        gpu_compare(c, "out", yg, y);
    }
}

static void run_sigmoid_mul(ctx *c)
{
    const ath_tensor *x = in(c, "x"), *g = in(c, "gate");
    ath_tensor *y = out(c, "out");
    NEED(x, g, y);
    RUN(ath_sigmoid_mul_ref(x, g, y));
    compare(c, "out", y);
    if (c->gpu) {
        ath_tensor *yg = dev_out(c, y);
        NEED(yg);
        GPU_RUN(ath_sigmoid_mul_gpu(&c->gpu->d, dev(c, x), dev(c, g), yg));
        gpu_compare(c, "out", yg, y);
    }
}

static void run_add(ctx *c)
{
    const ath_tensor *a = in(c, "a"), *b = in(c, "b");
    ath_tensor *y = out(c, "out");
    NEED(a, b, y);
    RUN(ath_add_ref(a, b, y));
    compare(c, "out", y);
    if (c->gpu) {
        ath_tensor *yg = dev_out(c, y);
        NEED(yg);
        GPU_RUN(ath_add_gpu(&c->gpu->d, dev(c, a), dev(c, b), yg));
        gpu_compare(c, "out", yg, y);
    }
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
    ath_tensor *sg = c->gpu ? dev(c, state) : NULL;                                 /* before the update */
    RUN(ath_conv_ref(x, w, state, ATH_ACT_SILU, y));
    compare(c, "out", y);
    compare(c, "state_out", state);
    if (c->gpu) {
        ath_tensor *yg = dev_out(c, y);
        NEED(sg, yg);
        GPU_RUN(ath_conv_gpu(&c->gpu->d, dev(c, x), dev(c, w), sg, ATH_ACT_SILU, yg));
        gpu_compare(c, "out", yg, y);
        gpu_compare(c, "state_out", sg, state);
    }
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
    if (c->gpu) {
        ath_tensor *gg = dev_out(c, g), *bg = dev_out(c, beta);
        NEED(gg, bg);
        GPU_RUN(ath_gdn_gates_gpu(&c->gpu->d, dev(c, a), dev(c, b), dev(c, a_log), dev(c, dt_bias), gg, bg));
        gpu_compare(c, "g", gg, g);
        gpu_compare(c, "beta", bg, beta);
    }
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
    ath_tensor *sg = c->gpu ? dev(c, state) : NULL;                                 /* before the update */
    RUN(ath_l2norm_ref(q, eps, qn));
    RUN(ath_l2norm_ref(k, eps, kn));
    RUN(ath_gdn_ref(qn, kn, v, g, beta, q_scale, state, y));
    compare(c, "out", y);
    compare(c, "state_out", state);
    if (c->gpu) {
        /* the same composition on the GPU: its l2norm, then its recurrence */
        ath_tensor *qg = dev_out(c, qn), *kg = dev_out(c, kn), *yg = dev_out(c, y);
        NEED(sg, qg, kg, yg);
        GPU_RUN(ath_l2norm_gpu(&c->gpu->d, dev(c, q), eps, qg));
        GPU_RUN(ath_l2norm_gpu(&c->gpu->d, dev(c, k), eps, kg));
        GPU_RUN(ath_gdn_gpu(&c->gpu->d, qg, kg, dev(c, v), dev(c, g), dev(c, beta), q_scale, sg, &c->gpu->mem, yg));
        gpu_compare(c, "out", yg, y);
        gpu_compare(c, "state_out", sg, state);
    }
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

static void print_stats(const stat *table, int n, const tolerance *(*tol_of)(const char *, const char *))
{
    printf("  %-14s %-10s %-5s %5s  %-9s %-4s %-8s  %5s  %-9s %-8s\n", "op", "output", "phase", "n",
           "native", "ulp", "tol", "n", "exact", "tol");
    for (int i = 0; i < n; i++) {
        const stat *s = &table[i];
        const tolerance *tol = tol_of(s->op, s->slot);
        printf("  %-14s %-10s %-5s %5d  %-9.2e %-4d %-8.1e  %5d  %-9.2e %-8.1e\n", s->op, s->slot,
               s->phase == 'p' ? "pre" : "dec", s->n_native, s->native, s->ulp, tol ? tol->native : -1,
               s->n_exact, s->exact, tol ? tol->exact : -1);
    }
    if (getenv("ATH_GOLDEN_WORST"))
        for (int i = 0; i < n; i++)
            printf("  worst %-14s %-10s native %s, exact %s\n", table[i].op, table[i].slot,
                   table[i].worst_native, table[i].worst_exact);
}

/* Every op and output compared, within its tolerance. */
static void check_stats(ath_test *t_, const char *who, const stat *table, int n,
                        const tolerance *(*tol_of)(const char *, const char *))
{
    for (int i = 0; i < n; i++) {
        const stat *st = &table[i];
        const tolerance *tol = tol_of(st->op, st->slot);
        char msg[320];
        if (!tol) {
            snprintf(msg, sizeof msg, "%s %.23s %.15s: no tolerance", who, st->op, st->slot);
            ath_test_fail_(t_, __FILE__, __LINE__, msg);
            continue;
        }
        if (!(st->native <= tol->native)) {
            snprintf(msg, sizeof msg, "%s %.23s %.15s native %.3g > %.3g at %.95s", who, st->op, st->slot,
                     st->native, tol->native, st->worst_native);
            ath_test_fail_(t_, __FILE__, __LINE__, msg);
        }
        if (!(st->exact <= tol->exact)) {
            snprintf(msg, sizeof msg, "%s %.23s %.15s exact %.3g > %.3g at %.95s", who, st->op, st->slot,
                     st->exact, tol->exact, st->worst_exact);
            ath_test_fail_(t_, __FILE__, __LINE__, msg);
        }
    }
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

    ath_pool pool = {.n_threads = 0};
    if (s == ATH_OK) {
        s = ath_pool_init(&pool, ath_pool_default_threads(), &a);
        ATH_EXPECT_OK(s);
    }
    /* the GPU replay, when this build and this machine have a GPU */
    gpu dev_ctx, *gp = NULL;
    if (s == ATH_OK && ath_gpu_count() > 0) {
        s = ath_device_open(&dev_ctx.d, 0);
        ATH_EXPECT_OK(s);
        if (s == ATH_OK) {
            ath_dev_arena_init(&dev_ctx.mem, &dev_ctx.d, &a, 0);
            gp = &dev_ctx;
        }
    }
    int replayed = 0;
    for (size_t i = 0; s == ATH_OK && i < n; i++) {
        ctx c = {.t = t_, .gpu = gp, .a = &scratch, .g = &g, .m = &m, .pool = &pool, .rec = ath_json_at(records, i)};
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
            ath_dev_arena_pos dp = gp ? ath_dev_arena_mark(&gp->mem) : (ath_dev_arena_pos){0};
            c.exact = mode == 1;
            handlers[h].run(&c);
            if (gp) {   /* the replay waited for its results; free its device memory */
                ATH_EXPECT_OK(ath_stream_sync(&gp->d, ATH_STREAM_COMPUTE));
                ath_dev_arena_reset(&gp->mem, dp);
            }
            ath_arena_reset(&scratch, p);
        }
        replayed += !c.bad;
    }
    ath_pool_free(&pool);
    printf("  %d of %zu records replayed (%s)\n", replayed, n, golden_dir());
    print_stats(stats, n_stats, tolerance_of);
    check_stats(t_, "ref", stats, n_stats, tolerance_of);
    for (size_t h = 0; s == ATH_OK && h < ATH_ARRAY_LEN(handlers); h++) {
        bool seen = false;
        for (int i = 0; i < n_stats; i++)
            seen = seen || strcmp(stats[i].op, handlers[h].op) == 0;
        ATH_EXPECT(seen);
    }
    if (gp) {
        printf("  GPU kernels against the _ref functions, same inputs (%s):\n", gp->d.name);
        print_stats(gpu_stats, n_gpu_stats, gpu_tolerance_of);
        check_stats(t_, "gpu", gpu_stats, n_gpu_stats, gpu_tolerance_of);
        ATH_EXPECT_EQ_INT(n_gpu_stats, n_stats);   /* every op and output replayed on the GPU */
        ath_dev_arena_free(&gp->mem);
        ath_device_close(&gp->d);
    } else {
        printf("  GPU replay: skipped (%s)\n", strcmp(ath_gpu_backend(), "none") == 0 ? "built without a GPU backend"
                                                                                        : "no usable GPU device");
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
