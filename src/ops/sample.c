/* sample.c: choosing the next token from the logits. See ops.h.
 *
 * Greedy takes the largest logit. Otherwise the candidates are ordered from
 * the most likely (largest logit, then lowest id), filtered, and the token
 * is drawn by inverse transform: u in [0, 1) is scaled by the total
 * probability kept and the first candidate whose cumulative probability
 * passes it is chosen. Probabilities are exp((l − l_max) / T), in double;
 * they need no normalization, because only ratios matter.
 *
 * Top-k keeps a sorted list of the k best while scanning the vocabulary,
 * which costs about one comparison per token when k is small (model cards
 * recommend about 20). Without top-k every token is a candidate and the
 * whole vocabulary is sorted (heapsort: glibc's qsort may allocate).
 */
#include "ops/check.h"
#include "ops/launch.h"
#include "ops/ops.h"

#include <math.h>

static const char op[] = "sample";

typedef struct {
    float   l;
    int32_t id;
} cand;

/* a comes before b: larger logit, then lower id. */
static bool before(cand a, cand b) { return a.l > b.l || (a.l == b.l && a.id < b.id); }

/* Heapsort, so that c[0] is first by `before`: a max-heap whose root is
 * the candidate that goes last, moved to the end one at a time. */
static void sift_down(cand *c, int64_t i, int64_t end)
{
    for (;;) {
        int64_t k = 2 * i + 1;
        if (k >= end)
            return;
        if (k + 1 < end && before(c[k], c[k + 1]))
            k++;
        if (!before(c[i], c[k]))
            return;
        cand t = c[i];
        c[i] = c[k];
        c[k] = t;
        i = k;
    }
}

static void heap_sort(cand *c, int64_t n)
{
    for (int64_t i = n / 2 - 1; i >= 0; i--)
        sift_down(c, i, n);
    for (int64_t end = n - 1; end > 0; end--) {
        cand t = c[0];
        c[0] = c[end];
        c[end] = t;
        sift_down(c, 0, end);
    }
}

static bool params_ok(const ath_sampler *p)
{
    return p->temperature >= 0 && p->top_k >= 0 && p->top_p > 0 && p->top_p <= 1 && p->min_p >= 0 && p->min_p <= 1;
}

static bool greedy(const ath_sampler *p) { return p->temperature == 0 || p->top_k == 1; }

ath_status ath_sample_ref(const ath_tensor *logits, const ath_sampler *p, double u, ath_arena *scratch,
                          int32_t *out)
{
    ATH_CHECK(ops_float(logits, op, "logits"));
    if (!p || !scratch || !out)
        return ops_fail(ATH_ERR_INVALID_ARG, op, "missing sampler, scratch or output");
    if (logits->ndim < 1 || logits->ndim > 2 || (logits->ndim == 2 && logits->shape[0] != 1) ||
        ops_last(logits) < 1 || ops_last(logits) > INT32_MAX)
        return ops_fail(ATH_ERR_INVALID_ARG, op, "logits must be [vocab] or [1, vocab]");
    if (!params_ok(p) || !(u >= 0 && u < 1))
        return ops_fail(ATH_ERR_INVALID_ARG, op, "temperature %g, top_k %d, top_p %g, min_p %g, u %g",
                        (double)p->temperature, p->top_k, (double)p->top_p, (double)p->min_p, u);
    ATH_CHECK(OPS_PLACE(op, OPS_HOST, logits));
    const int64_t V = ops_last(logits);

    if (greedy(p)) {
        cand best = {ld(logits, 0), 0};
        for (int64_t i = 1; i < V; i++) {
            cand c = {ld(logits, i), (int32_t)i};
            if (before(c, best))
                best = c;
        }
        *out = best.id;
        return ATH_OK;
    }

    ath_arena_pos pos = ath_arena_mark(scratch);
    int64_t k = p->top_k > 0 && p->top_k < V ? p->top_k : V;
    cand *c = ath_arena_alloc(scratch, (size_t)V * sizeof *c, _Alignof(cand));
    double *w = ath_arena_alloc(scratch, (size_t)k * sizeof *w, _Alignof(double));
    if (!c || !w) {
        ath_arena_reset(scratch, pos);
        return ATH_ERR_NOMEM;
    }
    int64_t n = 0;
    if (k < V) {
        /* c[0 .. n) sorted; a token enters if it beats the last one */
        for (int64_t i = 0; i < V; i++) {
            cand x = {ld(logits, i), (int32_t)i};
            if (n == k && !before(x, c[n - 1]))
                continue;
            int64_t j = n < k ? n++ : n - 1;
            while (j > 0 && before(x, c[j - 1])) {
                c[j] = c[j - 1];
                j--;
            }
            c[j] = x;
        }
    } else {
        for (int64_t i = 0; i < V; i++)
            c[i] = (cand){ld(logits, i), (int32_t)i};
        n = V;
        heap_sort(c, n);
    }

    /* Relative probabilities, and the top-p and min-p cuts. */
    const double T = p->temperature, top = c[0].l;
    double total = 0;
    for (int64_t i = 0; i < n; i++) {
        w[i] = exp(((double)c[i].l - top) / T);
        total += w[i];
    }
    int64_t keep = n;
    if (p->top_p < 1) {
        double cum = 0;
        for (int64_t i = 0; i < n; i++) {
            cum += w[i];
            if (cum >= p->top_p * total) {
                keep = i + 1;
                break;
            }
        }
    }
    while (keep > 1 && w[keep - 1] < p->min_p * w[0])
        keep--;

    double kept = 0;
    for (int64_t i = 0; i < keep; i++)
        kept += w[i];
    double r = u * kept, cum = 0;
    int32_t id = c[keep - 1].id;
    for (int64_t i = 0; i < keep; i++) {
        cum += w[i];
        if (cum > r) {
            id = c[i].id;
            break;
        }
    }
    *out = id;
    ath_arena_reset(scratch, pos);
    return ATH_OK;
}

bool ath_sample_gpu_fits(const ath_sampler *p)
{
    return p && params_ok(p) && (greedy(p) || (p->top_k > 0 && p->top_k <= ATH_SAMPLE_GPU_MAX_K));
}

ath_status ath_sample_gpu(const ath_device *d, const ath_tensor *logits, const ath_sampler *p, const double *u,
                          ath_dev_arena *scratch, int32_t *out)
{
    ATH_CHECK(ops_float(logits, op, "logits"));
    if (!p || !scratch || !out || !u)
        return ops_fail(ATH_ERR_INVALID_ARG, op, "missing sampler, scratch, u or output");
    if (logits->ndim < 1 || logits->ndim > 2 || (logits->ndim == 2 && logits->shape[0] != 1) ||
        ops_last(logits) < 1 || ops_last(logits) > INT32_MAX)
        return ops_fail(ATH_ERR_INVALID_ARG, op, "logits must be [vocab] or [1, vocab]");
    if (!params_ok(p))
        return ops_fail(ATH_ERR_INVALID_ARG, op, "temperature %g, top_k %d, top_p %g, min_p %g",
                        (double)p->temperature, p->top_k, (double)p->top_p, (double)p->min_p);
    if (!ath_sample_gpu_fits(p))
        return ops_fail(ATH_ERR_UNSUPPORTED, op, "top_k %d on the GPU: 1 to %d", p->top_k, ATH_SAMPLE_GPU_MAX_K);
    ATH_CHECK(OPS_PLACE(op, OPS_GPU, logits));
    const int64_t V = ops_last(logits);
    const int32_t k = greedy(p) ? 1 : (int32_t)(p->top_k < V ? p->top_k : V);
    const ath_dev_arena_pos mark = ath_dev_arena_mark(scratch);
    void *work = ath_dev_arena_alloc(scratch, ath_sample_launch_bytes(V, k), 0);
    ath_status st = work ? ath_sample_launch(d, logits, p, k, u, work, out) : ATH_ERR_NOMEM;
    ath_dev_arena_reset(scratch, mark);   /* the next work on the stream comes after the kernels */
    return st;
}
