/* moe.c: the mixture-of-experts ops (step 12). See ops.h and
 * docs/contract/qwen35-35b-a3b.md, MoE op contract.
 *
 * The reference (Qwen3_5MoeTopKRouter, Qwen3_5MoeExperts,
 * Qwen3_5MoeSparseMoeBlock) computes softmax over all experts in fp32, takes
 * the top k, renormalizes, then for every expert that some token chose runs
 * its MLP on those tokens and adds the result, scaled by the weight, into a
 * zeroed output (index_add_, in order of expert index), and finally adds the
 * gated shared expert. Here the routed experts are added in the same order
 * of expert index, in fp32 with fused multiply-adds, and the result is
 * rounded once.
 *
 * An expert's MLP follows the canonical order of src/ops/q8_0.h for Q8_0
 * weights, with SiLU through the canonical exponential: the reference here,
 * the CPU fast path and the GPU kernel give the same bits, so where an
 * expert sits never changes a result (AGENTS.md invariant 7, option (b)).
 */
#include "ops/check.h"
#include "ops/launch.h"
#include "ops/ops.h"
#include "ops/q8_0.h"

#include <math.h>

/* ---- router ---------------------------------------------------------------------- */

static ath_status route_args(const ath_tensor *logits, const ath_tensor *ids, const ath_tensor *w, ops_where where)
{
    static const char op[] = "moe_route";
    ATH_CHECK(ops_tensor(logits, ATH_DTYPE_F32, op, "logits"));
    ATH_CHECK(ops_tensor(ids, ATH_DTYPE_I32, op, "ids"));
    ATH_CHECK(ops_float(w, op, "w"));
    if (logits->ndim != 2 || ids->ndim != 2 || !ath_tensor_same_shape(ids, w) || ids->shape[0] != logits->shape[0] ||
        ids->shape[1] < 1 || ids->shape[1] > ATH_MOE_MAX_K || ids->shape[1] > logits->shape[1])
        return ops_fail(ATH_ERR_INVALID_ARG, op, "shapes must be logits [T, n], ids and w [T, k], 1 <= k <= min(n, %d)",
                        ATH_MOE_MAX_K);
    return OPS_PLACE(op, where, logits, ids, w);
}

ath_status ath_moe_route_ref(const ath_tensor *logits, ath_tensor *ids, ath_tensor *w)
{
    ATH_CHECK(route_args(logits, ids, w, OPS_HOST));
    const int64_t T = logits->shape[0], n = logits->shape[1], k = ids->shape[1];
    const float *L = logits->data;
    int32_t *id = ids->data;
    for (int64_t t = 0; t < T; t++) {
        const float *l = L + t * n;
        float mx = -INFINITY;
        for (int64_t e = 0; e < n; e++)
            mx = l[e] > mx ? l[e] : mx;
        float sum = 0.0f;
        for (int64_t e = 0; e < n; e++)
            sum += expf(l[e] - mx);
        /* the k largest probabilities, the lower index first on ties: a
         * selection by repeated scans, k is small */
        int32_t pick[ATH_MOE_MAX_K];
        float p[ATH_MOE_MAX_K];
        for (int64_t j = 0; j < k; j++) {
            int32_t best = -1;
            float bp = 0.0f;
            for (int64_t e = 0; e < n; e++) {
                bool taken = false;
                for (int64_t i = 0; i < j; i++)
                    taken |= pick[i] == e;
                const float pe = expf(l[e] - mx) / sum;
                if (!taken && (best < 0 || pe > bp)) {
                    best = (int32_t)e;
                    bp = pe;
                }
            }
            pick[j] = best;
            p[j] = bp;
        }
        float ps = 0.0f;                         /* the sum in order of rank, as torch's sum over top-k */
        for (int64_t j = 0; j < k; j++)
            ps += p[j];
        /* sorted by expert index, weights renormalized */
        for (int64_t j = 0; j < k; j++) {
            int64_t r = 0;
            for (int64_t i = 0; i < k; i++)
                r += pick[i] < pick[j];
            id[t * k + r] = pick[j];
            st(w, t * k + r, p[j] / ps);
        }
    }
    return ATH_OK;
}

ath_status ath_moe_route_gpu(const ath_device *d, const ath_tensor *logits, ath_tensor *ids, ath_tensor *w)
{
    ATH_CHECK(route_args(logits, ids, w, OPS_GPU));
    return ath_moe_route_launch(d, logits, ids, w);
}

/* ---- one expert ------------------------------------------------------------------- */

ath_status ath_moe_expert_ref(const ath_tensor *x, const ath_tensor *gate, const ath_tensor *up,
                              const ath_tensor *down, ath_tensor *y)
{
    static const char op[] = "moe_expert";
    ATH_CHECK(ops_float(x, op, "x"));
    ATH_CHECK(ops_float(y, op, "y"));
    if (x->ndim != 2 || !ath_tensor_same_shape(x, y) || gate->ndim != 2 || !ath_tensor_same_shape(gate, up) ||
        down->ndim != 2 || gate->shape[1] != x->shape[1] || down->shape[0] != x->shape[1] ||
        down->shape[1] != gate->shape[0])
        return ops_fail(ATH_ERR_INVALID_ARG, op, "shapes must be x, y [T, H], gate, up [I, H], down [H, I]");
    ATH_CHECK(OPS_PLACE(op, OPS_HOST, x, gate, up, down, y));
    const int64_t T = x->shape[0], I = gate->shape[0];
    /* g and u, rounded to y's dtype as matmul outputs, then h, then down;
     * one token at a time, so that the working memory is two rows of I */
    for (int64_t t = 0; t < T; t++) {
        float g[I > 0 ? I : 1], u[I > 0 ? I : 1];
        ath_tensor xr, gt, ut, yr;
        ATH_CHECK(ath_tensor_slice(x, 0, t, 1, &xr));
        ATH_CHECK(ath_tensor_slice(y, 0, t, 1, &yr));
        ATH_CHECK(ath_tensor_init(&gt, ATH_DTYPE_F32, 2, (const int64_t[]){1, I}, g, ATH_TIER_RAM));
        ATH_CHECK(ath_tensor_init(&ut, ATH_DTYPE_F32, 2, (const int64_t[]){1, I}, u, ATH_TIER_RAM));
        ATH_CHECK(ath_matmul_ref(&xr, gate, &gt));
        ATH_CHECK(ath_matmul_ref(&xr, up, &ut));
        for (int64_t i = 0; i < I; i++)
            g[i] = ath_round_to(y->dtype,
                                ath_q8_silu(ath_round_to(y->dtype, g[i])) * ath_round_to(y->dtype, u[i]));
        /* h (in g) as y's dtype, the input of the down projection */
        ath_tensor ht;
        uint16_t hb[y->dtype == ATH_DTYPE_BF16 && I > 0 ? I : 1];
        if (y->dtype == ATH_DTYPE_BF16) {
            for (int64_t i = 0; i < I; i++)
                hb[i] = ath_f32_to_bf16(g[i]);
            ATH_CHECK(ath_tensor_init(&ht, ATH_DTYPE_BF16, 2, (const int64_t[]){1, I}, hb, ATH_TIER_RAM));
        } else {
            ATH_CHECK(ath_tensor_init(&ht, ATH_DTYPE_F32, 2, (const int64_t[]){1, I}, g, ATH_TIER_RAM));
        }
        ATH_CHECK(ath_matmul_ref(&ht, down, &yr));
    }
    return ATH_OK;
}

/* ---- combination ------------------------------------------------------------------- */

static ath_status combine_args(const ath_tensor *y, const ath_tensor *w, const ath_tensor *ys, const ath_tensor *sg,
                               const ath_tensor *out, ops_where where)
{
    static const char op[] = "moe_combine";
    ATH_CHECK(ops_float(y, op, "y"));
    ATH_CHECK(ops_float(w, op, "w"));
    ATH_CHECK(ops_float(out, op, "out"));
    if ((ys == NULL) != (sg == NULL))
        return ops_fail(ATH_ERR_INVALID_ARG, op, "the shared expert's output and gate come together");
    if (y->ndim != 3 || w->ndim != 2 || out->ndim != 2 || w->shape[0] != y->shape[0] || w->shape[1] != y->shape[1] ||
        out->shape[0] != y->shape[0] || out->shape[1] != y->shape[2])
        return ops_fail(ATH_ERR_INVALID_ARG, op, "shapes must be y [T, k, H], w [T, k], out [T, H]");
    if (ys) {
        ATH_CHECK(ops_float(ys, op, "ys"));
        ATH_CHECK(ops_float(sg, op, "sg"));
        if (!ath_tensor_same_shape(ys, out) || sg->ndim != 2 || sg->shape[0] != out->shape[0] || sg->shape[1] != 1)
            return ops_fail(ATH_ERR_INVALID_ARG, op, "shapes must be ys [T, H], sg [T, 1]");
        return OPS_PLACE(op, where, y, w, ys, sg, out);
    }
    return OPS_PLACE(op, where, y, w, out);
}

ath_status ath_moe_combine_ref(const ath_tensor *y, const ath_tensor *w, const ath_tensor *ys, const ath_tensor *sg,
                               ath_tensor *out)
{
    ATH_CHECK(combine_args(y, w, ys, sg, out, OPS_HOST));
    const int64_t T = y->shape[0], k = y->shape[1], H = y->shape[2];
    for (int64_t t = 0; t < T; t++) {
        const float gate = ys ? 1.0f / (1.0f + expf(-ld(sg, t))) : 0.0f;
        for (int64_t h = 0; h < H; h++) {
            float acc = 0.0f;
            for (int64_t j = 0; j < k; j++)
                acc = fmaf(ld(w, t * k + j), ld(y, (t * k + j) * H + h), acc);
            if (ys)
                acc = acc + ld(ys, t * H + h) * gate;
            st(out, t * H + h, acc);
        }
    }
    return ATH_OK;
}

ath_status ath_moe_combine_gpu(const ath_device *d, const ath_tensor *y, const ath_tensor *w, const ath_tensor *ys,
                               const ath_tensor *sg, ath_tensor *out)
{
    ATH_CHECK(combine_args(y, w, ys, sg, out, OPS_GPU));
    return ath_moe_combine_launch(d, y, w, ys, sg, out);
}

/* ---- the routed experts on the device ----------------------------------------------- */

ath_status ath_moe_experts_gpu(const ath_device *d, const ath_tensor *x, const ath_tensor *ids,
                               const void *const *tab, ath_dtype wdt, int64_t n_experts, int64_t inter, ath_tensor *h,
                               ath_tensor *y)
{
    static const char op[] = "moe_experts";
    ATH_CHECK(ops_float(x, op, "x"));
    ATH_CHECK(ops_tensor(ids, ATH_DTYPE_I32, op, "ids"));
    ATH_CHECK(ops_tensor(h, x->dtype, op, "h"));
    ATH_CHECK(ops_tensor(y, x->dtype, op, "y"));
    if (!tab || n_experts < 1 || inter < 1 ||
        (wdt != ATH_DTYPE_Q8_0 && wdt != ATH_DTYPE_Q8_0_R16 && wdt != ATH_DTYPE_BF16 && wdt != ATH_DTYPE_F32))
        return ops_fail(ATH_ERR_INVALID_ARG, op, "a table of experts, Q8_0, Q8_0_R16, bf16 or f32");
    if (x->ndim != 2 || ids->ndim != 2 || ids->shape[0] != x->shape[0] || y->ndim != 3 ||
        y->shape[0] != x->shape[0] || y->shape[1] != ids->shape[1] || y->shape[2] != x->shape[1] || h->ndim != 3 ||
        h->shape[0] != x->shape[0] || h->shape[1] != ids->shape[1] || h->shape[2] != inter)
        return ops_fail(ATH_ERR_INVALID_ARG, op, "shapes must be x [T, H], ids [T, k], h [T, k, I], y [T, k, H]");
    const int64_t blk = ath_dtype_block(wdt);
    if (blk > 1 && (x->shape[1] % blk || inter % blk))
        return ops_fail(ATH_ERR_INVALID_ARG, op, "%s experts need H and I multiples of %lld", ath_dtype_name(wdt),
                        (long long)blk);
    ATH_CHECK(OPS_PLACE(op, OPS_GPU, x, ids, h, y));
    return ath_moe_experts_launch(d, x, ids, tab, wdt, n_experts, inter, h, y);
}

/* ---- experts split between the GPU and the CPU --------------------------------------- */

#define JOB_RING 4096   /* jobs of passes that are not captured, reused in turn */

typedef struct {
    ath_moe_ws            *ws;
    const ath_moe_experts *ex;
    int64_t                T;
    ath_dtype              dt;
} cpu_job;

struct ath_moe_ws {
    const ath_device *d;
    ath_pool         *pool;
    int64_t           max_tokens, k, H, I;
    void             *x, *ids, *y;     /* pinned: [max_tokens, H], [max_tokens, k], [max_tokens, k, H] */
    ath_event         ready, done;
    ath_arena         cpu;             /* the CPU's working memory                                    */
    ath_arena         kept;            /* jobs of captured passes: they live as long as ws            */
    cpu_job           ring[JOB_RING];
    int64_t           next;
};

ath_status ath_moe_ws_create(ath_moe_ws **out, const ath_device *d, ath_pool *pool, int64_t max_tokens, int64_t k,
                             int64_t hidden, int64_t inter, ath_arena *a)
{
    if (!out || !d || max_tokens < 1 || k < 1 || hidden < 1 || inter < 1)
        return ATH_ERR_INVALID_ARG;
    ath_moe_ws *ws = ath_arena_zalloc(a, sizeof *ws, _Alignof(ath_moe_ws));
    if (!ws)
        return ATH_ERR_NOMEM;
    *ws = (ath_moe_ws){.d = d, .pool = pool, .max_tokens = max_tokens, .k = k, .H = hidden, .I = inter};
    ath_arena_init(&ws->cpu, 0);
    ath_arena_init(&ws->kept, 0);
    ath_status s = ath_pinned_alloc(d, (size_t)(max_tokens * hidden) * sizeof(float), &ws->x);
    if (s == ATH_OK)
        s = ath_pinned_alloc(d, (size_t)(max_tokens * k) * sizeof(int32_t), &ws->ids);
    if (s == ATH_OK)
        s = ath_pinned_alloc(d, (size_t)(max_tokens * k * hidden) * sizeof(float), &ws->y);
    if (s == ATH_OK)
        s = ath_event_create(d, &ws->ready, false);
    if (s == ATH_OK)
        s = ath_event_create(d, &ws->done, false);
    if (s != ATH_OK) {
        ath_moe_ws_free(ws);
        return s;
    }
    *out = ws;
    return ATH_OK;
}

void ath_moe_ws_free(ath_moe_ws *ws)
{
    if (!ws)
        return;
    if (ws->ready.handle || ws->done.handle)
        ath_stream_sync(ws->d, ATH_STREAM_D2H);    /* no CPU job may still be running */
    ath_event_destroy(ws->d, &ws->ready);
    ath_event_destroy(ws->d, &ws->done);
    ath_pinned_free(ws->d, ws->x);
    ath_pinned_free(ws->d, ws->ids);
    ath_pinned_free(ws->d, ws->y);
    ath_arena_free(&ws->cpu);
    ath_arena_free(&ws->kept);
    ws->x = ws->ids = ws->y = NULL;
}

/* The CPU's share, on a thread of the runtime (ath_host_call). It cannot
 * return a status: an error is logged, and the outputs it did not write
 * keep what was there. */
static void cpu_run(void *ctx)
{
    const cpu_job *j = ctx;
    ath_moe_ws *ws = j->ws;
    ath_tensor x, ids, y;
    ath_status s = ath_tensor_init(&x, j->dt, 2, (const int64_t[]){j->T, ws->H}, ws->x, ATH_TIER_PINNED);
    if (s == ATH_OK)
        s = ath_tensor_init(&ids, ATH_DTYPE_I32, 2, (const int64_t[]){j->T, ws->k}, ws->ids, ATH_TIER_PINNED);
    if (s == ATH_OK)
        s = ath_tensor_init(&y, j->dt, 3, (const int64_t[]){j->T, ws->k, ws->H}, ws->y, ATH_TIER_PINNED);
    if (s == ATH_OK)
        s = ath_moe_experts_cpu(&x, &ids, j->ex, &y, ws->pool, &ws->cpu);
    if (s != ATH_OK)
        ath_log(ATH_LOG_ERROR, "moe: the CPU's experts failed: %s", ath_status_str(s));
}

/* A job that lasts as long as the work that runs it: kept for ever when a
 * graph records it, otherwise a slot of the ring (after a full turn of the
 * ring, the side stream is drained so that no slot is still in use). */
static ath_status new_job(ath_moe_ws *ws, cpu_job **out)
{
    bool capturing = false;
    ATH_CHECK(ath_stream_capturing(ws->d, ATH_STREAM_COMPUTE, &capturing));
    if (capturing) {
        *out = ath_arena_alloc(&ws->kept, sizeof **out, _Alignof(cpu_job));
        return *out ? ATH_OK : ATH_ERR_NOMEM;
    }
    if (ws->next == JOB_RING) {
        ATH_CHECK(ath_stream_sync(ws->d, ATH_STREAM_D2H));
        ws->next = 0;
    }
    *out = &ws->ring[ws->next++];
    return ATH_OK;
}

ath_status ath_moe_experts_split(const ath_device *d, const ath_tensor *x, const ath_tensor *ids,
                                 const ath_moe_experts *ex, ath_moe_ws *ws, ath_tensor *h, ath_tensor *y)
{
    static const char op[] = "moe_experts_split";
    if (!ex || !ws || ws->d != d)
        return ops_fail(ATH_ERR_INVALID_ARG, op, "an expert table and the device's working state");
    const int64_t T = x->shape[0], H = x->shape[1], k = ids->ndim == 2 ? ids->shape[1] : 0;
    if (ex->host_tab && (k != ws->k || H != ws->H || ex->inter != ws->I))
        return ops_fail(ATH_ERR_INVALID_ARG, op, "the working state was made for other shapes");
    const size_t xes = ath_dtype_size(x->dtype);
    for (int64_t t0 = 0; t0 < T; t0 += ws->max_tokens) {
        const int64_t n = T - t0 < ws->max_tokens ? T - t0 : ws->max_tokens;
        ath_tensor xc, ic, hc, yc;
        ATH_CHECK(ath_tensor_slice(x, 0, t0, n, &xc));
        ATH_CHECK(ath_tensor_slice(ids, 0, t0, n, &ic));
        ATH_CHECK(ath_tensor_slice(h, 0, t0, n, &hc));
        ATH_CHECK(ath_tensor_slice(y, 0, t0, n, &yc));
        if (ex->host_tab) {
            /* the side stream: inputs to pinned memory, then the CPU */
            cpu_job *j;
            ATH_CHECK(new_job(ws, &j));
            *j = (cpu_job){.ws = ws, .ex = ex, .T = n, .dt = x->dtype};
            ATH_CHECK(ath_event_record(d, &ws->ready, ATH_STREAM_COMPUTE));
            ATH_CHECK(ath_stream_wait(d, ATH_STREAM_D2H, &ws->ready));
            ATH_CHECK(ath_copy(d, ws->x, xc.data, (size_t)(n * H) * xes, ATH_STREAM_D2H));
            ATH_CHECK(ath_copy(d, ws->ids, ic.data, (size_t)(n * k) * sizeof(int32_t), ATH_STREAM_D2H));
            ATH_CHECK(ath_host_call(d, ATH_STREAM_D2H, cpu_run, j));
            ATH_CHECK(ath_event_record(d, &ws->done, ATH_STREAM_D2H));
        }
        /* the device's experts meanwhile, then the CPU's outputs merged in */
        ATH_CHECK(ath_moe_experts_gpu(d, &xc, &ic, ex->dev_tab, ex->dev_dtype, ex->n_experts, ex->inter, &hc, &yc));
        if (ex->host_tab) {
            ATH_CHECK(ath_stream_wait(d, ATH_STREAM_COMPUTE, &ws->done));
            ATH_CHECK(ath_moe_merge_launch(d, &ic, ex->dev_tab, ex->n_experts, ws->y, &yc));
        }
    }
    return ATH_OK;
}
