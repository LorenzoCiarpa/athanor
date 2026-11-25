/* gdn.c: Gated DeltaNet gates and the gated delta rule, recurrent form. See ops.h.
 *
 * Each value head keeps a state S [dk, dv] that maps keys to values. Per token:
 * decay the whole state by α = exp(g); read what it predicts for the key,
 * u = Sᵀk; write the correction β (v − u) along k; read the output with the
 * query. Columns of S (one per value dimension) never interact, so the update
 * runs column by column and needs no scratch memory. This is the
 * token-by-token form (torch_recurrent_gated_delta_rule in the reference); the
 * chunked form used for prefill by the reference and, from step 10, by the GPU
 * computes the same recurrence (contract, Gated DeltaNet layer).
 *
 * The query and key are L2-normalized here (step 10), with the expression of
 * ath_l2norm_ref, so the values are the bits that op would have stored.
 */
#include "ops/check.h"
#include "ops/launch.h"
#include "ops/ops.h"

#include <math.h>

/* torch.nn.functional.softplus with beta 1 and threshold 20. */
static float softplus(float x)
{
    return x > 20.0f ? x : log1pf(expf(x));
}

static ath_status gates_args(const ath_tensor *a, const ath_tensor *b, const ath_tensor *a_log,
                             const ath_tensor *dt_bias, const ath_tensor *g, const ath_tensor *beta,
                             ops_where where)
{
    static const char op[] = "gdn_gates";
    ATH_CHECK(ops_float(a, op, "a"));
    ATH_CHECK(ops_float(b, op, "b"));
    ATH_CHECK(ops_float(a_log, op, "a_log"));
    ATH_CHECK(ops_float(dt_bias, op, "dt_bias"));
    ATH_CHECK(ops_float(g, op, "g"));
    ATH_CHECK(ops_float(beta, op, "beta"));
    if (a->ndim != 2 || !ath_tensor_same_shape(a, b) || !ath_tensor_same_shape(a, g) ||
        !ath_tensor_same_shape(a, beta) || a_log->ndim != 1 || a_log->shape[0] != a->shape[1] ||
        !ath_tensor_same_shape(a_log, dt_bias))
        return ops_fail(ATH_ERR_INVALID_ARG, op, "shapes must be a, b, g, beta [T, nv], a_log, dt_bias [nv]");
    return OPS_PLACE(op, where, a, b, a_log, dt_bias, g, beta);
}

ath_status ath_gdn_gates_ref(const ath_tensor *a, const ath_tensor *b, const ath_tensor *a_log,
                             const ath_tensor *dt_bias, ath_tensor *g, ath_tensor *beta)
{
    ATH_CHECK(gates_args(a, b, a_log, dt_bias, g, beta, OPS_HOST));
    const int64_t T = a->shape[0], nv = a->shape[1];
    for (int64_t t = 0; t < T; t++)
        for (int64_t h = 0; h < nv; h++) {
            int64_t e = t * nv + h;
            st(g, e, -expf(ld(a_log, h)) * softplus(ld(a, e) + ld(dt_bias, h)));
            st(beta, e, 1.0f / (1.0f + expf(-ld(b, e))));
        }
    return ATH_OK;
}

static const char gdn_op[] = "gdn";

static ath_status gdn_args(const ath_tensor *q, const ath_tensor *k, const ath_tensor *v, const ath_tensor *g,
                           const ath_tensor *beta, const ath_tensor *state, const ath_tensor *out,
                           ops_where where)
{
    const char *op = gdn_op;
    ATH_CHECK(ops_float_rows(q, op, "q"));
    ATH_CHECK(ops_float_rows(k, op, "k"));
    ATH_CHECK(ops_float_rows(v, op, "v"));
    ATH_CHECK(ops_float(g, op, "g"));
    ATH_CHECK(ops_float(beta, op, "beta"));
    ATH_CHECK(ops_tensor(state, ATH_DTYPE_F32, op, "state"));
    ATH_CHECK(ops_float(out, op, "out"));
    if (q->ndim != 3 || !ath_tensor_same_shape(q, k) || v->ndim != 3 || !ath_tensor_same_shape(v, out) ||
        g->ndim != 2 || !ath_tensor_same_shape(g, beta) || state->ndim != 3)
        return ops_fail(ATH_ERR_INVALID_ARG, op,
                        "shapes must be q, k [T, nk, dk], v, out [T, nv, dv], g, beta [T, nv], "
                        "state [nv, dk, dv]");
    const int64_t T = q->shape[0], nk = q->shape[1], dk = q->shape[2];
    const int64_t nv = v->shape[1], dv = v->shape[2];
    if (v->shape[0] != T || g->shape[0] != T || g->shape[1] != nv || state->shape[0] != nv ||
        state->shape[1] != dk || state->shape[2] != dv || nk <= 0 || nv % nk != 0)
        return ops_fail(ATH_ERR_INVALID_ARG, op,
                        "q, k [%lld, %lld, %lld] and v [%lld, %lld, %lld] need g, beta [T, nv], "
                        "state [nv, dk, dv], nv a multiple of nk",
                        (long long)T, (long long)nk, (long long)dk, (long long)v->shape[0],
                        (long long)nv, (long long)dv);
    return OPS_PLACE(op, where, q, k, v, g, beta, state, out);
}

/* 1 / |x| over x[base .. base + n), as ath_l2norm_ref computes it. */
static float l2_inv(const ath_tensor *x, int64_t base, int64_t n, float eps)
{
    float ss = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float v = ld(x, base + i);
        ss += v * v;
    }
    return 1.0f / sqrtf(ss / 1.0f + eps);
}

ath_status ath_gdn_ref(const ath_tensor *q, const ath_tensor *k, const ath_tensor *v,
                       const ath_tensor *g, const ath_tensor *beta, float q_scale, float l2_eps,
                       ath_tensor *state, ath_tensor *out)
{
    ATH_CHECK(gdn_args(q, k, v, g, beta, state, out, OPS_HOST));
    const int64_t T = q->shape[0], nk = q->shape[1], dk = q->shape[2];
    const int64_t nv = v->shape[1], dv = v->shape[2];
    const int64_t group = nv / nk;
    float *S0 = state->data;
    for (int64_t t = 0; t < T; t++)
        for (int64_t h = 0; h < nv; h++) {
            float *S = S0 + h * dk * dv;
            const int64_t qb = t * q->stride[0] + (h / group) * dk;   /* query/key head of value head h */
            const int64_t kb = t * k->stride[0] + (h / group) * dk;
            const int64_t vb = t * v->stride[0] + h * dv, ob = (t * nv + h) * dv;
            const float alpha = expf(ld(g, t * nv + h));
            const float b = ld(beta, t * nv + h);
            const float iq = l2_inv(q, qb, dk, l2_eps), ik = l2_inv(k, kb, dk, l2_eps);
            for (int64_t j = 0; j < dv; j++) {
                float u = 0.0f;
                for (int64_t i = 0; i < dk; i++) {
                    S[i * dv + j] *= alpha;
                    u += S[i * dv + j] * (ld(k, kb + i) * ik);
                }
                float delta = (ld(v, vb + j) - u) * b;
                float o = 0.0f;
                for (int64_t i = 0; i < dk; i++) {
                    S[i * dv + j] += (ld(k, kb + i) * ik) * delta;
                    o += S[i * dv + j] * ((ld(q, qb + i) * iq) * q_scale);
                }
                st(out, ob + j, o);
            }
        }
    return ATH_OK;
}

ath_status ath_gdn_gates_gpu(const ath_device *d, const ath_tensor *a, const ath_tensor *b,
                             const ath_tensor *a_log, const ath_tensor *dt_bias, ath_tensor *g, ath_tensor *beta)
{
    ATH_CHECK(gates_args(a, b, a_log, dt_bias, g, beta, OPS_GPU));
    return ath_gdn_gates_launch(d, a, b, a_log, dt_bias, g, beta);
}

ath_status ath_gdn_gpu(const ath_device *d, const ath_tensor *q, const ath_tensor *k, const ath_tensor *v,
                       const ath_tensor *g, const ath_tensor *beta, float q_scale, float l2_eps, ath_tensor *state,
                       ath_dev_arena *scratch, ath_tensor *out)
{
    ATH_CHECK(gdn_args(q, k, v, g, beta, state, out, OPS_GPU));
    if (q->shape[2] > ATH_GDN_MAX_DK)
        return ops_fail(ATH_ERR_UNSUPPORTED, gdn_op, "key head dim %lld > %d on the GPU", (long long)q->shape[2],
                        ATH_GDN_MAX_DK);
    /* prompts: the chunked form, with its working memory from scratch (step 10) */
    if (scratch && q->shape[0] >= ATH_GDN_CHUNK_MIN_T && ath_gdn_chunk_fits(d, q, k, v, g)) {
        const ath_dev_arena_pos mark = ath_dev_arena_mark(scratch);
        void *work = ath_dev_arena_alloc(scratch, ath_gdn_chunk_bytes(d, q, v), 0);
        if (!work)
            return ATH_ERR_NOMEM;
        ath_status st = ath_gdn_chunk_launch(d, q, k, v, g, beta, q_scale, l2_eps, state, work, out);
        ath_dev_arena_reset(scratch, mark);   /* the next work on the stream comes after the kernels */
        return st;
    }
    return ath_gdn_launch(d, q, k, v, g, beta, q_scale, l2_eps, state, out);
}
