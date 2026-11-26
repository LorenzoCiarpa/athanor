/* attn.c: causal softmax attention with grouped-query heads. See ops.h.
 *
 * Naive and exact: for each query and head, one pass for the largest score,
 * one pass that accumulates exp(s_j − max) and exp(s_j − max) · v_j, and a
 * final division. Scores are recomputed in the second pass instead of being
 * stored, so the function needs no memory beyond the dv accumulators.
 * The reference (eager_attention_forward) rounds the scores and the
 * probabilities to bf16; this function keeps both in fp32 (docs/MODELS.md:
 * softmax is an fp32 island).
 */
#include "ops/check.h"
#include "ops/launch.h"
#include "ops/ops.h"

#include <math.h>

static const char op[] = "attn";

static ath_status args(const ath_tensor *q, const ath_tensor *k, const ath_tensor *v, const ath_tensor *pos,
                       const ath_tensor *out, ops_where where)
{
    ATH_CHECK(ops_float(q, op, "q"));
    ATH_CHECK(ops_float(k, op, "k"));
    ATH_CHECK(ops_float(v, op, "v"));
    ATH_CHECK(ops_float(out, op, "out"));
    if (q->ndim != 3 || k->ndim != 3 || v->ndim != 3 || out->ndim != 3)
        return ops_fail(ATH_ERR_INVALID_ARG, op, "q, k, v and out must be [tokens, heads, dim]");
    const int64_t T = q->shape[0], nh = q->shape[1], d = q->shape[2];
    const int64_t S = k->shape[0], nkv = k->shape[1], dv = v->shape[2];
    if (k->shape[2] != d || v->shape[0] != S || v->shape[1] != nkv || out->shape[0] != T ||
        out->shape[1] != nh || out->shape[2] != dv || (!pos && S < T) || nkv <= 0 || nh % nkv != 0)
        return ops_fail(ATH_ERR_INVALID_ARG, op,
                        "q [T, nh, d], k [S, nkv, d], v [S, nkv, dv], out [T, nh, dv], nh a multiple of nkv, "
                        "S >= T without positions");
    if (pos) {
        ATH_CHECK(ops_tensor(pos, ATH_DTYPE_I32, op, "pos"));
        if (pos->ndim != 1 || pos->shape[0] != T)
            return ops_fail(ATH_ERR_INVALID_ARG, op, "pos must be [T]");
        ATH_CHECK(OPS_PLACE(op, where, pos));
        for (int64_t t = 0; where == OPS_HOST && t < T; t++)
            if (((const int32_t *)pos->data)[t] < 0 || ((const int32_t *)pos->data)[t] >= S)
                return ops_fail(ATH_ERR_RANGE, op, "position %d of query %lld outside the %lld keys",
                                ((const int32_t *)pos->data)[t], (long long)t, (long long)S);
    }
    if (dv > ATH_ATTN_MAX_DV)
        return ops_fail(ATH_ERR_UNSUPPORTED, op, "value head dim %lld > %d", (long long)dv,
                        ATH_ATTN_MAX_DV);
    return OPS_PLACE(op, where, q, k, v, out);
}

/* Keys or values: an f32 or bf16 tensor, or (step 14) an int8 cache with
 * its row scales, read back as q · s rounded to rnd (the queries' dtype). */
typedef struct {
    const ath_tensor *t, *s;
    ath_dtype         rnd;
} kvt;

static float kvld(kvt a, int64_t i, int64_t d)
{
    if (!a.s)
        return ld(a.t, i);
    return ath_round_to(a.rnd, (float)((const int8_t *)a.t->data)[i] * ((const float *)a.s->data)[i / d]);
}

static void attn_core(const ath_tensor *q, kvt k, kvt v, const ath_tensor *pos, float scale, ath_tensor *out)
{
    const int64_t T = q->shape[0], nh = q->shape[1], d = q->shape[2];
    const int64_t S = k.t->shape[0], nkv = k.t->shape[1], dv = v.t->shape[2];
    const int64_t group = nh / nkv;
    float acc[ATH_ATTN_MAX_DV];
    for (int64_t t = 0; t < T; t++) {
        const int64_t last = pos ? ((const int32_t *)pos->data)[t] : S - T + t;   /* last key it sees */
        for (int64_t h = 0; h < nh; h++) {
            const int64_t kh = h / group;
            const int64_t qb = (t * nh + h) * d;
            float max = -INFINITY;
            for (int64_t j = 0; j <= last; j++) {
                float s = 0.0f;
                for (int64_t i = 0; i < d; i++)
                    s += ld(q, qb + i) * kvld(k, (j * nkv + kh) * d + i, d);
                s *= scale;
                if (s > max)
                    max = s;
            }
            float sum = 0.0f;
            for (int64_t i = 0; i < dv; i++)
                acc[i] = 0.0f;
            for (int64_t j = 0; j <= last; j++) {
                float s = 0.0f;
                for (int64_t i = 0; i < d; i++)
                    s += ld(q, qb + i) * kvld(k, (j * nkv + kh) * d + i, d);
                float e = expf(s * scale - max);
                sum += e;
                for (int64_t i = 0; i < dv; i++)
                    acc[i] += e * kvld(v, (j * nkv + kh) * dv + i, dv);
            }
            for (int64_t i = 0; i < dv; i++)
                st(out, (t * nh + h) * dv + i, acc[i] / sum);
        }
    }
}

ath_status ath_attn_ref(const ath_tensor *q, const ath_tensor *k, const ath_tensor *v, const ath_tensor *pos,
                        float scale, ath_tensor *out)
{
    ATH_CHECK(args(q, k, v, pos, out, OPS_HOST));
    attn_core(q, (kvt){k, NULL, q->dtype}, (kvt){v, NULL, q->dtype}, pos, scale, out);
    return ATH_OK;
}

/* An int8 cache (step 14): k and v ATH_DTYPE_I8, their scales [S, nkv] f32;
 * the rest as args(). */
static ath_status args_q8(const ath_tensor *q, const ath_tensor *k, const ath_tensor *ks, const ath_tensor *v,
                          const ath_tensor *vs, const ath_tensor *pos, const ath_tensor *out, ops_where where)
{
    ATH_CHECK(ops_tensor(k, ATH_DTYPE_I8, op, "k"));
    ATH_CHECK(ops_tensor(v, ATH_DTYPE_I8, op, "v"));
    ATH_CHECK(ops_tensor(ks, ATH_DTYPE_F32, op, "k scale"));
    ATH_CHECK(ops_tensor(vs, ATH_DTYPE_F32, op, "v scale"));
    if (k->ndim != 3 || v->ndim != 3 || ks->ndim != 2 || vs->ndim != 2 || ks->shape[0] != k->shape[0] ||
        ks->shape[1] != k->shape[1] || vs->shape[0] != v->shape[0] || vs->shape[1] != v->shape[1])
        return ops_fail(ATH_ERR_INVALID_ARG, op, "k [S, nkv, d] and v [S, nkv, dv] in i8, their scales [S, nkv]");
    /* the shapes and places of args(), with float stand-ins for the int8 tensors */
    ath_tensor kf = *k, vf = *v;
    kf.dtype = vf.dtype = q->dtype;
    ATH_CHECK(args(q, &kf, &vf, pos, out, where));
    return OPS_PLACE(op, where, ks, vs);
}

ath_status ath_attn_q8_ref(const ath_tensor *q, const ath_tensor *k, const ath_tensor *ks, const ath_tensor *v,
                           const ath_tensor *vs, const ath_tensor *pos, float scale, ath_tensor *out)
{
    ATH_CHECK(args_q8(q, k, ks, v, vs, pos, out, OPS_HOST));
    attn_core(q, (kvt){k, ks, q->dtype}, (kvt){v, vs, q->dtype}, pos, scale, out);
    return ATH_OK;
}

ath_status ath_attn_q8_gpu(const ath_device *d, const ath_tensor *q, const ath_tensor *k, const ath_tensor *ks,
                           const ath_tensor *v, const ath_tensor *vs, const ath_tensor *pos, float scale,
                           ath_dev_arena *scratch, ath_tensor *out)
{
    ATH_CHECK(args_q8(q, k, ks, v, vs, pos, out, OPS_GPU));
    if (q->shape[2] > ATH_ATTN_MAX_DV)
        return ops_fail(ATH_ERR_UNSUPPORTED, op, "head dim %lld > %d on the GPU", (long long)q->shape[2],
                        ATH_ATTN_MAX_DV);
    const int64_t T = q->shape[0], nh = q->shape[1], S = k->shape[0], dv = v->shape[2];
    float *part = NULL;
    ath_dev_arena_pos mark = {0};
    if (scratch && T <= ATH_ATTN_SPLIT_MAX_T && T > 0) {
        const int64_t pieces = (S + ATH_ATTN_SPLIT - 1) / ATH_ATTN_SPLIT;
        mark = ath_dev_arena_mark(scratch);
        part = ath_dev_arena_alloc(scratch, (size_t)(T * nh * pieces * (dv + 2)) * sizeof(float), 0);
        if (!part)
            return ATH_ERR_NOMEM;
    }
    ath_status st = ath_attn_q8_launch(d, q, k, ks, v, vs, pos, scale, part, out);
    if (part)
        ath_dev_arena_reset(scratch, mark);
    return st;
}

ath_status ath_attn_gpu(const ath_device *d, const ath_tensor *q, const ath_tensor *k, const ath_tensor *v,
                        const ath_tensor *pos, float scale, ath_dev_arena *scratch, ath_tensor *out)
{
    ATH_CHECK(args(q, k, v, pos, out, OPS_GPU));
    if (q->shape[2] > ATH_ATTN_MAX_DV)
        return ops_fail(ATH_ERR_UNSUPPORTED, op, "head dim %lld > %d on the GPU", (long long)q->shape[2],
                        ATH_ATTN_MAX_DV);
    const int64_t T = q->shape[0], nh = q->shape[1], S = k->shape[0], dv = v->shape[2];
    /* prompts: the tensor cores, where the kernel takes the tensors (step 10) */
    if (T >= ATH_ATTN_MMA_MIN_T && ath_attn_mma_fits(d, q, k, v, out))
        return ath_attn_mma_launch(d, q, k, v, pos, scale, out);
    /* split-K: the partial softmax of every piece of every query, from scratch */
    float *part = NULL;
    ath_dev_arena_pos mark = {0};
    if (scratch && T <= ATH_ATTN_SPLIT_MAX_T && T > 0) {
        const int64_t pieces = (S + ATH_ATTN_SPLIT - 1) / ATH_ATTN_SPLIT;
        mark = ath_dev_arena_mark(scratch);
        part = ath_dev_arena_alloc(scratch, (size_t)(T * nh * pieces * (dv + 2)) * sizeof(float), 0);
        if (!part)
            return ATH_ERR_NOMEM;
    }
    ath_status st = ath_attn_launch(d, q, k, v, pos, scale, part, out);
    if (part)
        ath_dev_arena_reset(scratch, mark);   /* the next work on the stream comes after the kernels */
    return st;
}
