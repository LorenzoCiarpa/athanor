/* dispatch.c: one entry point per op, which runs it where a forward pass
 * runs. See ops.h.
 *
 * The choice is the caller's: a device, or none for the CPU. Each
 * implementation checks that the tensors are on its side (ops_place), so a
 * tensor left on the wrong side is an error, never a silent copy: the
 * engine moves data before it runs ops (core/tensor.h, ath_tier).
 */
#include "ops/ops.h"

ath_status ath_embed(const ath_device *d, const ath_tensor *table, const ath_tensor *ids, ath_tensor *out)
{
    return d ? ath_embed_gpu(d, table, ids, out) : ath_embed_ref(table, ids, out);
}

ath_status ath_matmul(const ath_device *d, ath_pool *pool, const ath_tensor *x, const ath_tensor *w, ath_tensor *out)
{
    return d ? ath_matmul_gpu(d, x, w, out) : ath_matmul_fast(x, w, out, pool);
}

ath_status ath_rmsnorm(const ath_device *d, const ath_tensor *x, const ath_tensor *w, float w_offset, float eps,
                       ath_tensor *out)
{
    return d ? ath_rmsnorm_gpu(d, x, w, w_offset, eps, out) : ath_rmsnorm_ref(x, w, w_offset, eps, out);
}

ath_status ath_rmsnorm_gated(const ath_device *d, const ath_tensor *x, const ath_tensor *z, const ath_tensor *w,
                             float eps, ath_tensor *out)
{
    return d ? ath_rmsnorm_gated_gpu(d, x, z, w, eps, out) : ath_rmsnorm_gated_ref(x, z, w, eps, out);
}

ath_status ath_l2norm(const ath_device *d, const ath_tensor *x, float eps, ath_tensor *out)
{
    return d ? ath_l2norm_gpu(d, x, eps, out) : ath_l2norm_ref(x, eps, out);
}

ath_status ath_rope(const ath_device *d, const ath_tensor *x, const ath_tensor *pos, int32_t rope_dim, float theta,
                    ath_tensor *out)
{
    return d ? ath_rope_gpu(d, x, pos, rope_dim, theta, out) : ath_rope_ref(x, pos, rope_dim, theta, out);
}

ath_status ath_swiglu(const ath_device *d, const ath_tensor *gate, const ath_tensor *up, ath_tensor *out)
{
    return d ? ath_swiglu_gpu(d, gate, up, out) : ath_swiglu_ref(gate, up, out);
}

ath_status ath_sigmoid_mul(const ath_device *d, const ath_tensor *x, const ath_tensor *gate, ath_tensor *out)
{
    return d ? ath_sigmoid_mul_gpu(d, x, gate, out) : ath_sigmoid_mul_ref(x, gate, out);
}

ath_status ath_add(const ath_device *d, const ath_tensor *a, const ath_tensor *b, ath_tensor *out)
{
    return d ? ath_add_gpu(d, a, b, out) : ath_add_ref(a, b, out);
}

ath_status ath_attn(const ath_device *d, const ath_tensor *q, const ath_tensor *k, const ath_tensor *v,
                    const ath_tensor *pos, float scale, ath_dev_arena *scratch, ath_tensor *out)
{
    return d ? ath_attn_gpu(d, q, k, v, pos, scale, scratch, out) : ath_attn_ref(q, k, v, pos, scale, out);
}

ath_status ath_matmul_n(const ath_device *d, ath_pool *pool, const ath_tensor *x, int n, const ath_tensor *const *w,
                        ath_tensor *const *out)
{
    if (d)
        return ath_matmul_n_gpu(d, x, n, w, out);
    if (n < 1 || n > ATH_MATMUL_N_MAX || !w || !out)
        return ATH_ERR_INVALID_ARG;
    for (int i = 0; i < n; i++)
        ATH_CHECK(ath_matmul_fast(x, w[i], out[i], pool));
    return ATH_OK;
}

ath_status ath_scatter_rows(const ath_device *d, const ath_tensor *src, const ath_tensor *idx, ath_tensor *dst)
{
    return d ? ath_scatter_rows_gpu(d, src, idx, dst) : ath_scatter_rows_ref(src, idx, dst);
}

ath_status ath_conv(const ath_device *d, const ath_tensor *x, const ath_tensor *w, ath_tensor *state, ath_act act,
                    ath_tensor *out)
{
    return d ? ath_conv_gpu(d, x, w, state, act, out) : ath_conv_ref(x, w, state, act, out);
}

ath_status ath_gdn_gates(const ath_device *d, const ath_tensor *a, const ath_tensor *b, const ath_tensor *a_log,
                         const ath_tensor *dt_bias, ath_tensor *g, ath_tensor *beta)
{
    return d ? ath_gdn_gates_gpu(d, a, b, a_log, dt_bias, g, beta) : ath_gdn_gates_ref(a, b, a_log, dt_bias, g, beta);
}

ath_status ath_gdn(const ath_device *d, const ath_tensor *q, const ath_tensor *k, const ath_tensor *v,
                   const ath_tensor *g, const ath_tensor *beta, float q_scale, ath_tensor *state,
                   ath_dev_arena *scratch, ath_tensor *out)
{
    return d ? ath_gdn_gpu(d, q, k, v, g, beta, q_scale, state, scratch, out)
             : ath_gdn_ref(q, k, v, g, beta, q_scale, state, out);
}
