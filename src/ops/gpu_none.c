/* gpu_none.c: the kernel launchers (launch.h) in a build without a GPU
 * backend (`make`). Every _gpu op has checked its arguments when it gets
 * here; there is no device to run on.
 */
#include "ops/launch.h"

static ath_status none(const char *op)
{
    ath_log(ATH_LOG_ERROR, "%s: this build has no GPU backend (make cuda)", op);
    return ATH_ERR_UNSUPPORTED;
}

ath_status ath_embed_launch(const ath_device *d, const ath_tensor *table, const ath_tensor *ids, ath_tensor *out)
{
    (void)d, (void)table, (void)ids, (void)out;
    return none("embed");
}

ath_status ath_matmul_launch(const ath_device *d, const ath_tensor *x, const ath_tensor *w, ath_tensor *out)
{
    (void)d, (void)x, (void)w, (void)out;
    return none("matmul");
}

ath_status ath_norm_launch(const ath_device *d, ath_norm_kind kind, const ath_tensor *x, const ath_tensor *z,
                           const ath_tensor *w, float w_offset, float eps, ath_tensor *out)
{
    (void)d, (void)kind, (void)x, (void)z, (void)w, (void)w_offset, (void)eps, (void)out;
    return none("norm");
}

ath_status ath_rope_launch(const ath_device *d, const ath_tensor *x, const ath_tensor *pos, int32_t rope_dim,
                           const float *freq, ath_tensor *out)
{
    (void)d, (void)x, (void)pos, (void)rope_dim, (void)freq, (void)out;
    return none("rope");
}

ath_status ath_elem_launch(const ath_device *d, ath_elem_kind kind, const ath_tensor *a, const ath_tensor *b,
                           ath_tensor *out)
{
    (void)d, (void)kind, (void)a, (void)b, (void)out;
    return none("elementwise op");
}

ath_status ath_attn_launch(const ath_device *d, const ath_tensor *q, const ath_tensor *k, const ath_tensor *v,
                           float scale, ath_tensor *out)
{
    (void)d, (void)q, (void)k, (void)v, (void)scale, (void)out;
    return none("attn");
}

ath_status ath_conv_launch(const ath_device *d, const ath_tensor *x, const ath_tensor *w, ath_tensor *state,
                           ath_act act, ath_tensor *out)
{
    (void)d, (void)x, (void)w, (void)state, (void)act, (void)out;
    return none("conv");
}

ath_status ath_gdn_gates_launch(const ath_device *d, const ath_tensor *a, const ath_tensor *b,
                                const ath_tensor *a_log, const ath_tensor *dt_bias, ath_tensor *g, ath_tensor *beta)
{
    (void)d, (void)a, (void)b, (void)a_log, (void)dt_bias, (void)g, (void)beta;
    return none("gdn_gates");
}

ath_status ath_gdn_launch(const ath_device *d, const ath_tensor *q, const ath_tensor *k, const ath_tensor *v,
                          const ath_tensor *g, const ath_tensor *beta, float q_scale, ath_tensor *state,
                          ath_tensor *out)
{
    (void)d, (void)q, (void)k, (void)v, (void)g, (void)beta, (void)q_scale, (void)state, (void)out;
    return none("gdn");
}
