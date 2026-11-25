/* launch.h: the kernel launchers, one per GPU op. Private to src/ops/.
 *
 * Each ath_<op>_gpu in ops.h checks its arguments in the op's .c file, with
 * the same code as ath_<op>_ref, then calls the launcher here, defined in
 * the op's .cu file: it makes the device current, queues the kernels on the
 * device's compute stream and returns. In a build without a GPU backend,
 * src/ops/gpu_none.c defines every launcher as a failure
 * (ATH_ERR_UNSUPPORTED).
 *
 * Launchers trust their arguments: shapes, dtypes and placement were
 * checked by the caller, and the limits below as well.
 */
#ifndef ATH_OPS_LAUNCH_H
#define ATH_OPS_LAUNCH_H

#include "backend/backend.h"
#include "ops/ops.h"

#ifdef __cplusplus
extern "C" {
#endif

ath_status ath_embed_launch(const ath_device *d, const ath_tensor *table, const ath_tensor *ids, ath_tensor *out);
ath_status ath_matmul_launch(const ath_device *d, const ath_tensor *x, const ath_tensor *w, ath_tensor *out);

/* The three norms share one kernel (norm.cu). */
typedef enum { ATH_NORM_RMS, ATH_NORM_RMS_GATED, ATH_NORM_L2 } ath_norm_kind;
ath_status ath_norm_launch(const ath_device *d, ath_norm_kind kind, const ath_tensor *x, const ath_tensor *z,
                           const ath_tensor *w, float w_offset, float eps, ath_tensor *out);

/* freq[i], i < rope_dim / 2: the frequencies, computed on the host by
 * rope.c with the reference's own expression (rope.cu says why). */
ath_status ath_rope_launch(const ath_device *d, const ath_tensor *x, const ath_tensor *pos, int32_t rope_dim,
                           const float *freq, ath_tensor *out);

/* The elementwise ops share one kernel (act.cu). */
typedef enum { ATH_ELEM_SWIGLU, ATH_ELEM_SIGMOID_MUL, ATH_ELEM_ADD } ath_elem_kind;
ath_status ath_elem_launch(const ath_device *d, ath_elem_kind kind, const ath_tensor *a, const ath_tensor *b,
                           ath_tensor *out);

ath_status ath_attn_launch(const ath_device *d, const ath_tensor *q, const ath_tensor *k, const ath_tensor *v,
                           float scale, ath_tensor *out);
ath_status ath_conv_launch(const ath_device *d, const ath_tensor *x, const ath_tensor *w, ath_tensor *state,
                           ath_act act, ath_tensor *out);
ath_status ath_gdn_gates_launch(const ath_device *d, const ath_tensor *a, const ath_tensor *b,
                                const ath_tensor *a_log, const ath_tensor *dt_bias, ath_tensor *g, ath_tensor *beta);
ath_status ath_gdn_launch(const ath_device *d, const ath_tensor *q, const ath_tensor *k, const ath_tensor *v,
                          const ath_tensor *g, const ath_tensor *beta, float q_scale, ath_tensor *state,
                          ath_tensor *out);

#ifdef __cplusplus
}
#endif

#endif /* ATH_OPS_LAUNCH_H */
