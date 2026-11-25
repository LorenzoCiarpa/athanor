/* check.h: argument checks and element access shared by the ops: their
 * _ref, _fast and _gpu functions check the same shapes and dtypes, and
 * differ in where the tensors must be (ops_place). Private to src/ops/ (the
 * public interface is ops.h).
 */
#ifndef ATH_OPS_CHECK_H
#define ATH_OPS_CHECK_H

#include "core/tensor.h"

#include <stdarg.h>
#include <stdio.h>

/* Log "<op>: <message>" and return ATH_ERR_INVALID_ARG (or the given status). */
static inline ath_status ops_fail(ath_status s, const char *op, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));

static inline ath_status ops_fail(ath_status s, const char *op, const char *fmt, ...)
{
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    ath_log(ATH_LOG_ERROR, "%s: %s", op, msg);
    return s;
}

/* A tensor of dtype `dtype` (or, with dtype ATH_DTYPE_COUNT, f32 or bf16),
 * present and contiguous. Where it lives is checked by ops_place. */
static inline ath_status ops_tensor(const ath_tensor *t, ath_dtype dtype, const char *op,
                                    const char *name)
{
    if (!t || !t->data)
        return ops_fail(ATH_ERR_INVALID_ARG, op, "%s: missing", name);
    if (dtype == ATH_DTYPE_COUNT ? t->dtype != ATH_DTYPE_F32 && t->dtype != ATH_DTYPE_BF16
                                 : t->dtype != dtype)
        return ops_fail(ATH_ERR_TYPE, op, "%s: dtype %s not supported", name,
                        ath_dtype_name(t->dtype));
    if (!ath_tensor_is_contiguous(t))
        return ops_fail(ATH_ERR_INVALID_ARG, op, "%s: not contiguous", name);
    return ATH_OK;
}

/* Where an implementation needs its tensors: _ref and _fast read and write
 * them on the host (tier RAM or PINNED), _gpu on the device (tier VRAM). */
typedef enum { OPS_HOST, OPS_GPU } ops_where;

/* Every one of the n tensors where `where` needs it; ATH_ERR_UNSUPPORTED
 * otherwise. Checked after the shapes and dtypes. */
static inline ath_status ops_place(const char *op, ops_where where, const ath_tensor *const *t, int n)
{
    for (int i = 0; i < n; i++) {
        const ath_tier tier = t[i]->tier;
        const bool host = tier == ATH_TIER_RAM || tier == ATH_TIER_PINNED;
        if (where == OPS_HOST && !host)
            return ops_fail(ATH_ERR_UNSUPPORTED, op, "a tensor in %s: the reference needs host memory",
                            ath_tier_name(tier));
        if (where == OPS_GPU && tier != ATH_TIER_VRAM)
            return ops_fail(ATH_ERR_UNSUPPORTED, op, "a tensor in %s: the GPU kernel needs device memory",
                            ath_tier_name(tier));
    }
    return ATH_OK;
}

/* OPS_PLACE(op, where, x, w, out): ops_place over the tensors listed. */
#define OPS_PLACE(op, where, ...)                                                          \
    ops_place((op), (where), (const ath_tensor *const[]){__VA_ARGS__},                     \
              (int)(sizeof((const ath_tensor *const[]){__VA_ARGS__}) / sizeof(const ath_tensor *)))

/* An f32 or bf16 tensor (activations and weights). */
static inline ath_status ops_float(const ath_tensor *t, const char *op, const char *name)
{
    return ops_tensor(t, ATH_DTYPE_COUNT, op, name);
}

static inline int64_t ops_last(const ath_tensor *t)
{
    return t->ndim > 0 ? t->shape[t->ndim - 1] : 1;
}

/* Number of rows: the product of every dimension but the last. */
static inline int64_t ops_rows(const ath_tensor *t)
{
    int64_t last = ops_last(t);
    return last > 0 ? ath_tensor_numel(t) / last : 0;
}

static inline float ld(const ath_tensor *t, int64_t i)
{
    return ath_load_f32(t->data, t->dtype, i);
}

static inline void st(const ath_tensor *t, int64_t i, float v)
{
    ath_store_f32(t->data, t->dtype, i, v);
}

#endif /* ATH_OPS_CHECK_H */
