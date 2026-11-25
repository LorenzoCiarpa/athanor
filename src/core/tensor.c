/* tensor.c: dtype tables and tensor views. See tensor.h. */
#include "core/tensor.h"

#include <string.h>

static const struct {
    const char *name;
    size_t      size;
} dtype_info[ATH_DTYPE_COUNT] = {
    [ATH_DTYPE_F32]     = {"f32", 4},
    [ATH_DTYPE_F16]     = {"f16", 2},
    [ATH_DTYPE_BF16]    = {"bf16", 2},
    [ATH_DTYPE_I32]     = {"i32", 4},
    [ATH_DTYPE_I64]     = {"i64", 8},
    [ATH_DTYPE_U8]      = {"u8", 1},
    [ATH_DTYPE_F8_E4M3] = {"f8_e4m3", 1},
    [ATH_DTYPE_F8_E5M2] = {"f8_e5m2", 1},
};

size_t ath_dtype_size(ath_dtype dt)
{
    return (unsigned)dt < ATH_DTYPE_COUNT ? dtype_info[dt].size : 0;
}

bool ath_dtype_is_quant(ath_dtype dt)
{
    return dt == ATH_DTYPE_Q8_0 || dt == ATH_DTYPE_Q8_0_R16;
}

int64_t ath_dtype_block(ath_dtype dt)
{
    if (dt == ATH_DTYPE_Q8_0)
        return ATH_Q8_0_BLOCK;
    if (dt == ATH_DTYPE_Q8_0_R16)
        return ATH_Q8_0_R16_BLOCK;
    return ath_dtype_size(dt) ? 1 : 0;
}

/* Bytes of one block. */
static size_t block_bytes(ath_dtype dt)
{
    if (dt == ATH_DTYPE_Q8_0)
        return ATH_Q8_0_BYTES;
    if (dt == ATH_DTYPE_Q8_0_R16)
        return ATH_Q8_0_R16_BYTES;
    return ath_dtype_size(dt);
}

size_t ath_dtype_bytes(ath_dtype dt, int64_t n)
{
    int64_t b = ath_dtype_block(dt);
    return b ? (size_t)(n / b) * block_bytes(dt) : 0;
}

const char *ath_dtype_name(ath_dtype dt)
{
    if (dt == ATH_DTYPE_Q8_0)
        return "q8_0";
    if (dt == ATH_DTYPE_Q8_0_R16)
        return "q8_0r16";
    return (unsigned)dt < ATH_DTYPE_COUNT ? dtype_info[dt].name : "?";
}

const char *ath_tier_name(ath_tier tier)
{
    switch (tier) {
    case ATH_TIER_RAM:    return "ram";
    case ATH_TIER_PINNED: return "pinned";
    case ATH_TIER_VRAM:   return "vram";
    case ATH_TIER_DISK:   return "disk";
    }
    return "?";
}

ath_status ath_tensor_init(ath_tensor *t, ath_dtype dtype, int32_t ndim, const int64_t *shape,
                           void *data, ath_tier tier)
{
    const int64_t block = ath_dtype_block(dtype);
    if (block == 0)
        return ATH_ERR_UNSUPPORTED;
    if (ndim < 0 || ndim > ATH_MAX_DIMS)
        return ATH_ERR_RANGE;
    if (block > 1 && (ndim == 0 || shape[ndim - 1] % block != 0))
        return ATH_ERR_INVALID_ARG;

    /* The product of the non-zero dimensions times the bytes of a block must
     * fit in int64_t and size_t. That bounds every stride and the byte size,
     * also when a zero dimension makes the tensor empty. */
    uint64_t limit = (uint64_t)INT64_MAX < (uint64_t)SIZE_MAX ? (uint64_t)INT64_MAX : (uint64_t)SIZE_MAX;
    uint64_t bytes = block_bytes(dtype);
    for (int32_t i = 0; i < ndim; i++) {
        if (shape[i] < 0)
            return ATH_ERR_RANGE;
        if (shape[i] == 0)
            continue;
        if (bytes > limit / (uint64_t)shape[i])
            return ATH_ERR_RANGE;
        bytes *= (uint64_t)shape[i];
    }

    memset(t, 0, sizeof *t);
    t->dtype = dtype;
    t->tier = tier;
    t->ndim = ndim;
    t->data = data;
    int64_t stride = 1;
    for (int32_t i = ndim - 1; i >= 0; i--) {
        t->shape[i] = shape[i];
        t->stride[i] = stride;
        stride *= shape[i] > 0 ? shape[i] : 1;
    }
    return ATH_OK;
}

int64_t ath_tensor_numel(const ath_tensor *t)
{
    int64_t n = 1;
    for (int32_t i = 0; i < t->ndim; i++)
        n *= t->shape[i];
    return n;
}

size_t ath_tensor_nbytes(const ath_tensor *t)
{
    return ath_dtype_bytes(t->dtype, ath_tensor_numel(t));
}

bool ath_tensor_is_contiguous(const ath_tensor *t)
{
    /* Dimensions of size 1 can have any stride without changing the layout. */
    int64_t expect = 1;
    for (int32_t i = t->ndim - 1; i >= 0; i--) {
        if (t->shape[i] == 0)
            return true;
        if (t->shape[i] != 1 && t->stride[i] != expect)
            return false;
        expect *= t->shape[i];
    }
    return true;
}

ath_status ath_tensor_slice(const ath_tensor *t, int32_t dim, int64_t start, int64_t len,
                            ath_tensor *out)
{
    if (dim < 0 || dim >= t->ndim)
        return ATH_ERR_INVALID_ARG;
    if (start < 0 || len < 0 || start > t->shape[dim] || len > t->shape[dim] - start)
        return ATH_ERR_RANGE;
    const int64_t block = ath_dtype_block(t->dtype);
    if (block > 1 && ((start * t->stride[dim]) % block || (len * t->stride[dim]) % block))
        return ATH_ERR_INVALID_ARG;
    ath_tensor v = *t;
    v.shape[dim] = len;
    if (t->data)
        v.data = (char *)t->data + ath_dtype_bytes(t->dtype, start * t->stride[dim]);
    *out = v;
    return ATH_OK;
}

ath_status ath_tensor_reshape(const ath_tensor *t, int32_t ndim, const int64_t *shape,
                              ath_tensor *out)
{
    if (!ath_tensor_is_contiguous(t))
        return ATH_ERR_INVALID_ARG;
    ath_tensor v;
    ATH_CHECK(ath_tensor_init(&v, t->dtype, ndim, shape, t->data, t->tier));
    if (ath_tensor_numel(&v) != ath_tensor_numel(t))
        return ATH_ERR_INVALID_ARG;
    *out = v;
    return ATH_OK;
}

bool ath_tensor_same_shape(const ath_tensor *a, const ath_tensor *b)
{
    if (a->ndim != b->ndim)
        return false;
    for (int32_t i = 0; i < a->ndim; i++)
        if (a->shape[i] != b->shape[i])
            return false;
    return true;
}
