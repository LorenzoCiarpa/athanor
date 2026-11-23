/* tensor.h: element types and the tensor descriptor.
 *
 * An ath_tensor does not own memory. It describes memory that lives somewhere
 * else (an mmap of a checkpoint, an arena, VRAM): its element type, shape,
 * strides and the tier it sits in. Views (slices, reshapes) are new
 * descriptors over the same bytes; nothing is copied.
 *
 * Layout is row-major: the last dimension is the fastest varying. Strides are
 * counted in elements, not bytes.
 */
#ifndef ATH_TENSOR_H
#define ATH_TENSOR_H

#include "core/base.h"

#define ATH_MAX_DIMS 8

typedef enum {
    ATH_DTYPE_F32,
    ATH_DTYPE_F16,
    ATH_DTYPE_BF16,
    ATH_DTYPE_I32,
    ATH_DTYPE_I64,
    ATH_DTYPE_U8,
    ATH_DTYPE_F8_E4M3,     /* 4 exponent bits, 3 mantissa bits, no infinities (OCP "fn") */
    ATH_DTYPE_F8_E5M2,
    ATH_DTYPE_COUNT,       /* number of plain element types */

    /* Values 64..127 are reserved for athanor's quantized block formats
     * (phase 2, milestone Q1). They pack several elements with shared scales,
     * so they have no per-element size: ath_dtype_size() returns 0 for them. */
    ATH_DTYPE_QUANT_FIRST = 64,
    ATH_DTYPE_QUANT_LAST  = 127
} ath_dtype;

/* Bytes per element; 0 for reserved or unknown values. */
size_t      ath_dtype_size(ath_dtype dt);
/* Lower-case name ("bf16"); "?" for reserved or unknown values. */
const char *ath_dtype_name(ath_dtype dt);

/* Where the bytes live. Kernels never branch on the tier; the engine moves
 * data before launching them. Memory-mapped checkpoint data is ATH_TIER_RAM:
 * host-addressable, paged in from the file by the kernel on first touch. */
typedef enum {
    ATH_TIER_RAM,
    ATH_TIER_PINNED,
    ATH_TIER_VRAM,
    ATH_TIER_DISK
} ath_tier;

const char *ath_tier_name(ath_tier tier);

typedef struct {
    ath_dtype dtype;
    ath_tier  tier;
    int32_t   ndim;                 /* 0 is a scalar with one element */
    int64_t   shape[ATH_MAX_DIMS];
    int64_t   stride[ATH_MAX_DIMS]; /* in elements */
    void     *data;                 /* first element; may be NULL for a pure descriptor */
} ath_tensor;

/* Contiguous row-major descriptor. Checks the dtype, that 0 <= ndim <=
 * ATH_MAX_DIMS, that no dimension is negative and that the byte size fits in
 * size_t. Unused shape and stride entries are zeroed. */
ath_status ath_tensor_init(ath_tensor *t, ath_dtype dtype, int32_t ndim, const int64_t *shape,
                           void *data, ath_tier tier);

int64_t ath_tensor_numel(const ath_tensor *t);
/* numel * element size: the bytes the elements occupy if packed. */
size_t  ath_tensor_nbytes(const ath_tensor *t);
bool    ath_tensor_is_contiguous(const ath_tensor *t);

/* View of elements [start, start + len) along dimension dim. The view shares
 * memory with t; it is contiguous only if t is and dim is the outermost
 * dimension with more than one element. */
ath_status ath_tensor_slice(const ath_tensor *t, int32_t dim, int64_t start, int64_t len,
                            ath_tensor *out);

/* Same elements with a different shape. t must be contiguous and the element
 * count must not change. */
ath_status ath_tensor_reshape(const ath_tensor *t, int32_t ndim, const int64_t *shape,
                              ath_tensor *out);

#endif /* ATH_TENSOR_H */
