/* embed.cu: token embedding lookup on the GPU. See ops.h and embed.c.
 *
 * One thread per output element. The ids live on the device, so the host
 * cannot check them before the launch: an id outside the table gives a row
 * of NaN (visible downstream) instead of a read outside the table.
 */
#include "ops/cuda/kernel.cuh"
#include "ops/q8_0.h"

static __global__ void embed_kernel(const void *table, int tdt, const int32_t *ids, void *out, int odt, int64_t n,
                                    int64_t H, int64_t V)
{
    K_FOR(e, n * H) {
        const int32_t id = ids[e / H];
        const float v = id >= 0 && id < V ? k_ld(table, tdt, (int64_t)id * H + e % H) : __int_as_float(0x7fc00000);
        k_st(out, odt, e, v);
    }
}

/* A Q8_0 table (step 12): element j of a row is d · q[j % 32] of its block,
 * exact in f32, then rounded to out's dtype, as embed.c. */
static __global__ void embed_q8_0(const uint8_t *table, const int32_t *ids, void *out, int odt, int64_t n, int64_t H,
                                  int64_t V)
{
    const int64_t row = H / ATH_Q8_0_BLOCK * ATH_Q8_0_BYTES;
    K_FOR(e, n * H) {
        const int32_t id = ids[e / H];
        const int64_t j = e % H;
        float v = __int_as_float(0x7fc00000);
        if (id >= 0 && id < V) {
            const uint8_t *blk = table + id * row + j / ATH_Q8_0_BLOCK * ATH_Q8_0_BYTES;
            v = ath_q8_0_scale(blk) * (float)(int8_t)blk[2 + j % ATH_Q8_0_BLOCK];
        }
        k_st(out, odt, e, v);
    }
}

extern "C" ath_status ath_embed_launch(const ath_device *d, const ath_tensor *table, const ath_tensor *ids,
                                       ath_tensor *out)
{
    const int64_t n = ids->shape[0], H = table->shape[1], V = table->shape[0];
    if (n * H == 0)
        return ATH_OK;
    ATH_CHECK(ath_gpu_use(d));
    if (table->dtype == ATH_DTYPE_Q8_0) {
        embed_q8_0<<<k_blocks(n * H, 256), 256, 0, k_stream(d)>>>((const uint8_t *)table->data,
                                                                 (const int32_t *)ids->data, out->data, out->dtype, n,
                                                                 H, V);
        return ath_gpu_launched("embed_q8_0");
    }
    embed_kernel<<<k_blocks(n * H, 256), 256, 0, k_stream(d)>>>(table->data, table->dtype, (const int32_t *)ids->data,
                                                                 out->data, out->dtype, n, H, V);
    return ath_gpu_launched("embed_kernel");
}
