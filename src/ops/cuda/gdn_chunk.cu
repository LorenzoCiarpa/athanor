/* gdn_chunk.cu: the gated delta rule in chunks, for prompts (step 10). See
 * ops.h and gdn.c; gdn.cu has the recurrent kernel, which decode keeps.
 *
 * The recurrence (gdn.c) walks the tokens one by one: for a prompt of
 * thousands of tokens that is thousands of dependent steps. The chunked
 * form of the reference (torch_chunk_gated_delta_rule, contract) computes
 * the same thing with one dependent step per chunk of GC = 64 tokens.
 *
 * Within a chunk, for one value head, let γ_i = g_0 + … + g_i (so that the
 * decay from the chunk's start to token i is e^γ_i), Γ_ij = e^(γ_i − γ_j)
 * for j ≤ i, S the state before the chunk and Δ the rows β_i (v_i − u_i) the
 * recurrence writes into the state. Unrolling the recurrence over the chunk
 * gives (docs/program/gdn-gpu-kernel.md derives it):
 *
 *   (I + A) Δ = βV − (β e^γ K) S,   A_ij = β_i (k_i · k_j) Γ_ij for j < i
 *   O = (q_scale e^γ Q) S + M Δ,    M_ij = q_scale (q_i · k_j) Γ_ij for j ≤ i
 *   S' = e^γ_last S + Σ_j e^(γ_last − γ_j) k_j Δ_jᵀ
 *
 * With T = (I + A)⁻¹ (unit lower triangular), W = T (β e^γ K), U = T (βV),
 * Q'' = q_scale e^γ Q − M W and O_U = M U, the chunk is
 *
 *   Δ = U − W S,   O = Q'' S + O_U,   S' = e^γ_last S + Kᵀ (e^(γ_last − γ) ⊙ Δ)
 *
 * W, U, Q'' and O_U do not depend on S: the first kernel (prep) computes
 * them for every chunk at once, one block per chunk and value head. The
 * second (scan) walks the chunks in order with the state, one block per
 * value head and 32 columns of the state (columns never interact, gdn.c):
 * three small matrix products per chunk instead of 64 steps.
 *
 * Everything is fp32, as in the reference's chunked function. The products
 * are summed in a fixed order, so a call gives the same bits every time, and
 * a prompt fed in pieces whose lengths are multiples of 64 gives the bits of
 * the prompt at once (the chunks are the same). Decay factors are never
 * above 1 (γ only decreases), so nothing overflows.
 *
 * Shapes: dk, dv up to GD (128), a multiple of 4. The tokens go in rounds of
 * chunks (round_chunks: as many as fill the GPU GDN_WAVES times with prep
 * blocks), so that the scratch memory (W, Q'', U, O_U and γ for a round)
 * stays bounded whatever the prompt's length, and in L2 for the scan.
 */
#include "ops/cuda/mma.cuh"

#define GC 64                 /* tokens per chunk                         */
#define GD 128                /* largest dk and dv                        */
#define GV 32                 /* state columns per scan block             */
#define GDN_WAVES 4           /* prep blocks per round: this many per SM  */

/* ---- prep: W, Q'', U, O_U and γ of one chunk and value head ----------------------- */

/* Four warps. Every product is a matrix product on the CUDA cores, an 8 × 8
 * tile per lane: each 16-byte load from shared memory (which costs the
 * same four cycles whether the lanes share its address or not) feeds 16
 * multiply-adds, so the loads keep up with the arithmetic.
 *
 * Shared memory: two [GC][XP] tiles (x0: the keys, then β e^γ K, W, βV,
 * U; x1: the queries, then A and Mᵀ), Tᵀ, and γ, β, β e^γ. A triangular
 * matrix is kept transposed (Tᵀ[j][i] = T_ij, zeros above its diagonal),
 * so that a lane's 8 rows at one j are two loads. */
#define PT 128                /* threads per prep block                   */
#define XP (GD + 4)           /* row of an x tile: rows 4 banks apart     */
#define AP (GC + 4)           /* row of a transposed triangular matrix    */
#define PREP_FLOATS (2 * GC * XP + GC * AP + 3 * GC + 3 * 256)

/* o[a][e] = Σ_j at[j][8 ti + a] x[j][c_e] over every j (at is zero above the
 * diagonal), the lane's rows 8 ti + a and columns c_e = 4 tj + (e < 4 ?
 * e : 60 + e) with ti from the warp and the half-warp, tj = lane % 16. */
static __device__ __forceinline__ void tri_mul(const float (*at)[AP], const float (*x)[XP], float o[8][8])
{
    const int lane = threadIdx.x % ATH_WARP, ti = lane / 16 + 2 * (threadIdx.x / ATH_WARP), c = 4 * (lane % 16);
#pragma unroll
    for (int a = 0; a < 8; a++)
#pragma unroll
        for (int e = 0; e < 8; e++)
            o[a][e] = 0.0f;
#pragma unroll 4
    for (int j = 0; j < GC; j++) {
        const float4 a0 = *(const float4 *)&at[j][8 * ti], a1 = *(const float4 *)&at[j][8 * ti + 4];
        const float4 x0 = *(const float4 *)&x[j][c], x1 = *(const float4 *)&x[j][c + 64];
        const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int r = 0; r < 8; r++)
#pragma unroll
            for (int e = 0; e < 8; e++)
                o[r][e] = __fmaf_rn(av[r], xv[e], o[r][e]);
    }
}

/* The lane's tile of a product into rows of n columns: dst[i][c] row-major
 * (stride floats per row), or transposed, dst[c][i] with GC per row. */
static __device__ __forceinline__ void tri_store(float *dst, int64_t stride, int n, const float o[8][8])
{
    const int lane = threadIdx.x % ATH_WARP, ti = lane / 16 + 2 * (threadIdx.x / ATH_WARP), c = 4 * (lane % 16);
#pragma unroll
    for (int a = 0; a < 8; a++)
#pragma unroll
        for (int h = 0; h < 2; h++)
            if (c + 64 * h < n)
                *(float4 *)&dst[(8 * ti + a) * stride + c + 64 * h] =
                    make_float4(o[a][4 * h], o[a][4 * h + 1], o[a][4 * h + 2], o[a][4 * h + 3]);
}

static __device__ __forceinline__ void tri_store_t(float *dst, int n, const float o[8][8])
{
    const int lane = threadIdx.x % ATH_WARP, ti = lane / 16 + 2 * (threadIdx.x / ATH_WARP), c = 4 * (lane % 16);
#pragma unroll
    for (int e = 0; e < 8; e++) {
        const int col = c + (e < 4 ? e : 60 + e);
        if (col < n) {
            *(float4 *)&dst[col * GC + 8 * ti] = make_float4(o[0][e], o[1][e], o[2][e], o[3][e]);
            *(float4 *)&dst[col * GC + 8 * ti + 4] = make_float4(o[4][e], o[5][e], o[6][e], o[7][e]);
        }
    }
}

static __global__ void __launch_bounds__(PT)
gdn_prep(const float *q, const float *k, const void *v, int vdt, const float *g, const void *beta, int bdt,
         float q_scale, int64_t T, int64_t c0, int64_t nk, int dk, int64_t nv, int dv, float *Qp, float *W,
         float *U, float *OU, float *gam)
{
    extern __shared__ __align__(16) float pm[];
    float (*x0)[XP] = (float (*)[XP])pm;
    float (*x1)[XP] = (float (*)[XP])(pm + GC * XP);
    float (*aa)[GC] = (float (*)[GC])(pm + GC * XP);                  /* A, where the queries were */
    float (*mt)[AP] = (float (*)[AP])(pm + GC * XP + GC * GC);        /* Mᵀ, after A               */
    float (*tt)[AP] = (float (*)[AP])(pm + 2 * GC * XP);              /* Tᵀ                        */
    float *gm = pm + 2 * GC * XP + GC * AP, *bt = gm + GC, *be = bt + GC;   /* γ, β, β e^γ           */
    float (*xt)[16][16] = (float (*)[16][16])(be + GC);               /* blocks of T being built   */
    const int tid = threadIdx.x, lane = tid % ATH_WARP, w = tid / ATH_WARP;
    const int64_t c = c0 + blockIdx.x, h = blockIdx.y, kh = h / (nv / nk), t0 = c * GC;
    const int n = (int)(T - t0 < GC ? T - t0 : GC);
    const int64_t out = ((int64_t)blockIdx.x * nv + h) * GC;     /* this chunk's rows in the round's scratch */

    /* the keys into x0, the queries into x1; rows past n (padding) and
     * columns past dk zeros */
    for (int e = tid; e < GC * GD / 4; e += PT) {
        const int i = e / (GD / 4), d = 4 * (e % (GD / 4));
        const bool ok = i < n && d < dk;
        const int64_t at = ((t0 + i) * nk + kh) * dk + d;
        k_cp16(&x0[i][d], ok ? k + at : k, ok);
        k_cp16(&x1[i][d], ok ? q + at : q, ok);
    }
    k_cp_commit();
    /* γ (in token order, as the reference's cumsum), β; padding: zeros, γ constant */
    if (tid < GC) {
        bt[tid] = tid < n ? k_ld(beta, bdt, (t0 + tid) * nv + h) : 0.0f;
        gm[tid] = tid < n ? g[(t0 + tid) * nv + h] : 0.0f;
    }
    for (int e = tid; e < GC * AP; e += PT)
        (&tt[0][0])[e] = e / AP == e % AP ? 1.0f : 0.0f;
    __syncthreads();
    if (tid == 0)
        for (int i = 1; i < GC; i++)
            gm[i] = __fadd_rn(gm[i - 1], gm[i]);
    k_cp_wait_all();
    __syncthreads();
    if (tid < GC) {
        gam[out + tid] = gm[tid];
        be[tid] = __fmul_rn(bt[tid], expf(gm[tid]));
    }

    /* k_i · k_j and q_i · k_j as one product: rows R = 8 ti + a of [K; Q]
     * (warps 0, 1 the keys, 2, 3 the queries), columns j = tj + 8 b */
    {
        const int ti = lane / 8 + 4 * w, tj = lane % 8;
        const float (*rx)[XP] = ti < 8 ? x0 + 8 * ti : x1 + 8 * (ti - 8);
        float acc[8][8];
#pragma unroll
        for (int a = 0; a < 8; a++)
#pragma unroll
            for (int b = 0; b < 8; b++)
                acc[a][b] = 0.0f;
        for (int d = 0; d < dk; d += 4) {
            float4 r[8], cl[8];
#pragma unroll
            for (int a = 0; a < 8; a++)
                r[a] = *(const float4 *)&rx[a][d];
#pragma unroll
            for (int b = 0; b < 8; b++)
                cl[b] = *(const float4 *)&x0[tj + 8 * b][d];
#pragma unroll
            for (int a = 0; a < 8; a++)
#pragma unroll
                for (int b = 0; b < 8; b++) {
                    acc[a][b] = __fmaf_rn(r[a].x, cl[b].x, acc[a][b]);
                    acc[a][b] = __fmaf_rn(r[a].y, cl[b].y, acc[a][b]);
                    acc[a][b] = __fmaf_rn(r[a].z, cl[b].z, acc[a][b]);
                    acc[a][b] = __fmaf_rn(r[a].w, cl[b].w, acc[a][b]);
                }
        }
        __syncthreads();                               /* the queries are read: A and Mᵀ replace them */
#pragma unroll
        for (int a = 0; a < 8; a++)
#pragma unroll
            for (int b = 0; b < 8; b++) {
                const int i = 8 * (ti & 7) + a, j = tj + 8 * b;
                const float decay = j <= i ? expf(__fsub_rn(gm[i], gm[j])) : 0.0f;
                if (ti < 8)
                    aa[i][j] = j < i ? __fmul_rn(__fmul_rn(bt[i], acc[a][b]), decay) : 0.0f;
                else
                    mt[j][i] = j <= i ? __fmul_rn(__fmul_rn(q_scale, acc[a][b]), decay) : 0.0f;
            }
    }
    __syncthreads();

    /* T = (I + A)⁻¹ in blocks of 16 (FLA's solve_tril does the same). The
     * diagonal blocks D_b = (I + A_bb)⁻¹ by forward substitution, a column
     * per thread: for j < i, T_ij = −A_ij − Σ_{j<m<i} A_im T_mj, which needs
     * only the column's own entries above (its row of Tᵀ). Then the blocks
     * below, nearest the diagonal first: T_bc = −D_b Σ_{c≤e<b} A_be T_ec. */
    if (tid < GC) {
        const int j = tid, b0 = j & ~15;
        for (int i = j + 1; i < b0 + 16; i++) {
            float s = -aa[i][j];
            for (int m = j + 1; m < i; m++)
                s = __fmaf_rn(-aa[i][m], tt[j][m], s);
            tt[j][i] = s;
        }
    }
    __syncthreads();
    for (int dist = 1; dist < GC / 16; dist++) {
        const int nb = GC / 16 - dist;
        for (int e = tid; e < nb * 256; e += PT) {
            const int bi = e / 256, r = (e / 16) % 16, cc = e % 16, b = bi + dist, row = 16 * b + r;
            const int col = 16 * bi + cc;
            float x = 0.0f;
            for (int m = 16 * bi; m < 16 * b; m++)
                x = __fmaf_rn(aa[row][m], tt[col][m], x);
            xt[bi][r][cc] = x;
        }
        __syncthreads();
        for (int e = tid; e < nb * 256; e += PT) {
            const int bi = e / 256, r = (e / 16) % 16, cc = e % 16, b = bi + dist;
            float x = 0.0f;
            for (int m = 0; m <= r; m++)
                x = __fmaf_rn(tt[16 * b + m][16 * b + r], xt[bi][m][cc], x);
            tt[16 * bi + cc][16 * b + r] = -x;
        }
        __syncthreads();
    }

    /* the values: a bf16 chunk is 16 KiB, A's place now that T is built;
     * they load while W and Q'' are computed */
    const bool vstage = vdt == ATH_DTYPE_BF16 && dv % 8 == 0;
    uint16_t *vs = (uint16_t *)&aa[0][0];
    if (vstage) {
        for (int e = tid; e < GC * GD / 8; e += PT) {
            const int i = e / (GD / 8), d = 8 * (e % (GD / 8));
            const bool ok = i < n && d < dv;
            const uint16_t *src = (const uint16_t *)v + ((t0 + i) * nv + h) * dv + d;
            k_cp16(&vs[i * GD + d], ok ? src : (const uint16_t *)v, ok);
        }
        k_cp_commit();
    }

    /* W = T (β e^γ K): scale the key rows in place, then the product; W goes
     * out transposed and stays in x0 for Q'' */
    for (int e = tid; e < GC * GD / 4; e += PT) {
        const int i = e / (GD / 4), d = 4 * (e % (GD / 4));
        float4 *p = (float4 *)&x0[i][d];
        const float f = be[i];
        *p = make_float4(__fmul_rn(p->x, f), __fmul_rn(p->y, f), __fmul_rn(p->z, f), __fmul_rn(p->w, f));
    }
    __syncthreads();
    float o[8][8];
    tri_mul(tt, x0, o);
    tri_store_t(W + out * dk, dk, o);
    __syncthreads();                                   /* every lane has read the scaled keys */
    tri_store(&x0[0][0], XP, GD, o);
    __syncthreads();

    /* Q'' = q_scale e^γ Q − M W */
    tri_mul(mt, x0, o);
    {
        const int ti = lane / 16 + 2 * w, cq = 4 * (lane % 16);
#pragma unroll
        for (int a = 0; a < 8; a++) {
            const int i = 8 * ti + a;
            const float f = __fmul_rn(q_scale, expf(gm[i]));
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                const int d = cq + 64 * hh;
                const float4 qv = i < n && d < dk ? *(const float4 *)&q[((t0 + i) * nk + kh) * dk + d]
                                                  : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                o[a][4 * hh] = __fsub_rn(__fmul_rn(f, qv.x), o[a][4 * hh]);
                o[a][4 * hh + 1] = __fsub_rn(__fmul_rn(f, qv.y), o[a][4 * hh + 1]);
                o[a][4 * hh + 2] = __fsub_rn(__fmul_rn(f, qv.z), o[a][4 * hh + 2]);
                o[a][4 * hh + 3] = __fsub_rn(__fmul_rn(f, qv.w), o[a][4 * hh + 3]);
            }
        }
    }
    tri_store_t(Qp + out * dk, dk, o);
    __syncthreads();                                   /* every lane has read W */

    /* U = T (βV), then O_U = M U */
    if (vstage) {
        k_cp_wait_all();
        __syncthreads();
    }
    for (int e = tid; e < GC * GD / 4; e += PT) {
        const int i = e / (GD / 4), d = 4 * (e % (GD / 4));
        float4 x = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float f = bt[i];
        if (vstage) {
            const uint2 r = *(const uint2 *)&vs[i * GD + d];             /* zeros where padding */
            x = make_float4(__fmul_rn(f, __uint_as_float(r.x << 16)), __fmul_rn(f, __uint_as_float(r.x & 0xffff0000u)),
                            __fmul_rn(f, __uint_as_float(r.y << 16)), __fmul_rn(f, __uint_as_float(r.y & 0xffff0000u)));
        } else if (i < n && d < dv) {
            const int64_t at = ((t0 + i) * nv + h) * dv + d;
            x = make_float4(__fmul_rn(f, k_ld(v, vdt, at)), __fmul_rn(f, k_ld(v, vdt, at + 1)),
                            __fmul_rn(f, k_ld(v, vdt, at + 2)), __fmul_rn(f, k_ld(v, vdt, at + 3)));
        }
        *(float4 *)&x0[i][d] = x;
    }
    __syncthreads();
    tri_mul(tt, x0, o);
    tri_store(U + out * dv, dv, dv, o);
    __syncthreads();
    tri_store(&x0[0][0], XP, GD, o);
    __syncthreads();
    tri_mul(mt, x0, o);
    tri_store(OU + out * dv, dv, dv, o);
}

/* ---- scan: the chunks in order, with the state ------------------------------------- */

/* Four warps. The 64 × 32 products Q'' S and W S: warp w takes rows
 * 16 w … 16 w + 15, a lane a 4 × 4 tile of them, over every d. The state:
 * warp w keeps rows 32 w … 32 w + 31 of the block's 128 × 32 part in
 * registers, 8 × 4 per lane, and updates them. W and Q'' come transposed
 * from prep, [dk][GC], so that a lane's 4 rows at one d are one load. */
#define ST 128                /* threads per scan block                   */
#define TILE (GC * GD)        /* floats in an operand tile                */

/* acc[a][b] = Σ_d x[d][16 w + 4 pr + a] S[d][4 pc + b], pr = lane / 8,
 * pc = lane % 8: rows 16 w + 4 pr + a and columns 4 pc + b of the product. */
static __device__ __forceinline__ void scan_rows(const float *xt, const float (*sm)[GV], int dk, float acc[4][4])
{
    const int lane = threadIdx.x % ATH_WARP, w = threadIdx.x / ATH_WARP, r0 = 16 * w + 4 * (lane / 8);
    const int c0 = 4 * (lane % 8);
#pragma unroll
    for (int a = 0; a < 4; a++)
#pragma unroll
        for (int b = 0; b < 4; b++)
            acc[a][b] = 0.0f;
#pragma unroll 8
    for (int d = 0; d < dk; d++) {
        const float4 x = *(const float4 *)&xt[d * GC + r0], y = *(const float4 *)&sm[d][c0];
        const float av[4] = {x.x, x.y, x.z, x.w}, bv[4] = {y.x, y.y, y.z, y.w};
#pragma unroll
        for (int a = 0; a < 4; a++)
#pragma unroll
            for (int b = 0; b < 4; b++)
                acc[a][b] = __fmaf_rn(av[a], bv[b], acc[a][b]);
    }
}

/* A contiguous [rows][n] block of global memory into a tile, asynchronously. */
static __device__ __forceinline__ void scan_copy(float *x, const float *src, int floats)
{
    for (int e = threadIdx.x; e < floats / 4; e += ST)
        k_cp16(&x[4 * e], src + 4 * e, true);
}

/* The chunk's keys, [GC][dk], from rows of `stride` floats starting at k;
 * rows at or past n zeros. */
static __device__ __forceinline__ void scan_keys(float *x, const float *k, int64_t stride, int dk, int n)
{
    for (int e = threadIdx.x; e < GC * dk / 4; e += ST) {
        const int i = (4 * e) / dk, d = (4 * e) % dk;
        const bool ok = i < n;
        k_cp16(&x[i * dk + d], ok ? k + i * stride + d : k, ok);
    }
}

static __global__ void __launch_bounds__(ST)
gdn_scan(const float *k, const float *Qp, const float *W, const float *U, const float *OU, const float *gam,
         float *state, void *out, int odt, int64_t T, int64_t c0, int64_t nc, int64_t nk, int dk, int64_t nv, int dv)
{
    extern __shared__ __align__(16) unsigned char scan_smem[];
    float *const xb0 = (float *)scan_smem;                              /* two operand tiles, */
#define XB(i) (xb0 + (i) * TILE)                                        /* so the compiler sees shared memory */
    float (*sm)[GV] = (float (*)[GV])(xb0 + 2 * TILE);                   /* S [GD][GV]        */
    float (*dl)[GV] = sm + GD;                                           /* e^(γ_last − γ) Δ [GC][GV] */
    const int lane = threadIdx.x % ATH_WARP, w = threadIdx.x / ATH_WARP, lr = lane / 8, lc = lane % 8;
    const int64_t h = blockIdx.x, kh = h / (nv / nk), j0 = (int64_t)blockIdx.y * GV;
    const int jn = (int)(dv - j0 < GV ? dv - j0 : GV);            /* live columns of this block */
    float *S = state + h * dk * dv;

    /* the state: rows 32 w + 8 lr + a, columns 4 lc + b of this lane */
    float st[8][4];
#pragma unroll
    for (int a = 0; a < 8; a++) {
        const int row = 32 * w + 8 * lr + a;
#pragma unroll
        for (int b = 0; b < 4; b++) {
            const int col = 4 * lc + b;
            st[a][b] = row < dk && col < jn ? S[row * dv + j0 + col] : 0.0f;
        }
        *(float4 *)&sm[row][4 * lc] = make_float4(st[a][0], st[a][1], st[a][2], st[a][3]);
    }

    /* operands in order Q''_c, W_c, K_c, Q''_c+1, ...: two tiles in turn, so
     * that each loads while the one before is used; chunk c has Q'' and K
     * in tile c % 2, W in the other. A consumed tile takes the partial
     * products before the next load. */
    scan_copy(XB(0), Qp + h * GC * dk, GC * dk);
    k_cp_commit();
    for (int64_t ci = 0; ci < nc; ci++) {
        const int b = (int)(ci & 1);
        const int64_t c = c0 + ci, t0 = c * GC, rows = (ci * nv + h) * GC;
        const int n = (int)(T - t0 < GC ? T - t0 : GC);
        const float glast = gam[rows + GC - 1];
        const int r0 = 16 * w + 4 * (lane / 8), cc = 4 * (lane % 8);   /* this lane's rows and columns */
        float acc[4][4], pre[4][4];

        /* O = Q'' S + O_U */
        k_cp_wait_all();
        __syncthreads();
        scan_copy(XB(b ^ 1), W + rows * dk, GC * dk);
        k_cp_commit();
#pragma unroll
        for (int a = 0; a < 4; a++)
#pragma unroll
            for (int q = 0; q < 4; q++)
                pre[a][q] = cc + q < jn ? OU[(rows + r0 + a) * dv + j0 + cc + q] : 0.0f;
        scan_rows(XB(b), sm, dk, acc);
#pragma unroll
        for (int a = 0; a < 4; a++)
#pragma unroll
            for (int q = 0; q < 4; q++)
                if (r0 + a < n && cc + q < jn)
                    k_st(out, odt, ((t0 + r0 + a) * nv + h) * dv + j0 + cc + q, __fadd_rn(acc[a][q], pre[a][q]));

        /* Δ = U − W S, kept as e^(γ_last − γ_i) Δ_i for the update */
        k_cp_wait_all();
        __syncthreads();
        scan_keys(XB(b), k + (t0 * nk + kh) * dk, nk * dk, dk, n);
        k_cp_commit();
#pragma unroll
        for (int a = 0; a < 4; a++)
#pragma unroll
            for (int q = 0; q < 4; q++)
                pre[a][q] = cc + q < jn ? U[(rows + r0 + a) * dv + j0 + cc + q] : 0.0f;
        scan_rows(XB(b ^ 1), sm, dk, acc);
#pragma unroll
        for (int a = 0; a < 4; a++) {
            const float f = expf(__fsub_rn(glast, gam[rows + r0 + a]));
            *(float4 *)&dl[r0 + a][cc] =
                make_float4(__fmul_rn(f, __fsub_rn(pre[a][0], acc[a][0])), __fmul_rn(f, __fsub_rn(pre[a][1], acc[a][1])),
                            __fmul_rn(f, __fsub_rn(pre[a][2], acc[a][2])), __fmul_rn(f, __fsub_rn(pre[a][3], acc[a][3])));
        }

        /* S = e^γ_last S + Kᵀ (e^(γ_last − γ) Δ) */
        k_cp_wait_all();
        __syncthreads();
        if (ci + 1 < nc) {
            scan_copy(XB(b ^ 1), Qp + ((ci + 1) * nv + h) * GC * dk, GC * dk);
            k_cp_commit();
        }
        {
            const float *x = XB(b);
            const float decay = expf(glast);
#pragma unroll
            for (int a = 0; a < 8; a++)
#pragma unroll
                for (int bb = 0; bb < 4; bb++)
                    st[a][bb] = __fmul_rn(st[a][bb], decay);
            const int r0 = 32 * w + 8 * lr;
            if (r0 < dk) {
#pragma unroll 4
                for (int i = 0; i < GC; i++) {
                    const float4 k0 = *(const float4 *)&x[i * dk + r0], k1 = *(const float4 *)&x[i * dk + r0 + 4];
                    const float4 dd = *(const float4 *)&dl[i][4 * lc];
                    const float kv[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
                    const float dv4[4] = {dd.x, dd.y, dd.z, dd.w};
#pragma unroll
                    for (int a = 0; a < 8; a++)
#pragma unroll
                        for (int bb = 0; bb < 4; bb++)
                            st[a][bb] = __fmaf_rn(kv[a], dv4[bb], st[a][bb]);
                }
            }
        }
        __syncthreads();                               /* every lane has read S and Δ */
#pragma unroll
        for (int a = 0; a < 8; a++)
            *(float4 *)&sm[32 * w + 8 * lr + a][4 * lc] = make_float4(st[a][0], st[a][1], st[a][2], st[a][3]);
    }
#pragma unroll
    for (int a = 0; a < 8; a++) {
        const int row = 32 * w + 8 * lr + a;
#pragma unroll
        for (int b = 0; b < 4; b++) {
            const int col = 4 * lc + b;
            if (row < dk && col < jn)
                S[row * dv + j0 + col] = st[a][b];
        }
    }
}

#undef XB

#define PREP_SMEM (PREP_FLOATS * sizeof(float))
#define SCAN_SMEM (2 * TILE * sizeof(float) + sizeof(float[GD + GC][GV]))

/* Scratch for one round: W and Q'' [chunks, nv, GC, dk], U and O_U
 * [chunks, nv, GC, dv], γ [chunks, nv, GC]. */
static size_t round_floats(int64_t chunks, int64_t nv, int64_t dk, int64_t dv)
{
    return (size_t)(chunks * nv * GC * (2 * dk + 2 * dv + 1));
}

extern "C" bool ath_gdn_chunk_fits(const ath_device *d, const ath_tensor *q, const ath_tensor *k,
                                   const ath_tensor *v, const ath_tensor *g)
{
    const int64_t dk = q->shape[2], dv = v->shape[2];
    return d->smem_per_block >= PREP_SMEM && d->smem_per_block >= SCAN_SMEM && q->dtype == ATH_DTYPE_F32 &&
           k->dtype == ATH_DTYPE_F32 && g->dtype == ATH_DTYPE_F32 && dk <= GD && dv <= GD && dk % 4 == 0 &&
           dv % 4 == 0 && dk > 0 && dv > 0 && (uintptr_t)q->data % 16 == 0 && (uintptr_t)k->data % 16 == 0;
}

/* Chunks per round: enough prep blocks for GDN_WAVES full waves over the
 * device's multiprocessors, all of the call's chunks at most. */
static int64_t round_chunks(const ath_device *d, int64_t nc, int64_t nv)
{
    int64_t r = ((int64_t)GDN_WAVES * d->n_sm + nv - 1) / nv;
    r = r < 1 ? 1 : r;
    return nc < r ? nc : r;
}

extern "C" size_t ath_gdn_chunk_bytes(const ath_device *d, const ath_tensor *q, const ath_tensor *v)
{
    const int64_t nc = (q->shape[0] + GC - 1) / GC, nv = v->shape[1];
    return round_floats(round_chunks(d, nc, nv), nv, q->shape[2], v->shape[2]) * sizeof(float);
}

extern "C" ath_status ath_gdn_chunk_launch(const ath_device *d, const ath_tensor *q, const ath_tensor *k,
                                           const ath_tensor *v, const ath_tensor *g, const ath_tensor *beta,
                                           float q_scale, ath_tensor *state, void *work, ath_tensor *out)
{
    const int64_t T = q->shape[0], nk = q->shape[1], dk = q->shape[2], nv = v->shape[1], dv = v->shape[2];
    const int64_t nc = (T + GC - 1) / GC;
    if (T == 0 || nv == 0)
        return ATH_OK;
    if (nv > 65535) {
        ath_log(ATH_LOG_ERROR, "gdn: %lld value heads, more than one launch holds", (long long)nv);
        return ATH_ERR_UNSUPPORTED;
    }
    ATH_CHECK(ath_gpu_use(d));
    if (cudaFuncSetAttribute(gdn_prep, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PREP_SMEM) != cudaSuccess ||
        cudaFuncSetAttribute(gdn_scan, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SCAN_SMEM) != cudaSuccess)
        return ath_gpu_launched("gdn chunks (shared memory)");
    const int64_t rc = round_chunks(d, nc, nv);
    float *Wb = (float *)work, *Qb = Wb + rc * nv * GC * dk, *Ub = Qb + rc * nv * GC * dk;
    float *Ob = Ub + rc * nv * GC * dv, *gb = Ob + rc * nv * GC * dv;
    for (int64_t c0 = 0; c0 < nc; c0 += rc) {
        const int64_t n = nc - c0 < rc ? nc - c0 : rc;
        gdn_prep<<<dim3((unsigned)n, (unsigned)nv), PT, PREP_SMEM, k_stream(d)>>>(
            (const float *)q->data, (const float *)k->data, v->data, v->dtype, (const float *)g->data, beta->data,
            beta->dtype, q_scale, T, c0, nk, (int)dk, nv, (int)dv, Qb, Wb, Ub, Ob, gb);
        ATH_CHECK(ath_gpu_launched("gdn_prep"));
        gdn_scan<<<dim3((unsigned)nv, (unsigned)((dv + GV - 1) / GV)), ST, SCAN_SMEM, k_stream(d)>>>(
            (const float *)k->data, Qb, Wb, Ub, Ob, gb, (float *)state->data, out->data, out->dtype, T, c0, n, nk,
            (int)dk, nv, (int)dv);
        ATH_CHECK(ath_gpu_launched("gdn_scan"));
    }
    return ATH_OK;
}
