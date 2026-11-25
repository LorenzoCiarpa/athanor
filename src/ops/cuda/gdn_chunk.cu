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
 * at most GDN_ROUND chunks, so that the scratch memory (W, Q'', U, O_U and γ
 * for a round) stays bounded whatever the prompt's length.
 */
#include "ops/cuda/mma.cuh"

#define GC 64                 /* tokens per chunk                         */
#define GD 128                /* largest dk and dv                        */
#define GDP (GD + 4)          /* padded row: rows 4 banks apart           */
#define GT 256                /* threads per block                        */
#define GV 32                 /* state columns per scan block             */
#define GDN_ROUND 8           /* chunks per round                         */

/* ---- prep: W, Q'', U, O_U and γ of one chunk and value head ----------------------- */

/* out[i][c] = Σ_{j ≤ i} a[i][j] x[j][c], with a[i][i] taken as 1 when unit
 * (T, whose diagonal is not stored): rows ti + 16 r, columns 4 tj + 64 s of
 * a [GC][GC + 1] and x [GC][GDP] matrix in shared memory, into the thread's
 * 4 × 8 tile. */
static __device__ __forceinline__ void lower_mul(const float (*a)[GC + 1], const float (*x)[GDP], bool unit,
                                                 float out[4][8])
{
    const int ti = threadIdx.x / 16, tj = threadIdx.x % 16;
#pragma unroll
    for (int r = 0; r < 4; r++)
#pragma unroll
        for (int e = 0; e < 8; e++)
            out[r][e] = 0.0f;
    #pragma unroll 4
    for (int j = 0; j < GC; j++) {
        const float4 x0 = *(const float4 *)&x[j][4 * tj], x1 = *(const float4 *)&x[j][4 * tj + 64];
#pragma unroll
        for (int r = 0; r < 4; r++) {
            if (j >= 16 * (r + 1))
                continue;                          /* past every row of this block: a[i][j] = 0 */
            const int i = ti + 16 * r;
            const float w = j < i ? a[i][j] : j > i ? 0.0f : unit ? 1.0f : a[i][i];
            out[r][0] = __fmaf_rn(w, x0.x, out[r][0]);
            out[r][1] = __fmaf_rn(w, x0.y, out[r][1]);
            out[r][2] = __fmaf_rn(w, x0.z, out[r][2]);
            out[r][3] = __fmaf_rn(w, x0.w, out[r][3]);
            out[r][4] = __fmaf_rn(w, x1.x, out[r][4]);
            out[r][5] = __fmaf_rn(w, x1.y, out[r][5]);
            out[r][6] = __fmaf_rn(w, x1.z, out[r][6]);
            out[r][7] = __fmaf_rn(w, x1.w, out[r][7]);
        }
    }
}

/* The thread's 4 × 8 tile into rows of n columns (columns past n dropped). */
static __device__ __forceinline__ void tile_store(float *dst, int64_t stride, int n, const float t[4][8])
{
    const int ti = threadIdx.x / 16, tj = threadIdx.x % 16;
#pragma unroll
    for (int r = 0; r < 4; r++)
#pragma unroll
        for (int s = 0; s < 2; s++) {
            const int c = 4 * tj + 64 * s;
            if (c < n)
                *(float4 *)&dst[(ti + 16 * r) * stride + c] =
                    make_float4(t[r][4 * s], t[r][4 * s + 1], t[r][4 * s + 2], t[r][4 * s + 3]);
        }
}

static __device__ __forceinline__ void tile_to_smem(float (*x)[GDP], const float t[4][8])
{
    tile_store(&x[0][0], GDP, GD, t);
}

static __global__ void __launch_bounds__(GT)
gdn_prep(const float *q, const float *k, const void *v, int vdt, const float *g, const void *beta, int bdt,
         float q_scale, int64_t T, int64_t c0, int64_t nk, int dk, int64_t nv, int dv, float *Qp, float *W,
         float *U, float *OU, float *gam)
{
    extern __shared__ __align__(16) unsigned char prep_smem[];
    float (*xs)[GDP] = (float (*)[GDP])prep_smem;                  /* k rows, then W, then βv, then U */
    float (*ta)[GC + 1] = (float (*)[GC + 1])(prep_smem + sizeof(float[GC][GDP]));   /* A */
    float (*tt)[GC + 1] = ta + GC;                                  /* T                               */
    float (*ma)[GC + 1] = tt + GC;                                  /* M                               */
    float *gm = &ma[GC][0], *bt = gm + GC, *be = bt + GC;           /* γ, β, β e^γ                     */
    float (*xt)[16][16] = (float (*)[16][16])(be + GC);             /* blocks of T being built         */
    const int tid = threadIdx.x, ti = tid / 16, tj = tid % 16;
    const int64_t c = c0 + blockIdx.x, h = blockIdx.y, kh = h / (nv / nk), t0 = c * GC;
    const int n = (int)(T - t0 < GC ? T - t0 : GC);
    const int64_t out = ((int64_t)blockIdx.x * nv + h) * GC;     /* this chunk's rows in the round's scratch */

    /* γ (in token order, as the reference's cumsum), β; rows past n are padding: zeros, γ constant */
    if (tid < GC) {
        bt[tid] = tid < n ? k_ld(beta, bdt, (t0 + tid) * nv + h) : 0.0f;
        gm[tid] = tid < n ? g[(t0 + tid) * nv + h] : 0.0f;
    }
    __syncthreads();
    if (tid == 0)
        for (int i = 1; i < GC; i++)
            gm[i] = __fadd_rn(gm[i - 1], gm[i]);
    /* the k rows in xs; the q rows, until the dot products are done, where
     * A, T and M will go */
    float (*qs)[GDP] = (float (*)[GDP])&ta[0][0];
    for (int e = tid; e < GC * GD / 4; e += GT) {
        const int i = e / (GD / 4), d = 4 * (e % (GD / 4));
        const bool ok = i < n && d < dk;
        const int64_t at = ((t0 + i) * nk + kh) * dk + d;
        *(float4 *)&xs[i][d] = ok ? *(const float4 *)&k[at] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        *(float4 *)&qs[i][d] = ok ? *(const float4 *)&q[at] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    __syncthreads();
    if (tid < GC) {
        gam[out + tid] = gm[tid];
        be[tid] = __fmul_rn(bt[tid], expf(gm[tid]));
    }

    /* k_i · k_j and q_i · k_j: rows ti + 16 r, columns tj + 16 s */
    float kk[4][4], qk[4][4];
#pragma unroll
    for (int r = 0; r < 4; r++)
#pragma unroll
        for (int s = 0; s < 4; s++)
            kk[r][s] = qk[r][s] = 0.0f;
    #pragma unroll 4
    for (int d = 0; d < dk; d += 4) {
        float4 ki[4], kj[4], qi[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
            const int i = ti + 16 * r;
            ki[r] = *(const float4 *)&xs[i][d];
            kj[r] = *(const float4 *)&xs[tj + 16 * r][d];
            qi[r] = *(const float4 *)&qs[i][d];
        }
#pragma unroll
        for (int r = 0; r < 4; r++)
#pragma unroll
            for (int s = 0; s <= r; s++) {          /* blocks on or below the diagonal only */
                kk[r][s] = __fmaf_rn(ki[r].x, kj[s].x, kk[r][s]);
                kk[r][s] = __fmaf_rn(ki[r].y, kj[s].y, kk[r][s]);
                kk[r][s] = __fmaf_rn(ki[r].z, kj[s].z, kk[r][s]);
                kk[r][s] = __fmaf_rn(ki[r].w, kj[s].w, kk[r][s]);
                qk[r][s] = __fmaf_rn(qi[r].x, kj[s].x, qk[r][s]);
                qk[r][s] = __fmaf_rn(qi[r].y, kj[s].y, qk[r][s]);
                qk[r][s] = __fmaf_rn(qi[r].z, kj[s].z, qk[r][s]);
                qk[r][s] = __fmaf_rn(qi[r].w, kj[s].w, qk[r][s]);
            }
    }
    __syncthreads();                                   /* the q rows are read: A and M replace them */
#pragma unroll
    for (int r = 0; r < 4; r++)
#pragma unroll
        for (int s = 0; s < 4; s++) {
            const int i = ti + 16 * r, j = tj + 16 * s;
            const float decay = j <= i ? expf(__fsub_rn(gm[i], gm[j])) : 0.0f;
            ta[i][j] = j < i ? __fmul_rn(__fmul_rn(bt[i], kk[r][s]), decay) : 0.0f;
            ma[i][j] = j <= i ? __fmul_rn(__fmul_rn(q_scale, qk[r][s]), decay) : 0.0f;
        }
    __syncthreads();

    /* T = (I + A)⁻¹ in blocks of 16 (FLA's solve_tril does the same). The
     * diagonal blocks D_b = (I + A_bb)⁻¹ by forward substitution, a column
     * per thread: for j < i, T_ij = −A_ij − Σ_{j<m<i} A_im T_mj, which needs
     * only the column's own entries above. Then the blocks below, nearest
     * the diagonal first: T_bc = −D_b Σ_{c≤e<b} A_be T_ec. T is stored with
     * its diagonal (1) and zeros above it. */
    for (int e = tid; e < GC * GC; e += GT)
        tt[e / GC][e % GC] = e / GC == e % GC ? 1.0f : 0.0f;
    __syncthreads();
    if (tid < GC) {
        const int b0 = tid & ~15, j = tid;
        for (int i = j + 1; i < b0 + 16; i++) {
            float s = -ta[i][j];
            for (int m = j + 1; m < i; m++)
                s = __fmaf_rn(-ta[i][m], tt[m][j], s);
            tt[i][j] = s;
        }
    }
    __syncthreads();
    for (int dist = 1; dist < GC / 16; dist++) {
        /* X_bc = Σ_{c≤e<b} A_be T_ec for the blocks at this distance, one
         * element per thread and block */
        const int nb = GC / 16 - dist, r = tid / 16, cc = tid % 16;
        for (int bi = 0; bi < nb; bi++) {
            const int b = bi + dist, c = bi, row = 16 * b + r, col = 16 * c + cc;
            float x = 0.0f;
            for (int m = 16 * c; m < 16 * b; m++)
                x = __fmaf_rn(ta[row][m], tt[m][col], x);
            xt[bi][r][cc] = x;
        }
        __syncthreads();
        for (int bi = 0; bi < nb; bi++) {
            const int b = bi + dist, c = bi, row = 16 * b + r, col = 16 * c + cc;
            float x = 0.0f;
            for (int m = 0; m <= r; m++)
                x = __fmaf_rn(tt[16 * b + r][16 * b + m], xt[bi][m][cc], x);
            tt[row][col] = -x;
        }
        __syncthreads();
    }

    /* W = T (β e^γ K): scale the k rows in place, then the product */
    for (int e = tid; e < GC * GD; e += GT) {
        const int i = e / GD, d = e % GD;
        xs[i][d] = __fmul_rn(xs[i][d], be[i]);
    }
    __syncthreads();
    float acc[4][8];
    lower_mul(tt, xs, true, acc);
    tile_store(W + out * dk, dk, dk, acc);
    __syncthreads();                                   /* every thread has read the k rows */
    tile_to_smem(xs, acc);
    __syncthreads();

    /* Q'' = q_scale e^γ Q − M W */
    lower_mul(ma, xs, false, acc);
#pragma unroll
    for (int r = 0; r < 4; r++) {
        const int i = ti + 16 * r;
        const float f = __fmul_rn(q_scale, expf(gm[i]));
#pragma unroll
        for (int s = 0; s < 2; s++) {
            const int d = 4 * tj + 64 * s;
            const float4 qv = i < n && d < dk ? *(const float4 *)&q[((t0 + i) * nk + kh) * dk + d]
                                              : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            acc[r][4 * s] = __fsub_rn(__fmul_rn(f, qv.x), acc[r][4 * s]);
            acc[r][4 * s + 1] = __fsub_rn(__fmul_rn(f, qv.y), acc[r][4 * s + 1]);
            acc[r][4 * s + 2] = __fsub_rn(__fmul_rn(f, qv.z), acc[r][4 * s + 2]);
            acc[r][4 * s + 3] = __fsub_rn(__fmul_rn(f, qv.w), acc[r][4 * s + 3]);
        }
    }
    tile_store(Qp + out * dk, dk, dk, acc);
    __syncthreads();                                   /* every thread has read W */

    /* U = T (βV), then O_U = M U */
    for (int e = tid; e < GC * GD; e += GT) {
        const int i = e / GD, d = e % GD;
        xs[i][d] = i < n && d < dv ? __fmul_rn(bt[i], k_ld(v, vdt, ((t0 + i) * nv + h) * dv + d)) : 0.0f;
    }
    __syncthreads();
    lower_mul(tt, xs, true, acc);
    tile_store(U + out * dv, dv, dv, acc);
    __syncthreads();
    tile_to_smem(xs, acc);
    __syncthreads();
    lower_mul(ma, xs, false, acc);
    tile_store(OU + out * dv, dv, dv, acc);
}

/* ---- scan: the chunks in order, with the state ------------------------------------- */

/* Rows of a [GC][n] matrix into a padded shared tile, asynchronously; rows
 * at or past `rows` zeros. */
static __device__ __forceinline__ void scan_load(float (*x)[GDP], const float *src, int64_t stride, int n, int rows)
{
    for (int e = threadIdx.x; e < GC * (GD / 4); e += GT) {
        const int i = e / (GD / 4), d = 4 * (e % (GD / 4));
        if (d < n) {
            const bool ok = i < rows;
            k_cp16(&x[i][d], ok ? src + i * stride + d : src, ok);
        }
    }
}

/* o[r][e] = Σ_d x[ti + 16 r][d] S[d][2 tj + e] over d < dk: a 64 × dk tile
 * times the state's 32 columns, the thread's 4 × 2 outputs. */
static __device__ __forceinline__ void scan_mul(const float (*x)[GDP], const float (*sm)[GV], int dk, float o[4][2])
{
    const int ti = threadIdx.x / 16, tj = threadIdx.x % 16;
#pragma unroll
    for (int r = 0; r < 4; r++)
        o[r][0] = o[r][1] = 0.0f;
    #pragma unroll 4
    for (int d = 0; d < dk; d += 4) {
        float4 a[4];
        float2 b[4];
#pragma unroll
        for (int r = 0; r < 4; r++)
            a[r] = *(const float4 *)&x[ti + 16 * r][d];
#pragma unroll
        for (int u = 0; u < 4; u++)
            b[u] = *(const float2 *)&sm[d + u][2 * tj];
#pragma unroll
        for (int r = 0; r < 4; r++) {
            const float av[4] = {a[r].x, a[r].y, a[r].z, a[r].w};
#pragma unroll
            for (int u = 0; u < 4; u++) {
                o[r][0] = __fmaf_rn(av[u], b[u].x, o[r][0]);
                o[r][1] = __fmaf_rn(av[u], b[u].y, o[r][1]);
            }
        }
    }
}

static __global__ void __launch_bounds__(GT)
gdn_scan(const float *k, const float *Qp, const float *W, const float *U, const float *OU, const float *gam,
         float *state, void *out, int odt, int64_t T, int64_t c0, int64_t nc, int64_t nk, int dk, int64_t nv, int dv)
{
    extern __shared__ __align__(16) unsigned char scan_smem[];
    float (*xb)[GC][GDP] = (float (*)[GC][GDP])scan_smem;              /* two operand tiles */
    float (*sm)[GV] = (float (*)[GV])(scan_smem + 2 * sizeof(float[GC][GDP]));   /* S [GD][GV]        */
    float (*dl)[GV] = sm + GD;                                          /* e^(γ_last − γ) Δ [GC][GV] */
    const int tid = threadIdx.x, ti = tid / 16, tj = tid % 16;
    const int64_t h = blockIdx.x, kh = h / (nv / nk), j0 = (int64_t)blockIdx.y * GV;
    const int jn = (int)(dv - j0 < GV ? dv - j0 : GV);            /* live columns of this block */
    float *S = state + h * dk * dv;

    /* this thread's part of the state for the update: rows 8 ti … 8 ti + 7, columns 2 tj, 2 tj + 1 */
    float st[8][2];
#pragma unroll
    for (int r = 0; r < 8; r++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int row = 8 * ti + r, col = 2 * tj + e;
            st[r][e] = row < dk && col < jn ? S[row * dv + j0 + col] : 0.0f;
            sm[row][col] = st[r][e];
        }

    /* operands in order Q''_c, W_c, K_c, Q''_c+1, ...: two tiles in turn, so
     * that each loads while the one before is used; chunk c has Q'' and K
     * in tile c % 2, W in the other */
    scan_load(xb[0], Qp + h * GC * dk, dk, dk, GC);
    k_cp_commit();
    for (int64_t ci = 0; ci < nc; ci++) {
        const int b = (int)(ci & 1);
        const int64_t c = c0 + ci, t0 = c * GC, rows = (ci * nv + h) * GC;
        const int n = (int)(T - t0 < GC ? T - t0 : GC);
        const float glast = gam[rows + GC - 1];

        /* O = Q'' S + O_U */
        k_cp_wait_all();
        __syncthreads();
        scan_load(xb[b ^ 1], W + rows * dk, dk, dk, GC);
        k_cp_commit();
        {
            float o[4][2];
            scan_mul(xb[b], sm, dk, o);
#pragma unroll
            for (int r = 0; r < 4; r++) {
                const int i = ti + 16 * r;
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int col = 2 * tj + e;
                    if (i < n && col < jn)
                        k_st(out, odt, ((t0 + i) * nv + h) * dv + j0 + col,
                             __fadd_rn(o[r][e], OU[(rows + i) * dv + j0 + col]));
                }
            }
        }

        /* Δ = U − W S, kept as e^(γ_last − γ_i) Δ_i for the update */
        k_cp_wait_all();
        __syncthreads();
        scan_load(xb[b], k + (t0 * nk + kh) * dk, nk * dk, dk, n);
        k_cp_commit();
        {
            float o[4][2];
            scan_mul(xb[b ^ 1], sm, dk, o);
#pragma unroll
            for (int r = 0; r < 4; r++) {
                const int i = ti + 16 * r;
                const float f = expf(__fsub_rn(glast, gam[rows + i]));
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int col = 2 * tj + e;
                    const float u = col < jn ? U[(rows + i) * dv + j0 + col] : 0.0f;
                    dl[i][col] = __fmul_rn(f, __fsub_rn(u, o[r][e]));
                }
            }
        }

        /* S = e^γ_last S + Kᵀ (e^(γ_last − γ) Δ) */
        k_cp_wait_all();
        __syncthreads();
        if (ci + 1 < nc) {
            scan_load(xb[b ^ 1], Qp + ((ci + 1) * nv + h) * GC * dk, dk, dk, GC);
            k_cp_commit();
        }
        {
            const float (*x)[GDP] = xb[b];
            const float decay = expf(glast);
#pragma unroll
            for (int r = 0; r < 8; r++) {
                st[r][0] = __fmul_rn(st[r][0], decay);
                st[r][1] = __fmul_rn(st[r][1], decay);
            }
            #pragma unroll 4
            for (int i = 0; i < GC; i++) {
                const float4 a0 = *(const float4 *)&x[i][8 * ti], a1 = *(const float4 *)&x[i][8 * ti + 4];
                const float2 bb = *(const float2 *)&dl[i][2 * tj];
                const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
                for (int r = 0; r < 8; r++) {
                    st[r][0] = __fmaf_rn(av[r], bb.x, st[r][0]);
                    st[r][1] = __fmaf_rn(av[r], bb.y, st[r][1]);
                }
            }
        }
        __syncthreads();                               /* every thread has read S and Δ */
#pragma unroll
        for (int r = 0; r < 8; r++) {
            sm[8 * ti + r][2 * tj] = st[r][0];
            sm[8 * ti + r][2 * tj + 1] = st[r][1];
        }
    }
#pragma unroll
    for (int r = 0; r < 8; r++)
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int row = 8 * ti + r, col = 2 * tj + e;
            if (row < dk && col < jn)
                S[row * dv + j0 + col] = st[r][e];
        }
}

#define PREP_SMEM (sizeof(float[GC][GDP]) + sizeof(float[3 * GC][GC + 1]) + 3 * GC * sizeof(float) + \
                   sizeof(float[GC / 16 - 1][16][16]))
#define SCAN_SMEM (2 * sizeof(float[GC][GDP]) + sizeof(float[GD + GC][GV]))

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

extern "C" size_t ath_gdn_chunk_bytes(const ath_tensor *q, const ath_tensor *v)
{
    const int64_t T = q->shape[0], nc = (T + GC - 1) / GC;
    return round_floats(nc < GDN_ROUND ? nc : GDN_ROUND, v->shape[1], q->shape[2], v->shape[2]) * sizeof(float);
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
    const int64_t rc = nc < GDN_ROUND ? nc : GDN_ROUND;
    float *Wb = (float *)work, *Qb = Wb + rc * nv * GC * dk, *Ub = Qb + rc * nv * GC * dk;
    float *Ob = Ub + rc * nv * GC * dv, *gb = Ob + rc * nv * GC * dv;
    for (int64_t c0 = 0; c0 < nc; c0 += GDN_ROUND) {
        const int64_t n = nc - c0 < GDN_ROUND ? nc - c0 : GDN_ROUND;
        gdn_prep<<<dim3((unsigned)n, (unsigned)nv), GT, PREP_SMEM, k_stream(d)>>>(
            (const float *)q->data, (const float *)k->data, v->data, v->dtype, (const float *)g->data, beta->data,
            beta->dtype, q_scale, T, c0, nk, (int)dk, nv, (int)dv, Qb, Wb, Ub, Ob, gb);
        ATH_CHECK(ath_gpu_launched("gdn_prep"));
        gdn_scan<<<dim3((unsigned)nv, (unsigned)((dv + GV - 1) / GV)), GT, SCAN_SMEM, k_stream(d)>>>(
            (const float *)k->data, Qb, Wb, Ub, Ob, gb, (float *)state->data, out->data, out->dtype, T, c0, n, nk,
            (int)dk, nv, (int)dv);
        ATH_CHECK(ath_gpu_launched("gdn_scan"));
    }
    return ATH_OK;
}
