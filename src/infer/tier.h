/* tier.h: where the routed experts of a MoE model live (step 12).
 *
 * A model larger than the GPU keeps its routed experts in RAM and a share
 * of them in device memory. Qwen3.6-35B-A3B in Q8_0 has 10,240 experts of
 * 3.34 MB (34.2 GB) beside 2.6 GB of other weights: the other weights go to
 * the device (the engine's copies), every expert is read once into host
 * memory, regrouped for the CPU (ATH_DTYPE_Q8_0_R16), and as many as the
 * device's memory allows get a slot there, the same number in every MoE
 * layer. A forward pass computes the experts in slots on the GPU and the
 * others on the CPU, with the same bits (src/ops/moe.c; invariant 7).
 *
 * The slots follow use: the CPU's share counts how often each expert is
 * chosen, and between passes (ath_place_update) the most used experts of a
 * layer move into its slots, a few at a time. A move copies the expert into
 * the slot over the bus and then changes two tables, the one in device
 * memory that the kernels read and the host flags the CPU reads; no pass
 * is running then, so a captured decode step simply sees the new places.
 *
 * Only Q8_0 experts are split between the GPU and the CPU: their two paths
 * share one arithmetic order (src/ops/q8_0.h). Experts in another dtype
 * must all fit in device memory (ATH_ERR_UNSUPPORTED otherwise).
 *
 * Pipeline code: devices, memory, copies through backend.h, computation
 * through ops.h (invariant 11). Which params are experts, and of which
 * layer, comes from the contract (roles, layer, expert), never from names.
 */
#ifndef ATH_TIER_H
#define ATH_TIER_H

#include "backend/backend.h"
#include "model/model.h"
#include "ops/ops.h"

typedef struct {
    uint64_t vram_reserve;   /* device memory left free for states and working memory;
                              * 0: ATH_PLACE_RESERVE                                        */
    int32_t  slots;          /* experts per layer in device memory; -1: as many as fit     */
    int64_t  cpu_tokens;     /* tokens per chunk of the CPU's share; 0: 512                */
    int32_t  period;         /* passes between two updates of the slots; 0: 4              */
    int32_t  moves;          /* moves per layer and update at most; 0: 2                   */
    bool     draft;          /* the draft head too (step 13): the engine copies its params
                              * and places its layers' experts like the others             */
} ath_place_opts;

#define ATH_PLACE_RESERVE ((uint64_t)2 << 30)   /* 2 GiB */

typedef struct {
    const ath_model  *m;
    const ath_device *d;
    ath_place_opts     o;
    int32_t           n_layers, n_experts, slots;   /* slots per MoE layer                  */
    int64_t           inter, hidden;
    uint64_t          expert_bytes;                 /* one expert's three matrices, Q8_0     */
    ath_moe_experts  *layers;                       /* [n_layers], for ath_exec.experts      */
    const void      **host_tab;                     /* [n_layers][3 n]                       */
    void            **dev_host;                     /* [n_layers][3 n]: host mirror of dev   */
    void             *dev_tab;                      /* device [n_layers][3 n]                */
    uint8_t          *on_dev;                       /* [n_layers][n]                         */
    uint32_t         *uses;                         /* [n_layers][n]                         */
    int32_t          *slot_of;                      /* [n_layers][n]: slot or -1             */
    int32_t          *in_slot;                      /* [n_layers][slots]: expert or -1       */
    void             *slot_mem;                     /* device: [n_layers][slots] experts     */
    void             *host_mem;                     /* the regrouped host copies (mmap)      */
    size_t            host_bytes;
    void             *staging;                      /* pinned: one expert in Q8_0, for moves */
    ath_pool          pool;                         /* the CPU's threads                     */
    ath_moe_ws       *ws;
    int64_t           passes;                       /* updates seen, for the period          */
    uint64_t          moved;                        /* experts moved into slots so far       */
} ath_placement;

/* Place the routed experts of m on device d: every expert read into host
 * memory (regrouped for the CPU when Q8_0), slots in device memory for as
 * many as fit after the reserve (or opts->slots), filled with the first
 * experts of each layer. Bookkeeping in a, which must outlive the tier.
 * ATH_ERR_UNSUPPORTED for a model without routed experts. */
ath_status ath_place_open(ath_placement *t, const ath_model *m, const ath_device *d, const ath_place_opts *opts,
                         ath_arena *a);

/* Between passes, with no work of a pass left on the device: every
 * opts->period calls, move the most chosen experts of each layer into its
 * slots, at most opts->moves per layer, and age the counts. */
ath_status ath_place_update(ath_placement *t);

/* Wait for the device, free the slots, the host copies and the threads. */
void ath_place_close(ath_placement *t);

#endif /* ATH_TIER_H */
