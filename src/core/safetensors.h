/* safetensors.h: zero-copy reader for safetensors checkpoints.
 *
 * File layout (https://github.com/huggingface/safetensors):
 *
 *   [8 bytes]  N, little-endian uint64: length of the header
 *   [N bytes]  JSON object, UTF-8, possibly padded with spaces:
 *                "__metadata__": {string: string}            (optional)
 *                "<tensor name>": {"dtype": "BF16",
 *                                  "shape": [248320, 1024],
 *                                  "data_offsets": [begin, end]}
 *   [rest]     data section; offsets are relative to its start
 *
 * Every file is mapped with mmap and each tensor becomes an ath_tensor whose
 * data pointer points straight into the mapping: opening a checkpoint copies
 * no weights. A checkpoint split into shards has a
 * model.safetensors.index.json whose "weight_map" names the shard of every
 * tensor; ath_safetensors_open handles both layouts.
 *
 * Validation follows the reference implementation: dtypes and shapes must be
 * known and consistent with the byte ranges, ranges must lie inside the data
 * section, and the tensors of a shard must cover its data section exactly,
 * without gaps or overlaps. For sharded checkpoints the index and the shard
 * headers must agree tensor by tensor.
 */
#ifndef ATH_SAFETENSORS_H
#define ATH_SAFETENSORS_H

#include "core/arena.h"
#include "core/file.h"
#include "core/json.h"
#include "core/tensor.h"

typedef struct {
    const char     *path;
    ath_file        file;
    uint64_t        header_bytes;   /* N                                     */
    uint64_t        data_offset;    /* 8 + N: file offset of the data section */
    uint64_t        data_bytes;     /* file size - data_offset               */
    const ath_json *header;         /* parsed header, including __metadata__ */
} ath_safetensors_shard;

typedef struct {
    const char *name;
    ath_tensor  t;        /* view into the shard mapping (read-only memory) */
    uint32_t    shard;    /* index into ath_safetensors.shards              */
    uint64_t    begin;    /* data_offsets, relative to the data section     */
    uint64_t    end;
} ath_safetensors_entry;

typedef struct {
    ath_safetensors_shard *shards;            /* sorted by file name          */
    uint32_t               n_shards;
    ath_safetensors_entry *tensors;           /* sorted by name, see below    */
    uint32_t               n_tensors;
    uint64_t               tensor_bytes;      /* sum of end - begin           */
    uint64_t               data_bytes;        /* sum of shard data sections   */
    int64_t                index_total_size;  /* index metadata.total_size, -1 if absent */
} ath_safetensors;

/* Open the checkpoint in dir: model.safetensors.index.json and its shards if
 * the index exists, otherwise model.safetensors. Names, headers and tables
 * are allocated in a, which must outlive st. On failure everything opened so
 * far is closed. ATH_ERR_PARSE for a malformed header, ATH_ERR_FORMAT for an
 * inconsistent one, ATH_ERR_UNSUPPORTED for an unknown dtype. */
ath_status ath_safetensors_open(ath_safetensors *st, ath_arena *a, const char *dir);

/* Open a single .safetensors file. */
ath_status ath_safetensors_open_file(ath_safetensors *st, ath_arena *a, const char *path);

/* Tensor by exact name, binary search; NULL if absent. */
const ath_safetensors_entry *ath_safetensors_find(const ath_safetensors *st, const char *name);

/* Unmap every shard. Tensor views become invalid. */
void ath_safetensors_close(ath_safetensors *st);

/* Order used for st->tensors: byte order, except that runs of digits compare
 * as numbers, so "layers.2" sorts before "layers.10". */
int ath_safetensors_name_cmp(const char *a, const char *b);

#endif /* ATH_SAFETENSORS_H */
