/* ckpt.h: a checkpoint, the named tensors of a model directory.
 *
 * Two formats hold weights: safetensors (Hugging Face's, one file or shards
 * with an index: core/safetensors.h) and GGUF (llama.cpp's, one file:
 * core/gguf.h). ath_ckpt opens whichever a directory holds and presents one
 * table of named tensors, sorted by name, each a zero-copy view of the
 * mapping. Families bind params by name from this table (src/model/model.h);
 * tensor names are the format's own, so a family that reads both knows both
 * naming schemes.
 *
 * A directory holds model.safetensors.index.json with its shards, or
 * model.safetensors, or exactly one .gguf file. Files named mmproj-*.gguf (a
 * vision projector, as llama.cpp's converters write it) are not the text
 * model and are passed over. Safetensors win when both are present.
 */
#ifndef ATH_CKPT_H
#define ATH_CKPT_H

#include "core/gguf.h"
#include "core/safetensors.h"

typedef enum {
    ATH_CKPT_SAFETENSORS,
    ATH_CKPT_GGUF
} ath_ckpt_format;

/* Lower-case name ("safetensors", "gguf"). */
const char *ath_ckpt_format_name(ath_ckpt_format f);

typedef struct {
    const char *name;
    ath_tensor  t;       /* view into the mapping (read-only memory)     */
    uint64_t    bytes;   /* bytes of the tensor in the file              */
    uint32_t    shard;   /* file: the safetensors shard; 0 for GGUF      */
    uint64_t    offset;  /* offset of its first byte in that file        */
} ath_ckpt_tensor;

typedef struct {
    ath_ckpt_format  format;
    const char      *path;           /* the directory, or the .gguf file           */
    ath_ckpt_tensor *tensors;        /* sorted by name (ath_safetensors_name_cmp)  */
    uint32_t         n_tensors;
    uint64_t         tensor_bytes;   /* sum of bytes                               */
    ath_safetensors  st;             /* format ATH_CKPT_SAFETENSORS                */
    ath_gguf         gguf;           /* format ATH_CKPT_GGUF                       */
} ath_ckpt;

/* Open the checkpoint in dir. Tables live in a, which must outlive c. On
 * failure everything is closed. ATH_ERR_NOT_FOUND if dir holds no
 * checkpoint, ATH_ERR_FORMAT if it holds several .gguf files; the readers'
 * errors otherwise. */
ath_status ath_ckpt_open(ath_ckpt *c, ath_arena *a, const char *dir);

/* Tensor by exact name, binary search; NULL if absent. */
const ath_ckpt_tensor *ath_ckpt_find(const ath_ckpt *c, const char *name);

/* Unmap. Tensor views become invalid. */
void ath_ckpt_close(ath_ckpt *c);

#endif /* ATH_CKPT_H */
