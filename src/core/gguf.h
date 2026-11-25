/* gguf.h: zero-copy reader for GGUF files (llama.cpp's checkpoint format).
 *
 * File layout, version 3 (ggml's docs/gguf.md; version 2 is the same layout,
 * version 1 used 32-bit counts and is refused), all little-endian:
 *
 *   u32 magic "GGUF", u32 version, u64 n_tensors, u64 n_kv
 *   n_kv × key/value:   string key, u32 type, value
 *   n_tensors × info:   string name, u32 n_dims, u64 ne[n_dims],
 *                       u32 ggml type, u64 offset (from the data section)
 *   padding to the alignment (metadata general.alignment, default 32)
 *   data section: every tensor at its offset, a multiple of the alignment
 *
 * A string is a u64 byte count and the bytes, not NUL-terminated. Value
 * types: 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool, 8 string,
 * 9 array (u32 element type, u64 count, elements), 10 u64, 11 i64, 12 f64.
 * ne[0] is the fastest-varying dimension, so a tensor of ne [2048, 248320]
 * is athanor's row-major [248320, 2048]: the shape is ne reversed.
 *
 * The file is mapped with mmap and every tensor becomes an ath_tensor that
 * points into the mapping, as for safetensors (core/safetensors.h). Tensor
 * types athanor reads: F32, F16, BF16 and Q8_0; a file with any other type is
 * ATH_ERR_UNSUPPORTED. Validation: every count, string and array lies inside
 * the file, names are unique and NUL-free, offsets are aligned, tensors do
 * not overlap and lie inside the file, and the gap between one tensor and
 * the next is padding only (less than the alignment).
 *
 * Metadata keys and string values are copied into the arena with a NUL
 * terminator; arrays are kept as a pointer into the mapping with their
 * element type and count (a tokenizer's 248,320 strings are not copied).
 */
#ifndef ATH_GGUF_H
#define ATH_GGUF_H

#include "core/arena.h"
#include "core/file.h"
#include "core/tensor.h"

typedef enum {
    ATH_GGUF_U8 = 0, ATH_GGUF_I8 = 1, ATH_GGUF_U16 = 2, ATH_GGUF_I16 = 3,
    ATH_GGUF_U32 = 4, ATH_GGUF_I32 = 5, ATH_GGUF_F32 = 6, ATH_GGUF_BOOL = 7,
    ATH_GGUF_STRING = 8, ATH_GGUF_ARRAY = 9, ATH_GGUF_U64 = 10, ATH_GGUF_I64 = 11,
    ATH_GGUF_F64 = 12
} ath_gguf_type;

typedef struct {
    const char   *key;
    ath_gguf_type type;
    union {
        uint64_t u;      /* U8, U16, U32, U64, BOOL                       */
        int64_t  i;      /* I8, I16, I32, I64                             */
        double   f;      /* F32, F64                                      */
        struct {
            const char *ptr;   /* NUL-terminated copy in the arena        */
            size_t      len;
        } str;
        struct {
            ath_gguf_type type;   /* element type                         */
            uint64_t      count;
            const void   *data;   /* first element, in the mapping        */
        } arr;
    } v;
} ath_gguf_kv;

typedef struct {
    const char *name;
    ath_tensor  t;          /* view into the mapping (read-only memory)  */
    uint32_t    ggml_type;  /* as stored: 0 F32, 1 F16, 8 Q8_0, 30 BF16   */
    uint64_t    offset;     /* from the start of the data section        */
    uint64_t    bytes;
} ath_gguf_tensor;

typedef struct {
    const char      *path;
    ath_file         file;
    uint32_t         version;
    uint64_t         alignment;
    uint64_t         data_offset;   /* file offset of the data section   */
    ath_gguf_kv     *kv;            /* in file order                     */
    uint64_t         n_kv;
    ath_gguf_tensor *tensors;       /* sorted by name (ath_safetensors_name_cmp) */
    uint32_t         n_tensors;
    uint64_t         tensor_bytes;  /* sum of bytes                      */
} ath_gguf;

/* Open and validate the file at path. Tables live in a, which must outlive
 * g. On failure everything opened so far is closed. ATH_ERR_FORMAT for a
 * malformed or inconsistent file, ATH_ERR_UNSUPPORTED for version 1 or an
 * unknown value or tensor type. */
ath_status ath_gguf_open(ath_gguf *g, ath_arena *a, const char *path);

/* Metadata entry by exact key; NULL if absent. */
const ath_gguf_kv *ath_gguf_get(const ath_gguf *g, const char *key);

/* Tensor by exact name, binary search; NULL if absent. */
const ath_gguf_tensor *ath_gguf_find(const ath_gguf *g, const char *name);

/* Unmap. Tensor views become invalid. */
void ath_gguf_close(ath_gguf *g);

#endif /* ATH_GGUF_H */
