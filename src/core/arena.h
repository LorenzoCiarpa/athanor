/* arena.h: lifetime-scoped linear allocator.
 *
 * An arena hands out memory by bumping a pointer inside a block. When a block
 * is full a new one is chained after it; nothing is freed individually.
 * Everything allocated from an arena dies together, either with
 * ath_arena_reset() back to a saved position or with ath_arena_free().
 *
 *   ath_arena a;
 *   ath_arena_init(&a, 0);                  // default block size
 *   float *x = ath_arena_alloc(&a, n * sizeof *x, _Alignof(float));
 *   ath_arena_pos p = ath_arena_mark(&a);
 *   ... scratch allocations ...
 *   ath_arena_reset(&a, p);                 // scratch gone, blocks kept for reuse
 *   ath_arena_free(&a);
 *
 * Allocation functions return NULL on failure instead of an ath_status, like
 * every allocator; callers map NULL to ATH_ERR_NOMEM.
 */
#ifndef ATH_ARENA_H
#define ATH_ARENA_H

#include "core/base.h"

#define ATH_ARENA_DEFAULT_BLOCK ((size_t)1 << 20)   /* 1 MiB */

typedef struct ath_arena_block ath_arena_block;

typedef struct {
    ath_arena_block *first;       /* chain of blocks, oldest first          */
    ath_arena_block *cur;         /* block currently being filled, or NULL  */
    size_t           block_size;  /* minimum size of a new block, in bytes  */
} ath_arena;

/* A saved position, returned by ath_arena_mark(). */
typedef struct {
    ath_arena_block *block;
    size_t           used;
} ath_arena_pos;

/* No memory is reserved until the first allocation. block_size 0 selects
 * ATH_ARENA_DEFAULT_BLOCK. */
void ath_arena_init(ath_arena *a, size_t block_size);

/* Release every block. The arena can be used again afterwards. */
void ath_arena_free(ath_arena *a);

/* size bytes aligned to align (a power of two). Memory is not zeroed.
 * A request larger than the block size gets a block of its own. */
void *ath_arena_alloc(ath_arena *a, size_t size, size_t align);

/* Same as ath_arena_alloc, zero-filled. */
void *ath_arena_zalloc(ath_arena *a, size_t size, size_t align);

/* Copy of the first n bytes of s, NUL-terminated. */
char *ath_arena_strndup(ath_arena *a, const char *s, size_t n);

/* Save and restore the allocation position. Reset frees (for reuse) every
 * allocation made after the mark; the blocks themselves are kept. A mark is
 * valid until the arena is reset to an earlier position or freed. */
ath_arena_pos ath_arena_mark(const ath_arena *a);
void          ath_arena_reset(ath_arena *a, ath_arena_pos pos);

/* Bytes handed out (including alignment padding) and bytes reserved in blocks. */
size_t ath_arena_used(const ath_arena *a);
size_t ath_arena_reserved(const ath_arena *a);

#endif /* ATH_ARENA_H */
