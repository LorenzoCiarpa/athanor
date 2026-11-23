/* arena.c: linear allocator with chained blocks. See arena.h.
 *
 * Invariant: every block after `cur` in the chain is empty. Allocation fills
 * `cur`; when it does not fit, the next block is reused if it is large
 * enough, otherwise a new block is inserted right after `cur`. This keeps
 * blocks that a reset left behind available for later allocations.
 *
 * This file is the only place in athanor's host code that calls malloc().
 */
#include "core/arena.h"

#include <stdlib.h>
#include <string.h>

struct ath_arena_block {
    ath_arena_block *next;
    size_t           size;   /* bytes in data[] */
    size_t           used;   /* bytes of data[] handed out */
    _Alignas(16) unsigned char data[];
};

void ath_arena_init(ath_arena *a, size_t block_size)
{
    a->first = NULL;
    a->cur = NULL;
    a->block_size = block_size ? block_size : ATH_ARENA_DEFAULT_BLOCK;
}

void ath_arena_free(ath_arena *a)
{
    ath_arena_block *b = a->first;
    while (b) {
        ath_arena_block *next = b->next;
        free(b);
        b = next;
    }
    a->first = NULL;
    a->cur = NULL;
}

/* Take size bytes aligned to align from b, or return NULL if they do not fit.
 * Alignment is computed on the actual address, so any power of two works. */
static void *block_take(ath_arena_block *b, size_t size, size_t align)
{
    uintptr_t base = (uintptr_t)b->data;
    uintptr_t at = (base + b->used + (align - 1)) & ~(uintptr_t)(align - 1);
    size_t off = (size_t)(at - base);
    if (off > b->size || size > b->size - off)
        return NULL;
    b->used = off + size;
    return (void *)at;
}

void *ath_arena_alloc(ath_arena *a, size_t size, size_t align)
{
    if (align == 0 || (align & (align - 1)) != 0)
        return NULL;
    if (a->cur) {
        void *p = block_take(a->cur, size, align);
        if (p)
            return p;
    }
    /* Worst case the block start needs align - 1 bytes of padding. */
    if (size > SIZE_MAX - sizeof(ath_arena_block) - align)
        return NULL;
    size_t need = size + align - 1;

    ath_arena_block *next = a->cur ? a->cur->next : a->first;
    if (next && next->size >= need) {
        next->used = 0;
        a->cur = next;
        return block_take(next, size, align);
    }

    size_t bsize = need > a->block_size ? need : a->block_size;
    ath_arena_block *b = malloc(sizeof(ath_arena_block) + bsize);
    if (!b)
        return NULL;
    b->size = bsize;
    b->used = 0;
    b->next = next;
    if (a->cur)
        a->cur->next = b;
    else
        a->first = b;
    a->cur = b;
    return block_take(b, size, align);
}

void *ath_arena_zalloc(ath_arena *a, size_t size, size_t align)
{
    void *p = ath_arena_alloc(a, size, align);
    if (p)
        memset(p, 0, size);
    return p;
}

char *ath_arena_strndup(ath_arena *a, const char *s, size_t n)
{
    if (n == SIZE_MAX)
        return NULL;
    char *d = ath_arena_alloc(a, n + 1, 1);
    if (!d)
        return NULL;
    memcpy(d, s, n);
    d[n] = '\0';
    return d;
}

ath_arena_pos ath_arena_mark(const ath_arena *a)
{
    ath_arena_pos p = {a->cur, a->cur ? a->cur->used : 0};
    return p;
}

void ath_arena_reset(ath_arena *a, ath_arena_pos pos)
{
    /* With a NULL block the next allocation starts again from the first
     * block; the blocks after the mark become the empty tail. */
    a->cur = pos.block;
    if (pos.block) {
        ATH_ASSERT(pos.used <= pos.block->size);
        pos.block->used = pos.used;
    }
}

size_t ath_arena_used(const ath_arena *a)
{
    size_t n = 0;
    if (!a->cur)
        return 0;
    for (const ath_arena_block *b = a->first; b; b = b->next) {
        n += b->used;
        if (b == a->cur)
            break;
    }
    return n;
}

size_t ath_arena_reserved(const ath_arena *a)
{
    size_t n = 0;
    for (const ath_arena_block *b = a->first; b; b = b->next)
        n += b->size;
    return n;
}
