/* devarena.c: linear allocation in device memory. See backend.h.
 *
 * The same scheme as the core arenas (src/core/arena.c): blocks chained in
 * order, a pointer bumped inside the current one, mark and reset. The
 * difference is where the bookkeeping lives. A core arena keeps its block
 * header inside the block; here the block is device memory, which the host
 * cannot write, so each block has a record allocated in a host arena.
 *
 * After a reset the blocks past the reset point are kept and reused in
 * order. A request that does not fit in the next kept block gets a new
 * block inserted before it, so the order of blocks always follows the
 * order of allocation and a mark stays meaningful.
 */
#include "backend/backend.h"

struct ath_dev_block {
    ath_dev_block *next;
    char          *data;   /* device memory */
    size_t         size;
    size_t         used;
};

void ath_dev_arena_init(ath_dev_arena *a, const ath_device *d, ath_arena *host, size_t block_size)
{
    *a = (ath_dev_arena){
        .dev = d, .host = host, .block_size = block_size ? block_size : ATH_DEV_ARENA_DEFAULT_BLOCK};
}

void ath_dev_arena_free(ath_dev_arena *a)
{
    for (ath_dev_block *b = a->first; b; b = b->next) {
        ath_dev_free(a->dev, b->data);
        b->data = NULL;
    }
    a->first = a->cur = NULL;
}

/* A new block of at least `size` bytes, linked after `prev` (or first). */
static ath_dev_block *new_block(ath_dev_arena *a, ath_dev_block *prev, size_t size)
{
    size_t bytes = size > a->block_size ? size : a->block_size;
    ath_dev_block *b = ath_arena_zalloc(a->host, sizeof *b, _Alignof(ath_dev_block));
    if (!b)
        return NULL;
    void *mem;
    if (ath_dev_alloc(a->dev, bytes, &mem) != ATH_OK)
        return NULL;   /* the record stays in the host arena, unused */
    b->data = mem;
    b->size = bytes;
    if (prev) {
        b->next = prev->next;
        prev->next = b;
    } else {
        b->next = a->first;
        a->first = b;
    }
    return b;
}

void *ath_dev_arena_alloc(ath_dev_arena *a, size_t size, size_t align)
{
    if (!align)
        align = ATH_DEV_ALIGN;
    if ((align & (align - 1)) || align > ATH_DEV_ALIGN || size > SIZE_MAX - ATH_DEV_ALIGN)
        return NULL;
    if (!size)
        size = 1;
    if (!a->cur) {
        a->cur = a->first;
        if (a->cur)
            a->cur->used = 0;
    }
    if (a->cur) {
        size_t off = (a->cur->used + align - 1) & ~(align - 1);
        if (off <= a->cur->size && size <= a->cur->size - off) {
            a->cur->used = off + size;
            return a->cur->data + off;
        }
        /* The next kept block, if it is large enough; otherwise a new block
         * goes in front of it. */
        ath_dev_block *next = a->cur->next;
        if (next && size <= next->size) {
            next->used = size;
            a->cur = next;
            return next->data;
        }
    }
    ath_dev_block *b = new_block(a, a->cur, size);
    if (!b)
        return NULL;
    b->used = size;
    a->cur = b;
    return b->data;
}

ath_dev_arena_pos ath_dev_arena_mark(const ath_dev_arena *a)
{
    return (ath_dev_arena_pos){a->cur, a->cur ? a->cur->used : 0};
}

void ath_dev_arena_reset(ath_dev_arena *a, ath_dev_arena_pos pos)
{
    a->cur = pos.block;
    if (a->cur)
        a->cur->used = pos.used;
}

size_t ath_dev_arena_used(const ath_dev_arena *a)
{
    size_t n = 0;
    for (ath_dev_block *b = a->first; b && a->cur; b = b->next) {
        n += b->used;
        if (b == a->cur)
            break;
    }
    return n;
}

size_t ath_dev_arena_reserved(const ath_dev_arena *a)
{
    size_t n = 0;
    for (ath_dev_block *b = a->first; b; b = b->next)
        n += b->size;
    return n;
}
