/* buf.h: a growable byte buffer in an arena.
 *
 * Text built piece by piece (a rendered prompt, decoded tokens, JSON) goes
 * into an ath_buf. It lives in an arena: growing allocates a block twice
 * as large and copies, and the old block stays in the arena until the arena
 * is reset, so the memory used is at most about twice the final size. The
 * data is always NUL-terminated (not counted in len), so it can be printed
 * as a C string when it holds no NUL bytes.
 */
#ifndef ATH_BUF_H
#define ATH_BUF_H

#include "core/arena.h"
#include "core/base.h"

typedef struct {
    char      *data;   /* NULL until the first append; then NUL-terminated */
    size_t     len;
    size_t     cap;    /* bytes available in data, the NUL included       */
    ath_arena *a;
} ath_buf;

/* An empty buffer that allocates in a. */
void ath_buf_init(ath_buf *b, ath_arena *a);

/* Append n bytes (n may be 0). ATH_ERR_NOMEM if the arena cannot grow. */
ath_status ath_buf_append(ath_buf *b, const void *p, size_t n);
ath_status ath_buf_puts(ath_buf *b, const char *s);
ath_status ath_buf_putc(ath_buf *b, char c);
ath_status ath_buf_printf(ath_buf *b, const char *fmt, ...) __attribute__((format(printf, 2, 3)));

/* Forget the contents, keep the memory. */
void ath_buf_clear(ath_buf *b);

/* The contents as a C string ("" when empty). */
const char *ath_buf_str(const ath_buf *b);

#endif /* ATH_BUF_H */
