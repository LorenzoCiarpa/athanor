/* buf.c: a growable byte buffer in an arena. See buf.h. */
#include "core/buf.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

void ath_buf_init(ath_buf *b, ath_arena *a)
{
    *b = (ath_buf){.a = a};
}

/* Room for n more bytes and the NUL. */
static ath_status reserve(ath_buf *b, size_t n)
{
    if (n > SIZE_MAX / 2 - b->len)
        return ATH_ERR_NOMEM;
    size_t need = b->len + n + 1;
    if (need <= b->cap)
        return ATH_OK;
    size_t cap = b->cap ? b->cap : 64;
    while (cap < need)
        cap *= 2;
    char *d = ath_arena_alloc(b->a, cap, 1);
    if (!d)
        return ATH_ERR_NOMEM;
    if (b->len)
        memcpy(d, b->data, b->len);
    d[b->len] = '\0';
    b->data = d;
    b->cap = cap;
    return ATH_OK;
}

ath_status ath_buf_append(ath_buf *b, const void *p, size_t n)
{
    ATH_CHECK(reserve(b, n));
    if (n)
        memcpy(b->data + b->len, p, n);
    b->len += n;
    b->data[b->len] = '\0';
    return ATH_OK;
}

ath_status ath_buf_puts(ath_buf *b, const char *s) { return ath_buf_append(b, s, strlen(s)); }

ath_status ath_buf_putc(ath_buf *b, char c) { return ath_buf_append(b, &c, 1); }

ath_status ath_buf_printf(ath_buf *b, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0)
        return ATH_ERR_INVALID_ARG;
    ATH_CHECK(reserve(b, (size_t)n));
    va_start(ap, fmt);
    vsnprintf(b->data + b->len, (size_t)n + 1, fmt, ap);
    va_end(ap);
    b->len += (size_t)n;
    return ATH_OK;
}

void ath_buf_clear(ath_buf *b)
{
    b->len = 0;
    if (b->data)
        b->data[0] = '\0';
}

const char *ath_buf_str(const ath_buf *b) { return b->data ? b->data : ""; }
