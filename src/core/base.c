/* base.c: status strings, logging and sorting. See base.h. */
#include "core/base.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *ath_status_str(ath_status s)
{
    static const char *const names[ATH_STATUS_COUNT] = {
        [ATH_OK]              = "ok",
        [ATH_ERR_INVALID_ARG] = "invalid argument",
        [ATH_ERR_NOMEM]       = "out of memory",
        [ATH_ERR_IO]          = "I/O error",
        [ATH_ERR_NOT_FOUND]   = "not found",
        [ATH_ERR_PARSE]       = "parse error",
        [ATH_ERR_FORMAT]      = "invalid format",
        [ATH_ERR_TYPE]        = "wrong type",
        [ATH_ERR_RANGE]       = "out of range",
        [ATH_ERR_UNSUPPORTED] = "unsupported",
        [ATH_ERR_DEVICE]      = "device error",
    };
    if ((unsigned)s >= ATH_STATUS_COUNT || !names[s])
        return "unknown status";
    return names[s];
}

static const char *const level_names[] = {"none", "error", "warn", "info", "debug", "trace"};

/* The threshold is read from the environment on every call instead of being
 * cached: athanor keeps no global state, and logging is never on a hot path. */
static int log_threshold(void)
{
    const char *e = getenv("ATH_LOG");
    if (!e || !*e)
        return ATH_LOG_WARN;
    if (e[0] >= '0' && e[0] <= '9' && e[1] == '\0')
        return e[0] - '0' > ATH_LOG_TRACE ? ATH_LOG_TRACE : e[0] - '0';
    for (int i = 0; i < (int)ATH_ARRAY_LEN(level_names); i++)
        if (strcmp(e, level_names[i]) == 0)
            return i;
    if (strcmp(e, "off") == 0)
        return 0;
    return ATH_LOG_WARN;
}

bool ath_log_enabled(ath_log_level level)
{
    return (int)level <= log_threshold();
}

void ath_log(ath_log_level level, const char *fmt, ...)
{
    if (!ath_log_enabled(level))
        return;
    int l = (int)level;
    if (l < ATH_LOG_ERROR || l > ATH_LOG_TRACE)
        l = ATH_LOG_ERROR;
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "athanor %s: ", level_names[l]);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

/* ---- sorting ----------------------------------------------------------------- */

static void swap_elem(char *a, char *b, size_t size, void *tmp)
{
    memcpy(tmp, a, size);
    memcpy(a, b, size);
    memcpy(b, tmp, size);
}

static void sift_down(char *base, size_t root, size_t end, size_t size, ath_cmp_fn cmp, void *ctx, void *tmp)
{
    for (;;) {
        size_t child = 2 * root + 1;
        if (child >= end)
            return;
        if (child + 1 < end && cmp(base + child * size, base + (child + 1) * size, ctx) < 0)
            child++;
        if (cmp(base + root * size, base + child * size, ctx) >= 0)
            return;
        swap_elem(base + root * size, base + child * size, size, tmp);
        root = child;
    }
}

void ath_sort(void *base, size_t n, size_t size, ath_cmp_fn cmp, void *ctx, void *tmp)
{
    char *b = base;
    if (n < 2)
        return;
    for (size_t i = n / 2; i-- > 0;)
        sift_down(b, i, n, size, cmp, ctx, tmp);
    for (size_t end = n - 1; end > 0; end--) {
        swap_elem(b, b + end * size, size, tmp);
        sift_down(b, 0, end, size, cmp, ctx, tmp);
    }
}
