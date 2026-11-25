/* base.h: the definitions every other file in athanor includes.
 *
 * Fixed-width types, the status codes that every fallible function returns,
 * ATH_CHECK for propagating them, logging and debug assertions.
 * Nothing here allocates memory or keeps state between calls.
 */
#ifndef ATH_BASE_H
#define ATH_BASE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* C linkage when included from CUDA code (src/backend/cuda/gpu.cu), which nvcc
 * compiles as C++. */
#ifdef __cplusplus
extern "C" {
#endif

/* ---- status codes --------------------------------------------------------- */

/* Every fallible function returns an ath_status. ATH_OK is zero so that
 * `if (s)` reads as "if failed". Library code never calls exit() or abort():
 * it returns a status and lets the caller decide. */
typedef enum {
    ATH_OK = 0,
    ATH_ERR_INVALID_ARG,   /* caller passed something meaningless          */
    ATH_ERR_NOMEM,         /* an allocation failed                         */
    ATH_ERR_IO,            /* the operating system reported an I/O error   */
    ATH_ERR_NOT_FOUND,     /* file, key or tensor does not exist           */
    ATH_ERR_PARSE,         /* malformed text (JSON syntax)                 */
    ATH_ERR_FORMAT,        /* well-formed but inconsistent data            */
    ATH_ERR_TYPE,          /* a value has a different type than requested  */
    ATH_ERR_RANGE,         /* out of bounds, overflow or truncation        */
    ATH_ERR_UNSUPPORTED,   /* valid input that athanor does not handle yet */
    ATH_ERR_DEVICE,        /* the GPU runtime or driver reported an error  */
    ATH_STATUS_COUNT
} ath_status;

/* Short English description, never NULL (unknown values give "unknown status"). */
const char *ath_status_str(ath_status s);

/* ---- logging -------------------------------------------------------------- */

/* Messages go to stderr. The environment variable ATH_LOG selects the most
 * verbose level printed: none, error, warn (default), info, debug, trace,
 * or the numbers 0 to 5 in the same order. */
typedef enum {
    ATH_LOG_ERROR = 1,
    ATH_LOG_WARN  = 2,
    ATH_LOG_INFO  = 3,
    ATH_LOG_DEBUG = 4,
    ATH_LOG_TRACE = 5
} ath_log_level;

bool ath_log_enabled(ath_log_level level);
void ath_log(ath_log_level level, const char *fmt, ...) __attribute__((format(printf, 2, 3)));

/* ---- status propagation --------------------------------------------------- */

/* Evaluate expr; if it fails, return its status from the current function.
 * At ATH_LOG=debug every hop is printed, which gives a poor man's backtrace. */
#define ATH_CHECK(expr)                                                         \
    do {                                                                        \
        ath_status ath_check_s_ = (expr);                                       \
        if (ath_check_s_ != ATH_OK) {                                           \
            ath_log(ATH_LOG_DEBUG, "%s:%d: %s -> %s", __FILE__, __LINE__,       \
                    #expr, ath_status_str(ath_check_s_));                       \
            return ath_check_s_;                                                \
        }                                                                       \
    } while (0)

/* ---- assertions ----------------------------------------------------------- */

/* ATH_ASSERT checks programmer errors (broken invariants), never input errors:
 * input errors return a status. It is active only in debug builds
 * (make DEBUG=1), where it logs the location and traps. In release builds the
 * condition is not evaluated. */
#ifdef ATH_DEBUG
#define ATH_ASSERT(cond)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            ath_log(ATH_LOG_ERROR, "%s:%d: assertion failed: %s",               \
                    __FILE__, __LINE__, #cond);                                 \
            __builtin_trap();                                                   \
        }                                                                       \
    } while (0)
#else
#define ATH_ASSERT(cond) ((void)sizeof(!(cond)))
#endif

#define ATH_ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

/* ---- sorting -------------------------------------------------------------- */

/* Sort n elements of size bytes in place by cmp (negative, zero, positive),
 * which receives ctx. A heapsort instead of qsort(): glibc's qsort may
 * allocate a merge buffer with malloc, and loading a checkpoint must not
 * touch the heap outside arenas. tmp is scratch space for one element. Not
 * stable. */
typedef int (*ath_cmp_fn)(const void *a, const void *b, void *ctx);
void ath_sort(void *base, size_t n, size_t size, ath_cmp_fn cmp, void *ctx, void *tmp);

#ifdef __cplusplus
}
#endif

#endif /* ATH_BASE_H */
