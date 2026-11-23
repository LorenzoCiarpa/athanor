/* alloc_trace.c: LD_PRELOAD shim that reports every heap allocation.
 *
 * Development only (see check_alloc.sh). Each malloc, calloc and realloc
 * prints one line to stderr:
 *
 *   [alloc] <function> <bytes> <object file> <offset of the caller in it>
 *
 * The offset can be resolved with addr2line -f -e <object file> <offset>.
 * The shim forwards to glibc's internal entry points, and writes with
 * write(2) so that reporting never allocates by itself.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

extern void *__libc_malloc(size_t n);
extern void *__libc_calloc(size_t n, size_t size);
extern void *__libc_realloc(void *p, size_t n);

static void report(const char *fn, size_t n, void *caller)
{
    Dl_info info = {0};
    const char *obj = "?";
    uintptr_t off = (uintptr_t)caller;
    if (dladdr(caller, &info) && info.dli_fname) {
        obj = info.dli_fname;
        off = (uintptr_t)caller - (uintptr_t)info.dli_fbase;
    }
    char line[1024];
    int l = snprintf(line, sizeof line, "[alloc] %s %zu %s 0x%lx\n", fn, n, obj,
                     (unsigned long)(off - 1)); /* -1: inside the call instruction */
    if (l > 0 && write(2, line, (size_t)l) < 0)
        return;
}

void *malloc(size_t n)
{
    report("malloc", n, __builtin_return_address(0));
    return __libc_malloc(n);
}

void *calloc(size_t n, size_t size)
{
    report("calloc", n * size, __builtin_return_address(0));
    return __libc_calloc(n, size);
}

void *realloc(void *p, size_t n)
{
    report("realloc", n, __builtin_return_address(0));
    return __libc_realloc(p, n);
}
