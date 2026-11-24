/* util.h: helpers shared by the test suites: a scratch directory for
 * synthetic files and the location of the real checkpoint. Header-only, like
 * test.h; each suite is a single translation unit.
 */
#ifndef ATH_TEST_UTIL_H
#define ATH_TEST_UTIL_H

#include "core/file.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ---- helpers: a scratch directory for synthetic files ---------------------- */

typedef struct {
    char path[512];
    char files[16][64];
    int  n_files;
} tmpdir;

static inline bool tmpdir_make(tmpdir *d)
{
    const char *base = getenv("TMPDIR");
    if (!base || !*base)
        base = "/tmp";
    snprintf(d->path, sizeof d->path, "%s/athanor-test-XXXXXX", base);
    d->n_files = 0;
    return mkdtemp(d->path) != NULL;
}

static inline bool tmpdir_write(tmpdir *d, const char *name, const void *data, size_t n)
{
    char p[1024];
    if (ath_path_join(p, sizeof p, d->path, name) != ATH_OK)
        return false;
    FILE *f = fopen(p, "wb");
    if (!f)
        return false;
    bool ok = fwrite(data, 1, n, f) == n;
    ok = (fclose(f) == 0) && ok;
    for (int i = 0; ok && i < d->n_files; i++)
        if (strcmp(d->files[i], name) == 0)
            return true;
    if (ok && d->n_files < (int)ATH_ARRAY_LEN(d->files)) {
        snprintf(d->files[d->n_files], sizeof d->files[0], "%s", name);
        d->n_files++;
    }
    return ok;
}

/* The real checkpoint used by the tests that need one. These tests check
 * format-level invariants only, so any checkpoint directory works. */
static inline const char *model_dir(void)
{
    const char *d = getenv("ATH_MODEL_DIR");
    return (d && *d) ? d : "models/Qwen3.5-0.8B";
}

static inline bool model_file(char *out, size_t cap, const char *name)
{
    return ath_path_join(out, cap, model_dir(), name) == ATH_OK && ath_file_exists(out);
}

static inline void tmpdir_remove(tmpdir *d)
{
    char p[1024];
    for (int i = 0; i < d->n_files; i++)
        if (ath_path_join(p, sizeof p, d->path, d->files[i]) == ATH_OK)
            unlink(p);
    rmdir(d->path);
}

#endif /* ATH_TEST_UTIL_H */
