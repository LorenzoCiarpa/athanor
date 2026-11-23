/* file.h: read-only access to files on disk.
 *
 * A model checkpoint is opened with mmap(): the kernel maps the file into the
 * address space and pages it in on first touch, so "loading" a 1.7 GB file
 * costs nothing until its bytes are used, and the page cache is shared with
 * every other process reading the same file. Nothing is copied.
 *
 * pread() is the explicit alternative for reading a byte range into a buffer
 * you own; the disk tier (step 12) is built on it.
 */
#ifndef ATH_FILE_H
#define ATH_FILE_H

#include "core/base.h"

typedef struct {
    int         fd;     /* kept open for pread; -1 when closed            */
    size_t      size;   /* file size in bytes                             */
    const void *data;   /* read-only mapping of the whole file, NULL if size is 0 */
} ath_file;

/* Open a regular file and map it read-only. On failure f is left closed.
 * ATH_ERR_NOT_FOUND if it does not exist, ATH_ERR_INVALID_ARG if it is not
 * a regular file, ATH_ERR_IO or ATH_ERR_NOMEM otherwise. */
ath_status ath_file_open(ath_file *f, const char *path);

/* Read exactly n bytes at offset off into dst. ATH_ERR_RANGE if the range
 * extends past the end of the file. Retries short reads and EINTR. */
ath_status ath_file_pread(const ath_file *f, void *dst, size_t n, uint64_t off);

/* Unmap and close. Safe on a closed or zero-initialized-then-failed file. */
void ath_file_close(ath_file *f);

/* true if path names an existing regular file. */
bool ath_file_exists(const char *path);

/* out = dir + "/" + name. ATH_ERR_RANGE if it does not fit in cap bytes. */
ath_status ath_path_join(char *out, size_t cap, const char *dir, const char *name);

#endif /* ATH_FILE_H */
