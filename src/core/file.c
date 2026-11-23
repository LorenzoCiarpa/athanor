/* file.c: mmap and pread on POSIX. See file.h. */
#include "core/file.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static ath_status status_from_errno(int e)
{
    switch (e) {
    case ENOENT:
    case ENOTDIR:
        return ATH_ERR_NOT_FOUND;
    case ENOMEM:
        return ATH_ERR_NOMEM;
    case EINVAL:
        return ATH_ERR_INVALID_ARG;
    default:
        return ATH_ERR_IO;
    }
}

ath_status ath_file_open(ath_file *f, const char *path)
{
    f->fd = -1;
    f->size = 0;
    f->data = NULL;

    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        int e = errno;
        ath_log(ATH_LOG_ERROR, "cannot open %s: %s", path, strerror(e));
        return status_from_errno(e);
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        int e = errno;
        ath_log(ATH_LOG_ERROR, "cannot stat %s: %s", path, strerror(e));
        close(fd);
        return status_from_errno(e);
    }
    if (!S_ISREG(st.st_mode)) {
        ath_log(ATH_LOG_ERROR, "%s is not a regular file", path);
        close(fd);
        return ATH_ERR_INVALID_ARG;
    }
    if ((uintmax_t)st.st_size > SIZE_MAX) {
        close(fd);
        return ATH_ERR_RANGE;
    }

    size_t size = (size_t)st.st_size;
    const void *data = NULL;
    if (size > 0) {
        /* MAP_PRIVATE + PROT_READ: the mapping can never write to the file. */
        void *p = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) {
            int e = errno;
            ath_log(ATH_LOG_ERROR, "cannot mmap %s: %s", path, strerror(e));
            close(fd);
            return status_from_errno(e);
        }
        data = p;
    }
    f->fd = fd;
    f->size = size;
    f->data = data;
    return ATH_OK;
}

ath_status ath_file_pread(const ath_file *f, void *dst, size_t n, uint64_t off)
{
    if (f->fd < 0)
        return ATH_ERR_INVALID_ARG;
    if (off > f->size || n > f->size - off)
        return ATH_ERR_RANGE;
    unsigned char *p = dst;
    while (n > 0) {
        /* Linux transfers at most about 2 GiB per call: loop on short reads. */
        ssize_t r = pread(f->fd, p, n, (off_t)off);
        if (r < 0) {
            if (errno == EINTR)
                continue;
            ath_log(ATH_LOG_ERROR, "pread failed: %s", strerror(errno));
            return ATH_ERR_IO;
        }
        if (r == 0) /* the file shrank after it was opened */
            return ATH_ERR_IO;
        p += r;
        n -= (size_t)r;
        off += (uint64_t)r;
    }
    return ATH_OK;
}

void ath_file_close(ath_file *f)
{
    if (f->data)
        munmap((void *)f->data, f->size);
    if (f->fd >= 0)
        close(f->fd);
    f->fd = -1;
    f->size = 0;
    f->data = NULL;
}

bool ath_file_exists(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

ath_status ath_path_join(char *out, size_t cap, const char *dir, const char *name)
{
    size_t dl = strlen(dir);
    const char *sep = (dl > 0 && dir[dl - 1] == '/') ? "" : "/";
    int n = snprintf(out, cap, "%s%s%s", dir, sep, name);
    if (n < 0 || (size_t)n >= cap)
        return ATH_ERR_RANGE;
    return ATH_OK;
}
