/* sha256.h: SHA-256 (FIPS 180-4).
 *
 * Benchmark reports record the hash of every weight file they ran
 * (docs/BENCHMARKS.md); the keys of on-disk prefix snapshots (step 14) will
 * be hashes of token prefixes. Streaming: init, any number of updates,
 * final.
 */
#ifndef ATH_CORE_SHA256_H
#define ATH_CORE_SHA256_H

#include "core/base.h"

typedef struct {
    uint32_t h[8];
    uint64_t len;        /* bytes hashed so far        */
    uint8_t  buf[64];    /* the block being filled     */
    uint32_t n;          /* bytes in buf               */
} ath_sha256;

void ath_sha256_init(ath_sha256 *c);
void ath_sha256_update(ath_sha256 *c, const void *data, size_t n);
void ath_sha256_final(ath_sha256 *c, uint8_t digest[32]);

/* The digest as 64 lower-case hex digits and a NUL. */
void ath_sha256_hex(const uint8_t digest[32], char hex[65]);

/* The hash of a whole file, read through its read-only mapping. */
ath_status ath_sha256_file(const char *path, char hex[65]);

#endif /* ATH_CORE_SHA256_H */
