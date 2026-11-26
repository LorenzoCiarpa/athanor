/* prefix.h: snapshots of a sequence at prefixes of its tokens (step 14).
 *
 * A recurrent state cannot be rolled back, so a prompt that begins with
 * tokens a sequence already ran starts from a snapshot of the state taken
 * after them (src/net/gen.h keeps one on the device). This store keeps
 * more of them, trimmed to their tokens (ath_state_snap_*, model.h): in
 * host memory under a byte budget, and in a folder on disk under another,
 * the least recently used leaving first. A prompt starts from its longest
 * stored prefix.
 *
 * On disk a snapshot is one file, named by the hash of the store's
 * identity (the model and the state's dtypes) and of its tokens: a header
 * (magic, version, that identity, the dtypes, the token counts, the
 * payload's size and hash, the header's own hash), the tokens, the
 * payload. A file is written under a temporary name, synced and renamed:
 * whole or absent. Opening the store reads the folder's headers; a file
 * of another model or other dtypes, or one that does not check out, is not
 * used (and is left where it is). The model's identity is a hash of its
 * family, its tensors' names, dtypes, shapes and sizes, the first and last
 * 4 KiB of each tensor's data, and its state specs.
 *
 * A snapshot goes to disk when it leaves host memory and when the store
 * closes, unless it is superseded: a longer stored snapshot begins with
 * its tokens, its tokens were only ever followed one way, and it was not
 * saved as shared. So the earlier turns of a conversation are not
 * written, while its latest turn is, and so is a shared prefix: the system
 * prompt and tools that the next session of an agent begins with too,
 * saved as such (and a prefix that turned out to branch).
 *
 * One thread at a time.
 */
#ifndef ATH_INFER_PREFIX_H
#define ATH_INFER_PREFIX_H

#include "model/model.h"

#define ATH_PREFIX_RAM_DEFAULT  (8ull << 30)    /* bytes of snapshots in host memory */
#define ATH_PREFIX_DISK_DEFAULT (64ull << 30)   /* bytes of snapshot files           */
#define ATH_PREFIX_MIN_TOKENS   64              /* shorter prefixes are not stored   */
#define ATH_PREFIX_SHARED_MIN   1024            /* a shared part worth a snapshot of
                                                 * its own (serve, chat)             */

typedef struct ath_prefix_store ath_prefix_store;

typedef struct {
    const char *dir;          /* snapshot folder (created if missing); NULL: host memory only */
    uint64_t    ram_bytes;    /* 0: ATH_PREFIX_RAM_DEFAULT                                    */
    uint64_t    disk_bytes;   /* 0: ATH_PREFIX_DISK_DEFAULT                                   */
    int32_t     min_tokens;   /* 0: ATH_PREFIX_MIN_TOKENS                                     */
} ath_prefix_opts;

typedef struct {
    uint32_t ram_count, disk_count;   /* snapshots in host memory, on disk (one can be both) */
    uint64_t ram_bytes, disk_bytes;
    /* the last save and load: tokens, bytes, seconds, and whether the load read the disk */
    int32_t  save_tokens, load_tokens;
    uint64_t save_bytes, load_bytes;
    double   save_s, load_s;
    bool     load_disk;
} ath_prefix_stats;

/* A store for the states of m with activation dtype act and KV dtype kv.
 * With a folder, the snapshots in it that belong to this model and these
 * dtypes are known at once (their payloads stay on disk until loaded). */
ath_status ath_prefix_open(ath_prefix_store **out, const ath_model *m, ath_dtype act, ath_dtype kv,
                           const ath_prefix_opts *o);
/* Write to disk what must go there, then free the store. NULL is fine. */
void       ath_prefix_close(ath_prefix_store *ps);

/* The length of the longest stored prefix of ids[0, n) shorter than n; 0
 * if there is none. */
int32_t    ath_prefix_find(ath_prefix_store *ps, const int32_t *ids, int32_t n);

/* Load into s the snapshot of ids[0, len), len as ath_prefix_find gave it;
 * ids[len] (the token that follows, if len < n) tells the store how the
 * prefix was continued. From disk when it is not in host memory.
 * ATH_ERR_NOT_FOUND if there is no such snapshot, or if its file no longer
 * reads back whole (the store forgets it); s is unchanged then. */
ath_status ath_prefix_load(ath_prefix_store *ps, const int32_t *ids, int32_t len, int32_t n, ath_state *s);

/* Store the state s, which holds ids[0, s->n_past); shared: a prefix
 * other prompts begin with too (never superseded). Prefixes under the
 * minimum length are not stored; a prefix already stored is only marked
 * used (and shared, if it is now). */
ath_status ath_prefix_save(ath_prefix_store *ps, const int32_t *ids, const ath_state *s, bool shared);

/* true if ids[0, n) itself is stored. */
bool       ath_prefix_has(ath_prefix_store *ps, const int32_t *ids, int32_t n);

void       ath_prefix_get_stats(const ath_prefix_store *ps, ath_prefix_stats *out);

#endif /* ATH_INFER_PREFIX_H */
