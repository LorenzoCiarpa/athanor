/* bench.c: `athanor bench`: prefill and decode speed, as docs/BENCHMARKS.md
 * asks a report to measure and record them.
 *
 * Two kinds of test, named as llama-bench names them so that the two
 * tools' tables line up:
 *
 *   ppN        a prompt of N tokens into an empty state: prefill tokens/s,
 *              and the time to the first token (prefill plus sampling it)
 *   tgN@dD     N tokens decoded one at a time, greedy, after a prompt of D
 *              tokens (the context depth): decode tokens/s
 *
 * Token ids are pseudo-random (a seeded generator, as llama-bench uses
 * random ids), which costs the model the same as text. Every test runs
 * --warmup times unmeasured and --reps times measured, from a reset state;
 * the report gives the median with the minimum and the maximum.
 *
 * Decode on a GPU runs through the decoder of src/infer/engine.h (step 9),
 * as `athanor chat` does: one recorded graph per token, the greedy choice
 * made on the device. --direct runs each token as one forward call whose
 * logits come back to the host, where the next token is chosen
 * (ath_sample_ref): the decode of step 8. The JSON report (--json) records the hardware, the
 * software, the model files with their SHA-256, the workload, the method
 * and every measured run, and memory as the driver and the kernel see it.
 */
#include "commands.h"

#include "backend/backend.h"
#include "core/buf.h"
#include "core/file.h"
#include "core/json.h"
#include "core/rng.h"
#include "core/sha256.h"
#include "core/threads.h"
#include "infer/engine.h"
#include "infer/spec.h"
#include "model/model.h"
#include "ops/ops.h"
#include "version.h"

#include <dirent.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <time.h>
#include <unistd.h>

#define MAX_LIST 8
#define MAX_REPS 50
#define MAX_FILES 16

void ath_cmd_bench_usage(FILE *out)
{
    fprintf(out,
            "usage: athanor bench <model_dir> [options]\n\n"
            "Measure prefill and decode speed (docs/BENCHMARKS.md). Tests are named as\n"
            "llama-bench names them: ppN is a prompt of N tokens, tgN@dD is N tokens\n"
            "decoded one by one after a prompt of D tokens. Token ids are pseudo-random.\n\n"
            "  -p, --prompt <list>   prompt lengths, comma-separated (default 512,4096; 0: none)\n"
            "  -n, --gen <n>         tokens decoded per tg test (default 128; 0: none)\n"
            "  -d, --depth <list>    context depths of the tg tests (default 0,4096,32768)\n"
            "  -r, --reps <n>        measured runs per test (default 5, at most %d)\n"
            "  --warmup <n>          unmeasured runs per test first (default 1)\n"
            "  --chunk <n>           prompt tokens per forward call (default 2048)\n"
            "  --device <cpu|gpu>    where the model runs (default: gpu if there is one)\n"
            "  --threads <n>         CPU threads (default: all)\n"
            "  --prompt-ids <file>   token ids (comma- or space-separated) as the prompt instead of\n"
            "                        pseudo-random ones: ppN runs them, and tgM decodes M tokens\n"
            "                        greedily after them, as other engines measure a given prompt\n"
            "  --experts <n>         MoE: routed experts per layer kept in device memory\n"
            "                        (default: as many as fit; the rest run on the CPU)\n"
            "  --seed <n>            seed of the token ids (default 1)\n"
            "  --json <file>         write the report there\n"
            "  --direct              decode with one forward call per token and the logits on\n"
            "                        the host (step 8), not with the decoder's graph\n"
            "  --kv <bf16|q8>        the KV cache in bf16 (default) or in int8 with a scale per 32\n"
            "                        values (step 14)\n"
            "  --mtp                 decode speculatively with the model's draft head (multi-token\n"
            "                        prediction, step 13): tg tests also report the guesses kept\n"
            "  --no-hash             do not hash the weight files for the report\n",
            MAX_REPS);
}

typedef struct {
    const char *model, *device, *json;
    const char *prompt_ids;   /* a file of token ids: the prompt of both tests (step 12) */
    int64_t     prompts[MAX_LIST], depths[MAX_LIST];
    int32_t     n_prompts, n_depths;
    int32_t     gen, reps, warmup, chunk, threads;
    int32_t     experts;    /* MoE: routed experts per layer in device memory; -1 as many as fit */
    uint64_t    seed;
    bool        hash, direct;
    bool        mtp;        /* decode with the draft head (step 13) */
    bool        kv8;        /* the KV cache in int8 (step 14) */
} options;

typedef struct {
    char    name[48];
    int64_t n;           /* tokens measured: prompt length (pp) or tokens decoded (tg) */
    int64_t depth;       /* tokens in the state before them                           */
    bool    decode;
    double  s[MAX_REPS]; /* seconds of each measured run                              */
    double  ttft[MAX_REPS];
    int32_t runs;
    double  kept;        /* --mtp: the share of passes whose guess was kept           */
} result;

typedef struct {
    ath_model    m;
    ath_engine  *e;            /* NULL on the CPU */
    ath_exec     exec;
    ath_pool     pool;
    bool         pool_open;
    ath_state    st;
    ath_tensor   logits;       /* f32 [1, vocab] in host memory */
    ath_arena   *scratch;
    int32_t     *ids;          /* the pseudo-random token ids    */
    int64_t      n_ids;
    ath_sampler  greedy;
    ath_decoder  dc;
    bool         decoder;      /* tg through dc */
    ath_spec     sp;
    bool         spec;         /* tg through sp (--mtp) */
    uint64_t     passes, kept; /* sp's counts over a test's measured runs */
    uint64_t     dev_used_before, dev_used_peak;
} bench;

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static bool parse_int(const char *s, int64_t lo, int64_t hi, int64_t *out)
{
    char *end;
    long long v = strtoll(s, &end, 10);
    if (end == s || *end || v < lo || v > hi)
        return false;
    *out = v;
    return true;
}

/* A comma-separated list of non-negative integers; zeros dropped unless keep0. */
static bool parse_list(const char *s, bool keep0, int64_t *out, int32_t *n)
{
    *n = 0;
    char buf[256];
    if (strlen(s) >= sizeof buf)
        return false;
    strcpy(buf, s);
    for (char *save = NULL, *tok = strtok_r(buf, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        if (*n == MAX_LIST || !parse_int(tok, 0, INT32_MAX, &out[*n]))
            return false;
        if (out[*n] > 0 || keep0)
            (*n)++;
    }
    return true;
}

/* ---- running ------------------------------------------------------------------- */

static void note_memory(bench *b)
{
    uint64_t free_b, total_b;
    if (b->e && ath_device_mem(&b->e->dev, &free_b, &total_b) == ATH_OK && total_b - free_b > b->dev_used_peak)
        b->dev_used_peak = total_b - free_b;
}

/* ids[0 .. n) into the state, in pieces of at most chunk tokens; the logits
 * of the last token if want. */
static ath_status feed(bench *b, const int32_t *ids, int64_t n, int32_t chunk, bool want)
{
    for (int64_t i = 0; i < n; i += chunk) {
        int64_t k = n - i < chunk ? n - i : chunk;
        ath_batch bt = {.tokens = ids + i, .n_tokens = (int32_t)k, .logits = want && i + k == n ? &b->logits : NULL};
        ATH_CHECK(ath_model_forward(&b->m, &bt, &b->st, NULL, b->scratch, &b->exec));
    }
    return ATH_OK;
}

static ath_status run_pp(bench *b, const options *o, int64_t n, double *secs, double *ttft)
{
    ATH_CHECK(ath_state_reset(&b->st, &b->m));
    double t0 = now_s();
    ATH_CHECK(feed(b, b->ids, n, o->chunk, true));
    double t1 = now_s();
    int32_t tok;
    ATH_CHECK(ath_sample_ref(&b->logits, &b->greedy, 0.0, b->scratch, &tok));
    *secs = t1 - t0;
    *ttft = now_s() - t0;
    return ATH_OK;
}

/* Speculative decode: the context and one more token as the prompt (the
 * draft head runs over it), then gen tokens. */
static ath_status run_tg_spec(bench *b, const options *o, int64_t depth, double *secs)
{
    ATH_CHECK(ath_state_reset(&b->st, &b->m));
    ATH_CHECK(ath_spec_feed(&b->sp, b->ids, (int32_t)depth + 1, b->logits.data));
    int32_t tok;
    ATH_CHECK(ath_sample_ref(&b->logits, &b->greedy, 0.0, b->scratch, &tok));
    const uint64_t p0 = b->sp.passes, k0 = b->sp.accepted;
    double t0 = now_s();
    ATH_CHECK(ath_spec_begin(&b->sp, tok));
    for (int32_t i = 1; i < o->gen;) {
        const double u[2] = {0, 0};
        int32_t out[2], k;
        ATH_CHECK(ath_spec_step(&b->sp, u, out, &k));
        i += k;
    }
    *secs = now_s() - t0;
    b->passes += b->sp.passes - p0;
    b->kept += b->sp.accepted - k0;
    return ATH_OK;
}

static ath_status run_tg(bench *b, const options *o, int64_t depth, double *secs)
{
    if (b->spec)
        return run_tg_spec(b, o, depth, secs);
    ATH_CHECK(ath_state_reset(&b->st, &b->m));
    ATH_CHECK(feed(b, b->ids, depth, o->chunk, false));
    if (b->e)   /* a pass without logits returns once queued: the prompt must be done first */
        ATH_CHECK(ath_stream_sync(&b->e->dev, ATH_STREAM_COMPUTE));
    int32_t tok = b->ids[depth];
    double t0 = now_s();
    for (int32_t i = 0; b->decoder && i < o->gen; i++)
        ATH_CHECK(ath_decoder_step(&b->dc, tok, 0.0, &tok));
    for (int32_t i = 0; !b->decoder && i < o->gen; i++) {
        ath_batch bt = {.tokens = &tok, .n_tokens = 1, .logits = &b->logits};
        ATH_CHECK(ath_model_forward(&b->m, &bt, &b->st, NULL, b->scratch, &b->exec));
        ATH_CHECK(ath_sample_ref(&b->logits, &b->greedy, 0.0, b->scratch, &tok));
    }
    *secs = now_s() - t0;
    return ATH_OK;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

/* Median, minimum and maximum of v[0 .. n). */
static void stats(const double *v, int32_t n, double *med, double *lo, double *hi)
{
    double s[MAX_REPS];
    memcpy(s, v, (size_t)n * sizeof *s);
    qsort(s, (size_t)n, sizeof *s, cmp_double);
    *lo = s[0];
    *hi = s[n - 1];
    *med = n % 2 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

static ath_status run_test(bench *b, const options *o, result *r)
{
    for (int32_t i = 0; i < o->warmup + o->reps; i++) {
        double secs = 0, ttft = 0;
        if (i == o->warmup)
            b->passes = b->kept = 0;
        ATH_CHECK(r->decode ? run_tg(b, o, r->depth, &secs) : run_pp(b, o, r->n, &secs, &ttft));
        note_memory(b);
        if (i >= o->warmup) {
            r->s[r->runs] = secs;
            r->ttft[r->runs++] = ttft;
        }
    }
    double tps[MAX_REPS], med, lo, hi;
    for (int32_t i = 0; i < r->runs; i++)
        tps[i] = (double)r->n / r->s[i];
    stats(tps, r->runs, &med, &lo, &hi);
    printf("%-16s %8.1f tok/s  (min %.1f, max %.1f)", r->name, med, lo, hi);
    if (r->decode && b->spec) {
        r->kept = b->passes ? (double)b->kept / (double)b->passes : 0;
        printf("  %.3f ms/token, %.0f%% of the guesses kept\n", 1e3 / med, 100 * r->kept);
    } else if (r->decode)
        printf("  %.3f ms/token\n", 1e3 / med);
    else {
        stats(r->ttft, r->runs, &med, &lo, &hi);
        printf("  TTFT %.1f ms\n", 1e3 * med);
    }
    fflush(stdout);
    return ATH_OK;
}

/* ---- the machine and the model, for the report ------------------------------- */

/* The first line of path starting with prefix, after it and any ": ", into
 * out; "" if there is none. */
static void read_field(const char *path, const char *prefix, char *out, size_t cap)
{
    out[0] = '\0';
    FILE *f = fopen(path, "r");
    if (!f)
        return;
    char line[512];
    size_t pl = strlen(prefix);
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, prefix, pl))
            continue;
        const char *v = line + pl;
        while (*v == ' ' || *v == '\t' || *v == ':')
            v++;
        snprintf(out, cap, "%s", v);
        out[strcspn(out, "\n")] = '\0';
        break;
    }
    fclose(f);
}

/* The model of the block device that holds path, from sysfs; "" if unknown. */
static void storage_of(const char *path, char *out, size_t cap)
{
    out[0] = '\0';
    struct stat sb;
    char sys[128], real[PATH_MAX], model[PATH_MAX + 16];
    if (stat(path, &sb) || !snprintf(sys, sizeof sys, "/sys/dev/block/%u:%u", major(sb.st_dev), minor(sb.st_dev)) ||
        !realpath(sys, real))
        return;
    for (int up = 0; up < 2 && !out[0]; up++) {     /* the partition, then its disk */
        snprintf(model, sizeof model, "%s/device/model", real);
        read_field(model, "", out, cap);
        char *slash = strrchr(real, '/');
        if (!slash)
            break;
        *slash = '\0';
    }
}

static ath_status put_str(ath_buf *o, const char *key, const char *v)
{
    ATH_CHECK(ath_buf_printf(o, "\"%s\": ", key));
    return ath_json_write_str(o, v, strlen(v));
}

static ath_status write_stats(ath_buf *o, const char *key, const double *v, int32_t n, double scale_num, bool rate)
{
    double x[MAX_REPS], med, lo, hi;
    for (int32_t i = 0; i < n; i++)
        x[i] = rate ? scale_num / v[i] : scale_num * v[i];
    stats(x, n, &med, &lo, &hi);
    ATH_CHECK(ath_buf_printf(o, "\"%s\": {\"median\": ", key));
    ATH_CHECK(ath_json_write_double(o, med));
    ATH_CHECK(ath_buf_puts(o, ", \"min\": "));
    ATH_CHECK(ath_json_write_double(o, lo));
    ATH_CHECK(ath_buf_puts(o, ", \"max\": "));
    ATH_CHECK(ath_json_write_double(o, hi));
    ATH_CHECK(ath_buf_puts(o, ", \"runs\": ["));
    for (int32_t i = 0; i < n; i++) {
        if (i)
            ATH_CHECK(ath_buf_puts(o, ", "));
        ATH_CHECK(ath_json_write_double(o, x[i]));
    }
    return ath_buf_puts(o, "]}");
}

static ath_status write_report(const bench *b, const options *o, const result *res, int32_t n_res, ath_arena *a)
{
    ath_buf out;
    ath_buf_init(&out, a);
    ath_buf *j = &out;
    char s1[512], s2[512], s3[1100];
    time_t t = time(NULL);
    struct tm tm;
    gmtime_r(&t, &tm);
    strftime(s1, sizeof s1, "%Y-%m-%dT%H:%M:%SZ", &tm);
    ATH_CHECK(ath_buf_puts(j, "{\n  \"athanor_bench\": 1,\n  "));
    ATH_CHECK(put_str(j, "date", s1));

    /* hardware */
    ATH_CHECK(ath_buf_puts(j, ",\n  \"hardware\": {"));
    if (b->e) {
        const ath_device *d = &b->e->dev;
        ATH_CHECK(put_str(j, "gpu", d->name));
        ATH_CHECK(ath_buf_printf(j, ", \"gpu_memory_bytes\": %llu, \"compute_capability\": \"%d.%d\", ",
                                 (unsigned long long)d->mem_total, d->cc_major, d->cc_minor));
        ATH_CHECK(put_str(j, "pci_bus_id", d->pci_bus_id));
        char lower[32];
        snprintf(lower, sizeof lower, "%s", d->pci_bus_id);
        for (char *c = lower; *c; c++)
            *c = (char)(*c >= 'A' && *c <= 'F' ? *c - 'A' + 'a' : *c);
        snprintf(s3, sizeof s3, "/sys/bus/pci/devices/%s/max_link_speed", lower);
        read_field(s3, "", s1, sizeof s1);
        snprintf(s3, sizeof s3, "/sys/bus/pci/devices/%s/max_link_width", lower);
        read_field(s3, "", s2, sizeof s2);
        snprintf(s3, sizeof s3, "%s x%s (maximum of the link)", s1, s2);
        ATH_CHECK(ath_buf_puts(j, ", "));
        ATH_CHECK(put_str(j, "pcie", s3));
        read_field("/proc/driver/nvidia/version", "NVRM version", s1, sizeof s1);
        ATH_CHECK(ath_buf_puts(j, ", "));
        ATH_CHECK(put_str(j, "driver", s1));
        ATH_CHECK(ath_buf_puts(j, ", "));
    }
    read_field("/proc/cpuinfo", "model name", s1, sizeof s1);
    ATH_CHECK(put_str(j, "cpu", s1));
    ATH_CHECK(ath_buf_printf(j, ", \"cpu_threads\": %ld, ", sysconf(_SC_NPROCESSORS_ONLN)));
    read_field("/proc/meminfo", "MemTotal", s1, sizeof s1);
    ATH_CHECK(put_str(j, "ram", s1));
    ATH_CHECK(ath_buf_puts(j, ", \"ram_speed\": \"not recorded (reading it needs root)\", "));
    storage_of(o->model, s1, sizeof s1);
    ATH_CHECK(put_str(j, "storage", s1));
    ATH_CHECK(ath_buf_puts(j, "}"));

    /* software */
    ATH_CHECK(ath_buf_puts(j, ",\n  \"software\": {"));
    ATH_CHECK(put_str(j, "athanor_commit", ATH_COMMIT));
    ATH_CHECK(ath_buf_puts(j, ", "));
    ATH_CHECK(put_str(j, "compiler", "gcc " __VERSION__));
    ATH_CHECK(ath_buf_puts(j, ", "));
    ATH_CHECK(put_str(j, "gpu_backend", ath_gpu_backend()));
    if (b->e)
        ATH_CHECK(ath_buf_printf(j, ", \"runtime_version\": %d, \"driver_api_version\": %d, \"blas_version\": %d",
                                 b->e->dev.runtime_version, b->e->dev.driver_version, b->e->dev.blas_version));
    ATH_CHECK(ath_buf_puts(j, "}"));

    /* model */
    ATH_CHECK(ath_buf_puts(j, ",\n  \"model\": {"));
    ATH_CHECK(put_str(j, "dir", o->model));
    ATH_CHECK(ath_buf_puts(j, ", "));
    ATH_CHECK(put_str(j, "family", b->m.family->name));
    /* the format and the dtypes the params have, as stored (step 12: a GGUF of Q8_0) */
    char dts[128] = "";
    bool seen[ATH_DTYPE_QUANT_LAST + 1] = {false};
    for (uint32_t i = 0; i < b->m.n_params; i++) {
        const ath_dtype dt = b->m.params[i].t.dtype;
        if ((b->m.params[i].flags & ATH_P_UNUSED) || (unsigned)dt > ATH_DTYPE_QUANT_LAST || seen[dt])
            continue;
        seen[dt] = true;
        snprintf(dts + strlen(dts), sizeof dts - strlen(dts), "%s%s", dts[0] ? ", " : "", ath_dtype_name(dt));
    }
    ATH_CHECK(ath_buf_printf(j, ", \"weights\": \"as stored in the checkpoint (%s: %s)\", \"files\": [",
                             ath_ckpt_format_name(b->m.ckpt.format), dts));
    DIR *dir = opendir(o->model);
    int32_t nf = 0;
    for (struct dirent *de; dir && (de = readdir(dir)) && nf < MAX_FILES;) {
        size_t l = strlen(de->d_name);
        if ((l < 12 || strcmp(de->d_name + l - 12, ".safetensors")) &&
            (l < 5 || strcmp(de->d_name + l - 5, ".gguf") || !strncmp(de->d_name, "mmproj-", 7)))
            continue;
        char path[ATH_PATH_MAX], hex[65] = "not hashed (--no-hash)", meta[ATH_PATH_MAX];
        ATH_CHECK(ath_path_join(path, sizeof path, o->model, de->d_name));
        if (o->hash)
            ATH_CHECK(ath_sha256_file(path, hex));
        snprintf(meta, sizeof meta, "%s/.cache/huggingface/download/%s.metadata", o->model, de->d_name);
        read_field(meta, "", s1, sizeof s1);   /* first line: the revision */
        struct stat sb;
        ATH_CHECK(ath_buf_puts(j, nf++ ? ", {" : "{"));
        ATH_CHECK(put_str(j, "name", de->d_name));
        ATH_CHECK(ath_buf_printf(j, ", \"bytes\": %lld, ", stat(path, &sb) ? -1LL : (long long)sb.st_size));
        ATH_CHECK(put_str(j, "sha256", hex));
        ATH_CHECK(ath_buf_puts(j, ", "));
        ATH_CHECK(put_str(j, "hf_revision", s1));
        ATH_CHECK(ath_buf_puts(j, "}"));
    }
    if (dir)
        closedir(dir);
    ATH_CHECK(ath_buf_puts(j, "]}"));

    /* workload and method */
    ATH_CHECK(ath_buf_printf(j,
                             ",\n  \"workload\": {\"tokens\": \"pseudo-random ids (ath_rng, seed %llu)\", "
                             "\"batch\": 1, \"prompt_chunk\": %d, \"context\": %d, \"activations\": \"bf16\", "
                             "\"sampling\": \"greedy\", \"decode\": \"%s\", \"device\": \"%s\"}",
                             (unsigned long long)o->seed, o->chunk, b->st.capacity,
                             b->spec      ? "speculative: the draft head's guess verified in one pass with the "
                                            "next token, forward calls from the host, ath_sample_ref (step 13)"
                             : b->decoder ? "decoder: one recorded graph per token, the choice on the GPU (step 9)"
                                          : "one forward call per token, logits to the host, ath_sample_ref (step 8)",
                             b->e ? "gpu" : "cpu"));
    if (o->prompt_ids) {
        ATH_CHECK(ath_buf_puts(j, ",\n  \"prompt_ids\": "));
        ATH_CHECK(ath_json_write_str(j, o->prompt_ids, strlen(o->prompt_ids)));
    }
    if (b->e && b->e->tier) {   /* where the routed experts were (step 12) */
        const ath_placement *t = b->e->tier;
        ATH_CHECK(ath_buf_printf(j,
                                 ",\n  \"experts\": {\"per_layer\": %lld, \"in_device_memory\": %d, "
                                 "\"bytes_each\": %llu, \"host_bytes\": %llu, \"moved\": %llu, \"period\": %d, "
                                 "\"moves\": %d, \"cpu_threads\": %d}",
                                 (long long)t->n_experts, t->slots, (unsigned long long)t->expert_bytes,
                                 (unsigned long long)t->host_bytes, (unsigned long long)t->moved, t->o.period,
                                 t->o.moves, ath_pool_threads(&t->pool)));
    }
    ATH_CHECK(ath_buf_printf(j,
                             ",\n  \"method\": {\"warmup_runs\": %d, \"measured_runs\": %d, \"statistic\": "
                             "\"median, with min and max\", \"timer\": \"CLOCK_MONOTONIC around whole calls; "
                             "a GPU pass waits for its logits\"}",
                             o->warmup, o->reps));

    /* results */
    ATH_CHECK(ath_buf_puts(j, ",\n  \"results\": ["));
    for (int32_t i = 0; i < n_res; i++) {
        const result *r = &res[i];
        ATH_CHECK(ath_buf_puts(j, i ? ",\n    {" : "\n    {"));
        ATH_CHECK(put_str(j, "test", r->name));
        ATH_CHECK(ath_buf_printf(j, ", \"%s\": %lld, \"depth\": %lld, ", r->decode ? "generated" : "prompt",
                                 (long long)r->n, (long long)r->depth));
        ATH_CHECK(write_stats(j, "tokens_per_s", r->s, r->runs, (double)r->n, true));
        if (!r->decode) {
            ATH_CHECK(ath_buf_puts(j, ", "));
            ATH_CHECK(write_stats(j, "ttft_ms", r->ttft, r->runs, 1e3, false));
        } else {
            ATH_CHECK(ath_buf_puts(j, ", "));
            ATH_CHECK(write_stats(j, "ms_per_token", r->s, r->runs, 1e3 / (double)r->n, false));
            if (b->spec)
                ATH_CHECK(ath_buf_printf(j, ", \"guesses_kept\": %.4f", r->kept));
        }
        ATH_CHECK(ath_buf_puts(j, "}"));
    }
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    ATH_CHECK(ath_buf_printf(j,
                             "\n  ],\n  \"memory\": {\"device_used_before_bytes\": %llu, "
                             "\"device_used_peak_bytes\": %llu, \"device_note\": \"all processes, from the "
                             "driver (cudaMemGetInfo); before = after this process created its context\", "
                             "\"process_peak_rss_bytes\": %lld}\n}\n",
                             (unsigned long long)b->dev_used_before, (unsigned long long)b->dev_used_peak,
                             (long long)ru.ru_maxrss * 1024));
    FILE *f = fopen(o->json, "w");
    if (!f) {
        fprintf(stderr, "athanor bench: cannot write %s\n", o->json);
        return ATH_ERR_IO;
    }
    fputs(ath_buf_str(j), f);
    fclose(f);
    return ATH_OK;
}

/* ---- the command ------------------------------------------------------------------ */

static int parse(int argc, char **argv, options *o)
{
    *o = (options){.gen = 128, .reps = 5, .warmup = 1, .chunk = 2048, .seed = 1, .hash = true, .experts = -1};
    parse_list("512,4096", false, o->prompts, &o->n_prompts);
    parse_list("0,4096,32768", true, o->depths, &o->n_depths);
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        int64_t x;
        bool ok = true, takes = true;
        if (!strcmp(a, "-p") || !strcmp(a, "--prompt"))
            ok = v && parse_list(v, false, o->prompts, &o->n_prompts);
        else if (!strcmp(a, "-d") || !strcmp(a, "--depth"))
            ok = v && parse_list(v, true, o->depths, &o->n_depths);
        else if ((!strcmp(a, "-n") || !strcmp(a, "--gen")) && v && parse_int(v, 0, 1 << 20, &x))
            o->gen = (int32_t)x;
        else if ((!strcmp(a, "-r") || !strcmp(a, "--reps")) && v && parse_int(v, 1, MAX_REPS, &x))
            o->reps = (int32_t)x;
        else if (!strcmp(a, "--warmup") && v && parse_int(v, 0, 100, &x))
            o->warmup = (int32_t)x;
        else if (!strcmp(a, "--chunk") && v && parse_int(v, 1, INT32_MAX, &x))
            o->chunk = (int32_t)x;
        else if (!strcmp(a, "--threads") && v && parse_int(v, 1, 1024, &x))
            o->threads = (int32_t)x;
        else if (!strcmp(a, "--experts") && v && parse_int(v, 0, 1 << 20, &x))
            o->experts = (int32_t)x;
        else if (!strcmp(a, "--prompt-ids") && v)
            o->prompt_ids = v;
        else if (!strcmp(a, "--seed") && v && parse_int(v, 0, INT64_MAX, &x))
            o->seed = (uint64_t)x;
        else if (!strcmp(a, "--device") && v && (!strcmp(v, "cpu") || !strcmp(v, "gpu")))
            o->device = v;
        else if (!strcmp(a, "--json") && v)
            o->json = v;
        else if (!strcmp(a, "--direct"))
            o->direct = true, takes = false;
        else if (!strcmp(a, "--mtp"))
            o->mtp = true, takes = false;
        else if (!strcmp(a, "--kv") && v && (!strcmp(v, "bf16") || !strcmp(v, "q8")))
            o->kv8 = !strcmp(v, "q8");
        else if (!strcmp(a, "--no-hash"))
            o->hash = false, takes = false;
        else if (a[0] != '-' && !o->model)
            o->model = a, takes = false;
        else
            ok = false;
        if (!ok) {
            fprintf(stderr, "athanor bench: bad argument %s%s%s\n", a, v && takes ? " " : "", v && takes ? v : "");
            return ATH_EXIT_USAGE;
        }
        if (takes)
            i++;
    }
    if (!o->model) {
        ath_cmd_bench_usage(stderr);
        return ATH_EXIT_USAGE;
    }
    return ATH_EXIT_OK;
}

int ath_cmd_bench(ath_arena *a, int argc, char **argv)
{
    options o;
    int rc = parse(argc, argv, &o);
    if (rc != ATH_EXIT_OK)
        return rc;
    bench b = {.greedy = {.temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0}};
    ath_arena scratch;
    ath_arena_init(&scratch, 0);
    b.scratch = &scratch;

    /* the largest context a test needs */
    int64_t ctx = 1;
    /* --prompt-ids: one prompt of those tokens, and decoding after all but the
     * last (whose step gives the first token generated) */
    int32_t *pids = NULL;
    int64_t n_pids = 0;
    if (o.prompt_ids) {
        ath_file f;
        if (ath_file_open(&f, o.prompt_ids) != ATH_OK || !f.size) {
            fprintf(stderr, "athanor bench: cannot read %s\n", o.prompt_ids);
            return ATH_EXIT_FAILURE;
        }
        pids = ath_arena_alloc(a, f.size * sizeof *pids, 64);
        const char *c = f.data, *end = c + f.size;
        while (pids && c < end) {
            while (c < end && (*c < '0' || *c > '9'))
                c++;
            int64_t v = 0;
            bool any = false;
            while (c < end && *c >= '0' && *c <= '9')
                v = v * 10 + (*c++ - '0'), any = true;
            if (any)
                pids[n_pids++] = (int32_t)v;
        }
        ath_file_close(&f);
        if (!pids || n_pids < 2) {
            fprintf(stderr, "athanor bench: %s holds fewer than 2 token ids\n", o.prompt_ids);
            return ATH_EXIT_FAILURE;
        }
        o.n_prompts = 1, o.prompts[0] = n_pids;
        o.n_depths = 1, o.depths[0] = n_pids - 1;
    }
    for (int32_t i = 0; i < o.n_prompts; i++)
        ctx = o.prompts[i] + 1 > ctx ? o.prompts[i] + 1 : ctx;
    for (int32_t i = 0; o.gen > 0 && i < o.n_depths; i++)
        ctx = o.depths[i] + o.gen + 1 > ctx ? o.depths[i] + o.gen + 1 : ctx;

    ath_status st = ath_model_open(&b.m, a, o.model);
    bool gpu = o.device ? !strcmp(o.device, "gpu") : ath_gpu_count() > 0;
    if (st == ATH_OK && gpu) {
        ath_device probe;
        uint64_t free_b, total_b;
        if (ath_device_open(&probe, 0) == ATH_OK) {
            if (ath_device_mem(&probe, &free_b, &total_b) == ATH_OK)
                b.dev_used_before = total_b - free_b;
            ath_device_close(&probe);
        }
        b.e = ath_arena_zalloc(a, sizeof *b.e, _Alignof(ath_engine));
        const ath_place_opts po = {.slots = o.experts, .draft = o.mtp};
        st = b.e ? ath_engine_open_tier(b.e, &b.m, 0, &po, a) : ATH_ERR_NOMEM;
        if (st == ATH_OK) {
            b.exec = b.e->exec;
            st = ath_engine_state_kv(b.e, &b.st, ATH_DTYPE_BF16, o.kv8 ? ATH_DTYPE_I8 : ATH_DTYPE_BF16, (int32_t)ctx, a);
        }
        if (st == ATH_OK && !o.direct && !o.mtp) {
            st = ath_decoder_open(&b.dc, b.e, &b.st, &b.greedy, a);
            b.decoder = st == ATH_OK;
        }
    } else if (st == ATH_OK) {
        st = ath_pool_init(&b.pool, o.threads ? o.threads : ath_pool_default_threads(), a);
        b.pool_open = st == ATH_OK;
        b.exec = (ath_exec){.pool = &b.pool};
        if (st == ATH_OK)
            st = ath_state_init_kv(&b.st, &b.m, ATH_DTYPE_BF16, o.kv8 ? ATH_DTYPE_I8 : ATH_DTYPE_BF16, (int32_t)ctx, a);
    }
    if (st == ATH_OK && o.mtp) {
        st = ath_spec_open(&b.sp, &b.m, &b.st, &b.exec, &b.greedy, o.chunk, a);
        b.spec = st == ATH_OK;
        if (st == ATH_ERR_UNSUPPORTED)
            fprintf(stderr, "athanor bench: --mtp: the model has no draft head\n");
    }
    if (st == ATH_OK) {
        float *lg = ath_arena_alloc(a, (size_t)b.m.vocab_size * sizeof(float), 64);
        b.n_ids = ctx;
        b.ids = ath_arena_alloc(a, (size_t)ctx * sizeof *b.ids, 64);
        st = lg && b.ids ? ath_tensor_init(&b.logits, ATH_DTYPE_F32, 2, (int64_t[]){1, b.m.vocab_size}, lg,
                                           ATH_TIER_RAM)
                         : ATH_ERR_NOMEM;
        ath_rng rng;
        ath_rng_init(&rng, o.seed);
        for (int64_t i = 0; st == ATH_OK && i < ctx; i++)
            b.ids[i] = (int32_t)(ath_rng_u64(&rng) % (uint64_t)b.m.vocab_size);
        for (int64_t i = 0; st == ATH_OK && i < n_pids; i++) {
            if (pids[i] < 0 || pids[i] >= b.m.vocab_size)
                st = ATH_ERR_RANGE;
            else
                b.ids[i] = pids[i];
        }
    }

    result *res = ath_arena_zalloc(a, sizeof *res * (size_t)(MAX_LIST * 2), _Alignof(result));
    int32_t n_res = 0;
    if (st == ATH_OK && !res)
        st = ATH_ERR_NOMEM;
    if (st == ATH_OK)
        printf("athanor bench: %s on %s, context %lld, %d runs after %d warm-up\n", b.m.family->name,
               b.e ? b.e->dev.name : "the CPU", (long long)ctx, o.reps, o.warmup);
    for (int32_t i = 0; st == ATH_OK && i < o.n_prompts; i++) {
        result *r = &res[n_res++];
        r->n = o.prompts[i];
        snprintf(r->name, sizeof r->name, "pp%lld", (long long)r->n);
        st = run_test(&b, &o, r);
    }
    for (int32_t i = 0; st == ATH_OK && o.gen > 0 && i < o.n_depths; i++) {
        result *r = &res[n_res++];
        r->n = o.gen, r->depth = o.depths[i], r->decode = true;
        snprintf(r->name, sizeof r->name, "tg%d@d%lld", o.gen, (long long)r->depth);
        st = run_test(&b, &o, r);
    }
    if (st == ATH_OK && o.json)
        st = write_report(&b, &o, res, n_res, a);
    if (st == ATH_OK && o.json)
        printf("report: %s\n", o.json);

    if (b.decoder)
        ath_decoder_close(&b.dc);
    if (b.spec)
        ath_spec_close(&b.sp);
    if (b.e)
        ath_engine_close(b.e);
    if (b.pool_open)
        ath_pool_free(&b.pool);
    ath_model_close(&b.m);
    ath_arena_free(&scratch);
    if (st != ATH_OK) {
        fprintf(stderr, "athanor bench: %s\n", ath_status_str(st));
        return ATH_EXIT_FAILURE;
    }
    return ATH_EXIT_OK;
}
