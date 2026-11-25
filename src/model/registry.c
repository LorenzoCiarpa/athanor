/* registry.c: compiled-in families, model loading and the coverage check.
 * See model.h.
 *
 * The family list comes from the Makefile: FAMILIES=<name> writes the line
 * ATH_FAMILY(<name>) into the generated header families.h, and each family
 * defines `const ath_family ath_family_<name>`. This file therefore names no
 * family (AGENTS.md, invariant 2) and a build with a different FAMILIES list
 * needs no edit here.
 */
#include "model/model.h"

#include "core/file.h"

#include <stdio.h>
#include <string.h>

#define ATH_FAMILY(f) extern const ath_family ath_family_##f;
#include "families.h"
#undef ATH_FAMILY

#define ATH_FAMILY(f) &ath_family_##f,
static const ath_family *const families[] = {
#include "families.h"
    NULL, /* keeps the initializer valid when FAMILIES is empty */
};
#undef ATH_FAMILY

size_t ath_family_count(void) { return ATH_ARRAY_LEN(families) - 1; }

const ath_family *ath_family_at(size_t i) { return i < ath_family_count() ? families[i] : NULL; }

static const char *const role_names[] = {
    "embed", "lm_head", "norm", "mix_in", "mix_out", "mlp_in", "mlp_out",
    "router", "expert_in", "expert_out", "conv", "gate", "other",
};
_Static_assert(ATH_ARRAY_LEN(role_names) == ATH_ROLE_COUNT, "one name per role");

const char *ath_role_name(ath_role r)
{
    return (unsigned)r < ATH_ROLE_COUNT ? role_names[r] : "?";
}

/* ---- loading ------------------------------------------------------------------ */

static ath_status load_config(ath_arena *a, const char *dir, const ath_json **cfg)
{
    char path[ATH_PATH_MAX];
    ATH_CHECK(ath_path_join(path, sizeof path, dir, "config.json"));
    if (!ath_file_exists(path)) {
        ath_log(ATH_LOG_ERROR, "%s: not found", path);
        return ATH_ERR_NOT_FOUND;
    }
    ath_file f;
    ATH_CHECK(ath_file_open(&f, path));
    ath_json_error err;
    ath_status s = ath_json_parse(a, f.data, f.size, cfg, &err);
    ath_file_close(&f);
    if (s == ATH_ERR_PARSE)
        ath_log(ATH_LOG_ERROR, "%s: line %zu column %zu: %s", path, err.line, err.col, err.msg);
    return s;
}

static const ath_family *find_family(const ath_json *cfg)
{
    for (size_t i = 0; i < ath_family_count(); i++)
        if (families[i]->match(cfg))
            return families[i];
    return NULL;
}

static void log_no_family(const char *dir, const ath_json *cfg)
{
    const char *type = "(none)";
    ath_json_to_str(ath_json_get(cfg, "model_type"), &type, NULL);
    char list[256] = "";
    size_t n = 0;
    for (size_t i = 0; i < ath_family_count() && n < sizeof list; i++)
        n += (size_t)snprintf(list + n, sizeof list - n, "%s%s", i ? ", " : "", families[i]->name);
    ath_log(ATH_LOG_ERROR, "%s: no compiled-in family handles model_type '%s' (families: %s)", dir,
            type, ath_family_count() ? list : "none");
}

/* ---- coverage ----------------------------------------------------------------- */

/* Per checkpoint tensor: viewed by a param the text path uses, viewed by an
 * ATH_P_UNUSED param, in an ignored group. */
enum { MARK_USED = 1, MARK_UNUSED = 2, MARK_GROUP = 4 };

/* Mark the checkpoint tensor behind every param, checking that the param is a
 * contiguous view inside it. */
static ath_status mark_params(const ath_model *m, uint8_t *mark)
{
    const ath_ckpt *st = &m->ckpt;
    for (uint32_t i = 0; i < m->n_params; i++) {
        const ath_param *p = &m->params[i];
        ATH_ASSERT((unsigned)p->role < ATH_ROLE_COUNT);
        ATH_ASSERT(p->layer >= -1 && p->layer < m->n_layers);
        ATH_ASSERT(p->expert >= -1 && (p->expert < m->n_experts || p->expert == -1));
        const ath_ckpt_tensor *e = p->name ? ath_ckpt_find(st, p->name) : NULL;
        if (!e) {
            ath_log(ATH_LOG_ERROR, "%s: param %u names no checkpoint tensor ('%s')",
                    m->family->name, i, p->name ? p->name : "(null)");
            return ATH_ERR_FORMAT;
        }
        const char *lo = e->t.data, *at = p->t.data;
        size_t      n = ath_tensor_nbytes(&p->t), size = ath_tensor_nbytes(&e->t);
        if (!at || !ath_tensor_is_contiguous(&p->t) || at < lo || (size_t)(at - lo) > size ||
            n > size - (size_t)(at - lo)) {
            ath_log(ATH_LOG_ERROR, "%s: param %u (%s, expert %d) is not a contiguous view inside its tensor",
                    m->family->name, i, p->name, p->expert);
            return ATH_ERR_FORMAT;
        }
        mark[e - st->tensors] |= (p->flags & ATH_P_UNUSED) ? MARK_UNUSED : MARK_USED;
    }
    return ATH_OK;
}

/* Count the tensors of every ignored group, and check that each is in one
 * group only and bound as unused, never as a param the text path uses. */
static ath_status mark_groups(ath_model *m, uint8_t *mark)
{
    const ath_ckpt *st = &m->ckpt;
    for (uint32_t g = 0; g < m->n_ignored; g++) {
        ath_ignored *ig = &m->ignored[g];
        size_t       plen = strlen(ig->prefix);
        ig->n_tensors = 0;
        ig->bytes = 0;
        for (uint32_t i = 0; i < st->n_tensors; i++) {
            const ath_ckpt_tensor *e = &st->tensors[i];
            if (strncmp(e->name, ig->prefix, plen) != 0)
                continue;
            const char *bad = (mark[i] & MARK_GROUP)    ? "is in another ignored group too"
                              : (mark[i] & MARK_USED)   ? "is bound to a param the text path uses"
                              : !(mark[i] & MARK_UNUSED) ? "is not bound"
                                                         : NULL;
            if (bad) {
                ath_log(ATH_LOG_ERROR, "%s: tensor %s of ignored group '%s' %s", m->family->name, e->name,
                        ig->prefix, bad);
                return ATH_ERR_FORMAT;
            }
            mark[i] |= MARK_GROUP;
            ig->n_tensors++;
            ig->bytes += e->bytes;
        }
    }
    return ATH_OK;
}

/* Every checkpoint tensor must be bound: a tensor the family does not know
 * about means the checkpoint is not what the family thinks it is. A tensor
 * bound as unused must be in an ignored group, so that the family says why. */
static ath_status check_coverage(ath_model *m, ath_arena *a)
{
    const ath_ckpt *st = &m->ckpt;
    ath_arena_pos pos = ath_arena_mark(a);
    uint8_t *mark = ath_arena_zalloc(a, st->n_tensors ? st->n_tensors : 1, 1);
    if (!mark)
        return ATH_ERR_NOMEM;

    ath_status s = mark_params(m, mark);
    if (s == ATH_OK)
        s = mark_groups(m, mark);
    if (s == ATH_OK) {
        uint32_t unbound = 0;
        m->n_bound = 0;
        for (uint32_t i = 0; i < st->n_tensors; i++) {
            if ((mark[i] & MARK_UNUSED) && !(mark[i] & MARK_GROUP)) {
                ath_log(ATH_LOG_ERROR, "%s: tensor %s is bound as unused but in no ignored group",
                        m->family->name, st->tensors[i].name);
                s = ATH_ERR_FORMAT;
            }
            if (mark[i] & (MARK_USED | MARK_UNUSED)) {
                m->n_bound++;
                continue;
            }
            if (unbound < 10)
                ath_log(ATH_LOG_ERROR, "%s: tensor %s is not bound", m->family->name, st->tensors[i].name);
            unbound++;
        }
        if (unbound > 10)
            ath_log(ATH_LOG_ERROR, "%s: ... %u unbound tensors in total", m->family->name, unbound);
        if (unbound)
            s = ATH_ERR_FORMAT;
    }
    ath_arena_reset(a, pos);
    return s;
}

/* ---- public ------------------------------------------------------------------- */

ath_status ath_model_open(ath_model *m, ath_arena *a, const char *dir)
{
    memset(m, 0, sizeof *m);
    ATH_CHECK(load_config(a, dir, &m->config));
    m->family = find_family(m->config);
    if (!m->family) {
        log_no_family(dir, m->config);
        return ATH_ERR_UNSUPPORTED;
    }
    ATH_CHECK(ath_ckpt_open(&m->ckpt, a, dir));
    ath_status s = m->family->load(m, a);
    if (s == ATH_OK)
        s = check_coverage(m, a);
    if (s != ATH_OK) {
        ath_ckpt_close(&m->ckpt);
        memset(m, 0, sizeof *m);
    }
    return s;
}

void ath_model_close(ath_model *m)
{
    ath_ckpt_close(&m->ckpt);
}
