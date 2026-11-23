/* test.h: the minimal test runner shared by every tests/test_*.c.
 *
 *   ATH_TEST(arena_grows) {
 *       ATH_EXPECT(x == 3);                    // records a failure, continues
 *       ATH_REQUIRE(p != NULL);                // records a failure, ends the test
 *       ATH_EXPECT_OK(ath_something());
 *   }
 *   ...
 *   static const ath_test_case tests[] = { ATH_TEST_ENTRY(arena_grows), ... };
 *   int main(int argc, char **argv) { return ath_test_main(tests, ATH_ARRAY_LEN(tests), argc, argv); }
 *
 * `./build/test_core arena` runs only the tests whose name contains "arena".
 * Library logging is silenced unless ATH_LOG is set, because many tests feed
 * deliberately broken input. Header-only: tests are single translation units.
 */
#ifndef ATH_TEST_H
#define ATH_TEST_H

#include "core/base.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *name;
    int         failures;
    bool        skipped;
} ath_test;

typedef struct {
    const char *name;
    void      (*fn)(ath_test *t);
} ath_test_case;

#define ATH_TEST(name) static void test_##name(ath_test *t_)
#define ATH_TEST_ENTRY(name) {#name, test_##name}

static inline void ath_test_fail_(ath_test *t, const char *file, int line, const char *what)
{
    t->failures++;
    printf("  FAIL %s (%s:%d): %s\n", t->name, file, line, what);
}

#define ATH_EXPECT(cond)                                                      \
    do {                                                                      \
        if (!(cond))                                                          \
            ath_test_fail_(t_, __FILE__, __LINE__, #cond);                    \
    } while (0)

#define ATH_REQUIRE(cond)                                                     \
    do {                                                                      \
        if (!(cond)) {                                                        \
            ath_test_fail_(t_, __FILE__, __LINE__, "required: " #cond);       \
            return;                                                           \
        }                                                                     \
    } while (0)

#define ATH_EXPECT_STATUS(expr, want)                                         \
    do {                                                                      \
        ath_status got_ = (expr), want_ = (want);                             \
        if (got_ != want_) {                                                  \
            char msg_[512];                                                   \
            snprintf(msg_, sizeof msg_, "%s: got '%s', want '%s'", #expr,     \
                     ath_status_str(got_), ath_status_str(want_));            \
            ath_test_fail_(t_, __FILE__, __LINE__, msg_);                     \
        }                                                                     \
    } while (0)

#define ATH_EXPECT_OK(expr) ATH_EXPECT_STATUS(expr, ATH_OK)

#define ATH_REQUIRE_OK(expr)                                                  \
    do {                                                                      \
        ath_status got_ = (expr);                                             \
        if (got_ != ATH_OK) {                                                 \
            char msg_[512];                                                   \
            snprintf(msg_, sizeof msg_, "required ok: %s: got '%s'", #expr,   \
                     ath_status_str(got_));                                   \
            ath_test_fail_(t_, __FILE__, __LINE__, msg_);                     \
            return;                                                           \
        }                                                                     \
    } while (0)

#define ATH_EXPECT_EQ_INT(a, b)                                               \
    do {                                                                      \
        int64_t a_ = (int64_t)(a), b_ = (int64_t)(b);                         \
        if (a_ != b_) {                                                       \
            char msg_[512];                                                   \
            snprintf(msg_, sizeof msg_, "%s == %s: %" PRId64 " != %" PRId64,  \
                     #a, #b, a_, b_);                                         \
            ath_test_fail_(t_, __FILE__, __LINE__, msg_);                     \
        }                                                                     \
    } while (0)

#define ATH_EXPECT_EQ_U64(a, b)                                               \
    do {                                                                      \
        uint64_t a_ = (uint64_t)(a), b_ = (uint64_t)(b);                      \
        if (a_ != b_) {                                                       \
            char msg_[512];                                                   \
            snprintf(msg_, sizeof msg_, "%s == %s: %" PRIu64 " != %" PRIu64,  \
                     #a, #b, a_, b_);                                         \
            ath_test_fail_(t_, __FILE__, __LINE__, msg_);                     \
        }                                                                     \
    } while (0)

#define ATH_EXPECT_STR_EQ(a, b)                                               \
    do {                                                                      \
        const char *a_ = (a), *b_ = (b);                                      \
        if (!a_ || !b_ || strcmp(a_, b_) != 0) {                              \
            char msg_[512];                                                   \
            snprintf(msg_, sizeof msg_, "%s == %s: \"%s\" != \"%s\"", #a, #b, \
                     a_ ? a_ : "(null)", b_ ? b_ : "(null)");                 \
            ath_test_fail_(t_, __FILE__, __LINE__, msg_);                     \
        }                                                                     \
    } while (0)

#define ATH_EXPECT_NEAR(a, b, tol)                                            \
    do {                                                                      \
        double a_ = (double)(a), b_ = (double)(b);                            \
        if (!(fabs(a_ - b_) <= (tol))) {                                      \
            char msg_[512];                                                   \
            snprintf(msg_, sizeof msg_, "%s ~= %s: %.17g vs %.17g", #a, #b,   \
                     a_, b_);                                                 \
            ath_test_fail_(t_, __FILE__, __LINE__, msg_);                     \
        }                                                                     \
    } while (0)

/* End the test as skipped (for example when the real model is not downloaded). */
#define ATH_SKIP(why)                                                         \
    do {                                                                      \
        t_->skipped = true;                                                   \
        printf("  SKIP %s: %s\n", t_->name, why);                             \
        return;                                                               \
    } while (0)

static inline int ath_test_main(const ath_test_case *cases, size_t n, int argc, char **argv)
{
    if (!getenv("ATH_LOG"))
        setenv("ATH_LOG", "none", 1);
    const char *filter = argc > 1 ? argv[1] : NULL;
    int passed = 0, failed = 0, skipped = 0;
    for (size_t i = 0; i < n; i++) {
        if (filter && !strstr(cases[i].name, filter))
            continue;
        ath_test t = {cases[i].name, 0, false};
        cases[i].fn(&t);
        if (t.failures)
            failed++;
        else if (t.skipped)
            skipped++;
        else
            passed++;
    }
    printf("%d passed, %d failed, %d skipped\n", passed, failed, skipped);
    return failed ? 1 : 0;
}

#endif /* ATH_TEST_H */
