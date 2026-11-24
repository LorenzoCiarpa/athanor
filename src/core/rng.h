/* rng.h: a small, seedable pseudo-random generator.
 *
 * xoshiro256** (Blackman and Vigna), seeded through splitmix64: fast, good
 * statistical quality, and the same sequence on every machine for the same
 * seed, so a sampled generation can be reproduced. Not for cryptography.
 * The state is a value the caller owns: no global generator.
 */
#ifndef ATH_RNG_H
#define ATH_RNG_H

#include "core/base.h"

typedef struct {
    uint64_t s[4];
} ath_rng;

void     ath_rng_init(ath_rng *r, uint64_t seed);
uint64_t ath_rng_u64(ath_rng *r);
/* Uniform in [0, 1), 53 random bits. */
double   ath_rng_uniform(ath_rng *r);

#endif /* ATH_RNG_H */
