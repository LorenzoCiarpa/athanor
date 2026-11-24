/* rng.c: xoshiro256** seeded by splitmix64. See rng.h. */
#include "core/rng.h"

static uint64_t splitmix64(uint64_t *x)
{
    uint64_t z = (*x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

void ath_rng_init(ath_rng *r, uint64_t seed)
{
    for (int i = 0; i < 4; i++)
        r->s[i] = splitmix64(&seed);
}

static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

uint64_t ath_rng_u64(ath_rng *r)
{
    uint64_t *s = r->s;
    const uint64_t out = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return out;
}

double ath_rng_uniform(ath_rng *r) { return (double)(ath_rng_u64(r) >> 11) * 0x1.0p-53; }
