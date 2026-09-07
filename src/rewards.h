#pragma once

static inline real_t reward_logp(const real_t log_pk, const int_t xk) {  // probability mass statistic
    return xk*log_pk - lgac[xk];
}

static inline real_t reward_llr(const int_t N, const real_t log_pk, const int_t xk, const real_t lambda) {  // log-likelihood ratio
    if (xk == 0) return 0;
    const real_t log_base = LOG((real_t)xk/N) - log_pk;
//    return -xk*log_base*(1 + lambda*log_base/2);
    return -xk*log_base;
}

static real_t* restrict log_primes = NULL;
static inline real_t reward_primes(const int_t k, const int_t xk) {  // probability mass statistic
    return xk*log_primes[k];
}

__attribute__((optimize("no-fast-math"), noinline))
static real_t reward_power(const int_t N, const int_t k, const real_t log_pk, const int_t xk, const real_t lambda) {  // Power divergence statistic (general lambda)
//    return reward_primes(k, xk);

    // Special cases:
    if (real_isnan(lambda)) return reward_logp(log_pk, xk);  // undefined lambda, use probability mass statistic
    if (ABS(lambda) <= EPS) return reward_llr(N, log_pk, xk, lambda);  // log-likelihood ratio is the limit lambda -> 0

    if (xk == 0) return 0;
    const real_t base = xk/(N*EXP(log_pk));
    return -1/(lambda*(lambda+1))*xk*(POW(base, lambda) - 1);
}

void fill_primes(int_t k_max) {
    k_max = k_max < 6 ? 6 : k_max;
    int_t n = (int_t)(k_max*(LOG(k_max) + LOG(LOG(k_max))));

    if (log_primes != NULL) free(log_primes);
    log_primes = malloc(k_max*sizeof(real_t));

    int8_t* prime = malloc((n+1) * sizeof(int8_t));
    for (int i=0; i<=n; i++) prime[i] = 1;

    prime[0] = prime[1] = 0;
    for (int p = 2; p*p <= n; p++) {
        if (prime[p]) {
            for (int i = p*p; i <= n; i += p) prime[i] = 0;
        }
    }

    int_t k=0;
    for (int_t p = 2; p <= n; p++) {
        if (k>=k_max) break;
        if (prime[p]) log_primes[k++] = LOG(p);
//        if (prime[p]) log_primes[k++] = sqrt((real_t)p);
    }

    free(prime);
}
