#include "base.h"
#pragma once

static void recursive(
        const int_t N,
        const int_t K,
        const real_t* restrict log_probs,
        real_t* restrict p_acc,
        const real_t r0, const real_t r_acc,
        const real_t logp_acc,
        const int_t K_remain, const int_t N_remain
) {
    if (K_remain == 1) {
        const real_t log_p = log_probs[K-1];
        const real_t r_final = r_acc + REWARD(N, log_p, N_remain);
        const real_t logp_final = logp_acc + reward_logp(N, log_p, N_remain);
        if (r_final <= r0) {
            const real_t p = EXP(MAX_EXP + logp_final);
            *p_acc = *p_acc + p;
        }
        return;
    }
    const int_t k = K-K_remain;
    for (int_t n=0; n<=N_remain; n++) {
        const real_t log_p = log_probs[k];
        const real_t dr = REWARD(N, log_p, n);
        const real_t dp = reward_logp(N, log_p, n);
        recursive(N, K, log_probs, p_acc, r0, r_acc + dr, logp_acc + dp, K_remain-1, N_remain-n);
    }
}

static void exhaustive(multinomial* mult, multinomial_result* mult_res, options_t* options) {
    const int_t N = mult->N;
    const int_t K = mult->K;
    const real_t r0 = mult->r0;
    real_t p_acc = 0;
    const real_t logp_acc = lgac[N];
    recursive(N, K, mult->log_probs, &p_acc, r0, 0, logp_acc, K, N);
    p_acc = p_acc*EXP(-MAX_EXP);
    if (options->verbose) printf("exhaustive = %.8e\n", p_acc);
    mult_res->pval = p_acc;
    mult_res->converged = 1;
    mult_res->terms = -1;
}
