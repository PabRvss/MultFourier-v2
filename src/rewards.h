#include "base.h"
#pragma once

// Reward signature: log-prob log_pk for category k, observed xk

static inline real_t reward_logp(int_t N, real_t log_pk, int_t xk) {  // standard p-value
    return xk*log_pk - lgac[xk];
}

static inline real_t reward_llr(int_t N, real_t log_pk, int_t xk) {  // log-likelihood ratio
    if (xk == 0) return 0;
    return -xk*(LOG((real_t)xk/N) - log_pk);
}

static inline real_t reward_pearson(int_t N, real_t log_pk, int_t xk) {  // Pearson statistic (Chi-squared)
    return -xk*xk/(N*EXP(log_pk))/10.0;
//    real_t obs = -2521.68282184;
//    return -MIN(xk*xk/(N*EXP(log_pk)), -obs);
}
