#pragma once

#include "dd_math.h"

#define NEG_HUGE_GAMMA (-1e30)

typedef struct {
    vec_t log_mod;
    vddc_t unit;
} vgdresult_t;

typedef struct {
    vgdresult_t sum;
    vgdresult_t dif;
} vgdsum_diff_t;

static inline vec_t vec_exp_libm(vec_t a) {
#ifdef USE_SIMD
    vec_t r;
    for (int s=0; s<STRIDE; s++) r[s] = exp(a[s]);
    return r;
#else
    return exp(a);
#endif
}

static inline __attribute__((always_inline)) void vgd_exp_pair(vec_t log_a, vec_t log_b, vdd_t* restrict f_a, vdd_t* restrict f_b) {
    const vec_t d = log_a - log_b;
    const vdd_t f_small = {vec_exp_libm(-VEC_MAX(d, -d)), VEC_ZERO};
    const vec_mask_t a_is_max = VEC_CMP_GE(d, VEC_ZERO);
    f_a->hi = VEC_SELECT(a_is_max, VEC_ONE, f_small.hi);
    f_a->lo = VEC_SELECT(a_is_max, VEC_ZERO, f_small.lo);
    f_b->hi = VEC_SELECT(a_is_max, f_small.hi, VEC_ONE);
    f_b->lo = VEC_SELECT(a_is_max, f_small.lo, VEC_ZERO);
}

static inline __attribute__((always_inline)) void vgd_add_inplace(vgdresult_t* restrict a, const vgdresult_t b) {
    const vec_t log_mod = VEC_MAX(a->log_mod, b.log_mod);
    vdd_t f1, f2;
    vgd_exp_pair(a->log_mod, b.log_mod, &f1, &f2);
    a->log_mod = log_mod;
    a->unit = vddc_add(vddc_scale(f1, a->unit), vddc_scale(f2, b.unit));
}

static inline __attribute__((always_inline)) vgdsum_diff_t vgd_sum_diff(const vgdresult_t* restrict a, const vgdresult_t* restrict b) {
    const vec_t log_mod = VEC_MAX(a->log_mod, b->log_mod);
    vdd_t f1, f2;
    vgd_exp_pair(a->log_mod, b->log_mod, &f1, &f2);
    const vddc_t sa = vddc_scale(f1, a->unit);
    const vddc_t sb = vddc_scale(f2, b->unit);

    vgdsum_diff_t res;
    res.sum.log_mod = log_mod;
    res.sum.unit = vddc_add(sa, sb);
    res.dif.log_mod = log_mod;
    res.dif.unit = vddc_sub(sa, sb);
    return res;
}

static inline __attribute__((always_inline)) vgdresult_t vgd_product(const vgdresult_t* restrict a, const vgdresult_t* restrict b) {
    const vgdresult_t res = {
        .log_mod = a->log_mod + b->log_mod,
        .unit = vddc_mul(a->unit, b->unit),
    };
    return res;
}

static inline __attribute__((always_inline)) vgdresult_t vgd_cproduct(const ddc_t a, const vgdresult_t b) {
    const vgdresult_t res = {
        .log_mod = b.log_mod,
        .unit = vddc_mul(vddc_from_ddc(a), b.unit),
    };
    return res;
}

static void* eval_gammas_fft_dd_thread(void* args_void) {
    args_gamma_t* args = args_void;

    const multinomial* restrict mult = args->mult;
    const cell_t* restrict global = mult->matrix;
    const int_t thread = args->thread;
    const int_t vecs_per_thread = args->vecs_per_thread;
    real_t* restrict res_arr = args->res_arr;
    const vec_t* restrict gammas_vec = args->gammas_vec;

    const int_t N = mult->N;
    const int_t K = mult->K;

    const vgdresult_t ZERO_G = {
        .log_mod = VEC_SET1(NEG_HUGE_GAMMA),
        .unit = {vdd_from_vec(VEC_ONE), vdd_from_vec(VEC_ZERO)},
    };
    const ddc_t DDC_ONE = {dd_from_double(1.0), dd_from_double(0.0)};

    int_t M = 1;
    while (M<N+1) M<<=1;
    M<<=1;

    vgdresult_t* restrict A = malloc(M*sizeof(vgdresult_t));
    vgdresult_t* restrict B = malloc(M*sizeof(vgdresult_t));
    vgdresult_t* restrict C = malloc(M*sizeof(vgdresult_t));
    ddc_t* restrict F = malloc(M*sizeof(ddc_t));

    for (int_t i_vec=0; i_vec<vecs_per_thread; i_vec++) {
        const int_t idx_vec = vecs_per_thread*thread + i_vec;
        const vec_t gamma = gammas_vec[idx_vec];

        const vec_t nu = find_nu_gamma_vec(global, N, K, gamma);
        args->l_rate_arr[idx_vec] = nu;

        for (int_t i=0; i<M; i++) A[i] = ZERO_G;
        A[0].log_mod = VEC_ZERO;

        vec_t l_prev = VEC_ZERO;
        for (int_t k=K-1; k>=0; k--) {  // reversed
            const int_t index_k = k*(N+1);
            const vec_t l_k = nu;

            const vec_t d_k = l_k - l_prev;
            for (int_t j=0; j<M; j++) A[j].log_mod = A[j].log_mod + d_k*VEC_SET1((real_t)j);
            l_prev = l_k;

            for (int_t i=0; i<=N; i++) {
                const int_t index = index_k + i;
                const vec_t raw_i = -global[index].reward*gamma + VEC_SET1(global[index].log_prob);
                B[i].log_mod = raw_i + l_k*VEC_SET1((real_t)i);
                B[i].unit = (vddc_t){vdd_from_vec(VEC_ONE), vdd_from_vec(VEC_ZERO)};
            }
            for (int_t i=N+1; i<M; i++) {
                B[i] = ZERO_G;
            }

            F[0] = DDC_ONE;

            int_t N_curr = M;
            int_t blocks = 1;

            for (; N_curr>THRESHOLD_PROD; N_curr>>=1) {
                const int_t width = N_curr/2;
                for (int_t i=0; i<blocks; i++) {
                    const int_t base = i*N_curr;
                    const ddc_t f_sqrt = ddc_sqrt(F[base]);

                    for (int_t j = 0; j<width; j++) {
                        const vgdresult_t a1 = A[base + j];
                        const vgdresult_t a2 = vgd_cproduct(f_sqrt, A[base + width + j]);
                        const vgdsum_diff_t sd1 = vgd_sum_diff(&a1, &a2);
                        A[base + j] = sd1.sum;
                        A[base + width + j] = sd1.dif;

                        const vgdresult_t b1 = B[base + j];
                        const vgdresult_t b2 = vgd_cproduct(f_sqrt, B[base + width + j]);
                        const vgdsum_diff_t sd2 = vgd_sum_diff(&b1, &b2);
                        B[base + j] = sd2.sum;
                        B[base + width + j] = sd2.dif;
                    }

                    F[base] = f_sqrt;
                    F[base + width] = (ddc_t){dd_neg(f_sqrt.re), dd_neg(f_sqrt.im)};
                }
                blocks <<= 1;
            }

            for (int_t b=0; b<blocks; b++) {
                const int_t width = N_curr;
                const int_t base = b*width;

                for (int_t i=0; i<width; i++) {
                    vgdresult_t acc = ZERO_G;
                    for (int_t l=i+1; l<width; l++) vgd_add_inplace(&acc, vgd_product(&A[base + width + i - l], &B[base + l]));
                    acc = vgd_cproduct(F[base], acc);
                    for (int_t l=0; l<=i; l++) vgd_add_inplace(&acc, vgd_product(&A[base + i - l], &B[base + l]));
                    C[base + i] = acc;
                }
            }

            blocks >>= 1;
            for (; N_curr<M; N_curr<<=1) {
                const int_t width2 = N_curr;
                for (int_t i=0; i<blocks; i++) {
                    const int_t base = i*width2*2;
                    const ddc_t f_sqrt = F[base];
                    const ddc_t recip = ddc_reciprocal(f_sqrt);
                    const ddc_t half_inv_f = {dd_mul_d(recip.re, 0.5), dd_mul_d(recip.im, 0.5)};
                    const ddc_t half = {dd_from_double(0.5), dd_from_double(0.0)};

                    for (int_t j = 0; j<width2; j++) {
                        const vgdresult_t m1 = C[base + j];
                        const vgdresult_t m2 = C[base + width2 + j];
                        const vgdsum_diff_t sd = vgd_sum_diff(&m1, &m2);
                        C[base + j] = vgd_cproduct(half, sd.sum);
                        C[base + width2 + j] = vgd_cproduct(half_inv_f, sd.dif);
                    }

                    F[base] = ddc_mul(f_sqrt, f_sqrt);
                }
                blocks >>= 1;
            }

            for (int_t i=N+1; i<M; i++) C[i] = ZERO_G;
            vgdresult_t* restrict temp = A;
            A = C;
            C = temp;
        }

        const vdd_t mag = vddc_abs(A[N].unit);
#ifdef USE_SIMD
        for (int s=0; s<STRIDE; s++) {
            const real_t resN = A[N].log_mod[s] - l_prev[s]*(real_t)N + log(mag.hi[s] + mag.lo[s]) + lgac[N];
            const real_t s2 = M_PI*gamma[s]/mult->T;
            real_t sinc = LOG((1-EXP(-s2))/s2);
            if (s2 < -MAX_EXP) sinc = -s2 - LOG(-s2);
            if (ABS(s2)<=EPS) sinc = -s2*(0.5 - s2/24);
            res_arr[idx_vec*STRIDE + s] = resN + sinc;
        }
#else
        const real_t resN = A[N].log_mod - l_prev*(real_t)N + log(mag.hi + mag.lo) + lgac[N];
        const real_t s2 = M_PI*gamma/mult->T;
        real_t sinc = LOG((1-EXP(-s2))/s2);
        if (s2 < -MAX_EXP) sinc = -s2 - LOG(-s2);
        if (ABS(s2)<=EPS) sinc = -s2*(0.5 - s2/24);
        res_arr[idx_vec] = resN + sinc;
#endif
    }

    free(A);
    free(B);
    free(C);
    free(F);

    return 0;
}
