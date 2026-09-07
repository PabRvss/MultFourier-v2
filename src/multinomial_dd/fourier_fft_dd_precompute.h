#pragma once

#include "dd_math.h"

typedef struct {
    dd_t fa;
    dd_t fb;
} merge_dd_t;

typedef struct {
    real_t sum_log;
    real_t dif_log;
    merge_dd_t m;
} log_sum_diff_dd_t;

static const dd_t DD_ONE = {1.0, 0.0};
static const dd_t DD_ZERO = {0.0, 0.0};

static inline merge_dd_t log_add_inplace_dd(real_t* restrict a_log, const real_t b_log) {
    merge_dd_t m;
    if (*a_log == -INFINITY && b_log == -INFINITY) { m.fa = DD_ONE; m.fb = DD_ZERO; return m; }

    if (*a_log > b_log) {
        m.fa = DD_ONE;
        m.fb = dd_exp_libm(b_log - *a_log);
    } else {
        m.fa = dd_exp_libm(*a_log - b_log);
        m.fb = DD_ONE;
        *a_log = b_log;
    }
    return m;
}

static inline log_sum_diff_dd_t log_sum_diff_dd(const real_t a_log, const real_t b_log) {
    log_sum_diff_dd_t res;
    if (a_log == -INFINITY && b_log == -INFINITY) {
        res.sum_log = -INFINITY;
        res.dif_log = -INFINITY;
        res.m.fa = DD_ZERO;
        res.m.fb = DD_ZERO;
        return res;
    }

    if (a_log > b_log) {
        res.m.fa = DD_ONE;
        res.m.fb = dd_exp_libm(b_log - a_log);
        res.sum_log = a_log;
        res.dif_log = a_log;
    } else {
        res.m.fa = dd_exp_libm(a_log - b_log);
        res.m.fb = DD_ONE;
        res.sum_log = b_log;
        res.dif_log = b_log;
    }
    return res;
}

typedef struct {
    vddc_t unit;
} uresult_dd_t;

typedef struct {
    uresult_dd_t sum;
    uresult_dd_t dif;
} usum_diff_dd_t;

static inline __attribute__((always_inline)) void unit_add_inplace_dd(uresult_dd_t* restrict a, const uresult_dd_t b, const merge_dd_t m) {
    a->unit = vddc_add(vddc_scale_dd(m.fa, a->unit), vddc_scale_dd(m.fb, b.unit));
}

static inline __attribute__((always_inline)) usum_diff_dd_t unit_sum_diff_dd(const uresult_dd_t* restrict a, const uresult_dd_t* restrict b, const merge_dd_t m) {
    usum_diff_dd_t res;
    const vddc_t fa = vddc_scale_dd(m.fa, a->unit);
    const vddc_t fb = vddc_scale_dd(m.fb, b->unit);
    res.sum.unit = vddc_add(fa, fb);
    res.dif.unit = vddc_sub(fa, fb);
    return res;
}

static inline __attribute__((always_inline)) uresult_dd_t unit_product_dd(const uresult_dd_t* restrict a, const uresult_dd_t* restrict b) {
    const uresult_dd_t res = { .unit = vddc_mul(a->unit, b->unit) };
    return res;
}

static inline __attribute__((always_inline)) uresult_dd_t unit_cproduct_dd(const ddc_t a, const uresult_dd_t b) {
    const uresult_dd_t res = { .unit = vddc_mul(vddc_from_ddc(a), b.unit) };
    return res;
}

typedef struct {
    int_t M;
    int_t K;
    int_t N;
    int_t calls_per_round;
    merge_dd_t* restrict factors;
    real_t resN_exp;
} fourier_fft_dd_plan_t;

static int_t fourier_fft_dd_count_calls(const int_t M) {
    int_t count = 0;
    int_t N_curr = M, blocks = 1;
    for (; N_curr>THRESHOLD_PROD; N_curr>>=1) {
        const int_t width = N_curr/2;
        count += 2*blocks*width;
        blocks <<= 1;
    }
    count += blocks*N_curr*N_curr;
    blocks >>= 1;
    for (; N_curr<M; N_curr<<=1) {
        count += blocks*N_curr;
        blocks >>= 1;
    }
    return count;
}

static fourier_fft_dd_plan_t* build_fourier_fft_dd_plan(const multinomial* restrict mult) {
    const cell_t* restrict global = mult->matrix;
    const int_t N = mult->N;
    const int_t K = mult->K;

    int_t M = 1;
    while (M<N+1) M<<=1;
    M<<=1;

    const int_t calls_per_round = fourier_fft_dd_count_calls(M);

    fourier_fft_dd_plan_t* plan = malloc(sizeof(fourier_fft_dd_plan_t));
    plan->M = M;
    plan->K = K;
    plan->N = N;
    plan->calls_per_round = calls_per_round;
    plan->factors = malloc((size_t)K*calls_per_round*sizeof(merge_dd_t));

    const real_t nu_sp = mult->l_rate;
    real_t* restrict l_rate = malloc(K*sizeof(real_t));
    for (int_t k=0; k<K; k++) l_rate[k] = nu_sp;

    real_t* restrict A = malloc(M*sizeof(real_t));
    real_t* restrict B = malloc(M*sizeof(real_t));
    real_t* restrict C = malloc(M*sizeof(real_t));

    for (int_t i=0; i<M; i++) A[i] = -INFINITY;
    A[0] = 0;

    real_t l_prev = 0;
    for (int_t k=K-1; k>=0; k--) {  // reversed
        const int_t index_k = k*(N+1);
        const real_t l_k = l_rate[k];
        int_t idx = (int_t)k*calls_per_round;

        const real_t d_k = l_k - l_prev;
        for (int_t j=0; j<M; j++) A[j] += d_k * (real_t)j;
        l_prev = l_k;

        for (int_t i=0; i<=N; i++) {
            const int_t index = index_k + i;
            B[i] = global[index].log_prob + l_k*(real_t)i;
        }
        for (int_t i=N+1; i<M; i++) B[i] = -INFINITY;

        int_t N_curr = M;
        int_t blocks = 1;

        for (; N_curr>THRESHOLD_PROD; N_curr>>=1) {
            const int_t width = N_curr/2;
            for (int_t i=0; i<blocks; i++) {
                const int_t base = i*N_curr;

                for (int_t j = 0; j<width; j++) {
                    const log_sum_diff_dd_t sd1 = log_sum_diff_dd(A[base + j], A[base + width + j]);
                    plan->factors[idx++] = sd1.m;
                    A[base + j] = sd1.sum_log;
                    A[base + width + j] = sd1.dif_log;

                    const log_sum_diff_dd_t sd2 = log_sum_diff_dd(B[base + j], B[base + width + j]);
                    plan->factors[idx++] = sd2.m;
                    B[base + j] = sd2.sum_log;
                    B[base + width + j] = sd2.dif_log;
                }
            }
            blocks <<= 1;
        }

        for (int_t b=0; b<blocks; b++) {
            const int_t width = N_curr;
            const int_t base = b*width;

            for (int_t i=0; i<width; i++) {
                real_t acc = -INFINITY;
                for (int_t l=i+1; l<width; l++) {
                    const real_t p_log = A[base + width + i - l] + B[base + l];
                    plan->factors[idx++] = log_add_inplace_dd(&acc, p_log);
                }
                for (int_t l=0; l<=i; l++) {
                    const real_t p_log = A[base + i - l] + B[base + l];
                    plan->factors[idx++] = log_add_inplace_dd(&acc, p_log);
                }
                C[base + i] = acc;
            }
        }

        blocks >>= 1;
        for (; N_curr<M; N_curr<<=1) {
            const int_t width2 = N_curr;
            for (int_t i=0; i<blocks; i++) {
                const int_t base = i*width2*2;

                for (int_t j = 0; j<width2; j++) {
                    const log_sum_diff_dd_t sd = log_sum_diff_dd(C[base + j], C[base + width2 + j]);
                    plan->factors[idx++] = sd.m;
                    C[base + j] = sd.sum_log;
                    C[base + width2 + j] = sd.dif_log;
                }
            }
            blocks >>= 1;
        }

        for (int_t i=N+1; i<M; i++) C[i] = -INFINITY;
        real_t* restrict temp = A;
        A = C;
        C = temp;
    }

    plan->resN_exp = exp((double)A[N] - (double)(l_prev*(real_t)N) + lgac[N]);

    free(l_rate);
    free(A);
    free(B);
    free(C);

    return plan;
}

static void free_fourier_fft_dd_plan(fourier_fft_dd_plan_t* plan) {
    free(plan->factors);
    free(plan);
}

static void* eval_fourier_fft_dd_precompute_thread(void* args_void) {
    arguments_t* args = args_void;

    const multinomial* restrict mult = args->mult;
    const cell_t* restrict global = mult->matrix;
    const fourier_fft_dd_plan_t* restrict plan = args->plan;
    complex_t* restrict res_arr = args->res_arr;

    const int_t N = mult->N;
    const int_t K = mult->K;
    const int_t M = plan->M;
    const int_t calls_per_round = plan->calls_per_round;

    const uresult_dd_t U_ZERO = {.unit = vddc_from_ddc((ddc_t){dd_from_double(1), dd_from_double(0)})};

    uresult_dd_t* restrict A = malloc(M*sizeof(uresult_dd_t));
    uresult_dd_t* restrict B = malloc(M*sizeof(uresult_dd_t));
    uresult_dd_t* restrict C = malloc(M*sizeof(uresult_dd_t));
    ddc_t* restrict F = malloc(M*sizeof(ddc_t));

    const real_t s2_re = M_PI*mult->gamma/mult->T;
    const ddc_t DDC_ONE = {dd_from_double(1), dd_from_double(0)};
    const ddc_t DDC_HALF = {dd_from_double(0.5), dd_from_double(0)};

    for (int_t n_outer=args->start; n_outer<args->end; n_outer++) {
        for (int_t i=0; i<M; i++) A[i] = U_ZERO;

        for (int_t k=K-1; k>=0; k--) {  // reversed
            const int_t index_k = k*(N+1);
            const merge_dd_t* restrict mf = &plan->factors[(int_t)k*calls_per_round];
            int_t idx = 0;

            for (int_t i=0; i<=N; i++) {
                const int_t index = index_k + i;
                const real_t rewardT = global[index].reward;
#ifdef USE_SIMD
                vec_t n_vec;
                for (int s=0; s<STRIDE; s++) n_vec[s] = (real_t)(n_outer*STRIDE + s);
                const vdd_t arg_dd = vec_two_prod(n_vec, VEC_SET1(rewardT));
                const vddc_t cs = vdd_cis(arg_dd);
                B[i].unit = (vddc_t){cs.re, vdd_neg(cs.im)};  // conjugated
#else
                const dd_t arg_dd = two_prod((double)n_outer, rewardT);
                const ddc_t cs = dd_cis(arg_dd);
                B[i].unit = vddc_from_ddc((ddc_t){cs.re, dd_neg(cs.im)});  // conjugated
#endif
            }
            for (int_t i=N+1; i<M; i++) B[i] = U_ZERO;

            F[0] = DDC_ONE;

            int_t N_curr = M;
            int_t blocks = 1;

            for (; N_curr>THRESHOLD_PROD; N_curr>>=1) {
                const int_t width = N_curr/2;
                for (int_t i=0; i<blocks; i++) {
                    const int_t base = i*N_curr;
                    const ddc_t f_sqrt = ddc_sqrt(F[base]);

                    for (int_t j = 0; j<width; j++) {
                        const uresult_dd_t a1 = A[base + j];
                        const uresult_dd_t a2 = unit_cproduct_dd(f_sqrt, A[base + width + j]);
                        const usum_diff_dd_t sd1 = unit_sum_diff_dd(&a1, &a2, mf[idx++]);
                        A[base + j] = sd1.sum;
                        A[base + width + j] = sd1.dif;

                        const uresult_dd_t b1 = B[base + j];
                        const uresult_dd_t b2 = unit_cproduct_dd(f_sqrt, B[base + width + j]);
                        const usum_diff_dd_t sd2 = unit_sum_diff_dd(&b1, &b2, mf[idx++]);
                        B[base + j] = sd2.sum;
                        B[base + width + j] = sd2.dif;
                    }

                    F[base] = f_sqrt;
                    F[base + width] = ddc_neg(f_sqrt);
                }
                blocks <<= 1;
            }

            for (int_t b=0; b<blocks; b++) {
                const int_t width = N_curr;
                const int_t base = b*width;

                for (int_t i=0; i<width; i++) {
                    uresult_dd_t acc = U_ZERO;
                    for (int_t l=i+1; l<width; l++) {
                        const uresult_dd_t p = unit_product_dd(&A[base + width + i - l], &B[base + l]);
                        unit_add_inplace_dd(&acc, p, mf[idx++]);
                    }
                    acc = unit_cproduct_dd(F[base], acc);
                    for (int_t l=0; l<=i; l++) {
                        const uresult_dd_t p = unit_product_dd(&A[base + i - l], &B[base + l]);
                        unit_add_inplace_dd(&acc, p, mf[idx++]);
                    }
                    C[base + i] = acc;
                }
            }

            blocks >>= 1;
            for (; N_curr<M; N_curr<<=1) {
                const int_t width2 = N_curr;
                for (int_t i=0; i<blocks; i++) {
                    const int_t base = i*width2*2;
                    const ddc_t f_sqrt = F[base];
                    const ddc_t half_inv_f = ddc_scale(dd_from_double(0.5), ddc_reciprocal(f_sqrt));

                    for (int_t j = 0; j<width2; j++) {
                        const uresult_dd_t m1 = C[base + j];
                        const uresult_dd_t m2 = C[base + width2 + j];
                        const usum_diff_dd_t sd = unit_sum_diff_dd(&m1, &m2, mf[idx++]);
                        C[base + j] = unit_cproduct_dd(DDC_HALF, sd.sum);
                        C[base + width2 + j] = unit_cproduct_dd(half_inv_f, sd.dif);
                    }

                    F[base] = ddc_mul(f_sqrt, f_sqrt);
                }
                blocks >>= 1;
            }

            for (int_t i=N+1; i<M; i++) C[i] = U_ZERO;
            uresult_dd_t* restrict temp = A;
            A = C;
            C = temp;
        }

        const vec_t unit_re_d = A[N].unit.re.hi + A[N].unit.re.lo;
        const vec_t unit_im_d = A[N].unit.im.hi + A[N].unit.im.lo;

#ifdef USE_SIMD
        for (int s=0; s<STRIDE; s++) {
            const int_t n = n_outer*STRIDE + s;
            const complex_t unit = unit_re_d[s] + I*unit_im_d[s];
            const complex_t s2 = s2_re + I*(n*M_PI);
            complex_t sinc = (1-EXPC(-s2))/s2;
            if ((n == 0) && (ABS(s2_re)<=EPS)) sinc = 1 - s2*(0.5 - s2/6);
            res_arr[n] = plan->resN_exp*(unit*sinc);
        }
#else
        const complex_t unit = unit_re_d + I*unit_im_d;
        const complex_t s2 = s2_re + I*(n_outer*M_PI);
        complex_t sinc = (1-EXPC(-s2))/s2;
        if ((n_outer == 0) && (ABS(s2_re)<=EPS)) sinc = 1 - s2*(0.5 - s2/6);
        res_arr[n_outer] = plan->resN_exp*(unit*sinc);
#endif
    }

    free(A);
    free(B);
    free(C);
    free(F);

    return 0;
}
