#pragma once

#include "dd_math.h"

static inline dd_t dd_exp_libm(double a) { return (dd_t){exp(a), 0.0}; }

typedef struct {
    real_t log_mod;
    vddc_t unit;
} vddresult_t;

typedef struct {
    vddresult_t sum;
    vddresult_t dif;
} vddsum_diff_t;

static inline __attribute__((always_inline)) void vdd_add_inplace(vddresult_t* restrict a, const vddresult_t b) {
    if (a->log_mod == -INFINITY && b.log_mod == -INFINITY) return;

    if (a->log_mod > b.log_mod) {
        const dd_t f = dd_exp_libm(b.log_mod - a->log_mod);
        a->unit = vddc_add(a->unit, vddc_scale_dd(f, b.unit));
    } else {
        const dd_t f = dd_exp_libm(a->log_mod - b.log_mod);
        a->log_mod = b.log_mod;
        a->unit = vddc_add(vddc_scale_dd(f, a->unit), b.unit);
    }
}

static inline __attribute__((always_inline)) vddsum_diff_t vdd_sum_diff(const vddresult_t* restrict a, const vddresult_t* restrict b) {
    vddsum_diff_t res;
    if (a->log_mod == -INFINITY && b->log_mod == -INFINITY) {
        const vddc_t zero = vddc_from_ddc((ddc_t){dd_from_double(0), dd_from_double(0)});
        res.sum.log_mod = -INFINITY;
        res.sum.unit = zero;
        res.dif.log_mod = -INFINITY;
        res.dif.unit = zero;
        return res;
    }

    if (a->log_mod > b->log_mod) {
        const dd_t f = dd_exp_libm(b->log_mod - a->log_mod);
        const vddc_t fb = vddc_scale_dd(f, b->unit);
        res.sum.log_mod = a->log_mod;
        res.sum.unit = vddc_add(a->unit, fb);
        res.dif.log_mod = a->log_mod;
        res.dif.unit = vddc_sub(a->unit, fb);
    } else {
        const dd_t f = dd_exp_libm(a->log_mod - b->log_mod);
        const vddc_t fa = vddc_scale_dd(f, a->unit);
        res.sum.log_mod = b->log_mod;
        res.sum.unit = vddc_add(fa, b->unit);
        res.dif.log_mod = b->log_mod;
        res.dif.unit = vddc_sub(fa, b->unit);
    }
    return res;
}

static inline __attribute__((always_inline)) vddresult_t vdd_product(const vddresult_t* restrict a, const vddresult_t* restrict b) {
    const vddresult_t res = {
        .log_mod = a->log_mod + b->log_mod,
        .unit = vddc_mul(a->unit, b->unit),
    };
    return res;
}

static inline __attribute__((always_inline)) vddresult_t vdd_cproduct(const ddc_t a, const vddresult_t b) {
    const vddresult_t res = {
        .log_mod = b.log_mod,
        .unit = vddc_mul(vddc_from_ddc(a), b.unit),
    };
    return res;
}

static void* eval_fourier_fft_dd_thread(void* args_void) {
    arguments_t* args = args_void;

    const multinomial* restrict mult = args->mult;
    const cell_t* restrict global = mult->matrix;
    const vddresult_t ZERO = {.log_mod = -INFINITY, .unit = vddc_from_ddc((ddc_t){dd_from_double(1), dd_from_double(0)})};
    const ddc_t DDC_ONE = {dd_from_double(1), dd_from_double(0)};
    const ddc_t DDC_HALF = {dd_from_double(0.5), dd_from_double(0.0)};

    const int_t N = mult->N;
    const int_t K = mult->K;

    int_t M = 1;
    while (M<N+1) M<<=1;
    M<<=1;

    const real_t nu_sp = mult->l_rate;
    real_t* restrict l_rate = malloc(K*sizeof(real_t));
    for (int_t k=0; k<K; k++) l_rate[k] = nu_sp;

    vddresult_t* restrict A = malloc(M*sizeof(vddresult_t));
    vddresult_t* restrict B = malloc(M*sizeof(vddresult_t));
    vddresult_t* restrict C = malloc(M*sizeof(vddresult_t));
    ddc_t* restrict F = malloc(M*sizeof(ddc_t));

    complex_t* restrict res_arr = args->res_arr;
    const real_t s2_re = M_PI*mult->gamma/mult->T;

    for (int_t n_outer=args->start; n_outer<args->end; n_outer++) {
        for (int_t i=0; i<M; i++) A[i] = ZERO;
        A[0].log_mod = 0;

        real_t l_prev = 0;
        for (int_t k=K-1; k>=0; k--) {  // reversed
            const int_t index_k = k*(N+1);
            const real_t l_k = l_rate[k];

            const real_t d_k = l_k - l_prev;
            for (int_t j=0; j<M; j++) A[j].log_mod += d_k * (real_t)j;
            l_prev = l_k;

            for (int_t i=0; i<=N; i++) {
                const int_t index = index_k + i;
                const real_t rewardT = global[index].reward;
                B[i].log_mod = global[index].log_prob + l_k*(real_t)i;
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
            for (int_t i=N+1; i<M; i++) {
                B[i] = ZERO;
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
                        const vddresult_t a1 = A[base + j];
                        const vddresult_t a2 = vdd_cproduct(f_sqrt, A[base + width + j]);
                        const vddsum_diff_t sd1 = vdd_sum_diff(&a1, &a2);
                        A[base + j] = sd1.sum;
                        A[base + width + j] = sd1.dif;

                        const vddresult_t b1 = B[base + j];
                        const vddresult_t b2 = vdd_cproduct(f_sqrt, B[base + width + j]);
                        const vddsum_diff_t sd2 = vdd_sum_diff(&b1, &b2);
                        B[base + j] = sd2.sum;
                        B[base + width + j] = sd2.dif;
                    }

                    F[base] = f_sqrt;
                    F[base + width] = (ddc_t){dd_neg(f_sqrt.re), dd_neg(f_sqrt.im)};
                }
                blocks <<= 1;
            }

            // direct convolution at the leaf blocks
            for (int_t b=0; b<blocks; b++) {
                const int_t width = N_curr;
                const int_t base = b*width;

                for (int_t i=0; i<width; i++) {
                    vddresult_t acc = ZERO;
                    for (int_t l=i+1; l<width; l++) vdd_add_inplace(&acc, vdd_product(&A[base + width + i - l], &B[base + l]));
                    acc = vdd_cproduct(F[base], acc);
                    for (int_t l=0; l<=i; l++) vdd_add_inplace(&acc, vdd_product(&A[base + i - l], &B[base + l]));
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

                    for (int_t j = 0; j<width2; j++) {
                        const vddresult_t m1 = C[base + j];
                        const vddresult_t m2 = C[base + width2 + j];
                        const vddsum_diff_t sd = vdd_sum_diff(&m1, &m2);
                        C[base + j] = vdd_cproduct(DDC_HALF, sd.sum);
                        C[base + width2 + j] = vdd_cproduct(half_inv_f, sd.dif);
                    }

                    F[base] = ddc_mul(f_sqrt, f_sqrt);
                }
                blocks >>= 1;
            }

            for (int_t i=N+1; i<M; i++) C[i] = ZERO;
            vddresult_t* restrict temp = A;
            A = C;
            C = temp;
        }

        const double resN_exp = exp((double)A[N].log_mod - (double)(l_prev*(real_t)N) + lgac[N]);
        const vec_t unit_re_d = A[N].unit.re.hi + A[N].unit.re.lo;
        const vec_t unit_im_d = A[N].unit.im.hi + A[N].unit.im.lo;

#ifdef USE_SIMD
        for (int s=0; s<STRIDE; s++) {
            const int_t n = n_outer*STRIDE + s;
            const complex_t unit = unit_re_d[s] + I*unit_im_d[s];
            const complex_t s2 = s2_re + I*(n*M_PI);
            complex_t sinc = (1-EXPC(-s2))/s2;
            if ((n == 0) && (ABS(s2_re)<=EPS)) sinc = 1 - s2*(0.5 - s2/6);
            res_arr[n] = resN_exp*(unit*sinc);
        }
#else
        const complex_t unit = unit_re_d + I*unit_im_d;
        const complex_t s2 = s2_re + I*(n_outer*M_PI);
        complex_t sinc = (1-EXPC(-s2))/s2;
        if ((n_outer == 0) && (ABS(s2_re)<=EPS)) sinc = 1 - s2*(0.5 - s2/6);
        res_arr[n_outer] = resN_exp*(unit*sinc);
#endif
    }

    free(A);
    free(B);
    free(C);
    free(F);
    free(l_rate);

    return 0;
}
