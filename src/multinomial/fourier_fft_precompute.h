#pragma once

typedef struct {
    real_t fa;
    real_t fb;
} merge_t;

typedef struct {
    real_t sum_log;
    real_t dif_log;
    merge_t m;
} log_sum_diff_t;

static inline merge_t log_add_inplace(real_t* restrict a_log, const real_t b_log) {
    merge_t m;
    if (*a_log == -INFINITY && b_log == -INFINITY) { m.fa = 1; m.fb = 0; return m; }

    if (*a_log > b_log) {
        m.fa = 1;
        m.fb = EXP(b_log - *a_log);
    } else {
        m.fa = EXP(*a_log - b_log);
        m.fb = 1;
        *a_log = b_log;
    }
    return m;
}

static inline log_sum_diff_t log_sum_diff(const real_t a_log, const real_t b_log) {
    log_sum_diff_t res;
    if (a_log == -INFINITY && b_log == -INFINITY) {
        res.sum_log = -INFINITY;
        res.dif_log = -INFINITY;
        res.m.fa = 0;
        res.m.fb = 0;
        return res;
    }

    if (a_log > b_log) {
        res.m.fa = 1;
        res.m.fb = EXP(b_log - a_log);
        res.sum_log = a_log;
        res.dif_log = a_log;
    } else {
        res.m.fa = EXP(a_log - b_log);
        res.m.fb = 1;
        res.sum_log = b_log;
        res.dif_log = b_log;
    }
    return res;
}

typedef struct {
    vec_t unit_re;
    vec_t unit_im;
} uresult_t;

typedef struct {
    uresult_t sum;
    uresult_t dif;
} usum_diff_t;

static inline void unit_add_inplace(uresult_t* restrict a, const uresult_t b, const merge_t m) {
    a->unit_re = m.fa*a->unit_re + m.fb*b.unit_re;
    a->unit_im = m.fa*a->unit_im + m.fb*b.unit_im;
}

static inline usum_diff_t unit_sum_diff(const uresult_t* restrict a, const uresult_t* restrict b, const merge_t m) {
    usum_diff_t res;
    res.sum.unit_re = m.fa*a->unit_re + m.fb*b->unit_re;
    res.sum.unit_im = m.fa*a->unit_im + m.fb*b->unit_im;
    res.dif.unit_re = m.fa*a->unit_re - m.fb*b->unit_re;
    res.dif.unit_im = m.fa*a->unit_im - m.fb*b->unit_im;
    return res;
}

static inline uresult_t unit_product(const uresult_t* restrict a, const uresult_t* restrict b) {
    const uresult_t res = {
        .unit_re = a->unit_re*b->unit_re - a->unit_im*b->unit_im,
        .unit_im = a->unit_re*b->unit_im + a->unit_im*b->unit_re,
    };
    return res;
}

static inline uresult_t unit_cproduct(const complex_t a, const uresult_t b) {
    const uresult_t res = {
        .unit_re = creal(a)*b.unit_re - cimag(a)*b.unit_im,
        .unit_im = creal(a)*b.unit_im + cimag(a)*b.unit_re,
    };
    return res;
}

typedef struct {
    int_t M;
    int_t K;
    int_t N;
    int_t calls_per_round;   // number of merge call-sites in one category round (same every k)
    merge_t* restrict factors;  // K * calls_per_round, in call order
    real_t resN_exp;         // exp(A[N].log_mod - l_prev*N + lgac[N]) after all K rounds
} fourier_fft_plan_t;

static int_t fourier_fft_count_calls(const int_t M) {
    int_t count = 0;
    int_t N_curr = M, blocks = 1;
    for (; N_curr>THRESHOLD_PROD; N_curr>>=1) {
        const int_t width = N_curr/2;
        count += 2*blocks*width;  // one merge for A's halves, one for B's halves, per (block,j)
        blocks <<= 1;
    }
    count += blocks*N_curr*N_curr;  // leaf: each of the blocks*N_curr outputs sums N_curr terms
    blocks >>= 1;
    for (; N_curr<M; N_curr<<=1) {
        count += blocks*N_curr;  // ascend: one merge per (block,j)
        blocks >>= 1;
    }
    return count;
}

static fourier_fft_plan_t* build_fourier_fft_plan(const multinomial* restrict mult) {
    const cell_t* restrict global = mult->matrix;
    const int_t N = mult->N;
    const int_t K = mult->K;

    int_t M = 1;
    while (M<N+1) M<<=1;
    M<<=1;

    const int_t calls_per_round = fourier_fft_count_calls(M);

    fourier_fft_plan_t* plan = malloc(sizeof(fourier_fft_plan_t));
    plan->M = M;
    plan->K = K;
    plan->N = N;
    plan->calls_per_round = calls_per_round;
    plan->factors = malloc((size_t)K*calls_per_round*sizeof(merge_t));

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
                    const log_sum_diff_t sd1 = log_sum_diff(A[base + j], A[base + width + j]);
                    plan->factors[idx++] = sd1.m;
                    A[base + j] = sd1.sum_log;
                    A[base + width + j] = sd1.dif_log;

                    const log_sum_diff_t sd2 = log_sum_diff(B[base + j], B[base + width + j]);
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
                    plan->factors[idx++] = log_add_inplace(&acc, p_log);
                }
                for (int_t l=0; l<=i; l++) {
                    const real_t p_log = A[base + i - l] + B[base + l];
                    plan->factors[idx++] = log_add_inplace(&acc, p_log);
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
                    const log_sum_diff_t sd = log_sum_diff(C[base + j], C[base + width2 + j]);
                    plan->factors[idx++] = sd.m;
                    // cproduct(0.5, sum) / cproduct(half_inv_f, dif) leave log_mod unchanged
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

static void free_fourier_fft_plan(fourier_fft_plan_t* plan) {
    free(plan->factors);
    free(plan);
}

static void* eval_fourier_fft_precompute_thread(void* args_void) {
    arguments_t* args = args_void;

    const multinomial* restrict mult = args->mult;
    const cell_t* restrict global = mult->matrix;
    const fourier_fft_plan_t* restrict plan = args->plan;
    complex_t* restrict res_arr = args->res_arr;

    const int_t N = mult->N;
    const int_t K = mult->K;
    const int_t M = plan->M;
    const int_t calls_per_round = plan->calls_per_round;

    const uresult_t U_ZERO = {.unit_re = VEC_ONE, .unit_im = VEC_ZERO};

    uresult_t* restrict A = malloc(M*sizeof(uresult_t));
    uresult_t* restrict B = malloc(M*sizeof(uresult_t));
    uresult_t* restrict C = malloc(M*sizeof(uresult_t));
    complex_t* restrict F = malloc(M*sizeof(complex_t));

    const real_t s2_re = M_PI*mult->gamma/mult->T;

    for (int_t n=args->start; n<args->end; n++) {
        for (int_t i=0; i<M; i++) A[i] = U_ZERO;

        for (int_t k=K-1; k>=0; k--) {  // reversed
            const int_t index_k = k*(N+1);
            const merge_t* restrict mf = &plan->factors[(int_t)k*calls_per_round];
            int_t idx = 0;

            for (int_t i=0; i<=N; i++) {
                const int_t index = index_k + i;
                const real_t rewardT = global[index].reward;
#ifdef USE_SIMD
                for (int s=0; s<STRIDE; s++) {
                    const real_t arg = (n*STRIDE + s)*rewardT;
                    B[i].unit_re[s] = cos(arg);
                    B[i].unit_im[s] = -sin(arg);
                }
#else
                const real_t arg = n*rewardT;
                B[i].unit_re = cos(arg);
                B[i].unit_im = -sin(arg);
#endif
            }
            for (int_t i=N+1; i<M; i++) B[i] = U_ZERO;

            F[0] = 1;

            int_t N_curr = M;
            int_t blocks = 1;

            for (; N_curr>THRESHOLD_PROD; N_curr>>=1) {
                const int_t width = N_curr/2;
                for (int_t i=0; i<blocks; i++) {
                    const int_t base = i*N_curr;
                    const complex_t f_sqrt = CSQRT(F[base]);

                    for (int_t j = 0; j<width; j++) {
                        const uresult_t a1 = A[base + j];
                        const uresult_t a2 = unit_cproduct(f_sqrt, A[base + width + j]);
                        const usum_diff_t sd1 = unit_sum_diff(&a1, &a2, mf[idx++]);
                        A[base + j] = sd1.sum;
                        A[base + width + j] = sd1.dif;

                        const uresult_t b1 = B[base + j];
                        const uresult_t b2 = unit_cproduct(f_sqrt, B[base + width + j]);
                        const usum_diff_t sd2 = unit_sum_diff(&b1, &b2, mf[idx++]);
                        B[base + j] = sd2.sum;
                        B[base + width + j] = sd2.dif;
                    }

                    F[base] = f_sqrt;
                    F[base + width] = -f_sqrt;
                }
                blocks <<= 1;
            }

            for (int_t b=0; b<blocks; b++) {
                const int_t width = N_curr;
                const int_t base = b*width;

                for (int_t i=0; i<width; i++) {
                    uresult_t acc = U_ZERO;
                    for (int_t l=i+1; l<width; l++) {
                        const uresult_t p = unit_product(&A[base + width + i - l], &B[base + l]);
                        unit_add_inplace(&acc, p, mf[idx++]);
                    }
                    acc = unit_cproduct(F[base], acc);
                    for (int_t l=0; l<=i; l++) {
                        const uresult_t p = unit_product(&A[base + i - l], &B[base + l]);
                        unit_add_inplace(&acc, p, mf[idx++]);
                    }
                    C[base + i] = acc;
                }
            }

            blocks >>= 1;
            for (; N_curr<M; N_curr<<=1) {
                const int_t width2 = N_curr;
                for (int_t i=0; i<blocks; i++) {
                    const int_t base = i*width2*2;
                    const complex_t f_sqrt = F[base];
                    const complex_t half_inv_f = 1.0/(2.0*f_sqrt);

                    for (int_t j = 0; j<width2; j++) {
                        const uresult_t m1 = C[base + j];
                        const uresult_t m2 = C[base + width2 + j];
                        const usum_diff_t sd = unit_sum_diff(&m1, &m2, mf[idx++]);
                        C[base + j] = unit_cproduct(0.5, sd.sum);
                        C[base + width2 + j] = unit_cproduct(half_inv_f, sd.dif);
                    }

                    F[base] = f_sqrt*f_sqrt;
                }
                blocks >>= 1;
            }

            for (int_t i=N+1; i<M; i++) C[i] = U_ZERO;
            uresult_t* restrict temp = A;
            A = C;
            C = temp;
        }

#ifdef USE_SIMD
        vec_t s2_im; for (int_t i=0; i<STRIDE; i++) s2_im[i] = (n*STRIDE + i)*M_PI;

        for (int_t i=0; i<STRIDE; i++) {
            const complex_t unit = A[N].unit_re[i] + I*A[N].unit_im[i];
            const complex_t s2 = s2_re + I*s2_im[i];
            complex_t sinc = (1-EXPC(-s2))/s2;
            if ((n*STRIDE+i == 0) && (ABS(s2_re)<=EPS)) sinc = 1 - s2*(0.5 - s2/6);
            res_arr[n*STRIDE+i] = plan->resN_exp*(unit*sinc);
        }
#else
        const complex_t s2 = s2_re + I*n*M_PI;
        complex_t sinc = (1-EXPC(-s2))/s2;
        if ((n == 0) && (ABS(s2_re)<=EPS)) sinc = 1 - s2*(0.5 - s2/6);

        const complex_t unit = A[N].unit_re + I*A[N].unit_im;
        res_arr[n] = plan->resN_exp*(unit*sinc);
#endif
    }

    free(A);
    free(B);
    free(C);
    free(F);

    return 0;
}
