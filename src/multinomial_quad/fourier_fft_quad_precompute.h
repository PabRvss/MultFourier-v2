#pragma once

typedef struct {
    __float128 fa;
    __float128 fb;
} merge_q_t;

typedef struct {
    __float128 sum_log;
    __float128 dif_log;
    merge_q_t m;
} log_sum_diff_q_t;

static inline merge_q_t log_add_inplace_q(__float128* restrict a_log, const __float128 b_log) {
    merge_q_t m;
    if (*a_log == -HUGE_VALQ && b_log == -HUGE_VALQ) { m.fa = 1; m.fb = 0; return m; }

    if (*a_log > b_log) {
        m.fa = 1;
        m.fb = expq(b_log - *a_log);
    } else {
        m.fa = expq(*a_log - b_log);
        m.fb = 1;
        *a_log = b_log;
    }
    return m;
}

static inline log_sum_diff_q_t log_sum_diff_q(const __float128 a_log, const __float128 b_log) {
    log_sum_diff_q_t res;
    if (a_log == -HUGE_VALQ && b_log == -HUGE_VALQ) {
        res.sum_log = -HUGE_VALQ;
        res.dif_log = -HUGE_VALQ;
        res.m.fa = 0;
        res.m.fb = 0;
        return res;
    }

    if (a_log > b_log) {
        res.m.fa = 1;
        res.m.fb = expq(b_log - a_log);
        res.sum_log = a_log;
        res.dif_log = a_log;
    } else {
        res.m.fa = expq(a_log - b_log);
        res.m.fb = 1;
        res.sum_log = b_log;
        res.dif_log = b_log;
    }
    return res;
}

typedef struct {
    __complex128 unit;
} uresult_q_t;

typedef struct {
    uresult_q_t sum;
    uresult_q_t dif;
} usum_diff_q_t;

static inline void unit_add_inplace_q(uresult_q_t* restrict a, const uresult_q_t b, const merge_q_t m) {
    a->unit = m.fa*a->unit + m.fb*b.unit;
}

static inline usum_diff_q_t unit_sum_diff_q(const uresult_q_t* restrict a, const uresult_q_t* restrict b, const merge_q_t m) {
    usum_diff_q_t res;
    res.sum.unit = m.fa*a->unit + m.fb*b->unit;
    res.dif.unit = m.fa*a->unit - m.fb*b->unit;
    return res;
}

static inline uresult_q_t unit_product_q(const uresult_q_t* restrict a, const uresult_q_t* restrict b) {
    const uresult_q_t res = { .unit = a->unit * b->unit };
    return res;
}

static inline uresult_q_t unit_cproduct_q(const __complex128 a, const uresult_q_t b) {
    const uresult_q_t res = { .unit = a * b.unit };
    return res;
}

typedef struct {
    int_t M;
    int_t K;
    int_t N;
    int_t calls_per_round;
    merge_q_t* restrict factors;  // K * calls_per_round, in call order
    __float128 resN_exp;          // expq(A[N].log_mod - l_prev*N + lgac[N])
} fourier_fft_quad_plan_t;

static int_t fourier_fft_quad_count_calls(const int_t M) {
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

static fourier_fft_quad_plan_t* build_fourier_fft_quad_plan(const multinomial* restrict mult) {
    const cell_t* restrict global = mult->matrix;
    const int_t N = mult->N;
    const int_t K = mult->K;

    int_t M = 1;
    while (M<N+1) M<<=1;
    M<<=1;

    const int_t calls_per_round = fourier_fft_quad_count_calls(M);

    fourier_fft_quad_plan_t* plan = malloc(sizeof(fourier_fft_quad_plan_t));
    plan->M = M;
    plan->K = K;
    plan->N = N;
    plan->calls_per_round = calls_per_round;
    plan->factors = malloc((size_t)K*calls_per_round*sizeof(merge_q_t));

    const real_t nu_sp = mult->l_rate;
    real_t* restrict l_rate = malloc(K*sizeof(real_t));
    for (int_t k=0; k<K; k++) l_rate[k] = nu_sp;

    __float128* restrict A = malloc(M*sizeof(__float128));
    __float128* restrict B = malloc(M*sizeof(__float128));
    __float128* restrict C = malloc(M*sizeof(__float128));

    for (int_t i=0; i<M; i++) A[i] = -HUGE_VALQ;
    A[0] = 0;

    __float128 l_prev = 0;
    for (int_t k=K-1; k>=0; k--) {  // reversed
        const int_t index_k = k*(N+1);
        const __float128 l_k = l_rate[k];
        int_t idx = (int_t)k*calls_per_round;

        const __float128 d_k = l_k - l_prev;
        for (int_t j=0; j<M; j++) A[j] += d_k * (__float128)j;
        l_prev = l_k;

        for (int_t i=0; i<=N; i++) {
            const int_t index = index_k + i;
            B[i] = (__float128)global[index].log_prob + l_k*(__float128)i;
        }
        for (int_t i=N+1; i<M; i++) B[i] = -HUGE_VALQ;

        int_t N_curr = M;
        int_t blocks = 1;

        for (; N_curr>THRESHOLD_PROD; N_curr>>=1) {
            const int_t width = N_curr/2;
            for (int_t i=0; i<blocks; i++) {
                const int_t base = i*N_curr;

                for (int_t j = 0; j<width; j++) {
                    const log_sum_diff_q_t sd1 = log_sum_diff_q(A[base + j], A[base + width + j]);
                    plan->factors[idx++] = sd1.m;
                    A[base + j] = sd1.sum_log;
                    A[base + width + j] = sd1.dif_log;

                    const log_sum_diff_q_t sd2 = log_sum_diff_q(B[base + j], B[base + width + j]);
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
                __float128 acc = -HUGE_VALQ;
                for (int_t l=i+1; l<width; l++) {
                    const __float128 p_log = A[base + width + i - l] + B[base + l];
                    plan->factors[idx++] = log_add_inplace_q(&acc, p_log);
                }
                for (int_t l=0; l<=i; l++) {
                    const __float128 p_log = A[base + i - l] + B[base + l];
                    plan->factors[idx++] = log_add_inplace_q(&acc, p_log);
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
                    const log_sum_diff_q_t sd = log_sum_diff_q(C[base + j], C[base + width2 + j]);
                    plan->factors[idx++] = sd.m;
                    C[base + j] = sd.sum_log;
                    C[base + width2 + j] = sd.dif_log;
                }
            }
            blocks >>= 1;
        }

        for (int_t i=N+1; i<M; i++) C[i] = -HUGE_VALQ;
        __float128* restrict temp = A;
        A = C;
        C = temp;
    }

    plan->resN_exp = expq(A[N] - l_prev*(__float128)N + (__float128)lgac[N]);

    free(l_rate);
    free(A);
    free(B);
    free(C);

    return plan;
}

static void free_fourier_fft_quad_plan(fourier_fft_quad_plan_t* plan) {
    free(plan->factors);
    free(plan);
}

static void* eval_fourier_fft_quad_precompute_thread(void* args_void) {
    arguments_t* args = args_void;

    const multinomial* restrict mult = args->mult;
    const cell_t* restrict global = mult->matrix;
    const fourier_fft_quad_plan_t* restrict plan = args->plan;
    complex_t* restrict res_arr = args->res_arr;

    const int_t N = mult->N;
    const int_t K = mult->K;
    const int_t M = plan->M;
    const int_t calls_per_round = plan->calls_per_round;

    const uresult_q_t U_ZERO_Q = { .unit = 1.0q };

    uresult_q_t* restrict A = malloc(M*sizeof(uresult_q_t));
    uresult_q_t* restrict B = malloc(M*sizeof(uresult_q_t));
    uresult_q_t* restrict C = malloc(M*sizeof(uresult_q_t));
    __complex128* restrict F = malloc(M*sizeof(__complex128));

    const __float128 s2_re = M_PIq*(__float128)mult->gamma/(__float128)mult->T;

    for (int_t n_outer=args->start; n_outer<args->end; n_outer++) {
#ifdef USE_SIMD
    for (int_t s=0; s<STRIDE; s++) {
        const int_t n = n_outer*STRIDE + s;
#else
    for (int_t s=0; s<1; s++) {
        const int_t n = n_outer;
#endif
        for (int_t i=0; i<M; i++) A[i] = U_ZERO_Q;

        for (int_t k=K-1; k>=0; k--) {  // reversed
            const int_t index_k = k*(N+1);
            const merge_q_t* restrict mf = &plan->factors[(int_t)k*calls_per_round];
            int_t idx = 0;

            for (int_t i=0; i<=N; i++) {
                const int_t index = index_k + i;
                const real_t rewardT = global[index].reward;
                const real_t arg = (real_t)n*rewardT;
                B[i].unit = cosq((__float128)arg) - I*sinq((__float128)arg);
            }
            for (int_t i=N+1; i<M; i++) B[i] = U_ZERO_Q;

            F[0] = 1;

            int_t N_curr = M;
            int_t blocks = 1;

            for (; N_curr>THRESHOLD_PROD; N_curr>>=1) {
                const int_t width = N_curr/2;
                for (int_t i=0; i<blocks; i++) {
                    const int_t base = i*N_curr;
                    const __complex128 f_sqrt = csqrtq(F[base]);

                    for (int_t j = 0; j<width; j++) {
                        const uresult_q_t a1 = A[base + j];
                        const uresult_q_t a2 = unit_cproduct_q(f_sqrt, A[base + width + j]);
                        const usum_diff_q_t sd1 = unit_sum_diff_q(&a1, &a2, mf[idx++]);
                        A[base + j] = sd1.sum;
                        A[base + width + j] = sd1.dif;

                        const uresult_q_t b1 = B[base + j];
                        const uresult_q_t b2 = unit_cproduct_q(f_sqrt, B[base + width + j]);
                        const usum_diff_q_t sd2 = unit_sum_diff_q(&b1, &b2, mf[idx++]);
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
                    uresult_q_t acc = U_ZERO_Q;
                    for (int_t l=i+1; l<width; l++) {
                        const uresult_q_t p = unit_product_q(&A[base + width + i - l], &B[base + l]);
                        unit_add_inplace_q(&acc, p, mf[idx++]);
                    }
                    acc = unit_cproduct_q(F[base], acc);
                    for (int_t l=0; l<=i; l++) {
                        const uresult_q_t p = unit_product_q(&A[base + i - l], &B[base + l]);
                        unit_add_inplace_q(&acc, p, mf[idx++]);
                    }
                    C[base + i] = acc;
                }
            }

            blocks >>= 1;
            for (; N_curr<M; N_curr<<=1) {
                const int_t width2 = N_curr;
                for (int_t i=0; i<blocks; i++) {
                    const int_t base = i*width2*2;
                    const __complex128 f_sqrt = F[base];
                    const __complex128 half_inv_f = 1.0q/(2.0q*f_sqrt);

                    for (int_t j = 0; j<width2; j++) {
                        const uresult_q_t m1 = C[base + j];
                        const uresult_q_t m2 = C[base + width2 + j];
                        const usum_diff_q_t sd = unit_sum_diff_q(&m1, &m2, mf[idx++]);
                        C[base + j] = unit_cproduct_q(0.5q, sd.sum);
                        C[base + width2 + j] = unit_cproduct_q(half_inv_f, sd.dif);
                    }

                    F[base] = f_sqrt*f_sqrt;
                }
                blocks >>= 1;
            }

            for (int_t i=N+1; i<M; i++) C[i] = U_ZERO_Q;
            uresult_q_t* restrict temp = A;
            A = C;
            C = temp;
        }

        const __float128 s2_im = (__float128)n*M_PIq;
        const __complex128 s2 = s2_re + I*s2_im;
        __complex128 sinc = (1.0q - cexpq(-s2))/s2;
        if ((n == 0) && (fabsq(s2_re)<=(__float128)EPS)) sinc = 1.0q - s2*(0.5q - s2/6.0q);

        const __complex128 res_q = plan->resN_exp*(A[N].unit*sinc);
        res_arr[n] = (real_t)crealq(res_q) + I*(real_t)cimagq(res_q);
    }
    }

    free(A);
    free(B);
    free(C);
    free(F);

    return 0;
}
