#pragma once

typedef struct {
    result_t sum;
    result_t dif;
} sum_diff_t;

static inline void add_inplace(result_t* restrict a, const result_t b) {
    if (a->log_mod == -INFINITY && b.log_mod == -INFINITY) return;

    if (a->log_mod > b.log_mod) {
        const real_t f = EXP(b.log_mod - a->log_mod);
        a->unit_re += f*b.unit_re;
        a->unit_im += f*b.unit_im;
    } else {
        const real_t f = EXP(a->log_mod - b.log_mod);
        a->log_mod = b.log_mod;
        a->unit_re = f*a->unit_re + b.unit_re;
        a->unit_im = f*a->unit_im + b.unit_im;
    }
}

static inline sum_diff_t sum_diff(const result_t* restrict a, const result_t* restrict b) {
    sum_diff_t res;
    if (a->log_mod == -INFINITY && b->log_mod == -INFINITY) {
        res.sum.log_mod = -INFINITY;
        res.sum.unit_re = VEC_ZERO;
        res.sum.unit_im = VEC_ZERO;

        res.dif.log_mod = -INFINITY;
        res.dif.unit_re = VEC_ZERO;
        res.dif.unit_im = VEC_ZERO;

        return res;
    }

    if (a->log_mod > b->log_mod) {
        const real_t f = EXP(b->log_mod - a->log_mod);

        res.sum.log_mod = a->log_mod;
        res.sum.unit_re = a->unit_re + f*b->unit_re;
        res.sum.unit_im = a->unit_im + f*b->unit_im;

        res.dif.log_mod = a->log_mod;
        res.dif.unit_re = a->unit_re - f*b->unit_re;
        res.dif.unit_im = a->unit_im - f*b->unit_im;
    } else {
        const real_t f = EXP(a->log_mod - b->log_mod);

        res.sum.log_mod = b->log_mod;
        res.sum.unit_re = f*a->unit_re + b->unit_re;
        res.sum.unit_im = f*a->unit_im + b->unit_im;

        res.dif.log_mod = b->log_mod;
        res.dif.unit_re = f*a->unit_re - b->unit_re;
        res.dif.unit_im = f*a->unit_im - b->unit_im;
    }
    return res;
}

static inline result_t product(const result_t* restrict a, const result_t* restrict b) {
    const result_t res = {
        .log_mod = a->log_mod + b->log_mod,
        .unit_re = a->unit_re*b->unit_re - a->unit_im*b->unit_im,
        .unit_im = a->unit_re*b->unit_im + a->unit_im*b->unit_re,
    };
    return res;
}

static inline result_t cproduct(const complex_t a, const result_t b) {
    const result_t res = {
        .log_mod = b.log_mod,
        .unit_re = creal(a)*b.unit_re - cimag(a)*b.unit_im,
        .unit_im = creal(a)*b.unit_im + cimag(a)*b.unit_re,
    };
    return res;
}

static void* eval_fourier_fft_thread(void* args_void) {
    arguments_t* args = args_void;

    const multinomial* restrict mult = args->mult;
    const cell_t* restrict global = mult->matrix;
    const result_t ZERO = {.log_mod = -INFINITY, .unit_re = VEC_ONE, .unit_im = VEC_ZERO};

    const int_t N = mult->N;
    const int_t K = mult->K;

    int_t M = 1;
    while (M<N+1) M<<=1;
    M<<=1;

    real_t* restrict l_rate = malloc(K*sizeof(real_t));
    for (int_t k=0; k<K; k++) {
        const int_t index_k = k*(N+1);
        l_rate[k] = N > 0 ? -(global[index_k+N].log_prob - global[index_k+0].log_prob)/(real_t)N : 0;
    }

    result_t* restrict A = malloc(M*sizeof(result_t));
    result_t* restrict B = malloc(M*sizeof(result_t));
    result_t* restrict C = malloc(M*sizeof(result_t));
    complex_t* restrict F = malloc(M*sizeof(complex_t));

    for (int_t n=args->start; n<args->end; n++) {
        for (int_t i=0; i<M; i++) A[i] = ZERO;
        A[0].log_mod = 0;

        real_t l_prev = 0;
        for (int_t k=K-1; k>=0; k--) {  // reversed
            const int_t index_k = k*(N+1);
            const real_t l_k = l_rate[k];

            const real_t d_k = l_k - l_prev;
            for (int_t j=0; j<M; j++) A[j].log_mod += d_k * (real_t)j;
            l_prev = l_k;

            // Update B for k, scaled by this category's own l_k alone.
            for (int_t i=0; i<=N; i++) {
                const int_t index = index_k + i;
                const real_t rewardT = global[index].reward;
                B[i].log_mod = global[index].log_prob + l_k*(real_t)i;
#ifdef USE_SIMD
                for (int s=0; s<STRIDE; s++) {
                    const real_t arg = (n*STRIDE + s)*rewardT;
                    B[i].unit_re[s] = cos(arg);
                    B[i].unit_im[s] = -sin(arg);
                }
#else
                const real_t arg = n*rewardT;
                B[i].unit_re = cos(arg);
                B[i].unit_im = -sin(arg);  // conjugated
#endif
            }
            for (int_t i=N+1; i<M; i++) {
                B[i] = ZERO;
            }

            F[0] = 1;

            int_t N_curr = M;
            int_t blocks = 1;

            for (; N_curr>THRESHOLD_PROD; N_curr>>=1) {
                const int_t width = N_curr/2;
                for (int_t i=0; i<blocks; i++) {
                    const int_t base = i*N_curr;
                    const complex_t f_sqrt = CSQRT(F[base]);

                    for (int_t j = 0; j<width; j++) {
                        const result_t a1 = A[base + j];
                        const result_t a2 = cproduct(f_sqrt, A[base + width + j]);

                        const sum_diff_t sd1 = sum_diff(&a1, &a2);
                        A[base + j] = sd1.sum;
                        A[base + width + j] = sd1.dif;

                        const result_t b1 = B[base + j];
                        const result_t b2 = cproduct(f_sqrt, B[base + width + j]);

                        const sum_diff_t sd2 = sum_diff(&b1, &b2);
                        B[base + j] = sd2.sum;
                        B[base + width + j] = sd2.dif;
                    }

                    F[base] = f_sqrt;
                    F[base + width] = -f_sqrt;
                }
                blocks <<= 1;
            }

            // direct convolution at the leaf rings
            for (int_t b=0; b<blocks; b++) {
                const int_t width = N_curr;
                const int_t base = b*width;

                for (int_t i=0; i<width; i++) {
                    result_t acc = ZERO;
                    for (int_t l=i+1; l<width; l++) add_inplace(&acc, product(&A[base + width + i - l], &B[base + l]));
                    acc = cproduct(F[base], acc);
                    for (int_t l=0; l<=i; l++) add_inplace(&acc, product(&A[base + i - l], &B[base + l]));
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
                        const result_t m1 = C[base + j];
                        const result_t m2 = C[base + width2 + j];

                        const sum_diff_t sd = sum_diff(&m1, &m2);
                        C[base + j] = cproduct(0.5, sd.sum);
                        C[base + width2 + j] = cproduct(half_inv_f, sd.dif);
                    }

                    F[base] = f_sqrt*f_sqrt;
                }
                blocks >>= 1;
            }

            for (int_t i=N+1; i<M; i++) C[i] = ZERO;
            result_t* restrict temp = A;
            A = C;
            C = temp;
        }

        qcomplex_t* restrict res_arr = args->res_arr;
        const real_t s2_re = M_PI*mult->gamma/mult->T;
        const quad_t resN_exp = exp_quad(A[N].log_mod - l_prev*(real_t)N + lgac[N]);
#ifdef USE_SIMD
        vec_t s2_im; for (int_t i=0; i<STRIDE; i++) s2_im[i] = (n*STRIDE + i)*M_PI;

        for (int_t i=0; i<STRIDE; i++) {
            const complex_t unit = A[N].unit_re[i] + I*A[N].unit_im[i];
            const complex_t s2 = s2_re + I*s2_im[i];
            complex_t sinc = (1-EXPC(-s2))/s2;
            if ((n*STRIDE+i == 0) && (ABS(s2_re)<=EPS)) sinc = 1 - s2*(0.5 - s2/6);
            res_arr[n*STRIDE+i] = resN_exp*(unit*sinc);
        }
#else
        const complex_t s2 = s2_re + I*n*M_PI;
        complex_t sinc = (1-EXPC(-s2))/s2;
        if ((n == 0) && (ABS(s2_re)<=EPS)) sinc = 1 - s2*(0.5 - s2/6);

        const complex_t unit = A[N].unit_re + I*A[N].unit_im;
        res_arr[n] = resN_exp*(unit*sinc);
#endif
    }

    free(A);
    free(B);
    free(C);
    free(F);
    free(l_rate);

    return 0;
}

static void* eval_fourier_fft_sp_thread(void* args_void) {
    arguments_t* args = args_void;

    const multinomial* restrict mult = args->mult;
    const cell_t* restrict global = mult->matrix;
    const result_t ZERO = {.log_mod = -INFINITY, .unit_re = VEC_ONE, .unit_im = VEC_ZERO};

    const int_t N = mult->N;
    const int_t K = mult->K;

    int_t M = 1;
    while (M<N+1) M<<=1;
    M<<=1;

    const real_t nu_sp = mult->l_rate;
    real_t* restrict l_rate = malloc(K*sizeof(real_t));
    for (int_t k=0; k<K; k++) l_rate[k] = nu_sp;

    result_t* restrict A = malloc(M*sizeof(result_t));
    result_t* restrict B = malloc(M*sizeof(result_t));
    result_t* restrict C = malloc(M*sizeof(result_t));
    complex_t* restrict F = malloc(M*sizeof(complex_t));

    for (int_t n=args->start; n<args->end; n++) {
        for (int_t i=0; i<M; i++) A[i] = ZERO;
        A[0].log_mod = 0;

        real_t l_prev = 0;
        for (int_t k=K-1; k>=0; k--) {  // reversed
            const int_t index_k = k*(N+1);
            const real_t l_k = l_rate[k];

            const real_t d_k = l_k - l_prev;
            for (int_t j=0; j<M; j++) A[j].log_mod += d_k * (real_t)j;
            l_prev = l_k;

            // Update B for k, scaled by this category's own l_k alone.
            for (int_t i=0; i<=N; i++) {
                const int_t index = index_k + i;
                const real_t rewardT = global[index].reward;
                B[i].log_mod = global[index].log_prob + l_k*(real_t)i;
#ifdef USE_SIMD
                for (int s=0; s<STRIDE; s++) {
                    const real_t arg = (n*STRIDE + s)*rewardT;
                    B[i].unit_re[s] = cos(arg);
                    B[i].unit_im[s] = -sin(arg);
                }
#else
                const real_t arg = n*rewardT;
                B[i].unit_re = cos(arg);
                B[i].unit_im = -sin(arg);  // conjugated
#endif
            }
            for (int_t i=N+1; i<M; i++) {
                B[i] = ZERO;
            }

            F[0] = 1;

            int_t N_curr = M;
            int_t blocks = 1;

            for (; N_curr>THRESHOLD_PROD; N_curr>>=1) {
                const int_t width = N_curr/2;
                for (int_t i=0; i<blocks; i++) {
                    const int_t base = i*N_curr;
                    const complex_t f_sqrt = CSQRT(F[base]);

                    for (int_t j = 0; j<width; j++) {
                        const result_t a1 = A[base + j];
                        const result_t a2 = cproduct(f_sqrt, A[base + width + j]);

                        const sum_diff_t sd1 = sum_diff(&a1, &a2);
                        A[base + j] = sd1.sum;
                        A[base + width + j] = sd1.dif;

                        const result_t b1 = B[base + j];
                        const result_t b2 = cproduct(f_sqrt, B[base + width + j]);

                        const sum_diff_t sd2 = sum_diff(&b1, &b2);
                        B[base + j] = sd2.sum;
                        B[base + width + j] = sd2.dif;
                    }

                    F[base] = f_sqrt;
                    F[base + width] = -f_sqrt;
                }
                blocks <<= 1;
            }

            // direct convolution at the leaf rings
            for (int_t b=0; b<blocks; b++) {
                const int_t width = N_curr;
                const int_t base = b*width;

                for (int_t i=0; i<width; i++) {
                    result_t acc = ZERO;
                    for (int_t l=i+1; l<width; l++) add_inplace(&acc, product(&A[base + width + i - l], &B[base + l]));
                    acc = cproduct(F[base], acc);
                    for (int_t l=0; l<=i; l++) add_inplace(&acc, product(&A[base + i - l], &B[base + l]));
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
                        const result_t m1 = C[base + j];
                        const result_t m2 = C[base + width2 + j];

                        const sum_diff_t sd = sum_diff(&m1, &m2);
                        C[base + j] = cproduct(0.5, sd.sum);
                        C[base + width2 + j] = cproduct(half_inv_f, sd.dif);
                    }

                    F[base] = f_sqrt*f_sqrt;
                }
                blocks >>= 1;
            }

            for (int_t i=N+1; i<M; i++) C[i] = ZERO;
            result_t* restrict temp = A;
            A = C;
            C = temp;
        }

        qcomplex_t* restrict res_arr = args->res_arr;
        const real_t s2_re = M_PI*mult->gamma/mult->T;
        const quad_t resN_exp = exp_quad(A[N].log_mod - l_prev*(real_t)N + lgac[N]);
#ifdef USE_SIMD
        vec_t s2_im; for (int_t i=0; i<STRIDE; i++) s2_im[i] = (n*STRIDE + i)*M_PI;

        for (int_t i=0; i<STRIDE; i++) {
            const complex_t unit = A[N].unit_re[i] + I*A[N].unit_im[i];
            const complex_t s2 = s2_re + I*s2_im[i];
            complex_t sinc = (1-EXPC(-s2))/s2;
            if ((n*STRIDE+i == 0) && (ABS(s2_re)<=EPS)) sinc = 1 - s2*(0.5 - s2/6);
            res_arr[n*STRIDE+i] = resN_exp*(unit*sinc);
        }
#else
        const complex_t s2 = s2_re + I*n*M_PI;
        complex_t sinc = (1-EXPC(-s2))/s2;
        if ((n == 0) && (ABS(s2_re)<=EPS)) sinc = 1 - s2*(0.5 - s2/6);

        const complex_t unit = A[N].unit_re + I*A[N].unit_im;
        res_arr[n] = resN_exp*(unit*sinc);
#endif
    }

    free(A);
    free(B);
    free(C);
    free(F);
    free(l_rate);

    return 0;
}
