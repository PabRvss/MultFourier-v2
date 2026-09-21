#pragma once

typedef struct {
    __float128 log_mod;
    __complex128 unit;
} gqresult_t;

typedef struct {
    gqresult_t sum;
    gqresult_t dif;
} gqsum_diff_t;

static inline void gq_add_inplace(gqresult_t* restrict a, const gqresult_t b) {
    if (a->log_mod == -HUGE_VALQ && b.log_mod == -HUGE_VALQ) return;

    if (a->log_mod > b.log_mod) {
        const __float128 f = expq(b.log_mod - a->log_mod);
        a->unit += f*b.unit;
    } else {
        const __float128 f = expq(a->log_mod - b.log_mod);
        a->log_mod = b.log_mod;
        a->unit = f*a->unit + b.unit;
    }
}

static inline gqsum_diff_t gq_sum_diff(const gqresult_t* restrict a, const gqresult_t* restrict b) {
    gqsum_diff_t res;
    if (a->log_mod == -HUGE_VALQ && b->log_mod == -HUGE_VALQ) {
        res.sum.log_mod = -HUGE_VALQ;
        res.sum.unit = 0.0q;
        res.dif.log_mod = -HUGE_VALQ;
        res.dif.unit = 0.0q;
        return res;
    }

    if (a->log_mod > b->log_mod) {
        const __float128 f = expq(b->log_mod - a->log_mod);
        res.sum.log_mod = a->log_mod;
        res.sum.unit = a->unit + f*b->unit;
        res.dif.log_mod = a->log_mod;
        res.dif.unit = a->unit - f*b->unit;
    } else {
        const __float128 f = expq(a->log_mod - b->log_mod);
        res.sum.log_mod = b->log_mod;
        res.sum.unit = f*a->unit + b->unit;
        res.dif.log_mod = b->log_mod;
        res.dif.unit = f*a->unit - b->unit;
    }
    return res;
}

static inline gqresult_t gq_product(const gqresult_t* restrict a, const gqresult_t* restrict b) {
    const gqresult_t res = {
        .log_mod = a->log_mod + b->log_mod,
        .unit = a->unit * b->unit,
    };
    return res;
}

static inline gqresult_t gq_cproduct(const __complex128 a, const gqresult_t b) {
    const gqresult_t res = {
        .log_mod = b.log_mod,
        .unit = a * b.unit,
    };
    return res;
}

static void* eval_gammas_fft_quad_thread(void* args_void) {
    args_gamma_t* args = args_void;

    const multinomial* restrict mult = args->mult;
    const cell_t* restrict global = mult->matrix;
    const int_t thread = args->thread;
    const int_t vecs_per_thread = args->vecs_per_thread;
    real_t* restrict res_arr = args->res_arr;
    const vec_t* restrict gammas_vec = args->gammas_vec;

    const int_t N = mult->N;
    const int_t K = mult->K;

    const gqresult_t ZERO_G = {.log_mod = -HUGE_VALQ, .unit = 1.0q};

    int_t M = 1;
    while (M<N+1) M<<=1;
    M<<=1;

    gqresult_t* restrict A = malloc(M*sizeof(gqresult_t));
    gqresult_t* restrict B = malloc(M*sizeof(gqresult_t));
    gqresult_t* restrict C = malloc(M*sizeof(gqresult_t));
    __complex128* restrict F = malloc(M*sizeof(__complex128));

    for (int_t i_vec=0; i_vec<vecs_per_thread; i_vec++) {
        const int_t idx_vec = vecs_per_thread*thread + i_vec;

        const vec_t nu_vec = find_nu_gamma_vec(global, N, K, gammas_vec[idx_vec]);
        args->l_rate_arr[idx_vec] = nu_vec;

#ifdef USE_SIMD
        for (int s=0; s<STRIDE; s++) {
            const real_t gamma_s = gammas_vec[idx_vec][s];
            const real_t nu = nu_vec[s];
#else
        for (int s=0; s<1; s++) {
            const real_t gamma_s = gammas_vec[idx_vec];
            const real_t nu = nu_vec;
#endif
            for (int_t i=0; i<M; i++) A[i] = ZERO_G;
            A[0].log_mod = 0;

            real_t l_prev = 0;
            for (int_t k=K-1; k>=0; k--) {  // reversed
                const int_t index_k = k*(N+1);
                const real_t l_k = nu;

                const real_t d_k = l_k - l_prev;
                for (int_t j=0; j<M; j++) A[j].log_mod += (__float128)d_k * (__float128)j;
                l_prev = l_k;

                for (int_t i=0; i<=N; i++) {
                    const int_t index = index_k + i;
                    const real_t raw_i = -global[index].reward*gamma_s + global[index].log_prob;
                    B[i].log_mod = (__float128)raw_i + (__float128)l_k*(__float128)i;
                    B[i].unit = 1.0q;
                }
                for (int_t i=N+1; i<M; i++) {
                    B[i] = ZERO_G;
                }

                F[0] = 1;

                int_t N_curr = M;
                int_t blocks = 1;

                for (; N_curr>THRESHOLD_PROD; N_curr>>=1) {
                    const int_t width = N_curr/2;
                    for (int_t i=0; i<blocks; i++) {
                        const int_t base = i*N_curr;
                        const __complex128 f_sqrt = csqrtq(F[base]);

                        for (int_t j = 0; j<width; j++) {
                            const gqresult_t a1 = A[base + j];
                            const gqresult_t a2 = gq_cproduct(f_sqrt, A[base + width + j]);
                            const gqsum_diff_t sd1 = gq_sum_diff(&a1, &a2);
                            A[base + j] = sd1.sum;
                            A[base + width + j] = sd1.dif;

                            const gqresult_t b1 = B[base + j];
                            const gqresult_t b2 = gq_cproduct(f_sqrt, B[base + width + j]);
                            const gqsum_diff_t sd2 = gq_sum_diff(&b1, &b2);
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
                        gqresult_t acc = ZERO_G;
                        for (int_t l=i+1; l<width; l++) gq_add_inplace(&acc, gq_product(&A[base + width + i - l], &B[base + l]));
                        acc = gq_cproduct(F[base], acc);
                        for (int_t l=0; l<=i; l++) gq_add_inplace(&acc, gq_product(&A[base + i - l], &B[base + l]));
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
                            const gqresult_t m1 = C[base + j];
                            const gqresult_t m2 = C[base + width2 + j];
                            const gqsum_diff_t sd = gq_sum_diff(&m1, &m2);
                            C[base + j] = gq_cproduct(0.5q, sd.sum);
                            C[base + width2 + j] = gq_cproduct(half_inv_f, sd.dif);
                        }

                        F[base] = f_sqrt*f_sqrt;
                    }
                    blocks >>= 1;
                }

                for (int_t i=N+1; i<M; i++) C[i] = ZERO_G;
                gqresult_t* restrict temp = A;
                A = C;
                C = temp;
            }

            const __float128 mag = cabsq(A[N].unit);
            const double resN = (double)(A[N].log_mod - (__float128)l_prev*(__float128)N) + log((double)mag) + lgac[N];

            const real_t s2 = M_PI*gamma_s/mult->T;
            real_t sinc = LOG((1-EXP(-s2))/s2);
            if (s2 < -MAX_EXP) sinc = -s2 - LOG(-s2);
            if (ABS(s2)<=EPS) sinc = -s2*(0.5 - s2/24);

#ifdef USE_SIMD
            res_arr[idx_vec*STRIDE + s] = resN + sinc;
#else
            res_arr[idx_vec] = resN + sinc;
#endif
        }
    }

    free(A);
    free(B);
    free(C);
    free(F);

    return 0;
}
