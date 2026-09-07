#pragma once

#define NEG_HUGE_GAMMA (-1e30)

typedef struct {
    vec_t log_mod;
    vec_t unit_re;
    vec_t unit_im;
} gresult_t;

typedef struct {
    gresult_t sum;
    gresult_t dif;
} gsumdiff_t;

static inline vec_t vexp_(const vec_t x) {
#ifdef USE_SIMD
    vec_t res;
    for (int s=0; s<STRIDE; s++) res[s] = EXP(x[s]);
    return res;
#else
    return EXP(x);
#endif
}

static inline void gadd_(gresult_t* restrict a, const gresult_t b) {
    const vec_t log_mod = VEC_MAX(a->log_mod, b.log_mod);
    const vec_t f1 = vexp_(a->log_mod - log_mod);
    const vec_t f2 = vexp_(b.log_mod - log_mod);
    a->log_mod = log_mod;
    a->unit_re = f1*a->unit_re + f2*b.unit_re;
    a->unit_im = f1*a->unit_im + f2*b.unit_im;
}

static inline gsumdiff_t gsumdiff_(const gresult_t* restrict a, const gresult_t* restrict b) {
    const vec_t log_mod = VEC_MAX(a->log_mod, b->log_mod);
    const vec_t f1 = vexp_(a->log_mod - log_mod);
    const vec_t f2 = vexp_(b->log_mod - log_mod);

    gsumdiff_t res;
    res.sum.log_mod = log_mod;
    res.sum.unit_re = f1*a->unit_re + f2*b->unit_re;
    res.sum.unit_im = f1*a->unit_im + f2*b->unit_im;

    res.dif.log_mod = log_mod;
    res.dif.unit_re = f1*a->unit_re - f2*b->unit_re;
    res.dif.unit_im = f1*a->unit_im - f2*b->unit_im;
    return res;
}

static inline gresult_t gprod_(const gresult_t* restrict a, const gresult_t* restrict b) {
    const gresult_t res = {
        .log_mod = a->log_mod + b->log_mod,
        .unit_re = a->unit_re*b->unit_re - a->unit_im*b->unit_im,
        .unit_im = a->unit_re*b->unit_im + a->unit_im*b->unit_re,
    };
    return res;
}

static inline gresult_t gcprod_(const complex_t a, const gresult_t b) {
    const gresult_t res = {
        .log_mod = b.log_mod,
        .unit_re = creal(a)*b.unit_re - cimag(a)*b.unit_im,
        .unit_im = creal(a)*b.unit_im + cimag(a)*b.unit_re,
    };
    return res;
}

#define GAMMA_NEWTON_ITERS 30
#define GAMMA_NU_BOUND 100.0

static inline vec_t find_nu_gamma_vec(const cell_t* restrict global, const int_t N, const int_t K, const vec_t gamma) {
    if (N <= 0) return VEC_ZERO;

    vec_t lo = VEC_SET1(-GAMMA_NU_BOUND);
    vec_t hi = VEC_SET1(GAMMA_NU_BOUND);
    vec_t nu = VEC_ZERO;

    for (int_t iter=0; iter<GAMMA_NEWTON_ITERS; iter++) {
        vec_t sum_m = VEC_ZERO, sum_var = VEC_ZERO;
        for (int_t k=0; k<K; k++) {
            const int_t index_k = k*(N+1);

            vec_t max_exp = VEC_SET1(-INFINITY);
            for (int_t i=0; i<=N; i++) {
                const cell_t c = global[index_k+i];
                const vec_t e = -c.reward*gamma + VEC_SET1(c.log_prob) + nu*VEC_SET1((real_t)i);
                max_exp = VEC_MAX(e, max_exp);
            }

            vec_t Z = VEC_ZERO, S1 = VEC_ZERO, S2 = VEC_ZERO;
            for (int_t i=0; i<=N; i++) {
                const cell_t c = global[index_k+i];
                const vec_t e = -c.reward*gamma + VEC_SET1(c.log_prob) + nu*VEC_SET1((real_t)i) - max_exp;
#ifdef USE_SIMD
                vec_t w; for (int s=0; s<STRIDE; s++) w[s] = EXP(e[s]);
#else
                const vec_t w = EXP(e);
#endif
                Z += w;
                S1 += w*VEC_SET1((real_t)i);
                S2 += w*VEC_SET1((real_t)i)*VEC_SET1((real_t)i);
            }

            const vec_t m = S1/Z;
            sum_m += m;
            sum_var += S2/Z - m*m;
        }

        const vec_t f = sum_m - VEC_SET1((real_t)N);

        const vec_mask_t f_pos = VEC_CMP_GT(f, VEC_ZERO);
        hi = VEC_SELECT(f_pos, nu, hi);
        lo = VEC_SELECT(f_pos, lo, nu);

        const vec_t fp = VEC_MAX(sum_var, VEC_SET1(1e-300));
        const vec_t newton_nu = nu - f/fp;
        const vec_t bisect_nu = VEC_SET1(0.5)*(lo + hi);
        const vec_mask_t newton_ok = VEC_CMP_GT(newton_nu, lo) & VEC_CMP_LT(newton_nu, hi);
        nu = VEC_SELECT(newton_ok, newton_nu, bisect_nu);
    }
    return nu;
}

static void* eval_gammas_fft_thread(void* args_void) {
    args_gamma_t* args = args_void;

    const multinomial* restrict mult = args->mult;
    const cell_t* restrict global = mult->matrix;
    const int_t thread = args->thread;
    const int_t vecs_per_thread = args->vecs_per_thread;
    real_t* restrict res_arr = args->res_arr;
    const vec_t* restrict gammas_vec = args->gammas_vec;

    const int_t N = mult->N;
    const int_t K = mult->K;

    const gresult_t ZERO_G = {.log_mod = NEG_HUGE_GAMMA*VEC_ONE, .unit_re = VEC_ONE, .unit_im = VEC_ZERO};

    int_t M = 1;
    while (M<N+1) M<<=1;
    M<<=1;

    gresult_t* restrict A = malloc(M*sizeof(gresult_t));
    gresult_t* restrict B = malloc(M*sizeof(gresult_t));
    gresult_t* restrict C = malloc(M*sizeof(gresult_t));
    complex_t* restrict F = malloc(M*sizeof(complex_t));

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
            for (int_t j=0; j<M; j++) A[j].log_mod += d_k * (real_t)j;
            l_prev = l_k;

            for (int_t i=0; i<=N; i++) {
                const int_t index = index_k + i;
                B[i].log_mod = -global[index].reward*gamma + global[index].log_prob + l_k*(real_t)i;
                B[i].unit_re = VEC_ONE;
                B[i].unit_im = VEC_ZERO;
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
                    const complex_t f_sqrt = CSQRT(F[base]);

                    for (int_t j = 0; j<width; j++) {
                        const gresult_t a1 = A[base + j];
                        const gresult_t a2 = gcprod_(f_sqrt, A[base + width + j]);

                        const gsumdiff_t sd1 = gsumdiff_(&a1, &a2);
                        A[base + j] = sd1.sum;
                        A[base + width + j] = sd1.dif;

                        const gresult_t b1 = B[base + j];
                        const gresult_t b2 = gcprod_(f_sqrt, B[base + width + j]);

                        const gsumdiff_t sd2 = gsumdiff_(&b1, &b2);
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
                    gresult_t acc = ZERO_G;
                    for (int_t l=i+1; l<width; l++) gadd_(&acc, gprod_(&A[base + width + i - l], &B[base + l]));
                    acc = gcprod_(F[base], acc);
                    for (int_t l=0; l<=i; l++) gadd_(&acc, gprod_(&A[base + i - l], &B[base + l]));
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
                        const gresult_t m1 = C[base + j];
                        const gresult_t m2 = C[base + width2 + j];

                        const gsumdiff_t sd = gsumdiff_(&m1, &m2);
                        C[base + j] = gcprod_(0.5, sd.sum);
                        C[base + width2 + j] = gcprod_(half_inv_f, sd.dif);
                    }

                    F[base] = f_sqrt*f_sqrt;
                }
                blocks >>= 1;
            }

            for (int_t i=N+1; i<M; i++) C[i] = ZERO_G;
            gresult_t* restrict temp = A;
            A = C;
            C = temp;
        }

#ifdef USE_SIMD
        for (int s=0; s<STRIDE; s++) {
            const real_t resN = A[N].log_mod[s] - l_prev[s]*(real_t)N
                               + LOG(cabs(A[N].unit_re[s] + I*A[N].unit_im[s])) + lgac[N];
            const real_t s2 = M_PI*gamma[s]/mult->T;
            real_t sinc = LOG((1-EXP(-s2))/s2);
            if (s2 < -MAX_EXP) sinc = -s2 - LOG(-s2);
            if (ABS(s2)<=EPS) sinc = -s2*(0.5 - s2/24);
            res_arr[idx_vec*STRIDE + s] = resN + sinc;
        }
#else
        const real_t resN = A[N].log_mod - l_prev*(real_t)N
                           + LOG(cabs(A[N].unit_re + I*A[N].unit_im)) + lgac[N];
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
