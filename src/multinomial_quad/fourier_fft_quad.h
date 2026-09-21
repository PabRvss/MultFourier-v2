#pragma once

typedef struct {
    __float128 log_mod;
    __complex128 unit;
} qresult_t;

typedef struct {
    qresult_t sum;
    qresult_t dif;
} qsum_diff_t;

static inline void qadd_inplace(qresult_t* restrict a, const qresult_t b) {
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

static inline qsum_diff_t qsum_diff(const qresult_t* restrict a, const qresult_t* restrict b) {
    qsum_diff_t res;
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

static inline qresult_t qproduct(const qresult_t* restrict a, const qresult_t* restrict b) {
    const qresult_t res = {
        .log_mod = a->log_mod + b->log_mod,
        .unit = a->unit * b->unit,
    };
    return res;
}

static inline qresult_t qcproduct(const __complex128 a, const qresult_t b) {
    const qresult_t res = {
        .log_mod = b.log_mod,
        .unit = a * b.unit,
    };
    return res;
}

static void* eval_fourier_fft_quad_thread(void* args_void) {
    arguments_t* args = args_void;

    const multinomial* restrict mult = args->mult;
    const cell_t* restrict global = mult->matrix;
    const qresult_t ZERO = {.log_mod = -HUGE_VALQ, .unit = 1.0q};

    const int_t N = mult->N;
    const int_t K = mult->K;

    int_t M = 1;
    while (M<N+1) M<<=1;
    M<<=1;

    const real_t nu_sp = mult->l_rate;
    real_t* restrict l_rate = malloc(K*sizeof(real_t));
    for (int_t k=0; k<K; k++) l_rate[k] = nu_sp;

    qresult_t* restrict A = malloc(M*sizeof(qresult_t));
    qresult_t* restrict B = malloc(M*sizeof(qresult_t));
    qresult_t* restrict C = malloc(M*sizeof(qresult_t));
    __complex128* restrict F = malloc(M*sizeof(__complex128));

    complex_t* restrict res_arr = args->res_arr;
    const __float128 s2_re = M_PIq*(__float128)mult->gamma/(__float128)mult->T;

    for (int_t n_outer=args->start; n_outer<args->end; n_outer++) {
#ifdef USE_SIMD
    for (int_t s=0; s<STRIDE; s++) {
        const int_t n = n_outer*STRIDE + s;
#else
    for (int_t s=0; s<1; s++) {
        const int_t n = n_outer;
#endif
        for (int_t i=0; i<M; i++) A[i] = ZERO;
        A[0].log_mod = 0;

        real_t l_prev = 0;
        for (int_t k=K-1; k>=0; k--) {  // reversed
            const int_t index_k = k*(N+1);
            const real_t l_k = l_rate[k];

            const real_t d_k = l_k - l_prev;
            for (int_t j=0; j<M; j++) A[j].log_mod += (__float128)d_k * (__float128)j;
            l_prev = l_k;

            // Update B for k, scaled by this category's own l_k alone.
            for (int_t i=0; i<=N; i++) {
                const int_t index = index_k + i;
                const real_t rewardT = global[index].reward;
                const real_t arg = (real_t)n*rewardT;
                B[i].log_mod = (__float128)global[index].log_prob + (__float128)l_k*(__float128)i;
                B[i].unit = cosq((__float128)arg) - I*sinq((__float128)arg);  // conjugated
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
                    const __complex128 f_sqrt = csqrtq(F[base]);

                    for (int_t j = 0; j<width; j++) {
                        const qresult_t a1 = A[base + j];
                        const qresult_t a2 = qcproduct(f_sqrt, A[base + width + j]);

                        const qsum_diff_t sd1 = qsum_diff(&a1, &a2);
                        A[base + j] = sd1.sum;
                        A[base + width + j] = sd1.dif;

                        const qresult_t b1 = B[base + j];
                        const qresult_t b2 = qcproduct(f_sqrt, B[base + width + j]);

                        const qsum_diff_t sd2 = qsum_diff(&b1, &b2);
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
                    qresult_t acc = ZERO;
                    for (int_t l=i+1; l<width; l++) {
                        const qresult_t p = qproduct(&A[base + width + i - l], &B[base + l]);
                        qadd_inplace(&acc, p);
                    }
                    acc = qcproduct(F[base], acc);
                    for (int_t l=0; l<=i; l++) {
                        const qresult_t p = qproduct(&A[base + i - l], &B[base + l]);
                        qadd_inplace(&acc, p);
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
                        const qresult_t m1 = C[base + j];
                        const qresult_t m2 = C[base + width2 + j];

                        const qsum_diff_t sd = qsum_diff(&m1, &m2);
                        C[base + j] = qcproduct(0.5q, sd.sum);
                        C[base + width2 + j] = qcproduct(half_inv_f, sd.dif);
                    }

                    F[base] = f_sqrt*f_sqrt;
                }
                blocks >>= 1;
            }

            for (int_t i=N+1; i<M; i++) C[i] = ZERO;
            qresult_t* restrict temp = A;
            A = C;
            C = temp;
        }

        const __float128 resN_log = A[N].log_mod - (__float128)l_prev*(__float128)N + (__float128)lgac[N];
        const __float128 s2_im = (__float128)n*M_PIq;
        const __complex128 s2 = s2_re + I*s2_im;
        __complex128 sinc = (1.0q - cexpq(-s2))/s2;
        if ((n == 0) && (fabsq(s2_re)<=(__float128)EPS)) sinc = 1.0q - s2*(0.5q - s2/6.0q);

        const __complex128 res_q = expq(resN_log)*(A[N].unit*sinc);
        res_arr[n] = (real_t)crealq(res_q) + I*(real_t)cimagq(res_q);
    }
    }

    free(A);
    free(B);
    free(C);
    free(F);
    free(l_rate);

    return 0;
}
