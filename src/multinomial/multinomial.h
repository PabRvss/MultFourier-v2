#pragma once

typedef struct {
    real_t log_prob;
    real_t reward;
    real_t shift;
} cell_t;

typedef struct {
    int_t N;
    int_t K;
    const real_t* restrict probs;
    real_t* restrict log_probs;
    cell_t* restrict matrix;
    double p0;
    real_t r0;
    real_t T;
    real_t W;
    real_t gamma;
    real_t l_rate;
} multinomial;

typedef struct {
    double pval;
    int_t converged;
    int_t terms;
    double p0;
    double eval_time;
    double total_time;
    double err_rel_est;
    double n_eff;
    double nu_max_ratio;
    int_t nu_at_max;
    double gamma;
    double W;
    int_t status;
    int_t method;
} multinomial_result;

static multinomial* get_mult(const int_t N, const int_t K, const real_t* restrict probs, const options_t* options) {
    multinomial* mult = malloc(sizeof(multinomial));
    mult->K = K;
    mult->N = N;
    mult->probs = probs;

    mult->log_probs = malloc(K*sizeof(real_t));
    for (int_t k=0; k<K; k++) {
        real_t logp = LOG(mult->probs[k]);
        mult->log_probs[k] = mult->probs[k] > 0 ? logp : 0;
    }

    const size_t matrix_size = K*(N+1);
    const size_t mem = matrix_size*sizeof(cell_t);
    mult->matrix = malloc(mem);

    return mult;
}

static void free_mult(multinomial* mult) {
    free(mult->log_probs);
    free(mult->matrix);
    free(mult);
}

static void fill_p0(multinomial* mult, const int_t* restrict x0, const options_t* options) {
    int_t K = mult->K;
    int_t N = mult->N;
    real_t log_p0 = 0;
    for (int_t k=0; k<K; k++) log_p0 += reward_logp(mult->log_probs[k], x0[k]);
    mult->p0 = EXP(lgac[N] + (double)log_p0);
}

static void fill_probs(multinomial* mult) {
    int_t K = mult->K;
    int_t N = mult->N;
    int_t index = 0;
    for (int_t k=0; k<K; k++) {
        const real_t pk = mult->probs[k];
        for (int_t i=0; i<=N; i++) {
            cell_t cell;
            const real_t t = i==0? 0:i*LOG(pk);
            cell.log_prob = -lgac[i] + t;
            mult->matrix[index++] = cell;
        }
    }
}

static void fill_rewards(multinomial* mult, const int_t* restrict x0, const options_t* options) {
    const int_t K = mult->K;
    const int_t N = mult->N;
    real_t r0 = 0;
    int_t index = 0;
    for (int_t k=0; k < K; k++) {
        const real_t rk = reward_power(N, k, mult->log_probs[k], x0[k], options->lambda);
        r0 += rk;
        for (int_t i=0; i<=N; i++) {
            const real_t reward = i == x0[k] ? rk : reward_power(N, k, mult->log_probs[k], i, options->lambda);
            mult->matrix[index++].reward = reward-rk;
        }
    }
    mult->r0 = r0;
}

static void fill_interval(multinomial* mult, const options_t* options) {
    const int_t K = mult->K;
    const int_t N = mult->N;

    real_t* arr_min = malloc((N+1)*sizeof(real_t));
    real_t* arr_max = malloc((N+1)*sizeof(real_t));

    for (int_t n=0; n<=N; n++) {
        arr_min[n] = INFINITY;
        arr_max[n] = -INFINITY;
    }
    arr_min[0] = 0;
    arr_max[0] = 0;

    for (int_t k=K-1; k>=0; k--) {
        for (int_t i=N; i>=0; i--) {
            const int_t index_k = k*(N+1);
            real_t new_min = INFINITY;
            real_t new_max = -INFINITY;
            for (int_t j=0; j<=i; j++) {
                real_t reward = mult->matrix[index_k + i-j].reward;
                real_t total_min = arr_min[j] + reward;
                real_t total_max = arr_max[j] + reward;
                new_min = MIN(new_min, total_min);
                new_max = MAX(new_max, total_max);
            }
            arr_min[i] = new_min;
            arr_max[i] = new_max;
        }
    }

    const real_t s_min = arr_min[N];
    const real_t s_max = arr_max[N];
    const real_t width = s_max - s_min;
    const real_t W_max = MAX(s_max, -s_min);
    const real_t W_eff = W_max/options->undersampling;

    mult->W = W_eff;
    mult->T = W_eff == 0? 1.0 : M_PI/W_eff;
    free(arr_min);
    free(arr_max);
}

typedef struct {
    const vec_t* restrict gammas_vec;
    const multinomial* restrict mult;
    vec_t* restrict vec_arr;
    vec_t* restrict shifts_arr;
    real_t* restrict res_arr;
    vec_t* restrict l_rate_arr;
    int_t thread;
    int_t vecs_per_thread;
} args_gamma_t;

#include "multinomial_fft.h"
#ifdef ENABLE_QUADMATH
#include "../multinomial_quad/multinomial_fft_quad.h"
#endif
#include "../multinomial_dd/multinomial_fft_dd.h"

static void* eval_gammas_thread(void* args_void) {
    args_gamma_t* args = args_void;
    const multinomial* restrict mult = args->mult;
    const cell_t* restrict global = mult->matrix;
    const int_t thread = args->thread;
    const int_t vecs_per_thread = args->vecs_per_thread;
    vec_t* restrict shifts_arr = args->shifts_arr;
    vec_t* restrict vec_arr = args->vec_arr;
    real_t* restrict res_arr = args->res_arr;
    const vec_t* restrict gammas_vec = args->gammas_vec;

    const int_t N = mult->N;
    const int_t K = mult->K;

    for (int_t i_vec=0; i_vec<vecs_per_thread; i_vec++) {
        const int_t idx_vec = vecs_per_thread*thread + i_vec;
        const vec_t gamma = gammas_vec[idx_vec];

        vec_arr[0] = VEC_ZERO;
        for (int_t i=1; i<=N; i++) vec_arr[i] = -INFINITY*VEC_ONE;

        for (int_t k=K-1; k>=0; k--) {
            const int_t index_k = k*(N+1);
            for (int_t i=N; i>=0; i--) {
                vec_t max_exp_i = -INFINITY*VEC_ONE;
                for (int_t j=0; j<=i; j++) {
                    const cell_t cell_ij = global[index_k + i-j];
                    const real_t reward = cell_ij.reward;
                    const real_t log_prob = cell_ij.log_prob;
                    const vec_t exponent = -reward*gamma + log_prob + vec_arr[j];
                    max_exp_i = VEC_MAX(exponent, max_exp_i);
                }

                vec_t res = VEC_ZERO;
                for (int_t j=0; j<=i; j++) {
                    const cell_t cell_ij = global[index_k + i-j];
                    const real_t reward = cell_ij.reward;
                    const real_t log_prob = cell_ij.log_prob;
                    const vec_t exponent = -reward*gamma + log_prob + vec_arr[j] - max_exp_i;
#ifdef USE_SIMD
                    for (int s=0; s<STRIDE; s++) res[s] += EXP(exponent[s]);
                }
                vec_arr[i] = max_exp_i; for (int s=0; s<STRIDE; s++) vec_arr[i][s] += LOG(res[s]);
#else
                    res += EXP(exponent);
                }
                vec_arr[i] = max_exp_i + LOG(res);
#endif
                shifts_arr[K*(N+1)*idx_vec + index_k + i] = max_exp_i;
            }
        }

        const vec_t resN = vec_arr[N] + lgac[N];

#ifdef USE_SIMD
        for (int s=0;s<STRIDE; s++) {
            const real_t s2 = M_PI*gamma[s]/mult->T;
            real_t sinc = LOG((1-EXP(-s2))/s2);
            if (s2 < -MAX_EXP) sinc = -s2 - LOG(-s2);
            if (ABS(s2)<=EPS) sinc = -s2*(0.5 - s2/24);
            res_arr[idx_vec*STRIDE + s] = resN[s] + sinc;
        }
#else
        const real_t s2 = M_PI*gamma/mult->T;
        real_t sinc = LOG((1-EXP(-s2))/s2);
        if (s2 < -MAX_EXP) sinc = -s2 - LOG(-s2);
        if (ABS(s2)<=EPS) sinc = -s2*(0.5 - s2/24);
        res_arr[idx_vec] = resN + sinc;
#endif
    }
    return 0;
}

static void eval_gammas(
        const multinomial* restrict mult,
        const vec_t* restrict gammas_vec,
        vec_t* restrict vec_arrs,
        vec_t* restrict shift_arrs,
        real_t* restrict res_arr,
        vec_t* restrict l_rate_arr,
        const options_t* options,
        const int_t vecs_per_thread
) {
    const int_t N = mult->N;
    const int_t threads = options->threads;
    pthread_t* tid = threads > 1 ? malloc(threads*sizeof(pthread_t)): NULL;
    args_gamma_t* args_array = malloc(threads*sizeof(args_gamma_t));

    for (int_t i=0; i<threads; i++) {
        args_gamma_t* args_i = &args_array[i];
        args_i->mult = mult;
        args_i->gammas_vec = gammas_vec;
        args_i->vec_arr = &vec_arrs[i*(N+1)];
        args_i->shifts_arr = shift_arrs;
        args_i->res_arr = res_arr;
        args_i->l_rate_arr = l_rate_arr;
        args_i->thread = i;
        args_i->vecs_per_thread = vecs_per_thread;
    }
    fft_precision_t gamma_precision = options->fft_precision == FFT_NONE ? FFT_NONE : options->gamma_precision;

    void* (*eval_fn)(void*);
    switch (gamma_precision) {
        case FFT_REAL: eval_fn = eval_gammas_fft_thread; break;
        case FFT_DD:   eval_fn = eval_gammas_fft_dd_thread; break;
#ifdef ENABLE_QUADMATH
        case FFT_QUAD: eval_fn = eval_gammas_fft_quad_thread; break;
#endif
        case FFT_NONE: default: eval_fn = eval_gammas_thread; break;
    }
    for (int_t i=0; i<threads; i++) {
        if (threads > 1) {
            pthread_create(&tid[i], NULL, eval_fn, &args_array[i]);
        } else {
            eval_fn(&args_array[0]);
        }
    }
    if (threads > 1) for (int_t i=0; i<threads; i++) pthread_join(tid[i], NULL);

    if (threads > 1) free(tid);
    free(args_array);
}

static void fill_gamma(multinomial* mult, multinomial_result* mult_res, const options_t* options) {
    struct timespec start, end;
    timespec_get(&start, TIME_UTC);

    const int_t N = mult->N;
    const int_t K = mult->K;
    real_t final_eval = NAN;
    real_t gamma = NAN;

    const int_t threads = options->threads;
    int_t vecs_per_thread = options->vecs_per_thread;
    if (threads * vecs_per_thread * STRIDE < 8) {
        vecs_per_thread = (8 + threads * STRIDE - 1) / (threads * STRIDE);
    }
    const int_t vecs = threads*vecs_per_thread;
    const int_t n_gammas = vecs*STRIDE;

    const fft_precision_t gamma_precision = options->fft_precision == FFT_NONE ? FFT_NONE : options->gamma_precision;

    vec_t* restrict gammas_vec = malloc(vecs*sizeof(vec_t));
    real_t* restrict gammas = malloc(n_gammas*sizeof(real_t));
    vec_t* restrict vec_arrs = malloc(vecs*(N+1)*sizeof(vec_t));
    real_t* restrict res_arr = malloc(n_gammas*sizeof(real_t));
    vec_t* restrict shift_arrs = malloc(vecs*K*(N+1)*sizeof(vec_t));
    vec_t* restrict l_rate_arr = malloc(vecs*sizeof(vec_t));

    real_t gamma1 = -2;
    real_t gamma2 = 2;
    int_t eval_count = 0;

    if (N==0) goto finish;

#ifdef USE_SIMD
    vec_t step;
    for (int s=0; s<STRIDE; s++) step[s] = s;
#else
    real_t step = 0;
#endif

    int_t j_min = -1;
    while (gamma2 - gamma1 > options->eps_gamma) {
        const real_t delta = (n_gammas > 1) ? ((gamma2 - gamma1)/((real_t)n_gammas - 1)) : 0.0;
        
        for (int_t i=0; i<vecs; i++) {
            gammas_vec[i] = gamma1 + ((real_t)(STRIDE*i)+step)*delta;
#ifdef USE_SIMD
            for (int s=0; s<STRIDE; s++) gammas[i*STRIDE + s] = gammas_vec[i][s];
#else
            gammas[i] = gammas_vec[i];
#endif
        }

        eval_gammas(mult, gammas_vec, vec_arrs, shift_arrs, res_arr, l_rate_arr, options, vecs_per_thread);
        eval_count += n_gammas;

        j_min = -1;
        real_t val_min = INFINITY;
        for (int_t j=0; j<n_gammas; j++) {
            if (res_arr[j] < val_min && res_arr[j] > -INFINITY) {
                j_min = j;
                val_min = res_arr[j];
            }
        }
        if (j_min < 0) j_min = 0;
        gamma = gammas[j_min];
        gamma1 = gamma - delta;
        gamma2 = gamma + delta;
        final_eval = val_min;
    }

    const int_t i_min = j_min / STRIDE;
    const int_t s_min = j_min % STRIDE;

    mult->l_rate = 0;
    if (gamma_precision == FFT_NONE) {
        const vec_t* restrict shifts_min = &shift_arrs[i_min*K*(N+1)];
        for (int_t k=0; k<K; k++) {
            const int_t index_k = k*(N+1);
            for (int_t i=0; i<=N; i++) {
#ifdef USE_SIMD
                mult->matrix[index_k + i].shift = shifts_min[index_k + i][s_min];
#else
                mult->matrix[index_k + i].shift = shifts_min[index_k + i];
#endif
            }
        }
    } else {
#ifdef USE_SIMD
        mult->l_rate = l_rate_arr[i_min][s_min];
#else
        mult->l_rate = l_rate_arr[i_min];
#endif
    }

    finish:
    mult->gamma = gamma;
    if (isnan(gamma) || !isfinite(gamma)) {
        mult_res->status = 3; // Could not solve the optimization of gamma
    }
    if (N == 0) mult->l_rate = 0;

    free(gammas);
    free(gammas_vec);
    free(res_arr);
    free(vec_arrs);
    free(shift_arrs);
    free(l_rate_arr);

    timespec_get(&end, TIME_UTC);
    mult_res->eval_time = get_dt(start, end);
}

static void fill_cells(multinomial* mult) {
    const int_t K = mult->K;
    const int_t N = mult->N;
    const real_t T = mult->T;
    for (int_t k=0; k<K; k++) {
        const int_t index_k = k*(N+1);
        for (int_t i=0; i<=N; i++) {
            const int_t index = index_k + i;
            const real_t reward = mult->matrix[index].reward;
            mult->matrix[index].reward = reward*T;
            mult->matrix[index].log_prob = -mult->gamma*reward + mult->matrix[index].log_prob;
        }
    }
}

static void fill_mult1(multinomial* mult, multinomial_result* mult_res, const int_t* restrict x0, const options_t* options) {
    fill_p0(mult, x0, options);
    fill_probs(mult);
    fill_rewards(mult, x0, options);
}

static void fill_mult2(multinomial* mult, multinomial_result* mult_res, const int_t* restrict x0, const options_t* options) {
    fill_interval(mult, options);
    mult_res->W = mult->W;
    fill_gamma(mult, mult_res, options);
    fill_cells(mult);
}