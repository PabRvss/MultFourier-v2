#include "base.h"
#pragma once

/*
 * Numero de puntos de evaluacion en cada iteracion de la busqueda de
 * gamma (fill_gamma). Este valor es un parametro puramente numerico del
 * algoritmo de line-search y es intencionalmente independiente de
 * options->threads (que ahora es vestigial, ya que la ejecucion es
 * secuencial). Desacoplar ambos conceptos evita la division por cero
 * 1/(1-1) que ocurria cuando threads=1: con threads=1 no hay suficientes
 * puntos para contraer el intervalo [gamma1, gamma2] de forma simetrica.
 */
#define N_GAMMA_POINTS 8

typedef struct {
    real_t log_prob;  // term i-j
    real_t reward;
    real_t shift;  // exponent shift i
} cell_t;

typedef struct {
    // Basic data
    int_t N;  // trials
    int_t K;  // categories
    const real_t* restrict probs;  // probabilities (size K)

    // Calculated data
    real_t* restrict log_probs;  // log-probabilities (size K)
    cell_t* restrict matrix;  // cells (size K*(N+1))

    double p0;  // realization probability
    real_t r0;  // realization reward value

    real_t T;  // sampling interval
    real_t gamma;  // gamma

    int_t eval_count;

} multinomial;

typedef struct {
    double pval;
    int_t converged;
    int_t terms;
    double p0;
} multinomial_result;

static multinomial* get_mult(const int_t N, const int_t K, const real_t* restrict probs, options_t* options) {
    multinomial* mult = malloc(sizeof(multinomial));
    mult->K = K;
    mult->N = N;
    mult->probs = probs;

    mult->log_probs = malloc(K*sizeof(real_t));
    for (int_t k=0; k<K; k++) {
        real_t logp = LOG(mult->probs[k]);
        mult->log_probs[k] = mult->probs[k] > 0 ? logp : 0;
    }

    size_t matrix_size = K*(N+1);
    size_t mem = matrix_size*sizeof(cell_t);
    mult->matrix = malloc(mem);

    size_t div = (1 << 20);
    double mb = (double)mem/div;
    if (options->verbose) printf("Matrix size = %llu bytes (%0.3f MB)\n", (unsigned long long)mem, (double)mb);

    mult->eval_count = 0;

    return mult;
}

static void free_mult(multinomial* mult) {
    free(mult->log_probs);
    free(mult->matrix);
    free(mult);
}

static void fill_p0(multinomial* mult, const int_t* restrict x0) {
    int_t K = mult->K;
    int_t N = mult->N;

    real_t log_p0 = 0;
    for (int_t k=0; k<K; k++) log_p0 += reward_logp(N, mult->log_probs[k], x0[k]);
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

static void fill_rewards(multinomial* mult, const int_t* restrict x0) {
    const int_t K = mult->K;
    const int_t N = mult->N;

    real_t r0 = 0;
    int_t index = 0;
    for (int_t k=0; k < K; k++) {
        const real_t rk = REWARD(N, mult->log_probs[k], x0[k]);
        r0 += rk;
        for (int_t i=0; i<=N; i++) {
            const real_t reward = REWARD(N, mult->log_probs[k], i);
            mult->matrix[index++].reward = reward-rk;
        }
    }
    mult->r0 = r0;
}

static void fill_interval(multinomial* mult, options_t* options) {
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

    for (int_t k=K-1; k>=0; k--) {  // reversed
        for (int_t i=N; i>=0; i--) {  // reversed
            int_t index_k = k*(N+1);

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

    const real_t W = s_max - s_min;
    const real_t W_max = MAX(s_max, -s_min);
    const real_t W_eff = W_max/options->undersampling;

    mult->T = M_PI/W_eff;
    if (W == 0) mult->T = 1;
    mult->T = mult->T;

    if (options->verbose) {
        printf("\tI in %.3f + [%.3f, %.3f]\n", mult->r0, s_min, s_max);
        printf("\tW = %.3f -> %.3f, T = %f\n", W, W_eff, mult->T);
    }

    free(arr_min);
    free(arr_max);
}

typedef struct {
    vec_t gamma;
    const multinomial* restrict mult;

    vec_t* restrict vec_arr;  // size N+1
    vec_t* restrict shifts;  // size K*(N+1)
    real_t* restrict res_arr;

    int_t thread;
} args_gamma_t;

/*
 * eval_gamma_thread contiene el kernel de calculo puro. Se mantiene el
 * nombre y la firma (void*) por compatibilidad con el resto del codigo,
 * pero ya no se invoca a traves de pthread_create: se llama directamente
 * como una funcion secuencial normal.
 */
static void* eval_gamma_thread(void* args_void) {
    args_gamma_t* args = (args_gamma_t*)args_void;

    const multinomial* restrict mult = args->mult;
    const cell_t* restrict global = mult->matrix;
    const int_t thread = args->thread;
    vec_t* restrict shifts = args->shifts;
    vec_t* restrict vec_arr = args->vec_arr;
    real_t* restrict res_arr = args->res_arr;

    const vec_t gamma = args->gamma;
    const int_t N = mult->N;
    const int_t K = mult->K;

    vec_arr[0] = VEC_ZERO;
    for (int_t i=1; i<=N; i++) vec_arr[i] = -INFINITY*VEC_ONE;

    for (int_t k=K-1; k>=0; k--) {  // reversed
        const int_t index_k = k*(N+1);
        for (int_t i=N; i>=0; i--) {  // reversed
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
                res += VEC_EXP(exponent);
            }

            vec_arr[i] = max_exp_i + VEC_LOG(res);
            shifts[index_k + i] = max_exp_i;
        }
    }

    const vec_t resN = vec_arr[N] + lgac[N];

    const real_t s2 = M_PI*gamma/mult->T;
    real_t sinc = LOG((1-EXP(-s2))/s2);
    if (s2 < -MAX_EXP) sinc = -s2 - LOG(-s2);
    if (ABS(s2)<=EPS) sinc = -s2*(0.5 - s2/24);
    res_arr[thread] = resN + sinc;

    return 0;
}

/*
 * eval_gamma_parallel: pese al nombre (conservado por compatibilidad con
 * fill_gamma), ejecuta ahora una ruta 100% secuencial. No hay creacion de
 * hilos: se invoca eval_gamma_thread() directamente para cada indice, en
 * el hilo actual. Esto elimina toda dependencia de pthread_create /
 * pthread_join / <pthread.h>, cumpliendo el requisito de CRAN de no
 * asumir soporte de threading manual.
 */
static void eval_gamma_parallel(
        const multinomial* restrict mult,
        const vec_t* restrict gammas,
        vec_t* restrict vec_arrs,
        vec_t* restrict shift_arrs,
        real_t* restrict res_arr,
        const int_t n_points
) {
    const int_t N = mult->N;
    const int_t K = mult->K;

    args_gamma_t args_i;
    for (int_t i=0; i<n_points; i++) {
        args_i.mult = mult;
        args_i.gamma = gammas[i];
        args_i.vec_arr = &vec_arrs[i*(N+1)];
        args_i.shifts = &shift_arrs[i*K*(N+1)];
        args_i.res_arr = res_arr;
        args_i.thread = i;

        eval_gamma_thread(&args_i);
    }
}

static void fill_gamma(multinomial* mult, options_t* options) {
    struct timespec start, end;
    timespec_get(&start, TIME_UTC);

    const int_t N = mult->N;
    const int_t K = mult->K;
    real_t final_eval = NAN;
    real_t gamma = NAN;

    /* n_points ya NO depende de options->threads */
    const int_t n_points = N_GAMMA_POINTS;

    vec_t* restrict gammas_vec = malloc(n_points*sizeof(vec_t));
    real_t* restrict gammas = malloc(n_points*sizeof(real_t));
    real_t* restrict res_arr = malloc(n_points*sizeof(real_t));
    vec_t* restrict vec_arrs = malloc(n_points*(N+1)*sizeof(vec_t));
    vec_t* restrict shift_arrs = malloc(n_points*K*(N+1)*sizeof(vec_t));

    real_t gamma1 = -2;
    real_t gamma2 = 2;

    if (N==0) goto finish;

    /* factor = n_points/(n_points - 1) = 8/7, bien definido siempre */
    const real_t factor = (real_t)n_points/((real_t)n_points - 1);

    int_t j_min = -1;
    while (gamma2 - gamma1 > options->eps_gamma) {
        real_t delta = factor*(gamma2 - gamma1)/n_points;
        for (int_t i=0; i<n_points; i++) {
            gammas_vec[i] = gamma1 + (real_t)i*delta;
            gammas[i] = gammas_vec[i];
        }

        eval_gamma_parallel(mult, gammas_vec, vec_arrs, shift_arrs, res_arr, n_points);
        mult->eval_count += n_points;

        j_min = -1;
        real_t val_min = INFINITY;
        for (int_t j=0; j<n_points; j++) {
            if (res_arr[j] < val_min) {
                j_min = j;
                val_min = res_arr[j];
            }
        }
        gamma = gammas[j_min];
        gamma1 = gamma - delta;
        gamma2 = gamma + delta;
        final_eval = val_min;
    }

    vec_t* shifts_min = &shift_arrs[j_min*K*(N+1)];
    for (int_t k=0; k<K; k++) {
        const int_t index_k = k*(N+1);
        for (int_t i=0; i<=N; i++)
            mult->matrix[index_k + i].shift = shifts_min[index_k + i];
    }

    finish:
    mult->gamma = gamma;

    free(gammas);
    free(gammas_vec);
    free(res_arr);
    free(vec_arrs);
    free(shift_arrs);

    timespec_get(&end, TIME_UTC);
    double dt = get_dt(start, end);

    if (options->verbose) {
        printf("\tgamma = %.6f, log eval(-gamma) = %.6f, PI*gamma/T = %.6e\n", mult->gamma, final_eval, M_PI*mult->gamma/mult->T);
        printf("\t%lld evaluations in %f seconds\n", mult->eval_count, dt);
    }
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

static void fill_mult1(multinomial* mult, const int_t* restrict x0, options_t* options) {
  const int_t verbose = options->verbose;

  if (verbose) printf("fill p0");
  fill_p0(mult, x0);
  if (verbose) printf(" = %.8e\n", mult->p0);

  if (verbose) printf("fill probabilities\n");
  fill_probs(mult);

  if (verbose) printf("fill rewards");
  fill_rewards(mult, x0);
  if (verbose) printf(", r0 = %.8f\n", mult->r0);
}


static void fill_mult2(multinomial* mult, const int_t* restrict x0, options_t* options) {
    const int_t verbose = options->verbose;

    if (verbose) printf("fill interval\n");
    fill_interval(mult, options);

    if (verbose) printf("optimize gamma\n");
    fill_gamma(mult, options);

    if (verbose) printf("transform cells\n");
    fill_cells(mult);
}