#pragma once

typedef struct {
    vec_t cos;
    vec_t sin;
} local_t;

typedef struct {
    real_t log_mod;
    vec_t unit_re;
    vec_t unit_im;
} result_t;

typedef struct {
    const multinomial* mult;
    local_t* restrict local;
    result_t* restrict vec_arr;
    qcomplex_t* restrict res_arr;
    const void* restrict plan;
    int_t thread;
    int_t start;
    int_t end;
} arguments_t;

#include "error_bound.h"
#include "fourier_fft.h"
#ifdef ENABLE_QUADMATH
#include "../multinomial_quad/fourier_fft_quad.h"
#endif
#include "fourier_fft_precompute.h"
#ifdef ENABLE_QUADMATH
#include "../multinomial_quad/fourier_fft_quad_precompute.h"
#endif
#include "../multinomial_dd/fourier_fft_dd.h"
#include "../multinomial_dd/fourier_fft_dd_precompute.h"

static void* eval_fourier_thread(void* args_void) {
    arguments_t* args = args_void;
    const multinomial* restrict mult = args->mult;
    const cell_t* restrict global = mult->matrix;
    local_t* restrict local = args->local;
    result_t* restrict vec_arr = args->vec_arr;
    qcomplex_t* restrict res_arr = args->res_arr;

    const int_t N = mult->N;
    const int_t K = mult->K;

    for (int_t n=args->start; n<args->end; n++) {
        for (int_t i=0; i<=N; i++) {
            vec_arr[i].log_mod = -INFINITY;
            vec_arr[i].unit_re = VEC_ONE;
            vec_arr[i].unit_im = VEC_ZERO;
        }
        vec_arr[0].log_mod = 0;

        for (int_t k=K-1; k>=0; k--) {
            const int_t index_k = k*(N+1);
            for (int_t i=0; i<=N; i++) {
                const int_t index = index_k + i;
                const real_t rewardT = global[index].reward;
#ifdef USE_SIMD
                for (int s=0; s<STRIDE; s++) {
                    const real_t arg = (n*STRIDE + s)*rewardT;
                    local[i].cos[s] = cos(arg);
                    local[i].sin[s] = sin(arg);
                }
#else
                const real_t arg = n*rewardT;
                local[i].cos = cos(arg);
                local[i].sin = sin(arg);
#endif
            }

            for (int_t i=N; i>=0; i--) {
                const real_t shift_ki = -global[index_k + i].shift;
                vec_t res_re = VEC_ZERO;
                vec_t res_im = VEC_ZERO;

                for (int_t j=0; j<=i; j++) {
                    const int_t ij = i-j;
                    const vec_t term_unit_re = local[ij].cos;
                    const vec_t term_unit_im = local[ij].sin;
                    const result_t vecj = vec_arr[j];
                    const vec_t z_re = term_unit_re*vecj.unit_re + term_unit_im*vecj.unit_im;
                    const vec_t z_im = term_unit_re*vecj.unit_im - term_unit_im*vecj.unit_re;
                    const real_t exponent_re = global[index_k + ij].log_prob + shift_ki + vecj.log_mod;
                    const real_t term_mod = EXP(exponent_re);

                    res_re += term_mod*z_re;
                    res_im += term_mod*z_im;
                }
                vec_arr[i].log_mod = -shift_ki;
                vec_arr[i].unit_re = res_re;
                vec_arr[i].unit_im = res_im;
            }
        }

        const real_t s2_re = M_PI*mult->gamma/mult->T;
        const quad_t resN_exp = exp_quad(vec_arr[N].log_mod + lgac[N]);
#ifdef USE_SIMD
        vec_t s2_im; for (int_t i=0; i<STRIDE; i++) s2_im[i] = (n*STRIDE + i)*M_PI;
        for (int_t i=0; i<STRIDE; i++) {
            const complex_t unit = vec_arr[N].unit_re[i] + I*vec_arr[N].unit_im[i];
            const complex_t s2 = s2_re + I*s2_im[i];
            complex_t sinc = (1-EXPC(-s2))/s2;
            if ((n*STRIDE+i == 0) && (ABS(s2_re)<=EPS)) sinc = 1 - s2*(0.5 - s2/6);
            res_arr[n*STRIDE+i] = resN_exp*(unit*sinc);
        }
#else
        const complex_t s2 = s2_re + I*n*M_PI;
        complex_t sinc = (1-EXPC(-s2))/s2;
        if ((n == 0) && (ABS(s2_re)<=EPS)) sinc = 1 - s2*(0.5 - s2/6);
        const complex_t unit = vec_arr[N].unit_re + I*vec_arr[N].unit_im;
        res_arr[n] = resN_exp*(unit*sinc);
#endif
    }
    return 0;
}

static void* (*select_fourier_eval_fn(const options_t* restrict options))(void*) {
    switch (options->fft_precision) {
        case FFT_NONE: return eval_fourier_thread;
        case FFT_REAL: return options->use_fft_precompute ? eval_fourier_fft_precompute_thread : eval_fourier_fft_sp_thread;
        case FFT_DD:   return options->use_fft_precompute ? eval_fourier_fft_dd_precompute_thread : eval_fourier_fft_dd_thread;
#ifdef ENABLE_QUADMATH
        case FFT_QUAD: return options->use_fft_precompute ? eval_fourier_fft_quad_precompute_thread : eval_fourier_fft_quad_thread;
#endif
        default: return eval_fourier_thread;
    }
}

static void eval_fourier_parallel(
        const multinomial* restrict mult, const int_t n_min, const int_t n_max,
        local_t* restrict local_arrs, result_t* restrict vec_arrs,
        qcomplex_t* res_arr, options_t* options, const void* restrict plan
) {
    const int_t threads = options->threads;
    const int_t N = mult->N;
    pthread_t* tid = (pthread_t*)malloc(threads*sizeof(pthread_t));
    arguments_t* args_array = malloc(threads*sizeof(arguments_t));

    void* (*eval_fn)(void*) = select_fourier_eval_fn(options);

    int_t step = (n_max-n_min+threads-1)/threads;
    for (int_t i=0; i<threads; i++) {
        int_t start = n_min + i*step;
        int_t end = start + step;
        if (end > n_max) end = n_max;

        args_array[i].mult = mult;
        args_array[i].local = &local_arrs[i*(N+1)];
        args_array[i].vec_arr = &vec_arrs[i*(N+1)];
        args_array[i].res_arr = res_arr;
        args_array[i].plan = plan;
        args_array[i].thread = i;
        args_array[i].start = start;
        args_array[i].end = end;
        pthread_create(&tid[i], NULL, eval_fn, &args_array[i]);
    }
    for (size_t i=0; i<threads; i++) pthread_join(tid[i], NULL);

    free(tid);
    free(args_array);
}

static void eval_fourier_sequential(
        const multinomial* restrict mult, const int_t n_min, const int_t n_max,
        local_t* restrict local_arrs, result_t* restrict vec_arrs,
        qcomplex_t* res_arr, options_t* options, const void* restrict plan
) {
    arguments_t args;
    args.mult = mult;
    args.local = local_arrs;
    args.vec_arr = vec_arrs;
    args.res_arr = res_arr;
    args.plan = plan;
    args.thread = 0;
    args.start = n_min;
    args.end = n_max;

    select_fourier_eval_fn(options)(&args);
}

static void series(multinomial* mult, multinomial_result* mult_res, options_t* options) {
    const int_t threads = options->threads;
    const int_t vecs_per_thread = options->vecs_per_thread;
    const int_t N = mult->N;

    const int_t step = threads*vecs_per_thread*STRIDE;
    const int_t max_chunks = (options->max_iter+step-1)/step;

    local_t* local_arrs = malloc(threads*(N+1)*sizeof(local_t));
    result_t* vec_arrs = malloc(threads*(N+1)*sizeof(result_t));
    qcomplex_t* res_arr = malloc(max_chunks*step*sizeof(qcomplex_t));
    quad_t* sums_arr = malloc(max_chunks*step*sizeof(quad_t));

    void* plan = NULL;
    if (options->use_fft_precompute) {
        switch (options->fft_precision) {
            case FFT_REAL: plan = build_fourier_fft_plan(mult); break;
            case FFT_DD:   plan = build_fourier_fft_dd_plan(mult); break;
#ifdef ENABLE_QUADMATH
            case FFT_QUAD: plan = build_fourier_fft_quad_plan(mult); break;
#endif
            case FFT_NONE: default: break;
        }
    }

    const int_t B = options->B;
    const quad_t threshold = (B > 0) ? (options->eps_rel / (quad_t)B) : 0.0;
    quad_t p = 0;
    quad_t c = 0;
    quad_t f = 0;
    int converged_full = 0;
    int_t converged_count = 0;
    quad_t p_prev = INFINITY;

    err_diag_t diag;
    err_diag_init(&diag, max_chunks*step);
    int_t iter = 0;

    struct timespec t_series_start;
    timespec_get(&t_series_start, TIME_UTC);

    for (int_t n=0; n < max_chunks; n++) {
        /* Check max_time after each chunk */
        if (options->max_time > 0.0 && isfinite(options->max_time)) {
            struct timespec t_now;
            timespec_get(&t_now, TIME_UTC);
            if (get_dt(t_series_start, t_now) >= options->max_time) {
                mult_res->status = 1; // Maximum time reached
                goto finish;
            }
        }

        /* Check for numerical explosion */
        if (!finiteq_quad(p) || isnanq_quad(p) || fabsq_quad(p) > 1e100) {
            p = NAN;
            mult_res->status = 4; // Magnitude explosion / numerical instability
            converged_full = 0;
            goto finish;
        }

        const int_t n_min = n*threads*vecs_per_thread;
        const int_t n_max = (n+1)*threads*vecs_per_thread;
        if (threads<=1) eval_fourier_sequential(mult, n_min, n_max, local_arrs, vec_arrs, res_arr, options, plan);
        else eval_fourier_parallel(mult, n_min, n_max, local_arrs, vec_arrs, res_arr, options, plan);

        if (options->error_bound) err_diag_accum(&diag, res_arr, n_min*STRIDE, n_max*STRIDE, mult->gamma, mult->T);

        if (n==0) {
            if (res_arr[0] == 0.0) {
                p = 0;
                sums_arr[0] = 0;
                converged_full = 1;
                mult_res->status = 0; // Converged
                goto finish;
            }
            res_arr[0] /= 2;
        }

        for (iter=n_min*STRIDE; (iter<n_max*STRIDE) && (iter<options->max_iter); iter++) {
            const quad_t delta_re = crealq_quad(res_arr[iter]);
            if (!finiteq_quad(delta_re) || isnanq_quad(delta_re) || fabsq_quad(delta_re) > 1e100) {
                p = NAN;
                mult_res->status = 4; // Magnitude explosion
                converged_full = 0;
                goto finish;
            }

            const quad_t y = delta_re - c;
            const quad_t t = f + y;
            c = (t - f) - y;
            f = t;
            p = f;
            sums_arr[iter] = p;

            const quad_t frac = fabsq_quad(delta_re/p_prev);
            const int converged = (B > 0) && (p > 0.0) && (frac <= threshold);
            converged_count = converged ? converged_count + 1 : 0;
            p_prev = p;

            if (B > 0 && converged_count >= B) {
                converged_full = 1;
                mult_res->status = 0; // Converged
                goto finish;
            }
        }
    }

    finish:
    /* Windowed series averaging to dampen Gibbs oscillations */
    if (iter > 0 && options->average_window > 0.0) {
        quad_t p_average = 0;
        double weight = 0.0;
        const int_t window = (int_t)(options->average_window * iter);
        for (int_t i=0; i<=window; i++) {
            const int_t j = iter - window + i;
            if (j >= 0) {
                const double w = options->average_flat ? 1.0 : (double)(i + 1);
                p_average += sums_arr[j] * w;
                weight += w;
            }
        }
        if (weight > 0.0) {
            p_average /= weight;
            p = p_average;
        }
    }

    p += mult->p0/2;

    mult_res->pval = (double)p;
    mult_res->converged = converged_full;
    mult_res->terms = iter;

    if (!converged_full && mult_res->status == 0) {
        mult_res->status = 2; // Maximum number of terms reached
    }

    mult_res->err_rel_est = NAN;
    mult_res->n_eff = NAN;
    mult_res->nu_max_ratio = NAN;
    mult_res->nu_at_max = -1;
    if (options->error_bound) {
        int_t n_at = -1;
        mult_res->err_rel_est = err_rel_estimate(&diag, (double)p, mult->gamma, mult->T);
        mult_res->n_eff = err_n_eff(&diag);
        mult_res->nu_max_ratio = err_max_revival(&diag, &n_at);
        mult_res->nu_at_max = n_at;
    }
    err_diag_free(&diag);

    if (plan) {
        switch (options->fft_precision) {
            case FFT_REAL: free_fourier_fft_plan(plan); break;
            case FFT_DD:   free_fourier_fft_dd_plan(plan); break;
#ifdef ENABLE_QUADMATH
            case FFT_QUAD: free_fourier_fft_quad_plan(plan); break;
#endif
            case FFT_NONE: default: break;
        }
    }

    free(local_arrs);
    free(vec_arrs);
    free(res_arr);
    free(sums_arr);
}