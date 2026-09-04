#include "base.h"
#pragma once

typedef struct {  // data dependent on n
    vec_t cos;  // cosine n
    vec_t sin;  // sine n
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
    complex double* restrict res_arr;

    int_t thread;
    int_t start;
    int_t end;
} arguments_t;

/*
 * eval_fourier_thread contiene el kernel de calculo puro. Se conserva la
 * firma (void*) por compatibilidad, pero ya no se lanza via
 * pthread_create: se invoca directamente como funcion secuencial.
 */
static void* eval_fourier_thread(void* args_void) {
    arguments_t* args = (arguments_t*)args_void;

    const multinomial* restrict mult = args->mult;
    const cell_t* restrict global = mult->matrix;
    local_t* restrict local = args->local;
    result_t* restrict vec_arr = args->vec_arr;
    complex double* restrict res_arr = args->res_arr;

    const int_t N = mult->N;
    const int_t K = mult->K;

    for (int_t n=args->start; n<args->end; n++) {
        for (int_t i=0; i<=N; i++) {
            vec_arr[i].log_mod = -INFINITY;
            vec_arr[i].unit_re = VEC_ONE;
            vec_arr[i].unit_im = VEC_ZERO;
        }
        vec_arr[0].log_mod = 0;

        real_t term_mod;
        vec_t term_unit_re;
        vec_t term_unit_im;

        for (int_t k=K-1; k>=0; k--) {  // reversed
            const int_t index_k = k*(N+1);

            // Update sincos for k
            for (int_t i=0; i<=N; i++) {
                const int_t index = index_k + i;
                const real_t rewardT = global[index].reward;
                const real_t arg = n*rewardT;
                local[i].cos = cos(arg);
                local[i].sin = sin(arg);
            }

            for (int_t i=N; i>=0; i--) {  // reversed
                const real_t shift_ki = -global[index_k + i].shift;
                vec_t res_re = VEC_ZERO;
                vec_t res_im = VEC_ZERO;

                for (int_t j=0; j<=i; j++) {
                    const int_t ij = i-j;
                    const real_t exponent_ij = global[index_k + ij].log_prob;
                    term_unit_re = local[ij].cos;
                    term_unit_im = local[ij].sin;

                    const result_t vecj = vec_arr[j];
                    const vec_t z_re = term_unit_re*vecj.unit_re + term_unit_im*vecj.unit_im;
                    const vec_t z_im = term_unit_re*vecj.unit_im - term_unit_im*vecj.unit_re;

                    const real_t exponent_re = exponent_ij + shift_ki + vecj.log_mod;
                    term_mod = EXP(exponent_re);

                    res_re += term_mod*z_re;
                    res_im += term_mod*z_im;
                }

                vec_arr[i].log_mod = -shift_ki;
                vec_arr[i].unit_re = res_re;
                vec_arr[i].unit_im = res_im;
            }
        }

        const real_t s2_re = M_PI*mult->gamma/mult->T;
        const double resN_exp = exp((double)vec_arr[N].log_mod + lgac[N]);

        const complex_t s2 = s2_re + I*n*M_PI;
        complex_t sinc = (1-EXPC(-s2))/s2;
        if ((n == 0) && (ABS(s2_re)<=EPS)) sinc = 1 - s2*(0.5 - s2/6);

        const complex_t unit = vec_arr[N].unit_re + I*vec_arr[N].unit_im;
        res_arr[n] = resN_exp*(unit*sinc);
    }
    return 0;
}

/*
 * eval_fourier_parallel: nombre conservado por compatibilidad con
 * series(), pero ejecuta una ruta 100% secuencial: reparte el rango
 * [n_min, n_max) en "threads" bloques y los procesa uno tras otro
 * llamando directamente a eval_fourier_thread(), sin pthread_create ni
 * pthread_join.
 */
static void eval_fourier_parallel(
        const multinomial* restrict mult, const int_t n_min, const int_t n_max,
        local_t* restrict local_arrs, result_t* restrict vec_arrs,
        complex double* res_arr,
        options_t* options
) {
    const int_t threads = options->threads;
    const int_t N = mult->N;

    int_t step = (n_max-n_min+threads-1)/threads;

    arguments_t args_i;
    for (int_t i=0; i<threads; i++) {
        int_t start = n_min + i*step;
        int_t end = start + step;
        if (end > n_max) end = n_max;

        args_i.mult = mult;
        args_i.local = &local_arrs[i*(N+1)];
        args_i.vec_arr = &vec_arrs[i*(N+1)];
        args_i.res_arr = res_arr;

        args_i.thread = i;
        args_i.start = start;
        args_i.end = end;

        eval_fourier_thread(&args_i);
    }
}

static void eval_fourier_series(
        const multinomial* restrict mult, const int_t n_min, const int_t n_max,
        local_t* restrict local_arrs, result_t* restrict vec_arrs,
        complex double* res_arr
) {
    arguments_t args;
    arguments_t* args_i = &args;
    args_i->mult = mult;
    args_i->local = local_arrs;
    args_i->vec_arr = vec_arrs;
    args_i->res_arr = res_arr;

    args_i->thread = 0;
    args_i->start = n_min;
    args_i->end = n_max;

    eval_fourier_thread(args_i);
}

static void series(multinomial* mult, multinomial_result* mult_res, options_t* options) {
    const int_t verbose = options->verbose;
    const int_t threads = options->threads;
    const int_t vecs_per_thread = options->vecs_per_thread;
    const int_t N = mult->N;

    const int_t step = threads*vecs_per_thread*STRIDE;
    const int_t max_iter = (options->max_iter+step-1)/step;
    if (options->verbose) {
        printf("Running Sequential\n");
        printf("p0 =\t%.10e\n", mult->p0);
    }

    local_t* local_arrs = malloc(threads*(N+1)*sizeof(local_t));
    result_t* vec_arrs = malloc(threads*(N+1)*sizeof(result_t));
    complex double* res_arr = malloc(max_iter*step*sizeof(complex double));

    const double threshold = 1 + options->eps_rel/CONV_COUNT;
    double delta_acc = 0;
    double delta_acc_im = 0;
    int_t terms_acc = 0;

    double p = 0;
    double p_im = 0;
    long double c = 0;

    int converged_full = 0;
    int_t converged_count = 0;
    double p_prev = INFINITY;

    if (verbose) printf("terms\tcumulative sum\t\tmean change\n");
    int_t iter = 0;
    for (int_t n=0; n < max_iter; n++) {
        if (!isfinite(p) || isnan(p)) {
            p = NAN;
            converged_full = 0;
            goto finish;
        }

        const int_t n_min = n*threads*vecs_per_thread;
        const int_t n_max = (n+1)*threads*vecs_per_thread;
        if (threads<=1) eval_fourier_series(mult, n_min, n_max, local_arrs, vec_arrs, res_arr);
        else eval_fourier_parallel(mult, n_min, n_max, local_arrs, vec_arrs, res_arr, options);

        if (n==0) {
            if (res_arr[0] == 0.0) {  // sum of underflows
                if (verbose) printf("%04lld\t%.10e\n", iter, p);
                p = 0;
                converged_full = 1;
                goto finish;
            }
            res_arr[0] /= 2;
        }

        for (iter=n_min*STRIDE; iter<n_max*STRIDE; iter++) {
            double delta_re = creal(res_arr[iter]);
            double delta_im = cimag(res_arr[iter]);
            double y = delta_re - c;
            double t = p + y;
            c = (t - p) - y;
            p = t;
            p_im += delta_im;

            if (verbose) {
                delta_acc += delta_re;
                delta_acc_im += delta_im;
                terms_acc++;
                if (iter % options->print_freq == 0) {
                    if (iter == 0) {
                        printf("%04lld\t%.10e\n", iter, p);
                    }
                    else {
                        printf("%04lld\t%.10e\t%.10e\n", iter, p, delta_acc / terms_acc);
                    }
                    delta_acc = 0;
                    terms_acc = 0;
                }
            }

            const double frac = p/p_prev;
            const int converged = (p > 0.0) & (frac <= threshold) & (1/frac <= threshold);
            converged_count = converged ? converged_count + 1 : 0;
            p_prev = p;

            if (converged_count >= CONV_COUNT) {
                if (verbose) {
                    printf("*%04lld\t%.10e\t%.10e\n", iter, p, delta_acc);
                }
                converged_full = 1;
                goto finish;
            }
        }
    }

    finish:
    p += mult->p0/2;
    if (verbose) {
        printf("p = %.10e\n", p);
        printf("converged: "); printf(converged_full ? "true" : "false"); printf("\n");
    }

    mult_res->pval = p;
    mult_res->converged = converged_full;
    mult_res->terms = iter;

    free(local_arrs);
    free(vec_arrs);
    free(res_arr);
}