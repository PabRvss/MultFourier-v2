#pragma once

typedef enum {
    FFT_NONE,
    FFT_REAL,
    FFT_DD,
    FFT_QUAD,
} fft_precision_t;

typedef struct {
    real_t lambda;

    real_t eps_rel;
    int_t max_iter;
    int_t B;
    real_t undersampling;

    real_t eps_gamma;
    real_t enum_cutoff;

    fft_precision_t gamma_precision;
    fft_precision_t fft_precision;
    int_t use_fft_precompute;
    int_t error_bound;  // compute truncation diagnostics in series() (see multinomial/error_bound.h)

    int_t threads;
    int_t vecs_per_thread;

    int_t verbose;
    int_t print_freq;
} options_t;

static void set_options_default(options_t* options) {
    // options->lambda = NAN;  // for pmf statistic
    options->lambda = 0.0;  // 0 for LLR, 1 for chi-squared

    options->eps_rel = 1e-3;
    options->max_iter = 10000;
    options->B = 30;
    options->undersampling = 1.0;

    options->eps_gamma = 1e-2;
    // options->enum_cutoff = INFINITY;
    options->enum_cutoff = 0.0;

    options->gamma_precision = FFT_REAL;
    options->fft_precision = FFT_REAL;
    options->use_fft_precompute = 1;
    options->error_bound = 0;

    options->threads = 8;
    options->vecs_per_thread = 1;

    options->verbose = 1;
    options->print_freq = 100;
}

static void set_options(
    options_t* options,

    real_t eps_rel,
    int_t max_iter,
    real_t undersampling,

    real_t eps_gamma,
    real_t enum_cutoff,

    int_t threads,
    int_t vecs_per_thread,

    int_t verbose,
    int_t print_freq
) {
    set_options_default(options);

    options->lambda = 0;  // TODO: complete

    options->eps_rel = eps_rel;
    options->max_iter = max_iter;
    options->undersampling = undersampling;

    options->eps_gamma = eps_gamma;
    options->enum_cutoff = enum_cutoff;

    options->threads = threads;
    options->vecs_per_thread = vecs_per_thread;

    options->verbose = verbose;
    options->print_freq = print_freq;
}
