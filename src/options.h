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

    int_t average_flat;
    real_t average_window;

    int_t threads;
    int_t vecs_per_thread;

    double max_time;    // max execution time in seconds (< 0 or INFINITY for no limit)

    int_t verbose;
    int_t print_freq;
} options_t;

static void set_options_default(options_t* options) {
    options->lambda = 0.0;  // 0 for LLR, 1 for chi-squared, NAN for pmf

    options->eps_rel = 1e-3;
    options->max_iter = 10000;
    options->B = 30;
    options->undersampling = 1.0;

    options->eps_gamma = 1e-2;
    options->enum_cutoff = 0.0;

    options->gamma_precision = FFT_REAL;
    options->fft_precision = FFT_REAL;
    options->use_fft_precompute = 1;
    options->error_bound = 0;

    options->average_flat = 0;
    options->average_window = 0.0;

    options->threads = 1;
    options->vecs_per_thread = 1;

    options->max_time = -1.0;

    options->verbose = 0;
    options->print_freq = 100;
}
