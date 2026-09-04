#include "base.h"
#pragma once

#define REWARD reward_logp  // pmf statistic
//#define REWARD reward_llr  // log-likelihood ratio statistic
//#define REWARD reward_pearson  // Pearson statistic

typedef struct {
    real_t eps_rel;
    int_t max_iter;
    real_t undersampling;

    real_t eps_gamma;
    real_t enum_cutoff;

    int_t threads;
    int_t vecs_per_thread;

    int_t verbose;
    int_t print_freq;
} options_t;

static void set_options_default(options_t* options) {
    options->eps_rel = 1e-3;
    options->max_iter = 1000;
    options->undersampling = 1;

    options->eps_gamma = 1e-2;
    options->enum_cutoff = 6.5;

    options->threads = 8;
    options->vecs_per_thread = 1;

    options->verbose = 1;
    options->print_freq = 10;
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
