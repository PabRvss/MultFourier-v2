# Helper interno para construir el objeto S3 multfourier
build_multfourier_s3 <- function(raw, x, p, elapsed, requested_method, stat_label) {
  method_name <- if (requested_method == "flexible") {
    if (!is.null(raw$method) && raw$method == 1L) "fourier" else "exhaustive"
  } else {
    requested_method
  }
  
  gamma_val <- if (method_name == "fourier") as.numeric(raw$gamma) else NA_real_
  time_gamma_val <- if (method_name == "fourier") as.numeric(raw$eval_time) else NA_real_
  w_val <- if (method_name == "fourier") as.numeric(raw$W) else NA_real_
  n_terms_val <- if (method_name == "fourier") as.integer(raw$terms) else NA_integer_
  
  # Status ID y mensajes sin punto al final:
  # 0: Converged
  # 1: Maximum time reached
  # 2: Maximum number of terms reached
  # 3: Could not solve the optimization of gamma
  # 4: Numerical instability detected (magnitude explosion)
  status_id <- as.integer(raw$status)
  status_msg <- switch(
    as.character(status_id),
    "0" = "Converged",
    "1" = "Maximum time reached",
    "2" = "Maximum number of terms reached",
    "3" = "Could not solve the optimization of gamma",
    "4" = "Numerical instability detected (magnitude explosion)",
    "Unknown status"
  )
  
  # Si C reporto status 0 pero termino sin converger en modo fourier:
  if (method_name == "fourier" && status_id == 0L && !isTRUE(raw$converged == 1L)) {
    status_id <- 2L
    status_msg <- "Maximum number of terms reached"
  }
  
  out <- list(
    x = as.numeric(x),
    p = as.numeric(p),
    pval = as.numeric(raw$pval),
    Time_gamma = time_gamma_val,
    W = w_val,
    stat = as.character(stat_label),
    Gamma = gamma_val,
    n_terms = n_terms_val,
    time = if (!is.null(raw$total_time) && raw$total_time > 0) as.numeric(raw$total_time) else as.numeric(elapsed),
    status = as.integer(status_id),
    message = as.character(status_msg),
    method = as.character(method_name)
  )
  
  class(out) <- "multfourier"
  for (nm in names(out)) {
    attr(out, nm) <- out[[nm]]
  }
  
  if (!is.null(raw$err_rel_est)) attr(out, "err_rel_est") <- as.numeric(raw$err_rel_est)
  if (!is.null(raw$err_rel_ext)) attr(out, "err_rel_ext") <- as.numeric(raw$err_rel_ext)
  if (!is.null(raw$n_eff)) attr(out, "n_eff") <- as.numeric(raw$n_eff)
  if (!is.null(raw$nu_max_ratio)) attr(out, "nu_max_ratio") <- as.numeric(raw$nu_max_ratio)
  
  out
}

# Helper interno para resolver y validar el estadistico y lambda
resolve_stat_lambda <- function(stat, lambda, has_lambda = FALSE) {
  stat <- match.arg(stat, c("llr", "pd", "chi2", "pmf"))
  if (has_lambda && stat != "pd") {
    warning('Parameter lambda is ignored since stat is not "pd"')
  }
  if (stat == "chi2") {
    lambda_val <- 1.0
    stat_label <- "Pearson's Chi-Square"
  } else if (stat == "llr") {
    lambda_val <- 0.0
    stat_label <- "Log-Likelihood Ratio"
  } else if (stat == "pmf") {
    lambda_val <- NaN
    stat_label <- "Probability Mass Function"
  } else { # "pd"
    if (length(lambda) != 1L || !is.numeric(lambda) || !is.finite(lambda)) {
      stop("'lambda' must be a single finite real number.")
    }
    lambda_val <- as.numeric(lambda)
    stat_label <- sprintf("power-divergence (lambda = %s)", format(lambda_val))
  }
  list(stat = stat, lambda = lambda_val, label = stat_label)
}

# Helper interno para acotar n_threads a los nucleos disponibles
resolve_n_threads <- function(n_threads) {
  max_cores <- parallel::detectCores()
  if (is.na(max_cores) || max_cores < 1L) max_cores <- 1L
  n_threads <- as.integer(n_threads)
  if (is.na(n_threads) || n_threads < 1L) {
    stop("'n_threads' must be a positive integer greater than or equal to 1.")
  }
  min(n_threads, max_cores)
}

#' Multinomial p-value computation via flexible method
#'
#' Evaluates whether the dimension of the multinomial support allows exact
#' enumeration (iterative bisection) or routes to the Fourier inversion series method.
#'
#' @param x Numeric vector of observed counts for each category.
#' @param p Numeric vector of theoretical probabilities under the null hypothesis (must sum to 1).
#' @param stat Test statistic: \code{"llr"} (Log-Likelihood Ratio), \code{"pd"} (Power Divergence), \code{"chi2"} (Pearson's Chi-squared), or \code{"pmf"} (Multinomial Point Probability Mass). Default is \code{"llr"}.
#' @param lambda Real parameter for Power Divergence when \code{stat = "pd"}. Default is 1. Ignored if \code{stat != "pd"}.
#' @param undersampling Undersampling factor (> 0) for the period in the Fourier series. Default is 1.
#' @param rel_eps Relative convergence tolerance (>= 0). Default is 0.001.
#' @param B Integer >= 0; number of consecutive terms within tolerance required for convergence. Default is 30.
#' @param max_terms Maximum number of harmonic terms to evaluate in the series (positive integer). Default is 10000.
#' @param max_time Maximum execution time in seconds (optional).
#' @param avg_window Fraction of final partial sums to average (between 0 and 1). Default is 0.
#' @param avg_flat Logical; if TRUE, uses a flat moving average; if FALSE, applies linear weighting towards the end. Default is FALSE.
#' @param n_threads Number of execution threads (positive integer >= 1). Default is \code{min(2L, parallel::detectCores())}.
#' @param precision Arithmetic precision: "double" (standard) or "double-double" (extended precision). Default is "double".
#' @param precompute Logical; if TRUE, uses precomputed transforms for acceleration. Default is TRUE.
#' @param engine Computational engine: \code{"speedup"} (default, using Poisson series, Newton saddlepoint search, O(K) bounds, and Shanks extrapolation) or \code{"standard"} (classic polynomial convolution and bisection algorithms). Default is "speedup".
#' @param verbose Logical; if TRUE, prints progress information. Default is FALSE.
#' @param ... Additional internal parameters passed to the C engine (e.g. \code{enum_cutoff = 7.4}, the threshold for \eqn{\log_{10}} support size under which exact bisection is selected, corresponding to approximately 1 second of execution).
#' @return Returns an S3 object of class \code{"multfourier"} with the following attributes:
#' \describe{
#'   \item{x}{The input vector of observed realizations for each category.}
#'   \item{p}{The input vector of probabilities for each category under the null hypothesis.}
#'   \item{pval}{The computed p-value.}
#'   \item{Time_gamma}{The time spent optimizing gamma in seconds.}
#'   \item{W}{The effective width of the distribution support interval.}
#'   \item{stat}{A string describing the test statistic used (e.g. \code{"Log-Likelihood Ratio"}, \code{"Pearson's Chi-Square"}, \code{"Probability Mass Function"}, or \code{"power-divergence (lambda = ...)"}).}
#'   \item{Gamma}{The optimal gamma obtained in the first part of the method.}
#'   \item{n_terms}{The number of terms of the Fourier sum evaluated.}
#'   \item{time}{The total execution time of the algorithm in seconds.}
#'   \item{status}{The final status ID of the algorithm upon completion:
#'     \itemize{
#'       \item \code{0}: Converged.
#'       \item \code{1}: Maximum time reached.
#'       \item \code{2}: Maximum number of terms reached.
#'       \item \code{3}: Could not solve the optimization of gamma.
#'       \item \code{4}: Numerical instability detected (magnitude explosion).
#'     }
#'   }
#'   \item{message}{The finishing status displayed as a human-readable message.}
#'   \item{method}{A string with value \code{"fourier"} or \code{"exhaustive"}.}
#' }
#' @useDynLib MultFourier, c_run_multfourier
#' @export
pval_flexible <- function(x,
                          p = rep(1/length(x), length(x)),
                          stat = c("llr", "pd", "chi2", "pmf"),
                          lambda = 1,
                          undersampling = 1,
                          rel_eps = 0.001,
                          B = 30,
                          max_terms = 10000,
                          max_time = NULL,
                          avg_window = 0,
                          avg_flat = FALSE,
                          n_threads = min(2L, parallel::detectCores()),
                          precision = c("double", "double-double"),
                          precompute = TRUE,
                          engine = c("speedup", "standard"),
                          verbose = FALSE,
                          ...) {
  if (length(x) != length(p)) stop("'x' and 'p' must have the same length.")
  if (any(x < 0) || any(p <= 0)) stop("Counts must be non-negative and probabilities strictly positive.")
  if (undersampling <= 0) stop("'undersampling' must be strictly positive.")
  if (rel_eps < 0) stop("'rel_eps' must be non-negative.")
  if (B < 0) stop("'B' must be an integer greater than or equal to 0.")
  if (max_terms <= 0) stop("'max_terms' must be a positive integer.")
  if (avg_window < 0 || avg_window > 1) stop("'avg_window' must be between 0 and 1.")

  st <- resolve_stat_lambda(stat, lambda, has_lambda = !missing(lambda))
  threads_val <- resolve_n_threads(n_threads)
  precision <- match.arg(precision)
  engine <- match.arg(engine)

  opts <- list(
    method = "flexible",
    stat = st$stat,
    lambda = st$lambda,
    undersampling = as.numeric(undersampling),
    avg_window = as.numeric(avg_window),
    avg_flat = as.logical(avg_flat),
    rel_eps = as.numeric(rel_eps),
    max_terms = as.integer(max_terms),
    B = as.integer(B),
    n_threads = as.integer(threads_val),
    precision = precision,
    precompute = as.logical(precompute),
    engine = engine,
    speedup = (engine == "speedup"),
    max_time = if (!is.null(max_time) && is.finite(max_time)) as.numeric(max_time) else -1.0,
    verbose = as.logical(verbose),
    ...
  )

  t0 <- proc.time()
  raw <- .Call(c_run_multfourier, as.double(x), as.double(p), opts)
  elapsed <- (proc.time() - t0)[["elapsed"]]

  build_multfourier_s3(raw, x, p, elapsed, "flexible", st$label)
}

#' Exact multinomial p-value computation via iterative bisection
#'
#' @param x Numeric vector of observed counts for each category.
#' @param p Numeric vector of theoretical probabilities under the null hypothesis.
#' @param stat Test statistic: \code{"llr"}, \code{"pd"}, \code{"chi2"}, or \code{"pmf"}. Default is \code{"llr"}.
#' @param lambda Real parameter for Power Divergence when \code{stat = "pd"}. Default is 1. Ignored if \code{stat != "pd"}.
#' @param n_threads Number of execution threads (integer >= 1). Default is 1.
#' @param max_time Maximum execution time in seconds (optional).
#' @param engine Computational engine: \code{"speedup"} (default, uses pruning bounds via \code{fast_exhaustive}) or \code{"standard"} (iterative bisection in last categories). Default is "speedup".
#' @param verbose Logical; if TRUE, prints progress information. Default is FALSE.
#' @param ... Additional internal parameters passed to the C engine.
#' @return Returns an S3 object of class \code{"multfourier"} with the following attributes:
#' \describe{
#'   \item{x}{The input vector of observed realizations for each category.}
#'   \item{p}{The input vector of probabilities for each category under the null hypothesis.}
#'   \item{pval}{The computed p-value.}
#'   \item{stat}{A string describing the test statistic used.}
#'   \item{time}{The total execution time of the algorithm in seconds.}
#'   \item{status}{The final status ID of the algorithm upon completion (\code{0}: Converged, \code{1}: Maximum time reached).}
#'   \item{message}{The finishing status displayed as a human-readable message.}
#'   \item{method}{A string with value \code{"exhaustive"}.}
#' }
#' @export
pval_exhaustive <- function(x,
                            p = rep(1/length(x), length(x)),
                            stat = c("llr", "pd", "chi2", "pmf"),
                            lambda = 1,
                            n_threads = 1,
                            max_time = NULL,
                            engine = c("speedup", "standard"),
                            verbose = FALSE,
                            ...) {
  if (length(x) != length(p)) stop("'x' and 'p' must have the same length.")
  if (any(x < 0) || any(p <= 0)) stop("Counts must be non-negative and probabilities strictly positive.")

  st <- resolve_stat_lambda(stat, lambda, has_lambda = !missing(lambda))
  threads_val <- resolve_n_threads(n_threads)
  engine <- match.arg(engine)

  opts <- list(
    method = "exhaustive",
    stat = st$stat,
    lambda = st$lambda,
    n_threads = as.integer(threads_val),
    engine = engine,
    speedup = (engine == "speedup"),
    max_time = if (!is.null(max_time) && is.finite(max_time)) as.numeric(max_time) else -1.0,
    verbose = as.logical(verbose),
    ...
  )

  t0 <- proc.time()
  raw <- .Call(c_run_multfourier, as.double(x), as.double(p), opts)
  elapsed <- (proc.time() - t0)[["elapsed"]]

  build_multfourier_s3(raw, x, p, elapsed, "exhaustive", st$label)
}

#' Multinomial p-value computation via Fourier series inversion
#'
#' @inheritParams pval_flexible
#' @return Returns an S3 object of class \code{"multfourier"} with the following attributes:
#' \describe{
#'   \item{x}{The input vector of observed realizations for each category.}
#'   \item{p}{The input vector of probabilities for each category under the null hypothesis.}
#'   \item{pval}{The computed p-value.}
#'   \item{Time_gamma}{The time spent optimizing gamma in seconds.}
#'   \item{W}{The effective width of the distribution support interval.}
#'   \item{stat}{A string describing the test statistic used (e.g. \code{"Log-Likelihood Ratio"}, \code{"Pearson's Chi-Square"}, \code{"Probability Mass Function"}, or \code{"power-divergence (lambda = ...)"}).}
#'   \item{Gamma}{The optimal gamma obtained in the first part of the method.}
#'   \item{n_terms}{The number of terms of the Fourier sum evaluated.}
#'   \item{time}{The total execution time of the algorithm in seconds.}
#'   \item{status}{The final status ID of the algorithm upon completion:
#'     \itemize{
#'       \item \code{0}: Converged.
#'       \item \code{1}: Maximum time reached.
#'       \item \code{2}: Maximum number of terms reached.
#'       \item \code{3}: Could not solve the optimization of gamma.
#'       \item \code{4}: Numerical instability detected (magnitude explosion).
#'     }
#'   }
#'   \item{message}{The finishing status displayed as a human-readable message.}
#'   \item{method}{A string with value \code{"fourier"}.}
#' }
#' @export
pval_fourier <- function(x,
                         p = rep(1/length(x), length(x)),
                         stat = c("llr", "pd", "chi2", "pmf"),
                         lambda = 1,
                         undersampling = 1,
                         rel_eps = 0.001,
                         B = 30,
                         max_terms = 10000,
                         max_time = NULL,
                         avg_window = 0,
                         avg_flat = FALSE,
                         n_threads = min(2L, parallel::detectCores()),
                         precision = c("double", "double-double"),
                         precompute = TRUE,
                         engine = c("speedup", "standard"),
                         verbose = FALSE,
                         ...) {
  if (length(x) != length(p)) stop("'x' and 'p' must have the same length.")
  if (any(x < 0) || any(p <= 0)) stop("Counts must be non-negative and probabilities strictly positive.")
  if (undersampling <= 0) stop("'undersampling' must be strictly positive.")
  if (rel_eps < 0) stop("'rel_eps' must be non-negative.")
  if (B < 0) stop("'B' must be an integer greater than or equal to 0.")
  if (max_terms <= 0) stop("'max_terms' must be a positive integer.")
  if (avg_window < 0 || avg_window > 1) stop("'avg_window' must be between 0 and 1.")

  st <- resolve_stat_lambda(stat, lambda, has_lambda = !missing(lambda))
  threads_val <- resolve_n_threads(n_threads)
  precision <- match.arg(precision)
  engine <- match.arg(engine)

  opts <- list(
    method = "fourier",
    stat = st$stat,
    lambda = st$lambda,
    undersampling = as.numeric(undersampling),
    avg_window = as.numeric(avg_window),
    avg_flat = as.logical(avg_flat),
    rel_eps = as.numeric(rel_eps),
    max_terms = as.integer(max_terms),
    B = as.integer(B),
    n_threads = as.integer(threads_val),
    precision = precision,
    precompute = as.logical(precompute),
    engine = engine,
    speedup = (engine == "speedup"),
    max_time = if (!is.null(max_time) && is.finite(max_time)) as.numeric(max_time) else -1.0,
    verbose = as.logical(verbose),
    ...
  )

  t0 <- proc.time()
  raw <- .Call(c_run_multfourier, as.double(x), as.double(p), opts)
  elapsed <- (proc.time() - t0)[["elapsed"]]

  build_multfourier_s3(raw, x, p, elapsed, "fourier", st$label)
}

#' Print method for multfourier objects
#'
#' @param x Object of class \code{"multfourier"} returned by \code{pval_fourier}, \code{pval_exhaustive}, or \code{pval_flexible}.
#' @param ... Additional arguments (currently unused).
#'
#' @return Invisibly returns the input object.
#' @export
print.multfourier <- function(x, ...) {
  cat("Multinomial Test (MultFourier)\n")
  cat("------------------------------\n")
  cat(sprintf("%-11s: %s\n", "Method", x$method))
  cat(sprintf("%-11s: %s\n", "Statistic", x$stat))
  cat(sprintf("%-11s: %s\n", "p-value", format(x$pval, digits = 8)))
  cat(sprintf("%-11s: %s\n", "Status", paste0(x$status, " (", x$message, ")")))
  if (!is.null(x$Gamma) && !is.na(x$Gamma)) {
    cat(sprintf("%-11s: %s\n", "Gamma", format(x$Gamma, digits = 6)))
  }
  if (!is.null(x$Time_gamma) && !is.na(x$Time_gamma)) {
    cat(sprintf("%-11s: %s seconds\n", "Time gamma", format(x$Time_gamma, digits = 4)))
  }
  if (!is.null(x$W) && !is.na(x$W)) {
    cat(sprintf("%-11s: %s\n", "W", format(x$W, digits = 6)))
  }
  if (!is.null(x$n_terms) && !is.na(x$n_terms)) {
    cat(sprintf("%-11s: %s\n", "Terms", x$n_terms))
  }
  cat(sprintf("%-11s: %s seconds\n", "Time", sprintf("%.6f", x$time)))
  invisible(x)
}