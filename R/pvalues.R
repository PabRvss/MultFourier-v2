# Helper interno para construir el objeto S3 multfourier
build_multfourier_s3 <- function(raw, x, p, elapsed, requested_method, stat_label) {
  method_name <- if (requested_method == "flexible") {
    if (!is.null(raw$method) && raw$method == 1L) "fourier" else "exact"
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
    stat = as.character(stat_label),
    method = as.character(method_name),
    status = as.integer(status_id),
    message = as.character(status_msg),
    time = if (!is.null(raw$total_time) && raw$total_time > 0) as.numeric(raw$total_time) else as.numeric(elapsed),
    gamma = gamma_val,
    time_gamma = time_gamma_val,
    W = w_val,
    n_terms = n_terms_val,
    err_rel_ext = diag_val(raw$err_rel_ext),
    err_rel_est = diag_val(raw$err_rel_est),
    n_eff = diag_val(raw$n_eff),
    nu_max_ratio = diag_val(raw$nu_max_ratio)
  )

  class(out) <- "multfourier"
  out
}

# Diagnosticos de la serie de Fourier: NA para el metodo exacto o si faltan
diag_val <- function(v) if (is.null(v)) NA_real_ else as.numeric(v)

# Nombres antiguos de componentes, aceptados con aviso (usar los nuevos)
multfourier_old_names <- c(Gamma = "gamma", Time_gamma = "time_gamma")

#' @export
`$.multfourier` <- function(x, name) {
  if (name %in% names(multfourier_old_names)) {
    new <- multfourier_old_names[[name]]
    warning(sprintf("'%s' is deprecated; use '%s' instead.", name, new), call. = FALSE)
    name <- new
  }
  .subset2(x, name)
}

# Helper interno para validar x y p (y reescalar p si rescale_p = TRUE, como
# chisq.test(): mismo valor por defecto y misma tolerancia)
validate_x_p <- function(x, p, rescale_p = FALSE) {
  if (!is.numeric(x) || length(x) < 1L) stop("'x' must be a numeric vector of counts.")
  if (!is.numeric(p)) stop("'p' must be a numeric vector of probabilities.")
  if (length(x) != length(p)) stop("'x' and 'p' must have the same length.")
  if (anyNA(x) || any(!is.finite(x))) stop("'x' must not contain NA, NaN or infinite values.")
  if (any(x < 0)) stop("'x' must contain non-negative counts.")
  if (any(abs(x - round(x)) > 1e-8)) stop("'x' must contain integer counts.")
  if (anyNA(p) || any(!is.finite(p))) stop("'p' must not contain NA, NaN or infinite values.")
  if (any(p <= 0)) stop("'p' must contain strictly positive probabilities.")
  if (!is.logical(rescale_p) || length(rescale_p) != 1L || is.na(rescale_p)) {
    stop("'rescale_p' must be TRUE or FALSE.")
  }
  if (rescale_p) p <- p / sum(p)
  if (abs(sum(p) - 1) > sqrt(.Machine$double.eps)) {
    stop(sprintf("'p' must sum to 1 (sum = %s); use rescale_p = TRUE to rescale it to p / sum(p).",
                 format(sum(p), digits = 8)))
  }
  list(x = round(x), p = p)
}

# Helper interno para resolver y validar el estadistico y lambda
resolve_stat_lambda <- function(stat, lambda = NULL) {
  stat <- match.arg(stat, c("llr", "pd", "chi2", "pmf"))
  if (!is.null(lambda) && stat != "pd") {
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
    if (is.null(lambda)) {
      stop('\'lambda\' must be given when stat = "pd" (e.g. lambda = 2/3).')
    }
    if (length(lambda) != 1L || !is.numeric(lambda) || !is.finite(lambda)) {
      stop("'lambda' must be a single finite real number.")
    }
    if (abs(lambda + 1) < 1e-8) {
      stop("'lambda = -1' is not supported; use a value different from -1.")
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


#' Multinomial test p-value by Fourier series inversion
#'
#' Computes the \pvalue{} of a multinomial goodness-of-fit test,
#' \deqn{p = P\{T(Y) \ge T(x)\}, \qquad Y \sim \mathrm{Multinomial}(N, p), \quad N = \sum_k x_k,}{p = P{T(Y) >= T(x)},   Y ~ Multinomial(N, p),   N = sum_k x_k,}
#' for an additive test statistic \eqn{T}, by summing a truncated Fourier series
#' whose terms are values of the moment-generating function of the statistic.
#' The series is truncated by an adaptive stopping rule, so the result is an
#' approximation whose accuracy depends on the statistic and the instance; always
#' check the returned \code{status}.
#'
#' @param x Vector of observed counts, one per category (non-negative integers,
#'   without missing values). Their sum is the number of trials \eqn{N}.
#' @param p Vector of category probabilities under the null hypothesis, all
#'   strictly positive. They must sum to 1 (up to a tolerance of
#'   \code{sqrt(.Machine$double.eps)}), unless \code{rescale_p = TRUE}.
#'   Default: equiprobable categories.
#' @param stat Test statistic: \code{"llr"} (log-likelihood ratio \eqn{G^2}, the
#'   default), \code{"chi2"} (Pearson's \eqn{\mathcal{X}^2}{X^2}), \code{"pd"}
#'   (Cressie--Read power divergence with parameter \code{lambda}) or
#'   \code{"pmf"} (probability mass function of the multinomial, i.e. the exact
#'   multinomial test). See Details.
#' @param lambda Parameter \eqn{\lambda} of the power-divergence statistic; used
#'   only when \code{stat = "pd"}, and then required (there is no default);
#'   a warning is issued if it is given for another statistic.
#' @param rescale_p Logical; if \code{TRUE}, \code{p} is rescaled to
#'   \code{p / sum(p)}, so it can be given as any vector of positive weights. If
#'   \code{FALSE} (default) and \code{p} does not sum to 1, an error is given.
#'   Same argument and default as \code{rescale.p} in
#'   \code{\link[stats]{chisq.test}}.
#' @param undersampling Positive factor dividing the half-width \eqn{W} of the
#'   Fourier period. The default 1 uses the smallest \eqn{W} for which the series
#'   is exact. Values below 1 enlarge \eqn{W} (still exact, but more terms are
#'   needed); values above 1 shrink \eqn{W} below that minimum, so the series
#'   converges to a wrong value (aliasing). Intended for experimentation only.
#' @param rel_eps Relative tolerance of the stopping rule (\eqn{\ge 0}{>= 0}). Default
#'   0.001. It controls how stable the partial sums must be before the series is
#'   truncated; it is \emph{not} a bound on the error of the returned \pvalue{}.
#' @param B Non-negative integer used by the term-size stopping rule: the series
#'   stops after \code{B} consecutive terms each smaller, relative to the partial
#'   sum, than \code{rel_eps / B}. Default 50; \code{B = 0} uses an internal value.
#' @param max_terms Maximum number of series terms (positive integer). Default
#'   10000. If it is reached before the stopping rule is met, \code{status} is 2.
#' @param max_time Optional time limit in seconds. \code{NULL} (default) means no
#'   limit. When it is reached, \code{status} is 1.
#' @param avg_window Fraction (between 0 and 1) of the last partial sums to
#'   average into the reported value. Default 0 (no averaging). When positive, it
#'   replaces the extrapolated value of \code{engine = "speedup"}; averaging is a
#'   linear filter that can bias the result and is usually less accurate than the
#'   default.
#' @param avg_flat Logical; with \code{avg_window > 0}, \code{TRUE} gives equal
#'   weights and \code{FALSE} (default) weights that grow linearly towards the
#'   last partial sum.
#' @param n_threads Number of threads used to evaluate the series terms (positive
#'   integer, capped at the number of available cores). Default
#'   \code{min(2L, parallel::detectCores())}.
#' @param precision Floating-point precision of the term evaluator:
#'   \code{"double"} (default) or \code{"double-double"}. Only used with
#'   \code{engine = "standard"}; ignored by the default engine.
#' @param precompute Logical; if \code{TRUE} (default), precomputes the FFT plan
#'   of the term evaluator. Only used with \code{engine = "standard"}.
#' @param engine \code{"speedup"} (default) or \code{"standard"}. See Details.
#' @param verbose Logical; if \code{TRUE}, prints the partial sums and diagnostic
#'   information while running. Default \code{FALSE}.
#' @param ... Advanced options passed to the C engine, for example
#'   \code{print_freq} (how often partial sums are printed when
#'   \code{verbose = TRUE}; default 100) or \code{extrapolate} (0: no
#'   extrapolation; 1: report the extrapolated value; 2: also stop on it, the
#'   default of \code{engine = "speedup"}). Unknown names produce a warning.
#'
#' @details
#' \strong{Statistics.} All supported statistics are additive,
#' \eqn{T(y) = \sum_k f_k(y_k)}{T(y) = sum_k f_k(y_k)}, which is what the method requires. With
#' \eqn{E_k = N p_k}:
#' \itemize{
#'   \item \code{"llr"}: \eqn{G^2 = 2 \sum_k y_k \log(y_k / E_k)}{G^2 = 2 sum_k y_k log(y_k / E_k)}.
#'   \item \code{"chi2"}: \eqn{\mathcal{X}^2 = \sum_k (y_k - E_k)^2 / E_k}{X^2 = sum_k (y_k - E_k)^2 / E_k}.
#'   \item \code{"pd"}: \eqn{2 I^\lambda = \frac{2}{\lambda(\lambda+1)} \sum_k y_k [(y_k/E_k)^\lambda - 1]}{2 I^lambda = 2 / (lambda (lambda + 1)) sum_k y_k [(y_k / E_k)^lambda - 1]}
#'     (Cressie and Read, 1984); \eqn{\lambda \to 0}{lambda -> 0} gives \eqn{G^2} and
#'     \eqn{\lambda = 1} gives \eqn{\mathcal{X}^2}{X^2}.
#'   \item \code{"pmf"}: the probability mass function \eqn{P(y)} of the
#'     multinomial under the null hypothesis (exact multinomial test). Small
#'     values are the extreme ones, so the \pvalue{} is \eqn{P\{P(Y) \le P(x)\}}{P{P(Y) <= P(x)}},
#'     the total probability of the outcomes that are at most as likely as the
#'     observed one.
#' }
#'
#' \strong{Which statistics converge.} The \pvalue{} is written exactly as a Fourier
#' series for every additive statistic, but the number of terms needed for a given
#' accuracy depends on the statistic. For \code{"llr"} and \code{"pmf"} the series
#' typically converges in a few hundred to a few thousand terms. For \code{"chi2"},
#' and often for \code{"pd"} unless \eqn{\lambda} is close to 0 (\eqn{\lambda = 2/3}
#' has failed on real data), the values of the statistic lie on or near a
#' lattice, the terms do not decay, and the series can fail to
#' converge within \code{max_terms} (\code{status = 2}) or even return a negative
#' value. In that case use \code{\link{pval_exact}} when it is feasible.
#'
#' \strong{Stopping rule and reported value.} With \code{engine = "speedup"} the
#' partial sums are accelerated with Wynn's epsilon algorithm (in the safeguarded
#' form of QUADPACK's DQELG) and smoothed with a running median of five. The
#' series stops as soon as either (i) \code{B} consecutive terms are each smaller
#' than \code{rel_eps / B} relative to the partial sum, or (ii) the smoothed
#' extrapolated values vary by at most a relative \code{rel_eps} over a trailing
#' window covering the last half of the terms. The reported \pvalue{} is the last
#' smoothed extrapolated value. With \code{engine = "standard"} only rule (i) is
#' used and the reported \pvalue{} is the last partial sum.
#'
#' \strong{Engines.} \code{"speedup"} evaluates the terms with a Poissonized
#' Cauchy-integral evaluator, chooses the exponential tilt \code{gamma} by a
#' Newton saddlepoint search, computes the range of the statistic in closed form,
#' and uses the extrapolation described above. \code{"standard"} is the earlier
#' implementation: polynomial convolution evaluated by FFT, a grid search for the
#' tilt, a dynamic program for the range, and no extrapolation. Both compute the
#' same series.
#'
#' @return An object of class \code{"multfourier"}: a list with components
#' \describe{
#'   \item{x, p}{The input counts and null probabilities (\code{p} after
#'     rescaling, if \code{rescale_p = TRUE}).}
#'   \item{pval}{The computed \pvalue{}.}
#'   \item{stat}{Name of the test statistic, e.g. \code{"Log-Likelihood Ratio"}
#'     or \code{"power-divergence (lambda = 0.6667)"}.}
#'   \item{method}{\code{"fourier"}.}
#'   \item{time}{Total wall-clock time in seconds.}
#'   \item{gamma}{The exponential tilt \eqn{\gamma} applied to the statistic so
#'     that the series terms stay within floating-point range (a saddlepoint
#'     choice).}
#'   \item{time_gamma}{Time in seconds spent choosing \code{gamma}.}
#'   \item{W}{Half-width of the Fourier period, \eqn{\max(s_{\max}, -s_{\min})}{max(s_max, -s_min)}
#'     over the range \eqn{[s_{\min}, s_{\max}]}{[s_min, s_max]} of the statistic minus its
#'     observed value, divided by \code{undersampling}.}
#'   \item{n_terms}{Number of series terms summed.}
#'   \item{status}{\code{0}: the stopping rule was met (the message reads
#'     \code{"Converged"}); \code{1}: \code{max_time} was reached; \code{2}:
#'     \code{max_terms} was reached without meeting the stopping rule, so
#'     \code{pval} is unreliable; \code{3}: the search for \code{gamma} failed;
#'     \code{4}: numerical overflow in the evaluation of the terms. With a status
#'     other than 0, \code{pval} is the last value reached or \code{NA}.}
#'   \item{message}{Human-readable version of \code{status}.}
#'   \item{err_rel_ext}{Relative spread of the smoothed extrapolated values over
#'     the final window, i.e. the quantity that rule (ii) compares with
#'     \code{rel_eps}. When the series stopped by rule (i) it can be larger
#'     than \code{rel_eps}.}
#'   \item{err_rel_est}{Estimate of the relative truncation error of the raw
#'     partial sum (not of the extrapolated value that is reported), based on a
#'     random-phase argument; in our tests it tends to underestimate the error.}
#'   \item{n_eff}{Effective number of support points implied by the size of the
#'     late terms; used by \code{err_rel_est}.}
#'   \item{nu_max_ratio}{Largest size of the late transform values relative to
#'     the first one; values near 1 mean the terms are not decaying (a lattice-like
#'     statistic).}
#' }
#' The last four components (\code{err_rel_ext}, \code{err_rel_est},
#' \code{n_eff} and \code{nu_max_ratio}) are diagnostics of the series; they are heuristics,
#' not error bounds. The names \code{Gamma} and \code{Time_gamma} used by earlier
#' versions still work, with a deprecation warning.
#'
#' @references
#' Subiabre, F. and Thraves, C. Efficient computation of \pvalue{}s for multinomial
#' tests. Manuscript.
#'
#' Cressie, N. and Read, T. R. C. (1984). Multinomial goodness-of-fit tests.
#' \emph{Journal of the Royal Statistical Society, Series B}, 46(3), 440--464.
#'
#' @seealso \code{\link{pval_exact}} for the exact \pvalue{},
#'   \code{\link{pval_flexible}} to choose between the two automatically.
#'
#' @examples
#' p <- c(0.1, 0.2, 0.3, 0.4)
#' x <- c(35, 30, 60, 75)   # N = 200
#'
#' # Log-likelihood ratio
#' fit <- pval_fourier(x, p, stat = "llr")
#' fit
#' fit$pval                 # 0.00688
#' fit$status               # 0: stopping rule met
#' fit$n_terms              # 1208
#'
#' # Probability mass function (exact multinomial test)
#' fit_pmf <- pval_fourier(x, p, stat = "pmf")
#' fit_pmf$pval             # 0.00577
#' fit_pmf$status           # 0: stopping rule met
#'
#' # Power divergence: lambda is required
#' fit_pd <- pval_fourier(x, p, stat = "pd", lambda = 2/3)
#' fit_pd$pval              # 0.00384
#' fit_pd$status            # 0: stopping rule met
#' @export
pval_fourier <- function(x,
                         p = rep(1/length(x), length(x)),
                         stat = c("llr", "pd", "chi2", "pmf"),
                         lambda = NULL,
                         rescale_p = FALSE,
                         undersampling = 1,
                         rel_eps = 0.001,
                         B = 50,
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
  xp <- validate_x_p(x, p, rescale_p)
  x <- xp$x
  p <- xp$p
  if (undersampling <= 0) stop("'undersampling' must be strictly positive.")
  if (rel_eps < 0) stop("'rel_eps' must be non-negative.")
  if (B < 0) stop("'B' must be an integer greater than or equal to 0.")
  if (max_terms <= 0) stop("'max_terms' must be a positive integer.")
  if (avg_window < 0 || avg_window > 1) stop("'avg_window' must be between 0 and 1.")

  st <- resolve_stat_lambda(stat, lambda)
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

#' Exact multinomial test p-value
#'
#' Computes the exact \pvalue{} \eqn{P\{T(Y) \ge T(x)\}}{P{T(Y) >= T(x)}},
#' \eqn{Y \sim \mathrm{Multinomial}(N, p)}{Y ~ Multinomial(N, p)}, of a multinomial goodness-of-fit
#' test, up to floating-point rounding. Instead of enumerating the support, it
#' prunes whole groups of outcomes using bounds on the statistic (see Details).
#' \code{pval_exhaustive()} is a deprecated alias.
#'
#' @inheritParams pval_fourier
#' @param n_threads Number of threads (positive integer, capped at the number of
#'   available cores). Default \code{min(2L, parallel::detectCores())}.
#' @param max_time Optional time limit in seconds. \code{NULL} (default) means no
#'   limit. The limit is only checked between blocks of values of the first
#'   category, so a run can exceed it by a wide margin; if it is reached,
#'   \code{pval} is \code{NA} and \code{status} is 1.
#' @param engine \code{"speedup"} (default): branch pruning with bounds from a
#'   dynamic program, described in Details. \code{"standard"}: the earlier
#'   algorithm, which enumerates the first \eqn{K-2} categories and solves the last
#'   two by bisection.
#' @param verbose Logical; if \code{TRUE}, prints progress information. Default
#'   \code{FALSE}.
#' @param ... Advanced options passed to the C engine. Unknown names produce a
#'   warning.
#'
#' @details
#' The statistics are those of \code{\link{pval_fourier}}. Because the statistic
#' is additive, a dynamic program over the \eqn{(N+1)K} states (category, trials
#' left) first computes the exact minimum and maximum that the remaining
#' categories can add to it, at cost \eqn{O(KN^2)}. Counts are then assigned one
#' category at a time, in decreasing order of probability; whenever the bounds
#' show that every completion of a partial assignment is (or none is) at least as
#' extreme as the observed data, its whole probability is added (or it is
#' discarded) without visiting it. Only partial assignments near the boundary
#' \eqn{T(y) = T(x)} are expanded.
#'
#' The cost still grows quickly with the number of categories \eqn{K}. As a rough
#' guide, in our tests with the log-likelihood ratio on 8 threads, instances with
#' \eqn{K \le 6}{K <= 6} and \eqn{N \le 1000}{N <= 1000} took under five minutes, while \eqn{K \ge 8}{K >= 8}
#' with \eqn{N \ge 500}{N >= 500} did not finish within an hour. For large instances use
#' \code{\link{pval_fourier}}.
#'
#' @return An object of class \code{"multfourier"}: a list with components
#' \describe{
#'   \item{x, p}{The input counts and null probabilities (\code{p} after
#'     rescaling, if \code{rescale_p = TRUE}).}
#'   \item{pval}{The exact \pvalue{}, up to floating-point rounding; \code{NA} if
#'     \code{max_time} was reached.}
#'   \item{stat}{Name of the test statistic, e.g. \code{"Log-Likelihood Ratio"}
#'     or \code{"power-divergence (lambda = 0.6667)"}.}
#'   \item{method}{\code{"exact"}.}
#'   \item{status}{\code{0}: the computation finished; \code{1}:
#'     \code{max_time} was reached before it finished.}
#'   \item{message}{Human-readable version of \code{status}. For \code{status = 0}
#'     it reads \code{"Converged"}, a label shared with \code{\link{pval_fourier}}.}
#'   \item{time}{Total wall-clock time in seconds.}
#' }
#' The components that describe the Fourier series are also present, so that
#' results of both methods have the same structure, but are always \code{NA}:
#' \code{gamma}, \code{time_gamma}, \code{W}, \code{n_terms}, \code{err_rel_ext},
#' \code{err_rel_est}, \code{n_eff} and \code{nu_max_ratio}.
#'
#' @seealso \code{\link{pval_fourier}}, \code{\link{pval_flexible}}.
#'
#' @examples
#' p <- c(0.1, 0.2, 0.3, 0.4)
#' x <- c(35, 30, 60, 75)   # N = 200
#'
#' # Log-likelihood ratio
#' fit <- pval_exact(x, p, stat = "llr")
#' fit
#' fit$pval                 # 0.00684
#' fit$status               # 0: computation finished
#'
#' # Probability mass function (exact multinomial test)
#' fit_pmf <- pval_exact(x, p, stat = "pmf")
#' fit_pmf$pval             # 0.00578
#'
#' # Pearson's chi-square
#' fit_chi2 <- pval_exact(x, p, stat = "chi2")
#' fit_chi2$pval            # 0.00294
#' @export
pval_exact <- function(x,
                       p = rep(1/length(x), length(x)),
                       stat = c("llr", "pd", "chi2", "pmf"),
                       lambda = NULL,
                       rescale_p = FALSE,
                       n_threads = min(2L, parallel::detectCores()),
                       max_time = NULL,
                       engine = c("speedup", "standard"),
                       verbose = FALSE,
                       ...) {
  xp <- validate_x_p(x, p, rescale_p)
  x <- xp$x
  p <- xp$p

  st <- resolve_stat_lambda(stat, lambda)
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

  build_multfourier_s3(raw, x, p, elapsed, "exact", st$label)
}

#' @rdname pval_exact
#' @export
pval_exhaustive <- function(...) {
  .Deprecated("pval_exact")
  pval_exact(...)
}

#' Multinomial test p-value, choosing the method automatically
#'
#' Computes the \pvalue{} with \code{\link{pval_exact}} when the instance is small
#' and with \code{\link{pval_fourier}} otherwise.
#'
#' @inheritParams pval_fourier
#' @param enum_cutoff Threshold on \eqn{\log_{10}}{log10} of the support size
#'   \eqn{{N+K-1 \choose K-1}}{choose(N+K-1, K-1)} (the number of possible count vectors). Below it the
#'   exact method is used, at or above it the Fourier method. Default 10. See
#'   Details.
#' @param n_threads Number of threads, passed to whichever method is chosen.
#'   Default \code{min(2L, parallel::detectCores())}.
#' @param max_time Optional time limit in seconds, passed to the chosen method.
#' @param engine \code{"speedup"} (default) or \code{"standard"}, passed to the
#'   chosen method.
#'
#' @details
#' The cost of \code{\link{pval_exact}} depends strongly on the observed counts,
#' not only on the support size, and it is highest far in the tail, where small
#' \pvalue{}s are computed. With counts drawn from the null hypothesis it takes
#' about one second on 2 threads at a support of \eqn{10^{12}}{10^12} to \eqn{10^{13}}{10^13}
#' outcomes (log-likelihood ratio, \eqn{K = 5, 6, 7}); with counts far in the
#' tail it can take tens of seconds at a support of \eqn{10^{11}}{10^11} (for example
#' \eqn{K = 6}, \eqn{N = 400}, \eqn{p \approx 10^{-33}}{p ~ 10^-33}). The default
#' \code{enum_cutoff = 10} is a simple, conservative value that keeps the exact
#' method for instances where it is fast in both situations.
#'
#' When the Fourier method is chosen and does not meet its stopping rule
#' (\code{status != 0}), the result is returned as is; there is no automatic
#' fallback to the exact method.
#'
#' @return An object of class \code{"multfourier"}, as described in
#' \code{\link{pval_fourier}}; its \code{method} component (\code{"exact"} or
#' \code{"fourier"}) shows which method was used.
#'
#' @seealso \code{\link{pval_fourier}}, \code{\link{pval_exact}}.
#'
#' @examples
#' # Small instance (K = 4, N = 200, about 10^6 outcomes): exact method
#' p <- c(0.1, 0.2, 0.3, 0.4)
#' x <- c(35, 30, 60, 75)
#' fit <- pval_flexible(x, p, stat = "llr")
#' fit
#' fit$method               # "exact"
#' fit$pval                 # 0.00684
#' fit$status               # 0: computation finished
#'
#' # Large instance (K = 8, N = 2000, about 10^19 outcomes): Fourier method
#' x_large <- c(240, 260, 250, 270, 230, 250, 245, 255)
#' fit_large <- pval_flexible(x_large, stat = "llr")
#' fit_large$method         # "fourier"
#' fit_large$pval           # 0.761
#' fit_large$status         # 0: stopping rule met
#' @export
pval_flexible <- function(x,
                          p = rep(1/length(x), length(x)),
                          stat = c("llr", "pd", "chi2", "pmf"),
                          lambda = NULL,
                          rescale_p = FALSE,
                          enum_cutoff = 10,
                          undersampling = 1,
                          rel_eps = 0.001,
                          B = 50,
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
  xp <- validate_x_p(x, p, rescale_p)
  x <- xp$x
  p <- xp$p
  if (undersampling <= 0) stop("'undersampling' must be strictly positive.")
  if (rel_eps < 0) stop("'rel_eps' must be non-negative.")
  if (B < 0) stop("'B' must be an integer greater than or equal to 0.")
  if (max_terms <= 0) stop("'max_terms' must be a positive integer.")
  if (avg_window < 0 || avg_window > 1) stop("'avg_window' must be between 0 and 1.")

  st <- resolve_stat_lambda(stat, lambda)
  threads_val <- resolve_n_threads(n_threads)
  precision <- match.arg(precision)
  engine <- match.arg(engine)

  opts <- list(
    method = "flexible",
    stat = st$stat,
    lambda = st$lambda,
    enum_cutoff = as.numeric(enum_cutoff),
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

#' Print method for multfourier objects
#'
#' Prints the method, statistic, \pvalue{}, status and total runtime and, for the
#' Fourier method, also the tilt \code{gamma}, the time spent choosing it, the
#' period half-width \code{W} and the number of terms.
#'
#' @param x Object of class \code{"multfourier"} returned by
#'   \code{\link{pval_fourier}}, \code{\link{pval_exact}} or
#'   \code{\link{pval_flexible}}.
#' @param ... Additional arguments (currently unused).
#'
#' @return Invisibly returns \code{x}.
#'
#' @examples
#' p <- c(0.1, 0.2, 0.3, 0.4)
#' x <- c(35, 30, 60, 75)
#'
#' # Fourier method: also shows gamma, W and the number of terms
#' fit_fourier <- pval_fourier(x, p, stat = "llr")
#' fit_fourier              # same as print(fit_fourier)
#'
#' # Exact method
#' fit_exact <- pval_exact(x, p, stat = "llr")
#' fit_exact
#' @export
print.multfourier <- function(x, ...) {
  cat("Multinomial Test (MultFourier)\n")
  cat("------------------------------\n")
  cat(sprintf("%-11s: %s\n", "Method", x$method))
  cat(sprintf("%-11s: %s\n", "Statistic", x$stat))
  cat(sprintf("%-11s: %s\n", "p-value", format(x$pval, digits = 8)))
  cat(sprintf("%-11s: %s\n", "Status", paste0(x$status, " (", x$message, ")")))
  if (!is.null(x$gamma) && !is.na(x$gamma)) {
    cat(sprintf("%-11s: %s\n", "Gamma", format(x$gamma, digits = 6)))
  }
  if (!is.null(x$time_gamma) && !is.na(x$time_gamma)) {
    cat(sprintf("%-11s: %s seconds\n", "Time gamma", format(x$time_gamma, digits = 4)))
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