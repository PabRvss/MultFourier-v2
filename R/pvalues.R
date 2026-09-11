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
  if (!is.null(raw$n_eff)) attr(out, "n_eff") <- as.numeric(raw$n_eff)
  if (!is.null(raw$nu_max_ratio)) attr(out, "nu_max_ratio") <- as.numeric(raw$nu_max_ratio)
  
  out
}

# Helper interno para resolver y validar el estadistico y lambda
resolve_stat_lambda <- function(stat, lambda) {
  stat <- match.arg(stat, c("pd", "chi2", "llr", "pmf"))
  if (stat == "chi2") {
    lambda_val <- 1.0
    stat_label <- "chi2"
  } else if (stat == "llr") {
    lambda_val <- 0.0
    stat_label <- "llr"
  } else if (stat == "pmf") {
    lambda_val <- NaN
    stat_label <- "pmf"
  } else { # "pd"
    if (length(lambda) != 1L || !is.numeric(lambda) || !is.finite(lambda)) {
      stop("'lambda' debe ser un numero real finito.")
    }
    lambda_val <- as.numeric(lambda)
    stat_label <- sprintf("pd (lambda = %s)", format(lambda_val))
  }
  list(stat = stat, lambda = lambda_val, label = stat_label)
}

# Helper interno para acotar n_threads a los nucleos disponibles
resolve_n_threads <- function(n_threads) {
  max_cores <- parallel::detectCores()
  if (is.na(max_cores) || max_cores < 1L) max_cores <- 1L
  n_threads <- as.integer(n_threads)
  if (is.na(n_threads) || n_threads < 1L) {
    stop("'n_threads' debe ser un entero positivo mayor o igual a 1.")
  }
  min(n_threads, max_cores)
}

#' Calculo de p-valor mediante metodo flexible
#'
#' Evalua si la dimension del soporte multinomial permite enumeracion exacta
#' (biseccion iterativa) o si deriva al metodo de inversion de Fourier.
#'
#' @param x Vector de conteos observados en cada categoria.
#' @param p Vector de probabilidades teoricas bajo la hipotesis nula (suman 1).
#' @param stat Estadistico de prueba: \code{"pd"} (Power Divergence), \code{"chi2"} (Chi-cuadrado), \code{"llr"} (Log-Likelihood Ratio), o \code{"pmf"} (Probabilidad multinomial puntual). Por defecto \code{"pd"}.
#' @param lambda Parametro real para Power Divergence cuando \code{stat = "pd"}. Por defecto 1.
#' @param undersampling Factor de submuestreo (> 0) para el periodo en la serie de Fourier. Por defecto 1.
#' @param avg_window Fraccion de sumas parciales finales a promediar (entre 0 y 1). Por defecto 0.
#' @param avg_flat Logico; si TRUE usa promedio plano; si FALSE pondera linealmente segun proximidad al final. Por defecto FALSE.
#' @param rel_eps Tolerancia relativa de convergencia (>= 0). Por defecto 1e-3.
#' @param max_terms Numero maximo de armonicos a evaluar en la serie (entero positivo). Por defecto 10000.
#' @param B Entero >= 0; cantidad de terminos consecutivos requeridos para convergencia. Por defecto 30.
#' @param n_threads Cantidad de hilos de ejecucion (entero positivo >= 1). Por defecto \code{min(2L, parallel::detectCores())}.
#' @param precision Precision aritmetica: "double" (estandar) o "double-double" (alta precision).
#' @param precompute Logico; si TRUE, usa precomputo de transformadas para aceleracion.
#' @param max_time Tiempo maximo de ejecucion en segundos (opcional).
#' @param verbose Logico; si TRUE, imprime informacion del progreso.
#' @param ... Parametros adicionales internos pasados al motor de C.
#' @return Returns an S3 object of class \code{"multfourier"} with the following attributes:
#' \describe{
#'   \item{x}{The input vector of observed realizations for each category.}
#'   \item{p}{The input vector of probabilities for each category under the null hypothesis.}
#'   \item{pval}{The computed p-value.}
#'   \item{Time_gamma}{The time spent optimizing gamma in seconds.}
#'   \item{W}{The effective width of the distribution support interval.}
#'   \item{stat}{A string describing the test statistic used (e.g. \code{"chi2"}, \code{"llr"}, \code{"pmf"}, or \code{"pd (lambda = ...)"}).}
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
pval_flexible <- function(x, p = rep(1/length(x), length(x)),
                          stat = c("pd", "chi2", "llr", "pmf"),
                          lambda = 1,
                          undersampling = 1,
                          avg_window = 0,
                          avg_flat = FALSE,
                          rel_eps = 1e-3,
                          max_terms = 10000,
                          B = 30,
                          n_threads = min(2L, parallel::detectCores()),
                          precision = c("double", "double-double"),
                          precompute = TRUE,
                          max_time = NULL,
                          verbose = FALSE,
                          ...) {
  if (length(x) != length(p)) stop("Los vectores 'x' y 'p' deben tener la misma longitud.")
  if (any(x < 0) || any(p <= 0)) stop("Conteos no negativos y probabilidades positivas.")
  if (undersampling <= 0) stop("'undersampling' debe ser estrictamente mayor a cero.")
  if (avg_window < 0 || avg_window > 1) stop("'avg_window' debe estar entre 0 y 1.")
  if (rel_eps < 0) stop("'rel_eps' debe ser mayor o igual que 0.")
  if (max_terms <= 0) stop("'max_terms' debe ser un entero positivo.")
  if (B < 0) stop("'B' debe ser un entero mayor o igual que 0.")

  st <- resolve_stat_lambda(stat, lambda)
  threads_val <- resolve_n_threads(n_threads)
  precision <- match.arg(precision)

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
    max_time = if (!is.null(max_time) && is.finite(max_time)) as.numeric(max_time) else -1.0,
    verbose = as.logical(verbose),
    ...
  )

  t0 <- proc.time()
  raw <- .Call(c_run_multfourier, as.double(x), as.double(p), opts)
  elapsed <- (proc.time() - t0)[["elapsed"]]

  build_multfourier_s3(raw, x, p, elapsed, "flexible", st$label)
}

#' Calculo exacto de p-valor mediante biseccion iterativa
#'
#' @param x Vector de conteos observados.
#' @param p Vector de probabilidades teoricas bajo la hipotesis nula.
#' @param stat Estadistico de prueba: \code{"pd"}, \code{"chi2"}, \code{"llr"}, o \code{"pmf"}. Por defecto \code{"pd"}.
#' @param lambda Parametro real para Power Divergence cuando \code{stat = "pd"}. Por defecto 1.
#' @param n_threads Cantidad de hilos a usar (entero >= 1). Por defecto 1.
#' @param max_time Tiempo maximo de ejecucion en segundos (opcional).
#' @param verbose Logico; si TRUE, imprime informacion del proceso.
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
pval_exhaustive <- function(x, p = rep(1/length(x), length(x)),
                            stat = c("pd", "chi2", "llr", "pmf"),
                            lambda = 1,
                            n_threads = 1,
                            max_time = NULL,
                            verbose = FALSE,
                            ...) {
  if (length(x) != length(p)) stop("Los vectores 'x' y 'p' deben tener la misma longitud.")
  if (any(x < 0) || any(p <= 0)) stop("Conteos no negativos y probabilidades positivas.")

  st <- resolve_stat_lambda(stat, lambda)
  threads_val <- resolve_n_threads(n_threads)

  opts <- list(
    method = "exhaustive",
    stat = st$stat,
    lambda = st$lambda,
    n_threads = as.integer(threads_val),
    max_time = if (!is.null(max_time) && is.finite(max_time)) as.numeric(max_time) else -1.0,
    verbose = as.logical(verbose),
    ...
  )

  t0 <- proc.time()
  raw <- .Call(c_run_multfourier, as.double(x), as.double(p), opts)
  elapsed <- (proc.time() - t0)[["elapsed"]]

  build_multfourier_s3(raw, x, p, elapsed, "exhaustive", st$label)
}

#' Calculo de p-valor mediante inversion de series de Fourier
#'
#' @inheritParams pval_flexible
#' @return Returns an S3 object of class \code{"multfourier"} with the following attributes:
#' \describe{
#'   \item{x}{The input vector of observed realizations for each category.}
#'   \item{p}{The input vector of probabilities for each category under the null hypothesis.}
#'   \item{pval}{The computed p-value.}
#'   \item{Time_gamma}{The time spent optimizing gamma in seconds.}
#'   \item{W}{The effective width of the distribution support interval.}
#'   \item{stat}{A string describing the test statistic used (e.g. \code{"chi2"}, \code{"llr"}, \code{"pmf"}, or \code{"pd (lambda = ...)"}).}
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
pval_fourier <- function(x, p = rep(1/length(x), length(x)),
                         stat = c("pd", "chi2", "llr", "pmf"),
                         lambda = 1,
                         undersampling = 1,
                         avg_window = 0,
                         avg_flat = FALSE,
                         rel_eps = 1e-3,
                         max_terms = 10000,
                         B = 30,
                         n_threads = min(2L, parallel::detectCores()),
                         precision = c("double", "double-double"),
                         precompute = TRUE,
                         max_time = NULL,
                         verbose = FALSE,
                         ...) {
  if (length(x) != length(p)) stop("Los vectores 'x' y 'p' deben tener la misma longitud.")
  if (any(x < 0) || any(p <= 0)) stop("Conteos no negativos y probabilidades positivas.")
  if (undersampling <= 0) stop("'undersampling' debe ser estrictamente mayor a cero.")
  if (avg_window < 0 || avg_window > 1) stop("'avg_window' debe estar entre 0 y 1.")
  if (rel_eps < 0) stop("'rel_eps' debe ser mayor o igual que 0.")
  if (max_terms <= 0) stop("'max_terms' debe ser un entero positivo.")
  if (B < 0) stop("'B' debe ser un entero mayor o igual que 0.")

  st <- resolve_stat_lambda(stat, lambda)
  threads_val <- resolve_n_threads(n_threads)
  precision <- match.arg(precision)

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
    max_time = if (!is.null(max_time) && is.finite(max_time)) as.numeric(max_time) else -1.0,
    verbose = as.logical(verbose),
    ...
  )

  t0 <- proc.time()
  raw <- .Call(c_run_multfourier, as.double(x), as.double(p), opts)
  elapsed <- (proc.time() - t0)[["elapsed"]]

  build_multfourier_s3(raw, x, p, elapsed, "fourier", st$label)
}

#' Metodo print para objetos de clase multfourier
#'
#' @param x Objeto devuelto por pval_fourier, pval_exhaustive o pval_flexible.
#' @param ... Argumentos adicionales no utilizados.
#'
#' @return Retorna el objeto de forma invisible.
#' @export
print.multfourier <- function(x, ...) {
  cat("Multinomial Test (MultFourier)\n")
  cat("------------------------------\n")
  cat("Method:      ", x$method, "\n")
  cat("Statistic:   ", x$stat, "\n")
  cat("p-value:     ", format(x$pval, digits = 8), "\n")
  cat("Status:      ", x$status, paste0("(", x$message, ")"), "\n")
  if (!is.null(x$Gamma) && !is.na(x$Gamma)) {
    cat("Gamma:       ", format(x$Gamma, digits = 6), "\n")
  }
  if (!is.null(x$Time_gamma) && !is.na(x$Time_gamma)) {
    cat("Time gamma:  ", format(x$Time_gamma, digits = 4), "seconds\n")
  }
  if (!is.null(x$W) && !is.na(x$W)) {
    cat("W:           ", format(x$W, digits = 6), "\n")
  }
  if (!is.null(x$n_terms) && !is.na(x$n_terms)) {
    cat("Terms:       ", x$n_terms, "\n")
  }
  cat("Time:        ", sprintf("%.6f", x$time), "seconds\n")
  invisible(x)
}