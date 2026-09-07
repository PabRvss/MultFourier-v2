# Helper interno para construir el objeto S3 multfourier
build_multfourier_s3 <- function(raw, x, p, elapsed, requested_method, max_time = NULL) {
  method_name <- if (requested_method == "flexible") {
    if (!is.null(raw$method) && raw$method == 1L) "fourier" else "exhaustive"
  } else {
    requested_method
  }
  
  gamma_val <- if (method_name == "fourier") as.numeric(raw$gamma) else NA_real_
  n_terms_val <- if (method_name == "fourier") as.integer(raw$terms) else NA_integer_
  
  # Determinacion de status y mensaje:
  # 0: Converged.
  # 1: Maximum time reached.
  # 2: Maximum number of terms reached.
  # 3: Could not solve the optimization of gamma.
  if (method_name == "fourier" && (is.na(gamma_val) || is.nan(gamma_val) || is.infinite(gamma_val))) {
    status_id <- 3L
    status_msg <- "Could not solve the optimization of gamma."
  } else if (!is.null(max_time) && is.finite(max_time) && elapsed >= max_time && !isTRUE(raw$converged == 1L)) {
    status_id <- 1L
    status_msg <- "Maximum time reached."
  } else if (isTRUE(raw$converged == 1L)) {
    status_id <- 0L
    status_msg <- "Converged."
  } else if (method_name == "fourier") {
    status_id <- 2L
    status_msg <- "Maximum number of terms reached."
  } else {
    status_id <- 0L
    status_msg <- "Converged."
  }
  
  out <- list(
    x = as.numeric(x),
    p = as.numeric(p),
    pval = as.numeric(raw$pval),
    gamma = gamma_val,
    n_terms = n_terms_val,
    time = as.numeric(elapsed),
    p0 = as.numeric(raw$p0),
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

#' Calculo de p-valor mediante metodo flexible
#'
#' Evalua si la dimension del soporte multinomial permite enumeracion exacta
#' (biseccion iterativa) o si deriva al metodo de inversion de Fourier.
#'
#' @param x Vector de conteos observados en cada categoria.
#' @param p Vector de probabilidades teoricas bajo la hipotesis nula (suman 1).
#' @param precision Precision aritmetica: "double" (estandar) o "double-double" (alta precision).
#' @param precompute Logico; si TRUE, usa precomputo de transformadas para aceleracion.
#' @param rel_eps Tolerancia relativa de convergencia (tau >= 0, default 1e-3).
#' @param max_terms Numero maximo de armonicos a evaluar en la serie (max_L, entero positivo, default 10000).
#' @param B Entero >= 0; cantidad de terminos consecutivos requeridos para convergencia (default 30). Si B = 0, no se aplica parada temprana.
#' @param threads Cantidad de hilos de ejecucion (maximo 2 bajo politicas CRAN).
#' @param verbose Logico; si TRUE, imprime informacion del progreso.
#' @param tau Alias opcional para rel_eps.
#' @param max_L Alias opcional para max_terms.
#' @param max_time Tiempo maximo de ejecucion en segundos (opcional).
#'
#' @return Objeto S3 de clase \code{"multfourier"} con los atributos x, p, pval, gamma, n_terms, time, p0, status, message y method.
#' @useDynLib MultFourier, .registration = TRUE
#' @export
pval_flexible <- function(x, p = rep(1/length(x), length(x)), 
                          precision = c("double", "double-double"),
                          precompute = TRUE,
                          rel_eps = 1e-3, 
                          max_terms = 10000, 
                          B = 30,
                          threads = 2,
                          verbose = FALSE,
                          tau = NULL,
                          max_L = NULL,
                          max_time = NULL) {
  if (length(x) != length(p)) stop("Los vectores 'x' y 'p' deben tener la misma longitud.")
  if (any(x < 0) || any(p <= 0)) stop("Conteos no negativos y probabilidades positivas.")
  
  if (!is.null(tau)) rel_eps <- tau
  if (!is.null(max_L)) max_terms <- max_L

  if (rel_eps < 0) stop("'rel_eps' (o 'tau') debe ser mayor o igual que 0.")
  if (max_terms <= 0) stop("'max_terms' (o 'max_L') debe ser un entero positivo.")
  if (B < 0) stop("'B' debe ser un entero mayor o igual que 0.")

  precision <- match.arg(precision)
  prec_code <- if (precision == "double-double") 2L else 1L
  
  t0 <- proc.time()
  raw <- .Call(c_pval_flexible, 
               as.double(x), 
               as.double(p), 
               as.integer(prec_code), 
               as.logical(precompute), 
               as.numeric(rel_eps), 
               as.integer(max_terms), 
               as.integer(B),
               as.integer(threads), 
               as.logical(verbose))
  elapsed <- (proc.time() - t0)[["elapsed"]]
  
  build_multfourier_s3(raw, x, p, elapsed, "flexible", max_time)
}

#' Calculo exacto de p-valor mediante biseccion iterativa
#'
#' @param x Vector de conteos observados.
#' @param p Vector de probabilidades teoricas bajo la hipotesis nula.
#' @param threads Cantidad de hilos a usar (maximo 2).
#' @param verbose Logico; si TRUE, imprime informacion del proceso.
#' @param max_time Tiempo maximo de ejecucion en segundos (opcional).
#' @param ... Parametros adicionales ignorados (compatibilidad).
#'
#' @return Objeto S3 de clase \code{"multfourier"} con los atributos x, p, pval, gamma, n_terms, time, p0, status, message y method.
#' @export
pval_exhaustive <- function(x, p = rep(1/length(x), length(x)), threads = 1, verbose = FALSE, max_time = NULL, ...) {
  if (length(x) != length(p)) stop("Los vectores 'x' y 'p' deben tener la misma longitud.")
  if (any(x < 0) || any(p <= 0)) stop("Conteos no negativos y probabilidades positivas.")
  
  t0 <- proc.time()
  raw <- .Call(c_pval_exhaustive, 
               as.double(x), 
               as.double(p), 
               as.integer(threads), 
               as.logical(verbose))
  elapsed <- (proc.time() - t0)[["elapsed"]]
  
  build_multfourier_s3(raw, x, p, elapsed, "exhaustive", max_time)
}

#' Calculo de p-valor mediante inversion de series de Fourier
#'
#' @inheritParams pval_flexible
#' @return Objeto S3 de clase \code{"multfourier"} con los atributos x, p, pval, gamma, n_terms, time, p0, status, message y method.
#' @export
pval_fourier <- function(x, p = rep(1/length(x), length(x)), 
                         precision = c("double", "double-double"),
                         precompute = TRUE,
                         rel_eps = 1e-3, 
                         max_terms = 10000, 
                         B = 30,
                         threads = 2,
                         verbose = FALSE,
                         tau = NULL,
                         max_L = NULL,
                         max_time = NULL) {
  if (length(x) != length(p)) stop("Los vectores 'x' y 'p' deben tener la misma longitud.")
  if (any(x < 0) || any(p <= 0)) stop("Conteos no negativos y probabilidades positivas.")
  
  if (!is.null(tau)) rel_eps <- tau
  if (!is.null(max_L)) max_terms <- max_L

  if (rel_eps < 0) stop("'rel_eps' (o 'tau') debe ser mayor o igual que 0.")
  if (max_terms <= 0) stop("'max_terms' (o 'max_L') debe ser un entero positivo.")
  if (B < 0) stop("'B' debe ser un entero mayor o igual que 0.")

  precision <- match.arg(precision)
  prec_code <- if (precision == "double-double") 2L else 1L
  
  t0 <- proc.time()
  raw <- .Call(c_pval_fourier, 
               as.double(x), 
               as.double(p), 
               as.integer(prec_code), 
               as.logical(precompute), 
               as.numeric(rel_eps), 
               as.integer(max_terms), 
               as.integer(B),
               as.integer(threads), 
               as.logical(verbose))
  elapsed <- (proc.time() - t0)[["elapsed"]]
  
  build_multfourier_s3(raw, x, p, elapsed, "fourier", max_time)
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
  cat("Method:   ", x$method, "\n")
  cat("P-value:  ", format(x$pval, digits = 8), "\n")
  cat("Status:   ", x$status, paste0("(", x$message, ")"), "\n")
  if (!is.na(x$gamma)) {
    cat("Gamma:    ", format(x$gamma, digits = 6), "\n")
  }
  if (!is.na(x$n_terms)) {
    cat("Terms:    ", x$n_terms, "\n")
  }
  cat("P0:       ", format(x$p0, digits = 8), "\n")
  cat("Time:     ", format(x$time, digits = 4), "seconds\n")
  invisible(x)
}