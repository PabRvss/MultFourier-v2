#' Compute p-value for a Multinomial Test
#'
#' Calcula el p-value para un test Multinomial exacto, Chi-Cuadrado de
#' Pearson, razon de verosimilitud logaritmica, o de Power-Divergence.
#' Usa enumeracion completa si la cardinalidad del soporte de la
#' distribucion Multinomial es menor a 10^6; en caso contrario, usa el
#' metodo de Fourier.
#'
#' @param x Vector entero con las realizaciones de cada categoria.
#' @param p Vector numerico con las probabilidades de cada categoria (no
#'   negativas y que sumen uno). Debe tener el mismo largo que `x`.
#' @param stat Nombre del estadistico a calcular: "prob" (default),
#'   "pearson", "llr" o "power_div".
#' @param lambda Numerico con el valor lambda del estadistico de
#'   Power-Divergence. Solo se usa si `stat = "power_div"`.
#' @param max_time Tiempo maximo en segundos. Por defecto 600.
#' @param max_terms Numero maximo de terminos de la serie de Fourier. Por
#'   defecto 300.
#' @param rel_eps Tolerancia de error relativo. Por defecto 0.001.
#' @param undersampling Factor de submuestreo. Por defecto 1.
#' @param verbose Logico; si `TRUE`, imprime resultados intermedios.
#'
#' @return Un objeto `list` con los elementos `pval`, `converged`, `terms`
#'   y `p0`.
#'
#' @export
#' @useDynLib MultFourier, .registration = TRUE
#'
#' @examples
#' probs <- c(0.00040161, 0.00080321, 0.00200803, 0.00401606, 0.00682731,
#'            0.01044177, 0.01485944, 0.02008032, 0.02610442, 0.03293173,
#'            0.04056225, 0.04899598, 0.05823293, 0.06827309, 0.07911647,
#'            0.09076305, 0.10321285, 0.11646586, 0.13052209, 0.14538153)
#' x0 <- rep(10, length(probs))
#' result <- pval_flexible(x0, probs, max_terms = 300, verbose = TRUE)
#' print(result)
pval_flexible <- function(x, p, stat = "prob", lambda = 0, max_time = 600,
                           max_terms = 300, rel_eps = 0.001, undersampling = 1,
                           verbose = FALSE) {
  if (length(x) != length(p)) {
    stop("The lengths of 'x' and 'p' must be equal.")
  }

  N <- sum(x)
  K <- length(x)
  verbose_num <- if (isTRUE(verbose)) 1 else 0

  .Call(c_pval_flexible, as.double(N), as.double(K), as.double(x),
        as.double(p), as.double(max_terms), as.double(rel_eps),
        as.double(undersampling), as.double(verbose_num))
}

#' Compute p-value using an exhaustive method
#'
#' Calcula el p-value evaluando todos los elementos del soporte de la
#' distribucion Multinomial. Recomendado solo para instancias pequenas.
#'
#' @inheritParams pval_flexible
#'
#' @return Un objeto `list` con los elementos `pval`, `converged`, `terms`
#'   y `p0`.
#'
#' @export
#'
#' @examples
#' probs <- c(0.1, 0.2, 0.3, 0.4)
#' x0 <- rep(10, 4)
#' result <- pval_exhaustive(x0, probs, verbose = TRUE)
#' print(result)
pval_exhaustive <- function(x, p, stat = "prob", lambda = 0, max_time = 600,
                             verbose = FALSE) {
  if (length(x) != length(p)) {
    stop("The lengths of 'x' and 'p' must be equal.")
  }

  N <- sum(x)
  K <- length(x)
  verbose_num <- if (isTRUE(verbose)) 1 else 0

  .Call(c_pval_exhaustive, as.double(N), as.double(K), as.double(x),
        as.double(p), as.double(max_time), as.double(verbose_num))
}

#' Compute p-value using the Fourier method
#'
#' Calcula el p-value usando exclusivamente el metodo de series de
#' Fourier, sin importar el tamano del soporte de la distribucion.
#'
#' @inheritParams pval_flexible
#'
#' @return Un objeto `list` con los elementos `pval`, `converged`, `terms`
#'   y `p0`.
#'
#' @export
#'
#' @examples
#' probs <- c(0.00040161, 0.00080321, 0.00200803, 0.00401606, 0.00682731,
#'            0.01044177, 0.01485944, 0.02008032, 0.02610442, 0.03293173,
#'            0.04056225, 0.04899598, 0.05823293, 0.06827309, 0.07911647,
#'            0.09076305, 0.10321285, 0.11646586, 0.13052209, 0.14538153)
#' x0 <- rep(10, length(probs))
#' result <- pval_fourier(x0, probs, max_terms = 300, verbose = TRUE)
#' print(result)
pval_fourier <- function(x, p, stat = "prob", lambda = 0, max_time = 600,
                          max_terms = 300, rel_eps = 0.001, undersampling = 1,
                          verbose = FALSE) {
  if (length(x) != length(p)) {
    stop("The lengths of 'x' and 'p' must be equal.")
  }

  N <- sum(x)
  K <- length(x)
  verbose_num <- if (isTRUE(verbose)) 1 else 0

  .Call(c_pval_fourier, as.double(N), as.double(K), as.double(x),
        as.double(p), as.double(max_terms), as.double(rel_eps),
        as.double(undersampling), as.double(verbose_num))
}