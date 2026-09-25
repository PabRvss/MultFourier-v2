#' MultFourier: p-values of multinomial goodness-of-fit tests
#'
#' @details
#' \code{MultFourier} computes the p-value \eqn{P\{T(Y) \ge T(x)\}},
#' \eqn{Y \sim \mathrm{Multinomial}(N, p)}, of goodness-of-fit tests based on an
#' additive statistic \eqn{T}: the log-likelihood ratio \eqn{G^2}, Pearson's
#' \eqn{X^2}, the Cressie--Read power divergence, and the exact multinomial test
#' (ordering by probability). It is aimed at small p-values, far in the tail, and
#' at instances where the support of the multinomial is too large to sum over.
#'
#' The package provides three functions:
#' \itemize{
#'   \item \code{\link{pval_fourier}}: writes the p-value as a Fourier series whose
#'     terms are values of the moment-generating function of the statistic, and
#'     sums it until an adaptive stopping rule is met. Its cost grows slowly with
#'     \eqn{N} and \eqn{K}; its accuracy depends on the statistic (fast convergence
#'     for \eqn{G^2} and the probability ordering, possibly none for \eqn{X^2}).
#'   \item \code{\link{pval_exact}}: exact p-value by branch pruning with bounds
#'     from a dynamic program, without enumerating the support. Feasible for a
#'     moderate number of categories.
#'   \item \code{\link{pval_flexible}}: chooses between the two according to the
#'     size of the support.
#' }
#' All three return an object of class \code{"multfourier"}; check its
#' \code{status} component before using the p-value of \code{pval_fourier}.
#'
#' @references
#' Subiabre, F. and Thraves, C. Efficient computation of p-values for multinomial
#' tests. Manuscript.
#'
#' @seealso
#' Report bugs at \url{https://github.com/PabRvss/MultFourier-v2/issues}.
#'
#' @name MultFourier-package
#' @aliases MultFourier
#' @keywords package
#' @useDynLib MultFourier, c_run_multfourier
"_PACKAGE"
