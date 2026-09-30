#' MultFourier: p-values of multinomial goodness-of-fit tests
#'
#' @details
#' \code{MultFourier} computes the \pvalue{} \eqn{P\{T(Y) \ge T(x)\}}{P{T(Y) >= T(x)}} of a
#' multinomial goodness-of-fit test, where \eqn{x} is the observation (the vector
#' of observed counts in the \eqn{K} categories, with \eqn{N = \sum_k x_k}{N = sum_k x_k}
#' trials), \eqn{Y \sim \mathrm{Multinomial}(N, p)}{Y ~ Multinomial(N, p)} under the null hypothesis
#' with category probabilities \eqn{p}, and \eqn{T} is an additive statistic: the
#' log-likelihood ratio \eqn{G^2}, Pearson's \eqn{\mathcal{X}^2}{X^2}, the
#' Cressie--Read power divergence, or the probability mass function of the
#' multinomial (exact multinomial test). It is aimed at small \pvalue{}s, far in
#' the tail, and at instances where the support of the multinomial is too large
#' to sum over.
#'
#' The package provides three functions:
#' \itemize{
#'   \item \code{\link{pval_fourier}}: writes the \pvalue{} as a Fourier series whose
#'     terms are values of the moment-generating function of the statistic, and
#'     sums it until an adaptive stopping rule is met. Its cost grows slowly with
#'     \eqn{N} and \eqn{K}; its accuracy depends on the statistic (fast convergence
#'     for \eqn{G^2} and the probability mass function, possibly none for
#'     \eqn{\mathcal{X}^2}{X^2}).
#'   \item \code{\link{pval_exact}}: exact \pvalue{} by branch pruning with bounds
#'     from a dynamic program, without enumerating the support. Feasible for a
#'     moderate number of categories and trials.
#'   \item \code{\link{pval_flexible}}: chooses between the two according to the
#'     size of the support.
#' }
#' All three return an object of class \code{"multfourier"}; check its
#' \code{status} component before using the \pvalue{} of \code{pval_fourier}.
#'
#' @references
#' Subiabre, F. and Thraves, C. Efficient computation of \pvalue{}s for multinomial
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
