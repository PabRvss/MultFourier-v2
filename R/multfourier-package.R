#' MultFourier: Efficient Computation of Multinomial Test p-values
#'
#' @details
#' The \code{MultFourier} package provides efficient methods for computing p-values
#' in multinomial tests, including both exact and approximate approaches. The exact
#' p-value is computed using an exhaustive algorithm, which is recommended for small
#' datasets. For larger datasets, an approximation is provided using a Fourier series
#' algorithm, which is computationally efficient and provides control over precision
#' through parameters such as \code{max_terms}, \code{rel_eps}, and \code{undersampling}.
#'
#' The package includes the following methods:
#' \itemize{
#'   \item \code{pval_exhaustive}: Computes the exact p-value by enumerating all possible
#'     outcomes. Suitable for small datasets.
#'   \item \code{pval_fourier}: Approximates the p-value using a Fourier series
#'     expansion. Efficient for large datasets.
#'   \item \code{pval_flexible}: Automatically selects between the exhaustive and Fourier
#'     series methods based on the dataset size.
#' }
#'
#' Key features:
#' \itemize{
#'   \item Control over precision through parameters such as \code{max_terms},
#'     \code{rel_eps}, and \code{undersampling}.
#'   \item Support for multiple test statistics, including logP, Pearson's chi-square,
#'     and log-likelihood ratio.
#'   \item Verbose output for debugging and analysis.
#' }
#'
#' @references
#' \itemize{
#'   \item Thraves, C. and Subiabre, F. et al. (Year): "Title of the Paper". Journal Name, Volume(Issue), Pages.
#' }
#'
#' @seealso
#' \itemize{
#'   \item Report bugs at \url{https://github.com/PabRvss/MultFourier/issues}
#' }
#'
#' @name MultFourier-package
#' @aliases MultFourier
#' @keywords package
"_PACKAGE"
