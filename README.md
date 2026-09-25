# MultFourier

`MultFourier` is an R package that computes p-values of multinomial
goodness-of-fit tests,

$$
p = P\{T(Y) \ge T(x)\}, \qquad Y \sim \mathrm{Multinomial}(N, p_1, \ldots, p_K),
$$

for additive test statistics $T$: the log-likelihood ratio $G^2$, Pearson's
$X^2$, the Cressie–Read power divergence, and the exact multinomial test
(ordering outcomes by their probability). It is aimed at small p-values, far in
the tail, and at instances whose sample space is too large to sum over.

The package has three functions:

| Function | Method |
|---|---|
| `pval_fourier()` | Writes the p-value as a Fourier series whose terms are values of the moment-generating function of the statistic, and sums it until an adaptive stopping rule is met. Its cost grows slowly with $N$ and $K$. |
| `pval_exact()` | Exact p-value by branch pruning with bounds from a dynamic program, without enumerating the sample space. Feasible for a moderate number of categories. |
| `pval_flexible()` | Chooses between the two according to the size of the sample space. |

## Installation

```r
# install.packages("remotes")
remotes::install_github("PabRvss/MultFourier-v2", ref = "speedup", build_vignettes = TRUE)
```

The package contains C code, so it needs the standard tools for building R
packages from source: Xcode Command Line Tools on macOS, Rtools on Windows, or
a C compiler on Linux.

## Example

```r
library(MultFourier)

p <- c(0.1, 0.2, 0.3, 0.4)   # null probabilities
x <- c(35, 30, 60, 75)       # observed counts, N = 200

fit <- pval_fourier(x, p, stat = "llr")
fit$pval     # 0.00688
fit$status   # 0: the stopping rule was met

pval_exact(x, p, stat = "llr")$pval   # 0.00684

# Power divergence: lambda is required
pval_fourier(x, p, stat = "pd", lambda = 2/3)$pval
```

All three functions return an object of class `"multfourier"`, a list with the
p-value, the status, the running time and, for the Fourier method, the details
of the series (see `?pval_fourier`).

## Things to keep in mind

- **Check `status`.** The Fourier series is truncated by a stopping rule, so
  `pval_fourier()` returns an approximation. `status = 0` means the stopping
  rule was met; any other value means the result is not reliable.
- **The statistic matters.** For the log-likelihood ratio (`"llr"`) and the
  probability ordering (`"pmf"`) the series typically converges in a few hundred
  to a few thousand terms. For Pearson's $X^2$ (`"chi2"`), and often for the
  power divergence unless $\lambda$ is close to 0, it may not converge at all.
  In that case use `pval_exact()` when the instance is small enough.
- **`rel_eps` is not an error bound.** It sets how stable the partial sums must
  be before stopping (default `1e-3`); the actual relative error can be larger.

## Documentation

- Reference manual: `MultFourier.pdf` in this repository, or `help(package = "MultFourier")`.
- Worked example with election data: `vignette("polling_tables_multfourier", package = "MultFourier")`.

## Reference

Subiabre, F. and Thraves, C. *Efficient computation of p-values for multinomial
tests*. Manuscript.

## Authors

Charles Thraves, Pablo Rivas (maintainer), Felipe Subiabre.
