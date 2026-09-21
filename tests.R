library(MultFourier)

# Datos de prueba
x <- c(10, 20, 30)
p <- c(0.2, 0.3, 0.5)

# 1. Caso base (fourier default: llr)
fit <- pval_fourier(x, p)
print(fit)

# 2. Distintos estadísticos
fit_chi2 <- pval_fourier(x, p, stat = "chi2")
fit_llr  <- pval_fourier(x, p, stat = "llr")
fit_pmf  <- pval_fourier(x, p, stat = "pmf")
fit_pd   <- pval_fourier(x, p, stat = "pd", lambda = 2/3)

# 2.1 Warning cuando se suministra lambda para stat != 'pd'
fit_warn <- pval_fourier(x, p, stat = "chi2", lambda = 1.5)

# 3. Límite de términos alcanzado (status = 2)
fit_terms <- pval_fourier(x, p, max_terms = 10, rel_eps = 1e-12)
print(fit_terms)

# 4. Límite de tiempo alcanzado (status = 1)
fit_time <- pval_fourier(x, p, max_time = 0.0001, max_terms = 1e6)
print(fit_time)

# 5. Promedio de cola en ventana (plano vs ponderado)
fit_w_flat <- pval_fourier(x, p, avg_window = 0.2, avg_flat = TRUE)
fit_w_pond <- pval_fourier(x, p, avg_window = 0.2, avg_flat = FALSE)
c(sin_ventana = fit$pval, plana = fit_w_flat$pval, ponderada = fit_w_pond$pval)

# 6. Variación de undersampling
fit_u05 <- pval_fourier(x, p, undersampling = 0.5)
fit_u15 <- pval_fourier(x, p, undersampling = 1.5)
c(W_05 = fit_u05$W, W_10 = fit$W, W_15 = fit_u15$W) #Valores de W para diferente undersampling

# 7. Ejecución con múltiples hilos
fit_th <- pval_fourier(x, p, n_threads = 4)
print(fit_th)

# 8. Bisección exhaustiva y método flexible
x_small <- c(3, 2, 4)
p_small <- c(0.3, 0.3, 0.4)
fit_ex <- pval_exhaustive(x_small, p_small)
fit_fl <- pval_flexible(x_small, p_small)

# 9. Manejo de inestabilidad/explosión numérica (status = 4 sin crash de R)
fit_bad <- pval_fourier(x, p, stat = "pd", lambda = -1)
print(fit_bad)

# 10. Tests de la vignette (Escuela Centenario y leave-one-out)
votes <- matrix(
  c(
     96, 101, 173,  94, 106,  88,  96, 104,  88,  98, 109,  92, 104, 57, 50,
     37,  34,  31,  34,  37,  37,  35,  32,  40,  37,  37,  38,  36, 22, 25,
     39,  37,  36,  36,  26,  29,  28,  39,  39,  34,  28,  32,  35, 15, 27,
     99, 107, 110, 108, 127, 107, 108, 125,  89, 100, 109, 103,  98, 74, 70,
     46,  37,  42,  41,  34,  37,  45,  26,  62,  45,  48,  49,  41, 21, 21,
     83,  84,   8,  87,  70, 102,  88,  74,  82,  86,  69,  86,  86, 49, 44
  ),
  nrow = 6, byrow = TRUE,
  dimnames = list(
    c("Abstention", "Candidate A", "Candidate B", "Candidate C", "Null/blank", "Candidate D"),
    as.character(276:290)
  )
)

loo_p <- sapply(seq_len(ncol(votes)), function(m) {
  other <- rowSums(votes[, -m, drop = FALSE])
  other / sum(other)
})
colnames(loo_p) <- colnames(votes)

stopifnot(all(abs(colSums(loo_p) - 1) < 1e-12))
stopifnot(all(loo_p >= 0))

fit_vignette_278 <- pval_fourier(votes[, "278"], loo_p[, "278"], stat = "llr")
stopifnot(fit_vignette_278$status == 0L)
stopifnot(fit_vignette_278$pval < 1e-25)

fit_vignette_276 <- pval_fourier(votes[, "276"], loo_p[, "276"], stat = "llr")
stopifnot(fit_vignette_276$status == 0L)
stopifnot(fit_vignette_276$pval > 0.5)
cat("Tests de la vignette pasaron exitosamente.\n")
