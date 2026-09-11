library(MultFourier)

# Datos de prueba
x <- c(10, 20, 30)
p <- c(0.2, 0.3, 0.5)

# 1. Caso base (fourier default: pd con lambda = 1)
fit <- pval_fourier(x, p)
print(fit)

# 2. Distintos estadísticos
fit_chi2 <- pval_fourier(x, p, stat = "chi2")
fit_llr  <- pval_fourier(x, p, stat = "llr")
fit_pmf  <- pval_fourier(x, p, stat = "pmf")
fit_pd   <- pval_fourier(x, p, stat = "pd", lambda = 2/3)

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
