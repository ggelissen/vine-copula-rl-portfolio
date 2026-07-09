# ====================================================================
# marginals.r
# Process raw asset returns by fitting to ARMA-GARCH.
# Extract standardised residuals and transform to Unif() for copulas.
# ====================================================================

library(rugarch)
library(xts)

RUN_TESTS <- FALSE

source ("load_data.r")
raw_returns <- load_returns()
asset_names <- colnames(returns)


# Fit marginal for a single asset
fit_marginal <- function(return_series, asset_name = "asset") {
  # Specify the ARMA-GARCH model
  spec <- ugarchspec(
                    mean.model = list(armaOrder = c(1,0), include.mean = TRUE),
                    variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
                    distribution.model = "sstd"
                    )

  # Fit the model to the asset data
  fit <- ugarchfit(spec = spec, data = return_series, solver = "hybrid")

  # Extract residuals and transform to Unif(0,1)
  z <- as.numeric(residuals(fit, standardize = TRUE))
  z_sorted <- sort(z)  # store for later
  u <- rank(z) / (length(z) + 1)                                         # use empirical CDF (ranks)
  # u <- pdist(distribution = "sstd", q = z, mu = 0, sigma = 1,          # instead of parametric skewed-t
  #            skew = coef(fit)["skew"], shape = coef(fit)["shape"])     # since skewed-t gave imperfect results.

  cat(sprintf("\n=== %s ===\n", asset_name))
  cat(sprintf("Mean(z)=%.4f  Var(z)=%.4f\n", mean(z), var(z)))
  cat(sprintf("LB p-val (z): %.4f   LB p-val (z^2): %.4f\n",
              Box.test(z, lag = 10, type = "Ljung-Box")$p.value,
              Box.test(z^2, lag = 10, type = "Ljung-Box")$p.value))
  
  return(list(fit = fit, z = z, u = u, z_sorted = z_sorted))
}

# =========================================================================================

if (RUN_TESTS) {
  # Call function to fit all marginals
  marginals <- list()
  U <- NULL

  for (i in seq_along(asset_names)) {
    name <- asset_names[i]
    return <- as.numeric(returns[,i])
    marginals[[name]] <- fit_marginal(return, name)
    
    if (is.null(U)) {
      U <- marginals[[name]]$u
    } else {
      U <- cbind(U, marginals[[name]]$u)
    }
  }

  colnames(U) <- asset_names
  save(marginals, U, asset_names, file = "data/marginal_results.RData")
  print("Data saved to marginal_results.RData")
}

#################################
# Note (07/07/2026):
# All assets pass the Ljung‑Box test at lag 10 except DIVIDEND 
# (p=0.024 at lag 10 for squared residuals). As a robustness 
# check I also use lag 5 where DIVIDEND marginally fails, and 
# the main results are qualitatively unchanged.
#################################