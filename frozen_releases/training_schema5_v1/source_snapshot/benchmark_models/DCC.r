# ============================================================================
# DCC.r
# DCC‑GARCH(1,1) estimation and simulation
# ============================================================================

library(rmgarch)
library(mvtnorm)

# Fit DCC on a window of returns
fit_DCC <- function(ret_window, distribution = c("norm", "sstd"), seed = 42) {
  distribution <- match.arg(distribution)
  set.seed(seed)
  
  # Univariate GARCH(1,1) spec for each asset
  uspec <- ugarchspec(
    mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    distribution.model = if (distribution == "norm") "norm" else "sstd"
  )
  n_assets <- ncol(ret_window)
  uspec_list <- replicate(n_assets, uspec, simplify = FALSE)
  
  # DCC(1,1) spec
  dcc_spec <- dccspec(
    uspec = multispec(uspec_list),
    dccOrder = c(1, 1),
    distribution = if (distribution == "norm") "mvnorm" else "mvt"
  )
  
  dccfit(dcc_spec, data = ret_window, fit.control = list(eval.se = FALSE))
}

# Simulate one‑step ahead from a fitted DCC
simulate_DCC <- function(dcc_fit, n_sim = 10000, seed = 42) {
  set.seed(seed)
  fcst <- dccforecast(dcc_fit, n.ahead = 1)
  
  # Extract mean vector
  mu <- as.numeric(fcst@mforecast$mu)
  
  # Extract covariance matrix — it's stored as an array, take the first slice
  H_array <- fcst@mforecast$H[[1]]
  if (is.array(H_array) && length(dim(H_array)) == 3) {
    H <- matrix(H_array[, , 1], nrow = dim(H_array)[1], ncol = dim(H_array)[2])
  } else if (is.matrix(H_array)) {
    H <- H_array
  } else {
    stop("Unexpected format for covariance forecast")
  }
  
  # Simulate from multivariate normal
  sim_log <- mvtnorm::rmvnorm(n_sim, mean = mu, sigma = H)
  exp(sim_log)
}