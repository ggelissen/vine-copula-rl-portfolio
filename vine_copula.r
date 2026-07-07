# ============================================================
# vine_copula.r
# Fit static vine copula to uniform residuals, simulate returns,
# and compute portfolio moments
# ============================================================

library(rvinecopulib)
library(xts)

load("data/marginal_results.RData")
source("load_data.r")

# Fit a static D-vine to residuals.
# Select structures using AIC and choosing from (Gauss,t,Clayton,Gumbel,Frank,Joe).
d <- ncol(U)
vine_fit <- vinecop(
  data = U,
  var_types = rep("c", d),
  structure = dvine_structure(1:d),
  family_set = c("gaussian", "t", "clayton", "gumbel", "frank", "joe"),
  selcrit = "aic"
)

# Output vine structure
summary(vine_fit)
save(vine_fit, file = "data/vine_fit.RData")


# Simulate from vine structure
n_sim <- 10000
sim_U <- rvinecop(n_sim, vine_fit)
n_sim <- nrow(sim_U)
d <- ncol(sim_U)

sim_log_returns <- matrix(0, nrow = n_sim, ncol = d)
colnames(sim_log_returns) <- asset_names
orig_returns <- returns

for (i in 1:d) {
  name <- asset_names[i]
  model <- marginals[[name]]

  # create inverse empirical CDF
  nz <- length(model$z_sorted)
  prob_grid <- (1:nz) / (nz + 1)

  z_sim <- approx(x = prob_grid, y = model$z_sorted, xout = sim_U[, i],
                   rule = 2, ties = "ordered"
                  )$y
  
  # extract ARMA-GARCH coefficients
  cfit <- coef(model$fit)
  mu <- cfit["mu"]            
  ar1 <- cfit["ar1"]         
  omega <- cfit["omega"]
  alpha <- cfit["alpha1"]
  beta <- cfit["beta1"]

  # determine parameters for moment computation

  # AR(1)
  # E[mu] = mu / (1 - phi)
  if (abs(ar1) < 1) {
    mu_uncond <- mu / (1 - ar1)
  } else {
    mu_uncond <- mean(as.numeric(returns[, i]))
  }
  # GARCH
  # E[sigma^2] = omega / (1 - alpha - beta)
  if (alpha + beta < 1) {
    sigma2_uncond <- omega / (1 - alpha - beta)
  } else {
    sigma2_uncond <- var(as.numeric(returns[, i]))
  }
  sigma_uncond <- sqrt(sigma2_uncond)

  # simulate log returns
  sim_log_returns[, i] <- mu_uncond + sigma_uncond * z_sim
}

# compute vine moments
vine_mu <- colMeans(sim_log_returns)
vine_cov <- cov(sim_log_returns)

cat("\nVine‑implied mean log returns:\n")
print(round(vine_mu, 6))
cat("\nVine‑implied covariance matrix:\n")
print(round(vine_cov, 6))

# compute empirical moments
emp_mu <- colMeans(orig_returns)
emp_cov <- cov(orig_returns)

cat("\nEmpirical mean log returns:\n")
print(round(emp_mu, 6))
cat("\nEmpirical covariance matrix:\n")
print(round(emp_cov, 6))


# run Sharpe ratio comparison with Li-Ng framework
rf <- 0
w_vine <- solve(vine_cov, vine_mu - rf)
w_vine <- w_vine / sum(w_vine)

w_emp <- solve(emp_cov, emp_mu - rf)
w_emp <- w_emp / sum(w_emp)

sr_vine <- (t(w_vine) %*% vine_mu) / sqrt(t(w_vine) %*% vine_cov %*% w_vine)
sr_emp  <- (t(w_emp) %*% emp_mu)  / sqrt(t(w_emp) %*% emp_cov %*% w_emp)

cat(sprintf("\nVine‑based Sharpe ratio: %.4f\n", sr_vine))
cat(sprintf("Empirical Sharpe ratio: %.4f\n", sr_emp))