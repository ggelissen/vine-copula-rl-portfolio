# ============================================================

stop("Legacy standalone rolling-vine script is disabled: it leaks full-sample marginals and mismatches monthly horizons. Use the common research protocol.")
# dynamic_vine_rolling_window.r
# Rolling‑window D‑vine + Li‑Ng multi‑period mean–variance
# ============================================================

library(rvinecopulib)
library(xts)
source("helper/load_data.r")
source("benchmark_models/Li_Ng.r")
load("data/marginal_results.RData")
returns <- load_returns()

L <- 500                  # lookback window (days)
T_horizon <- 12           # number of rebalancing periods (months)
rebal_freq <- "months"    # rebalance monthly
n_sim <- 10000            # number of simulations per period
gamma <- 2                # risk aversion
w0 <- 100000              # initial wealth

# Find rebalancing dates with at least L days of history
all_dates <- index(returns)
valid_dates <- all_dates[L:length(all_dates)]
rebal_dates <- endpoints(returns[L:nrow(returns)], on = rebal_freq)
rebal_dates <- index(returns)[rebal_dates + L - 1]   # align with full index
rebal_dates <- tail(rebal_dates, T_horizon)
cat(sprintf("Rebalancing dates: %s\n", paste(rebal_dates, collapse = ", ")))

# Store the wealth and weights over time
weights_history <- vector("list", T_horizon)
wealth <- numeric(T_horizon + 1)
wealth[1] <- w0

# Backtesting loop
for (t in 1:T_horizon) {
  current_date <- rebal_dates[t]
  cat(sprintf("\n=== Period %d: %s ===\n", t, current_date))
  
  # Identify lookback window of daily data
  window_end <- which(index(returns) == current_date)
  window_start <- window_end - L + 1
  U_window <- U[window_start:window_end, ]
  
  # Fit static D‑vine
  vine_current <- vinecop(
    U_window,
    var_types = rep("c", ncol(U_window)),
    structure = dvine_structure(1:ncol(U_window)),
    family_set = c("gaussian", "t", "clayton", "gumbel", "frank", "joe"),
    selcrit = "aic"
  )
  
  # Simulate uniforms
  sim_U <- rvinecop(n_sim, vine_current)
  
  # Back‑transform to log returns
  sim_log_returns <- matrix(0, n_sim, ncol(sim_U))
  for (i in 1:ncol(sim_U)) {
    name <- asset_names[i]
    model <- marginals[[name]]
    prob_grid <- seq_len(length(model$z_sorted)) / (length(model$z_sorted) + 1)
    z_sim <- approx(prob_grid, model$z_sorted, xout = sim_U[, i], rule = 2)$y
    # unconditional mean and variance from AR(1)-GARCH
    cfit <- coef(model$fit)
    mu <- cfit["mu"]; ar1 <- cfit["ar1"]
    omega <- cfit["omega"]; alpha <- cfit["alpha1"]; beta <- cfit["beta1"]
    mu_uncond <- if (abs(ar1) < 1) mu / (1 - ar1) else mean(as.numeric(returns[, i]))
    sigma2 <- if (alpha + beta < 1) omega / (1 - alpha - beta) else var(as.numeric(returns[, i]))
    sigma <- sqrt(sigma2)
    sim_log_returns[, i] <- mu_uncond + sigma * z_sim
  }
  colnames(sim_log_returns) <- asset_names
  
  # Compute conditional moments
  cond_mean <- colMeans(sim_log_returns)
  cond_cov <- cov(sim_log_returns)
  
  # Build Li‑Ng compatible returns list (constant moments)
  T_rem <- T_horizon - t + 1   # periods remaining
  returns_list <- replicate(T_rem, sim_log_returns, simplify = FALSE)
  ref_col <- 7
  risk_cols <- setdiff(1:ncol(sim_log_returns), ref_col)

  for (j in 1:T_rem) {
    # convert log returns to simple returns
    ret_mat <- exp(sim_log_returns)
    returns_list[[j]] <- cbind(ret_mat[, ref_col], ret_mat[, risk_cols])
  }
  
  # Compute policy and take the weight
  policy_result <- compute_policy(returns_list, wealth[t], gamma, "E")

  ut <- policy_result$policy[[1]]$vt - policy_result$policy[[1]]$Kt * wealth[t]
  weights_history[[t]] <- ut / wealth[t]
  
  # Move to next month: compute return from the true data
  if (t < T_horizon) {
    next_date <- rebal_dates[t+1]
  } else {
    next_date <- index(returns)[nrow(returns)]
  }

  actual_returns <- returns[paste0(current_date+1, "/", next_date)]
  total_return <- exp(colSums(actual_returns)) 
  portf_return <- total_return[ref_col] * (1 - sum(weights_history[[t]])) +
    sum(total_return[risk_cols] * weights_history[[t]])
  wealth[t+1] <- wealth[t] * portf_return
  
  cat(sprintf("Wealth: %.2f, Weight vector: %s\n", wealth[t+1],
              paste(round(weights_history[[t]], 4), collapse = ", ")))
}

# Print performance metrics
final_wealth <- wealth[T_horizon+1]
total_return <- (final_wealth / w0 - 1) * 100
annual_return <- (final_wealth / w0)^(1/(T_horizon/12)) - 1   # assuming monthly horizon
cat(sprintf("\nFinal wealth: %.2f (%.2f%%)\n", final_wealth, total_return))
cat(sprintf("Annualised return: %.2f%%\n", annual_return*100))
