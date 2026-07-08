# =================================================================================
# pre_commitment.r
# Replicated Li & Ng (2000) pre-commitment policy for multi-period mean-variance portfolio selection
# =================================================================================

source("load_data.r")

compute_policy <- function(returns, w0, strat_param, strat) {
  
  T <- length(returns)         # number of time periods
  d <- ncol(returns[[1]]) - 1  # number of risky assets

  inv_E_ee_list <- vector("list", T)
  mean_e_list <- vector("list", T)
  E_r0e_list <- vector("list", T)

  A1 <- numeric(T)
  A2 <- numeric(T)
  B <- numeric(T)

  # Loop over periods
  for (t in 1:T) {
    returns_t <- returns[[t]]    # returns at time t
    N <- nrow(returns_t)         # number of returns at time t

    r0 <- returns_t[,1]          # reference asset returns
    e <- returns_t[,-1] - r0     # excess returns

    mean_r0 <- mean(r0)                         # expectation of reference asset
    mean_e <- colMeans(e)                       # expectation of risky assets
    E_ee <- (t(e) %*% e) / N                    # raw second moment
    E_r0e <- colMeans(r0 * e)                   # cross moment
    E2_r0 <- mean(r0^2)                         # second moment of reference asset
    inv_E_ee <- solve(E_ee + diag(1e-9,d))      # use diag to ensure invertibility

    inv_E_ee_list[[t]] <- inv_E_ee
    mean_e_list[[t]] <- mean_e
    E_r0e_list[[t]] <- E_r0e

    B[t] <- as.numeric(t(mean_e) %*% inv_E_ee %*% mean_e)    # multivariate squared Sharpe Ratio
    A1[t] <- mean_r0 - t(mean_e) %*% inv_E_ee %*% E_r0e      # expected return of 'hedged' reference asset
    A2[t] <- E2_r0 - t(E_r0e) %*% inv_E_ee %*% E_r0e         # second moment of residual
  }

  # Compute tail products
  tail_prod_A1 <- numeric(T)
  tail_prod_A2 <- numeric(T)
  tail_prod_A1[T] <- 1
  tail_prod_A2[T] <- 1

  if (T > 1) {
    for (t in (T-1):1) {
      tail_prod_A1[t] <- tail_prod_A1[t+1] * A1[t+1]
      tail_prod_A2[t] <- tail_prod_A2[t+1] * A2[t+1]
    }
  }

  # Calculate B1, B2 and aggregates
  B1 <- B * tail_prod_A1 / (2 * tail_prod_A2)
  B2 <- B * (tail_prod_A1 / (2 * tail_prod_A2))^2

  mu <- prod(A1)                   # multiplicative drift
  nu <- sum(tail_prod_A1 * B1)     # total speculative capacity
  tau <- prod(A2)                  # total second-moment multiplier

  a <- (nu / 2) - nu^2              # curvature of efficient frontier
  b <- (mu * nu) / a                # scaling coefficient
  c <- tau - mu^2 - (a * b^2)       # unavoidable risk

  if (a <= 0) warning("a <= 0: mean-variance frontier may be ill-conditioned.")
  
  # Determine optimal portfolio policy
  if (strat == "E") omega <- strat_param
  if (strat == "P1") omega <- nu / (2 * sqrt(a * (strat_param - c * w0^2)))
  if (strat == "P2") omega <- nu^2 / (2 * a * (strat_param - (mu + b * nu) * w0))
  
  spec_scalar <- 0.5 * (b * w0 + nu / (2 * omega * a))
  spec_mult <- tail_prod_A1 / tail_prod_A2

  policy <- vector("list", T)

  for (t in 1:T) {
    inv_E_ee <- inv_E_ee_list[[t]]
    mean_e <- mean_e_list[[t]]
    E_r0e <- E_r0e_list[[t]]

    Kt <- inv_E_ee %*% E_r0e                                   # hedging coefficient (indepedent of risk strategy)
    vt <- spec_scalar * spec_mult[t] * (inv_E_ee %*% mean_e)   # speculative demand (scales with risk tolerance)

    policy[[t]] <- list(Kt = Kt, vt = vt)
   }

  # Return outputs
  return(list(
    global = list(mu = mu, nu = nu, tau = tau, a = a, b = b, c = c),
    policy = policy,
    per_period = list(A1 = A1, A2 = A2, B = B, B1 = B1, B2 = B2)
  ))
}


run_simulation <- function(returns, policy, w0, seed=123) {
  set.seed(seed)

  T <- length(returns)         # number of time periods
  d <- ncol(returns[[1]]) - 1  # number of risky assets

  wealth <- numeric(T + 1)
  wealth[1] <- w0

  allocations <- list()
  ref_asset <- numeric(T)      # amount in reference asset

  for (t in 1:T) {
    # obtain hedging term and speculative term
    Kt <- policy[[t]]$Kt
    vt <- policy[[t]]$vt

    # compute allocations to risky and reference assets
    ut <- -Kt * wealth[t] + vt
    #ut <- pmax(pmin(ut / wealth[t], 1.0), -0.5) * wealth[t] # constraint on shorting
    ref_asset[t] <- wealth[t] - sum(ut)

    # ensure ref_asset is non-negative (no borrowing)
    if (ref_asset[t] < 0) {
      ut <- ut * (wealth[t] / sum(ut)) * 0.95   # scale to 95%
      ref_asset[t] <- wealth[t] - sum(ut)
    }

    # sample a return scenario for this period
    ret <- returns[[t]]
    idx <- sample(nrow(ret), 1)
    ret0 <- ret[idx, 1]
    ret_risk <- ret[idx, -1]

    # update wealth
    wealth[t+1] <- ret0 * ref_asset[t] + sum(ret_risk * ut)

    allocations[[t]] <- list(
      u_risk = ut,
      u_ref = ref_asset[t],
      returns = c(ret0, ret_risk)
    )
  }

  return(list(
    wealth = wealth,
    allocations = allocations,
    ref_asset = ref_asset
  ))
}


# ==============================================================================================

if (sys.nframe() == 0) {
  source("load_data.r")
  
  set.seed(123)
  
  T <- 12
  L <- 138
  freq <- "monthly"
  w0 <- 100000
  gamma <- 2
  strategy <- "E"
  
  returns <- load_returns()
  data <- preprocess_returns(returns, ref_col = 7, L = L, T = T, freq = freq)
  returns_list <- data$returns_list
  
  result <- compute_policy(returns_list, w0, gamma, strategy)
  sim <- run_simulation(returns_list, result$policy, w0)
  
  cat("\n========================================\n")
  cat("Li & Ng (2000) — Real Data Test\n")
  cat("========================================\n")
  cat(sprintf("Horizon: %d periods\n", T))
  cat(sprintf("Lookback: %d days per scenario set\n", L))
  cat(sprintf("Initial wealth: %.0f\n", w0))
  cat(sprintf("Risk aversion (gamma): %.1f\n", gamma))
  cat(sprintf("Strategy: %s\n\n", strategy))
  
  cat("Global parameters:\n")
  cat(sprintf("  mu (drift): %.6f\n", result$global$mu))
  cat(sprintf("  nu (speculative capacity): %.6f\n", result$global$nu))
  cat(sprintf("  tau (second-moment multiplier): %.6f\n", result$global$tau))
  
  cat("\nPeriod-by-period:\n")
  for (t in 1:T) {
    cat(sprintf("\nPeriod %d:\n", t))
    cat(sprintf("  Wealth start: %.2f\n", sim$wealth[t]))
    cat(sprintf("  Risky weights: %s\n",
                paste(round(sim$allocations[[t]]$u_risk / sim$wealth[t], 4), collapse = ", ")))
    cat(sprintf("  Reference weight (GOLD): %.4f\n",
                sim$allocations[[t]]$u_ref / sim$wealth[t]))
    cat(sprintf("  Wealth end: %.2f\n", sim$wealth[t+1]))
  }
  
  cat(sprintf("\nFinal wealth: %.2f\n", sim$wealth[T+1]))
  cat(sprintf("Total return: %.2f%%\n", 100 * (sim$wealth[T+1]/w0 - 1)))
}