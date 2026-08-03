# ==============================================================================
# expected_utility_single.r
# Dynamic portfolio selection with CRRA utility and rolling‑window D‑vine copula
# ==============================================================================

library(rvinecopulib)

source("helper/timer.r")

RUN_TESTS <- FALSE

crra_utility <- function(W, gamma) {
  if (gamma == 1) return(log(pmax(W, 1e-10)))
  pmax(W, 1e-10)^(1 - gamma) / (1 - gamma)
}

expected_utility <- function(W, gamma) {
  mean(crra_utility(W, gamma))
}


# Vine-based return simulation function
VineReturnSimulator <- function(marginals, asset_names, ref_col) {
  cdf_grids <- list()
  for (i in seq_along(asset_names)) {
    name <- asset_names[i]
    z_sorted <- marginals[[name]]$z_sorted
    n <- length(z_sorted)
    prob_grid <- seq_len(n) / (n + 1)
    cdf_grids[[name]] <- list(prob = prob_grid, z = z_sorted)
  }

  list(
    marginals = marginals,
    asset_names     = asset_names,
    ref_col         = ref_col,
    risk_cols       = setdiff(seq_along(asset_names), ref_col),
    cdf_grids       = cdf_grids,
    
    # Simulate one‑period‑ahead returns
    simulate_returns = function(vine_fit, n_sim = 10000, cores = 1L, prev_returns = NULL) {
      cores <- max(1L, as.integer(cores))
      sim_U <- rvinecop(n_sim, vine_fit, cores = cores)
      sim_log <- matrix(0, n_sim, length(asset_names))
      
      for (i in seq_along(asset_names)) {
        name <- asset_names[i]
        grid <- cdf_grids[[name]]
        z_sim <- approx(grid$prob, grid$z, xout = sim_U[, i], rule = 2)$y
        
        # Conditional mean for the first row, unconditional for the rest
        if (!is.null(prev_returns) && length(prev_returns) == length(asset_names)) {
          cond_mean <- marginals[[name]]$mu_ar + marginals[[name]]$ar1 * prev_returns[i]
          mu_vec <- rep(marginals[[name]]$mu_uncond, n_sim)
          mu_vec[1] <- cond_mean
        } else {
          mu_vec <- rep(marginals[[name]]$mu_uncond, n_sim)
        }
        sigma <- marginals[[name]]$sigma_uncond
        sim_log[, i] <- mu_vec + sigma * z_sim
      }
      
      colnames(sim_log) <- asset_names
      list(log = sim_log, gross = exp(sim_log))
    },
    
    # Simulate terminal wealth over multiple periods
    simulate_terminal_wealth = function(vine_fits, w0, weights_seq, 
                                         n_paths = 10000) {
      T_periods <- length(weights_seq)
      W <- rep(w0, n_paths)
      
      for (t in seq_len(T_periods)) {
        sim <- simulate_returns(vine_fits[[t]], n_sim = n_paths)
        R <- sim$gross
        
        R_ref  <- R[, ref_col]
        R_risk <- R[, risk_cols, drop = FALSE]
        excess <- R_risk - R_ref
        
        portf_return <- R_ref + as.vector(excess %*% weights_seq[[t]])
        W <- W * portf_return
      }
      
      W
    }
  )
}


# Expected utility portfolio optimizer (single-period)
optimise_eu_portfolio <- function(simulator, vine_fit, W0, gamma, n_sim = 10000) {
  sim    <- simulator$simulate_returns(vine_fit, n_sim)
  R      <- sim$gross
  R_ref  <- R[, simulator$ref_col]
  R_risk <- R[, simulator$risk_cols, drop = FALSE]
  d      <- ncol(R_risk)
  
  obj <- function(w) {
    # Penalise violations of constraints
    if (any(w < 0) || sum(w) > 1) return(1e10)
    w_full <- c(1 - sum(w), w)
    R_all  <- cbind(R_ref, R_risk)
    W1 <- W0 * as.vector(R_all %*% w_full)
    -expected_utility(W1, gamma)
  }

  w0 <- rep(1 / (d + 1), d)
  opt <- optim(w0, obj, method = "L-BFGS-B",
               lower = rep(0, d), upper = rep(1, d),
               control = list(maxit = 500, factr = 1e-10))
  opt$par
}


run_eu_backtest <- function(simulator, vine_fits, returns_xts, rebal_dates,
                             W0 = 100000, gamma = 2, n_sim = 10000) {
  timer <- start_timer("Single‑period EU")
  T_horizon <- length(rebal_dates)
  wealth <- numeric(T_horizon + 1)
  wealth[1] <- W0
  weights_history <- vector("list", T_horizon)
  
  for (t in seq_len(T_horizon)) {
    w_opt <- optimise_eu_portfolio(simulator, vine_fits[[t]], wealth[t], 
                                    gamma, n_sim)
    weights_history[[t]] <- w_opt
    
    # Next date
    current_date <- rebal_dates[t]
    if (t < T_horizon) {
      next_date <- rebal_dates[t + 1]
    } else {
      next_date <- index(returns_xts)[nrow(returns_xts)]
    }
    
    # Actual return over holding period
    actual_log <- returns_xts[paste0(current_date + 1, "/", next_date)]
    actual_gross <- exp(colSums(actual_log))
    
    R_ref  <- as.numeric(actual_gross[simulator$ref_col])
    R_risk <- as.numeric(actual_gross[simulator$risk_cols])
    portf_ret <- R_ref + sum((R_risk - R_ref) * w_opt)
    
    wealth[t + 1] <- wealth[t] * portf_ret
    
    #cat(sprintf("Period %d: %s → Wealth: %.2f | Weights: %s\n",
    #            t, current_date, wealth[t + 1],
    #            paste(round(w_opt, 4), collapse = ", ")))
  }
  
  # Metrics
  returns_p <- diff(wealth) / wealth[1:T_horizon]
  final_wealth  <- wealth[T_horizon + 1]
  total_return  <- (final_wealth / W0 - 1) * 100
  annual_return <- ((final_wealth / W0)^(1/(T_horizon/12)) - 1) * 100
  annual_vol    <- sd(returns_p) * sqrt(12) * 100
  sharpe        <- if (sd(returns_p) > 0) mean(returns_p) / sd(returns_p) * sqrt(12) else NA_real_
  max_dd        <- max(1 - wealth / cummax(wealth)) * 100
  
  metrics <- c(final_wealth = final_wealth,
               total_return = total_return,
               annual_return = annual_return,
               annual_vol    = annual_vol,
               sharpe_ratio  = sharpe,
               max_drawdown  = max_dd)
  
  stop_timer(timer)
  list(wealth = wealth, weights = weights_history, metrics = metrics)
}



build_simulator <- function(marginals, asset_names, ref_col = 7) {
  # Extract unconditional moments and AR(1) coefficients for each asset
  for (name in asset_names) {
    model <- marginals[[name]]
    if (identical(model$marginal_type, "component_ewma")) {
      mu <- model$mu_ar; ar1 <- model$ar1
      marginals[[name]]$mu_uncond <- if (abs(ar1) < 1) mu / (1 - ar1) else mu
      marginals[[name]]$sigma_uncond <- sqrt(mean(model$sigma^2))
    } else {
      cfit <- model$fit@fit$coef
      mu <- cfit["mu"]
      ar1 <- if ("ar1" %in% names(cfit)) cfit["ar1"] else 0
      # Fitted RMS volatility is valid for asymmetric/nonlinear GARCH variants.
      fitted_mean <- as.numeric(model$fit@fit$fitted.values)
      fitted_sigma <- as.numeric(model$fit@fit$sigma)
      marginals[[name]]$mu_uncond <- if (any(is.finite(fitted_mean))) mean(fitted_mean[is.finite(fitted_mean)]) else if (abs(ar1) < 1) mu / (1 - ar1) else mean(model$z)
      marginals[[name]]$sigma_uncond <- if (any(is.finite(fitted_sigma))) sqrt(mean(fitted_sigma[is.finite(fitted_sigma)]^2)) else sd(model$z)
    }
    # Store AR(1) intercept and coefficient for conditional mean
    marginals[[name]]$mu_ar <- mu
    marginals[[name]]$ar1 <- ar1
  }
  
  VineReturnSimulator(marginals, asset_names, ref_col)
}




# ============================================================================================

if (RUN_TESTS) {
  source("helper/load_data.r")
  source("benchmark_models/Li_Ng.r")
  load("data/marginal_results.RData")

  returns <- load_returns()
  sim <- build_simulator(marginals, asset_names, ref_col = 7)

  # ---- Rebalancing dates (same as benchmarks.r) ----
  L <- 500
  all_dates <- index(returns)
  rebal_dates <- endpoints(returns[L:nrow(returns)], on = "months")
  rebal_dates <- index(returns)[rebal_dates + L - 1]
  rebal_dates <- tail(rebal_dates, 36)

  # ---- Fit rolling‑window vine at each rebalancing date ----
  vine_fits <- vector("list", length(rebal_dates))
  for (t in seq_along(rebal_dates)) {
    current_date <- rebal_dates[t]
    window_end <- which(index(returns) == current_date)
    window_start <- window_end - L + 1
    U_window <- U[window_start:window_end, ]
    
    vine_fits[[t]] <- vinecop(
      U_window,
      var_types = rep("c", ncol(U_window)),
      structure = dvine_structure(1:ncol(U_window)),
      family_set = c("gaussian", "t", "clayton", "gumbel", "frank", "joe"),
      selcrit = "aic"
    )
    cat(sprintf("✓ Vine fit %d/%d (date: %s)\n", t, length(rebal_dates), current_date))
  }

  # ---- Run Expected Utility back‑test ----
  for (gamma in c(3)) {
    cat(sprintf("\nRunning Expected Utility back-test with gamma = %.1f...\n", gamma))
    eu_result <- run_eu_backtest(
      simulator   = sim,
      vine_fits   = vine_fits,
      returns_xts = returns,
      rebal_dates = rebal_dates,
      W0          = 100000,
      gamma       = gamma,
      n_sim       = 10000
    )

    # ---- Compare with existing benchmarks ----
    cat("\n\n")
    cat("===========================================================\n")
    cat("   EXPECTED UTILITY vs. MEAN–VARIANCE BENCHMARKS           \n")
    cat("===========================================================\n")
    cat(sprintf("%-25s %12s %10s %10s %10s %10s %10s\n",
                "Strategy", "Final W.", "Return%", "Ann.Ret%", "Vol%", "Sharpe", "Max DD"))
    cat("-----------------------------------------------------------\n")
    
    metrics <- eu_result$metrics
    cat(sprintf("%-25s %12.0f %10.2f %10.2f %10.2f %10.3f %10.2f\n",
                paste("Expected Utility (gamma=", gamma, ")", sep=""),
                metrics["final_wealth"],
                metrics["total_return"],
                metrics["annual_return"],
                metrics["annual_vol"],
                metrics["sharpe_ratio"],
                metrics["max_drawdown"]))
    cat("===========================================================\n\n")
  }

  # Save
  save(eu_result, vine_fits, file = "data/eu_backtest_result.RData")
  cat("✓ Results saved to data/eu_backtest_result.RData\n")
}
  
