# ============================================================================
# statistical_tests.r
# Diebold–Mariano test and portfolio turnover
# ============================================================================

library(sandwich)
library(lmtest)
library(parallel)

RUN_TESTS <- TRUE

source("helper/load_data.r")
source("Li_Ng.r")
source("expected_utility_single.r")
source("expected_utility_multi.r")
source("benchmarks.r")

load("data/marginal_results.RData")
returns <- load_returns()


# 1. Extract monthly out‑of‑sample returns for each strategy
extract_monthly_returns <- function(wealth_path) {
  # wealth_path: vector of wealth at each rebalancing date (length T+1)
  T <- length(wealth_path) - 1
  monthly_ret <- wealth_path[2:(T+1)] / wealth_path[1:T] - 1
  monthly_ret
}



# 2. Diebold–Mariano test
dm_test <- function(returns_A, returns_B, loss = "squared") {
  # returns_A, returns_B: vectors of out‑of‑sample returns
  if (loss == "squared") {
    d <- -(returns_A^2) + (returns_B^2)   # lower squared error = better → positive d means A better
  } else if (loss == "return") {
    d <- returns_A - returns_B
  } else if (loss == "utility") {
    # CRRA utility with gamma = 2 (match your risk aversion)
    gamma <- 2
    u_A <- (1 + returns_A)^(1 - gamma) / (1 - gamma)
    u_B <- (1 + returns_B)^(1 - gamma) / (1 - gamma)
    d <- u_A - u_B
  }
  
  n <- length(d)
  reg <- lm(d ~ 1)
  nw_se <- sqrt(vcovHAC(reg))[1, 1]
  
  dm_stat <- mean(d) / (nw_se / sqrt(n))
  p_value <- 2 * (1 - pnorm(abs(dm_stat)))
  
  c(DM = dm_stat, p_value = p_value)
}


# 3. Turnover
compute_turnover <- function(weights_history, returns_xts, rebal_dates, ref_col = 7) {
  T <- length(weights_history)
  turnover <- numeric(T - 1)
  
  risk_cols <- setdiff(1:ncol(returns_xts), ref_col)
  
  for (t in 2:T) {
    prev_date <- rebal_dates[t - 1]
    curr_date <- rebal_dates[t]
    
    # Previous weights after market movement (before rebalancing)
    actual_log <- returns_xts[paste0(prev_date + 1, "/", curr_date)]
    actual_gross <- exp(colSums(actual_log))
    
    w_prev <- weights_history[[t - 1]]
    R_ref  <- as.numeric(actual_gross[ref_col])
    R_risk <- as.numeric(actual_gross[risk_cols])
    
    portf_ret <- R_ref * (1 - sum(w_prev)) + sum(R_risk * w_prev)
    w_prev_evolved <- as.numeric(R_risk * w_prev) / portf_ret
    
    w_curr <- weights_history[[t]]
    turnover[t - 1] <- sum(abs(w_curr - w_prev_evolved))
  }
  
  list(mean_turnover = mean(turnover), turnover_series = turnover)
}



# 4. Run all tests
run_statistical_tests <- function(results_list, returns_xts, rebal_dates) {
  # results_list: output of run_all_benchmarks (list with empirical, static, rolling, nn_mv, eu_single, eu_multi, nn_eu)
  # Each has $wealth and $weights
  
  # Extract monthly returns
  monthly_ret <- list(
    empirical   = extract_monthly_returns(results_list$empirical$wealth),
    static      = extract_monthly_returns(results_list$static$wealth),
    rolling     = extract_monthly_returns(results_list$rolling$wealth),
    nn_mv       = extract_monthly_returns(results_list$nn_mv$wealth),
    eu_single   = extract_monthly_returns(results_list$eu_single$wealth),
    eu_multi    = extract_monthly_returns(results_list$eu_multi$wealth),
    nn_eu       = extract_monthly_returns(results_list$nn_eu$wealth)
  )
  
  names_list <- names(monthly_ret)
  n <- length(names_list)
  
  # ── DM test matrix ──
  cat("\n===========================================================\n")
  cat("   DIEBOLD–MARIANO TEST (pairwise p‑values)\n")
  cat("   H0: equal predictive accuracy\n")
  cat("   Positive DM statistic → row strategy beats column strategy\n")
  cat("===========================================================\n\n")
  
  dm_pvalues <- matrix(NA, n, n)
  rownames(dm_pvalues) <- names_list
  colnames(dm_pvalues) <- names_list
  
  for (i in 1:n) {
    for (j in 1:n) {
      if (i != j) {
        result <- dm_test(monthly_ret[[i]], monthly_ret[[j]], loss = "utility")
        dm_pvalues[i, j] <- result["p_value"]
      }
    }
  }
  
  # Print as formatted table
  cat(sprintf("%-18s", ""))
  for (j in 1:n) cat(sprintf("%10s", abbreviate(names_list[j], 8)))
  cat("\n")
  for (i in 1:n) {
    cat(sprintf("%-18s", abbreviate(names_list[i], 18)))
    for (j in 1:n) {
      if (i == j) {
        cat(sprintf("%10s", "—"))
      } else {
        p <- dm_pvalues[i, j]
        sig <- if (p < 0.01) "***" else if (p < 0.05) "**" else if (p < 0.10) "*" else ""
        cat(sprintf("%7.3f%-3s", p, sig))
      }
    }
    cat("\n")
  }
  cat("---\n")
  cat("*** p<0.01, ** p<0.05, * p<0.10\n\n")
  
  # ── Key pairwise comparisons ──
  cat("Key comparisons:\n")
  pairs <- list(
    c("rolling", "empirical"),
    c("rolling", "static"),
    c("rolling", "nn_mv"),
    c("nn_eu", "eu_single"),
    c("static", "empirical")
  )
  for (pair in pairs) {
    i <- match(pair[1], names_list)
    j <- match(pair[2], names_list)
    res <- dm_test(monthly_ret[[i]], monthly_ret[[j]], loss = "utility")
    cat(sprintf("  %-25s vs %-25s: DM = %+.3f, p = %.3f %s\n",
                names_list[i], names_list[j], res["DM"], res["p_value"],
                if (res["p_value"] < 0.05) "**" else if (res["p_value"] < 0.10) "*" else ""))
  }
  
  # ── Turnover ──
  cat("\n===========================================================\n")
  cat("   PORTFOLIO TURNOVER (average monthly)\n")
  cat("===========================================================\n\n")
  
  for (name in names_list) {
    if (!is.null(results_list[[name]]$weights)) {
      to <- compute_turnover(results_list[[name]]$weights, returns_xts, rebal_dates)
      cat(sprintf("  %-25s: %.4f\n", name, to$mean_turnover))
    }
  }
  
  cat("\n")
  return(invisible(list(dm_pvalues = dm_pvalues, monthly_ret = monthly_ret)))
}


# ============================================================================


# Set up rebalancing dates (same as benchmarks.r)
L <- 500
all_dates <- index(returns)
rebal_dates <- endpoints(returns[L:nrow(returns)], on = "months")
rebal_dates <- index(returns)[rebal_dates + L - 1]
rebal_dates <- tail(rebal_dates, 36)

# Run benchmarks first (or load saved results)
if (!file.exists("data/benchmark_results.RData")) {
  cat("Running benchmarks first...\n")
  results <- run_all_benchmarks(
    returns_xts  = returns,
    U            = U,
    marginals    = marginals,
    asset_names  = asset_names,
    rebal_dates  = rebal_dates,
    T_horizon    = 36,
    ref_col      = 7,
    L            = 500,
    w0           = 100000,
    gamma        = 2,
    n_sim        = 10000
  )
  save(results, file = "data/benchmark_results.RData")
} else {
  load("data/benchmark_results.RData")
}

# Run statistical tests
run_statistical_tests(results, returns, rebal_dates)
