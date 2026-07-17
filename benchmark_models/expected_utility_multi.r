# ============================================================================
# expected_utility_multi.r
# Multi‑period expected utility via backward induction (Bellman)
# ============================================================================

source("benchmark_models/expected_utility_single.r")
source("helper/timer.r")

RUN_TESTS <- FALSE

# Multi‑period optimisation via backward induction
optimise_eu_multi <- function(simulator, vine_fits, gamma,
                               n_sim = 5000) {
  d_risk <- length(simulator$risk_cols)
  T_periods <- length(vine_fits)

  R_sim_list <- lapply(vine_fits, function(vf) {
    simulator$simulate_returns(vf, n_sim)$gross
  })

  v <- rep(1 / (1 - gamma), T_periods + 1)
  w_opt_multi <- vector("list", T_periods)

  for (t in seq(T_periods, 1, by = -1)) {
    R      <- R_sim_list[[t]]
    R_ref  <- R[, simulator$ref_col]
    R_risk <- R[, simulator$risk_cols, drop = FALSE]

        obj <- function(w) {
      if (any(w < 0) || sum(w) > 1) return(1e10)
      w_full <- c(1 - sum(w), w)
      R_all  <- cbind(R_ref, R_risk)
      portf_return <- as.vector(R_all %*% w_full)
      eu <- if (gamma == 1) {
        mean(log(pmax(portf_return, 1e-10)))
      } else {
        mean(pmax(portf_return, 1e-10)^(1 - gamma))
      }
      -eu * v[t + 1]
    }

    w0 <- rep(1 / (d_risk + 1), d_risk)
    opt <- optim(w0, obj, method = "L-BFGS-B",
                 lower = rep(0, d_risk), upper = rep(1, d_risk),
                 control = list(maxit = 500, factr = 1e-8))
    w_opt_multi[[t]] <- opt$par

    w_full <- c(1 - sum(opt$par), opt$par)
    R_all  <- cbind(R_ref, R_risk)
    portf_return <- as.vector(R_all %*% w_full)
    v[t] <- if (gamma == 1) {
      mean(log(pmax(portf_return, 1e-10))) * v[t + 1]
    } else {
      mean(pmax(portf_return, 1e-10)^(1 - gamma)) * v[t + 1]
    }

    #cat(sprintf("  t=%d: v=%.6f, w=%s\n", t, v[t],
    #            paste(round(opt$par, 4), collapse = ", ")))
  }

  w_opt_multi
}

# Multi‑period back‑test
run_eu_multi_backtest <- function(simulator, returns_xts, U,
                                   rebal_dates, L = 500, W0 = 100000,
                                   gamma = 2, n_sim = 5000) {
  timer <- start_timer("Multi‑period EU")
  T_horizon <- length(rebal_dates)

  vine_fits <- lapply(rebal_dates, function(d) {
    we <- which(index(returns_xts) == d)
    U_window <- U[(we - L + 1):we, ]
    vinecop(U_window,
      var_types  = rep("c", ncol(U_window)),
      structure  = dvine_structure(1:ncol(U_window)),
      family_set = c("gaussian","t","clayton","gumbel","frank","joe"),
      selcrit    = "aic")
  })

  #cat(sprintf("Multi‑period EU: %d periods, %d sims\n",
  #            T_horizon, n_sim))
  w_opt_all <- optimise_eu_multi(simulator, vine_fits, gamma, n_sim)

  wealth <- numeric(T_horizon + 1)
  wealth[1] <- W0
  for (t in seq_len(T_horizon)) {
    w_opt <- w_opt_all[[t]]
    current_date <- rebal_dates[t]
    next_date <- if (t < T_horizon) rebal_dates[t + 1] else index(returns_xts)[nrow(returns_xts)]
    actual_log   <- returns_xts[paste0(current_date + 1, "/", next_date)]
    actual_gross <- exp(colSums(actual_log))
    R_ref  <- as.numeric(actual_gross[simulator$ref_col])
    R_risk <- as.numeric(actual_gross[simulator$risk_cols])
    wealth[t + 1] <- wealth[t] * (R_ref + sum((R_risk - R_ref) * w_opt))

    #cat(sprintf("Period %d: %s → Wealth: %.2f | w: %s\n",
    #            t, current_date, wealth[t + 1],
    #            paste(round(w_opt, 4), collapse = ", ")))
  }

  rets <- diff(wealth) / wealth[1:T_horizon]
  metrics <- c(
    final_wealth  = wealth[T_horizon + 1],
    total_return  = (wealth[T_horizon + 1] / W0 - 1) * 100,
    annual_return = ((wealth[T_horizon + 1] / W0)^(1/(T_horizon/12)) - 1) * 100,
    annual_vol    = sd(rets) * sqrt(12) * 100,
    sharpe_ratio  = ((wealth[T_horizon + 1] / W0)^(1/(T_horizon/12)) - 1) * 100 /
                    (sd(rets) * sqrt(12) * 100),
    max_drawdown  = max(1 - wealth / cummax(wealth)) * 100
  )

  stop_timer(timer)
  list(wealth = wealth, weights = w_opt_all, metrics = metrics)
}


# ============================================================================

if (RUN_TESTS) {
  source("helper/load_data.r")
  load("data/marginal_results.RData")
  returns <- load_returns()
  sim <- build_simulator(marginals, asset_names, ref_col = 7)

  L <- 500
  all_dates <- index(returns)
  rebal_dates <- endpoints(returns[L:nrow(returns)], on = "months")
  rebal_dates <- index(returns)[rebal_dates + L - 1]
  rebal_dates <- tail(rebal_dates, 36)

  eu_multi <- run_eu_multi_backtest(sim, returns, U, rebal_dates,
                                      L = L, gamma = 3, n_sim = 20000)

  cat("\n===========================================================\n")
  cat("   MULTI‑PERIOD EXPECTED UTILITY\n")
  cat("===========================================================\n")
  m <- eu_multi$metrics
  cat(sprintf("Final wealth:  %.0f\n", m["final_wealth"]))
  cat(sprintf("Total return:  %.2f%%\n", m["total_return"]))
  cat(sprintf("Annual return: %.2f%%\n", m["annual_return"]))
  cat(sprintf("Annual vol:    %.2f%%\n", m["annual_vol"]))
  cat(sprintf("Sharpe ratio:  %.3f\n", m["sharpe_ratio"]))
  cat(sprintf("Max drawdown:  %.2f%%\n", m["max_drawdown"]))
  cat("===========================================================\n")
}