# ==========================================================================
# benchmarks.r
# Runs benchmark tests between different model setups to compare performance
# ==========================================================================

library(rvinecopulib)
library(xts)

source("load_data.r")
source("pre_commitment.r")


# Helper: compute risk metrics from a wealth path
compute_metrics <- function(wealth, T_horizon, w0 = 100000) {
  # Period returns
  returns_p <- diff(wealth) / wealth[1:T_horizon]
  
  final_wealth  <- wealth[T_horizon + 1]
  total_return  <- (final_wealth / w0 - 1) * 100
  annual_return <- ((final_wealth / w0)^(1/(T_horizon/12)) - 1) * 100
  annual_vol    <- sd(returns_p) * sqrt(12) * 100
  sharpe        <- annual_return / annual_vol
  max_dd        <- max(1 - wealth / cummax(wealth)) * 100
  
  return(c(final_wealth = final_wealth,
           total_return = total_return,
           annual_return = annual_return,
           annual_vol    = annual_vol,
           sharpe_ratio  = sharpe,
           max_drawdown  = max_dd))
}


# Helper: plot wealth curves for all benchmarks
plot_wealth <- function(empirical, static_vine, rolling_vine, rebal_dates, 
                        save_path = NULL) {
  # Combine into one data frame for plotting
  dates <- c(rebal_dates[1], rebal_dates)  # start date + end of each period
  df <- data.frame(
    Date      = rep(dates, 3),
    Wealth    = c(empirical$wealth, static_vine$wealth, rolling_vine$wealth),
    Strategy  = rep(c("Empirical MV", "Static Vine MV", "Rolling Vine MV"), 
                    each = length(dates))
  )
  
  # Open plot device if saving
  if (!is.null(save_path)) pdf(save_path, width = 8, height = 5)
  
  # Base plot
  plot(dates, empirical$wealth, type = "n",
       xlab = "Date", ylab = "Wealth (€)",
       main = "Multi‑Period Mean–Variance Portfolio Performance",
       ylim = range(df$Wealth))
  
  # Add grid and lines
  grid()
  cols <- c("black", "blue", "red")
  ltys <- c(3, 2, 1)
  lwds <- c(2, 2, 2.5)
  
  lines(dates, empirical$wealth,  col = cols[1], lty = ltys[1], lwd = lwds[1])
  lines(dates, static_vine$wealth, col = cols[2], lty = ltys[2], lwd = lwds[2])
  lines(dates, rolling_vine$wealth,col = cols[3], lty = ltys[3], lwd = lwds[3])
  
  # Legend
  legend("topleft",
         legend = c("Empirical MV", "Static Vine MV", "Rolling Vine MV"),
         col = cols, lty = ltys, lwd = lwds, bty = "n")
  
  if (!is.null(save_path)) dev.off()
}



# Benchmark 1: Empirical Li–Ng (raw historical moments)
benchmark_empirical <- function(returns_xts, rebal_dates, T_horizon, ref_col = 7,
                                L = 500, w0 = 100000, gamma = 2) {
  risk_cols <- setdiff(1:ncol(returns_xts), ref_col)
  wealth_emp <- numeric(T_horizon + 1)
  wealth_emp[1] <- w0
  weights_history <- vector("list", T_horizon)

  for (t in 1:T_horizon) {
    current_date <- rebal_dates[t]
    window_end <- which(index(returns_xts) == current_date)
    window_start <- window_end - L + 1
    
    ret_window <- as.matrix(exp(returns_xts[window_start:window_end, ]))
    returns_list <- list(cbind(ret_window[, ref_col], ret_window[, risk_cols]))
    
    policy <- compute_policy(returns_list, wealth_emp[t], gamma, "E")
    ut <- policy$policy[[1]]$vt - policy$policy[[1]]$Kt * wealth_emp[t]
    w <- as.numeric(ut / wealth_emp[t])
    weights_history[[t]] <- w

    if (t < T_horizon) {
      next_date <- rebal_dates[t+1]
    } else {
      next_date <- index(returns_xts)[nrow(returns_xts)]
    }
    
    actual_log <- returns_xts[paste0(current_date + 1, "/", next_date)]
    actual_gross <- exp(colSums(actual_log))
    
    portf_ret <- as.numeric(actual_gross[ref_col]) * (1 - sum(w)) +
      sum(as.numeric(actual_gross[risk_cols]) * w)
    
    wealth_emp[t+1] <- wealth_emp[t] * portf_ret
  }

  metrics <- compute_metrics(wealth_emp, T_horizon, w0)
  
  return(list(wealth = wealth_emp, weights = weights_history, metrics = metrics))
}


# Benchmark 2: Static D‑vine MV (vine fitted once on pre‑sample, constant moments)
benchmark_static_vine <- function(returns_xts, U, marginals, asset_names,
                                  rebal_dates, T_horizon, ref_col = 7,
                                  L = 500, w0 = 100000, gamma = 2, n_sim = 10000) {
  risk_cols <- setdiff(1:ncol(returns_xts), ref_col)
  wealth <- numeric(T_horizon + 1)
  wealth[1] <- w0
  weights_history <- vector("list", T_horizon)

  pre_sample_end <- which(index(returns_xts) == rebal_dates[1]) - 1
  U_pre <- U[1:pre_sample_end, ]
  
  vine_static <- vinecop(
    U_pre,
    var_types = rep("c", ncol(U_pre)),
    structure = dvine_structure(1:ncol(U_pre)),
    family_set = c("gaussian", "t", "clayton", "gumbel", "frank", "joe"),
    selcrit = "aic"
  )
  
  sim_U <- rvinecop(n_sim, vine_static)
  sim_log_returns <- matrix(0, n_sim, ncol(sim_U))
  
  for (i in 1:ncol(sim_U)) {
    name <- asset_names[i]
    model <- marginals[[name]]
    prob_grid <- seq_len(length(model$z_sorted)) / (length(model$z_sorted) + 1)
    z_sim <- approx(prob_grid, model$z_sorted, xout = sim_U[, i], rule = 2)$y
    cfit <- model$fit@fit$coef
    mu <- cfit["mu"]; ar1 <- if ("ar1" %in% names(cfit)) cfit["ar1"] else 0
    omega <- cfit["omega"]; alpha <- cfit["alpha1"]; beta <- cfit["beta1"]
    mu_uncond <- if (abs(ar1) < 1) mu / (1 - ar1) else mean(as.numeric(returns_xts[, i]))
    sigma2 <- if (alpha + beta < 1) omega / (1 - alpha - beta) else var(as.numeric(returns_xts[, i]))
    sigma <- sqrt(sigma2)
    sim_log_returns[, i] <- mu_uncond + sigma * z_sim
  }
  
  sim_gross <- exp(sim_log_returns)
  returns_list <- list(cbind(sim_gross[, ref_col], sim_gross[, risk_cols]))

  for (t in 1:T_horizon) {
    current_date <- rebal_dates[t]
    policy <- compute_policy(returns_list, wealth[t], gamma, "E")
    ut <- policy$policy[[1]]$vt - policy$policy[[1]]$Kt * wealth[t]
    w <- as.numeric(ut / wealth[t])
    weights_history[[t]] <- w

    if (t < T_horizon) {
      next_date <- rebal_dates[t+1]
    } else {
      next_date <- index(returns_xts)[nrow(returns_xts)]
    }
    
    actual_log <- returns_xts[paste0(current_date + 1, "/", next_date)]
    actual_gross <- exp(colSums(actual_log))
    
    portf_ret <- as.numeric(actual_gross[ref_col]) * (1 - sum(w)) +
      sum(as.numeric(actual_gross[risk_cols]) * w)
    
    wealth[t+1] <- wealth[t] * portf_ret
  }

  metrics <- compute_metrics(wealth, T_horizon, w0)
  
  return(list(wealth = wealth, weights = weights_history, metrics = metrics))
}



# Benchmark 3: Rolling‑window D‑vine MV (vine re‑estimated at each rebalancing date)
benchmark_rolling_vine <- function(returns_xts, U, marginals, asset_names,
                                   rebal_dates, T_horizon, ref_col = 7,
                                   L = 500, w0 = 100000, gamma = 2, n_sim = 10000) {
  risk_cols <- setdiff(1:ncol(returns_xts), ref_col)
  wealth <- numeric(T_horizon + 1)
  wealth[1] <- w0
  weights_history <- vector("list", T_horizon)

  for (t in 1:T_horizon) {
    current_date <- rebal_dates[t]
    window_end <- which(index(returns_xts) == current_date)
    window_start <- window_end - L + 1
    U_window <- U[window_start:window_end, ]
    
    vine_current <- vinecop(
      U_window,
      var_types = rep("c", ncol(U_window)),
      structure = dvine_structure(1:ncol(U_window)),
      family_set = c("gaussian", "t", "clayton", "gumbel", "frank", "joe"),
      selcrit = "aic"
    )
    
    sim_U <- rvinecop(n_sim, vine_current)
    sim_log_returns <- matrix(0, n_sim, ncol(sim_U))
    
    for (i in 1:ncol(sim_U)) {
      name <- asset_names[i]
      model <- marginals[[name]]
      prob_grid <- seq_len(length(model$z_sorted)) / (length(model$z_sorted) + 1)
      z_sim <- approx(prob_grid, model$z_sorted, xout = sim_U[, i], rule = 2)$y
      cfit <- model$fit@fit$coef
      mu <- cfit["mu"]; ar1 <- if ("ar1" %in% names(cfit)) cfit["ar1"] else 0
      omega <- cfit["omega"]; alpha <- cfit["alpha1"]; beta <- cfit["beta1"]
      mu_uncond <- if (abs(ar1) < 1) mu / (1 - ar1) else mean(as.numeric(returns_xts[, i]))
      sigma2 <- if (alpha + beta < 1) omega / (1 - alpha - beta) else var(as.numeric(returns_xts[, i]))
      sigma <- sqrt(sigma2)
      sim_log_returns[, i] <- mu_uncond + sigma * z_sim
    }
    
    sim_gross <- exp(sim_log_returns)
    returns_list <- list(cbind(sim_gross[, ref_col], sim_gross[, risk_cols]))
    
    policy <- compute_policy(returns_list, wealth[t], gamma, "E")
    ut <- policy$policy[[1]]$vt - policy$policy[[1]]$Kt * wealth[t]
    w <- as.numeric(ut / wealth[t])
    weights_history[[t]] <- w

    if (t < T_horizon) {
      next_date <- rebal_dates[t+1]
    } else {
      next_date <- index(returns_xts)[nrow(returns_xts)]
    }
    
    actual_log <- returns_xts[paste0(current_date + 1, "/", next_date)]
    actual_gross <- exp(colSums(actual_log))
    
    portf_ret <- as.numeric(actual_gross[ref_col]) * (1 - sum(w)) +
      sum(as.numeric(actual_gross[risk_cols]) * w)
    
    wealth[t+1] <- wealth[t] * portf_ret
  }

  metrics <- compute_metrics(wealth, T_horizon, w0)
  
  return(list(wealth = wealth, weights = weights_history, metrics = metrics))
}



# Runner function: runs all three benchmarks and returns comparison table
run_all_benchmarks <- function(returns_xts, U, marginals, asset_names,
                               rebal_dates, T_horizon = 12, ref_col = 7,
                               L = 500, w0 = 100000, gamma = 2, n_sim = 10000,
                               save_plot = NULL) {
  
  cat("\n#############################################################\n")
  cat("#   MULTI‑PERIOD MEAN–VARIANCE PORTFOLIO BENCHMARKS         #\n")
  cat("#############################################################\n\n")
  
  empirical  <- benchmark_empirical(returns_xts, rebal_dates, T_horizon, ref_col, L, w0, gamma)
  cat(sprintf("✓ Empirical MV done  — Return: %.2f%%\n", empirical$metrics["total_return"]))
  
  static     <- benchmark_static_vine(returns_xts, U, marginals, asset_names,
                                       rebal_dates, T_horizon, ref_col, L, w0, gamma, n_sim)
  cat(sprintf("✓ Static Vine MV done — Return: %.2f%%\n", static$metrics["total_return"]))
  
  rolling    <- benchmark_rolling_vine(returns_xts, U, marginals, asset_names,
                                        rebal_dates, T_horizon, ref_col, L, w0, gamma, n_sim)
  cat(sprintf("✓ Rolling Vine MV done — Return: %.2f%%\n", rolling$metrics["total_return"]))
  
  # ── Risk metrics table ──
  metrics_table <- rbind(empirical$metrics, static$metrics, rolling$metrics)
  metrics_table <- as.matrix(metrics_table)
  rownames(metrics_table) <- c("Empirical MV", "Static Vine MV", "Rolling Vine MV")
  
  cat("\n===========================================================\n")
  cat("                   RISK & PERFORMANCE METRICS               \n")
  cat("===========================================================\n")
  cat(sprintf("%-20s %12s %10s %10s %10s %10s %10s\n",
              "Strategy", "Final W.", "Return%", "Ann.Ret%", "Vol%", "Sharpe", "MaxDD%"))
  cat("-----------------------------------------------------------\n")
  for (i in 1:3) {
    cat(sprintf("%-20s %12.0f %10.2f %10.2f %10.2f %10.3f %10.2f\n",
                rownames(metrics_table)[i],
                metrics_table[i, "final_wealth"],
                metrics_table[i, "total_return"],
                metrics_table[i, "annual_return"],
                metrics_table[i, "annual_vol"],
                metrics_table[i, "sharpe_ratio"],
                metrics_table[i, "max_drawdown"]))
  }
  cat("===========================================================\n\n")
  
  # ── Wealth plot ──
  plot_wealth(empirical, static, rolling, rebal_dates, save_path = save_plot)
  cat("✓ Wealth curve plotted.\n")
  
  return(list(empirical = empirical, static = static, rolling = rolling,
              metrics_table = metrics_table))
}




# ======================================================================

load("data/marginal_results.RData")
returns <- load_returns()

L <- 500
all_dates <- index(returns)
rebal_dates <- endpoints(returns[L:nrow(returns)], on = "months")
rebal_dates <- index(returns)[rebal_dates + L - 1]
rebal_dates <- tail(rebal_dates, 12)

results <- run_all_benchmarks(
  returns_xts  = returns,
  U            = U,
  marginals    = marginals,
  asset_names  = asset_names,
  rebal_dates  = rebal_dates,
  T_horizon    = 12,
  ref_col      = 7,
  L            = 500,
  w0           = 100000,
  gamma        = 2,
  n_sim        = 10000,
  save_plot    = "figures/wealth_curves.pdf"
)
