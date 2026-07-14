# ==========================================================================
# benchmarks.r
# Runs benchmark tests between different model setups to compare performance
# ==========================================================================

library(rvinecopulib)
library(xts)
library(torch)
library(parallel)

RUN_TESTS <- TRUE

source("helper/load_data.r")
source("Li_Ng.r")
source("DCC.r")
source("dynamic_vine_NN.r")
source("expected_utility_single.r")
source("expected_utility_multi.r")
source("helper/timer.r")
source("helper/plotting.r")


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



# Benchmark 1: Empirical Li–Ng (raw historical moments)
benchmark_empirical <- function(returns_xts, rebal_dates, T_horizon, ref_col = 7,
                                L = 500, w0 = 100000, gamma = 2) {
  timer <- start_timer("Empirical MV")

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
  
  stop_timer(timer)
  return(list(wealth = wealth_emp, weights = weights_history, metrics = metrics))
}



# Benchmark 2: DCC‑GARCH(.,.)
benchmark_DCC <- function(returns_xts, rebal_dates, T_horizon, ref_col = 7,
                          L = 500, w0 = 100000, gamma = 2, n_sim = 10000,
                          distribution = "norm") {
  timer <- start_timer(paste0("DCC-GARCH (", distribution, ")"))

  source("DCC.r")
  
  risk_cols <- setdiff(1:ncol(returns_xts), ref_col)
  all_cols  <- c(ref_col, risk_cols)
  wealth <- numeric(T_horizon + 1)
  wealth[1] <- w0
  weights_history <- vector("list", T_horizon)
  
  for (t in 1:T_horizon) {
    current_date <- rebal_dates[t]
    window_end   <- which(index(returns_xts) == current_date)
    window_start <- window_end - L + 1
    ret_window   <- returns_xts[window_start:window_end, all_cols]
    
    dcc_fit  <- fit_DCC(ret_window, distribution)
    sim_gross <- simulate_DCC(dcc_fit, n_sim)
    
    ref_sim  <- sim_gross[, 1]
    risk_sim <- sim_gross[, -1, drop = FALSE]
    
    returns_list <- list(cbind(ref_sim, risk_sim))
    policy <- compute_policy(returns_list, wealth[t], gamma, "E")
    ut <- policy$policy[[1]]$vt - policy$policy[[1]]$Kt * wealth[t]
    w <- as.numeric(ut / wealth[t])
    weights_history[[t]] <- w
    
    next_date <- if (t < T_horizon) rebal_dates[t + 1] else index(returns_xts)[nrow(returns_xts)]
    actual_log   <- returns_xts[paste0(current_date + 1, "/", next_date), all_cols]
    actual_gross <- exp(colSums(actual_log))
    R_ref  <- as.numeric(actual_gross[1])
    R_risk <- as.numeric(actual_gross[-1])
    wealth[t + 1] <- wealth[t] * (R_ref + sum((R_risk - R_ref) * w))
  }
  
  rets <- diff(wealth) / wealth[1:T_horizon]
  metrics <- c(
    final_wealth  = wealth[T_horizon + 1],
    total_return  = (wealth[T_horizon + 1] / w0 - 1) * 100,
    annual_return = ((wealth[T_horizon + 1] / w0)^(1/(T_horizon/12)) - 1) * 100,
    annual_vol    = sd(rets) * sqrt(12) * 100,
    sharpe_ratio  = ((wealth[T_horizon + 1] / w0)^(1/(T_horizon/12)) - 1) * 100 /
                    (sd(rets) * sqrt(12) * 100),
    max_drawdown  = max(1 - wealth / cummax(wealth)) * 100
  )
  
  stop_timer(timer)
  list(wealth = wealth, weights = weights_history, metrics = metrics)
}



# Benchmark 3: Static D‑vine MV (vine fitted once on pre‑sample, constant moments)
benchmark_static_vine <- function(returns_xts, U, marginals, asset_names,
                                  rebal_dates, T_horizon, ref_col = 7,
                                  L = 500, w0 = 100000, gamma = 2, n_sim = 10000) {
  timer <- start_timer("Static Vine MV")
  
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
  
  stop_timer(timer)
  return(list(wealth = wealth, weights = weights_history, metrics = metrics))
}



# Benchmark 4: Rolling‑window D‑vine MV (vine re‑estimated at each rebalancing date)
benchmark_rolling_vine <- function(returns_xts, U, marginals, asset_names,
                                   rebal_dates, T_horizon, ref_col = 7,
                                   L = 500, w0 = 100000, gamma = 2, n_sim = 10000) {
  timer <- start_timer("Rolling Vine MV")

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
  
  stop_timer(timer)
  return(list(wealth = wealth, weights = weights_history, metrics = metrics))
}


# Helper: train or load NN models
get_nn_models <- function(U, vine_fit, marginals, returns_xts,
                          rebal_dates, force_retrain = FALSE) {
  model_dir <- "data/nn_models"
  
  if (!force_retrain && dir.exists(model_dir)) {
    #cat("Loading pre‑trained NN models...\n")
    nn_models <- load_nn_models(model_dir)
    return(nn_models)
  }
  
  pre_sample_end <- which(index(returns_xts) == rebal_dates[1]) - 1
  U_train <- U[1:pre_sample_end, ]
  
  #cat("Training NNs on pre‑sample data...\n")
  timer_nn <- start_timer("NN Training")
  
  marg_states <- extract_marginal_states(marginals, U, returns_xts)
  z_train <- marg_states$z[1:pre_sample_end, ]
  sigma_train <- marg_states$sigma[1:pre_sample_end, ]
  
  nn_models <- train_all_edges(U_train, vine_fit, asset_names,
                                z_train, sigma_train,
                                epochs = 500, lr = 1e-3, patience = 50)
  stop_timer(timer_nn)
  
  save_nn_models(nn_models, model_dir)
  return(nn_models)
}



# Benchmark 5: NN‑driven D‑vine MV
benchmark_NN_vine <- function(returns_xts, U, marginals, asset_names,
                               rebal_dates, nn_models, full_vine, ref_col = 7,
                               L = 500, w0 = 100000, gamma = 2, n_sim = 10000) {
  timer <- start_timer("NN Vine MV")

  risk_cols <- setdiff(1:ncol(returns_xts), ref_col)
  wealth <- numeric(length(rebal_dates) + 1)
  wealth[1] <- w0
  weights_history <- vector("list", length(rebal_dates))
  
  sim <- build_simulator(marginals, asset_names, ref_col)

  marg_states <- extract_marginal_states(marginals, U, returns_xts)
  
  for (t in seq_along(rebal_dates)) {
    current_date <- rebal_dates[t]
    window_end   <- which(index(returns_xts) == current_date)
    window_start <- window_end - L + 1
    U_window <- U[window_start:window_end, ]
    
    # Extract real z and sigma for the current window
    z_window     <- marg_states$z[window_start:window_end, ]
    sigma_window <- marg_states$sigma[window_start:window_end, ]

    vine_nn <- build_nn_vine(nn_models, full_vine, U_window, z_window, sigma_window)
    
    sim_ret   <- sim$simulate_returns(vine_nn, n_sim)
    sim_gross <- sim_ret$gross
    
    returns_list <- list(cbind(sim_gross[, ref_col], sim_gross[, risk_cols]))
    policy <- compute_policy(returns_list, wealth[t], gamma, "E")
    ut <- policy$policy[[1]]$vt - policy$policy[[1]]$Kt * wealth[t]
    w <- as.numeric(ut / wealth[t])
    weights_history[[t]] <- w
    
    next_date <- if (t < length(rebal_dates)) rebal_dates[t + 1] else index(returns_xts)[nrow(returns_xts)]
    actual_log   <- returns_xts[paste0(current_date + 1, "/", next_date)]
    actual_gross <- exp(colSums(actual_log))
    R_ref  <- as.numeric(actual_gross[ref_col])
    R_risk <- as.numeric(actual_gross[risk_cols])
    wealth[t + 1] <- wealth[t] * (R_ref + sum((R_risk - R_ref) * w))
  }
  
  rets <- diff(wealth) / wealth[1:length(rebal_dates)]
  metrics <- c(
    final_wealth  = wealth[length(rebal_dates) + 1],
    total_return  = (wealth[length(rebal_dates) + 1] / w0 - 1) * 100,
    annual_return = ((wealth[length(rebal_dates) + 1] / w0)^(1/(length(rebal_dates)/12)) - 1) * 100,
    annual_vol    = sd(rets) * sqrt(12) * 100,
    sharpe_ratio  = ((wealth[length(rebal_dates) + 1] / w0)^(1/(length(rebal_dates)/12)) - 1) * 100 /
                    (sd(rets) * sqrt(12) * 100),
    max_drawdown  = max(1 - wealth / cummax(wealth)) * 100
  )
  
  stop_timer(timer)
  list(wealth = wealth, weights = weights_history, metrics = metrics)
}



# Benchmark 6: NN‑driven D‑vine Expected Utility
benchmark_NN_eu <- function(returns_xts, U, marginals, asset_names,
                             rebal_dates, nn_models, full_vine, ref_col = 7,
                             L = 500, w0 = 100000, gamma = 2, n_sim = 10000) {
  timer <- start_timer("NN Vine EU")

  sim <- build_simulator(marginals, asset_names, ref_col)
  wealth <- numeric(length(rebal_dates) + 1)
  wealth[1] <- w0
  weights_history <- vector("list", length(rebal_dates))

  # Extract z and sigma once (for the full sample, then subset in loop)
  marg_states <- extract_marginal_states(marginals, U, returns_xts)

  for (t in seq_along(rebal_dates)) {
    current_date <- rebal_dates[t]
    window_end   <- which(index(returns_xts) == current_date)
    window_start <- window_end - L + 1
    U_window     <- U[window_start:window_end, ]
    z_window     <- marg_states$z[window_start:window_end, ]
    sigma_window <- marg_states$sigma[window_start:window_end, ]

    vine_nn <- build_nn_vine(nn_models, full_vine, U_window, z_window, sigma_window)

    w_opt <- optimise_eu_portfolio(sim, vine_nn, wealth[t], gamma, n_sim)
    weights_history[[t]] <- w_opt

    next_date <- if (t < length(rebal_dates)) rebal_dates[t + 1] else index(returns_xts)[nrow(returns_xts)]
    actual_log   <- returns_xts[paste0(current_date + 1, "/", next_date)]
    actual_gross <- exp(colSums(actual_log))
    R_ref  <- as.numeric(actual_gross[ref_col])
    R_risk <- as.numeric(actual_gross[sim$risk_cols])
    wealth[t + 1] <- wealth[t] * (R_ref + sum((R_risk - R_ref) * w_opt))
  }

  rets <- diff(wealth) / wealth[1:length(rebal_dates)]
  metrics <- c(
    final_wealth  = wealth[length(rebal_dates) + 1],
    total_return  = (wealth[length(rebal_dates) + 1] / w0 - 1) * 100,
    annual_return = ((wealth[length(rebal_dates) + 1] / w0)^(1/(length(rebal_dates)/12)) - 1) * 100,
    annual_vol    = sd(rets) * sqrt(12) * 100,
    sharpe_ratio  = ((wealth[length(rebal_dates) + 1] / w0)^(1/(length(rebal_dates)/12)) - 1) * 100 /
                    (sd(rets) * sqrt(12) * 100),
    max_drawdown  = max(1 - wealth / cummax(wealth)) * 100
  )

  stop_timer(timer)
  list(wealth = wealth, weights = weights_history, metrics = metrics)
}



# Runner function: runs all benchmarks in parallel and returns comparison table
run_all_benchmarks <- function(returns_xts, U, marginals, asset_names,
                               rebal_dates, T_horizon = 12, ref_col = 7,
                               L = 500, w0 = 100000, gamma = 2, n_sim = 10000,
                               save_plot = "figures/wealth_curves.pdf") {
  
  # Build simulator and vine fits once (shared across EU strategies)
  sim_eu <- build_simulator(marginals, asset_names, ref_col)
  
  vine_fits_eu <- vector("list", length(rebal_dates))
  for (t in seq_along(rebal_dates)) {
    window_end   <- which(index(returns_xts) == rebal_dates[t])
    window_start <- window_end - L + 1
    U_window <- U[window_start:window_end, ]
    vine_fits_eu[[t]] <- vinecop(
      U_window,
      var_types  = rep("c", ncol(U_window)),
      structure  = dvine_structure(1:ncol(U_window)),
      family_set = c("gaussian","t","clayton","gumbel","frank","joe"),
      selcrit    = "aic"
    )
  }
  
  # Train NN models once
  nn_models <- get_nn_models(U, vine_fit, marginals, returns_xts, rebal_dates)
  
  # Define each benchmark as a task
  tasks_parallel <- list(
    empirical = function() benchmark_empirical(returns_xts, rebal_dates, T_horizon, ref_col, L, w0, gamma),
    dcc       = function() benchmark_DCC(returns_xts, rebal_dates, T_horizon, ref_col, L, w0, gamma, n_sim),
    static    = function() benchmark_static_vine(returns_xts, U, marginals, asset_names, rebal_dates, T_horizon, ref_col, L, w0, gamma, n_sim),
    rolling   = function() benchmark_rolling_vine(returns_xts, U, marginals, asset_names, rebal_dates, T_horizon, ref_col, L, w0, gamma, n_sim),
    eu_single = function() {
      eu <- run_eu_backtest(sim_eu, vine_fits_eu, returns_xts, rebal_dates, w0, gamma, n_sim)
      list(wealth = eu$wealth, weights = eu$weights, metrics = eu$metrics)
    },
    eu_multi  = function() run_eu_multi_backtest(sim_eu, returns_xts, U, rebal_dates, L, w0, gamma, n_sim)
  )
  
  n_cores <- min(detectCores() - 1, length(tasks_parallel))
  cl <- makeCluster(n_cores, type = "PSOCK")
  clusterEvalQ(cl, {
    rm(list = ls(envir = .GlobalEnv)[grep("^\\.ark_", ls(envir = .GlobalEnv))], 
      envir = .GlobalEnv)
  })

  # validate seed for cluster RNG
  .seed <- get0("seed", ifnotfound = NULL)
  if (is.null(.seed)) {
    .seed <- 123L
    
  } else {
    .seed <- suppressWarnings(as.integer(.seed))
    if (is.na(.seed) || length(.seed) != 1L) .seed <- 123L
  }

  clusterSetRNGStream(cl, .seed)
  
  export_list <- list(
    returns_xts = returns_xts, U = U, marginals = marginals, asset_names = asset_names,
    rebal_dates = rebal_dates, T_horizon = T_horizon, ref_col = ref_col,
    L = L, w0 = w0, gamma = gamma, n_sim = n_sim,
    sim_eu = sim_eu, vine_fits_eu = vine_fits_eu
  )
  for (name in names(export_list)) {
    clusterExport(cl, name, envir = environment())
  }
  
  # Export all needed functions to workers
  clusterExport(cl, c("benchmark_empirical", "benchmark_DCC", "benchmark_static_vine", "benchmark_rolling_vine",
                      "benchmark_NN_vine", "benchmark_NN_eu", "run_eu_backtest", "run_eu_multi_backtest", "fit_DCC", "simulate_DCC",
                      "optimise_eu_portfolio", "optimise_eu_multi", "compute_policy", "compute_metrics", "build_simulator", "build_nn_vine",
                      "extract_marginal_states", "start_timer", "stop_timer", "VineReturnSimulator", "crra_utility", "expected_utility",
                      "predict_rho_nn", "predict_vine_params", "prepare_nn_data"))
  
  clusterEvalQ(cl, {
    library(rvinecopulib)
    library(rmgarch)
    library(xts)
    library(copula)
    library(mvtnorm)
    set.seed(123)
  })

  cat("Starting parallel benchmarks...\n")
  cat(sprintf("Number of tasks: %d\n", length(tasks_parallel)))
  for (name in names(tasks_parallel)) {
    cat(sprintf("  Task: %s\n", name))
  }
  
  cat(sprintf("Running %d non‑NN benchmarks on %d cores...\n", length(tasks_parallel), n_cores))
  results <- tryCatch({
    parLapply(cl, tasks_parallel, function(f) f())
  }, error = function(e) {
    cat("Parallel error caught:\n")
    cat(conditionMessage(e), "\n")
    # Try running each task sequentially to find the culprit
    for (name in names(tasks_parallel)) {
      cat(sprintf("Testing %s sequentially...\n", name))
      tryCatch({
        tasks_parallel[[name]]()
        cat(sprintf("  %s: OK\n", name))
      }, error = function(e2) {
        cat(sprintf("  %s: FAILED — %s\n", name, conditionMessage(e2)))
      })
    }
    stop(e)
  })
  stopCluster(cl)
  
  # Run NN benchmarks on master (torch objects can't be serialised to workers)
  cat("Running NN benchmarks on master...\n")
  results$nn_mv  <- benchmark_NN_vine(returns_xts, U, marginals, asset_names, rebal_dates, nn_models, vine_fit, ref_col, L, w0, gamma, n_sim)
  results$nn_eu  <- benchmark_NN_eu(returns_xts, U, marginals, asset_names, rebal_dates, nn_models, vine_fit, ref_col, L, w0, gamma, n_sim)
  
  # Extract individual results
  empirical  <- results$empirical
  dcc        <- results$dcc
  static     <- results$static
  rolling    <- results$rolling
  nn_mv      <- results$nn_mv
  eu_single  <- results$eu_single
  eu_multi   <- results$eu_multi
  nn_eu      <- results$nn_eu
  
  cat("\n✓ All benchmarks complete.\n")
  cat(sprintf("✓ Empirical MV     — Return: %.2f%%\n", as.numeric(empirical$metrics["total_return"])))
  cat(sprintf("✓ DCC-GARCH        — Return: %.2f%%\n", as.numeric(dcc$metrics["total_return"])))
  cat(sprintf("✓ Static Vine MV   — Return: %.2f%%\n", as.numeric(static$metrics["total_return"])))
  cat(sprintf("✓ Rolling Vine MV  — Return: %.2f%%\n", as.numeric(rolling$metrics["total_return"])))
  cat(sprintf("✓ NN Vine MV       — Return: %.2f%%\n", as.numeric(nn_mv$metrics["total_return"])))
  cat(sprintf("✓ Myopic EU        — Return: %.2f%%\n", as.numeric(eu_single$metrics["total_return"])))
  cat(sprintf("✓ Multi‑period EU  — Return: %.2f%%\n", as.numeric(eu_multi$metrics["total_return"])))
  cat(sprintf("✓ NN Vine EU       — Return: %.2f%%\n", as.numeric(nn_eu$metrics["total_return"])))
  
  # ── Risk metrics table ──
  metrics_table <- matrix(NA, nrow = 8, ncol = 6)
  colnames(metrics_table) <- c("final_wealth", "total_return", "annual_return",
                                "annual_vol", "sharpe_ratio", "max_drawdown")
  rownames(metrics_table) <- c("Empirical MV", "DCC-GARCH", "Static Vine MV", "Rolling Vine MV",
                                "NN Vine MV", "Single-Period EU", "Multi-period EU", "NN Vine EU")
  
  metrics_table[1, ] <- as.numeric(empirical$metrics)
  metrics_table[2, ] <- as.numeric(dcc$metrics)
  metrics_table[3, ] <- as.numeric(static$metrics)
  metrics_table[4, ] <- as.numeric(rolling$metrics)
  metrics_table[5, ] <- as.numeric(nn_mv$metrics)
  metrics_table[6, ] <- as.numeric(eu_single$metrics)
  metrics_table[7, ] <- as.numeric(eu_multi$metrics)
  metrics_table[8, ] <- as.numeric(nn_eu$metrics)
  
  cat("\n===========================================================\n")
  cat("                   RISK & PERFORMANCE METRICS               \n")
  cat("===========================================================\n")
  cat(sprintf("%-20s %12s %10s %10s %10s %10s %10s\n",
              "Strategy", "Final W.", "Return%", "Ann.Ret%", "Vol%", "Sharpe", "MaxDD%"))
  cat("-----------------------------------------------------------\n")
  for (i in 1:8) {
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
  plot_wealth_curves(results, rebal_dates, save_path = save_plot)
  cat("✓ Wealth curve plotted.\n")
  
  return(list(empirical = empirical, dcc = dcc, static = static, rolling = rolling,
              nn_mv = nn_mv, eu_single = eu_single, eu_multi = eu_multi, nn_eu = nn_eu,
              metrics_table = metrics_table))
}



# ======================================================================

load("data/marginal_results.RData")
returns <- load_returns()

L <- 500
all_dates <- index(returns)
rebal_dates <- endpoints(returns[L:nrow(returns)], on = "months")
rebal_dates <- index(returns)[rebal_dates + L - 1]
rebal_dates <- tail(rebal_dates, 36)

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
  n_sim        = 10000,
  save_plot    = "figures/wealth_curves.pdf"
)
