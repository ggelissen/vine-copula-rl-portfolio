# Additional causal financial baselines for future/external evaluation.
#
# This module extends benchmark_weights.R without changing the consumed v4
# benchmark contract. Every generator emits only ex-ante target weights and a
# solver audit; the common Python evaluator remains the sole accounting path.

if (!exists("solve_constrained_weights")) {
  source("publication_pipeline_draft/benchmark_weights.R")
}

read_extended_benchmark_contract <- function(path) {
  contract <- read_benchmark_contract(path)
  required <- c(
    "minimum_history_months", "minimum_variance_turnover_penalty",
    "risk_parity_long_only", "risk_parity_ridge",
    "risk_parity_turnover_penalty", "mean_cvar_history_months",
    "mean_cvar_risk_aversion", "mean_cvar_scenario_seed",
    "momentum_lookback_months", "momentum_skip_months",
    "momentum_signal_scale", "black_litterman_tau",
    "black_litterman_view_confidence",
    "black_litterman_equilibrium_risk_aversion",
    "black_litterman_prior", "black_litterman_views")
  missing <- setdiff(required, names(contract))
  if (length(missing)) {
    benchmark_protocol_error("Extended benchmark contract is missing: ",
                             paste(missing, collapse = ", "))
  }
  scalar_numeric <- setdiff(required, c(
    "risk_parity_long_only", "black_litterman_prior",
    "black_litterman_views"))
  if (any(!vapply(contract[scalar_numeric], function(value) {
    length(value) == 1L && is.finite(as.numeric(value))
  }, logical(1)))) {
    benchmark_protocol_error("Extended numeric settings must be finite scalars.")
  }
  if (!isTRUE(contract$risk_parity_long_only) ||
      contract$minimum_history_months < 24L ||
      contract$mean_cvar_history_months < contract$minimum_history_months ||
      contract$momentum_lookback_months < 2L ||
      contract$momentum_skip_months < 0L ||
      contract$black_litterman_tau <= 0 ||
      contract$black_litterman_view_confidence <= 0 ||
      contract$black_litterman_view_confidence > 1) {
    benchmark_protocol_error("Extended benchmark settings are infeasible.")
  }
  if (!identical(contract$black_litterman_prior,
                 "equal_weight_equilibrium") ||
      !identical(contract$black_litterman_views,
                 "causal_12_minus_1_momentum_absolute_views")) {
    benchmark_protocol_error("Unsupported Black-Litterman prior/view declaration.")
  }
  contract
}

extended_latest_input_date <- function(monthly, decision_date) {
  available <- as.Date(monthly$periods$holding_end_date) <= as.Date(decision_date)
  if (!any(available)) benchmark_protocol_error("No completed data at ", decision_date)
  max(as.Date(monthly$periods$holding_end_date[available]))
}

minimum_variance_objective <- function(covariance, pretrade, contract,
                                       year_fraction = 1 / 12) {
  turnover_penalty <- as.numeric(contract$turnover_cost) +
    as.numeric(contract$minimum_variance_turnover_penalty)
  short_rate <- as.numeric(contract$annual_short_borrow_rate) * year_fraction
  cash_rate <- as.numeric(contract$annual_cash_borrow_rate) * year_fraction
  function(weights) {
    risk <- drop(crossprod(weights, covariance %*% weights))
    turnover <- sum(sqrt((weights - pretrade)^2 + 1e-12))
    financing <- short_rate * sum(pmax(-weights, 0)) +
      cash_rate * max(sum(weights) - 1, 0)
    as.numeric(risk + turnover_penalty * turnover + financing)
  }
}

risk_parity_objective <- function(covariance, pretrade, contract) {
  ridge <- as.numeric(contract$risk_parity_ridge)
  covariance <- covariance + diag(ridge, ncol(covariance))
  turnover_penalty <- as.numeric(contract$turnover_cost) +
    as.numeric(contract$risk_parity_turnover_penalty)
  target <- rep(1 / ncol(covariance), ncol(covariance))
  function(weights) {
    marginal <- as.numeric(covariance %*% weights)
    variance <- sum(weights * marginal)
    if (!is.finite(variance) || variance <= 0) return(1e12)
    contribution <- weights * marginal / variance
    sum((contribution - target)^2) +
      turnover_penalty * sum(sqrt((weights - pretrade)^2 + 1e-12))
  }
}

long_only_contract <- function(contract) {
  output <- contract
  output$max_short_weight <- 0
  output$gross_leverage <- abs(as.numeric(output$net_exposure))
  output
}

generate_moment_benchmark <- function(daily_log_returns, periods, contract,
                                      method = c("minimum_variance", "risk_parity"),
                                      solver = NULL) {
  method <- match.arg(method)
  monthly <- monthly_log_outcomes(daily_log_returns)
  asset_names <- colnames(daily_log_returns)
  solve_contract <- if (method == "risk_parity") long_only_contract(contract) else contract
  previous <- initial_equal_weight(asset_names, solve_contract)
  weights <- matrix(NA_real_, nrow(periods), length(asset_names))
  audits <- vector("list", nrow(periods))
  for (i in seq_len(nrow(periods))) {
    decision <- as.Date(periods$decision_date[i])
    history <- history_through_decision(
      monthly, decision, as.integer(contract$minimum_history_months))
    moments <- shrinkage_moments(history, contract)
    pretrade <- historical_pretrade_weight(
      daily_log_returns, periods, i, previous, solve_contract)
    year_fraction <- benchmark_period_year_fraction(periods[i, ], contract)
    objective <- if (method == "minimum_variance") {
      minimum_variance_objective(
        moments$covariance, pretrade, contract, year_fraction)
    } else {
      risk_parity_objective(moments$covariance, pretrade, contract)
    }
    result <- solve_constrained_weights(
      objective, previous, asset_names, solve_contract,
      context = sprintf("%s at %s", method, decision), solver = solver)
    assert_weight_vector(result$weights, asset_names, contract,
                         sprintf("%s common mandate at %s", method, decision))
    weights[i, ] <- result$weights
    previous <- result$weights
    audits[[i]] <- data.frame(
      method = method, decision_date = decision,
      latest_input_date = extended_latest_input_date(monthly, decision),
      convergence = result$convergence, iterations = result$iterations,
      solver_attempts = result$solver_attempts,
      solver_attempt_codes = result$solver_attempt_codes,
      solver_attempt_messages = result$solver_attempt_messages,
      solver_total_iterations = result$solver_total_iterations,
      objective = result$objective, solver_message = result$message,
      constraint_residual = result$constraint_residual,
      estimator = "oas_identity_covariance", stringsAsFactors = FALSE)
  }
  list(weights = canonical_weight_log(periods, weights, asset_names, contract),
       audit = do.call(rbind, audits))
}

causal_momentum_signal <- function(history, lookback, skip) {
  history <- as.matrix(history)
  required <- as.integer(lookback + skip)
  if (nrow(history) < required) {
    benchmark_protocol_error("Momentum history needs ", required, " months.")
  }
  end <- nrow(history) - as.integer(skip)
  indices <- seq.int(end - as.integer(lookback) + 1L, end)
  signal <- colSums(history[indices, , drop = FALSE]) / as.integer(lookback)
  trailing_scale <- apply(history[indices, , drop = FALSE], 2L, stats::sd)
  trailing_scale[!is.finite(trailing_scale) | trailing_scale <= 1e-8] <- 1
  signal / trailing_scale
}

generate_momentum_tilt <- function(daily_log_returns, periods, contract,
                                   solver = NULL) {
  monthly <- monthly_log_outcomes(daily_log_returns)
  asset_names <- colnames(daily_log_returns)
  previous <- initial_equal_weight(asset_names, contract)
  weights <- matrix(NA_real_, nrow(periods), length(asset_names))
  audits <- vector("list", nrow(periods))
  minimum <- max(as.integer(contract$minimum_history_months),
                 as.integer(contract$momentum_lookback_months +
                            contract$momentum_skip_months))
  for (i in seq_len(nrow(periods))) {
    decision <- as.Date(periods$decision_date[i])
    history <- history_through_decision(monthly, decision, minimum)
    moments <- shrinkage_moments(history, contract)
    raw_signal <- causal_momentum_signal(
      history, contract$momentum_lookback_months,
      contract$momentum_skip_months)
    # Rescale the cross-sectional signal to the causal magnitude of estimated
    # monthly means; this avoids an arbitrary return unit in the optimizer.
    magnitude <- max(stats::median(abs(moments$mu)), 1e-4)
    mu <- as.numeric(contract$momentum_signal_scale) * magnitude *
      raw_signal / max(stats::sd(raw_signal), 1e-8)
    pretrade <- historical_pretrade_weight(
      daily_log_returns, periods, i, previous, contract)
    result <- solve_constrained_weights(
      mean_variance_objective(
        mu, moments$covariance, pretrade, contract,
        benchmark_period_year_fraction(periods[i, ], contract)),
      previous, asset_names, contract,
      context = sprintf("momentum tilt at %s", decision), solver = solver)
    weights[i, ] <- result$weights
    previous <- result$weights
    audits[[i]] <- data.frame(
      method = "momentum_tilt", decision_date = decision,
      latest_input_date = extended_latest_input_date(monthly, decision),
      convergence = result$convergence, iterations = result$iterations,
      solver_attempts = result$solver_attempts,
      solver_attempt_codes = result$solver_attempt_codes,
      solver_attempt_messages = result$solver_attempt_messages,
      solver_total_iterations = result$solver_total_iterations,
      objective = result$objective, solver_message = result$message,
      constraint_residual = result$constraint_residual,
      signal = "12_minus_1_volatility_scaled", stringsAsFactors = FALSE)
  }
  list(weights = canonical_weight_log(periods, weights, asset_names, contract),
       audit = do.call(rbind, audits))
}

empirical_monthly_scenarios <- function(history, count, seed, maximum_history) {
  history <- tail(as.matrix(history), as.integer(maximum_history))
  if (nrow(history) < 24L || any(!is.finite(history))) {
    benchmark_protocol_error("Mean-CVaR empirical scenario history is invalid.")
  }
  set.seed(as.integer(seed))
  indices <- sample.int(nrow(history), as.integer(count), replace = TRUE)
  exp(history[indices, , drop = FALSE])
}

mean_cvar_objective <- function(scenario_gross, pretrade, contract,
                                year_fraction = 1 / 12) {
  scenarios <- as.matrix(scenario_gross)
  probability <- as.numeric(contract$cvar_probability)
  risk_aversion <- as.numeric(contract$mean_cvar_risk_aversion)
  short_rate <- as.numeric(contract$annual_short_borrow_rate) * year_fraction
  cash_rate <- as.numeric(contract$annual_cash_borrow_rate) * year_fraction
  function(weights) {
    gross <- 1 + scenarios %*% weights - sum(weights)
    turnover <- sum(abs(weights - pretrade))
    financing <- short_rate * sum(pmax(-weights, 0)) +
      cash_rate * max(sum(weights) - 1, 0)
    net <- as.numeric(gross) * exp(-as.numeric(contract$turnover_cost) * turnover -
                                  financing)
    if (any(!is.finite(net)) || any(net <= 1e-8)) return(1e12)
    returns <- net - 1
    tail_count <- max(1L, ceiling(probability * length(returns)))
    cvar_loss <- -mean(sort(returns)[seq_len(tail_count)])
    as.numeric(-mean(returns) + risk_aversion * cvar_loss)
  }
}

generate_mean_cvar <- function(daily_log_returns, periods, contract,
                               solver = NULL) {
  monthly <- monthly_log_outcomes(daily_log_returns)
  asset_names <- colnames(daily_log_returns)
  previous <- initial_equal_weight(asset_names, contract)
  weights <- matrix(NA_real_, nrow(periods), length(asset_names))
  audits <- vector("list", nrow(periods))
  for (i in seq_len(nrow(periods))) {
    decision <- as.Date(periods$decision_date[i])
    history <- history_through_decision(
      monthly, decision, as.integer(contract$minimum_history_months))
    scenario_seed <- as.integer(contract$mean_cvar_scenario_seed) + i
    scenarios <- empirical_monthly_scenarios(
      history, contract$scenario_count, scenario_seed,
      contract$mean_cvar_history_months)
    pretrade <- historical_pretrade_weight(
      daily_log_returns, periods, i, previous, contract)
    result <- solve_constrained_weights(
      mean_cvar_objective(
        scenarios, pretrade, contract,
        benchmark_period_year_fraction(periods[i, ], contract)),
      previous, asset_names, contract,
      context = sprintf("mean-CVaR at %s", decision), solver = solver)
    weights[i, ] <- result$weights
    previous <- result$weights
    audits[[i]] <- data.frame(
      method = "mean_cvar", decision_date = decision,
      latest_input_date = extended_latest_input_date(monthly, decision),
      convergence = result$convergence, iterations = result$iterations,
      solver_attempts = result$solver_attempts,
      solver_attempt_codes = result$solver_attempt_codes,
      solver_attempt_messages = result$solver_attempt_messages,
      solver_total_iterations = result$solver_total_iterations,
      objective = result$objective, solver_message = result$message,
      constraint_residual = result$constraint_residual,
      scenario_seed = scenario_seed,
      scenario_source = "causal_empirical_monthly_bootstrap",
      stringsAsFactors = FALSE)
  }
  list(weights = canonical_weight_log(periods, weights, asset_names, contract),
       audit = do.call(rbind, audits))
}

black_litterman_posterior <- function(covariance, equilibrium_weights,
                                      views, contract) {
  covariance <- as.matrix(covariance)
  n <- ncol(covariance)
  tau <- as.numeric(contract$black_litterman_tau)
  delta <- as.numeric(contract$black_litterman_equilibrium_risk_aversion)
  confidence <- as.numeric(contract$black_litterman_view_confidence)
  prior <- delta * as.numeric(covariance %*% equilibrium_weights)
  prior_covariance <- tau * covariance
  view_variance <- diag(pmax(diag(prior_covariance) * (1 - confidence) /
                              confidence, 1e-10), n)
  precision <- solve(prior_covariance) + solve(view_variance)
  posterior_covariance <- solve(precision)
  posterior_mean <- posterior_covariance %*%
    (solve(prior_covariance, prior) + solve(view_variance, views))
  list(mu = as.numeric(posterior_mean),
       covariance = covariance + posterior_covariance)
}

generate_black_litterman <- function(daily_log_returns, periods, contract,
                                     solver = NULL) {
  monthly <- monthly_log_outcomes(daily_log_returns)
  asset_names <- colnames(daily_log_returns)
  equilibrium <- rep(1 / length(asset_names), length(asset_names))
  previous <- initial_equal_weight(asset_names, contract)
  weights <- matrix(NA_real_, nrow(periods), length(asset_names))
  audits <- vector("list", nrow(periods))
  minimum <- max(as.integer(contract$minimum_history_months),
                 as.integer(contract$momentum_lookback_months +
                            contract$momentum_skip_months))
  for (i in seq_len(nrow(periods))) {
    decision <- as.Date(periods$decision_date[i])
    history <- history_through_decision(monthly, decision, minimum)
    moments <- shrinkage_moments(history, contract)
    raw_views <- causal_momentum_signal(
      history, contract$momentum_lookback_months,
      contract$momentum_skip_months)
    magnitude <- max(stats::median(abs(moments$mu)), 1e-4)
    views <- magnitude * raw_views / max(stats::sd(raw_views), 1e-8)
    posterior <- black_litterman_posterior(
      moments$covariance, equilibrium, views, contract)
    pretrade <- historical_pretrade_weight(
      daily_log_returns, periods, i, previous, contract)
    result <- solve_constrained_weights(
      mean_variance_objective(
        posterior$mu, posterior$covariance, pretrade, contract,
        benchmark_period_year_fraction(periods[i, ], contract)),
      previous, asset_names, contract,
      context = sprintf("Black-Litterman at %s", decision), solver = solver)
    weights[i, ] <- result$weights
    previous <- result$weights
    audits[[i]] <- data.frame(
      method = "black_litterman_momentum_views", decision_date = decision,
      latest_input_date = extended_latest_input_date(monthly, decision),
      convergence = result$convergence, iterations = result$iterations,
      solver_attempts = result$solver_attempts,
      solver_attempt_codes = result$solver_attempt_codes,
      solver_attempt_messages = result$solver_attempt_messages,
      solver_total_iterations = result$solver_total_iterations,
      objective = result$objective, solver_message = result$message,
      constraint_residual = result$constraint_residual,
      prior = contract$black_litterman_prior,
      views = contract$black_litterman_views,
      stringsAsFactors = FALSE)
  }
  list(weights = canonical_weight_log(periods, weights, asset_names, contract),
       audit = do.call(rbind, audits))
}

generate_extended_financial_benchmarks <- function(
    daily_log_returns, periods, contract, methods, solver = NULL) {
  allowed <- c("minimum_variance", "risk_parity", "mean_cvar",
               "momentum_tilt", "black_litterman_momentum_views")
  if (!length(methods) || any(!methods %in% allowed) || anyDuplicated(methods)) {
    benchmark_protocol_error("Invalid extended benchmark method set.")
  }
  output <- list()
  for (method in methods) {
    output[[method]] <- switch(
      method,
      minimum_variance = generate_moment_benchmark(
        daily_log_returns, periods, contract, "minimum_variance", solver),
      risk_parity = generate_moment_benchmark(
        daily_log_returns, periods, contract, "risk_parity", solver),
      mean_cvar = generate_mean_cvar(
        daily_log_returns, periods, contract, solver),
      momentum_tilt = generate_momentum_tilt(
        daily_log_returns, periods, contract, solver),
      black_litterman_momentum_views = generate_black_litterman(
        daily_log_returns, periods, contract, solver))
  }
  output
}
