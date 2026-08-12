# Causal benchmark weight generators for the locked portfolio experiment.
#
# This module never scores a strategy and never reads a future holding-period
# outcome when choosing its weight.  Each public generator returns only the
# canonical keys and w_<ASSET> columns.  Realised returns and implementation
# costs remain the exclusive responsibility of publication_pipeline.py.

`%||%` <- function(value, default) if (is.null(value)) default else value

.benchmark_vine_functions <- new.env(parent = globalenv())
ensure_dynamic_vine_functions <- function() {
  if (!exists("load_nn_dynamic_vine_fit", envir = .benchmark_vine_functions,
              inherits = FALSE)) {
    source("benchmark_models/dynamic_vine_NN.r",
           local = .benchmark_vine_functions)
  }
  invisible(.benchmark_vine_functions)
}

benchmark_protocol_error <- function(...) {
  stop(paste0(...), call. = FALSE)
}

bind_benchmark_audits <- function(audits) {
  if (!is.list(audits) || !length(audits) ||
      any(!vapply(audits, is.data.frame, logical(1)))) {
    benchmark_protocol_error("Benchmark audits must be a non-empty list of data frames.")
  }
  columns <- unique(unlist(lapply(audits, names), use.names = FALSE))
  normalized <- lapply(audits, function(frame) {
    missing <- setdiff(columns, names(frame))
    for (name in missing) frame[[name]] <- NA
    frame[columns]
  })
  output <- do.call(rbind, normalized)
  rownames(output) <- NULL
  output
}

require_benchmark_namespace <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    benchmark_protocol_error("Required benchmark package is unavailable: ", package)
  }
}

read_benchmark_contract <- function(path) {
  require_benchmark_namespace("jsonlite")
  if (!file.exists(path)) benchmark_protocol_error("Benchmark contract not found: ", path)
  contract <- jsonlite::fromJSON(path, simplifyVector = TRUE)
  required <- c(
    "schema_version", "evaluation_id", "net_exposure", "gross_leverage",
    "max_long_weight", "max_short_weight", "weight_tolerance",
    "turnover_cost", "annual_short_borrow_rate", "annual_cash_borrow_rate",
    "crra_gamma", "cvar_probability", "cvar_penalty", "scenario_count",
    "scenario_seed", "rolling_vine_lookback_months", "dcc_horizon_days",
    "mv_covariance_shrinkage_method", "mv_mean_shrinkage_method",
    "mv_risk_aversion",
    "optimizer_maxeval", "optimizer_max_restarts", "optimizer_xtol_rel",
    "optimizer_allowed_convergence_codes"
  )
  missing <- setdiff(required, names(contract))
  if (length(missing)) {
    benchmark_protocol_error("Benchmark contract is missing: ",
                             paste(missing, collapse = ", "))
  }
  if (!identical(as.integer(contract$schema_version), 1L)) {
    benchmark_protocol_error("Unsupported benchmark contract schema.")
  }
  character_fields <- c("schema_version", "evaluation_id",
                        "mv_covariance_shrinkage_method",
                        "mv_mean_shrinkage_method")
  numeric_fields <- setdiff(
    required, c(character_fields, "optimizer_allowed_convergence_codes"))
  if (any(!vapply(contract[numeric_fields], function(x)
    length(x) == 1L && is.finite(as.numeric(x)), logical(1)))) {
    benchmark_protocol_error("All numeric benchmark settings must be finite scalars.")
  }
  if (contract$gross_leverage < abs(contract$net_exposure) ||
      contract$max_long_weight <= 0 || contract$max_short_weight < 0 ||
      contract$scenario_count < 100L || contract$cvar_probability <= 0 ||
      contract$cvar_probability >= 0.5 || contract$optimizer_maxeval < 1L ||
      contract$optimizer_max_restarts < 0L ||
      contract$optimizer_max_restarts != floor(contract$optimizer_max_restarts)) {
    benchmark_protocol_error("Benchmark contract contains infeasible settings.")
  }
  if (!identical(contract$mv_covariance_shrinkage_method, "oas_identity") ||
      !identical(contract$mv_mean_shrinkage_method, "james_stein_zero")) {
    benchmark_protocol_error("Unsupported shrinkage estimator declaration.")
  }
  validate_optimizer_convergence_codes(contract$optimizer_allowed_convergence_codes)
  contract
}

validate_optimizer_convergence_codes <- function(codes) {
  numeric_codes <- suppressWarnings(as.numeric(codes))
  valid <- is.atomic(codes) && length(codes) == 4L &&
    length(numeric_codes) == 4L && all(is.finite(numeric_codes)) &&
    all(numeric_codes == floor(numeric_codes)) && !anyDuplicated(numeric_codes) &&
    setequal(as.integer(numeric_codes), 1:4)
  if (!isTRUE(valid)) {
    benchmark_protocol_error(
      "optimizer_allowed_convergence_codes must be exactly the unique integer set {1,2,3,4}.")
  }
  as.integer(numeric_codes)
}

weight_columns <- function(asset_names) paste0("w_", asset_names)

assert_weight_vector <- function(weights, asset_names, contract,
                                 context = "portfolio") {
  weights <- as.numeric(weights)
  tolerance <- as.numeric(contract$weight_tolerance)
  if (length(weights) != length(asset_names) || any(!is.finite(weights))) {
    benchmark_protocol_error(context, ": weights are missing or non-finite.")
  }
  if (abs(sum(weights) - contract$net_exposure) > tolerance) {
    benchmark_protocol_error(context, ": net-exposure constraint failed.")
  }
  if (sum(abs(weights)) > contract$gross_leverage + tolerance) {
    benchmark_protocol_error(context, ": gross-exposure constraint failed.")
  }
  if (max(weights) > contract$max_long_weight + tolerance ||
      min(weights) < -contract$max_short_weight - tolerance) {
    benchmark_protocol_error(context, ": position constraint failed.")
  }
  invisible(weights)
}

assert_canonical_weight_log <- function(frame, periods, asset_names, contract,
                                        context = "weight log") {
  required <- c("window_id", "decision_date", "holding_end_date",
                weight_columns(asset_names))
  if (!identical(names(frame), required)) {
    benchmark_protocol_error(context, ": canonical columns/order do not match.")
  }
  if (nrow(frame) != nrow(periods)) {
    benchmark_protocol_error(context, ": row count does not match evaluation periods.")
  }
  for (name in c("decision_date", "holding_end_date")) {
    if (!identical(as.Date(frame[[name]]), as.Date(periods[[name]]))) {
      benchmark_protocol_error(context, ": calendar mismatch in ", name, ".")
    }
  }
  expected_window <- if ("window_id" %in% names(periods))
    as.character(periods$window_id) else rep(contract$evaluation_id, nrow(periods))
  if (!identical(as.character(frame$window_id), expected_window)) {
    benchmark_protocol_error(context, ": window_id mismatch.")
  }
  for (i in seq_len(nrow(frame))) {
    assert_weight_vector(as.numeric(unlist(
                           frame[i, weight_columns(asset_names), drop = FALSE],
                           use.names = FALSE)),
                         asset_names, contract,
                         sprintf("%s row %d", context, i))
  }
  invisible(frame)
}

canonical_weight_log <- function(periods, weights, asset_names, contract) {
  weights <- as.matrix(weights)
  if (!identical(dim(weights), c(nrow(periods), length(asset_names)))) {
    benchmark_protocol_error("Weight matrix dimensions do not match periods/assets.")
  }
  out <- data.frame(
    window_id = if ("window_id" %in% names(periods))
      as.character(periods$window_id) else rep(contract$evaluation_id, nrow(periods)),
    decision_date = as.Date(periods$decision_date),
    holding_end_date = as.Date(periods$holding_end_date),
    weights, check.names = FALSE
  )
  names(out)[-(1:3)] <- weight_columns(asset_names)
  assert_canonical_weight_log(out, periods, asset_names, contract)
  out
}

initial_equal_weight <- function(asset_names, contract) {
  weights <- rep(as.numeric(contract$net_exposure) / length(asset_names),
                 length(asset_names))
  assert_weight_vector(weights, asset_names, contract, "initial equal weight")
  weights
}

generate_equal_weight <- function(periods, asset_names, contract) {
  weights <- matrix(initial_equal_weight(asset_names, contract),
                    nrow = nrow(periods), ncol = length(asset_names), byrow = TRUE)
  canonical_weight_log(periods, weights, asset_names, contract)
}

benchmark_period_year_fraction <- function(period, contract) {
  convention <- contract$financing_proration %||% "fixed_period_v1"
  if (identical(convention, "actual_calendar_days_v1")) {
    days <- as.numeric(as.Date(period$holding_end_date) -
                         as.Date(period$decision_date))
    basis <- as.numeric(contract$day_count_basis %||% 365)
    if (length(days) != 1L || !is.finite(days) || days <= 0 ||
        length(basis) != 1L || !is.finite(basis) || basis <= 0) {
      benchmark_protocol_error("Actual-calendar benchmark financing interval is invalid.")
    }
    return(days / basis)
  }
  if (!identical(convention, "fixed_period_v1")) {
    benchmark_protocol_error("Unsupported benchmark financing_proration: ", convention)
  }
  1 / 12
}

benchmark_pretrade_weight <- function(previous_target, previous_asset_gross,
                                      contract, context = "benchmark") {
  convention <- contract$turnover_convention %||% "target_to_target_v1"
  if (identical(convention, "target_to_target_v1")) {
    return(as.numeric(previous_target))
  }
  if (!identical(convention, "drifted_pretrade_v1")) {
    benchmark_protocol_error("Unsupported benchmark turnover_convention: ", convention)
  }
  gross <- as.numeric(previous_asset_gross)
  target <- as.numeric(previous_target)
  if (length(gross) != length(target) || any(!is.finite(gross)) || any(gross <= 0)) {
    benchmark_protocol_error(context, " has invalid previous-period asset gross returns.")
  }
  portfolio_gross <- 1 + sum(target * (gross - 1))
  if (!is.finite(portfolio_gross) || portfolio_gross <= 0) {
    benchmark_protocol_error(context, " has invalid previous-period portfolio gross return.")
  }
  target * gross / portfolio_gross
}

historical_pretrade_weight <- function(daily_log_returns, periods, index,
                                       previous_target, contract) {
  if (index <= 1L || identical(
      contract$turnover_convention %||% "target_to_target_v1",
      "target_to_target_v1")) {
    return(as.numeric(previous_target))
  }
  previous_period <- periods[index - 1L, , drop = FALSE]
  if (as.Date(previous_period$holding_end_date) > as.Date(periods$decision_date[index])) {
    benchmark_protocol_error("Previous holding outcome is unavailable at the next decision.")
  }
  gross <- realised_gross_for_period(
    daily_log_returns,
    previous_period$decision_date,
    previous_period$holding_end_date)
  benchmark_pretrade_weight(
    previous_target, gross, contract,
    sprintf("pretrade state at %s", periods$decision_date[index]))
}

# Convert daily log returns into completed monthly log-return outcomes.  The
# holding_end_date is used as the information timestamp: an outcome is visible
# at decision t only when holding_end_date <= t.
monthly_log_outcomes <- function(daily_log_returns) {
  if (!inherits(daily_log_returns, "xts")) {
    benchmark_protocol_error("daily_log_returns must be an xts object.")
  }
  source("helper/time_split.r", local = TRUE)
  periods <- build_monthly_periods(daily_log_returns, min_history = 0L)
  values <- do.call(rbind, lapply(seq_len(nrow(periods)), function(i) {
    log(as.numeric(realised_gross_for_period(
      daily_log_returns, periods$decision_date[i], periods$holding_end_date[i])))
  }))
  colnames(values) <- colnames(daily_log_returns)
  list(periods = periods, log_returns = values)
}

history_through_decision <- function(monthly, decision_date,
                                     minimum_observations = 24L) {
  available <- which(as.Date(monthly$periods$holding_end_date) <= as.Date(decision_date))
  if (length(available) < minimum_observations) {
    benchmark_protocol_error("Only ", length(available),
      " completed monthly observations are available at decision ", decision_date,
      "; need ", minimum_observations, ".")
  }
  monthly$log_returns[available, , drop = FALSE]
}

shrinkage_moments <- function(history, contract) {
  history <- as.matrix(history)
  if (nrow(history) < 3L || any(!is.finite(history))) {
    benchmark_protocol_error("Moment history is too short or non-finite.")
  }
  n <- nrow(history); p <- ncol(history)
  raw_mean <- colMeans(history)
  unbiased_covariance <- stats::cov(history)

  # Empirical-Bayes/James-Stein shrinkage of noisy expected returns toward
  # zero.  The intensity is re-estimated causally at every decision rather
  # than hand-tuned on portfolio performance.
  mean_noise <- mean(diag(unbiased_covariance)) / n
  mean_signal <- sum(raw_mean^2)
  mean_shrinkage <- if (mean_signal <= 0) 1 else
    pmin(1, pmax(0, (p - 2) * mean_noise / mean_signal))
  mu <- (1 - mean_shrinkage) * raw_mean

  # Oracle Approximating Shrinkage (OAS) toward a scaled identity target.
  # The maximum-likelihood covariance is used in the closed-form intensity;
  # the result is positive semidefinite and well conditioned in short samples.
  centred <- sweep(history, 2L, raw_mean, "-")
  empirical_covariance <- crossprod(centred) / n
  trace_mean <- sum(diag(empirical_covariance)) / p
  alpha <- mean(empirical_covariance^2)
  denominator <- (n + 1) * (alpha - trace_mean^2 / p)
  covariance_shrinkage <- if (!is.finite(denominator) || denominator <= 0) 1 else
    pmin(1, pmax(0, (alpha + trace_mean^2) / denominator))
  target <- diag(trace_mean, p)
  covariance <- (1 - covariance_shrinkage) * empirical_covariance +
    covariance_shrinkage * target
  # A frozen numerical ridge is not a data-dependent fallback; it makes the
  # declared positive-semidefinite shrinkage target strictly positive definite.
  ridge <- max(1e-10, 1e-8 * mean(diag(covariance)))
  covariance <- covariance + diag(ridge, ncol(history))
  list(mu = as.numeric(mu), covariance = covariance,
       mean_shrinkage = mean_shrinkage,
       covariance_shrinkage = covariance_shrinkage)
}

solve_constrained_weights <- function(objective, previous_weight, asset_names,
                                      contract, context = "optimizer",
                                      solver = NULL) {
  require_benchmark_namespace("nloptr")
  if (is.null(solver)) solver <- nloptr::slsqp
  lower <- rep(-as.numeric(contract$max_short_weight), length(asset_names))
  upper <- rep(as.numeric(contract$max_long_weight), length(asset_names))
  x0 <- as.numeric(previous_weight)
  assert_weight_vector(x0, asset_names, contract, paste(context, "initial point"))
  allowed_convergence <- validate_optimizer_convergence_codes(
    contract$optimizer_allowed_convergence_codes)
  max_attempts <- 1L + as.integer(contract$optimizer_max_restarts)
  attempt_codes <- rep(NA_integer_, max_attempts)
  attempt_messages <- rep("", max_attempts)
  attempt_iterations <- rep(NA_integer_, max_attempts)
  current_x0 <- x0
  accepted <- FALSE

  constraint_residual_for <- function(weights) {
    if (length(weights) != length(asset_names) || any(!is.finite(weights))) {
      return(Inf)
    }
    max(
      abs(sum(weights) - as.numeric(contract$net_exposure)),
      max(0, sum(abs(weights)) - as.numeric(contract$gross_leverage)),
      max(0, max(weights) - as.numeric(contract$max_long_weight)),
      max(0, -as.numeric(contract$max_short_weight) - min(weights)))
  }

  for (attempt in seq_len(max_attempts)) {
    result <- tryCatch(
      solver(
        x0 = current_x0, fn = objective, lower = lower, upper = upper,
        # Pin the current nloptr convention explicitly: hin <= 0.  Relying on
        # deprecatedBehavior's version-dependent default would reverse the gross
        # constraint after a package upgrade.
        hin = function(w) sum(abs(w)) - as.numeric(contract$gross_leverage),
        heq = function(w) sum(w) - as.numeric(contract$net_exposure),
        control = list(
          maxeval = as.integer(contract$optimizer_maxeval),
          xtol_rel = as.numeric(contract$optimizer_xtol_rel),
          check_derivatives = FALSE, print_level = 0L),
        deprecatedBehavior = FALSE
      ),
      error = function(error) error
    )
    if (inherits(result, "error")) {
      benchmark_protocol_error(context, " failed on attempt ", attempt,
                               ": ", conditionMessage(result))
    }
    convergence <- as.integer(result$convergence %||% NA_integer_)
    weights <- as.numeric(result$par %||% numeric())
    residual <- constraint_residual_for(weights)
    attempt_codes[attempt] <- convergence
    attempt_messages[attempt] <- as.character(result$message %||% "missing")
    attempt_iterations[attempt] <- as.integer(result$iter %||% NA_integer_)

    accepted <- is.finite(convergence) &&
      convergence %in% allowed_convergence && is.finite(residual) &&
      residual <= as.numeric(contract$weight_tolerance)
    if (accepted) break

    # Code 5 is not accepted.  A finite feasible incumbent can, however, be
    # used as the unchanged starting point for a deterministic continuation.
    # The number of continuations is frozen in the contract and every attempt
    # is exposed in the audit output.  No clipping, projection, or alternative
    # solver is used.
    retryable <- identical(convergence, 5L) && is.finite(residual) &&
      residual <= as.numeric(contract$weight_tolerance)
    if (!retryable || attempt >= max_attempts) {
      used <- seq_len(attempt)
      trail <- paste0("attempt", used, "=code", attempt_codes[used], "[",
                      attempt_messages[used], "]", collapse = "; ")
      benchmark_protocol_error(
        context, " did not meet the frozen convergence rule; code=", convergence,
        ", allowed=", paste(allowed_convergence, collapse = ","),
        ", attempts=", trail)
    }
    current_x0 <- weights
  }
  attempts_used <- attempt
  # No clipping or projection is permitted after the solve.
  assert_weight_vector(weights, asset_names, contract, context)
  constraint_residual <- constraint_residual_for(weights)
  used <- seq_len(attempts_used)
  list(weights = weights, convergence = convergence,
       message = as.character(result$message %||% ""),
       iterations = as.integer(result$iter %||% NA_integer_),
       solver_attempts = as.integer(attempts_used),
       solver_attempt_codes = paste(attempt_codes[used], collapse = ","),
       solver_attempt_messages = paste(attempt_messages[used], collapse = " | "),
       solver_total_iterations = if (all(is.na(attempt_iterations[used])))
         NA_integer_ else sum(attempt_iterations[used], na.rm = TRUE),
       objective = as.numeric(result$value %||% objective(weights)),
       constraint_residual = as.numeric(constraint_residual))
}

mean_variance_objective <- function(mu, covariance, previous_weight, contract,
                                    year_fraction = 1 / 12) {
  risk_aversion <- as.numeric(contract$mv_risk_aversion)
  turnover_penalty <- as.numeric(contract$turnover_cost) +
    as.numeric(contract$optimizer_turnover_penalty %||% 0)
  period_short_rate <- as.numeric(contract$annual_short_borrow_rate) * year_fraction
  period_cash_rate <- as.numeric(contract$annual_cash_borrow_rate) * year_fraction
  function(weights) {
    risk <- 0.5 * risk_aversion * drop(crossprod(weights, covariance %*% weights))
    expected_return <- sum(mu * weights)
    turnover <- sum(sqrt((weights - previous_weight)^2 + 1e-12))
    financing <- period_short_rate * sum(pmax(-weights, 0)) +
      period_cash_rate * max(sum(weights) - 1, 0)
    as.numeric(risk - expected_return + turnover_penalty * turnover + financing)
  }
}

generate_shrinkage_mean_variance <- function(daily_log_returns, periods,
                                             contract, solver = NULL) {
  monthly <- monthly_log_outcomes(daily_log_returns)
  asset_names <- colnames(daily_log_returns)
  previous <- initial_equal_weight(asset_names, contract)
  weights <- matrix(NA_real_, nrow(periods), length(asset_names))
  audit <- vector("list", nrow(periods))
  for (i in seq_len(nrow(periods))) {
    history <- history_through_decision(monthly, periods$decision_date[i])
    moments <- shrinkage_moments(history, contract)
    pretrade <- historical_pretrade_weight(
      daily_log_returns, periods, i, previous, contract)
    year_fraction <- benchmark_period_year_fraction(periods[i, ], contract)
    result <- solve_constrained_weights(
      mean_variance_objective(
        moments$mu, moments$covariance, pretrade, contract, year_fraction),
      previous, asset_names, contract,
      context = sprintf("shrinkage mean-variance at %s", periods$decision_date[i]),
      solver = solver)
    weights[i, ] <- result$weights
    previous <- result$weights
    audit[[i]] <- data.frame(
      method = "shrinkage_mean_variance", decision_date = periods$decision_date[i],
      latest_input_date = max(monthly$periods$holding_end_date[
        monthly$periods$holding_end_date <= periods$decision_date[i]]),
      convergence = result$convergence, iterations = result$iterations,
      solver_attempts = result$solver_attempts,
      solver_attempt_codes = result$solver_attempt_codes,
      solver_attempt_messages = result$solver_attempt_messages,
      solver_total_iterations = result$solver_total_iterations,
      objective = result$objective, solver_message = result$message,
      objective_year_fraction = year_fraction,
      objective_turnover_convention = contract$turnover_convention %||%
        "target_to_target_v1",
      constraint_residual = result$constraint_residual,
      stringsAsFactors = FALSE)
  }
  list(weights = canonical_weight_log(periods, weights, asset_names, contract),
       audit = do.call(rbind, audit))
}

extract_dcc_forecast <- function(forecast, horizon, asset_names) {
  covariance_value <- forecast@mforecast$H[[1L]]
  if (is.matrix(covariance_value)) {
    covariance_array <- array(covariance_value,
                              dim = c(nrow(covariance_value), ncol(covariance_value), 1L))
  } else if (is.array(covariance_value) && length(dim(covariance_value)) == 3L) {
    covariance_array <- covariance_value
  } else {
    benchmark_protocol_error("DCC covariance forecast has an unsupported shape.")
  }
  if (dim(covariance_array)[3L] < horizon) {
    benchmark_protocol_error("DCC forecast returned fewer days than requested.")
  }
  monthly_covariance <- apply(covariance_array[, , seq_len(horizon), drop = FALSE],
                              c(1L, 2L), sum)
  dimnames(monthly_covariance) <- list(asset_names, asset_names)
  if (any(!is.finite(monthly_covariance))) {
    benchmark_protocol_error("DCC covariance forecast is non-finite.")
  }
  monthly_covariance
}

fit_monthly_horizon_dcc <- function(daily_history, horizon, seed) {
  require_benchmark_namespace("rmgarch")
  require_benchmark_namespace("rugarch")
  set.seed(as.integer(seed))
  univariate <- rugarch::ugarchspec(
    mean.model = list(armaOrder = c(0L, 0L), include.mean = TRUE),
    variance.model = list(model = "sGARCH", garchOrder = c(1L, 1L)),
    distribution.model = "std")
  specification <- rmgarch::dccspec(
    uspec = rugarch::multispec(replicate(ncol(daily_history), univariate,
                                         simplify = FALSE)),
    dccOrder = c(1L, 1L), distribution = "mvt")
  fit <- tryCatch(
    rmgarch::dccfit(specification, data = as.matrix(daily_history),
                    fit.control = list(eval.se = FALSE)),
    error = function(error) error)
  if (inherits(fit, "error")) {
    benchmark_protocol_error("DCC fit failed: ", conditionMessage(fit))
  }
  convergence <- tryCatch(as.integer(fit@mfit$convergence), error = function(e) NA_integer_)
  if (!is.finite(convergence) || convergence != 0L) {
    benchmark_protocol_error("DCC fit did not converge; code=", convergence)
  }
  forecast <- tryCatch(rmgarch::dccforecast(fit, n.ahead = as.integer(horizon)),
                       error = function(error) error)
  if (inherits(forecast, "error")) {
    benchmark_protocol_error("DCC forecast failed: ", conditionMessage(forecast))
  }
  list(covariance = extract_dcc_forecast(
    forecast, as.integer(horizon), colnames(daily_history)),
    fit_convergence = convergence)
}

generate_dcc_garch <- function(daily_log_returns, periods, contract,
                               dcc_fit_function = fit_monthly_horizon_dcc,
                               solver = NULL) {
  monthly <- monthly_log_outcomes(daily_log_returns)
  asset_names <- colnames(daily_log_returns)
  previous <- initial_equal_weight(asset_names, contract)
  weights <- matrix(NA_real_, nrow(periods), length(asset_names))
  audit <- vector("list", nrow(periods))
  daily_dates <- as.Date(zoo::index(daily_log_returns))
  for (i in seq_len(nrow(periods))) {
    decision <- as.Date(periods$decision_date[i])
    daily_index <- which(daily_dates <= decision)
    if (length(daily_index) < 500L) {
      benchmark_protocol_error("DCC has fewer than 500 daily observations at ", decision)
    }
    history <- history_through_decision(monthly, decision)
    moments <- shrinkage_moments(history, contract)
    forecast <- dcc_fit_function(
      daily_log_returns[daily_index, ], as.integer(contract$dcc_horizon_days),
      as.integer(contract$dcc_seed) + i)
    covariance <- as.matrix(forecast$covariance)
    if (!identical(dim(covariance), c(length(asset_names), length(asset_names))) ||
        any(!is.finite(covariance))) {
      benchmark_protocol_error("DCC provider returned an invalid covariance at ", decision)
    }
    # The expected-return convention is intentionally identical to the
    # shrinkage mean-variance benchmark; DCC changes only the risk forecast.
    pretrade <- historical_pretrade_weight(
      daily_log_returns, periods, i, previous, contract)
    year_fraction <- benchmark_period_year_fraction(periods[i, ], contract)
    result <- solve_constrained_weights(
      mean_variance_objective(
        moments$mu, covariance, pretrade, contract, year_fraction),
      previous, asset_names, contract,
      context = sprintf("DCC-GARCH at %s", decision), solver = solver)
    weights[i, ] <- result$weights
    previous <- result$weights
    audit[[i]] <- data.frame(
      method = "dcc_garch", decision_date = decision,
      latest_input_date = max(daily_dates[daily_index]),
      convergence = result$convergence, iterations = result$iterations,
      solver_attempts = result$solver_attempts,
      solver_attempt_codes = result$solver_attempt_codes,
      solver_attempt_messages = result$solver_attempt_messages,
      solver_total_iterations = result$solver_total_iterations,
      objective = result$objective, solver_message = result$message,
      objective_year_fraction = year_fraction,
      objective_turnover_convention = contract$turnover_convention %||%
        "target_to_target_v1",
      constraint_residual = result$constraint_residual,
      stringsAsFactors = FALSE)
  }
  list(weights = canonical_weight_log(periods, weights, asset_names, contract),
       audit = do.call(rbind, audit))
}

crra_value <- function(gross, gamma) {
  if (any(!is.finite(gross)) || any(gross <= 0)) return(rep(-Inf, length(gross)))
  if (abs(gamma - 1) < 1e-12) log(gross) else
    (gross^(1 - gamma) - 1) / (1 - gamma)
}

scenario_objective <- function(scenario_gross, previous_weight, contract,
                               year_fraction = 1 / 12) {
  scenario_gross <- as.matrix(scenario_gross)
  gamma <- as.numeric(contract$crra_gamma)
  probability <- as.numeric(contract$cvar_probability)
  cvar_penalty <- as.numeric(contract$cvar_penalty)
  turnover_penalty <- as.numeric(contract$optimizer_turnover_penalty %||% 0)
  period_short_rate <- as.numeric(contract$annual_short_borrow_rate) * year_fraction
  period_cash_rate <- as.numeric(contract$annual_cash_borrow_rate) * year_fraction
  function(weights) {
    portfolio_gross <- 1 + scenario_gross %*% weights - sum(weights)
    turnover <- sum(abs(weights - previous_weight))
    short_notional <- sum(pmax(-weights, 0))
    cash_notional <- max(sum(weights) - 1, 0)
    cost <- as.numeric(contract$turnover_cost) * turnover +
      period_short_rate * short_notional + period_cash_rate * cash_notional
    net_gross <- as.numeric(portfolio_gross) * exp(-cost)
    if (any(!is.finite(net_gross)) || any(net_gross <= 1e-8)) return(1e12)
    loss <- 1 - net_gross
    tail_count <- max(1L, ceiling(probability * length(loss)))
    cvar <- mean(sort(loss, decreasing = TRUE)[seq_len(tail_count)])
    utility <- mean(crra_value(net_gross, gamma))
    as.numeric(-utility + cvar_penalty * cvar + turnover_penalty * turnover)
  }
}

generate_scenario_optimizer <- function(periods, asset_names, contract,
                                        scenario_provider, method,
                                        solver = NULL) {
  previous <- initial_equal_weight(asset_names, contract)
  weights <- matrix(NA_real_, nrow(periods), length(asset_names))
  audit <- vector("list", nrow(periods))
  for (i in seq_len(nrow(periods))) {
    decision <- as.Date(periods$decision_date[i])
    provided <- scenario_provider(i, decision)
    scenarios <- as.matrix(provided$scenarios)
    if (nrow(scenarios) != as.integer(contract$scenario_count) ||
        ncol(scenarios) != length(asset_names) ||
        any(!is.finite(scenarios)) || any(scenarios <= 0)) {
      benchmark_protocol_error(method, " scenario provider returned invalid data at ", decision)
    }
    latest_input_date <- as.Date(provided$latest_input_date)
    if (length(latest_input_date) != 1L || is.na(latest_input_date) ||
        latest_input_date > decision) {
      benchmark_protocol_error(method, " used future information at ", decision)
    }
    pretrade <- previous
    if (i > 1L && identical(
        contract$turnover_convention %||% "target_to_target_v1",
        "drifted_pretrade_v1")) {
      if (as.Date(periods$holding_end_date[i - 1L]) > decision) {
        benchmark_protocol_error(
          method, " previous holding outcome is unavailable at ", decision)
      }
      pretrade <- benchmark_pretrade_weight(
        previous, provided$previous_asset_gross, contract,
        sprintf("%s pretrade state at %s", method, decision))
    }
    year_fraction <- benchmark_period_year_fraction(periods[i, ], contract)
    result <- solve_constrained_weights(
      scenario_objective(scenarios, pretrade, contract, year_fraction), previous,
      asset_names, contract, context = sprintf("%s at %s", method, decision),
      solver = solver)
    weights[i, ] <- result$weights
    previous <- result$weights
    audit[[i]] <- data.frame(
      method = method, decision_date = decision,
      latest_input_date = latest_input_date,
      convergence = result$convergence, iterations = result$iterations,
      solver_attempts = result$solver_attempts,
      solver_attempt_codes = result$solver_attempt_codes,
      solver_attempt_messages = result$solver_attempt_messages,
      solver_total_iterations = result$solver_total_iterations,
      objective = result$objective,
      objective_year_fraction = year_fraction,
      objective_turnover_convention = contract$turnover_convention %||%
        "target_to_target_v1",
      solver_message = result$message,
      constraint_residual = result$constraint_residual,
      scenario_seed = as.integer(provided$scenario_seed),
      stringsAsFactors = FALSE)
  }
  list(weights = canonical_weight_log(periods, weights, asset_names, contract),
       audit = do.call(rbind, audit))
}

rank_to_training_distribution <- function(values, training_values) {
  sorted <- sort(as.numeric(training_values))
  probabilities <- (seq_along(sorted) - 0.5) / length(sorted)
  pmin(pmax(stats::approx(sorted, probabilities, xout = values,
                          rule = 2, ties = "ordered")$y, 1e-6), 1 - 1e-6)
}

serial_conditional_pit_benchmark <- function(previous_u, current_u, model) {
  rho <- as.numeric(model$rho); nu <- as.numeric(model$nu)
  previous_t <- stats::qt(pmin(pmax(previous_u, 1e-8), 1 - 1e-8), df = nu)
  current_t <- stats::qt(pmin(pmax(current_u, 1e-8), 1 - 1e-8), df = nu)
  scale <- sqrt((nu + previous_t^2) * (1 - rho^2) / (nu + 1))
  stats::pt((current_t - rho * previous_t) / scale, df = nu + 1)
}

load_vine_context <- function(daily_log_returns, training_marginals_file,
                              nn_vine_model_dir) {
  require_benchmark_namespace("rvinecopulib")
  if (!file.exists(training_marginals_file)) {
    benchmark_protocol_error("Training marginal artifact not found: ", training_marginals_file)
  }
  artifact <- new.env(parent = emptyenv())
  load(training_marginals_file, envir = artifact)
  required <- c("copula_monthly_log", "copula_innovation_u", "serial_copulas",
                "asset_names", "training_cutoff")
  missing <- required[!vapply(required, exists, logical(1), envir = artifact,
                             inherits = FALSE)]
  if (length(missing)) {
    benchmark_protocol_error("Training marginal artifact is missing: ",
                             paste(missing, collapse = ", "))
  }
  if (!identical(as.character(artifact$asset_names), colnames(daily_log_returns))) {
    benchmark_protocol_error("Training artifact asset order differs from realised data.")
  }
  vine_functions <- ensure_dynamic_vine_functions()
  nn_fit <- vine_functions$load_nn_dynamic_vine_fit(nn_vine_model_dir)
  if (as.integer(nn_fit$training_observations) != nrow(artifact$copula_innovation_u)) {
    benchmark_protocol_error("NN-vine fit and monthly innovation artifact disagree.")
  }
  monthly <- monthly_log_outcomes(daily_log_returns)
  raw_u <- vapply(seq_along(artifact$asset_names), function(j) {
    rank_to_training_distribution(monthly$log_returns[, j],
                                  artifact$copula_monthly_log[, j])
  }, numeric(nrow(monthly$log_returns)))
  colnames(raw_u) <- artifact$asset_names
  innovations <- matrix(NA_real_, nrow(raw_u) - 1L, ncol(raw_u),
                        dimnames = list(NULL, artifact$asset_names))
  for (j in seq_along(artifact$asset_names)) {
    innovations[, j] <- serial_conditional_pit_benchmark(
      head(raw_u[, j], -1L), tail(raw_u[, j], -1L), artifact$serial_copulas[[j]])
  }
  innovations <- pmin(pmax(innovations, 1e-6), 1 - 1e-6)
  list(
    artifact = artifact, nn_fit = nn_fit, monthly = monthly,
    daily_log_returns = daily_log_returns,
    raw_u = raw_u, innovations = innovations,
    innovation_dates = as.Date(monthly$periods$holding_end_date[-1L]),
    historical_sorted = lapply(seq_along(artifact$asset_names), function(j)
      sort(artifact$copula_monthly_log[, j])))
}

scale_training_vine <- function(vine, dependence_scale = 1) {
  require_benchmark_namespace("rvinecopulib")
  pair_copulas <- lapply(vine$pair_copulas, function(tree) {
    lapply(tree, function(pair) {
      if (!pair$family %in% c("student", "t")) {
        benchmark_protocol_error("Static benchmark expects all-t backbone edges.")
      }
      rho <- tanh(dependence_scale * atanh(as.numeric(pair$parameters[1L])))
      rvinecopulib::bicop_dist(
        "t", rotation = 0L,
        parameters = c(pmax(pmin(rho, 0.995), -0.995),
                       as.numeric(pair$parameters[2L])))
    })
  })
  rvinecopulib::vinecop_dist(pair_copulas = pair_copulas,
                            structure = vine$structure)
}

simulate_monthly_vine_scenarios <- function(vine, context, latest_index,
                                            n_draws, seed) {
  simulate_monthly_vine_scenarios_from_state(
    vine = vine, asset_names = context$artifact$asset_names,
    serial_copulas = context$artifact$serial_copulas,
    previous_u = context$raw_u[latest_index, ],
    historical_sorted = context$historical_sorted,
    n_draws = n_draws, seed = seed)
}

simulate_monthly_vine_scenarios_from_state <- function(
    vine, asset_names, serial_copulas, previous_u, historical_sorted,
    n_draws, seed) {
  set.seed(as.integer(seed))
  innovations <- rvinecopulib::rvinecop(as.integer(n_draws), vine,
                                        cores = 1L)
  previous_u <- matrix(as.numeric(previous_u), nrow = 1L)
  previous_u <- previous_u[rep(1L, nrow(innovations)), , drop = FALSE]
  current_u <- innovations
  for (j in seq_along(asset_names)) {
    model <- serial_copulas[[j]]
    rho <- as.numeric(model$rho); nu <- as.numeric(model$nu)
    previous_t <- stats::qt(pmin(pmax(previous_u[, j], 1e-8), 1 - 1e-8), df = nu)
    current_t <- rho * previous_t +
      sqrt((nu + previous_t^2) * (1 - rho^2) / (nu + 1)) *
      stats::qt(pmin(pmax(innovations[, j], 1e-8), 1 - 1e-8), df = nu + 1)
    current_u[, j] <- stats::pt(current_t, df = nu)
  }
  out <- matrix(NA_real_, nrow(current_u), length(asset_names))
  for (j in seq_along(asset_names)) {
    sorted <- historical_sorted[[j]]
    probabilities <- (seq_along(sorted) - 0.5) / length(sorted)
    out[, j] <- exp(stats::approx(probabilities, sorted, xout = current_u[, j],
                                  rule = 2)$y)
  }
  colnames(out) <- asset_names
  out
}

fit_rolling_vine_state <- function(context, raw_available, lookback) {
  selected <- tail(raw_available, min(as.integer(lookback), length(raw_available)))
  logs <- context$monthly$log_returns[selected, , drop = FALSE]
  if (nrow(logs) < 30L) benchmark_protocol_error("Rolling vine needs 30 monthly observations.")
  raw_u <- apply(logs, 2L, function(column)
    (rank(column, ties.method = "average") - 0.5) / length(column))
  dimnames(raw_u) <- dimnames(logs)
  serial_copulas <- lapply(seq_len(ncol(raw_u)), function(j) {
    pair <- cbind(head(raw_u[, j], -1L), tail(raw_u[, j], -1L))
    fit <- tryCatch(
      rvinecopulib::bicop(pair, family_set = "t", selcrit = "bic"),
      error = function(error) error)
    if (inherits(fit, "error") || length(fit$parameters) < 2L ||
        any(!is.finite(fit$parameters[1:2]))) {
      benchmark_protocol_error("Rolling serial t-copula fit failed for asset ",
                               context$artifact$asset_names[j], ".")
    }
    list(rho = pmax(pmin(as.numeric(fit$parameters[1L]), 0.95), -0.95),
         nu = pmax(as.numeric(fit$parameters[2L]), 2.05))
  })
  innovations <- matrix(NA_real_, nrow(raw_u) - 1L, ncol(raw_u),
                        dimnames = list(NULL, colnames(raw_u)))
  for (j in seq_len(ncol(raw_u))) {
    innovations[, j] <- serial_conditional_pit_benchmark(
      head(raw_u[, j], -1L), tail(raw_u[, j], -1L), serial_copulas[[j]])
  }
  innovations <- pmin(pmax(innovations, 1e-6), 1 - 1e-6)
  vine_functions <- ensure_dynamic_vine_functions()
  order <- vine_functions$select_dvine_order(innovations)
  truncation <- suppressWarnings(as.numeric(context$nn_fit$truncation_level))
  active_trees <- if (length(truncation) == 1L && is.finite(truncation)) {
    min(max(1L, as.integer(truncation)), ncol(innovations) - 1L)
  } else ncol(innovations) - 1L
  vine <- rvinecopulib::vinecop(
    innovations, structure = rvinecopulib::dvine_structure(order),
    family_set = "t", trunc_lvl = active_trees, selcrit = "bic")
  list(
    vine = vine, serial_copulas = serial_copulas,
    previous_u = tail(raw_u, 1L),
    historical_sorted = lapply(seq_len(ncol(logs)), function(j) sort(logs[, j])))
}

vine_provider_factory <- function(kind = c("static", "rolling", "dynamic_nn"),
                                  context, periods, contract) {
  kind <- match.arg(kind)
  asset_names <- context$artifact$asset_names
  static_vine <- scale_training_vine(
    context$nn_fit$backbone, context$nn_fit$dependence_scale %||% 1)
  function(index, decision_date) {
    available <- which(context$innovation_dates <= as.Date(decision_date))
    raw_available <- which(as.Date(context$monthly$periods$holding_end_date) <=
                             as.Date(decision_date))
    if (length(available) < 30L || !length(raw_available)) {
      benchmark_protocol_error(kind, " vine has insufficient causal history at ",
                               decision_date)
    }
    if (identical(kind, "static")) {
      vine <- static_vine
    } else if (identical(kind, "rolling")) {
      rolling_state <- fit_rolling_vine_state(
        context, raw_available, contract$rolling_vine_lookback_months)
      vine <- rolling_state$vine
    } else {
      selected <- available
      u <- context$innovations[selected, , drop = FALSE]
      vine_functions <- ensure_dynamic_vine_functions()
      states <- vine_functions$derive_nn_states(u)
      vine <- vine_functions$build_nn_vine_sequence(
        context$nn_fit, u, states$z, states$sigma,
        rebal_dates = tail(context$innovation_dates[selected], 1L),
        all_dates = context$innovation_dates[selected])[[1L]]
    }
    scenario_seed <- as.integer(contract$scenario_seed) +
      switch(kind, static = 100000L, rolling = 200000L, dynamic_nn = 300000L) +
      as.integer(index)
    scenarios <- if (identical(kind, "rolling")) {
      simulate_monthly_vine_scenarios_from_state(
        vine, asset_names, rolling_state$serial_copulas,
        rolling_state$previous_u, rolling_state$historical_sorted,
        as.integer(contract$scenario_count), scenario_seed)
    } else {
      simulate_monthly_vine_scenarios(
        vine, context, tail(raw_available, 1L),
        as.integer(contract$scenario_count), scenario_seed)
    }
    list(
      scenarios = scenarios,
      latest_input_date = max(context$monthly$periods$holding_end_date[raw_available]),
      scenario_seed = scenario_seed,
      previous_asset_gross = if (index > 1L) realised_gross_for_period(
        context$daily_log_returns,
        periods$decision_date[index - 1L],
        periods$holding_end_date[index - 1L]) else NULL)
  }
}

generate_vine_optimizers <- function(daily_log_returns, periods, contract,
                                     training_marginals_file,
                                     nn_vine_model_dir, solver = NULL) {
  context <- load_vine_context(daily_log_returns, training_marginals_file,
                               nn_vine_model_dir)
  asset_names <- context$artifact$asset_names
  kinds <- c(static_vine = "static", rolling_vine = "rolling",
             dynamic_nn_vine = "dynamic_nn")
  lapply(names(kinds), function(name) {
    generate_scenario_optimizer(
      periods, asset_names, contract,
      vine_provider_factory(kinds[[name]], context, periods, contract),
      method = name, solver = solver)
  }) |> setNames(names(kinds))
}
