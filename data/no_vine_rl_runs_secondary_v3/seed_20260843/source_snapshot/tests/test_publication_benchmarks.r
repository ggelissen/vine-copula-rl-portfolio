#!/usr/bin/env Rscript
# Development tests for every causal benchmark family.  Run from repository
# root before freezing evaluation code.  No result from the locked holdout is
# printed or used for tuning.

suppressPackageStartupMessages({
  library(xts)
  library(jsonlite)
})
source("helper/load_data.r")
source("helper/time_split.r")
source("publication_pipeline_draft/benchmark_weights.R")

contract <- read_benchmark_contract(
  "publication_pipeline_draft/config/benchmark_contract.json")
contract$scenario_count <- 256L
# Keep the frozen optimizer budget.  Reducing maxeval inside a protocol test
# can manufacture NLOPT_MAXEVAL_REACHED (code 5) even though the production
# contract has not failed.  The strict gate below still proves that a genuine
# code-5 result is rejected; successful integration solves must use the same
# convergence budget declared by the benchmark contract.
frozen_optimizer_maxeval <- as.integer(contract$optimizer_maxeval)
if (length(frozen_optimizer_maxeval) != 1L ||
    is.na(frozen_optimizer_maxeval) || frozen_optimizer_maxeval < 1L) {
  stop("The benchmark contract declares an invalid optimizer_maxeval.",
       call. = FALSE)
}
if (!identical(as.integer(contract$optimizer_max_restarts), 2L)) {
  stop("The benchmark contract must freeze two deterministic restarts.",
       call. = FALSE)
}

expect_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}
expect_error <- function(expression, pattern) {
  caught <- tryCatch({ force(expression); NULL }, error = identity)
  if (is.null(caught) || !grepl(pattern, conditionMessage(caught), fixed = TRUE)) {
    stop("Expected error containing '", pattern, "'.", call. = FALSE)
  }
}

expect_true(
  identical(sort(validate_optimizer_convergence_codes(
    contract$optimizer_allowed_convergence_codes)), 1:4),
  "The frozen optimizer convergence-code set is not exactly {1,2,3,4}.")
for (invalid_codes in list(c(1L, 2L, 3L), c(1L, 2L, 3L, 4L, 5L),
                           c(1L, 2L, 2L, 4L), c(1, 2, 3, 4.5), NULL)) {
  mutated_contract <- contract
  mutated_contract$optimizer_allowed_convergence_codes <- invalid_codes
  expect_error(
    validate_optimizer_convergence_codes(
      mutated_contract$optimizer_allowed_convergence_codes),
    "must be exactly the unique integer set {1,2,3,4}")
}
invalid_contract_path <- tempfile(fileext = ".json")
invalid_file_contract <- contract
invalid_file_contract$optimizer_allowed_convergence_codes <- c(1L, 2L, 2L, 4L)
jsonlite::write_json(invalid_file_contract, invalid_contract_path, auto_unbox = TRUE)
expect_error(
  read_benchmark_contract(invalid_contract_path),
  "must be exactly the unique integer set {1,2,3,4}")
unlink(invalid_contract_path)

audit_bind_test <- bind_benchmark_audits(list(
  data.frame(method = "analytic", objective = 1),
  data.frame(method = "scenario", objective = 2, scenario_seed = 10L)
))
expect_true(nrow(audit_bind_test) == 2L,
            "Benchmark audit binding lost rows.")
expect_true("scenario_seed" %in% names(audit_bind_test) &&
              is.na(audit_bind_test$scenario_seed[1L]),
            "Benchmark audit binding did not fill method-specific columns.")

drift_contract <- contract
drift_contract$turnover_convention <- "drifted_pretrade_v1"
drift_contract$financing_proration <- "actual_calendar_days_v1"
drift_contract$day_count_basis <- 365
drifted <- benchmark_pretrade_weight(
  c(0.6, 0.4), c(1.2, 0.9), drift_contract, "drift test")
expected_drifted <- c(0.6, 0.4) * c(1.2, 0.9) /
  (1 + sum(c(0.6, 0.4) * (c(1.2, 0.9) - 1)))
expect_true(isTRUE(all.equal(drifted, expected_drifted, tolerance = 1e-15)),
            "Benchmark pretrade drift is not self-financing.")
fraction <- benchmark_period_year_fraction(
  data.frame(decision_date = as.Date("2030-01-31"),
             holding_end_date = as.Date("2030-02-28")), drift_contract)
expect_true(isTRUE(all.equal(fraction, 28 / 365, tolerance = 1e-15)),
            "Benchmark objective does not use actual period fractions.")

returns <- load_returns()
split <- split_monthly_periods(
  build_monthly_periods(returns, min_history = 250L), 24L)
# Use only the historical training prefix.  The locked 24-month outcomes are
# neither passed to a generator nor summarized by these development tests.
returns <- returns[paste0("/", max(split$train$holding_end_date))]
test_period_count <- suppressWarnings(as.integer(
  Sys.getenv("BENCHMARK_TEST_PERIODS", "2")))
if (length(test_period_count) != 1L || is.na(test_period_count) ||
    test_period_count < 1L) {
  stop("BENCHMARK_TEST_PERIODS must be one positive integer.", call. = FALSE)
}
development_periods <- head(split$train, -2L)
if (test_period_count > nrow(development_periods)) {
  stop("BENCHMARK_TEST_PERIODS exceeds the available training-only periods.",
       call. = FALSE)
}
periods <- tail(development_periods, test_period_count)
periods$window_id <- contract$evaluation_id
asset_names <- colnames(returns)

# Future-data perturbation: everything strictly after the first decision is
# changed materially.  The first weight must remain bitwise/numerically equal.
perturbed <- returns
future <- as.Date(zoo::index(perturbed)) > as.Date(periods$decision_date[1L])
perturbed[future, ] <- perturbed[future, ] + 0.01

equal_a <- generate_equal_weight(periods, asset_names, contract)
equal_b <- generate_equal_weight(periods, asset_names, contract)
expect_true(identical(equal_a, equal_b), "Equal weight is not deterministic.")

mv_a <- generate_shrinkage_mean_variance(returns, periods, contract)
mv_b <- generate_shrinkage_mean_variance(returns, periods, contract)
mv_future <- generate_shrinkage_mean_variance(perturbed, periods, contract)
expect_true(isTRUE(all.equal(mv_a$weights, mv_b$weights, tolerance = 0)),
            "Mean-variance is not deterministically reproducible.")
expect_true(isTRUE(all.equal(mv_a$weights[1L, ], mv_future$weights[1L, ], tolerance = 0)),
            "Mean-variance used a future return.")

# A deterministic covariance provider isolates DCC plumbing/causality from the
# expensive rmgarch integration, which is tested separately below when enabled.
mock_dcc <- function(daily_history, horizon, seed) {
  covariance <- stats::cov(as.matrix(tail(daily_history, 500L))) * horizon
  list(covariance = covariance + diag(1e-8, ncol(covariance)),
       fit_convergence = 0L)
}
dcc_a <- generate_dcc_garch(returns, periods, contract,
                            dcc_fit_function = mock_dcc)
dcc_b <- generate_dcc_garch(returns, periods, contract,
                            dcc_fit_function = mock_dcc)
dcc_future <- generate_dcc_garch(perturbed, periods, contract,
                                 dcc_fit_function = mock_dcc)
expect_true(isTRUE(all.equal(dcc_a$weights, dcc_b$weights, tolerance = 0)),
            "DCC plumbing is not deterministic.")
expect_true(isTRUE(all.equal(dcc_a$weights[1L, ], dcc_future$weights[1L, ], tolerance = 0)),
            "DCC plumbing used a future return.")

# Static, rolling, and NN-vine optimizers share this tested direct-scenario
# engine.  The integration branch below additionally exercises each real vine.
scenario_provider <- function(index, decision_date) {
  set.seed(5000L + index)
  scenarios <- exp(matrix(rnorm(contract$scenario_count * length(asset_names),
                                sd = 0.04), ncol = length(asset_names)))
  list(scenarios = scenarios, latest_input_date = decision_date,
       scenario_seed = 5000L + index)
}
for (method in c("static_vine", "rolling_vine", "dynamic_nn_vine")) {
  first <- generate_scenario_optimizer(
    periods, asset_names, contract, scenario_provider, method)
  second <- generate_scenario_optimizer(
    periods, asset_names, contract, scenario_provider, method)
  expect_true(isTRUE(all.equal(first$weights, second$weights, tolerance = 0)),
              paste(method, "is not deterministic."))
}

failure_solver <- function(...) list(
  par = initial_equal_weight(asset_names, contract), convergence = -1L,
  message = "injected failure", value = 0)
expect_error(
  generate_scenario_optimizer(periods[1L, ], asset_names, contract,
                              scenario_provider, "injected", failure_solver),
  "did not meet the frozen convergence rule")

# NLopt code 5 means the maximum evaluation budget was reached. It produces a
# feasible point but is not evidence that the declared convergence criterion
# was met, so future releases must fail closed instead of silently scoring it.
maxeval_calls <- 0L
maxeval_solver <- function(...) {
  maxeval_calls <<- maxeval_calls + 1L
  list(par = initial_equal_weight(asset_names, contract), convergence = 5L,
       message = "NLOPT_MAXEVAL_REACHED", value = 0,
       iter = contract$optimizer_maxeval)
}
expect_error(
  generate_scenario_optimizer(periods[1L, ], asset_names, contract,
                              scenario_provider, "maxeval", maxeval_solver),
  "code=5")
expect_true(maxeval_calls == 1L + contract$optimizer_max_restarts,
            "The solver did not exhaust the frozen deterministic restart budget.")

continuation_calls <- 0L
continuation_solver <- function(...) {
  continuation_calls <<- continuation_calls + 1L
  list(par = initial_equal_weight(asset_names, contract),
       convergence = if (continuation_calls == 1L) 5L else 4L,
       message = if (continuation_calls == 1L)
         "NLOPT_MAXEVAL_REACHED" else "NLOPT_XTOL_REACHED",
       value = 0, iter = if (continuation_calls == 1L)
         contract$optimizer_maxeval else 1L)
}
continuation <- generate_scenario_optimizer(
  periods[1L, ], asset_names, contract, scenario_provider,
  "continuation", continuation_solver)
expect_true(continuation_calls == 2L,
            "The deterministic continuation was not used exactly once.")
expect_true(continuation$audit$solver_attempts[1L] == 2L &&
              identical(continuation$audit$solver_attempt_codes[1L], "5,4"),
            "The solver continuation audit is incomplete.")

for (object in list(equal_a, mv_a$weights, dcc_a$weights)) {
  assert_canonical_weight_log(object, periods, asset_names, contract)
}

if (tolower(Sys.getenv("RUN_EXPENSIVE_BENCHMARK_TESTS", "false")) %in%
    c("1", "true", "yes")) {
  dcc_real_a <- generate_dcc_garch(returns, periods, contract)
  dcc_real_b <- generate_dcc_garch(returns, periods, contract)
  dcc_real_future <- generate_dcc_garch(perturbed, periods, contract)
  expect_true(isTRUE(all.equal(dcc_real_a$weights, dcc_real_b$weights,
                               tolerance = 1e-12)),
              "Real DCC-GARCH is not reproducible.")
  expect_true(isTRUE(all.equal(dcc_real_a$weights[1L, ],
                               dcc_real_future$weights[1L, ], tolerance = 1e-12)),
              "Real DCC-GARCH used future data.")

  vine_a <- generate_vine_optimizers(
    returns, periods, contract,
    "data/training_marginal_results.RData", "data/nn_vine_models")
  vine_b <- generate_vine_optimizers(
    returns, periods, contract,
    "data/training_marginal_results.RData", "data/nn_vine_models")
  vine_future <- generate_vine_optimizers(
    perturbed, periods, contract,
    "data/training_marginal_results.RData", "data/nn_vine_models")
  for (method in names(vine_a)) {
    expect_true(isTRUE(all.equal(vine_a[[method]]$weights,
                                 vine_b[[method]]$weights, tolerance = 1e-12)),
                paste("Real", method, "is not reproducible."))
    expect_true(isTRUE(all.equal(vine_a[[method]]$weights[1L, ],
                                 vine_future[[method]]$weights[1L, ],
                                 tolerance = 1e-12)),
                paste("Real", method, "used future data."))
  }
}

cat("All publication benchmark protocol tests passed.\n")
