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
contract$optimizer_maxeval <- 750L

expect_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}
expect_error <- function(expression, pattern) {
  caught <- tryCatch({ force(expression); NULL }, error = identity)
  if (is.null(caught) || !grepl(pattern, conditionMessage(caught), fixed = TRUE)) {
    stop("Expected error containing '", pattern, "'.", call. = FALSE)
  }
}

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
  "did not converge")

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
