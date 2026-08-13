#!/usr/bin/env Rscript
# Training-prefix-only tests for the extended financial benchmark family.

suppressPackageStartupMessages({
  library(xts)
  library(jsonlite)
})
source("helper/load_data.r")
source("helper/time_split.r")
source("publication_pipeline_draft/benchmark_weights.R")
source("publication_pipeline_draft/extended_benchmark_weights.R")

expect_true <- function(value, message) {
  if (!isTRUE(value)) stop(message, call. = FALSE)
}

contract <- read_extended_benchmark_contract(
  "publication_pipeline_draft/config/benchmark_contract_v2.json")
contract$scenario_count <- 256L
returns <- load_returns()
split <- split_monthly_periods(
  build_monthly_periods(returns, min_history = 250L), 24L)
returns <- returns[paste0("/", max(split$train$holding_end_date))]
periods <- tail(head(split$train, -2L), 1L)
periods$window_id <- contract$evaluation_id
asset_names <- colnames(returns)

perturbed <- returns
future <- as.Date(zoo::index(perturbed)) > as.Date(periods$decision_date[1L])
perturbed[future, ] <- perturbed[future, ] + 0.02

methods <- c("minimum_variance", "risk_parity", "mean_cvar",
             "momentum_tilt", "black_litterman_momentum_views")
first <- generate_extended_financial_benchmarks(
  returns, periods, contract, methods)
second <- generate_extended_financial_benchmarks(
  returns, periods, contract, methods)
changed_future <- generate_extended_financial_benchmarks(
  perturbed, periods, contract, methods)

for (method in methods) {
  expect_true(isTRUE(all.equal(first[[method]]$weights,
                               second[[method]]$weights, tolerance = 0)),
              paste(method, "is not deterministic."))
  expect_true(isTRUE(all.equal(first[[method]]$weights,
                               changed_future[[method]]$weights, tolerance = 0)),
              paste(method, "used a future return."))
  assert_canonical_weight_log(
    first[[method]]$weights, periods, asset_names, contract,
    paste(method, "extended test"))
  expect_true(all(as.Date(first[[method]]$audit$latest_input_date) <=
                    as.Date(first[[method]]$audit$decision_date)),
              paste(method, "audit reports future data."))
  expect_true(all(first[[method]]$audit$convergence %in% 1:4),
              paste(method, "did not meet the frozen convergence rule."))
}

rp <- first$risk_parity$weights[1L, weight_columns(asset_names)]
expect_true(min(as.numeric(rp)) >= -contract$weight_tolerance,
            "Risk parity is not long-only.")

cat("All extended publication benchmark protocol tests passed.\n")
