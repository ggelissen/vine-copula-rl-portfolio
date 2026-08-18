#!/usr/bin/env Rscript
# Generate every non-RL benchmark weight log in one fail-closed transaction.
# The output directory is immutable and is created only after all methods pass.

suppressPackageStartupMessages({
  library(yaml)
  library(xts)
})

args <- commandArgs(trailingOnly = TRUE)
config_file <- if (length(args) >= 1L) args[[1L]] else "config/config.yaml"
contract_file <- if (length(args) >= 2L) args[[2L]] else
  "publication_pipeline_draft/config/benchmark_contract.json"
output_dir <- if (length(args) >= 3L) args[[3L]] else
  "publication_eval/benchmark_weights"
if (!file.exists(config_file)) stop("Config file not found: ", config_file)
if (dir.exists(output_dir) || file.exists(output_dir)) {
  stop("Benchmark output is immutable and already exists: ", output_dir)
}

config <- yaml::yaml.load_file(config_file)
source("helper/load_data.r")
source("helper/time_split.r")
source("publication_pipeline_draft/benchmark_weights.R")
contract <- read_benchmark_contract(contract_file)

# Cross-check the independent benchmark contract against the training mandate.
cross_checks <- c(
  net_exposure = config$environment$net_exposure,
  gross_leverage = config$environment$gross_leverage,
  max_long_weight = config$environment$max_long_weight,
  max_short_weight = config$environment$max_short_weight,
  turnover_cost = config$evaluation$eval_kappa,
  annual_short_borrow_rate = config$environment$short_borrow_rate,
  annual_cash_borrow_rate = config$environment$cash_borrow_rate,
  crra_gamma = config$evaluation$eval_gamma,
  cvar_penalty = config$evaluation$eval_lambda)
for (name in names(cross_checks)) {
  if (abs(as.numeric(contract[[name]]) - as.numeric(cross_checks[[name]])) > 1e-12) {
    stop("Benchmark contract and master configuration disagree on ", name)
  }
}
if (as.integer(config$evaluation$periods) != 24L) {
  stop("Locked evaluation must contain exactly 24 periods.")
}

returns <- load_returns()
asset_names <- colnames(returns)
period_split <- split_monthly_periods(
  build_monthly_periods(returns, min_history = as.integer(config$vine$L)),
  as.integer(config$evaluation$periods))
validate_period_split(period_split, as.integer(config$evaluation$periods))
validate_return_evaluation_contract(
  returns, period_split, as.integer(config$evaluation$periods))
validate_return_model_contract(
  returns, as.integer(Sys.getenv("REF_COL", as.character(config$vine$ref_col))),
  as.integer(Sys.getenv("VINE_TRUNCATION_LEVEL", if (is.null(config$vine$truncation_level))
    "0" else as.character(config$vine$truncation_level))))
periods <- period_split$evaluation
periods$window_id <- Sys.getenv("EVAL_WINDOW_ID", contract$evaluation_id)

requested <- Sys.getenv(
  "BENCHMARK_METHODS",
  unset = "equal_weight,shrinkage_mean_variance,dcc_garch,static_vine,rolling_vine,dynamic_nn_vine")
requested <- trimws(strsplit(requested, ",", fixed = TRUE)[[1L]])
allowed <- c("equal_weight", "shrinkage_mean_variance", "dcc_garch",
             "static_vine", "rolling_vine", "dynamic_nn_vine")
if (!length(requested) || any(!requested %in% allowed) || anyDuplicated(requested)) {
  stop("BENCHMARK_METHODS contains an invalid or duplicate method.")
}

results <- list(); audits <- list()
if ("equal_weight" %in% requested) {
  results$equal_weight <- generate_equal_weight(periods, asset_names, contract)
  audits$equal_weight <- data.frame(
    method = "equal_weight", decision_date = periods$decision_date,
    latest_input_date = periods$decision_date,
    convergence = NA_integer_, iterations = NA_integer_, objective = NA_real_,
    stringsAsFactors = FALSE)
}
if ("shrinkage_mean_variance" %in% requested) {
  generated <- generate_shrinkage_mean_variance(returns, periods, contract)
  results$shrinkage_mean_variance <- generated$weights
  audits$shrinkage_mean_variance <- generated$audit
}
if ("dcc_garch" %in% requested) {
  generated <- generate_dcc_garch(returns, periods, contract)
  results$dcc_garch <- generated$weights
  audits$dcc_garch <- generated$audit
}
vine_methods <- intersect(c("static_vine", "rolling_vine", "dynamic_nn_vine"),
                          requested)
if (length(vine_methods)) {
  vines <- generate_vine_optimizers(
    returns, periods, contract,
    training_marginals_file = Sys.getenv(
      "TRAINING_MARGINALS_FILE", config$vine$training_marginals_file),
    nn_vine_model_dir = Sys.getenv("NN_VINE_MODEL_DIR", config$vine$nn_model_dir),
    methods = vine_methods)
  for (name in vine_methods) {
    results[[name]] <- vines[[name]]$weights
    audits[[name]] <- vines[[name]]$audit
  }
}

# Validate all logs as one common calendar/asset/mandate family before writing.
for (name in names(results)) {
  assert_canonical_weight_log(results[[name]], periods, asset_names, contract,
                              paste(name, "weights"))
}
reference_keys <- results[[1L]][c("window_id", "decision_date", "holding_end_date")]
for (name in names(results)[-1L]) {
  if (!identical(reference_keys,
                 results[[name]][c("window_id", "decision_date", "holding_end_date")])) {
    stop("Benchmark calendars differ: ", name)
  }
}

parent <- dirname(output_dir)
dir.create(parent, recursive = TRUE, showWarnings = FALSE)
temporary <- tempfile(pattern = ".benchmark_weights_", tmpdir = parent)
dir.create(temporary, recursive = TRUE, showWarnings = FALSE)
on.exit(if (dir.exists(temporary)) unlink(temporary, recursive = TRUE), add = TRUE)
for (name in names(results)) {
  write.csv(results[[name]], file.path(temporary, paste0("weights_", name, ".csv")),
            row.names = FALSE)
}
write.csv(bind_benchmark_audits(audits), file.path(temporary, "solver_audit.csv"),
          row.names = FALSE)
manifest <- data.frame(
  method = names(results),
  rows = vapply(results, nrow, integer(1)),
  first_decision = vapply(results, function(x) as.character(min(x$decision_date)),
                          character(1)),
  last_decision = vapply(results, function(x) as.character(max(x$decision_date)),
                         character(1)),
  weight_file = paste0("weights_", names(results), ".csv"),
  md5 = unname(tools::md5sum(file.path(
    temporary, paste0("weights_", names(results), ".csv")))),
  stringsAsFactors = FALSE)
write.csv(manifest, file.path(temporary, "benchmark_manifest.csv"), row.names = FALSE)
if (!file.copy(contract_file, file.path(temporary, "benchmark_contract.json"))) {
  stop("Could not preserve the benchmark contract in the output.")
}
if (!file.rename(temporary, output_dir)) {
  stop("Could not atomically publish benchmark output directory.")
}
cat(sprintf("Generated %d causal benchmark logs in %s\n",
            length(results), normalizePath(output_dir, winslash = "/", mustWork = TRUE)))
