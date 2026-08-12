#!/usr/bin/env Rscript
# Generate the five added financial baselines for a frozen train/validation or
# future-evaluation calendar. This entry point deliberately does not replace
# generate_benchmark_weights.R, so the consumed v4 evidence remains immutable.

suppressPackageStartupMessages({
  library(xts)
})

args <- commandArgs(trailingOnly = TRUE)
returns_file <- if (length(args) >= 1L) args[[1L]] else
  "frozen_releases/global_liquid_etf_18/development_daily_log_returns.csv"
periods_file <- if (length(args) >= 2L) args[[2L]] else
  "publication_eval/external_periods.csv"
contract_file <- if (length(args) >= 3L) args[[3L]] else
  "publication_pipeline_draft/config/benchmark_contract_v2.json"
output_dir <- if (length(args) >= 4L) args[[4L]] else
  "publication_eval/extended_benchmark_weights"
return_manifest_file <- if (length(args) >= 5L) args[[5L]] else ""

for (path in c(returns_file, periods_file, contract_file, return_manifest_file)) {
  if (!file.exists(path)) stop("Required benchmark input not found: ", path)
}
if (dir.exists(output_dir) || file.exists(output_dir)) {
  stop("Extended benchmark output is immutable and already exists: ", output_dir)
}

source("publication_pipeline_draft/benchmark_weights.R")
source("publication_pipeline_draft/extended_benchmark_weights.R")
source("helper/load_data.r")
contract <- read_extended_benchmark_contract(contract_file)

returns <- load_returns(returns_file, "daily_log_returns", return_manifest_file)
asset_names <- colnames(returns)

periods <- read.csv(periods_file, stringsAsFactors = FALSE)
required_periods <- c("window_id", "decision_date", "holding_end_date")
if (!all(required_periods %in% names(periods))) {
  stop("Period file is missing canonical evaluation calendar columns.")
}
periods <- periods[required_periods]
periods$decision_date <- as.Date(periods$decision_date)
periods$holding_end_date <- as.Date(periods$holding_end_date)
if (anyNA(periods$decision_date) || anyNA(periods$holding_end_date) ||
    any(periods$decision_date >= periods$holding_end_date) ||
    is.unsorted(periods$decision_date, strictly = TRUE)) {
  stop("Evaluation period calendar is invalid.")
}

requested <- Sys.getenv(
  "EXTENDED_BENCHMARK_METHODS",
  unset = paste(c("minimum_variance", "risk_parity", "mean_cvar",
                  "momentum_tilt", "black_litterman_momentum_views"),
                collapse = ","))
requested <- trimws(strsplit(requested, ",", fixed = TRUE)[[1L]])
generated <- generate_extended_financial_benchmarks(
  returns, periods, contract, requested)
results <- lapply(generated, `[[`, "weights")
audits <- lapply(generated, `[[`, "audit")

for (name in names(results)) {
  assert_canonical_weight_log(results[[name]], periods, asset_names, contract,
                              paste(name, "weights"))
}
reference <- results[[1L]][required_periods]
for (name in names(results)[-1L]) {
  if (!identical(reference, results[[name]][required_periods])) {
    stop("Extended benchmark calendars differ: ", name)
  }
}

parent <- dirname(output_dir)
dir.create(parent, recursive = TRUE, showWarnings = FALSE)
temporary <- tempfile(pattern = ".extended_benchmarks_", tmpdir = parent)
dir.create(temporary, recursive = TRUE, showWarnings = FALSE)
on.exit(if (dir.exists(temporary)) unlink(temporary, recursive = TRUE), add = TRUE)
for (name in names(results)) {
  write.csv(results[[name]], file.path(temporary, paste0("weights_", name, ".csv")),
            row.names = FALSE)
}
write.csv(bind_benchmark_audits(audits),
          file.path(temporary, "solver_audit.csv"), row.names = FALSE)
manifest <- data.frame(
  method = names(results),
  rows = vapply(results, nrow, integer(1)),
  weight_file = paste0("weights_", names(results), ".csv"),
  md5 = unname(tools::md5sum(file.path(
    temporary, paste0("weights_", names(results), ".csv")))),
  stringsAsFactors = FALSE)
write.csv(manifest, file.path(temporary, "benchmark_manifest.csv"), row.names = FALSE)
if (!file.copy(contract_file, file.path(temporary, "benchmark_contract.json"))) {
  stop("Could not preserve the benchmark contract.")
}
if (!file.rename(temporary, output_dir)) {
  stop("Could not atomically publish extended benchmark output.")
}
cat(sprintf("Generated %d extended causal benchmarks in %s\n",
            length(results), normalizePath(output_dir, winslash = "/",
                                           mustWork = TRUE)))
