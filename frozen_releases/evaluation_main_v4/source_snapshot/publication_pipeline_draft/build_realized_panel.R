#!/usr/bin/env Rscript
# Export the realized asset panel consumed by publication_pipeline.py.
# Run only after the seed sweep. This adapter reads returns but never reads a
# policy checkpoint or writes into the training directories.

suppressPackageStartupMessages({
  library(yaml)
  library(xts)
})

args <- commandArgs(trailingOnly = TRUE)
config_file <- if (length(args) >= 1L) args[[1L]] else "config/config.yaml"
output_dir <- if (length(args) >= 2L) args[[2L]] else "publication_eval/inputs"
if (!file.exists(config_file)) stop("Config file not found: ", config_file)
if (dir.exists(output_dir)) {
  stop("Output directory already exists; realized evaluation inputs are immutable: ", output_dir)
}

config <- yaml::yaml.load_file(config_file)
source("helper/load_data.r")
source("helper/time_split.r")

returns <- load_returns()
periods <- build_monthly_periods(returns, min_history = as.integer(config$vine$L))
split <- split_monthly_periods(periods, as.integer(config$evaluation$periods))
validate_period_split(split, as.integer(config$evaluation$periods))
evaluation <- split$evaluation
asset_names <- colnames(returns)

last_calendar_day <- function(date) {
  first <- as.Date(format(as.Date(date), "%Y-%m-01"))
  as.Date(seq(first, by = "1 month", length.out = 2L)[2L]) - 1L
}

gross <- do.call(rbind, lapply(seq_len(nrow(evaluation)), function(index) {
  as.numeric(realised_gross_for_period(
    returns, evaluation$decision_date[index], evaluation$holding_end_date[index]
  ))
}))
colnames(gross) <- paste0("g_", asset_names)
trading_days <- evaluation$holding_end_idx - evaluation$decision_idx
days_from_calendar_end <- as.integer(mapply(
  function(date) last_calendar_day(date) - as.Date(date),
  evaluation$holding_end_date
))
# This is a conservative, auditable calendar flag rather than an assumption
# that the final observed date is necessarily a true month end.
is_complete <- trading_days >= 15L & days_from_calendar_end >= 0L &
  days_from_calendar_end <= 7L

panel <- data.frame(
  window_id = "locked_oos_v1",
  decision_date = evaluation$decision_date,
  holding_end_date = evaluation$holding_end_date,
  trading_days = trading_days,
  is_complete_period = is_complete,
  gross,
  check.names = FALSE
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(panel, file.path(output_dir, "realized_asset_gross.csv"), row.names = FALSE)

equal_weight <- panel[c("window_id", "decision_date", "holding_end_date")]
for (asset in asset_names) equal_weight[[paste0("w_", asset)]] <-
  as.numeric(config$environment$net_exposure) / length(asset_names)
write.csv(equal_weight, file.path(output_dir, "weights_equal_weight.csv"), row.names = FALSE)

calendar_audit <- panel[c(
  "window_id", "decision_date", "holding_end_date", "trading_days",
  "is_complete_period"
)]
calendar_audit$days_from_calendar_month_end <- days_from_calendar_end
write.csv(calendar_audit, file.path(output_dir, "calendar_audit.csv"), row.names = FALSE)

cat(sprintf(
  "Exported %d locked periods (%d complete; %d shortened) to %s\n",
  nrow(panel), sum(panel$is_complete_period), sum(!panel$is_complete_period),
  normalizePath(output_dir, winslash = "/", mustWork = TRUE)
))
