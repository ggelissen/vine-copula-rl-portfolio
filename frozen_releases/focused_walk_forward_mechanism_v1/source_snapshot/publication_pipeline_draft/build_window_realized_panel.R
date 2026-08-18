#!/usr/bin/env Rscript
# Build one immutable common-path realized panel from a frozen window input.

suppressPackageStartupMessages(library(xts))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5L) {
  stop(paste(
    "Usage: build_window_realized_panel.R <daily_log_returns.csv>",
    "<return_input_manifest.json> <evaluation_periods.csv>",
    "<expected_window_id> <output_dir>"))
}
returns_file <- args[[1L]]
manifest_file <- args[[2L]]
periods_file <- args[[3L]]
expected_window_id <- args[[4L]]
output_dir <- args[[5L]]
for (path in c(returns_file, manifest_file, periods_file)) {
  if (!file.exists(path)) stop("Required realized-panel input not found: ", path)
}
if (dir.exists(output_dir) || file.exists(output_dir)) {
  stop("Realized-panel output already exists: ", output_dir)
}

source("helper/load_data.r")
source("helper/time_split.r")
returns <- load_returns(returns_file, "daily_log_returns", manifest_file)
assets <- colnames(returns)
periods <- read.csv(periods_file, stringsAsFactors = FALSE,
                    check.names = FALSE)
required <- c("window_id", "decision_date", "holding_end_date")
if (!all(required %in% names(periods))) {
  stop("Period file lacks the canonical window/date columns.")
}
periods <- periods[required]
periods$window_id <- as.character(periods$window_id)
periods$decision_date <- as.Date(periods$decision_date)
periods$holding_end_date <- as.Date(periods$holding_end_date)
if (nrow(periods) != 24L || anyNA(periods$decision_date) ||
    anyNA(periods$holding_end_date) ||
    !identical(unique(periods$window_id), expected_window_id) ||
    any(periods$holding_end_date <= periods$decision_date) ||
    is.unsorted(periods$decision_date, strictly = TRUE)) {
  stop("The window must contain exactly 24 valid, ordered holding periods.")
}

return_dates <- as.Date(index(returns))
gross <- matrix(NA_real_, nrow(periods), length(assets),
                dimnames = list(NULL, paste0("g_", assets)))
trading_days <- integer(nrow(periods))
for (index in seq_len(nrow(periods))) {
  decision <- periods$decision_date[index]
  holding_end <- periods$holding_end_date[index]
  selected <- which(return_dates > decision & return_dates <= holding_end)
  if (!length(selected)) stop("No realized returns in holding period ", index)
  if (max(return_dates[selected]) != holding_end) {
    stop("Holding end is absent from frozen returns at period ", index)
  }
  trading_days[index] <- length(selected)
  gross[index, ] <- exp(colSums(as.matrix(returns[selected, , drop = FALSE])))
}
if (any(!is.finite(gross)) || any(gross <= 0)) {
  stop("Realized gross returns must be finite and positive.")
}

last_calendar_day <- function(date) {
  first <- as.Date(format(as.Date(date), "%Y-%m-01"))
  as.Date(seq(first, by = "1 month", length.out = 2L)[2L]) - 1L
}
days_from_month_end <- vapply(seq_len(nrow(periods)), function(i) {
  date <- periods$holding_end_date[i]
  as.integer(last_calendar_day(date) - date)
}, integer(1))
complete <- trading_days >= 15L & days_from_month_end >= 0L &
  days_from_month_end <= 7L
if (sum(complete) < 20L) {
  stop("Fewer than 20 of 24 periods pass the frozen completeness rule.")
}
panel <- data.frame(periods, trading_days = trading_days,
                    is_complete_period = complete, gross,
                    check.names = FALSE)
calendar <- data.frame(periods, trading_days = trading_days,
                       is_complete_period = complete,
                       days_from_calendar_month_end = days_from_month_end,
                       check.names = FALSE)

parent <- dirname(output_dir)
dir.create(parent, recursive = TRUE, showWarnings = FALSE)
temporary <- tempfile(".realized_window_", tmpdir = parent)
dir.create(temporary)
on.exit(if (dir.exists(temporary)) unlink(temporary, recursive = TRUE), add = TRUE)
write.csv(panel, file.path(temporary, "realized_asset_gross.csv"),
          row.names = FALSE)
write.csv(calendar, file.path(temporary, "calendar_audit.csv"),
          row.names = FALSE)
manifest <- data.frame(
  window_id = expected_window_id, rows = nrow(panel),
  complete_periods = sum(complete), asset_count = length(assets),
  first_decision = as.character(min(periods$decision_date)),
  last_holding_end = as.character(max(periods$holding_end_date)),
  returns_md5 = unname(tools::md5sum(returns_file)),
  periods_md5 = unname(tools::md5sum(periods_file)),
  stringsAsFactors = FALSE)
write.csv(manifest, file.path(temporary, "realized_panel_manifest.csv"),
          row.names = FALSE)
if (!file.rename(temporary, output_dir)) {
  stop("Could not atomically publish the realized window panel.")
}
cat(sprintf("Built %d-period common realized panel for %s (%d assets).\n",
            nrow(panel), expected_window_id, length(assets)))
