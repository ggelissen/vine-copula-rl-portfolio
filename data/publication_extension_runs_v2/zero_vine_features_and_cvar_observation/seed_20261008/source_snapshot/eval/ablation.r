# Artifact-only ablation analysis.
# Every row must originate from an actually trained policy evaluated on the
# same realised holding periods. This file deliberately contains no synthetic
# metric fallback.

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

annualised_path_metrics <- function(net_returns, periods_per_year = 12L,
                                    risk_free = 0) {
  r <- as.numeric(net_returns)
  if (length(r) < 2L || any(!is.finite(r)) || any(r <= -1)) {
    stop("net_returns must contain at least two finite simple returns above -100%.")
  }
  excess <- r - risk_free / periods_per_year
  wealth <- cumprod(1 + r)
  drawdown <- wealth / cummax(c(1, wealth))[-1L] - 1
  cvar_cut <- unname(quantile(r, 0.05, type = 8))
  c(
    observations = length(r),
    total_return = tail(wealth, 1L) - 1,
    cagr = tail(wealth, 1L)^(periods_per_year / length(r)) - 1,
    annual_vol = sd(r) * sqrt(periods_per_year),
    sharpe_ratio = if (sd(excess) > 0) mean(excess) / sd(excess) *
      sqrt(periods_per_year) else NA_real_,
    max_drawdown = -min(drawdown),
    cvar05_loss = -mean(r[r <= cvar_cut])
  )
}

read_experiment_manifest <- function(manifest_file, required_columns) {
  if (!file.exists(manifest_file)) {
    stop(sprintf("Missing experiment manifest: %s. No metrics were fabricated.", manifest_file))
  }
  manifest <- fread(manifest_file)
  missing <- setdiff(required_columns, names(manifest))
  if (length(missing)) stop("Manifest is missing columns: ", paste(missing, collapse = ", "))
  if (!nrow(manifest)) stop("Experiment manifest has no completed runs.")
  missing_files <- manifest[!file.exists(log_file), unique(log_file)]
  if (length(missing_files)) stop("Missing run logs: ", paste(missing_files, collapse = ", "))
  manifest
}

read_realised_run <- function(path) {
  x <- fread(path)
  required <- c("date", "net_return")
  if (length(setdiff(required, names(x)))) {
    stop(sprintf("%s must contain date and net_return columns.", path))
  }
  x[, date := as.Date(date)]
  if (anyNA(x$date) || anyDuplicated(x$date) || any(!is.finite(x$net_return))) {
    stop(sprintf("%s has invalid, duplicated, or non-finite observations.", path))
  }
  setorder(x, date)
  x
}

run_ablation <- function(manifest_file = "config/ablation_manifest.csv",
                         output_file = "data/ablation_results.csv", ...) {
  manifest <- read_experiment_manifest(
    manifest_file, c("variant", "label", "seed", "log_file")
  )
  # Apples-to-apples dates are mandatory.
  runs <- lapply(manifest$log_file, read_realised_run)
  reference_dates <- runs[[1L]]$date
  if (!all(vapply(runs, function(x) identical(x$date, reference_dates), logical(1)))) {
    stop("Ablation logs do not cover identical realised holding periods.")
  }
  per_run <- rbindlist(lapply(seq_len(nrow(manifest)), function(i) {
    as.data.table(as.list(annualised_path_metrics(runs[[i]]$net_return)))[
      , `:=`(variant = manifest$variant[i], label = manifest$label[i],
               seed = manifest$seed[i])]
  }), fill = TRUE)
  metric_cols <- setdiff(names(per_run), c("variant", "label", "seed"))
  result <- per_run[, c(
    list(n_training_seeds = .N),
    setNames(lapply(.SD, mean, na.rm = TRUE), paste0(metric_cols, "_mean")),
    setNames(lapply(.SD, function(x) sd(x, na.rm = TRUE) / sqrt(sum(is.finite(x)))),
             paste0(metric_cols, "_se"))
  ), by = .(variant, label), .SDcols = metric_cols]
  dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
  fwrite(result, output_file)
  attr(result, "per_run") <- per_run
  result
}

plot_ablation <- function(ablation_df, metric = "sharpe_ratio",
                          save_path = "figures/ablation.pdf") {
  value <- paste0(metric, "_mean"); error <- paste0(metric, "_se")
  if (!all(c("label", value, error) %in% names(ablation_df))) {
    stop("Requested metric is absent from the artifact-derived ablation table.")
  }
  p <- ggplot(ablation_df, aes(x = reorder(label, get(value)), y = get(value))) +
    geom_col(fill = "#2C7FB8") +
    geom_errorbar(aes(ymin = get(value) - get(error),
                      ymax = get(value) + get(error)), width = 0.2) +
    coord_flip() + theme_bw() + labs(x = NULL, y = metric)
  dir.create(dirname(save_path), recursive = TRUE, showWarnings = FALSE)
  ggsave(save_path, p, width = 8, height = 5)
  p
}

ablation_summary <- function(ablation_df) {
  if (!"sharpe_ratio_mean" %in% names(ablation_df)) {
    stop("Expected artifact-derived sharpe_ratio_mean.")
  }
  ablation_df[order(-sharpe_ratio_mean)]
}

