# Artifact-only hyperparameter sensitivity analysis.
# Configurations are trained independently; no response surface is invented.

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})
if (!exists("annualised_path_metrics")) source("eval/ablation.r")

run_sensitivity <- function(manifest_file = "config/sensitivity_manifest.csv",
                            output_file = "data/sensitivity_results.csv", ...) {
  manifest <- read_experiment_manifest(
    manifest_file, c("lambda", "kappa", "seed", "log_file")
  )
  runs <- lapply(manifest$log_file, read_realised_run)
  reference_dates <- runs[[1L]]$date
  if (!all(vapply(runs, function(x) identical(x$date, reference_dates), logical(1)))) {
    stop("Sensitivity logs do not cover identical realised holding periods.")
  }
  per_run <- rbindlist(lapply(seq_len(nrow(manifest)), function(i) {
    as.data.table(as.list(annualised_path_metrics(runs[[i]]$net_return)))[
      , `:=`(lambda = manifest$lambda[i], kappa = manifest$kappa[i],
               seed = manifest$seed[i])]
  }), fill = TRUE)
  metric_cols <- setdiff(names(per_run), c("lambda", "kappa", "seed"))
  result <- per_run[, c(
    list(n_training_seeds = .N),
    setNames(lapply(.SD, mean, na.rm = TRUE), paste0(metric_cols, "_mean")),
    setNames(lapply(.SD, function(x) sd(x, na.rm = TRUE) / sqrt(sum(is.finite(x)))),
             paste0(metric_cols, "_se"))
  ), by = .(lambda, kappa), .SDcols = metric_cols]
  dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
  fwrite(result, output_file)
  attr(result, "per_run") <- per_run
  result
}

plot_sensitivity_heatmap <- function(sensitivity_df,
                                     save_path = "figures/sensitivity_heatmap.pdf") {
  p <- ggplot(sensitivity_df, aes(factor(kappa), factor(lambda),
                                  fill = sharpe_ratio_mean)) +
    geom_tile() + geom_text(aes(label = sprintf("%.2f", sharpe_ratio_mean))) +
    scale_fill_viridis_c() + theme_bw() +
    labs(x = "Turnover cost coefficient", y = "CVaR coefficient",
         fill = "OOS Sharpe")
  dir.create(dirname(save_path), recursive = TRUE, showWarnings = FALSE)
  ggsave(save_path, p, width = 7, height = 5)
  p
}

sensitivity_summary <- function(sensitivity_df, benchmark_sharpe = NULL) {
  best <- sensitivity_df[which.max(sharpe_ratio_mean)]
  text <- sprintf("Best completed configuration: lambda=%g, kappa=%g, mean OOS Sharpe=%.3f across %d training seeds.",
                  best$lambda, best$kappa, best$sharpe_ratio_mean,
                  best$n_training_seeds)
  if (!is.null(benchmark_sharpe)) {
    text <- paste0(text, sprintf(" Benchmark Sharpe=%.3f.", benchmark_sharpe))
  }
  text
}

