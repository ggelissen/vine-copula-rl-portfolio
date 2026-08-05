#!/usr/bin/env Rscript
# Sequential, fail-closed publication replication runner. It trains and runs
# the no-holdout gate for every preregistered seed but never invokes evaluation.

suppressPackageStartupMessages(library(yaml))
args <- commandArgs(trailingOnly = TRUE)
config_file <- if (length(args)) args[[1L]] else "config/config.yaml"
if (!file.exists(config_file)) stop("Config file not found: ", config_file)
config <- yaml::yaml.load_file(config_file)
seeds <- unique(as.integer(unlist(config$publication$seeds)))
minimum_successful <- as.integer(config$publication$minimum_successful_seeds)
if (!length(seeds) || anyNA(seeds) || minimum_successful < 1L ||
    minimum_successful > length(seeds)) {
  stop("Invalid publication seed configuration.")
}

rscript <- file.path(R.home("bin"), "Rscript")
if (.Platform$OS.type == "windows") rscript <- paste0(rscript, ".exe")
if (!file.exists(rscript)) stop("Rscript executable not found: ", rscript)

summary_rows <- vector("list", length(seeds))
for (index in seq_along(seeds)) {
  seed <- seeds[[index]]
  output_dir <- file.path("data", "rl_runs", paste0("seed_", seed))
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  environment <- c(
    paste0("TRAIN_SEED=", seed),
    paste0("TRAIN_OUTPUT_DIR=", normalizePath(output_dir, winslash = "/", mustWork = TRUE)),
    "LC_ALL=C", "LANG=C", "LANGUAGE=C", "TZ=UTC")
  cat(sprintf("\n[%d/%d] Training seed %d\n", index, length(seeds), seed))
  training_status <- system2(
    rscript, c("--vanilla", "run_with_config.r", config_file), env = environment)
  sanity_status <- if (identical(training_status, 0L)) system2(
    rscript, c("--vanilla", "rl/training_sanity_check.r", config_file),
    env = environment) else 1L
  report_file <- file.path(output_dir, "sanity_no_holdout", "sanity_report.json")
  gate_pass <- FALSE
  if (identical(sanity_status, 0L) && file.exists(report_file)) {
    report <- yaml::yaml.load_file(report_file)
    gate_pass <- isTRUE(report$overall_pass) &&
      isTRUE(report$publication_behavior_pass)
  }
  summary_rows[[index]] <- data.frame(
    seed = seed, output_dir = output_dir,
    training_status = training_status, sanity_status = sanity_status,
    no_holdout_gate_pass = gate_pass)
  write.csv(do.call(rbind, summary_rows[seq_len(index)]),
            file.path("data", "rl_runs", "seed_sweep_status.csv"),
            row.names = FALSE)
}

summary <- do.call(rbind, summary_rows)
passed <- sum(summary$no_holdout_gate_pass)
cat(sprintf("\nNo-holdout gate passed for %d/%d seeds.\n", passed, nrow(summary)))
if (passed < minimum_successful) {
  stop(sprintf("Publication gate failed: need %d successful seeds.",
               minimum_successful))
}
cat("All preregistered training replications passed. Final OOS evaluation remains a separate locked action.\n")
