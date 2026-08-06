#!/usr/bin/env Rscript
# Sequential, fail-closed publication replication runner. It trains and runs
# the no-holdout gate for every preregistered seed but never invokes evaluation.

suppressPackageStartupMessages(library(yaml))
args <- commandArgs(trailingOnly = TRUE)
config_file <- if (length(args)) args[[1L]] else "config/config.yaml"
if (!file.exists(config_file)) stop("Config file not found: ", config_file)
config <- yaml::yaml.load_file(config_file)
seed_file <- Sys.getenv("SWEEP_SEEDS_FILE", unset = "")
if (nzchar(seed_file)) {
  if (!file.exists(seed_file)) stop("SWEEP_SEEDS_FILE not found: ", seed_file)
  seed_spec <- yaml::yaml.load_file(seed_file)
  seeds <- unique(as.integer(unlist(seed_spec$seeds)))
  minimum_successful <- as.integer(seed_spec$minimum_successful_seeds)
} else {
  seeds <- unique(as.integer(unlist(config$publication$seeds)))
  minimum_successful <- as.integer(config$publication$minimum_successful_seeds)
}
run_root <- Sys.getenv("SWEEP_ROOT_DIR", unset = file.path("data", "rl_runs"))
output_prefix <- Sys.getenv("SWEEP_OUTPUT_PREFIX", unset = "seed_")
worker_count <- as.integer(Sys.getenv("SWEEP_WORKER_COUNT", unset = "1"))
worker_index <- as.integer(Sys.getenv("SWEEP_WORKER_INDEX", unset = "1"))
if (is.na(worker_count) || is.na(worker_index) || worker_count < 1L ||
    worker_index < 1L || worker_index > worker_count) {
  stop("SWEEP_WORKER_INDEX must be in 1..SWEEP_WORKER_COUNT.")
}
status_default <- if (worker_count == 1L) {
  file.path(run_root, "seed_sweep_status.csv")
} else {
  file.path(run_root, sprintf("seed_sweep_status_worker_%d_of_%d.csv",
                              worker_index, worker_count))
}
status_file <- Sys.getenv("SWEEP_STATUS_FILE", unset = status_default)
vine_observation_mode <- Sys.getenv("VINE_OBSERVATION_MODE", unset = "full")
if (!vine_observation_mode %in% c("full", "zero")) {
  stop("VINE_OBSERVATION_MODE must be full or zero.")
}
if (!length(seeds) || anyNA(seeds) || minimum_successful < 1L ||
    minimum_successful > length(seeds)) {
  stop("Invalid publication seed configuration.")
}
all_seed_count <- length(seeds)
if (worker_count > 1L) {
  assigned <- ((seq_along(seeds) - 1L) %% worker_count) + 1L == worker_index
  seeds <- seeds[assigned]
  if (!length(seeds)) stop("This worker was assigned no seeds.")
  cat(sprintf("Parallel sweep worker %d/%d: %d of %d seeds assigned.\n",
              worker_index, worker_count, length(seeds), all_seed_count))
}

rscript <- file.path(R.home("bin"), "Rscript")
if (.Platform$OS.type == "windows") rscript <- paste0(rscript, ".exe")
if (!file.exists(rscript)) stop("Rscript executable not found: ", rscript)

summary_rows <- vector("list", length(seeds))
policy_metric <- function(frame, model, metric) {
  row <- frame[frame$model == model, , drop = FALSE]
  if (nrow(row) != 1L || !metric %in% names(row)) return(NA_real_)
  as.numeric(row[[metric]])
}
sensitivity_metric <- function(frame, model, perturbation, metric) {
  row <- frame[frame$model == model & frame$perturbation == perturbation,
               , drop = FALSE]
  if (nrow(row) != 1L || !metric %in% names(row)) return(NA_real_)
  as.numeric(row[[metric]])
}
for (index in seq_along(seeds)) {
  seed <- seeds[[index]]
  output_dir <- file.path(run_root, paste0(output_prefix, seed))
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  environment <- c(
    paste0("TRAIN_SEED=", seed),
    paste0("TRAIN_OUTPUT_DIR=", normalizePath(output_dir, winslash = "/", mustWork = TRUE)),
    paste0("VINE_OBSERVATION_MODE=", vine_observation_mode),
    "LC_ALL=C", "LANG=C", "LANGUAGE=C", "TZ=UTC")
  cat(sprintf("\n[%d/%d] Training seed %d\n", index, length(seeds), seed))
  training_status <- system2(
    rscript, c("--vanilla", "run_with_config.r", config_file), env = environment)
  sanity_status <- if (identical(training_status, 0L)) system2(
    rscript, c("--vanilla", "rl/training_sanity_check.r", config_file),
    env = environment) else 1L
  report_file <- file.path(output_dir, "sanity_no_holdout", "sanity_report.json")
  gate_pass <- FALSE
  policy_file <- file.path(output_dir, "sanity_no_holdout", "policy_summary.csv")
  sensitivity_file <- file.path(
    output_dir, "sanity_no_holdout", "state_sensitivity_summary.csv")
  policy <- data.frame(); sensitivity <- data.frame()
  if (identical(sanity_status, 0L) && file.exists(report_file)) {
    report <- yaml::yaml.load_file(report_file)
    gate_pass <- isTRUE(report$overall_pass) &&
      isTRUE(report$publication_behavior_pass)
    if (file.exists(policy_file)) {
      policy <- read.csv(policy_file, stringsAsFactors = FALSE)
    }
    if (file.exists(sensitivity_file)) {
      sensitivity <- read.csv(sensitivity_file, stringsAsFactors = FALSE)
    }
  }
  pretrained_reward <- policy_metric(policy, "pretrained", "mean_episode_reward")
  full_reward <- policy_metric(policy, "full", "mean_episode_reward")
  pretrained_wealth <- policy_metric(policy, "pretrained", "mean_terminal_wealth")
  full_wealth <- policy_metric(policy, "full", "mean_terminal_wealth")
  summary_rows[[index]] <- data.frame(
    seed = seed, output_dir = output_dir,
    training_status = training_status, sanity_status = sanity_status,
    no_holdout_gate_pass = gate_pass,
    pretrained_mean_reward = pretrained_reward,
    full_mean_reward = full_reward,
    finetune_reward_delta = full_reward - pretrained_reward,
    pretrained_mean_terminal_wealth = pretrained_wealth,
    full_mean_terminal_wealth = full_wealth,
    finetune_terminal_wealth_delta = full_wealth - pretrained_wealth,
    full_mean_cvar = policy_metric(policy, "full", "mean_cvar"),
    full_median_turnover = policy_metric(policy, "full", "median_turnover"),
    full_mean_leverage_gate = policy_metric(policy, "full", "mean_leverage_gate"),
    full_std_leverage_gate = policy_metric(policy, "full", "std_leverage_gate"),
    full_mean_normalized_entropy = policy_metric(
      policy, "full", "mean_normalized_direction_entropy"),
    full_mean_effective_positions = policy_metric(
      policy, "full", "mean_effective_positions"),
    full_zero_vine_median_action_l1 = sensitivity_metric(
      sensitivity, "full", "zero_vine", "median_action_l1_change"),
    full_zero_market_median_action_l1 = sensitivity_metric(
      sensitivity, "full", "zero_returns_volatility",
      "median_action_l1_change"),
    full_zero_market_median_gate_change = sensitivity_metric(
      sensitivity, "full", "zero_returns_volatility",
      "median_leverage_gate_abs_change"))
  dir.create(dirname(status_file), recursive = TRUE, showWarnings = FALSE)
  write.csv(do.call(rbind, summary_rows[seq_len(index)]), status_file,
            row.names = FALSE)
}

summary <- do.call(rbind, summary_rows)
passed <- sum(summary$no_holdout_gate_pass)
cat(sprintf("\nNo-holdout gate passed for %d/%d seeds.\n", passed, nrow(summary)))
if (worker_count > 1L) {
  cat("Worker shard completed; enforce the preregistered minimum after merging all shard status files.\n")
} else if (passed < minimum_successful) {
  stop(sprintf("Publication gate failed: need %d successful seeds.",
               minimum_successful))
} else {
  cat("All preregistered training replications passed. Final OOS evaluation remains a separate locked action.\n")
}
