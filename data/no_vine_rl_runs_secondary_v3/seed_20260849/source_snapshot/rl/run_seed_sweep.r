#!/usr/bin/env Rscript
# Sequential, fail-closed publication replication runner. It trains and runs
# the no-holdout gate for every preregistered seed but never invokes evaluation.

suppressPackageStartupMessages(library(yaml))
args <- commandArgs(trailingOnly = TRUE)
config_file <- if (length(args)) args[[1L]] else "config/config.yaml"
if (!file.exists(config_file)) stop("Config file not found: ", config_file)
config <- yaml::yaml.load_file(config_file)
seed_file <- Sys.getenv("SWEEP_SEEDS_FILE", unset = "")
seed_spec_vine_mode <- ""
if (nzchar(seed_file)) {
  if (!file.exists(seed_file)) stop("SWEEP_SEEDS_FILE not found: ", seed_file)
  seed_spec <- yaml::yaml.load_file(seed_file)
  seeds <- unique(as.integer(unlist(seed_spec$seeds)))
  minimum_successful <- as.integer(seed_spec$minimum_successful_seeds)
  if (!is.null(seed_spec$vine_observation_mode)) {
    seed_spec_vine_mode <- as.character(seed_spec$vine_observation_mode)
  }
} else {
  seeds <- unique(as.integer(unlist(config$publication$seeds)))
  minimum_successful <- as.integer(config$publication$minimum_successful_seeds)
}
run_root <- Sys.getenv("SWEEP_ROOT_DIR", unset = file.path("data", "rl_runs"))
reuse_completed_training <- tolower(Sys.getenv(
  "SWEEP_REUSE_COMPLETED_TRAINING", unset = "false")) %in%
  c("1", "true", "yes")
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
environment_vine_mode <- Sys.getenv("VINE_OBSERVATION_MODE", unset = "")
if (nzchar(seed_spec_vine_mode)) {
  if (nzchar(environment_vine_mode) &&
      !identical(environment_vine_mode, seed_spec_vine_mode)) {
    stop(sprintf(
      "VINE_OBSERVATION_MODE=%s conflicts with %s, which preregisters %s.",
      environment_vine_mode, seed_file, seed_spec_vine_mode))
  }
  vine_observation_mode <- seed_spec_vine_mode
} else {
  vine_observation_mode <- if (nzchar(environment_vine_mode)) {
    environment_vine_mode
  } else {
    "full"
  }
}
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
cat(sprintf("Sweep vine observation mode: %s\n", vine_observation_mode))

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
  if (reuse_completed_training) {
    required_recovery_files <- file.path(output_dir, c(
      "td3_lstm_vine_pretrained.pt", "td3_lstm_vine_full.pt",
      "run_manifest.rds", "vine_observation_mode.txt",
      "vine_observation_mode_repair.json"))
    if (any(!file.exists(required_recovery_files))) {
      stop("Recovery mode lacks required completed-training evidence for seed ",
           seed, ": ",
           paste(basename(required_recovery_files[!file.exists(
             required_recovery_files)]), collapse = ", "))
    }
    if (dir.exists(file.path(output_dir, "sanity_no_holdout"))) {
      stop("Recovery refuses to overwrite an existing sanity directory for seed ",
           seed)
    }
    cat(sprintf("Reusing cryptographically attested completed training for seed %d; training is not rerun.\n",
                seed))
    training_status <- 0L
  } else {
    training_status <- system2(
      rscript, c("--vanilla", "run_with_config.r", config_file), env = environment)
  }
  mode_file <- file.path(output_dir, "vine_observation_mode.txt")
  recorded_mode <- if (file.exists(mode_file)) {
    trimws(readLines(mode_file, warn = FALSE, n = 1L))
  } else {
    ""
  }
  mode_match <- identical(recorded_mode, vine_observation_mode)
  if (identical(training_status, 0L) && !mode_match) {
    warning(sprintf(
      "Seed %d wrote vine_observation_mode='%s'; expected '%s'. Marking training invalid.",
      seed, recorded_mode, vine_observation_mode))
    training_status <- 2L
  }
  sanity_status <- if (identical(training_status, 0L)) system2(
    rscript, c("--vanilla", "rl/training_sanity_check.r", config_file),
    env = environment) else 1L
  report_file <- file.path(output_dir, "sanity_no_holdout", "sanity_report.json")
  gate_pass <- FALSE
  report_signal_mask <- ""
  policy_file <- file.path(output_dir, "sanity_no_holdout", "policy_summary.csv")
  sensitivity_file <- file.path(
    output_dir, "sanity_no_holdout", "state_sensitivity_summary.csv")
  policy <- data.frame(); sensitivity <- data.frame()
  if (identical(sanity_status, 0L) && file.exists(report_file)) {
    report <- yaml::yaml.load_file(report_file)
    report_mode_match <- identical(
      as.character(report$vine_observation_mode), vine_observation_mode)
    report_signal_mask <- if (is.null(report$no_vine_signal_mask)) {
      ""
    } else {
      as.character(report$no_vine_signal_mask)
    }
    signal_mask_match <- !identical(vine_observation_mode, "zero") ||
      identical(report_signal_mask, "explicit_vine_and_scenario_cvar_v1")
    if (!report_mode_match) {
      warning(sprintf(
        "Seed %d sanity report mode '%s' does not match expected '%s'.",
        seed, as.character(report$vine_observation_mode), vine_observation_mode))
      sanity_status <- 2L
    }
    if (!signal_mask_match) {
      warning(sprintf(
        "Seed %d no-vine signal mask '%s' is invalid.",
        seed, report_signal_mask))
      sanity_status <- 2L
    }
    gate_pass <- report_mode_match && signal_mask_match &&
      isTRUE(report$overall_pass) &&
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
  status_row <- data.frame(
    seed = seed, output_dir = output_dir,
    vine_observation_mode = vine_observation_mode,
    no_vine_signal_mask = report_signal_mask,
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
  required_status_columns <- c(
    "seed", "training_status", "sanity_status", "no_holdout_gate_pass",
    "vine_observation_mode", "no_vine_signal_mask",
    "full_zero_vine_median_action_l1")
  missing_status_columns <- setdiff(required_status_columns, names(status_row))
  if (length(missing_status_columns)) {
    stop("Internal sweep status schema error; missing: ",
         paste(missing_status_columns, collapse = ", "))
  }
  summary_rows[[index]] <- status_row
  dir.create(dirname(status_file), recursive = TRUE, showWarnings = FALSE)
  write.csv(do.call(rbind, summary_rows[seq_len(index)]), status_file,
            row.names = FALSE)
}

summary <- do.call(rbind, summary_rows)
passed <- sum(summary$no_holdout_gate_pass)
cat(sprintf("\nNo-holdout gate passed for %d/%d seeds.\n", passed, nrow(summary)))
if (worker_count > 1L) {
  if (passed < nrow(summary)) {
    stop(sprintf(
      "Worker shard failed closed: %d/%d assigned seeds passed training, sanity, and behavior gates.",
      passed, nrow(summary)))
  }
  cat("Worker shard completed with every assigned seed passing; enforce the preregistered minimum after merging all shard status files.\n")
} else if (passed < minimum_successful) {
  stop(sprintf("Publication gate failed: need %d successful seeds.",
               minimum_successful))
} else {
  cat("All preregistered training replications passed. Final OOS evaluation remains a separate locked action.\n")
}
