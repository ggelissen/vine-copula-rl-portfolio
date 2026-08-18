#!/usr/bin/env Rscript
# =============================================================================
# evaluate_with_config.R — Launcher for evaluation
# =============================================================================

# ---- Ensure reticulate uses the correct Python ----
conda_prefix <- Sys.getenv("CONDA_PREFIX")
if (nzchar(conda_prefix)) {
  python_path <- file.path(conda_prefix, "bin", "python")
  if (file.exists(python_path)) {
    Sys.setenv(RETICULATE_PYTHON = python_path)
    cat("Using Python:", python_path, "\n")
  }
}

library(yaml)

# ---- Parse command-line argument for config file ----
args <- commandArgs(trailingOnly = TRUE)
config_file <- if (length(args) >= 1) args[1] else "config/config.yaml"
model_dir_override <- if (length(args) >= 2L) args[2L] else ""

if (!file.exists(config_file)) {
  stop(paste("Config file not found:", config_file))
}
cat("Loading configuration from:", config_file, "\n")
config <- yaml::yaml.load_file(config_file)

# Locked evaluation passes the frozen seed directory explicitly. Relying only
# on an inherited environment variable allowed the configuration fallback to
# evaluate seed 20260741 repeatedly under some launcher environments.
if (nzchar(model_dir_override)) {
  if (!dir.exists(model_dir_override)) {
    stop("Explicit evaluation model directory not found: ", model_dir_override)
  }
  Sys.setenv(EVAL_MODEL_DIR = normalizePath(
    model_dir_override, winslash = "/", mustWork = TRUE))
  cat("Explicit evaluation model directory:", Sys.getenv("EVAL_MODEL_DIR"), "\n")
}

set_default_env <- function(name, value) {
  if (!nzchar(Sys.getenv(name, unset = ""))) {
    do.call(Sys.setenv, setNames(list(as.character(value)), name))
  }
}

# Set environment variables for evaluation
set_default_env("EVAL_MODEL_DIR", config$evaluation$model_dir)
set_default_env("EVAL_OUTPUT_DIR", "data")
data_config <- config$data
set_default_env("RETURNS_DATA_FILE", if (is.null(data_config$returns_file))
  "data/portfolio_B_7assets_2013.csv" else data_config$returns_file)
set_default_env("RETURNS_DATA_KIND", if (is.null(data_config$returns_kind))
  "adjusted_levels" else data_config$returns_kind)
set_default_env("RETURNS_DATA_MANIFEST", if (is.null(data_config$returns_manifest))
  "" else data_config$returns_manifest)
Sys.setenv(EVALUATION_PERIODS = as.character(config$evaluation$periods))
Sys.setenv(EVAL_SEED = as.character(config$evaluation$eval_seed))
Sys.setenv(N_EVAL_SEEDS = as.character(config$evaluation$n_eval_seeds))
Sys.setenv(EVAL_GAMMA = as.character(config$evaluation$eval_gamma))
Sys.setenv(EVAL_LAMBDA = as.character(config$evaluation$eval_lambda))
Sys.setenv(EVAL_KAPPA = as.character(config$evaluation$eval_kappa))
Sys.setenv(TRAIN_DEVICE = "cpu")   # evaluation always on CPU
Sys.setenv(VINE_SIM_CORES = "1")   # evaluation uses single core
set_default_env("L", config$vine$L)
set_default_env("REF_COL", config$vine$ref_col)
set_default_env("N_SIM_CVAR", config$vine$n_sim_cvar)
Sys.setenv(ENV_SEQ_LEN = as.character(config$environment$seq_len))
Sys.setenv(ENV_HOLDING_DAYS = as.character(config$environment$holding_days))
Sys.setenv(ENV_GROSS_LEVERAGE = as.character(config$environment$gross_leverage))
Sys.setenv(ENV_NET_EXPOSURE = as.character(config$environment$net_exposure))
Sys.setenv(ENV_MAX_LONG_WEIGHT = as.character(config$environment$max_long_weight))
Sys.setenv(ENV_MAX_SHORT_WEIGHT = as.character(config$environment$max_short_weight))
Sys.setenv(ENV_SHORT_BORROW_RATE = as.character(config$environment$short_borrow_rate))
Sys.setenv(ENV_CASH_BORROW_RATE = as.character(config$environment$cash_borrow_rate))
Sys.setenv(ENV_UTILITY_MODE = as.character(config$environment$utility_mode))
ablation_zero_vine <- !is.null(config$ablation$zero_vine_state) &&
  isTRUE(config$ablation$zero_vine_state)
set_default_env("VINE_OBSERVATION_MODE", if (ablation_zero_vine) "zero" else "full")
set_default_env("VINE_FEATURE_MODE", Sys.getenv("VINE_OBSERVATION_MODE", "full"))
set_default_env("CVAR_OBSERVATION_MODE", Sys.getenv("VINE_OBSERVATION_MODE", "full"))
set_default_env("CVAR_REWARD_MODE", "full")
set_default_env("POLICY_INFERENCE_SERVER", "rl/policy_inference_server_v2.py")
Sys.setenv(VINE_MODEL = as.character(config$vine$model))
Sys.setenv(NN_VINE_EPOCHS = as.character(config$vine$nn_epochs))
Sys.setenv(NN_VINE_LR = as.character(config$vine$nn_learning_rate))
Sys.setenv(NN_VINE_PATIENCE = as.character(config$vine$nn_patience))
set_default_env("NN_VINE_MODEL_DIR", config$vine$nn_model_dir)
set_default_env("VINE_TRUNCATION_LEVEL", if (is.null(config$vine$truncation_level))
  0 else config$vine$truncation_level)
set_default_env("BENCHMARK_RESULTS_FILE", config$vine$benchmark_results_file)
set_default_env("TRAINING_MARGINALS_FILE", config$vine$training_marginals_file)
Sys.setenv(HIDDEN = as.character(config$agent$hidden))
Sys.setenv(NUM_LAYERS = as.character(config$agent$num_layers))
Sys.setenv(DIRECTION_LOGIT_BOUND = as.character(config$agent$direction_logit_bound))
Sys.setenv(PROJECTION_TEMPERATURE = as.character(config$agent$projection_temperature))
Sys.setenv(INITIAL_LEVERAGE_GATE = as.character(config$agent$initial_leverage_gate))
Sys.setenv(ENTROPY_COEF = as.character(config$agent$entropy_coef))
Sys.setenv(LEVERAGE_SOFT_TARGET = as.character(config$agent$leverage_soft_target))
Sys.setenv(LEVERAGE_PENALTY_COEF = as.character(config$agent$leverage_penalty_coef))
Sys.setenv(USE_AMP = tolower(as.character(config$agent$use_amp)))

# The ordinary locked evaluation is inaccessible until each training run has
# its local no-holdout sanity report. The causal and focused walk-forward
# studies instead use centralized frozen audits that bind every permitted
# checkpoint hash to its preregistered architecture and gate policy. These
# audited paths are weights-only and are enabled only by their replay drivers.
evaluation_model_dir <- Sys.getenv("EVAL_MODEL_DIR")
sanity_report_file <- file.path(evaluation_model_dir,
                                "sanity_no_holdout", "sanity_report.json")
gate_authorization <- Sys.getenv("EVAL_GATE_AUTHORIZATION", unset = "")
audited_causal_replay <- identical(
  gate_authorization, "causal_checkpoint_audit_v1")
audited_focused_replay <- identical(
  gate_authorization, "focused_checkpoint_audit_v1")
audited_synthetic_dose_replay <- identical(
  gate_authorization, "synthetic_dose_checkpoint_audit_v1")

if (nzchar(gate_authorization) &&
    !audited_causal_replay && !audited_focused_replay &&
    !audited_synthetic_dose_replay) {
  stop("Evaluation rejected an unknown gate authorization protocol.")
}

truthy <- function(value) {
  tolower(trimws(as.character(value))) %in% c("1", "true", "yes")
}

if (audited_causal_replay) {
  if (!truthy(Sys.getenv("EVAL_WEIGHTS_ONLY", unset = "false")) ||
      !identical(Sys.getenv("EVAL_CHECKPOINT_MODELS", unset = ""), "full")) {
    stop("Audited causal replay is restricted to weights-only full checkpoints.")
  }
  audit_manifest_file <- Sys.getenv("EVAL_CAUSAL_AUDIT_MANIFEST", unset = "")
  audit_table_file <- Sys.getenv("EVAL_CAUSAL_CHECKPOINT_AUDIT", unset = "")
  declared_checkpoint_sha256 <- tolower(Sys.getenv(
    "EVAL_CAUSAL_CHECKPOINT_SHA256", unset = ""))
  if (!file.exists(audit_manifest_file) || !file.exists(audit_table_file) ||
      !grepl("^[0-9a-f]{64}$", declared_checkpoint_sha256)) {
    stop("Audited causal replay lacks its frozen audit evidence.")
  }
  audit_manifest <- yaml::yaml.load_file(audit_manifest_file)
  audit_manifest_valid <- identical(
      audit_manifest$status, "causal_sweep_audit_passed") &&
    identical(as.integer(audit_manifest$job_count), 130L) &&
    isTRUE(audit_manifest$all_checkpoint_tensors_finite) &&
    isTRUE(audit_manifest$all_behavior_gate_enforcement_valid) &&
    isTRUE(audit_manifest$all_checkpoint_metadata_match) &&
    isTRUE(audit_manifest$mixed_revision_carry_forward) &&
    identical(as.integer(audit_manifest$v2_carried_count), 70L) &&
    identical(as.integer(audit_manifest$v3_carried_count), 31L) &&
    identical(as.integer(audit_manifest$v4_retry_count), 29L)
  if (!audit_manifest_valid) {
    stop("Audited causal replay rejected a nonconforming checkpoint audit manifest.")
  }
  checkpoint_prefix <- Sys.getenv("EVAL_CHECKPOINT_PREFIX", unset = "")
  checkpoint_file <- normalizePath(file.path(
    evaluation_model_dir, paste0(checkpoint_prefix, "_full.pt")),
    winslash = "/", mustWork = TRUE)
  audit_table <- read.csv(audit_table_file, stringsAsFactors = FALSE,
                          check.names = FALSE)
  required_audit_columns <- c(
    "checkpoint", "sha256", "all_tensors_finite", "behavior_gate_mode")
  if (!nrow(audit_table) ||
      !all(required_audit_columns %in% names(audit_table))) {
    stop("Audited causal replay found an incomplete checkpoint audit table.")
  }
  audit_root <- normalizePath(".", winslash = "/", mustWork = TRUE)
  audit_checkpoint_paths <- vapply(audit_table$checkpoint, function(path) {
    normalizePath(file.path(audit_root, path), winslash = "/", mustWork = TRUE)
  }, character(1))
  audit_row <- audit_table[audit_checkpoint_paths == checkpoint_file, , drop = FALSE]
  expected_gate_mode <- Sys.getenv(
    "PRETRAIN_BEHAVIOR_GATE_MODE", unset = "strict")
  if (nrow(audit_row) != 1L ||
      !identical(tolower(as.character(audit_row$sha256[[1L]])),
                 declared_checkpoint_sha256) ||
      !truthy(audit_row$all_tensors_finite[[1L]]) ||
      !identical(as.character(audit_row$behavior_gate_mode[[1L]]),
                 expected_gate_mode)) {
    stop("Checkpoint is not uniquely authorized by the frozen causal audit.")
  }
  cat("Authorized weights-only replay through frozen 130-checkpoint causal audit.\n")
} else if (audited_focused_replay) {
  if (!truthy(Sys.getenv("EVAL_WEIGHTS_ONLY", unset = "false")) ||
      !identical(Sys.getenv("EVAL_CHECKPOINT_MODELS", unset = ""), "full")) {
    stop("Audited focused replay is restricted to weights-only full checkpoints.")
  }
  audit_manifest_file <- Sys.getenv(
    "EVAL_FOCUSED_AUDIT_MANIFEST", unset = "")
  audit_table_file <- Sys.getenv(
    "EVAL_FOCUSED_CHECKPOINT_AUDIT", unset = "")
  declared_checkpoint_sha256 <- tolower(Sys.getenv(
    "EVAL_FOCUSED_CHECKPOINT_SHA256", unset = ""))
  if (!file.exists(audit_manifest_file) || !file.exists(audit_table_file) ||
      !grepl("^[0-9a-f]{64}$", declared_checkpoint_sha256)) {
    stop("Audited focused replay lacks its frozen audit evidence.")
  }
  audit_manifest <- yaml::yaml.load_file(audit_manifest_file)
  audit_manifest_valid <- identical(
      audit_manifest$status, "focused_window_sweep_audit_passed") &&
    identical(as.integer(audit_manifest$job_count), 15L) &&
    identical(as.integer(audit_manifest$experiment_count), 3L) &&
    identical(as.integer(audit_manifest$seeds_per_experiment), 5L) &&
    isTRUE(audit_manifest$all_checkpoint_tensors_finite) &&
    isTRUE(audit_manifest$all_behavior_gate_enforcement_valid) &&
    isTRUE(audit_manifest$all_checkpoint_metadata_match) &&
    identical(audit_manifest$confirmatory_claim_permitted, FALSE)
  if (!audit_manifest_valid) {
    stop("Audited focused replay rejected a nonconforming checkpoint audit manifest.")
  }
  checkpoint_prefix <- Sys.getenv("EVAL_CHECKPOINT_PREFIX", unset = "")
  checkpoint_file <- normalizePath(file.path(
    evaluation_model_dir, paste0(checkpoint_prefix, "_full.pt")),
    winslash = "/", mustWork = TRUE)
  audit_table <- read.csv(audit_table_file, stringsAsFactors = FALSE,
                          check.names = FALSE)
  required_audit_columns <- c(
    "checkpoint", "checkpoint_sha256", "all_tensors_finite",
    "behavior_gate_mode")
  if (!nrow(audit_table) ||
      !all(required_audit_columns %in% names(audit_table))) {
    stop("Audited focused replay found an incomplete checkpoint audit table.")
  }
  audit_checkpoint_paths <- vapply(audit_table$checkpoint, function(path) {
    normalizePath(path, winslash = "/", mustWork = TRUE)
  }, character(1))
  audit_row <- audit_table[audit_checkpoint_paths == checkpoint_file, , drop = FALSE]
  expected_gate_mode <- Sys.getenv(
    "PRETRAIN_BEHAVIOR_GATE_MODE", unset = "strict")
  if (nrow(audit_row) != 1L ||
      !identical(tolower(as.character(audit_row$checkpoint_sha256[[1L]])),
                 declared_checkpoint_sha256) ||
      !truthy(audit_row$all_tensors_finite[[1L]]) ||
      !identical(as.character(audit_row$behavior_gate_mode[[1L]]),
                 expected_gate_mode)) {
    stop("Checkpoint is not uniquely authorized by the focused frozen audit.")
  }
  cat("Authorized weights-only replay through frozen 15-checkpoint focused audit.\n")
} else if (audited_synthetic_dose_replay) {
  if (!truthy(Sys.getenv("EVAL_WEIGHTS_ONLY", unset = "false")) ||
      !identical(Sys.getenv("EVAL_CHECKPOINT_MODELS", unset = ""), "full")) {
    stop("Audited synthetic-dose replay is restricted to weights-only full checkpoints.")
  }
  audit_manifest_file <- Sys.getenv(
    "EVAL_DOSE_AUDIT_MANIFEST", unset = "")
  audit_table_file <- Sys.getenv(
    "EVAL_DOSE_CHECKPOINT_AUDIT", unset = "")
  declared_checkpoint_sha256 <- tolower(Sys.getenv(
    "EVAL_DOSE_CHECKPOINT_SHA256", unset = ""))
  if (!file.exists(audit_manifest_file) || !file.exists(audit_table_file) ||
      !grepl("^[0-9a-f]{64}$", declared_checkpoint_sha256)) {
    stop("Audited synthetic-dose replay lacks its frozen audit evidence.")
  }
  audit_manifest <- yaml::yaml.load_file(audit_manifest_file)
  audit_manifest_valid <- identical(
      audit_manifest$status, "synthetic_dose_sweep_audit_passed") &&
    identical(as.integer(audit_manifest$job_count), 20L) &&
    identical(as.integer(audit_manifest$experiment_count), 2L) &&
    identical(as.integer(audit_manifest$seeds_per_experiment), 10L) &&
    isTRUE(audit_manifest$all_checkpoint_tensors_finite) &&
    isTRUE(audit_manifest$all_behavior_gate_enforcement_valid) &&
    isTRUE(audit_manifest$all_checkpoint_metadata_match) &&
    identical(audit_manifest$confirmatory_claim_permitted, FALSE)
  if (!audit_manifest_valid) {
    stop("Audited synthetic-dose replay rejected a nonconforming audit manifest.")
  }
  checkpoint_prefix <- Sys.getenv("EVAL_CHECKPOINT_PREFIX", unset = "")
  checkpoint_file <- normalizePath(file.path(
    evaluation_model_dir, paste0(checkpoint_prefix, "_full.pt")),
    winslash = "/", mustWork = TRUE)
  audit_table <- read.csv(audit_table_file, stringsAsFactors = FALSE,
                          check.names = FALSE)
  required_audit_columns <- c(
    "checkpoint", "checkpoint_sha256", "all_tensors_finite",
    "behavior_gate_mode")
  if (!nrow(audit_table) ||
      !all(required_audit_columns %in% names(audit_table))) {
    stop("Audited synthetic-dose replay found an incomplete audit table.")
  }
  audit_checkpoint_paths <- vapply(audit_table$checkpoint, function(path) {
    normalizePath(path, winslash = "/", mustWork = TRUE)
  }, character(1))
  audit_row <- audit_table[audit_checkpoint_paths == checkpoint_file, , drop = FALSE]
  expected_gate_mode <- Sys.getenv(
    "PRETRAIN_BEHAVIOR_GATE_MODE", unset = "report_only")
  if (nrow(audit_row) != 1L ||
      !identical(tolower(as.character(audit_row$checkpoint_sha256[[1L]])),
                 declared_checkpoint_sha256) ||
      !truthy(audit_row$all_tensors_finite[[1L]]) ||
      !identical(as.character(audit_row$behavior_gate_mode[[1L]]),
                 expected_gate_mode)) {
    stop("Checkpoint is not uniquely authorized by the synthetic-dose audit.")
  }
  cat("Authorized weights-only replay through frozen 20-checkpoint synthetic-dose audit.\n")
} else {
  required_training_artifacts <- c(
    file.path(evaluation_model_dir, "training_episode_metrics.csv"),
    file.path(evaluation_model_dir, "training_update_metrics.csv"),
    file.path(evaluation_model_dir, "pretraining_behavior_gate.csv"),
    file.path(evaluation_model_dir, "finetune_validation_metrics.csv"),
    file.path(evaluation_model_dir, "finetune_selection.txt"),
    sanity_report_file)
  missing_training_artifacts <- required_training_artifacts[
    !file.exists(required_training_artifacts)]
  if (length(missing_training_artifacts)) {
    stop(paste0(
      "Evaluation is locked because required training gates are missing:\n - ",
      paste(missing_training_artifacts, collapse = "\n - "),
      "\nRun the corrected trainer and rl/training_sanity_check.r first."))
  }
  pretraining_gate <- read.csv(
    file.path(evaluation_model_dir, "pretraining_behavior_gate.csv"),
    stringsAsFactors = FALSE)
  if (!nrow(pretraining_gate) || !"pass" %in% names(pretraining_gate) ||
      !all(as.logical(pretraining_gate$pass))) {
    stop("Evaluation is locked because the pre-training behavioural gate did not pass.")
  }
  sanity_report <- yaml::yaml.load_file(sanity_report_file)
  if (!isTRUE(sanity_report$overall_pass) ||
      !isTRUE(sanity_report$publication_behavior_pass)) {
    stop("Evaluation is locked because the no-holdout publication behaviour gate did not pass.")
  }
}

source("rl/evaluate_rl.r")
