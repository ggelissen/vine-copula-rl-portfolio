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

suppressPackageStartupMessages({
  library(yaml)
})

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
Sys.setenv(EVALUATION_PERIODS = as.character(config$evaluation$periods))
Sys.setenv(EVAL_SEED = as.character(config$evaluation$eval_seed))
Sys.setenv(N_EVAL_SEEDS = as.character(config$evaluation$n_eval_seeds))
Sys.setenv(EVAL_GAMMA = as.character(config$evaluation$eval_gamma))
Sys.setenv(EVAL_LAMBDA = as.character(config$evaluation$eval_lambda))
Sys.setenv(EVAL_KAPPA = as.character(config$evaluation$eval_kappa))
Sys.setenv(TRAIN_DEVICE = "cpu")   # evaluation always on CPU
Sys.setenv(VINE_SIM_CORES = "1")   # evaluation uses single core
Sys.setenv(L = as.character(config$vine$L))
Sys.setenv(REF_COL = as.character(config$vine$ref_col))
Sys.setenv(N_SIM_CVAR = as.character(config$vine$n_sim_cvar))
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
Sys.setenv(VINE_MODEL = as.character(config$vine$model))
Sys.setenv(NN_VINE_EPOCHS = as.character(config$vine$nn_epochs))
Sys.setenv(NN_VINE_LR = as.character(config$vine$nn_learning_rate))
Sys.setenv(NN_VINE_PATIENCE = as.character(config$vine$nn_patience))
Sys.setenv(NN_VINE_MODEL_DIR = as.character(config$vine$nn_model_dir))
Sys.setenv(BENCHMARK_RESULTS_FILE = config$vine$benchmark_results_file)
Sys.setenv(TRAINING_MARGINALS_FILE = config$vine$training_marginals_file)
Sys.setenv(HIDDEN = as.character(config$agent$hidden))
Sys.setenv(NUM_LAYERS = as.character(config$agent$num_layers))
Sys.setenv(DIRECTION_LOGIT_BOUND = as.character(config$agent$direction_logit_bound))
Sys.setenv(PROJECTION_TEMPERATURE = as.character(config$agent$projection_temperature))
Sys.setenv(INITIAL_LEVERAGE_GATE = as.character(config$agent$initial_leverage_gate))
Sys.setenv(ENTROPY_COEF = as.character(config$agent$entropy_coef))
Sys.setenv(LEVERAGE_SOFT_TARGET = as.character(config$agent$leverage_soft_target))
Sys.setenv(LEVERAGE_PENALTY_COEF = as.character(config$agent$leverage_penalty_coef))
Sys.setenv(USE_AMP = tolower(as.character(config$agent$use_amp)))

# The locked holdout is inaccessible until the frozen checkpoints pass the
# no-holdout numerical and behavioural gate. JSON is valid YAML, so no extra
# parser dependency is required here.
evaluation_model_dir <- Sys.getenv("EVAL_MODEL_DIR")
sanity_report_file <- file.path(evaluation_model_dir,
                                "sanity_no_holdout", "sanity_report.json")
required_training_artifacts <- c(
  file.path(evaluation_model_dir, "training_episode_metrics.csv"),
  file.path(evaluation_model_dir, "training_update_metrics.csv"),
  file.path(evaluation_model_dir, "pretraining_behavior_gate.csv"),
  file.path(evaluation_model_dir, "finetune_validation_metrics.csv"),
  file.path(evaluation_model_dir, "finetune_selection.txt"),
  sanity_report_file)
missing_training_artifacts <- required_training_artifacts[!file.exists(required_training_artifacts)]
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

source("rl/evaluate_rl.r")
