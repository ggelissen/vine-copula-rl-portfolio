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

if (!file.exists(config_file)) {
  stop(paste("Config file not found:", config_file))
}
cat("Loading configuration from:", config_file, "\n")
config <- yaml::yaml.load_file(config_file)

# Set environment variables for evaluation
Sys.setenv(EVAL_MODEL_DIR = config$evaluation$model_dir)
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
Sys.setenv(ENV_SHORT_BORROW_RATE = as.character(config$environment$short_borrow_rate))
Sys.setenv(ENV_CASH_BORROW_RATE = as.character(config$environment$cash_borrow_rate))
Sys.setenv(ENV_UTILITY_MODE = as.character(config$environment$utility_mode))
Sys.setenv(VINE_MODEL = as.character(config$vine$model))
Sys.setenv(NN_VINE_EPOCHS = as.character(config$vine$nn_epochs))
Sys.setenv(NN_VINE_LR = as.character(config$vine$nn_learning_rate))
Sys.setenv(NN_VINE_PATIENCE = as.character(config$vine$nn_patience))
Sys.setenv(NN_VINE_MODEL_DIR = as.character(config$vine$nn_model_dir))
Sys.setenv(BENCHMARK_RESULTS_FILE = config$vine$benchmark_results_file)
Sys.setenv(TRAINING_MARGINALS_FILE = config$vine$training_marginals_file)
Sys.setenv(HIDDEN = as.character(config$agent$hidden))
Sys.setenv(NUM_LAYERS = as.character(config$agent$num_layers))

source("rl/evaluate_rl.r")
