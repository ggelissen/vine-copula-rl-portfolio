#!/usr/bin/env Rscript
# =============================================================================
# run_with_config.R — Launcher that reads YAML and sets environment variables,
# then sources the training script.
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
Sys.setenv(CONFIG_FILE = normalizePath(config_file, winslash = "/", mustWork = TRUE))

# Scheduler jobs may set a unique seed/output directory.  Preserve explicit
# process-level overrides while still supplying every unspecified setting from
# the YAML file.
set_default_env <- function(name, value) {
  if (!nzchar(Sys.getenv(name, unset = ""))) do.call(Sys.setenv, setNames(list(as.character(value)), name))
}

# ---- Set ALL environment variables from config ----
# General
set_default_env("TRAIN_SEED", config$general$seed)
set_default_env("TRAIN_OUTPUT_DIR", config$general$output_dir)
set_default_env("TRAIN_DEVICE", config$general$device)
set_default_env("TRAIN_SMOKE_TEST", tolower(as.character(config$general$smoke_test)))
set_default_env("TRAIN_VERBOSE", tolower(as.character(config$general$verbose)))
set_default_env("EVALUATION_PERIODS", config$evaluation$periods)

# Vine & Data
Sys.setenv(N_SIM_CVAR = as.character(config$vine$n_sim_cvar))
Sys.setenv(VINE_SIM_CORES = as.character(config$vine$sim_cores))
Sys.setenv(L = as.character(config$vine$L))
Sys.setenv(REF_COL = as.character(config$vine$ref_col))
Sys.setenv(PRETRAIN_RETURNS_FILE = config$vine$pretrain_returns_file)
Sys.setenv(FINETUNE_RETURNS_FILE = config$vine$finetune_returns_file)
Sys.setenv(TRAINING_MARGINALS_FILE = config$vine$training_marginals_file)
Sys.setenv(BENCHMARK_RESULTS_FILE = config$vine$benchmark_results_file)

# Environment
Sys.setenv(ENV_GAMMA = as.character(config$environment$gamma))
Sys.setenv(ENV_LAMBDA = as.character(config$environment$lambda))
Sys.setenv(ENV_KAPPA = as.character(config$environment$kappa))
Sys.setenv(ENV_T = as.character(config$environment$T))
Sys.setenv(ENV_W0 = as.character(config$environment$w0))
Sys.setenv(ENV_SEQ_LEN = as.character(config$environment$seq_len))
Sys.setenv(ENV_HOLDING_DAYS = as.character(config$environment$holding_days))
Sys.setenv(ENV_GROSS_LEVERAGE = as.character(config$environment$gross_leverage))
Sys.setenv(ENV_NET_EXPOSURE = as.character(config$environment$net_exposure))
Sys.setenv(ENV_SHORT_BORROW_RATE = as.character(config$environment$short_borrow_rate))
Sys.setenv(ENV_CASH_BORROW_RATE = as.character(config$environment$cash_borrow_rate))
Sys.setenv(ENV_UTILITY_MODE = as.character(config$environment$utility_mode))

# NN-driven vine configuration
Sys.setenv(VINE_MODEL = as.character(config$vine$model))
Sys.setenv(NN_VINE_EPOCHS = as.character(config$vine$nn_epochs))
Sys.setenv(NN_VINE_LR = as.character(config$vine$nn_learning_rate))
Sys.setenv(NN_VINE_PATIENCE = as.character(config$vine$nn_patience))
Sys.setenv(NN_VINE_MODEL_DIR = as.character(config$vine$nn_model_dir))

# Pre-training
Sys.setenv(PRETRAIN_EPISODES = as.character(config$pretraining$episodes))
Sys.setenv(PRETRAIN_BATCH_SIZE = as.character(config$pretraining$batch_size))
Sys.setenv(PRETRAIN_NOISE_SCALE = as.character(config$pretraining$noise_scale))
Sys.setenv(PRETRAIN_NOISE_DECAY = as.character(config$pretraining$noise_decay))
Sys.setenv(PRETRAIN_UPDATES_PER_STEP = as.character(config$pretraining$updates_per_step))

# Fine-tuning
Sys.setenv(FINETUNE_EPISODES = as.character(config$finetuning$episodes))
Sys.setenv(FINETUNE_BATCH_SIZE = as.character(config$finetuning$batch_size))
Sys.setenv(FINETUNE_NOISE_SCALE = as.character(config$finetuning$noise_scale))
Sys.setenv(FINETUNE_NOISE_DECAY = as.character(config$finetuning$noise_decay))
Sys.setenv(FINETUNE_UPDATES_PER_STEP = as.character(config$finetuning$updates_per_step))
if (nchar(config$finetuning$load_model_path) > 0) {
  Sys.setenv(LOAD_MODEL_PATH = config$finetuning$load_model_path)
}

# Agent
Sys.setenv(LR_ACTOR = as.character(config$agent$lr_actor))
Sys.setenv(LR_CRITIC = as.character(config$agent$lr_critic))
Sys.setenv(DISCOUNT = as.character(config$agent$discount))
Sys.setenv(TAU = as.character(config$agent$tau))
Sys.setenv(HIDDEN = as.character(config$agent$hidden))
Sys.setenv(NUM_LAYERS = as.character(config$agent$num_layers))
Sys.setenv(REPLAY_CAPACITY = as.character(config$agent$replay_capacity))
Sys.setenv(ENTROPY_COEF = as.character(config$agent$entropy_coef))
Sys.setenv(POLICY_DELAY = as.character(config$agent$policy_delay))
Sys.setenv(TARGET_POLICY_NOISE = as.character(config$agent$target_policy_noise))
Sys.setenv(TARGET_NOISE_CLIP = as.character(config$agent$target_noise_clip))
Sys.setenv(RANDOM_EXPLORATION_STEPS = as.character(config$agent$random_exploration_steps))
Sys.setenv(DETERMINISTIC_ALGORITHMS = tolower(as.character(config$agent$deterministic_algorithms)))
Sys.setenv(GRAD_CLIP_NORM = as.character(config$agent$grad_clip_norm))

# ---- Source the actual training script ----
source("rl/train_rl.r")
