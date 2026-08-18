# ============================================================================
# train_rl.r — Training algorithm for DRL agent
# ============================================================================
# Frozen causal gate wiring: report_only_v4_20260812. Economic diagnostics
# are reported without selecting seeds; non-finite and hard-constraint
# failures remain fatal in enforce_pretraining_behavior_gate().

suppressPackageStartupMessages({
  library(reticulate)
  library(parallel)
  library(rvinecopulib)
  library(zoo)
})

# ---- Read all parameters from environment ----
train_seed <- as.integer(Sys.getenv("TRAIN_SEED"))
output_dir <- Sys.getenv("TRAIN_OUTPUT_DIR")
device <- Sys.getenv("TRAIN_DEVICE")
smoke_test <- tolower(Sys.getenv("TRAIN_SMOKE_TEST")) %in% c("1", "true", "yes")
verbose <- tolower(Sys.getenv("TRAIN_VERBOSE")) %in% c("1", "true", "yes")
n_sim_cvar <- as.integer(Sys.getenv("N_SIM_CVAR"))
vine_sim_cores <- as.integer(Sys.getenv("VINE_SIM_CORES"))
available_physical_cores <- parallel::detectCores(logical = FALSE)
if (is.finite(available_physical_cores)) vine_sim_cores <- min(vine_sim_cores, available_physical_cores)
L <- as.integer(Sys.getenv("L"))
ref_col <- as.integer(Sys.getenv("REF_COL"))
synthetic_file <- Sys.getenv("SYNTHETIC_RETURNS_FILE", "data/synthetic_returns.RData")
training_marginals_file <- Sys.getenv("TRAINING_MARGINALS_FILE", "data/training_marginal_results.RData")
benchmark_file <- Sys.getenv("BENCHMARK_RESULTS_FILE")

env_gamma <- as.numeric(Sys.getenv("ENV_GAMMA"))
env_lambda <- as.numeric(Sys.getenv("ENV_LAMBDA"))
env_kappa <- as.numeric(Sys.getenv("ENV_KAPPA"))
env_T <- as.integer(Sys.getenv("ENV_T"))
evaluation_periods <- as.integer(Sys.getenv("EVALUATION_PERIODS", "24"))
env_w0 <- as.numeric(Sys.getenv("ENV_W0"))
env_seq_len <- as.integer(Sys.getenv("ENV_SEQ_LEN"))
env_holding_days <- as.integer(Sys.getenv("ENV_HOLDING_DAYS"))
env_gross_leverage <- as.numeric(Sys.getenv("ENV_GROSS_LEVERAGE"))
env_net_exposure <- as.numeric(Sys.getenv("ENV_NET_EXPOSURE"))
env_max_long_weight <- as.numeric(Sys.getenv("ENV_MAX_LONG_WEIGHT", "0.60"))
env_max_short_weight <- as.numeric(Sys.getenv("ENV_MAX_SHORT_WEIGHT", "0.20"))
env_short_borrow_rate <- as.numeric(Sys.getenv("ENV_SHORT_BORROW_RATE", "0.03"))
env_cash_borrow_rate <- as.numeric(Sys.getenv("ENV_CASH_BORROW_RATE", "0.02"))
env_utility_mode <- Sys.getenv("ENV_UTILITY_MODE", "terminal_wealth_crra")
vine_observation_mode <- Sys.getenv("VINE_OBSERVATION_MODE", "full")
if (!vine_observation_mode %in% c("full", "zero")) {
  stop("VINE_OBSERVATION_MODE must be 'full' or 'zero'.")
}
vine_feature_mode <- Sys.getenv("VINE_FEATURE_MODE", vine_observation_mode)
cvar_observation_mode <- Sys.getenv("CVAR_OBSERVATION_MODE", vine_observation_mode)
cvar_reward_mode <- Sys.getenv("CVAR_REWARD_MODE", "full")
if (any(!c(vine_feature_mode, cvar_observation_mode, cvar_reward_mode) %in%
        c("full", "zero"))) {
  stop("VINE_FEATURE_MODE, CVAR_OBSERVATION_MODE, and CVAR_REWARD_MODE must be 'full' or 'zero'.")
}
if (identical(vine_observation_mode, "zero") &&
    (!identical(vine_feature_mode, "zero") ||
     !identical(cvar_observation_mode, "zero"))) {
  stop("Legacy VINE_OBSERVATION_MODE=zero must mask both policy-visible vine signals.")
}
pretrain_data_mode <- Sys.getenv("PRETRAIN_DATA_MODE", "vine_synthetic")
if (!pretrain_data_mode %in% c("vine_synthetic", "historical_prefix_repeated",
                               "moving_block_bootstrap")) {
  stop("Unsupported PRETRAIN_DATA_MODE.")
}
pretrain_behavior_gate_mode <- tolower(Sys.getenv(
  "PRETRAIN_BEHAVIOR_GATE_MODE", "strict"))
if (!pretrain_behavior_gate_mode %in% c("strict", "report_only")) {
  stop("PRETRAIN_BEHAVIOR_GATE_MODE must be 'strict' or 'report_only'.")
}
rl_algorithm <- tolower(Sys.getenv("RL_ALGORITHM", "td3"))
policy_encoder <- tolower(Sys.getenv("POLICY_ENCODER", "lstm"))
checkpoint_prefix <- Sys.getenv("CHECKPOINT_PREFIX", "td3_lstm_vine")
run_finetune <- tolower(Sys.getenv("RUN_FINETUNE", "true")) %in%
  c("1", "true", "yes")
if (!rl_algorithm %in% c("td3", "ddpg", "sac", "ppo", "a2c")) stop("Unsupported RL_ALGORITHM.")
if (!policy_encoder %in% c("lstm", "mlp")) stop("POLICY_ENCODER must be lstm or mlp.")
if (!grepl("^[A-Za-z0-9_]+$", checkpoint_prefix)) stop("Unsafe CHECKPOINT_PREFIX.")
vine_model <- Sys.getenv("VINE_MODEL", "nn_dynamic_t_vine")
nn_vine_epochs <- as.integer(Sys.getenv("NN_VINE_EPOCHS", "200"))
nn_vine_lr <- as.numeric(Sys.getenv("NN_VINE_LR", "0.001"))
nn_vine_patience <- as.integer(Sys.getenv("NN_VINE_PATIENCE", "20"))

pretrain_episodes <- as.integer(Sys.getenv("PRETRAIN_EPISODES"))
pretrain_batch_size <- as.integer(Sys.getenv("PRETRAIN_BATCH_SIZE"))
pretrain_noise_scale <- as.numeric(Sys.getenv("PRETRAIN_NOISE_SCALE"))
pretrain_noise_decay <- as.numeric(Sys.getenv("PRETRAIN_NOISE_DECAY"))
pretrain_updates <- as.integer(Sys.getenv("PRETRAIN_UPDATES_PER_STEP"))
pretrain_random_exploration_steps <- as.integer(Sys.getenv("PRETRAIN_RANDOM_EXPLORATION_STEPS", "1000"))
pretrain_behavior_gate_window <- as.integer(Sys.getenv("PRETRAIN_BEHAVIOR_GATE_WINDOW", "100"))
pretrain_max_mean_leverage_gate <- as.numeric(Sys.getenv("PRETRAIN_MAX_MEAN_LEVERAGE_GATE", "0.95"))
pretrain_max_mean_gross_cap_fraction <- as.numeric(Sys.getenv("PRETRAIN_MAX_MEAN_GROSS_CAP_FRACTION", "0.75"))
pretrain_warn_position_cap_fraction <- as.numeric(Sys.getenv("PRETRAIN_WARN_POSITION_CAP_FRACTION", "0.75"))
pretrain_min_mean_normalized_entropy <- as.numeric(Sys.getenv("PRETRAIN_MIN_MEAN_NORMALIZED_ENTROPY", "0.70"))
pretrain_min_q05_normalized_entropy <- as.numeric(Sys.getenv("PRETRAIN_MIN_Q05_NORMALIZED_ENTROPY", "0.50"))
pretrain_min_mean_effective_positions <- as.numeric(Sys.getenv("PRETRAIN_MIN_MEAN_EFFECTIVE_POSITIONS", "2.50"))
pretrain_max_position_limit_violation <- as.numeric(Sys.getenv("PRETRAIN_MAX_POSITION_LIMIT_VIOLATION", "1e-6"))
pretrain_max_gate_gross_mae <- as.numeric(Sys.getenv("PRETRAIN_MAX_GATE_GROSS_MAE", "1e-5"))
pretrain_max_mean_turnover <- as.numeric(Sys.getenv("PRETRAIN_MAX_MEAN_TURNOVER", "1.0"))

finetune_episodes <- as.integer(Sys.getenv("FINETUNE_EPISODES"))
finetune_batch_size <- as.integer(Sys.getenv("FINETUNE_BATCH_SIZE"))
finetune_noise_scale <- as.numeric(Sys.getenv("FINETUNE_NOISE_SCALE"))
finetune_noise_decay <- as.numeric(Sys.getenv("FINETUNE_NOISE_DECAY"))
finetune_updates <- as.integer(Sys.getenv("FINETUNE_UPDATES_PER_STEP"))
finetune_random_exploration_steps <- as.integer(Sys.getenv("FINETUNE_RANDOM_EXPLORATION_STEPS", "0"))
finetune_lr_actor <- as.numeric(Sys.getenv("FINETUNE_LR_ACTOR", Sys.getenv("LR_ACTOR")))
finetune_lr_critic <- as.numeric(Sys.getenv("FINETUNE_LR_CRITIC", Sys.getenv("LR_CRITIC")))
finetune_max_selection_passes <- as.integer(Sys.getenv("FINETUNE_MAX_SELECTION_PASSES", "8"))
finetune_validation_patience <- as.integer(Sys.getenv("FINETUNE_VALIDATION_PATIENCE", "2"))
finetune_validation_min_delta <- as.numeric(Sys.getenv("FINETUNE_VALIDATION_MIN_DELTA", "0.005"))
load_model_path <- Sys.getenv("LOAD_MODEL_PATH", "")

lr_actor <- as.numeric(Sys.getenv("LR_ACTOR"))
lr_critic <- as.numeric(Sys.getenv("LR_CRITIC"))
discount <- as.numeric(Sys.getenv("DISCOUNT"))
tau <- as.numeric(Sys.getenv("TAU"))
hidden <- as.integer(Sys.getenv("HIDDEN"))
num_layers <- as.integer(Sys.getenv("NUM_LAYERS"))
replay_capacity <- as.integer(Sys.getenv("REPLAY_CAPACITY"))
entropy_coef <- as.numeric(Sys.getenv("ENTROPY_COEF"))
direction_logit_bound <- as.numeric(Sys.getenv("DIRECTION_LOGIT_BOUND", "1.0"))
projection_temperature <- as.numeric(Sys.getenv("PROJECTION_TEMPERATURE", "1.5"))
initial_leverage_gate <- as.numeric(Sys.getenv("INITIAL_LEVERAGE_GATE", "0.10"))
leverage_soft_target <- as.numeric(Sys.getenv("LEVERAGE_SOFT_TARGET", "0.80"))
leverage_penalty_coef <- as.numeric(Sys.getenv("LEVERAGE_PENALTY_COEF", "0.25"))
grad_clip_norm <- as.numeric(Sys.getenv("GRAD_CLIP_NORM"))
diagnostic_interval <- as.integer(Sys.getenv("DIAGNOSTIC_INTERVAL", "100"))

# ---- Ensure required variables are set ----
if (any(is.na(c(train_seed, output_dir, device, n_sim_cvar, vine_sim_cores, L, ref_col,
                synthetic_file, env_gamma, env_lambda, env_kappa,
                env_T, env_w0, env_seq_len, env_holding_days, env_gross_leverage,
                env_net_exposure, env_max_long_weight, env_max_short_weight,
                pretrain_episodes, pretrain_batch_size,
                pretrain_noise_scale, pretrain_noise_decay, pretrain_updates,
                pretrain_random_exploration_steps,
                pretrain_behavior_gate_window,
                pretrain_max_mean_leverage_gate,
                pretrain_max_mean_gross_cap_fraction,
                pretrain_warn_position_cap_fraction,
                pretrain_min_mean_normalized_entropy,
                pretrain_min_q05_normalized_entropy,
                pretrain_min_mean_effective_positions,
                pretrain_max_position_limit_violation,
                pretrain_max_gate_gross_mae,
                pretrain_max_mean_turnover,
                finetune_episodes, finetune_batch_size, finetune_noise_scale,
                finetune_noise_decay, finetune_updates,
                finetune_random_exploration_steps, finetune_lr_actor,
                finetune_lr_critic, finetune_max_selection_passes,
                finetune_validation_patience, finetune_validation_min_delta,
                lr_actor, lr_critic,
                discount, tau, hidden, num_layers, replay_capacity,
                entropy_coef, direction_logit_bound, projection_temperature,
                initial_leverage_gate,
                leverage_soft_target, leverage_penalty_coef,
                grad_clip_norm, diagnostic_interval)))) {
  stop("One or more required environment variables are not set. Check your launcher.")
}
if (!identical(vine_model, "nn_dynamic_t_vine")) stop("RL supports only VINE_MODEL=nn_dynamic_t_vine; rolling-window vines are intentionally disabled.")
if (!identical(env_utility_mode, "terminal_wealth_crra")) stop("Set ENV_UTILITY_MODE=terminal_wealth_crra for the multi-period CRRA objective.")
if (!is.finite(env_net_exposure) || env_net_exposure <= 0 ||
    env_gross_leverage < env_net_exposure) {
  stop("Schema-5 rank-partition actions require positive net exposure and gross >= net.")
}
if (abs(discount - 1) > 1e-12) stop("DISCOUNT must be 1.0 when using telescoping terminal-wealth CRRA utility.")
if (evaluation_periods != 24L) stop("The final historical holdout must contain exactly 24 monthly holding periods.")
if (any(c(pretrain_random_exploration_steps, finetune_random_exploration_steps) < 0L)) stop("Random exploration steps cannot be negative.")
if (pretrain_behavior_gate_window < 10L || pretrain_behavior_gate_window > pretrain_episodes) {
  stop("PRETRAIN_BEHAVIOR_GATE_WINDOW must be between 10 and PRETRAIN_EPISODES.")
}
if (any(c(pretrain_max_mean_leverage_gate,
          pretrain_max_mean_gross_cap_fraction,
          pretrain_warn_position_cap_fraction,
          pretrain_min_mean_normalized_entropy,
          pretrain_min_q05_normalized_entropy) <= 0) ||
    any(c(pretrain_max_mean_leverage_gate,
          pretrain_max_mean_gross_cap_fraction,
          pretrain_warn_position_cap_fraction,
          pretrain_min_mean_normalized_entropy,
          pretrain_min_q05_normalized_entropy) > 1) ||
    pretrain_min_mean_effective_positions <= 1 ||
    pretrain_max_position_limit_violation <= 0 ||
    pretrain_max_gate_gross_mae <= 0 || pretrain_max_mean_turnover <= 0) {
  stop("Invalid pre-training behavioural-gate thresholds.")
}
if (!is.finite(direction_logit_bound) || direction_logit_bound <= 0 ||
    !is.finite(projection_temperature) || projection_temperature <= 0 ||
    !is.finite(initial_leverage_gate) || initial_leverage_gate <= 0 ||
    initial_leverage_gate >= 1 || entropy_coef < 0 ||
    !is.finite(leverage_soft_target) || leverage_soft_target <= 0 ||
    leverage_soft_target >= pretrain_max_mean_leverage_gate ||
    !is.finite(leverage_penalty_coef) || leverage_penalty_coef < 0) {
  stop("Invalid bounded-action or allocation-regularisation settings.")
}
if (finetune_max_selection_passes < 1L || finetune_validation_patience < 1L ||
    finetune_validation_min_delta < 0) stop("Invalid fine-tuning validation settings.")
if (finetune_max_selection_passes != 1L) {
  stop("Publication protocol fixes historical fine-tuning to one pass; do not select pass count on one 24-month path.")
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
writeLines(vine_observation_mode,
           file.path(output_dir, "vine_observation_mode.txt"), useBytes = TRUE)
writeLines(vine_feature_mode,
           file.path(output_dir, "vine_feature_mode.txt"), useBytes = TRUE)
writeLines(cvar_observation_mode,
           file.path(output_dir, "cvar_observation_mode.txt"), useBytes = TRUE)
writeLines(cvar_reward_mode,
           file.path(output_dir, "cvar_reward_mode.txt"), useBytes = TRUE)
writeLines(pretrain_data_mode,
           file.path(output_dir, "pretrain_data_mode.txt"), useBytes = TRUE)
writeLines(pretrain_behavior_gate_mode,
           file.path(output_dir, "pretrain_behavior_gate_mode.txt"),
           useBytes = TRUE)
writeLines(rl_algorithm, file.path(output_dir, "rl_algorithm.txt"), useBytes = TRUE)
writeLines(policy_encoder, file.path(output_dir, "policy_encoder.txt"), useBytes = TRUE)
set.seed(train_seed)
source("helper/reproducibility.r")
write_run_manifest(output_dir, train_seed,
  config_file = Sys.getenv("CONFIG_FILE", "config/config.yaml"),
  data_files = c(Sys.getenv("RETURNS_DATA_FILE",
                            "data/portfolio_B_7assets_2013.csv"),
                 Sys.getenv("RETURNS_DATA_MANIFEST", ""), synthetic_file,
                 training_marginals_file))
cat(sprintf("Run mode: %s | seed: %d | CVaR simulations/step: %d | vine simulation cores: %d\n",
            if (smoke_test) "SMOKE TEST" else "full", train_seed, n_sim_cvar, vine_sim_cores))

if (device == "cpu") {
  Sys.setenv(CUDA_VISIBLE_DEVICES = "")
}

print_sep <- function() {
  cat(paste0("\n", paste(rep("=", 60), collapse = ""), "\n"))
}

source("rl/rl_environment.r")
source("helper/time_split.r")
if ("torch" %in% loadedNamespaces()) {
  stop(paste0(
    "R torch/Lantern was loaded in the training process. This conflicts with ",
    "Python PyTorch under reticulate. Training must use only the precomputed ",
    "vine states; do not source benchmark_models/dynamic_vine_NN.r here."))
}
if (!file.exists(training_marginals_file)) stop(sprintf("Training-only marginal file not found: %s\nRun rl/synthetic_returns.r first.", training_marginals_file))
load(training_marginals_file)
returns <- load_returns()
validate_return_model_contract(
  returns, ref_col, as.integer(Sys.getenv("VINE_TRUNCATION_LEVEL", "0")))
if (!identical(as.character(asset_names), colnames(returns))) {
  stop("Training marginal asset names/order do not match the active return-data contract.")
}
long_budget <- 0.5 * (env_gross_leverage + env_net_exposure)
short_budget <- 0.5 * (env_gross_leverage - env_net_exposure)
if (env_max_long_weight <= 0 ||
    env_max_long_weight * length(asset_names) < long_budget - 1e-10 ||
    env_max_short_weight < 0 ||
    (short_budget > 0 &&
       env_max_short_weight * length(asset_names) < short_budget - 1e-10)) {
  stop("Position limits cannot support the configured long and short budgets.")
}

# Reconstruct and validate the locked calendar; the NN vine itself was fitted
# once by synthetic_returns.r and is not redundantly re-estimated here.
period_split <- split_monthly_periods(
  build_monthly_periods(returns, min_history = L), evaluation_periods
)
validate_period_split(period_split, evaluation_periods)
validate_return_evaluation_contract(returns, period_split, evaluation_periods)
train_periods <- period_split$train
train_end <- tail(train_periods$decision_idx, 1L)

# Synthetic returns are generated and audited by rl/synthetic_returns.r.
if (!file.exists(synthetic_file)) stop(sprintf("Synthetic return bundle not found: %s\nRun Rscript --vanilla rl/synthetic_returns.r first.", synthetic_file))
cat(sprintf("Loading synthetic return bundle: %s\n", synthetic_file))
bundle <- new.env(parent = emptyenv()); load(synthetic_file, envir = bundle)
if (!exists("pretrain_returns", envir = bundle) || !exists("finetune_returns", envir = bundle) || !exists("pretrain_vine", envir = bundle)) stop("Synthetic bundle must contain pretrain_returns, finetune_returns, and pretrain_vine.")
pretrain_returns <- get("pretrain_returns", envir = bundle); finetune_returns <- get("finetune_returns", envir = bundle); pretrain_vine <- get("pretrain_vine", envir = bundle)
valid_episode_bundle <- function(x) is.list(x) && length(x) && all(vapply(x, function(ep) {
  is.list(ep) && is.list(ep$returns) && length(ep$returns) >= env_T &&
    is.list(ep$burnin_returns) && length(ep$burnin_returns) == env_seq_len
}, logical(1)))
if (!valid_episode_bundle(pretrain_returns) || !valid_episode_bundle(finetune_returns)) {
  stop("Synthetic bundle uses the obsolete flat format. Regenerate it with Rscript --vanilla rl/synthetic_returns.r.")
}
if (identical(attr(returns, "source_kind"), "daily_log_returns")) {
  valid_fractions <- function(episodes) all(vapply(episodes, function(ep) {
    fractions <- as.numeric(ep$holding_year_fractions)
    length(fractions) == env_T && all(is.finite(fractions)) &&
      all(fractions > 0) && all(fractions <= 1)
  }, logical(1)))
  if (!valid_fractions(pretrain_returns) || !valid_fractions(finetune_returns)) {
    stop("External-panel episodes lack valid holding-period financing fractions.")
  }
}
metadata <- if (exists("metadata", envir = bundle)) get("metadata", envir = bundle) else NULL
protocol_failures <- character()
require_protocol <- function(ok, message) {
  if (!isTRUE(ok)) protocol_failures <<- c(protocol_failures, message)
}
require_protocol(!is.null(metadata), "metadata is missing")
if (!is.null(metadata)) {
  expected_source <- switch(
    pretrain_data_mode,
    vine_synthetic = "synthetic_vine",
    historical_prefix_repeated = "historical_prefix_repeated_matched_updates",
    moving_block_bootstrap = "historical_moving_block_bootstrap")
  if (identical(pretrain_data_mode, "vine_synthetic")) {
    require_protocol(isTRUE(metadata$diagnostics_passed), "diagnostics_passed is not TRUE")
  } else {
    require_protocol(isTRUE(metadata$ablation_bundle), "ablation_bundle is not TRUE")
    require_protocol(identical(metadata$parent_pretrain_data_mode, "vine_synthetic"),
                     sprintf("parent_pretrain_data_mode=%s",
                             metadata$parent_pretrain_data_mode))
  }
  require_protocol(identical(metadata$pretrain_realised_source, expected_source),
                   sprintf("pretrain_realised_source=%s (expected %s)",
                           metadata$pretrain_realised_source, expected_source))
  require_protocol(identical(metadata$finetune_realised_source, "historical"),
                   sprintf("finetune_realised_source=%s", metadata$finetune_realised_source))
  require_protocol(identical(metadata$pretrain_vine_frequency, "monthly_holding_period_ranks"),
                   sprintf("pretrain_vine_frequency=%s", metadata$pretrain_vine_frequency))
  require_protocol(identical(metadata$cross_sectional_fit_input,
                             "monthly_one_step_serial_conditional_pit"),
                   sprintf("cross_sectional_fit_input=%s", metadata$cross_sectional_fit_input))
  require_protocol(identical(metadata$pretrain_vine_model, "nn_dynamic_t_vine"),
                   sprintf("pretrain_vine_model=%s", metadata$pretrain_vine_model))
  require_protocol(identical(metadata$finetune_vine_model, "nn_dynamic_t_vine"),
                   sprintf("finetune_vine_model=%s", metadata$finetune_vine_model))
  require_protocol(metadata$pretrain_vine_structure %in%
                     c("nn_dynamic_all_tree_dvine", "nn_dynamic_truncated_dvine"),
                   sprintf("pretrain_vine_structure=%s", metadata$pretrain_vine_structure))
  raw_truncation <- suppressWarnings(as.numeric(metadata$vine_truncation_level))
  # Schema-5 seven-asset bundles stored Inf to mean all trees. Preserve those
  # immutable training controls while new bundles store the explicit d-1 value.
  active_trees <- if (length(raw_truncation) == 1L &&
                      is.infinite(raw_truncation) && raw_truncation > 0) {
    length(asset_names) - 1L
  } else suppressWarnings(as.integer(raw_truncation))
  require_protocol(length(active_trees) == 1L && is.finite(active_trees) &&
                     active_trees >= 1L && active_trees < length(asset_names),
                   sprintf("vine_truncation_level=%s", metadata$vine_truncation_level))
  expected_dynamic_edges <- if (length(active_trees) == 1L &&
                                is.finite(active_trees) && active_trees >= 1L &&
                                active_trees < length(asset_names)) {
    sum(length(asset_names) - seq_len(active_trees))
  } else NA_real_
  require_protocol(length(metadata$dynamic_vine_edges) == 1L &&
                     is.finite(as.numeric(metadata$dynamic_vine_edges)) &&
                     as.numeric(metadata$dynamic_vine_edges) == expected_dynamic_edges,
                   sprintf("dynamic_vine_edges=%s", metadata$dynamic_vine_edges))
  require_protocol(identical(as.integer(metadata$sequence_length), env_seq_len),
                   sprintf("sequence_length=%s (expected %s)", metadata$sequence_length, env_seq_len))
  require_protocol(identical(as.integer(metadata$reserved_evaluation_steps), evaluation_periods),
                   sprintf("reserved_evaluation_steps=%s (expected %s)",
                           metadata$reserved_evaluation_steps, evaluation_periods))
  require_protocol(identical(as.character(metadata$asset_names),
                             as.character(asset_names)),
                   "synthetic bundle asset names/order do not match the active panel")
  if (identical(attr(returns, "source_kind"), "daily_log_returns")) {
    require_protocol(identical(as.character(metadata$source_data_sha256),
                               as.character(attr(returns, "source_sha256"))),
                     "synthetic bundle source-data hash does not match the active panel")
    require_protocol(identical(as.character(metadata$source_manifest_sha256),
                               as.character(attr(returns, "source_manifest_sha256"))),
                     "synthetic bundle source-manifest hash does not match the active panel")
  } else {
    require_protocol(identical(as.character(metadata$source_data_md5),
                               as.character(attr(returns, "source_md5"))),
                     "synthetic bundle source-data MD5 does not match the active panel")
  }
}
require_protocol(exists("train_end", envir = bundle, inherits = FALSE),
                 "train_end is missing")
if (exists("train_end", envir = bundle, inherits = FALSE)) {
  require_protocol(as.integer(train_end) ==
                     as.integer(get("train_end", envir = bundle, inherits = FALSE)),
                   "train_end does not match the locked calendar split")
}
if (length(protocol_failures)) {
  stop(paste0(
    "Training bundle protocol validation failed:\n - ",
    paste(protocol_failures, collapse = "\n - "),
    "\nRegenerate only if these fields do not describe the intended current protocol."))
}
cat(sprintf("Loaded %d pre-training and %d fine-tuning episodes.\n", length(pretrain_returns), length(finetune_returns)))
if (pretrain_episodes != length(pretrain_returns)) {
  stop("PRETRAIN_EPISODES must equal the generated episode count so all and only the synthetic data are used once.")
}
if (finetune_episodes != length(finetune_returns)) {
  stop("FINETUNE_EPISODES must equal the number of distinct historical trajectories; pass selection is handled separately.")
}

# ---- Create environments ----
make_training_environment <- function(episode_returns) {
  environment <- RLEnvironment$new(
    marginals, asset_names,
    vine = pretrain_vine, vine_sequence = NULL,
    ref_col = ref_col,
    gamma = env_gamma,
    lambda = env_lambda,
    kappa = env_kappa,
    T = env_T,
    w0 = env_w0,
    n_sim_cvar = n_sim_cvar,
    sim_cores = vine_sim_cores,
    seq_len = env_seq_len,
    holding_days = env_holding_days,
    gross_leverage = env_gross_leverage,
    net_exposure = env_net_exposure,
    max_long_weight = env_max_long_weight,
    max_short_weight = env_max_short_weight,
    short_borrow_rate = env_short_borrow_rate,
    cash_borrow_rate = env_cash_borrow_rate,
    utility_mode = env_utility_mode,
    vine_observation_mode = vine_observation_mode,
    vine_feature_mode = vine_feature_mode,
    cvar_observation_mode = cvar_observation_mode,
    cvar_reward_mode = cvar_reward_mode,
    episode_sampling = "sequential")
  environment$set_precomputed_returns(episode_returns)
  environment
}

env_pretrain <- make_training_environment(pretrain_returns)
# The publication gate is deliberately evaluated without exploration noise on
# a fixed, held-in synthetic diagnostic slice.  This tests the learned policy,
# not the stochastic trajectory used to populate replay memory.
env_pretrain_gate <- make_training_environment(
  tail(pretrain_returns, pretrain_behavior_gate_window))

# The last historical episode contains the final 24 months of the training
# prefix. Any earlier 24-month episode whose realised target overlaps it is
# purged from the diagnostic fit. This validation block is strictly before the
# separately locked OOS block and cannot adapt the preregistered one-pass refit.
selection_fit_count <- length(finetune_returns) - env_T
has_purged_finetune_validation <- selection_fit_count >= 1L
if (!has_purged_finetune_validation && finetune_max_selection_passes != 1L) {
  stop(paste0(
    "Too few historical episodes for purged fine-tuning model selection, ",
    "and more than one selection pass was requested."))
}
if (has_purged_finetune_validation) {
  finetune_selection_returns <- finetune_returns[seq_len(selection_fit_count)]
  finetune_validation_returns <- finetune_returns[length(finetune_returns)]
  purged_selection_episodes <-
    length(finetune_returns) - selection_fit_count - 1L
  if (purged_selection_episodes != env_T - 1L) {
    stop("Purged validation geometry is inconsistent with the episode horizon.")
  }
  finetune_selection_mode <- "fixed_one_pass_purged_validation_diagnostic_only"
  cat(sprintf(
    paste0("Fine-tuning diagnostic: %d fit episodes, %d purged overlapping ",
           "episodes, 1 validation episode; final refit uses all %d.\n"),
    selection_fit_count, purged_selection_episodes,
    length(finetune_returns)))
} else {
  # A complete target-disjoint validation episode would require more than T
  # overlapping historical trajectories. Short walk-forward prefixes cannot
  # provide that geometry. Because the pass count is preregistered at one, no
  # model selection is performed: train once from the pretrained checkpoint on
  # every available trajectory and record the unavailable diagnostic explicitly.
  finetune_selection_returns <- list()
  finetune_validation_returns <- list()
  purged_selection_episodes <- NA_integer_
  finetune_selection_mode <-
    "fixed_one_pass_all_history_no_validation_short_window"
  cat(sprintf(
    paste0("Fine-tuning diagnostic unavailable: %d historical episodes are ",
           "insufficient for a target-disjoint %d-step validation block; ",
           "the preregistered one-pass refit uses all episodes.\n"),
    length(finetune_returns), env_T))
}

balanced_pass_schedule <- function(episodes, passes, seed_offset, stage) {
  set.seed(train_seed + as.integer(seed_offset))
  orders <- lapply(seq_len(passes), function(pass) sample.int(length(episodes)))
  schedule <- do.call(c, lapply(orders, function(order) episodes[order]))
  order_table <- do.call(rbind, lapply(seq_along(orders), function(pass) {
    data.frame(stage = stage, pass = pass, position = seq_along(orders[[pass]]),
               original_episode = orders[[pass]])
  }))
  list(schedule = schedule, order = order_table)
}
if (has_purged_finetune_validation) {
  selection_schedule <- balanced_pass_schedule(
    finetune_selection_returns, finetune_max_selection_passes, 1001L,
    "selection_fit")
} else {
  selection_schedule <- list(
    schedule = list(),
    order = data.frame(stage = character(), pass = integer(),
                       position = integer(), original_episode = integer()))
}
refit_schedule <- balanced_pass_schedule(
  finetune_returns, finetune_max_selection_passes, 2001L,
  "all_history_refit")
write.csv(rbind(selection_schedule$order, refit_schedule$order),
          file.path(output_dir, "finetune_episode_schedule.csv"), row.names = FALSE)

if (has_purged_finetune_validation) {
  env_finetune_selection <- make_training_environment(selection_schedule$schedule)
  env_finetune_validation <- make_training_environment(
    finetune_validation_returns)
}
env_finetune_all <- make_training_environment(refit_schedule$schedule)

# cat("Pre-computed returns created.")
# # Break the HPC code here to only output the pre-computed returns and exit.
# stop("Exiting after creating pre-computed returns.")

# ---- Expose R functions to Python ----
r_env_pretrain_reset <- function() env_pretrain$reset()
r_env_pretrain_step <- function(action) env_pretrain$step(action)
r_env_pretrain_get_action_dim <- function() as.integer(env_pretrain$get_action_dim())
r_env_pretrain_get_obs_dim <- function() as.integer(env_pretrain$get_obs_dim())
r_env_pretrain_get_seq_len <- function() as.integer(env_pretrain$get_seq_len())
r_env_pretrain_get_history <- function() env_pretrain$get_history()

expose_environment <- function(environment) {
  list(
    reset = function() environment$reset(),
    step = function(action) environment$step(action),
    action_dim = function() as.integer(environment$get_action_dim()),
    obs_dim = function() as.integer(environment$get_obs_dim()),
    seq_len = function() as.integer(environment$get_seq_len()),
    history = function() environment$get_history())
}
if (has_purged_finetune_validation) {
  r_finetune_selection <- expose_environment(env_finetune_selection)
  r_finetune_validation <- expose_environment(env_finetune_validation)
}
r_finetune_all <- expose_environment(env_finetune_all)
r_pretrain_gate <- expose_environment(env_pretrain_gate)
r_env_pretrain_gate_reset <- r_pretrain_gate$reset
r_env_pretrain_gate_step <- r_pretrain_gate$step
r_env_pretrain_gate_get_action_dim <- r_pretrain_gate$action_dim
r_env_pretrain_gate_get_obs_dim <- r_pretrain_gate$obs_dim
r_env_pretrain_gate_get_seq_len <- r_pretrain_gate$seq_len
r_env_pretrain_gate_get_history <- r_pretrain_gate$history
if (has_purged_finetune_validation) {
  r_env_finetune_selection_reset <- r_finetune_selection$reset
  r_env_finetune_selection_step <- r_finetune_selection$step
  r_env_finetune_selection_get_action_dim <- r_finetune_selection$action_dim
  r_env_finetune_selection_get_obs_dim <- r_finetune_selection$obs_dim
  r_env_finetune_selection_get_seq_len <- r_finetune_selection$seq_len
  r_env_finetune_selection_get_history <- r_finetune_selection$history
  r_env_finetune_validation_reset <- r_finetune_validation$reset
  r_env_finetune_validation_step <- r_finetune_validation$step
  r_env_finetune_validation_get_action_dim <- r_finetune_validation$action_dim
  r_env_finetune_validation_get_obs_dim <- r_finetune_validation$obs_dim
  r_env_finetune_validation_get_seq_len <- r_finetune_validation$seq_len
  r_env_finetune_validation_get_history <- r_finetune_validation$history
}
r_env_finetune_all_reset <- r_finetune_all$reset
r_env_finetune_all_step <- r_finetune_all$step
r_env_finetune_all_get_action_dim <- r_finetune_all$action_dim
r_env_finetune_all_get_obs_dim <- r_finetune_all$obs_dim
r_env_finetune_all_get_seq_len <- r_finetune_all$seq_len
r_env_finetune_all_get_history <- r_finetune_all$history
r_finetune_has_purged_validation <- function()
  isTRUE(has_purged_finetune_validation)
r_finetune_selection_mode <- function() finetune_selection_mode
r_finetune_selection_episode_count <- function()
  as.integer(length(finetune_selection_returns))
r_finetune_all_episode_count <- function() as.integer(length(finetune_returns))

# ============================================================================
# Python Code — with File Logging
# ============================================================================
py_run_string("
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
import sys
import csv
import math
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
from rl.action_projection import (
    action_components as shared_action_components,
    portfolio_books as shared_portfolio_books,
)

SEED = int(os.environ.get('TRAIN_SEED'))
OUTPUT_DIR = os.environ.get('TRAIN_OUTPUT_DIR')
REQUESTED_DEVICE = os.environ.get('TRAIN_DEVICE')
SMOKE_TEST = os.environ.get('TRAIN_SMOKE_TEST', 'false').lower() in ('1', 'true', 'yes')
VERBOSE = os.environ.get('TRAIN_VERBOSE', 'false').lower() in ('1', 'true', 'yes')
PRETRAIN_EPISODES = int(os.environ.get('PRETRAIN_EPISODES'))
FINETUNE_EPISODES = int(os.environ.get('FINETUNE_EPISODES'))
PRETRAIN_BATCH_SIZE = int(os.environ.get('PRETRAIN_BATCH_SIZE'))
FINETUNE_BATCH_SIZE = int(os.environ.get('FINETUNE_BATCH_SIZE'))
LR_ACTOR = float(os.environ.get('LR_ACTOR'))
LR_CRITIC = float(os.environ.get('LR_CRITIC'))
FINETUNE_LR_ACTOR = float(os.environ.get('FINETUNE_LR_ACTOR', LR_ACTOR))
FINETUNE_LR_CRITIC = float(os.environ.get('FINETUNE_LR_CRITIC', LR_CRITIC))
DISCOUNT = float(os.environ.get('DISCOUNT'))
TAU = float(os.environ.get('TAU'))
HIDDEN = int(os.environ.get('HIDDEN'))
NUM_LAYERS = int(os.environ.get('NUM_LAYERS'))
REPLAY_CAPACITY = int(os.environ.get('REPLAY_CAPACITY'))
ENTROPY_COEF = float(os.environ.get('ENTROPY_COEF'))
DIRECTION_LOGIT_BOUND = float(os.environ.get('DIRECTION_LOGIT_BOUND', '1.0'))
PROJECTION_TEMPERATURE = float(os.environ.get('PROJECTION_TEMPERATURE', '1.5'))
INITIAL_LEVERAGE_GATE = float(os.environ.get('INITIAL_LEVERAGE_GATE', '0.10'))
LEVERAGE_SOFT_TARGET = float(os.environ.get('LEVERAGE_SOFT_TARGET', '0.80'))
LEVERAGE_PENALTY_COEF = float(os.environ.get('LEVERAGE_PENALTY_COEF', '0.25'))
GRAD_CLIP_NORM = float(os.environ.get('GRAD_CLIP_NORM'))
POLICY_DELAY = int(os.environ.get('POLICY_DELAY', '2'))
TARGET_POLICY_NOISE = float(os.environ.get('TARGET_POLICY_NOISE', '0.2'))
TARGET_NOISE_CLIP = float(os.environ.get('TARGET_NOISE_CLIP', '0.5'))
PRETRAIN_RANDOM_EXPLORATION_STEPS = int(os.environ.get('PRETRAIN_RANDOM_EXPLORATION_STEPS', '1000'))
PRETRAIN_BEHAVIOR_GATE_WINDOW = int(os.environ.get('PRETRAIN_BEHAVIOR_GATE_WINDOW', '100'))
PRETRAIN_MAX_MEAN_LEVERAGE_GATE = float(os.environ.get('PRETRAIN_MAX_MEAN_LEVERAGE_GATE', '0.95'))
PRETRAIN_MAX_MEAN_GROSS_CAP_FRACTION = float(os.environ.get('PRETRAIN_MAX_MEAN_GROSS_CAP_FRACTION', '0.75'))
PRETRAIN_WARN_POSITION_CAP_FRACTION = float(os.environ.get('PRETRAIN_WARN_POSITION_CAP_FRACTION', '0.75'))
PRETRAIN_MIN_MEAN_NORMALIZED_ENTROPY = float(os.environ.get('PRETRAIN_MIN_MEAN_NORMALIZED_ENTROPY', '0.70'))
PRETRAIN_MIN_Q05_NORMALIZED_ENTROPY = float(os.environ.get('PRETRAIN_MIN_Q05_NORMALIZED_ENTROPY', '0.50'))
PRETRAIN_MIN_MEAN_EFFECTIVE_POSITIONS = float(os.environ.get('PRETRAIN_MIN_MEAN_EFFECTIVE_POSITIONS', '2.50'))
PRETRAIN_MAX_POSITION_LIMIT_VIOLATION = float(os.environ.get('PRETRAIN_MAX_POSITION_LIMIT_VIOLATION', '1e-6'))
PRETRAIN_MAX_GATE_GROSS_MAE = float(os.environ.get('PRETRAIN_MAX_GATE_GROSS_MAE', '1e-5'))
PRETRAIN_MAX_MEAN_TURNOVER = float(os.environ.get('PRETRAIN_MAX_MEAN_TURNOVER', '1.0'))
PRETRAIN_BEHAVIOR_GATE_MODE = os.environ.get(
    'PRETRAIN_BEHAVIOR_GATE_MODE', 'strict').lower()
FINETUNE_RANDOM_EXPLORATION_STEPS = int(os.environ.get('FINETUNE_RANDOM_EXPLORATION_STEPS', '0'))
FINETUNE_MAX_SELECTION_PASSES = int(os.environ.get('FINETUNE_MAX_SELECTION_PASSES', '8'))
FINETUNE_VALIDATION_PATIENCE = int(os.environ.get('FINETUNE_VALIDATION_PATIENCE', '2'))
FINETUNE_VALIDATION_MIN_DELTA = float(os.environ.get('FINETUNE_VALIDATION_MIN_DELTA', '0.005'))
DIAGNOSTIC_INTERVAL = int(os.environ.get('DIAGNOSTIC_INTERVAL', '100'))
DETERMINISTIC_ALGORITHMS = os.environ.get('DETERMINISTIC_ALGORITHMS', 'true').lower() in ('1', 'true', 'yes')
USE_AMP = os.environ.get('USE_AMP', 'false').lower() in ('1', 'true', 'yes')
LOAD_MODEL_PATH = os.environ.get('LOAD_MODEL_PATH', '')
PRETRAIN_NOISE_SCALE = float(os.environ.get('PRETRAIN_NOISE_SCALE'))
PRETRAIN_NOISE_DECAY = float(os.environ.get('PRETRAIN_NOISE_DECAY'))
PRETRAIN_UPDATES = int(os.environ.get('PRETRAIN_UPDATES_PER_STEP'))
FINETUNE_NOISE_SCALE = float(os.environ.get('FINETUNE_NOISE_SCALE'))
FINETUNE_NOISE_DECAY = float(os.environ.get('FINETUNE_NOISE_DECAY'))
FINETUNE_UPDATES = int(os.environ.get('FINETUNE_UPDATES_PER_STEP'))
GROSS_LEVERAGE = float(os.environ.get('ENV_GROSS_LEVERAGE'))
NET_EXPOSURE = float(os.environ.get('ENV_NET_EXPOSURE'))
MAX_LONG_WEIGHT = float(os.environ.get('ENV_MAX_LONG_WEIGHT', '0.60'))
MAX_SHORT_WEIGHT = float(os.environ.get('ENV_MAX_SHORT_WEIGHT', '0.20'))
SHORT_BORROW_RATE = float(os.environ.get('ENV_SHORT_BORROW_RATE', '0.03'))
CASH_BORROW_RATE = float(os.environ.get('ENV_CASH_BORROW_RATE', '0.02'))
UTILITY_MODE = os.environ.get('ENV_UTILITY_MODE')
VINE_OBSERVATION_MODE = os.environ.get('VINE_OBSERVATION_MODE', 'full')
if VINE_OBSERVATION_MODE not in ('full', 'zero'):
    raise RuntimeError('VINE_OBSERVATION_MODE must be full or zero.')
VINE_FEATURE_MODE = os.environ.get('VINE_FEATURE_MODE', VINE_OBSERVATION_MODE)
CVAR_OBSERVATION_MODE = os.environ.get('CVAR_OBSERVATION_MODE', VINE_OBSERVATION_MODE)
CVAR_REWARD_MODE = os.environ.get('CVAR_REWARD_MODE', 'full')
PRETRAIN_DATA_MODE = os.environ.get('PRETRAIN_DATA_MODE', 'vine_synthetic')
RL_ALGORITHM = os.environ.get('RL_ALGORITHM', 'td3').lower()
POLICY_ENCODER = os.environ.get('POLICY_ENCODER', 'lstm').lower()
CHECKPOINT_PREFIX = os.environ.get('CHECKPOINT_PREFIX', 'td3_lstm_vine')
RUN_FINETUNE = os.environ.get('RUN_FINETUNE', 'true').lower() in ('1', 'true', 'yes')
SEQ_LEN = int(os.environ.get('ENV_SEQ_LEN'))
if any(mode not in ('full', 'zero') for mode in
       (VINE_FEATURE_MODE, CVAR_OBSERVATION_MODE, CVAR_REWARD_MODE)):
    raise RuntimeError('Invalid independent vine/CVaR masking mode.')
NO_VINE_SIGNAL_MASK = ('explicit_vine_and_scenario_cvar_v1'
                       if VINE_FEATURE_MODE == 'zero' and
                       CVAR_OBSERVATION_MODE == 'zero' else 'not_applicable')
if GROSS_LEVERAGE < abs(NET_EXPOSURE):
    raise RuntimeError('Gross leverage must be at least abs(net exposure).')
if NET_EXPOSURE <= 0:
    raise RuntimeError('Schema-5 rank-partition actions require positive net exposure.')
FULL_SHORT_BUDGET = 0.5 * (GROSS_LEVERAGE - NET_EXPOSURE)
FULL_LONG_BUDGET = NET_EXPOSURE + FULL_SHORT_BUDGET
SHORT_SUPPORT_SIZE = (int(math.ceil(FULL_SHORT_BUDGET / MAX_SHORT_WEIGHT - 1e-12))
                      if FULL_SHORT_BUDGET > 0 else 0)
if SHORT_SUPPORT_SIZE * MAX_SHORT_WEIGHT < FULL_SHORT_BUDGET - 1e-8:
    raise RuntimeError('Short position limit is infeasible for the rank partition.')

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
if DETERMINISTIC_ALGORITHMS:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
if DETERMINISTIC_ALGORITHMS:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if REQUESTED_DEVICE == 'cpu':
    device = torch.device('cpu')
elif torch.cuda.is_available():
    device = torch.device('cuda')
elif REQUESTED_DEVICE == 'cuda':
    raise RuntimeError('TRAIN_DEVICE=cuda was requested but PyTorch cannot see a CUDA GPU.')
else:
    device = torch.device('cpu')

# TF32 accelerates float32 matrix operations on Ampere-and-newer NVIDIA GPUs
# while retaining float32 model state.  It has no effect on older GPUs/CPUs.
if device.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
    except AttributeError:
        pass

# ================================================================
# Open log file for debug output
# ================================================================
log_file = open(os.path.join(OUTPUT_DIR, 'debug_output.txt'), 'w', buffering=1)
log_file.write('='*60 + '\\n')
log_file.write('PYTHON DEBUG LOG STARTED\\n')
log_file.write('='*60 + '\\n')

def log_print(*args, **kwargs):
    import time
    timestamp = time.strftime('%H:%M:%S')
    msg = ' '.join(str(a) for a in args)
    log_file.write(f'[{timestamp}] {msg}\\n')
    print(msg, flush=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
log_print(f'Run seed: {SEED}')
log_print('Run mode: ' + ('SMOKE TEST' if SMOKE_TEST else 'full'))
log_print(f'PyTorch: {torch.__version__}')
log_print(f'Device: {device}')
log_print(f'Action schema: interior_rank_partition_leverage_gate_v5; projection temperature={PROJECTION_TEMPERATURE}')
log_print(f'Experiment: algorithm={RL_ALGORITHM}; encoder={POLICY_ENCODER}; '
          f'vine_features={VINE_FEATURE_MODE}; cvar_observation={CVAR_OBSERVATION_MODE}; '
          f'cvar_reward={CVAR_REWARD_MODE}; pretraining={PRETRAIN_DATA_MODE}; '
          f'behavior_gate={PRETRAIN_BEHAVIOR_GATE_MODE}; finetune={RUN_FINETUNE}')
if device.type == 'cuda':
    log_print(f'GPU: {torch.cuda.get_device_name(device)}')

class VinePortfolioEnv(gym.Env):
    def __init__(self, reset_fn, step_fn, render_fn, get_history_fn, action_dim, obs_dim, seq_len):
        super().__init__()
        self.reset_fn = reset_fn
        self.step_fn = step_fn
        self.render_fn = render_fn
        self.get_history_fn = get_history_fn
        self.seq_len = int(seq_len)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.action_space = spaces.Box(low=-GROSS_LEVERAGE, high=GROSS_LEVERAGE,
                                       shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.seq_len, self.obs_dim), dtype=np.float32)
        self.history = None

    def reset(self):
        obs = self.reset_fn()
        self.history = self.get_history_fn()
        if len(self.history) == 0:
            self.history = np.zeros((self.seq_len, self.obs_dim), dtype=np.float32)
        return np.array(self.history, dtype=np.float32)

    def step(self, action):
        if isinstance(action, (list, np.ndarray)):
            action_list = np.array(action).flatten().tolist()
        else:
            action_list = list(action)
        res = self.step_fn(action_list)
        obs = np.array(res['observation'], dtype=np.float32)
        reward = float(res['reward'])
        done = bool(res['done'])
        info = dict(res['info'])
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = obs
        return self.history.copy(), reward, done, info

    def render(self, mode='human'):
        self.render_fn()

# ── LSTM Actor ───────────────────────────────────────────────────────────
class LSTMActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128, num_layers=2):
        super().__init__()
        self.input_norm = nn.LayerNorm(int(obs_dim))
        self.lstm = nn.LSTM(int(obs_dim), hidden, num_layers, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            # One cross-sectional score per asset plus one scalar gross-
            # leverage gate. High-ranked assets form the long book and the
            # lowest-ranked assets form a disjoint short book.
            nn.Linear(hidden, int(action_dim) + 1)
        )
        # Begin from a conservative, state-independent leverage prior.  The
        # old random head was driven to sigmoid(raw_gate) ~= 1 during warm-up
        # before the actor ever controlled an environment action.
        with torch.no_grad():
            self.fc[-1].weight[-1].zero_()
            self.fc[-1].bias[-1].fill_(
                math.log(INITIAL_LEVERAGE_GATE / (1.0 - INITIAL_LEVERAGE_GATE)))
    def forward(self, state_seq, hidden=None):
        out, hidden = self.lstm(self.input_norm(state_seq), hidden)
        out = self.layernorm(out)
        action = self.fc(out)
        return action, hidden

def action_components(raw_action):
    return shared_action_components(raw_action, DIRECTION_LOGIT_BOUND)

def portfolio_books(raw_action):
    return shared_portfolio_books(
        raw_action,
        direction_logit_bound=DIRECTION_LOGIT_BOUND,
        projection_temperature=PROJECTION_TEMPERATURE,
        net_exposure=NET_EXPOSURE,
        full_short_budget=FULL_SHORT_BUDGET,
        max_long_weight=MAX_LONG_WEIGHT,
        max_short_weight=MAX_SHORT_WEIGHT,
        short_support_size=SHORT_SUPPORT_SIZE,
    )

def portfolio_weights(raw_action):
    # Disjoint books make gross exposure exactly
    # abs(net) + gate * (gross_cap - abs(net)); the gate is identifiable.
    long_probs, short_probs, _, long_budget, short_budget = portfolio_books(raw_action)
    return long_budget * long_probs - short_budget * short_probs

def allocation_entropy(raw_action):
    long_probs, short_probs, _, _, short_budget = portfolio_books(raw_action)
    return book_entropy(long_probs, short_probs, short_budget).mean()

def book_entropy(long_probs, short_probs, short_budget):
    has_short_book = (short_budget > 1e-10).to(long_probs.dtype).squeeze(-1)
    short_entropy = torch.sum(short_probs * torch.log(short_probs + 1e-8), dim=-1)
    return -0.5 * (torch.sum(
        long_probs * torch.log(long_probs + 1e-8), dim=-1) +
        has_short_book * short_entropy)

def leverage_saturation_penalty(raw_action):
    _, leverage_gate = action_components(raw_action)
    return torch.relu(leverage_gate - LEVERAGE_SOFT_TARGET).square().mean()

def effective_leverage(weights):
    denominator = max(GROSS_LEVERAGE - abs(NET_EXPOSURE), 1e-12)
    return ((weights.abs().sum(dim=-1, keepdim=True) - abs(NET_EXPOSURE)) /
            denominator).clamp(0.0, 1.0)

# ── LSTM Critic with Shape Assertions ─────────────────────────────────
class LSTMCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128, num_layers=2):
        super().__init__()
        self.input_norm = nn.LayerNorm(int(obs_dim))
        self.lstm = nn.LSTM(int(obs_dim), hidden, num_layers, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden)
        self.fc = nn.Sequential(
            nn.Linear(hidden + int(action_dim), hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, state_seq, action, hidden=None):
        lstm_out, hidden = self.lstm(self.input_norm(state_seq), hidden)
        last_hidden = self.layernorm(lstm_out[:, -1, :])
        x = torch.cat([last_hidden, action], dim=-1)
        q = self.fc(x)
        return q, hidden

# ── Replay Buffer ──────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_seq, action, reward, next_state_seq, done):
        action = np.array(action, dtype=np.float32)
        self.buffer.append((
            np.array(state_seq, dtype=np.float32),
            np.array(action, dtype=np.float32),
            float(reward),
            np.array(next_state_seq, dtype=np.float32),
            float(done)
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.from_numpy(np.stack(states)).to(device, non_blocking=True)
        actions = torch.from_numpy(np.stack(actions)).to(device, non_blocking=True)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.from_numpy(np.stack(next_states)).to(device, non_blocking=True)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# ── DDPG Agent ──────────────────────────────────────────────────────────
class TD3Agent:
    def __init__(self, obs_dim, action_dim, hidden=128, num_layers=2,
                 lr_actor=1e-4, lr_critic=1e-4, gamma=1.0, tau=0.005,
                 entropy_coef=0.0, grad_clip_norm=1.0,
                 random_exploration_steps=0):
        self.actor = LSTMActor(obs_dim, action_dim, hidden, num_layers).to(device)
        self.actor_target = LSTMActor(obs_dim, action_dim, hidden, num_layers).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = LSTMCritic(obs_dim, action_dim, hidden, num_layers).to(device)
        self.critic2 = LSTMCritic(obs_dim, action_dim, hidden, num_layers).to(device)
        self.critic_target = LSTMCritic(obs_dim, action_dim, hidden, num_layers).to(device)
        self.critic2_target = LSTMCritic(obs_dim, action_dim, hidden, num_layers).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.tau = tau
        self.action_dim = int(action_dim)
        self.obs_dim = int(obs_dim)
        if self.action_dim <= SHORT_SUPPORT_SIZE:
            raise RuntimeError('Position limits leave no asset available for the long book.')
        if ((self.action_dim - SHORT_SUPPORT_SIZE) * MAX_LONG_WEIGHT <
                FULL_LONG_BUDGET - 1e-8):
            raise RuntimeError('Long position limit is infeasible for the rank partition.')
        self.update_count = 0
        self.entropy_coef = entropy_coef
        self.grad_clip_norm = grad_clip_norm
        self.total_actions = 0
        self.random_exploration_steps = int(random_exploration_steps)
        self.last_action_diagnostics = {}
        # Independent scalers prevent the old code from updating one shared
        # scale twice on delayed-policy iterations. Their states are persisted.
        amp_enabled = USE_AMP and device.type == 'cuda'
        self.critic_scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
        self.actor_scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)


    def select_action(self, state_seq, noise_scale=0.0):
        self.actor.eval()
        with torch.no_grad():
            if state_seq.ndim == 2:
                state_tensor = torch.from_numpy(np.asarray(state_seq, dtype=np.float32)).unsqueeze(0).to(device)
            else:
                state_tensor = torch.from_numpy(np.asarray(state_seq, dtype=np.float32)).to(device)
            if self.total_actions < self.random_exploration_steps:
                raw_action = torch.randn((state_tensor.shape[0], self.action_dim + 1), device=device)
            else:
                raw_action, _ = self.actor(state_tensor)
                raw_action = raw_action[:, -1, :]
        self.actor.train()
        # Exploration belongs in the raw directional/leverage outputs, before
        # the differentiable portfolio projection.
        if noise_scale > 0:
            raw_action = raw_action + torch.randn_like(raw_action) * noise_scale
        long_probs, short_probs, gate, long_budget, short_budget = portfolio_books(raw_action)
        projected = long_budget * long_probs - short_budget * short_probs
        entropy = book_entropy(long_probs, short_probs, short_budget)
        realised_gate = effective_leverage(projected)
        at_position_cap = torch.any(
            (projected >= MAX_LONG_WEIGHT - 1e-4) |
            (projected <= -MAX_SHORT_WEIGHT + 1e-4), dim=-1)
        self.last_action_diagnostics = {
            'leverage_gate': float(gate.mean().detach().cpu()),
            'effective_leverage': float(realised_gate.mean().detach().cpu()),
            'gate_gross_error': float(torch.abs(gate - realised_gate).mean().detach().cpu()),
            'position_at_cap': float(at_position_cap.to(torch.float32).mean().detach().cpu()),
            'direction_entropy': float(entropy.mean().detach().cpu())}
        action = projected.detach().cpu().numpy().flatten()
        self.total_actions += 1
        if VERBOSE:
            log_print(f'Action: {action[:3]}...')
        return action

    def deterministic_action(self, state_seq):
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.from_numpy(np.asarray(state_seq, dtype=np.float32))
            if state_tensor.ndim == 2:
                state_tensor = state_tensor.unsqueeze(0)
            state_tensor = state_tensor.to(device)
            raw_action, _ = self.actor(state_tensor)
            action = portfolio_weights(raw_action[:, -1, :])
        self.actor.train()
        return action.detach().cpu().numpy().flatten()

    def raw_deterministic_tensor(self, state_tensor):
        raw_action, _ = self.actor(state_tensor)
        return raw_action[:, -1, :]

    def sync_targets(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    def update(self, replay_buffer, batch_size=32):
        if len(replay_buffer) < batch_size:
            return
        diagnostic_due = (self.update_count + 1) % DIAGNOSTIC_INTERVAL == 0

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Critic update
        with torch.no_grad():
            next_raw_action, _ = self.actor_target(next_states)
            next_raw_action = next_raw_action[:, -1, :]
            smoothing = torch.randn_like(next_raw_action) * TARGET_POLICY_NOISE
            next_raw_action = next_raw_action + smoothing.clamp(-TARGET_NOISE_CLIP, TARGET_NOISE_CLIP)
            next_action = portfolio_weights(next_raw_action)
            target_q1, _ = self.critic_target(next_states, next_action)
            target_q2, _ = self.critic2_target(next_states, next_action)
            target_q = rewards + (1 - dones) * self.gamma * torch.minimum(target_q1, target_q2)

        with torch.amp.autocast('cuda', enabled=(USE_AMP and device.type == 'cuda')):
            current_q, _ = self.critic(states, actions)
            current_q2, _ = self.critic2(states, actions)
            critic_loss = nn.MSELoss()(current_q, target_q) + nn.MSELoss()(current_q2, target_q)
        if diagnostic_due and not torch.isfinite(critic_loss):
            raise RuntimeError(f'Non-finite critic loss at update {self.update_count + 1}.')

        self.critic_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        self.critic_scaler.scale(critic_loss).backward()
        self.critic_scaler.unscale_(self.critic_optimizer)
        self.critic_scaler.unscale_(self.critic2_optimizer)
        critic_grad = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        critic2_grad = torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.grad_clip_norm)
        self.critic_scaler.step(self.critic_optimizer)
        self.critic_scaler.step(self.critic2_optimizer)
        self.critic_scaler.update()

        self.update_count += 1
        # Delayed policy and target updates are the defining TD3 correction for
        # critic over-estimation and rapidly moving targets.
        actor_loss_tensor = None
        actor_grad_tensor = None
        actor_gate_tensor = None
        actor_entropy_tensor = None
        actor_leverage_penalty_tensor = None
        actor_gross_tensor = None
        actor_max_weight_tensor = None
        actor_gate_gross_mae_tensor = None
        if self.update_count % POLICY_DELAY == 0:
            with torch.amp.autocast('cuda', enabled=(USE_AMP and device.type == 'cuda')):
                pred_raw_action, _ = self.actor(states)
                pred_raw_action_last = pred_raw_action[:, -1, :]
                (pred_long_probs, pred_short_probs, diagnostic_gate,
                 pred_long_budget, pred_short_budget) = portfolio_books(
                    pred_raw_action_last)
                pred_action = (pred_long_budget * pred_long_probs -
                               pred_short_budget * pred_short_probs)
                actor_entropy = book_entropy(
                    pred_long_probs, pred_short_probs, pred_short_budget).mean()
                actor_leverage_penalty = torch.relu(
                    diagnostic_gate - LEVERAGE_SOFT_TARGET).square().mean()
                q_value, _ = self.critic(states, pred_action)
                actor_loss = -q_value.mean()
                if self.entropy_coef != 0.0:
                    actor_loss = actor_loss - self.entropy_coef * actor_entropy
                if LEVERAGE_PENALTY_COEF != 0.0:
                    actor_loss = actor_loss + LEVERAGE_PENALTY_COEF * actor_leverage_penalty
            if diagnostic_due and not torch.isfinite(actor_loss):
                raise RuntimeError(f'Non-finite actor loss at update {self.update_count}.')

            self.actor_optimizer.zero_grad()
            self.actor_scaler.scale(actor_loss).backward()
            self.actor_scaler.unscale_(self.actor_optimizer)
            actor_grad = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
            self.actor_scaler.step(self.actor_optimizer)
            self.actor_scaler.update()
            actor_loss_tensor = actor_loss.detach()
            actor_grad_tensor = actor_grad.detach()
            with torch.no_grad():
                actor_gate_tensor = diagnostic_gate.mean()
                actor_entropy_tensor = actor_entropy.detach()
                actor_leverage_penalty_tensor = actor_leverage_penalty.detach()
                actor_gross_tensor = pred_action.abs().sum(dim=-1).mean()
                actor_max_weight_tensor = pred_action.abs().max(dim=-1).values.mean()
                actor_gate_gross_mae_tensor = torch.abs(
                    diagnostic_gate - effective_leverage(pred_action)).mean()

            for target, source in zip(self.actor_target.parameters(), self.actor.parameters()):
                target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)
            for target, source in zip(self.critic_target.parameters(), self.critic.parameters()):
                target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)
            for target, source in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)

        if diagnostic_due:
            with torch.no_grad():
                twin_gap = torch.mean(torch.abs(current_q - current_q2))
            return {
                'update': self.update_count,
                'critic_loss': float(critic_loss.detach().cpu()),
                'actor_loss': None if actor_loss_tensor is None else float(actor_loss_tensor.cpu()),
                'q1_mean': float(current_q.mean().detach().cpu()),
                'q2_mean': float(current_q2.mean().detach().cpu()),
                'target_q_mean': float(target_q.mean().detach().cpu()),
                'twin_q_gap': float(twin_gap.detach().cpu()),
                'critic_grad_norm': float(critic_grad.detach().cpu()),
                'critic2_grad_norm': float(critic2_grad.detach().cpu()),
                'actor_grad_norm': None if actor_grad_tensor is None else float(actor_grad_tensor.cpu()),
                'actor_mean_leverage_gate': None if actor_gate_tensor is None else float(actor_gate_tensor.cpu()),
                'actor_mean_direction_entropy': None if actor_entropy_tensor is None else float(actor_entropy_tensor.cpu()),
                'actor_leverage_penalty': None if actor_leverage_penalty_tensor is None else float(actor_leverage_penalty_tensor.cpu()),
                'actor_mean_gross_exposure': None if actor_gross_tensor is None else float(actor_gross_tensor.cpu()),
                'actor_mean_max_abs_weight': None if actor_max_weight_tensor is None else float(actor_max_weight_tensor.cpu()),
                'actor_gate_gross_mae': None if actor_gate_gross_mae_tensor is None else float(actor_gate_gross_mae_tensor.cpu()),
                'critic_amp_scale': float(self.critic_scaler.get_scale()),
                'actor_amp_scale': float(self.actor_scaler.get_scale())}
        return None

    def save(self, path):
        parameter_count = sum(
            parameter.numel()
            for module in (self.actor, self.critic, self.critic2)
            for parameter in module.parameters() if parameter.requires_grad)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic2': self.critic2.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'update_count': self.update_count,
            'total_actions': self.total_actions,
            'critic_scaler': self.critic_scaler.state_dict(),
            'actor_scaler': self.actor_scaler.state_dict(),
            'architecture': {'obs_dim': self.obs_dim, 'action_dim': self.action_dim,
                             'seq_len': SEQ_LEN,
                             'actor_output_dim': self.action_dim + 1,
                             'hidden': self.actor.lstm.hidden_size, 'num_layers': self.actor.lstm.num_layers,
                             'agent': 'td3', 'state_normalization': 'layer_norm',
                             'action_mode': 'interior_rank_partition_leverage_gate_v5', 'gross_leverage': GROSS_LEVERAGE,
                             'net_exposure': NET_EXPOSURE, 'short_borrow_rate': SHORT_BORROW_RATE,
                             'cash_borrow_rate': CASH_BORROW_RATE, 'utility_mode': UTILITY_MODE,
                             'vine_observation_mode': VINE_OBSERVATION_MODE,
                             'vine_feature_mode': VINE_FEATURE_MODE,
                             'cvar_observation_mode': CVAR_OBSERVATION_MODE,
                             'cvar_reward_mode': CVAR_REWARD_MODE,
                             'pretrain_data_mode': PRETRAIN_DATA_MODE,
                             'pretrain_behavior_gate_mode': PRETRAIN_BEHAVIOR_GATE_MODE,
                             'rl_algorithm': RL_ALGORITHM,
                             'policy_encoder': POLICY_ENCODER,
                             'run_finetune': RUN_FINETUNE,
                             'no_vine_signal_mask': NO_VINE_SIGNAL_MASK,
                             'max_long_weight': MAX_LONG_WEIGHT,
                             'max_short_weight': MAX_SHORT_WEIGHT,
                             'direction_logit_bound': DIRECTION_LOGIT_BOUND,
                             'projection_temperature': PROJECTION_TEMPERATURE,
                             'initial_leverage_gate': INITIAL_LEVERAGE_GATE,
                             'allocation_entropy_coef': ENTROPY_COEF,
                             'leverage_soft_target': LEVERAGE_SOFT_TARGET,
                             'leverage_penalty_coef': LEVERAGE_PENALTY_COEF,
                             'short_support_size': SHORT_SUPPORT_SIZE,
                             'use_amp': USE_AMP,
                             'parameter_count': parameter_count,
                             'latent_entropy_objective': False,
                             'training_update_protocol': 'off_policy_replay',
                             'checkpoint_schema': 5}
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        architecture = ckpt.get('architecture', {})
        expected = {'obs_dim': self.obs_dim, 'action_dim': self.action_dim,
                    'seq_len': SEQ_LEN,
                    'actor_output_dim': self.action_dim + 1,
                    'hidden': self.actor.lstm.hidden_size,
                    'num_layers': self.actor.lstm.num_layers,
                    'agent': 'td3', 'state_normalization': 'layer_norm',
                    'action_mode': 'interior_rank_partition_leverage_gate_v5', 'gross_leverage': GROSS_LEVERAGE,
                    'net_exposure': NET_EXPOSURE, 'short_borrow_rate': SHORT_BORROW_RATE,
                    'cash_borrow_rate': CASH_BORROW_RATE, 'utility_mode': UTILITY_MODE,
                    'vine_observation_mode': VINE_OBSERVATION_MODE,
                    'vine_feature_mode': VINE_FEATURE_MODE,
                    'cvar_observation_mode': CVAR_OBSERVATION_MODE,
                    'cvar_reward_mode': CVAR_REWARD_MODE,
                    'pretrain_data_mode': PRETRAIN_DATA_MODE,
                    'pretrain_behavior_gate_mode': PRETRAIN_BEHAVIOR_GATE_MODE,
                    'rl_algorithm': RL_ALGORITHM,
                    'policy_encoder': POLICY_ENCODER,
                    'run_finetune': RUN_FINETUNE,
                    'max_long_weight': MAX_LONG_WEIGHT,
                    'max_short_weight': MAX_SHORT_WEIGHT,
                    'direction_logit_bound': DIRECTION_LOGIT_BOUND,
                    'projection_temperature': PROJECTION_TEMPERATURE,
                    'initial_leverage_gate': INITIAL_LEVERAGE_GATE,
                    'allocation_entropy_coef': ENTROPY_COEF,
                    'leverage_soft_target': LEVERAGE_SOFT_TARGET,
                    'leverage_penalty_coef': LEVERAGE_PENALTY_COEF,
                    'short_support_size': SHORT_SUPPORT_SIZE,
                    'use_amp': USE_AMP,
                    'parameter_count': sum(
                        parameter.numel()
                        for module in (self.actor, self.critic, self.critic2)
                        for parameter in module.parameters() if parameter.requires_grad),
                    'latent_entropy_objective': False,
                    'training_update_protocol': 'off_policy_replay',
                    'checkpoint_schema': 5}
        if VINE_OBSERVATION_MODE == 'zero':
            expected['no_vine_signal_mask'] = NO_VINE_SIGNAL_MASK
        # Schema-5 full-vine checkpoints predate this metadata field.  Treat
        # absence as 'full' only; a no-vine checkpoint must state 'zero'.
        actual_architecture = dict(architecture)
        actual_architecture.setdefault('vine_observation_mode', 'full')
        legacy_vine_mode = actual_architecture['vine_observation_mode']
        actual_architecture.setdefault('vine_feature_mode', legacy_vine_mode)
        actual_architecture.setdefault('cvar_observation_mode', legacy_vine_mode)
        actual_architecture.setdefault('cvar_reward_mode', 'full')
        actual_architecture.setdefault('pretrain_data_mode', 'vine_synthetic')
        actual_architecture.setdefault('pretrain_behavior_gate_mode', 'strict')
        actual_architecture.setdefault('rl_algorithm', 'td3')
        actual_architecture.setdefault('policy_encoder', 'lstm')
        actual_architecture.setdefault('run_finetune', True)
        actual_architecture.setdefault('seq_len', SEQ_LEN)
        actual_architecture.setdefault('parameter_count', expected['parameter_count'])
        actual_architecture.setdefault('latent_entropy_objective', False)
        actual_architecture.setdefault('training_update_protocol', 'off_policy_replay')
        mismatches = {k: (actual_architecture.get(k), v) for k, v in expected.items()
                      if actual_architecture.get(k) != v}
        if mismatches:
            raise RuntimeError(f'Checkpoint is incompatible with the long-short multi-period setup: {mismatches}. Retrain it.')
        def strip_prefix(state_dict):
            return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        self.actor.load_state_dict(strip_prefix(ckpt['actor']))
        self.critic.load_state_dict(strip_prefix(ckpt['critic']))
        self.critic2.load_state_dict(strip_prefix(ckpt['critic2']))
        self.actor_target.load_state_dict(strip_prefix(ckpt.get('actor_target', ckpt['actor'])))
        self.critic_target.load_state_dict(strip_prefix(ckpt.get('critic_target', ckpt['critic'])))
        self.critic2_target.load_state_dict(strip_prefix(ckpt.get('critic2_target', ckpt['critic2'])))
        self.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        self.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
        self.critic2_optimizer.load_state_dict(ckpt['critic2_optimizer'])
        self.update_count = ckpt['update_count']
        self.total_actions = int(ckpt.get('total_actions', 0))
        if 'critic_scaler' in ckpt:
            self.critic_scaler.load_state_dict(ckpt['critic_scaler'])
        if 'actor_scaler' in ckpt:
            self.actor_scaler.load_state_dict(ckpt['actor_scaler'])
        # Reset learning rates after loading
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.actor_optimizer.defaults['lr']
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = self.critic_optimizer.defaults['lr']
        for param_group in self.critic2_optimizer.param_groups:
            param_group['lr'] = self.critic2_optimizer.defaults['lr']

# ── Training Function ──────────────────────────────────────────────────
training_episode_rows = []
training_update_rows = []
validation_rows = []

def write_rows(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def train_stage(env, agent, episodes, stage, batch_size=32, noise_scale=0.0,
                noise_decay=0.999, log_interval=1, updates_per_step=3,
                replay_buffer=None, episode_offset=0, pass_index=None):
    log_print('='*60)
    log_print(f'TRAIN STAGE STARTED: stage={stage}, episodes={episodes}, batch_size={batch_size}, updates_per_step={updates_per_step}')
    log_print('='*60)
    if replay_buffer is None:
        replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
    episode_rewards = []
    for ep in range(episodes):
        state_seq = env.reset()
        episode_reward = 0.0
        global_episode = episode_offset + ep
        current_noise = noise_scale * (noise_decay ** global_episode)
        t = 0
        done = False
        turnover_values, cvar_values, gross_values = [], [], []
        leverage_gate_values, effective_leverage_values = [], []
        gate_gross_error_values, position_cap_values = [], []
        direction_entropy_values = []
        short_values, max_weight_values = [], []
        terminal_wealth = float('nan')
        while not done and t < 100:
            action = agent.select_action(state_seq, noise_scale=current_noise)
            if not np.isfinite(action).all():
                raise RuntimeError(f'Non-finite action in {stage}, episode {global_episode + 1}, step {t + 1}.')
            next_state_seq, reward, done, info = env.step(action)
            if not np.isfinite(reward):
                raise RuntimeError(f'Non-finite reward in {stage}, episode {global_episode + 1}, step {t + 1}.')
            episode_reward += reward
            if getattr(agent, 'on_policy', False):
                agent.record_outcome(reward, done, next_state_seq)
            else:
                replay_buffer.push(state_seq, action, reward, next_state_seq, done)
                for _ in range(updates_per_step):
                    diagnostics = agent.update(replay_buffer, batch_size)
                    if diagnostics is not None:
                        diagnostics.update({'stage': stage, 'episode': global_episode + 1,
                                            'step': t + 1, 'pass': pass_index})
                        training_update_rows.append(diagnostics)

            terminal_wealth = float(info['wealth'])
            turnover_values.append(float(info['turnover']))
            cvar_values.append(float(info['cvar']))
            gross_values.append(float(info['gross_exposure']))
            short_values.append(float(info['short_notional']))
            max_weight_values.append(float(np.max(np.abs(np.asarray(info['weights'], dtype=float)))))
            leverage_gate_values.append(float(agent.last_action_diagnostics['leverage_gate']))
            actual_effective_leverage = (
                (float(info['gross_exposure']) - abs(NET_EXPOSURE)) /
                max(GROSS_LEVERAGE - abs(NET_EXPOSURE), 1e-12))
            actual_weights = np.asarray(info['weights'], dtype=float)
            effective_leverage_values.append(float(actual_effective_leverage))
            gate_gross_error_values.append(float(abs(
                agent.last_action_diagnostics['leverage_gate'] -
                actual_effective_leverage)))
            position_cap_values.append(float(np.any(
                (actual_weights >= MAX_LONG_WEIGHT - 1e-4) |
                (actual_weights <= -MAX_SHORT_WEIGHT + 1e-4))))
            direction_entropy_values.append(float(agent.last_action_diagnostics['direction_entropy']))
            state_seq = next_state_seq
            t += 1

        if not done:
            raise RuntimeError(f'{stage} episode {global_episode + 1} exceeded the step limit.')
        if getattr(agent, 'on_policy', False):
            diagnostics = agent.finish_episode()
            if diagnostics is not None:
                diagnostics.update({'stage': stage, 'episode': global_episode + 1,
                                    'step': t, 'pass': pass_index})
                training_update_rows.append(diagnostics)
        episode_rewards.append(episode_reward)
        training_episode_rows.append({
            'stage': stage, 'pass': pass_index,
            'episode': global_episode + 1, 'reward': episode_reward,
            'noise': current_noise, 'terminal_wealth': terminal_wealth,
            'mean_turnover': float(np.mean(turnover_values)),
            'q95_turnover': float(np.quantile(turnover_values, 0.95)),
            'mean_cvar': float(np.mean(cvar_values)),
            'mean_gross_exposure': float(np.mean(gross_values)),
            'fraction_at_gross_cap': float(np.mean(np.asarray(gross_values) >= GROSS_LEVERAGE - 1e-4)),
            'mean_short_notional': float(np.mean(short_values)),
            'max_abs_weight': float(np.max(max_weight_values)),
            'mean_leverage_gate': float(np.mean(leverage_gate_values)),
            'q95_leverage_gate': float(np.quantile(leverage_gate_values, 0.95)),
            'std_leverage_gate': float(np.std(leverage_gate_values, ddof=1)),
            'mean_effective_leverage': float(np.mean(effective_leverage_values)),
            'gate_gross_mae': float(np.mean(gate_gross_error_values)),
            'fraction_at_position_cap': float(np.mean(position_cap_values)),
            'mean_direction_entropy': float(np.mean(direction_entropy_values)),
            'total_actions': agent.total_actions,
            'update_count': agent.update_count})
        log_print(f'Episode {global_episode+1}  Reward: {episode_reward:10.6f}  Wealth: {terminal_wealth:12.2f}  Turnover: {np.mean(turnover_values):.4f}  Gross: {np.mean(gross_values):.4f}  Gate: {np.mean(leverage_gate_values):.4f}')
        if (ep + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            log_print(f'Episode {global_episode+1:6d}  AvgReward: {avg_reward:10.6f}  Noise: {current_noise:.4f}')
    log_print('TRAIN STAGE COMPLETE')
    log_file.flush()
    return episode_rewards, replay_buffer

def evaluate_policy(env, agent, stage, pass_index):
    state_seq = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    turnover_values, gross_values, cvar_values, short_values = [], [], [], []
    terminal_wealth = float('nan')
    while not done and steps < 100:
        action = agent.deterministic_action(state_seq)
        state_seq, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        terminal_wealth = float(info['wealth'])
        turnover_values.append(float(info['turnover']))
        gross_values.append(float(info['gross_exposure']))
        cvar_values.append(float(info['cvar']))
        short_values.append(float(info['short_notional']))
    if not done or not np.isfinite(total_reward) or not np.isfinite(terminal_wealth):
        raise RuntimeError(f'Validation failed at pass {pass_index}.')
    row = {'stage': stage, 'pass': pass_index, 'reward': total_reward,
           'terminal_wealth': terminal_wealth,
           'mean_turnover': float(np.mean(turnover_values)),
           'mean_gross_exposure': float(np.mean(gross_values)),
           'fraction_at_gross_cap': float(np.mean(np.asarray(gross_values) >= GROSS_LEVERAGE - 1e-4)),
           'mean_cvar': float(np.mean(cvar_values)),
           'mean_short_notional': float(np.mean(short_values))}
    validation_rows.append(row)
    validation_turnover = row['mean_turnover']
    validation_gross = row['mean_gross_exposure']
    log_print(f'VALIDATION pass={pass_index} reward={total_reward:.6f} wealth={terminal_wealth:.2f} turnover={validation_turnover:.4f} gross={validation_gross:.4f}')
    return row

def flush_training_diagnostics():
    write_rows(os.path.join(OUTPUT_DIR, 'training_episode_metrics.csv'), training_episode_rows)
    write_rows(os.path.join(OUTPUT_DIR, 'training_update_metrics.csv'), training_update_rows)
    write_rows(os.path.join(OUTPUT_DIR, 'finetune_validation_metrics.csv'), validation_rows)

def evaluate_pretraining_policy(env, agent):
    # Evaluate the final actor deterministically on a fixed held-in synthetic
    # diagnostic slice.  Training rows contain exploration noise and therefore
    # cannot establish whether the learned policy itself is diversified.
    rows = []
    long_support = agent.action_dim - SHORT_SUPPORT_SIZE
    maximum_book_entropy = 0.5 * (
        math.log(long_support) +
        (math.log(SHORT_SUPPORT_SIZE) if SHORT_SUPPORT_SIZE > 0 else 0.0))
    was_training = agent.actor.training
    agent.actor.eval()
    with torch.no_grad():
        for episode in range(PRETRAIN_BEHAVIOR_GATE_WINDOW):
            state_seq = env.reset()
            done = False
            step = 0
            while not done and step < 100:
                state_tensor = torch.from_numpy(
                    np.asarray(state_seq, dtype=np.float32)).unsqueeze(0).to(device)
                raw_action = agent.raw_deterministic_tensor(state_tensor)
                books = portfolio_books(raw_action)
                long_probs, short_probs, leverage_gate, long_budget, short_budget = books
                weights_tensor = long_budget * long_probs - short_budget * short_probs
                entropy = float(book_entropy(
                    long_probs, short_probs, short_budget).detach().cpu().item())
                weights = weights_tensor.detach().cpu().numpy().reshape(-1)
                gate = float(leverage_gate.detach().cpu().item())
                state_seq, reward, done, info = env.step(weights)
                gross = float(np.sum(np.abs(weights)))
                abs_shares = np.abs(weights) / max(gross, 1e-12)
                effective_positions = float(1.0 / np.sum(abs_shares ** 2))
                long_prob_np = long_probs.detach().cpu().numpy().reshape(-1)
                short_prob_np = short_probs.detach().cpu().numpy().reshape(-1)
                expected_gross = abs(NET_EXPOSURE) + gate * (
                    GROSS_LEVERAGE - abs(NET_EXPOSURE))
                violation = max(float(np.max(weights) - MAX_LONG_WEIGHT),
                                float(-MAX_SHORT_WEIGHT - np.min(weights)), 0.0)
                rows.append({
                    'episode': episode + 1, 'step': step + 1,
                    'reward': float(reward), 'leverage_gate': gate,
                    'gross_exposure': gross,
                    'turnover': float(info['turnover']),
                    'gate_gross_error': abs(gross - expected_gross),
                    'position_limit_violation': violation,
                    'at_position_cap': bool(
                        np.max(weights) >= MAX_LONG_WEIGHT - 1e-4 or
                        np.min(weights) <= -MAX_SHORT_WEIGHT + 1e-4),
                    'direction_entropy': entropy,
                    'normalized_direction_entropy': entropy / maximum_book_entropy,
                    'effective_positions': effective_positions,
                    'long_hhi': float(np.sum(long_prob_np ** 2)),
                    'short_hhi': float(np.sum(short_prob_np ** 2)),
                    'dominant_long_asset': int(np.argmax(weights)) + 1})
                step += 1
            if not done:
                raise RuntimeError(
                    f'Deterministic pre-training policy episode {episode + 1} did not terminate.')
    if was_training:
        agent.actor.train()
    write_rows(os.path.join(
        OUTPUT_DIR, 'pretraining_policy_diagnostics.csv'), rows)
    return rows

def enforce_pretraining_behavior_gate(policy_rows, agent):
    if not policy_rows:
        raise RuntimeError('Deterministic pre-training policy diagnostics are empty.')
    values = lambda key: np.asarray([row[key] for row in policy_rows], dtype=float)
    maximum_specifications = [
        ('mean_leverage_gate', float(np.mean(values('leverage_gate'))),
         PRETRAIN_MAX_MEAN_LEVERAGE_GATE),
        ('fraction_at_gross_cap', float(np.mean(
            values('gross_exposure') >= GROSS_LEVERAGE - 1e-4)),
         PRETRAIN_MAX_MEAN_GROSS_CAP_FRACTION),
        ('gate_gross_mae', float(np.mean(values('gate_gross_error'))),
         PRETRAIN_MAX_GATE_GROSS_MAE),
        ('mean_turnover', float(np.mean(values('turnover'))),
         PRETRAIN_MAX_MEAN_TURNOVER),
        ('max_position_limit_violation',
         float(np.max(values('position_limit_violation'))),
         PRETRAIN_MAX_POSITION_LIMIT_VIOLATION)]
    minimum_specifications = [
        ('mean_normalized_direction_entropy',
         float(np.mean(values('normalized_direction_entropy'))),
         PRETRAIN_MIN_MEAN_NORMALIZED_ENTROPY),
        ('q05_normalized_direction_entropy',
         float(np.quantile(values('normalized_direction_entropy'), 0.05)),
         PRETRAIN_MIN_Q05_NORMALIZED_ENTROPY),
        ('mean_effective_positions', float(np.mean(values('effective_positions'))),
         PRETRAIN_MIN_MEAN_EFFECTIVE_POSITIONS)]
    gate_rows = [{
        'window_episodes': PRETRAIN_BEHAVIOR_GATE_WINDOW,
        'metric': metric, 'value': value, 'comparison': '<=',
        'threshold': threshold,
        'pass': bool(np.isfinite(value) and value <= threshold)}
        for metric, value, threshold in maximum_specifications]
    gate_rows.extend({
        'window_episodes': PRETRAIN_BEHAVIOR_GATE_WINDOW,
        'metric': metric, 'value': value, 'comparison': '>=',
        'threshold': threshold,
        'pass': bool(np.isfinite(value) and value >= threshold)}
        for metric, value, threshold in minimum_specifications)
    write_rows(os.path.join(OUTPUT_DIR, 'pretraining_behavior_gate.csv'), gate_rows)

    dominant_counts = np.bincount(
        np.asarray([row['dominant_long_asset'] for row in policy_rows], dtype=int),
        minlength=agent.action_dim + 1)[1:]
    warning_rows = [
        {'metric': 'fraction_at_position_cap',
         'value': float(np.mean(values('at_position_cap'))),
         'warning_threshold': PRETRAIN_WARN_POSITION_CAP_FRACTION},
        {'metric': 'mean_long_hhi', 'value': float(np.mean(values('long_hhi'))),
         'warning_threshold': ''},
        {'metric': 'mean_short_hhi', 'value': float(np.mean(values('short_hhi'))),
         'warning_threshold': ''},
        {'metric': 'max_dominant_long_asset_fraction',
         'value': float(np.max(dominant_counts) / np.sum(dominant_counts)),
         'warning_threshold': ''}]
    write_rows(os.path.join(
        OUTPUT_DIR, 'pretraining_behavior_warnings.csv'), warning_rows)
    if warning_rows[0]['value'] > PRETRAIN_WARN_POSITION_CAP_FRACTION:
        log_print('WARNING: deterministic position-cap contact remains high at '
                  f\"{warning_rows[0]['value']:.4f}; this is diagnostic only because \"
                  'constraint equality is not a violation or a collapse statistic.')

    failures = [row for row in gate_rows if not row['pass']]
    structural_metrics = {'gate_gross_mae', 'max_position_limit_violation'}
    fatal_failures = [row for row in failures
                      if PRETRAIN_BEHAVIOR_GATE_MODE == 'strict'
                      or row['metric'] in structural_metrics
                      or not np.isfinite(row['value'])]
    if fatal_failures:
        details = '; '.join(
            f\"{row['metric']}={row['value']:.6g} {row['comparison']} {row['threshold']:.6g} failed\"
            for row in fatal_failures)
        raise RuntimeError(
            'Pre-training behavioural gate failed; historical fine-tuning is locked: ' +
            details)
    if failures:
        details = '; '.join(
            f\"{row['metric']}={row['value']:.6g} {row['comparison']} {row['threshold']:.6g}\"
            for row in failures)
        log_print('PRE-TRAINING BEHAVIOUR GATE REPORTED WITHOUT SELECTION: ' +
                  details + '. Continuing under the frozen report_only causal-control protocol.')
        return
    log_print('PRE-TRAINING BEHAVIOUR GATE PASSED on deterministic held-in episodes: ' +
              ', '.join(f\"{row['metric']}={row['value']:.4f}\" for row in gate_rows))

def create_env(reset_fn, step_fn, render_fn, get_history_fn, action_dim, obs_dim, seq_len):
    return VinePortfolioEnv(reset_fn, step_fn, render_fn, get_history_fn, action_dim, obs_dim, seq_len)

def create_agent(obs_dim, action_dim, lr_actor=1e-4, lr_critic=1e-4,
                 gamma=1.0, tau=0.005, random_exploration_steps=0):
    if RL_ALGORITHM == 'td3' and POLICY_ENCODER == 'lstm':
        return TD3Agent(obs_dim, action_dim, hidden=HIDDEN, num_layers=NUM_LAYERS,
                        lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma, tau=tau,
                        entropy_coef=ENTROPY_COEF, grad_clip_norm=GRAD_CLIP_NORM,
                        random_exploration_steps=random_exploration_steps)
    from rl.recurrent_baselines import BaselineConfig, build_agent
    configuration = BaselineConfig(
        algorithm=RL_ALGORITHM, encoder=POLICY_ENCODER,
        obs_dim=int(obs_dim), action_dim=int(action_dim), seq_len=SEQ_LEN,
        hidden=HIDDEN, num_layers=NUM_LAYERS,
        lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma, tau=tau,
        entropy_coef=ENTROPY_COEF, grad_clip_norm=GRAD_CLIP_NORM,
        random_exploration_steps=random_exploration_steps,
        replay_capacity=REPLAY_CAPACITY, policy_delay=POLICY_DELAY,
        target_policy_noise=TARGET_POLICY_NOISE,
        target_noise_clip=TARGET_NOISE_CLIP,
        diagnostic_interval=DIAGNOSTIC_INTERVAL,
        direction_logit_bound=DIRECTION_LOGIT_BOUND,
        projection_temperature=PROJECTION_TEMPERATURE,
        net_exposure=NET_EXPOSURE, gross_leverage=GROSS_LEVERAGE,
        max_long_weight=MAX_LONG_WEIGHT, max_short_weight=MAX_SHORT_WEIGHT,
        initial_leverage_gate=INITIAL_LEVERAGE_GATE,
        leverage_soft_target=LEVERAGE_SOFT_TARGET,
        leverage_penalty_coef=LEVERAGE_PENALTY_COEF,
        short_borrow_rate=SHORT_BORROW_RATE, cash_borrow_rate=CASH_BORROW_RATE,
        utility_mode=UTILITY_MODE, vine_observation_mode=VINE_OBSERVATION_MODE,
        vine_feature_mode=VINE_FEATURE_MODE,
        cvar_observation_mode=CVAR_OBSERVATION_MODE,
        cvar_reward_mode=CVAR_REWARD_MODE,
        pretrain_data_mode=PRETRAIN_DATA_MODE,
        pretrain_behavior_gate_mode=PRETRAIN_BEHAVIOR_GATE_MODE,
        run_finetune=RUN_FINETUNE, use_amp=USE_AMP)
    return build_agent(configuration, device)

def save_agent(agent, path):
    agent.save(path)

def load_agent(agent, path):
    agent.load(path)

log_print('PYTHON: Framework ready.')
log_print(f'Action schema: interior_rank_partition_leverage_gate_v5 | gross={GROSS_LEVERAGE:.3f} | net={NET_EXPOSURE:.3f} | max_long={MAX_LONG_WEIGHT:.3f} | max_short={MAX_SHORT_WEIGHT:.3f}')
log_print(f'Risk/training: entropy_coef={ENTROPY_COEF:.6g} | leverage_target={LEVERAGE_SOFT_TARGET:.3f} | leverage_penalty={LEVERAGE_PENALTY_COEF:.6g} | direction_bound={DIRECTION_LOGIT_BOUND:.3f} | AMP={USE_AMP}')
log_file.flush()
")

# ============================================================================
# Run Training
# ============================================================================

# ---- Pre-training ----
print_sep()
cat(sprintf("Stage 1: Pre-training (%s)\n", pretrain_data_mode))
print_sep()

py_run_string("
log_print('='*60)
log_print('STAGE 1: PRE-TRAINING')
log_print('='*60)

env_pretrain = create_env(
    reset_fn = r.r_env_pretrain_reset,
    step_fn = r.r_env_pretrain_step,
    render_fn = lambda: None,
    get_history_fn = r.r_env_pretrain_get_history,
    action_dim = int(r.r_env_pretrain_get_action_dim()),
    obs_dim = int(r.r_env_pretrain_get_obs_dim()),
    seq_len = int(r.r_env_pretrain_get_seq_len())
)
env_pretrain_gate = create_env(
    reset_fn = r.r_env_pretrain_gate_reset,
    step_fn = r.r_env_pretrain_gate_step,
    render_fn = lambda: None,
    get_history_fn = r.r_env_pretrain_gate_get_history,
    action_dim = int(r.r_env_pretrain_gate_get_action_dim()),
    obs_dim = int(r.r_env_pretrain_gate_get_obs_dim()),
    seq_len = int(r.r_env_pretrain_gate_get_seq_len())
)

agent = create_agent(
    obs_dim = int(r.r_env_pretrain_get_obs_dim()),
    action_dim = int(r.r_env_pretrain_get_action_dim()),
    lr_actor = LR_ACTOR,
    lr_critic = LR_CRITIC,
    gamma = DISCOUNT,
    tau = TAU,
    random_exploration_steps = PRETRAIN_RANDOM_EXPLORATION_STEPS
)

pretrain_rewards, _ = train_stage(
    env_pretrain, agent,
    episodes = PRETRAIN_EPISODES,
    stage = 'pretrain',
    batch_size = PRETRAIN_BATCH_SIZE,
    noise_scale = PRETRAIN_NOISE_SCALE,
    noise_decay = PRETRAIN_NOISE_DECAY,
    log_interval = 10,
    updates_per_step = PRETRAIN_UPDATES
)
save_agent(agent, os.path.join(OUTPUT_DIR, CHECKPOINT_PREFIX + '_pretrained.pt'))
flush_training_diagnostics()
pretraining_policy_rows = evaluate_pretraining_policy(env_pretrain_gate, agent)
enforce_pretraining_behavior_gate(pretraining_policy_rows, agent)
log_print('Pre-training complete. Agent saved.')
")

# ---- Fine-tuning ----
print_sep()
cat("Stage 2: Fine-tuning on Real Data\n")
print_sep()

if (run_finetune) {
py_run_string("
log_print('='*60)
log_print('STAGE 2: FINE-TUNING')
log_print('='*60)

has_purged_finetune_validation = bool(r.r_finetune_has_purged_validation())
finetune_selection_mode = str(r.r_finetune_selection_mode())
if has_purged_finetune_validation:
    env_finetune = create_env(
        reset_fn = r.r_env_finetune_selection_reset,
        step_fn = r.r_env_finetune_selection_step,
        render_fn = lambda: None,
        get_history_fn = r.r_env_finetune_selection_get_history,
        action_dim = int(r.r_env_finetune_selection_get_action_dim()),
        obs_dim = int(r.r_env_finetune_selection_get_obs_dim()),
        seq_len = int(r.r_env_finetune_selection_get_seq_len())
    )
    env_finetune_validation = create_env(
        reset_fn = r.r_env_finetune_validation_reset,
        step_fn = r.r_env_finetune_validation_step,
        render_fn = lambda: None,
        get_history_fn = r.r_env_finetune_validation_get_history,
        action_dim = int(r.r_env_finetune_validation_get_action_dim()),
        obs_dim = int(r.r_env_finetune_validation_get_obs_dim()),
        seq_len = int(r.r_env_finetune_validation_get_seq_len())
    )
env_finetune_all = create_env(
    reset_fn = r.r_env_finetune_all_reset,
    step_fn = r.r_env_finetune_all_step,
    render_fn = lambda: None,
    get_history_fn = r.r_env_finetune_all_get_history,
    action_dim = int(r.r_env_finetune_all_get_action_dim()),
    obs_dim = int(r.r_env_finetune_all_get_obs_dim()),
    seq_len = int(r.r_env_finetune_all_get_seq_len())
)

pretrained_path = (LOAD_MODEL_PATH if LOAD_MODEL_PATH else
                   os.path.join(OUTPUT_DIR, CHECKPOINT_PREFIX + '_pretrained.pt'))
if not os.path.exists(pretrained_path):
    raise RuntimeError(f'Pretrained checkpoint not found: {pretrained_path}')

def new_finetune_agent():
    candidate = create_agent(
        obs_dim = int(r.r_env_finetune_all_get_obs_dim()),
        action_dim = int(r.r_env_finetune_all_get_action_dim()),
        lr_actor = FINETUNE_LR_ACTOR,
        lr_critic = FINETUNE_LR_CRITIC,
        gamma = DISCOUNT,
        tau = TAU,
        random_exploration_steps = FINETUNE_RANDOM_EXPLORATION_STEPS)
    load_agent(candidate, pretrained_path)
    candidate.sync_targets()
    return candidate

# Diagnose the preregistered one-pass adaptation on a chronologically later,
# target-disjoint pre-holdout block. No final-OOS return is consulted.
best_pass = 1
best_reward = float('nan')
selection_count = 0
if has_purged_finetune_validation:
    selection_agent = new_finetune_agent()
    selection_replay = None
    selection_count = int(r.r_finetune_selection_episode_count())
    best_reward = -float('inf')
    passes_without_improvement = 0
    selection_episode_offset = 0
    for pass_index in range(1, FINETUNE_MAX_SELECTION_PASSES + 1):
        _, selection_replay = train_stage(
            env_finetune, selection_agent,
            episodes = selection_count,
            stage = 'finetune_selection',
            batch_size = FINETUNE_BATCH_SIZE,
            noise_scale = FINETUNE_NOISE_SCALE,
            noise_decay = FINETUNE_NOISE_DECAY,
            log_interval = max(1, selection_count),
            updates_per_step = FINETUNE_UPDATES,
            replay_buffer = selection_replay,
            episode_offset = selection_episode_offset,
            pass_index = pass_index)
        selection_episode_offset += selection_count
        validation = evaluate_policy(env_finetune_validation, selection_agent,
                                     'finetune_validation', pass_index)
        if validation['reward'] > best_reward + FINETUNE_VALIDATION_MIN_DELTA:
            best_reward = validation['reward']
            best_pass = pass_index
            passes_without_improvement = 0
        else:
            passes_without_improvement += 1
            if passes_without_improvement >= FINETUNE_VALIDATION_PATIENCE:
                log_print(f'Fine-tuning selection stopped after pass {pass_index}; best pass={best_pass}.')
                break
    del selection_agent, selection_replay
else:
    log_print('Purged validation diagnostic skipped for short history; '
              'executing the preregistered one-pass all-history refit.')

# Refit from the same pretrained checkpoint on every permissible historical
# trajectory for the preregistered single pass.  Validation is diagnostic.
import gc
gc.collect()
if device.type == 'cuda':
    torch.cuda.empty_cache()
agent_finetune = new_finetune_agent()
all_count = int(r.r_finetune_all_episode_count())
finetune_rewards, _ = train_stage(
    env_finetune_all, agent_finetune,
    episodes = all_count * best_pass,
    stage = 'finetune_refit_all',
    batch_size = FINETUNE_BATCH_SIZE,
    noise_scale = FINETUNE_NOISE_SCALE,
    noise_decay = FINETUNE_NOISE_DECAY,
    log_interval = all_count,
    updates_per_step = FINETUNE_UPDATES)
save_agent(agent_finetune, os.path.join(OUTPUT_DIR, CHECKPOINT_PREFIX + '_full.pt'))
flush_training_diagnostics()
with open(os.path.join(OUTPUT_DIR, 'finetune_selection.txt'), 'w') as stream:
    stream.write(f'selection_mode={finetune_selection_mode}\\n')
    stream.write(f'best_pass={best_pass}\\n')
    stream.write(f'best_validation_reward={best_reward:.12g}\\n')
    stream.write(f'fit_episodes={selection_count}\\n')
    stream.write(f'all_history_episodes={all_count}\\n')
log_print(f'Fine-tuning complete. Selected {best_pass} pass(es); final agent saved.')
")
} else {
  pretrained_path <- file.path(output_dir, paste0(checkpoint_prefix, "_pretrained.pt"))
  full_path <- file.path(output_dir, paste0(checkpoint_prefix, "_full.pt"))
  if (!file.exists(pretrained_path) || !file.copy(pretrained_path, full_path,
                                                  overwrite = FALSE)) {
    stop("Could not publish the pretrained-only final checkpoint.")
  }
  writeLines(c(
    "selection_mode=disabled_by_preregistered_pretrained_only_ablation",
    "best_pass=0", "fit_episodes=0", "all_history_episodes=0"),
    file.path(output_dir, "finetune_selection.txt"), useBytes = TRUE)
  cat("Fine-tuning skipped by RUN_FINETUNE=false; pretrained checkpoint copied as final.\n")
}

print_sep()
cat("TRAINING COMPLETE\n")
print_sep()
