# No-holdout integrity and policy sanity check for the trained TD3-LSTM agent.
#
# This script deliberately reads only the historical fine-tuning episodes that
# were already used during training.  It never loads or constructs the final
# 24-month evaluation episode, so running it cannot reveal holdout performance.

suppressPackageStartupMessages({
  library(yaml)
  library(reticulate)
  library(qs)
})

args <- commandArgs(trailingOnly = TRUE)
config_file <- if (length(args)) args[[1L]] else "config/config.yaml"
if (!file.exists(config_file)) stop(sprintf("Configuration file not found: %s", config_file))
config <- yaml::yaml.load_file(config_file)

required_scalar <- function(value, label, mode = c("numeric", "integer", "character")) {
  mode <- match.arg(mode)
  converted <- switch(mode,
    numeric = as.numeric(value), integer = as.integer(value), character = as.character(value))
  if (length(converted) != 1L || is.na(converted) ||
      (mode != "character" && !is.finite(converted)) ||
      (mode == "character" && !nzchar(converted))) {
    stop(sprintf("Invalid configuration value: %s", label))
  }
  converted
}

seed <- required_scalar(Sys.getenv("TRAIN_SEED", as.character(config$general$seed)),
                        "general.seed", "integer")
model_dir <- required_scalar(Sys.getenv("TRAIN_OUTPUT_DIR", config$general$output_dir),
                             "general.output_dir", "character")
finetune_file <- required_scalar(Sys.getenv("FINETUNE_RETURNS_FILE", config$vine$finetune_returns_file),
                                 "vine.finetune_returns_file", "character")
gamma <- required_scalar(config$environment$gamma, "environment.gamma")
lambda <- required_scalar(config$environment$lambda, "environment.lambda")
kappa <- required_scalar(config$environment$kappa, "environment.kappa")
episode_length <- required_scalar(config$environment$T, "environment.T", "integer")
w0 <- required_scalar(config$environment$w0, "environment.w0")
seq_len <- required_scalar(config$environment$seq_len, "environment.seq_len", "integer")
gross_leverage <- required_scalar(config$environment$gross_leverage,
                                  "environment.gross_leverage")
net_exposure <- required_scalar(config$environment$net_exposure,
                                "environment.net_exposure")
max_long_weight <- required_scalar(config$environment$max_long_weight,
                                   "environment.max_long_weight")
max_short_weight <- required_scalar(config$environment$max_short_weight,
                                    "environment.max_short_weight")
short_borrow_rate <- required_scalar(config$environment$short_borrow_rate,
                                     "environment.short_borrow_rate")
cash_borrow_rate <- required_scalar(config$environment$cash_borrow_rate,
                                    "environment.cash_borrow_rate")
utility_mode <- required_scalar(config$environment$utility_mode,
                                "environment.utility_mode", "character")
hidden <- required_scalar(config$agent$hidden, "agent.hidden", "integer")
num_layers <- required_scalar(config$agent$num_layers, "agent.num_layers", "integer")
direction_logit_bound <- required_scalar(config$agent$direction_logit_bound,
                                         "agent.direction_logit_bound")
projection_temperature <- required_scalar(config$agent$projection_temperature,
                                          "agent.projection_temperature")
initial_leverage_gate <- required_scalar(config$agent$initial_leverage_gate,
                                         "agent.initial_leverage_gate")
allocation_entropy_coef <- required_scalar(config$agent$entropy_coef,
                                            "agent.entropy_coef")
leverage_soft_target <- required_scalar(config$agent$leverage_soft_target,
                                        "agent.leverage_soft_target")
leverage_penalty_coef <- required_scalar(config$agent$leverage_penalty_coef,
                                         "agent.leverage_penalty_coef")
warn_position_cap_fraction <- required_scalar(
  config$pretraining$warn_position_cap_fraction,
  "pretraining.warn_position_cap_fraction")
min_mean_normalized_entropy <- required_scalar(
  config$pretraining$min_mean_normalized_entropy,
  "pretraining.min_mean_normalized_entropy")
min_q05_normalized_entropy <- required_scalar(
  config$pretraining$min_q05_normalized_entropy,
  "pretraining.min_q05_normalized_entropy")
min_mean_effective_positions <- required_scalar(
  config$pretraining$min_mean_effective_positions,
  "pretraining.min_mean_effective_positions")
use_amp <- isTRUE(config$agent$use_amp)
configured_vine_mode <- if (!is.null(config$ablation$zero_vine_state) &&
                              isTRUE(config$ablation$zero_vine_state)) "zero" else "full"
vine_observation_mode <- Sys.getenv("VINE_OBSERVATION_MODE",
                                   unset = configured_vine_mode)
if (!vine_observation_mode %in% c("full", "zero")) {
  stop("VINE_OBSERVATION_MODE must be full or zero.")
}
vine_feature_mode <- Sys.getenv("VINE_FEATURE_MODE", vine_observation_mode)
cvar_observation_mode <- Sys.getenv("CVAR_OBSERVATION_MODE", vine_observation_mode)
cvar_reward_mode <- Sys.getenv("CVAR_REWARD_MODE", "full")
pretrain_data_mode <- Sys.getenv("PRETRAIN_DATA_MODE", "vine_synthetic")
pretrain_behavior_gate_mode <- tolower(Sys.getenv(
  "PRETRAIN_BEHAVIOR_GATE_MODE", "strict"))
rl_algorithm <- tolower(Sys.getenv("RL_ALGORITHM", "td3"))
policy_encoder <- tolower(Sys.getenv("POLICY_ENCODER", "lstm"))
checkpoint_prefix <- Sys.getenv("CHECKPOINT_PREFIX", "td3_lstm_vine")
run_finetune <- tolower(Sys.getenv("RUN_FINETUNE", "true")) %in%
  c("1", "true", "yes")
if (any(!c(vine_feature_mode, cvar_observation_mode, cvar_reward_mode) %in%
        c("full", "zero")) ||
    !pretrain_behavior_gate_mode %in% c("strict", "report_only") ||
    !grepl("^[A-Za-z0-9_]+$", checkpoint_prefix)) {
  stop("Invalid causal sanity-check mode or checkpoint prefix.")
}
expected_finetune_uses <- required_scalar(
  Sys.getenv("FINETUNE_EPISODES", as.character(config$finetuning$episodes)),
  "finetuning.episodes", "integer")

if (episode_length != 24L) stop("The sanity check expects the trained 24-step objective.")
if (!identical(utility_mode, "terminal_wealth_crra")) {
  stop("The sanity check supports only terminal_wealth_crra checkpoints.")
}
if (gross_leverage < abs(net_exposure)) stop("Gross leverage is below absolute net exposure.")
if (net_exposure <= 0) stop("Schema-5 rank-partition actions require positive net exposure.")
if (!file.exists(finetune_file)) {
  stop(sprintf("Fine-tuning episodes not found: %s\nRun this on the training machine after synthetic_returns.r.",
               finetune_file))
}

checkpoint_paths <- file.path(model_dir, c(
  pretrained = paste0(checkpoint_prefix, "_pretrained.pt"),
  full = paste0(checkpoint_prefix, "_full.pt")))
missing_checkpoints <- checkpoint_paths[!file.exists(checkpoint_paths)]
if (length(missing_checkpoints)) {
  stop(sprintf("Missing checkpoint(s): %s", paste(missing_checkpoints, collapse = ", ")))
}
pretraining_gate_file <- file.path(model_dir, "pretraining_behavior_gate.csv")
if (!file.exists(pretraining_gate_file)) {
  stop("Missing pretraining_behavior_gate.csv; the checkpoint is not eligible for sanity testing.")
}
pretraining_gate <- read.csv(pretraining_gate_file, stringsAsFactors = FALSE)
if (!nrow(pretraining_gate) || !"pass" %in% names(pretraining_gate) ||
    !all(as.logical(pretraining_gate$pass))) {
  stop("Pre-training behavioural gate did not pass; sanity testing is locked.")
}

# The portable QS contains only training-prefix historical episodes.  Avoid
# loading synthetic_returns.RData here because it also contains the much larger
# synthetic pre-training object and is unnecessary for this diagnostic.
finetune_returns <- qs::qread(finetune_file)
if (!is.list(finetune_returns) || !length(finetune_returns)) {
  stop("Fine-tuning artifact is empty or not an episode list.")
}
valid_episode <- vapply(finetune_returns, function(ep) {
  is.list(ep) && identical(ep$source, "historical_realised") &&
    is.list(ep$burnin_returns) && length(ep$burnin_returns) == seq_len &&
    is.list(ep$returns) && length(ep$returns) == episode_length &&
    is.list(ep$vine_states) && length(ep$vine_states) == episode_length
}, logical(1))
if (!all(valid_episode)) {
  stop(sprintf("Fine-tuning artifact contains %d invalid or non-training episodes.",
               sum(!valid_episode)))
}
if (expected_finetune_uses %% length(finetune_returns) != 0L) {
  stop("Configured fine-tuning episode count is not a balanced multiple of the stored trajectories.")
}

first_return <- finetune_returns[[1L]]$returns[[1L]]
action_dim <- ncol(as.matrix(first_return))
vine_dim <- length(finetune_returns[[1L]]$vine_states[[1L]])
obs_dim <- 1L + action_dim + action_dim + 1L + action_dim + 2L + vine_dim
if (action_dim < 2L || vine_dim < 1L) stop("Episode dimensions are invalid.")

project_long_short_weights <- function(weights) {
  weights <- as.numeric(weights)
  weights[!is.finite(weights)] <- 0
  base <- rep(net_exposure / action_dim, action_dim)
  centred <- weights - mean(weights)
  candidate <- base + centred
  if (sum(abs(candidate)) > gross_leverage + 1e-10 && sum(abs(centred)) > 0) {
    lower <- 0; upper <- 1
    for (iteration in seq_len(50L)) {
      middle <- (lower + upper) / 2
      if (sum(abs(base + middle * centred)) > gross_leverage) upper <- middle else lower <- middle
    }
    candidate <- base + lower * centred
  }
  candidate
}

crra_utility <- function(wealth_multiple) {
  if (gamma == 1) log(wealth_multiple) else
    (wealth_multiple^(1 - gamma) - 1) / (1 - gamma)
}

sanity_state <- new.env(parent = emptyenv())

advance_market_state <- function(gross_returns) {
  sanity_state$last_returns <- log(pmax(as.numeric(gross_returns), 1e-12))
  if (is.null(sanity_state$vol_history)) {
    initial_variance <- pmax(sanity_state$last_returns^2, 1e-6)
    sanity_state$vol_history <- matrix(rep(initial_variance, each = 20L),
                                       nrow = 20L, ncol = action_dim)
  }
  new_variance <- 0.97 * sanity_state$last_var +
    0.03 * sanity_state$last_returns^2
  sanity_state$vol_history <- rbind(
    sanity_state$vol_history[-1L, , drop = FALSE], new_variance)
  sanity_state$last_var <- new_variance
  sanity_state$last_vols <- sqrt(colMeans(sanity_state$vol_history)) * sqrt(12)
  invisible(NULL)
}

current_observation <- function() {
  # Preserve the frozen no-vine intervention while allowing the newer
  # component-level masks when the legacy intervention is not active.
  no_vine_observation <- identical(vine_observation_mode, "zero")
  vine_observation <- if (no_vine_observation) {
    numeric(vine_dim)
  } else if (identical(vine_feature_mode, "zero")) {
    numeric(vine_dim)
  } else {
    sanity_state$vine_state
  }
  # Match RLEnvironment exactly: scenario CVaR is a vine-derived feature and
  # must not remain as an indirect dependence channel in the no-vine control.
  cvar_observation <- if (no_vine_observation) 0 else if (
    identical(cvar_observation_mode, "zero")
  ) 0 else sanity_state$last_cvar * 100
  observation <- c(
    sanity_state$wealth / w0,
    sanity_state$last_returns * 100,
    sanity_state$last_vols * 100,
    cvar_observation,
    sanity_state$previous_action,
    sum(abs(sanity_state$previous_action)),
    sum(sanity_state$previous_action),
    vine_observation)
  observation[!is.finite(observation)] <- 0
  if (length(observation) != obs_dim) stop("Constructed observation has the wrong dimension.")
  as.numeric(observation)
}

sanity_reset_episode <- function(episode_index) {
  episode_index <- as.integer(episode_index)
  if (episode_index < 1L || episode_index > length(finetune_returns)) {
    stop("Episode index is out of bounds.")
  }
  episode <- finetune_returns[[episode_index]]
  sanity_state$episode <- episode
  sanity_state$step <- 1L
  sanity_state$wealth <- w0
  sanity_state$previous_action <- rep(net_exposure / action_dim, action_dim)
  sanity_state$last_returns <- rep(0, action_dim)
  sanity_state$last_vols <- rep(0.01, action_dim)
  sanity_state$last_var <- 0.01^2
  sanity_state$vol_history <- NULL
  sanity_state$last_cvar <- 0
  sanity_state$vine_state <- as.numeric(episode$vine_states[[1L]])
  history <- vector("list", seq_len)
  for (index in seq_len(seq_len)) {
    if (!is.null(episode$burnin_vine_states)) {
      sanity_state$vine_state <- as.numeric(episode$burnin_vine_states[[index]])
    }
    advance_market_state(episode$burnin_returns[[index]])
    history[[index]] <- current_observation()
  }
  sanity_state$vine_state <- as.numeric(episode$vine_states[[1L]])
  history[[seq_len]] <- current_observation()
  sanity_state$history <- do.call(rbind, history)
  sanity_state$history
}

sanity_get_history <- function() sanity_state$history

sanity_step <- function(action) {
  step <- sanity_state$step
  if (step > episode_length) stop("Episode is complete; reset before stepping again.")
  weights <- project_long_short_weights(action)
  if (max(weights) > max_long_weight + 1e-6 ||
      min(weights) < -max_short_weight - 1e-6) {
    stop("Policy action violates the configured single-asset position limits.")
  }
  return_matrix <- as.matrix(sanity_state$episode$returns[[step]])
  realised_gross <- as.numeric(return_matrix[1L, ])
  scenarios <- return_matrix[-1L, , drop = FALSE]
  if (!nrow(scenarios) || any(!is.finite(return_matrix)) || any(return_matrix <= 0)) {
    stop("Episode contains invalid realised or scenario returns.")
  }

  portfolio_gross <- 1 + sum(weights * (realised_gross - 1))
  turnover <- sum(abs(weights - sanity_state$previous_action))
  transaction_cost <- kappa * turnover
  short_notional <- sum(pmax(-weights, 0))
  cash_borrow_notional <- pmax(sum(weights) - 1, 0)
  holding_year_fraction <- if (!is.null(
      sanity_state$episode$holding_year_fractions)) {
    as.numeric(sanity_state$episode$holding_year_fractions[step])
  } else 1 / 12
  if (!is.finite(holding_year_fraction) || holding_year_fraction <= 0 ||
      holding_year_fraction > 1) stop("Invalid sanity holding-year fraction.")
  financing_cost <- (short_borrow_rate * short_notional +
                     cash_borrow_rate * cash_borrow_notional) *
    holding_year_fraction
  cost_multiplier <- exp(-transaction_cost - financing_cost)

  scenario_gross <- (1 + scenarios %*% weights - sum(weights)) * cost_multiplier
  losses <- 1 - as.numeric(scenario_gross)
  tail_count <- max(1L, ceiling(0.05 * length(losses)))
  cvar <- mean(sort(losses, decreasing = TRUE)[seq_len(tail_count)])

  net_portfolio_gross <- pmax(portfolio_gross * cost_multiplier, 1e-12)
  wealth_before <- sanity_state$wealth
  sanity_state$wealth <- wealth_before * net_portfolio_gross
  terminal_utility <- crra_utility(sanity_state$wealth / w0)
  previous_utility <- crra_utility(wealth_before / w0)
  utility_increment <- terminal_utility - previous_utility
  reward_cvar <- if (identical(cvar_reward_mode, "zero")) 0 else cvar
  reward <- utility_increment - lambda * reward_cvar

  sanity_state$previous_action <- weights * realised_gross / portfolio_gross
  if (any(!is.finite(sanity_state$previous_action))) {
    stop("Sanity replay produced non-finite drifted holdings.")
  }
  sanity_state$last_cvar <- cvar
  sanity_state$step <- step + 1L
  next_vine_index <- min(sanity_state$step, length(sanity_state$episode$vine_states))
  sanity_state$vine_state <- as.numeric(sanity_state$episode$vine_states[[next_vine_index]])
  advance_market_state(realised_gross)
  observation <- current_observation()
  sanity_state$history <- rbind(sanity_state$history[-1L, , drop = FALSE], observation)

  list(
    observation = observation,
    reward = reward,
    done = sanity_state$step > episode_length,
    info = list(
      wealth = sanity_state$wealth,
      portfolio_gross = portfolio_gross,
      net_portfolio_gross = net_portfolio_gross,
      cvar = cvar,
      turnover = turnover,
      transaction_cost = transaction_cost,
      financing_cost = financing_cost,
      short_notional = short_notional,
      utility = terminal_utility,
      gross_exposure = sum(abs(weights)),
      net_exposure = sum(weights),
      weights = weights,
      asset_gross_returns = realised_gross))
}

Sys.setenv(
  SANITY_SEED = as.character(seed),
  SANITY_MODEL_DIR = normalizePath(model_dir, winslash = "/", mustWork = TRUE),
  SANITY_OUTPUT_DIR = normalizePath(model_dir, winslash = "/", mustWork = TRUE),
  SANITY_OBS_DIM = as.character(obs_dim),
  SANITY_ACTION_DIM = as.character(action_dim),
  SANITY_VINE_DIM = as.character(vine_dim),
  SANITY_HIDDEN = as.character(hidden),
  SANITY_NUM_LAYERS = as.character(num_layers),
  SANITY_GROSS_LEVERAGE = as.character(gross_leverage),
  SANITY_NET_EXPOSURE = as.character(net_exposure),
  SANITY_MAX_LONG_WEIGHT = as.character(max_long_weight),
  SANITY_MAX_SHORT_WEIGHT = as.character(max_short_weight),
  SANITY_SHORT_BORROW_RATE = as.character(short_borrow_rate),
  SANITY_CASH_BORROW_RATE = as.character(cash_borrow_rate),
  SANITY_UTILITY_MODE = utility_mode,
  SANITY_DIRECTION_LOGIT_BOUND = as.character(direction_logit_bound),
  SANITY_PROJECTION_TEMPERATURE = as.character(projection_temperature),
  SANITY_INITIAL_LEVERAGE_GATE = as.character(initial_leverage_gate),
  SANITY_ALLOCATION_ENTROPY_COEF = as.character(allocation_entropy_coef),
  SANITY_LEVERAGE_SOFT_TARGET = as.character(leverage_soft_target),
  SANITY_LEVERAGE_PENALTY_COEF = as.character(leverage_penalty_coef),
  SANITY_WARN_POSITION_CAP_FRACTION = as.character(warn_position_cap_fraction),
  SANITY_MIN_MEAN_NORMALIZED_ENTROPY = as.character(min_mean_normalized_entropy),
  SANITY_MIN_Q05_NORMALIZED_ENTROPY = as.character(min_q05_normalized_entropy),
  SANITY_MIN_MEAN_EFFECTIVE_POSITIONS = as.character(min_mean_effective_positions),
  SANITY_USE_AMP = tolower(as.character(use_amp)),
  SANITY_VINE_OBSERVATION_MODE = vine_observation_mode,
  SANITY_VINE_FEATURE_MODE = vine_feature_mode,
  SANITY_CVAR_OBSERVATION_MODE = cvar_observation_mode,
  SANITY_CVAR_REWARD_MODE = cvar_reward_mode,
  SANITY_PRETRAIN_DATA_MODE = pretrain_data_mode,
  SANITY_PRETRAIN_BEHAVIOR_GATE_MODE = pretrain_behavior_gate_mode,
  SANITY_RL_ALGORITHM = rl_algorithm,
  SANITY_POLICY_ENCODER = policy_encoder,
  SANITY_CHECKPOINT_PREFIX = checkpoint_prefix,
  SANITY_RUN_FINETUNE = tolower(as.character(run_finetune)),
  VINE_FEATURE_MODE = vine_feature_mode,
  CVAR_OBSERVATION_MODE = cvar_observation_mode,
  CVAR_REWARD_MODE = cvar_reward_mode,
  SANITY_EPISODES = as.character(length(finetune_returns)),
  SANITY_EPISODE_LENGTH = as.character(episode_length),
  SANITY_SEQ_LEN = as.character(seq_len),
  SANITY_BASE_OBS_DIM = as.character(obs_dim - vine_dim)
)

if ("torch" %in% loadedNamespaces()) {
  stop("R torch/Lantern is loaded. Run this in a fresh Rscript process so Python PyTorch is isolated.")
}

py_run_string("
import os, sys, json, hashlib, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
from rl.action_projection import portfolio_books as shared_portfolio_books
from rl.checkpoint_attestation import resolve_architecture_mode
from rl.policy_inference_server_v2 import load_policy

SEED = int(os.environ['SANITY_SEED'])
MODEL_DIR = os.environ['SANITY_MODEL_DIR']
OUTPUT_DIR = os.path.join(os.environ['SANITY_OUTPUT_DIR'], 'sanity_no_holdout')
OBS_DIM = int(os.environ['SANITY_OBS_DIM'])
ACTION_DIM = int(os.environ['SANITY_ACTION_DIM'])
VINE_DIM = int(os.environ['SANITY_VINE_DIM'])
BASE_OBS_DIM = int(os.environ['SANITY_BASE_OBS_DIM'])
HIDDEN = int(os.environ['SANITY_HIDDEN'])
NUM_LAYERS = int(os.environ['SANITY_NUM_LAYERS'])
GROSS_LEVERAGE = float(os.environ['SANITY_GROSS_LEVERAGE'])
NET_EXPOSURE = float(os.environ['SANITY_NET_EXPOSURE'])
MAX_LONG_WEIGHT = float(os.environ['SANITY_MAX_LONG_WEIGHT'])
MAX_SHORT_WEIGHT = float(os.environ['SANITY_MAX_SHORT_WEIGHT'])
SHORT_BORROW_RATE = float(os.environ['SANITY_SHORT_BORROW_RATE'])
CASH_BORROW_RATE = float(os.environ['SANITY_CASH_BORROW_RATE'])
UTILITY_MODE = os.environ['SANITY_UTILITY_MODE']
DIRECTION_LOGIT_BOUND = float(os.environ['SANITY_DIRECTION_LOGIT_BOUND'])
PROJECTION_TEMPERATURE = float(os.environ['SANITY_PROJECTION_TEMPERATURE'])
INITIAL_LEVERAGE_GATE = float(os.environ['SANITY_INITIAL_LEVERAGE_GATE'])
ALLOCATION_ENTROPY_COEF = float(os.environ['SANITY_ALLOCATION_ENTROPY_COEF'])
LEVERAGE_SOFT_TARGET = float(os.environ['SANITY_LEVERAGE_SOFT_TARGET'])
LEVERAGE_PENALTY_COEF = float(os.environ['SANITY_LEVERAGE_PENALTY_COEF'])
WARN_POSITION_CAP_FRACTION = float(os.environ['SANITY_WARN_POSITION_CAP_FRACTION'])
MIN_MEAN_NORMALIZED_ENTROPY = float(os.environ['SANITY_MIN_MEAN_NORMALIZED_ENTROPY'])
MIN_Q05_NORMALIZED_ENTROPY = float(os.environ['SANITY_MIN_Q05_NORMALIZED_ENTROPY'])
MIN_MEAN_EFFECTIVE_POSITIONS = float(os.environ['SANITY_MIN_MEAN_EFFECTIVE_POSITIONS'])
USE_AMP = os.environ['SANITY_USE_AMP'].lower() in ('1', 'true', 'yes')
VINE_OBSERVATION_MODE = os.environ['SANITY_VINE_OBSERVATION_MODE']
VINE_FEATURE_MODE = os.environ['SANITY_VINE_FEATURE_MODE']
CVAR_OBSERVATION_MODE = os.environ['SANITY_CVAR_OBSERVATION_MODE']
CVAR_REWARD_MODE = os.environ['SANITY_CVAR_REWARD_MODE']
PRETRAIN_DATA_MODE = os.environ['SANITY_PRETRAIN_DATA_MODE']
PRETRAIN_BEHAVIOR_GATE_MODE = os.environ['SANITY_PRETRAIN_BEHAVIOR_GATE_MODE']
RL_ALGORITHM = os.environ['SANITY_RL_ALGORITHM']
POLICY_ENCODER = os.environ['SANITY_POLICY_ENCODER']
CHECKPOINT_PREFIX = os.environ['SANITY_CHECKPOINT_PREFIX']
RUN_FINETUNE = os.environ['SANITY_RUN_FINETUNE'].lower() in ('1', 'true', 'yes')
NO_VINE_SIGNAL_MASK = ('explicit_vine_and_scenario_cvar_v1'
                       if VINE_FEATURE_MODE == 'zero' and
                       CVAR_OBSERVATION_MODE == 'zero' else 'not_applicable')
N_EPISODES = int(os.environ['SANITY_EPISODES'])
EPISODE_LENGTH = int(os.environ['SANITY_EPISODE_LENGTH'])
SEQ_LEN = int(os.environ['SANITY_SEQ_LEN'])
FULL_SHORT_BUDGET = 0.5 * (GROSS_LEVERAGE - NET_EXPOSURE)
SHORT_SUPPORT_SIZE = (int(math.ceil(FULL_SHORT_BUDGET / MAX_SHORT_WEIGHT - 1e-12))
                      if FULL_SHORT_BUDGET > 0 else 0)
MAX_BOOK_ENTROPY = 0.5 * (
    math.log(ACTION_DIM - SHORT_SUPPORT_SIZE) +
    (math.log(SHORT_SUPPORT_SIZE) if SHORT_SUPPORT_SIZE > 0 else 0.0))
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(SEED); torch.manual_seed(SEED)

class LSTMActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_norm = nn.LayerNorm(OBS_DIM)
        self.lstm = nn.LSTM(OBS_DIM, HIDDEN, NUM_LAYERS, batch_first=True)
        self.layernorm = nn.LayerNorm(HIDDEN)
        self.fc = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                                nn.Linear(HIDDEN, ACTION_DIM + 1))
    def forward(self, state):
        output, _ = self.lstm(self.input_norm(state))
        return self.fc(self.layernorm(output[:, -1, :]))

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
    long_probs, short_probs, _, long_budget, short_budget = portfolio_books(raw_action)
    return long_budget * long_probs - short_budget * short_probs

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def tensor_audit(value, prefix='root'):
    records = []
    if torch.is_tensor(value):
        finite = bool(torch.isfinite(value).all().item()) if value.is_floating_point() or value.is_complex() else True
        maximum = float(value.detach().abs().max().item()) if value.numel() else 0.0
        records.append((prefix, int(value.numel()), finite, maximum))
    elif isinstance(value, dict):
        for key, child in value.items(): records.extend(tensor_audit(child, f'{prefix}.{key}'))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value): records.extend(tensor_audit(child, f'{prefix}[{index}]'))
    return records

def load_checkpoint(label, path):
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location='cpu')
    architecture = dict(checkpoint.get('architecture', {}))
    if int(architecture.get('checkpoint_schema', 0)) == 5:
        actual_architecture, mode_metadata_source = resolve_architecture_mode(
            Path(path), architecture, VINE_OBSERVATION_MODE)
        actual_architecture.setdefault('vine_feature_mode',
                                       actual_architecture['vine_observation_mode'])
        actual_architecture.setdefault('cvar_observation_mode',
                                       actual_architecture['vine_observation_mode'])
        actual_architecture.setdefault('cvar_reward_mode', 'full')
        actual_architecture.setdefault('pretrain_data_mode', 'vine_synthetic')
        actual_architecture.setdefault('pretrain_behavior_gate_mode', 'strict')
        actual_architecture.setdefault('rl_algorithm', 'td3')
        actual_architecture.setdefault('policy_encoder', 'lstm')
        actual_architecture.setdefault('run_finetune', True)
    else:
        actual_architecture = architecture
        mode_metadata_source = 'checkpoint_schema_6'
    expected = {
        'obs_dim': OBS_DIM, 'action_dim': ACTION_DIM,
        'actor_output_dim': ACTION_DIM + 1, 'hidden': HIDDEN,
        'num_layers': NUM_LAYERS,
        'state_normalization': 'layer_norm',
        'action_mode': 'interior_rank_partition_leverage_gate_v5',
        'gross_leverage': GROSS_LEVERAGE, 'net_exposure': NET_EXPOSURE,
        'short_borrow_rate': SHORT_BORROW_RATE,
        'cash_borrow_rate': CASH_BORROW_RATE, 'utility_mode': UTILITY_MODE,
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
        'allocation_entropy_coef': ALLOCATION_ENTROPY_COEF,
        'leverage_soft_target': LEVERAGE_SOFT_TARGET,
        'leverage_penalty_coef': LEVERAGE_PENALTY_COEF,
        'short_support_size': SHORT_SUPPORT_SIZE,
        'use_amp': USE_AMP}
    mismatches = {key: [actual_architecture.get(key), value]
                  for key, value in expected.items()
                  if actual_architecture.get(key) != value}
    tensors = tensor_audit(checkpoint)
    all_finite = bool(tensors) and all(record[2] for record in tensors)
    actor_state = {key.replace('_orig_mod.', ''): value
                   for key, value in checkpoint['actor'].items()}
    actor, _, loaded_architecture = load_policy(
        Path(path), OBS_DIM, ACTION_DIM, SEQ_LEN, torch.device('cpu'))
    target_state = {key.replace('_orig_mod.', ''): value
                    for key, value in checkpoint.get('actor_target', checkpoint['actor']).items()}
    common_target_keys = set(actor_state) & set(target_state)
    target_gap = (max(float((actor_state[key] - target_state[key]).abs().max().item())
                      for key in common_target_keys)
                  if common_target_keys else math.nan)
    integrity = {
        'model': label, 'path': path, 'sha256': sha256(path),
        'size_bytes': os.path.getsize(path), 'architecture_match': not mismatches,
        'architecture_mismatches': json.dumps(mismatches, sort_keys=True),
        'all_checkpoint_tensors_finite': all_finite,
        'tensor_count': len(tensors),
        'tensor_elements': sum(record[1] for record in tensors),
        'max_tensor_abs': max(record[3] for record in tensors) if tensors else math.nan,
        'actor_parameters': sum(value.numel() for value in actor_state.values()),
        'update_count': int(checkpoint.get('update_count', -1)),
        'actor_target_max_abs_gap': target_gap,
        'pytorch_version': torch.__version__,
        'mode_metadata_source': mode_metadata_source}
    if mismatches or not all_finite:
        raise RuntimeError(f'Checkpoint {label} failed integrity checks: {integrity}')
    return actor, integrity

actors, integrity_rows = {}, []
for label, filename in [('pretrained', 'td3_lstm_vine_pretrained.pt'),
                        ('full', 'td3_lstm_vine_full.pt')]:
    filename = f'{CHECKPOINT_PREFIX}_{label}.pt'
    actors[label], integrity = load_checkpoint(label, os.path.join(MODEL_DIR, filename))
    integrity_rows.append(integrity)

def actor_action(actor, state, include_diagnostics=False):
    tensor = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        raw_action = actor(tensor)
        if not bool(torch.isfinite(raw_action).all()):
            raise RuntimeError('Actor emitted a non-finite logit.')
        weights = portfolio_weights(raw_action).squeeze(0).numpy()
        if not include_diagnostics:
            return weights
        long_probs, short_probs, gate, _, short_budget = portfolio_books(raw_action)
        has_short_book = (short_budget > 1e-10).to(long_probs.dtype).squeeze(-1)
        entropy = -0.5 * (
            torch.sum(long_probs * torch.log(long_probs + 1e-8), dim=-1) +
            has_short_book * torch.sum(
                short_probs * torch.log(short_probs + 1e-8), dim=-1))
        gross = float(np.abs(weights).sum())
        absolute_shares = np.abs(weights) / max(gross, 1e-12)
        effective_positions = float(1.0 / np.square(absolute_shares).sum())
        return (weights, float(gate.item()), float(entropy.item()),
                float(entropy.item()) / MAX_BOOK_ENTROPY,
                effective_positions)

step_rows, sensitivity_rows = [], []
models = ['equal_weight', 'pretrained', 'full']
for model in models:
    actor = actors.get(model)
    for episode in range(1, N_EPISODES + 1):
        state = np.asarray(r.sanity_reset_episode(episode), dtype=np.float32)
        if state.shape != (SEQ_LEN, OBS_DIM) or not np.isfinite(state).all():
            raise RuntimeError(f'Invalid initial state for episode {episode}: {state.shape}')
        for step in range(1, EPISODE_LENGTH + 1):
            if actor is None:
                action = np.full(ACTION_DIM, NET_EXPOSURE / ACTION_DIM, dtype=np.float32)
                leverage_gate = direction_entropy = normalized_direction_entropy = math.nan
                effective_positions = float(ACTION_DIM)
            else:
                (action, leverage_gate, direction_entropy,
                 normalized_direction_entropy, effective_positions) = actor_action(
                    actor, state, include_diagnostics=True)
                perturbations = {}
                reversed_state = state[::-1].copy()
                perturbations['reverse_time'] = reversed_state
                zero_vine = state.copy(); zero_vine[:, BASE_OBS_DIM:] = 0
                perturbations['zero_vine'] = zero_vine
                zero_cvar = state.copy(); zero_cvar[:, 1 + 2 * ACTION_DIM] = 0
                perturbations['zero_cvar_observation'] = zero_cvar
                zero_market = state.copy(); zero_market[:, 1:1 + 2 * ACTION_DIM] = 0
                perturbations['zero_returns_volatility'] = zero_market
                neutral_holdings = state.copy()
                holdings_start = 2 * ACTION_DIM + 2
                neutral_holdings[:, holdings_start:holdings_start + ACTION_DIM] = NET_EXPOSURE / ACTION_DIM
                neutral_holdings[:, holdings_start + ACTION_DIM] = abs(NET_EXPOSURE)
                neutral_holdings[:, holdings_start + ACTION_DIM + 1] = NET_EXPOSURE
                perturbations['neutral_holdings'] = neutral_holdings
                for perturbation, modified in perturbations.items():
                    (changed_action, changed_gate, changed_entropy,
                     _, _) = actor_action(
                        actor, modified, include_diagnostics=True)
                    sensitivity_rows.append({
                        'model': model, 'episode': episode, 'step': step,
                        'perturbation': perturbation,
                        'action_l1_change': float(np.abs(action - changed_action).sum()),
                        'leverage_gate_abs_change': abs(leverage_gate - changed_gate),
                        'direction_entropy_abs_change': abs(
                            direction_entropy - changed_entropy)})
            result = r.sanity_step(action.tolist())
            info = dict(result['info'])
            realised_weights = np.asarray(info['weights'], dtype=float)
            asset_gross_returns = np.asarray(info['asset_gross_returns'], dtype=float)
            effective_leverage = ((float(info['gross_exposure']) - abs(NET_EXPOSURE)) /
                                  max(GROSS_LEVERAGE - abs(NET_EXPOSURE), 1e-12))
            gate_gross_error = (abs(leverage_gate - effective_leverage)
                                if np.isfinite(leverage_gate) else math.nan)
            position_at_cap = bool(np.any(
                (realised_weights >= MAX_LONG_WEIGHT - 1e-4) |
                (realised_weights <= -MAX_SHORT_WEIGHT + 1e-4)))
            row = {
                'model': model, 'episode': episode, 'step': step,
                'wealth': float(info['wealth']),
                'gross_return': float(info['portfolio_gross']) - 1.0,
                'net_return': float(info['net_portfolio_gross']) - 1.0,
                'reward': float(result['reward']), 'utility': float(info['utility']),
                'cvar': float(info['cvar']), 'turnover': float(info['turnover']),
                'transaction_cost': float(info['transaction_cost']),
                'financing_cost': float(info['financing_cost']),
                'short_notional': float(info['short_notional']),
                'gross_exposure': float(info['gross_exposure']),
                'net_exposure': float(info['net_exposure']),
                'leverage_gate': leverage_gate,
                'effective_leverage': effective_leverage,
                'gate_gross_error': gate_gross_error,
                'position_at_cap': position_at_cap,
                'direction_entropy': direction_entropy,
                'normalized_direction_entropy': normalized_direction_entropy,
                'effective_positions': effective_positions,
                # Rolling historical episodes overlap.  This key identifies
                # their shared realised calendar decision for de-duplicated
                # ex-post alignment diagnostics.
                'calendar_index': episode + step - 1}
            row.update({f'w_{index + 1}': float(weight)
                        for index, weight in enumerate(realised_weights)})
            row.update({f'r_{index + 1}': float(asset_return)
                        for index, asset_return in enumerate(asset_gross_returns)})
            step_rows.append(row)
            observation = np.asarray(result['observation'], dtype=np.float32)
            state = np.roll(state, -1, axis=0); state[-1] = observation

steps = pd.DataFrame(step_rows)
sensitivity = pd.DataFrame(sensitivity_rows)
weight_columns = [f'w_{index + 1}' for index in range(ACTION_DIM)]
return_columns = [f'r_{index + 1}' for index in range(ACTION_DIM)]

summary_rows = []
for model, frame in steps.groupby('model', sort=False):
    terminals = frame.sort_values(['episode', 'step']).groupby('episode').tail(1)
    drawdowns = []
    for _, episode_frame in frame.groupby('episode'):
        wealth = np.r_[100000.0, episode_frame.sort_values('step')['wealth'].to_numpy()]
        drawdowns.append(float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0)))
    # Equal weight has no actor, so its gate/entropy diagnostics are
    # intentionally undefined.  Exclude only those actor-only fields from the
    # finiteness contract; all realised portfolio and return fields remain
    # mandatory.  The previous omission of gate_gross_error made the otherwise
    # valid equal-weight control set overall_pass=False.
    actor_only_diagnostics = [
        'leverage_gate', 'gate_gross_error', 'direction_entropy',
        'normalized_direction_entropy']
    required_numeric = frame.drop(
        columns=actor_only_diagnostics).select_dtypes(
            include=[np.number]).to_numpy()
    all_finite = bool(np.isfinite(required_numeric).all())
    net_error = float(np.max(np.abs(frame['net_exposure'] - NET_EXPOSURE)))
    gross_violation = float(max(0.0, frame['gross_exposure'].max() - GROSS_LEVERAGE))
    weights_array = frame[weight_columns].to_numpy()
    long_weight_violation = float(max(0.0, weights_array.max() - MAX_LONG_WEIGHT))
    short_weight_violation = float(max(0.0, -MAX_SHORT_WEIGHT - weights_array.min()))
    # The rolling 24-month trajectories repeat the same realised calendar
    # decisions. Aggregate policy weights and count each outcome once for
    # ex-post alignment diagnostics instead of treating 1,464 rows as
    # independent observations.
    grouped = frame.groupby('calendar_index', sort=True)
    unique_weights = grouped[weight_columns].mean().to_numpy()
    unique_returns = grouped[return_columns].first().to_numpy()
    return_replication_error = float(max(
        (grouped[column].max() - grouped[column].min()).max()
        for column in return_columns))
    if model == 'equal_weight':
        top_hit_rate = bottom_hit_rate = weight_return_correlation = math.nan
    else:
        top_hit_rate = float(np.mean(
            np.argmax(unique_weights, axis=1) == np.argmax(unique_returns, axis=1)))
        bottom_hit_rate = float(np.mean(
            np.argmin(unique_weights, axis=1) == np.argmin(unique_returns, axis=1)))
        correlations = [np.corrcoef(weight, realised)[0, 1]
                        for weight, realised in zip(unique_weights, unique_returns)]
        weight_return_correlation = float(np.nanmean(correlations))
    summary_rows.append({
        'model': model, 'episodes': int(terminals.shape[0]),
        'steps': int(frame.shape[0]),
        'unique_calendar_decisions': int(unique_weights.shape[0]),
        'return_replication_error': return_replication_error,
        'all_values_finite': all_finite,
        'mean_terminal_wealth': float(terminals['wealth'].mean()),
        'median_terminal_wealth': float(terminals['wealth'].median()),
        'q05_terminal_wealth': float(terminals['wealth'].quantile(0.05)),
        'q95_terminal_wealth': float(terminals['wealth'].quantile(0.95)),
        'min_terminal_wealth': float(terminals['wealth'].min()),
        'max_terminal_wealth': float(terminals['wealth'].max()),
        'mean_episode_reward': float(frame.groupby('episode')['reward'].sum().mean()),
        'mean_max_drawdown': float(np.mean(drawdowns)),
        'worst_max_drawdown': float(np.min(drawdowns)),
        'mean_turnover': float(frame['turnover'].mean()),
        'median_turnover': float(frame['turnover'].median()),
        'q95_turnover': float(frame['turnover'].quantile(0.95)),
        'mean_cvar': float(frame['cvar'].mean()),
        'mean_gross_exposure': float(frame['gross_exposure'].mean()),
        'max_gross_exposure': float(frame['gross_exposure'].max()),
        'fraction_at_gross_cap': float((frame['gross_exposure'] >= GROSS_LEVERAGE - 1e-4).mean()),
        'max_net_exposure_error': net_error,
        'gross_exposure_violation': gross_violation,
        'long_weight_violation': long_weight_violation,
        'short_weight_violation': short_weight_violation,
        'fraction_steps_with_short': float((frame['short_notional'] > 1e-8).mean()),
        'mean_short_notional': float(frame['short_notional'].mean()),
        'mean_leverage_gate': float(frame['leverage_gate'].mean()),
        'q95_leverage_gate': float(frame['leverage_gate'].quantile(0.95)),
        'std_leverage_gate': float(frame['leverage_gate'].std()),
        'mean_effective_leverage': float(frame['effective_leverage'].mean()),
        'gate_gross_mae': float(frame['gate_gross_error'].mean()),
        'fraction_at_position_cap': float(frame['position_at_cap'].mean()),
        'mean_direction_entropy': float(frame['direction_entropy'].mean()),
        'mean_normalized_direction_entropy': float(
            frame['normalized_direction_entropy'].mean()),
        'q05_normalized_direction_entropy': float(
            frame['normalized_direction_entropy'].quantile(0.05)),
        'mean_effective_positions': float(frame['effective_positions'].mean()),
        'max_abs_weight': float(frame[weight_columns].abs().to_numpy().max()),
        'fraction_concentrated_weight': float((frame[weight_columns].abs().max(axis=1) > MAX_LONG_WEIGHT + 1e-6).mean()),
        'mean_cross_asset_weight_sd': float(frame[weight_columns].std(axis=1).mean()),
        'apparent_expost_top_hit_rate': top_hit_rate,
        'apparent_expost_bottom_hit_rate': bottom_hit_rate,
        'mean_cross_sectional_weight_next_return_correlation': weight_return_correlation,
        'hard_constraints_pass': bool(all_finite and net_error <= 1e-6 and
                                      gross_violation <= 1e-6 and
                                      long_weight_violation <= 1e-6 and
                                      short_weight_violation <= 1e-6 and
                                      return_replication_error <= 1e-10 and
                                      (frame['wealth'] > 0).all())})
summary = pd.DataFrame(summary_rows)

if not sensitivity.empty:
    sensitivity_summary = sensitivity.groupby(['model', 'perturbation'], as_index=False).agg(
        mean_action_l1_change=('action_l1_change', 'mean'),
        median_action_l1_change=('action_l1_change', 'median'),
        q95_action_l1_change=('action_l1_change', lambda x: x.quantile(0.95)),
        mean_leverage_gate_abs_change=('leverage_gate_abs_change', 'mean'),
        median_leverage_gate_abs_change=('leverage_gate_abs_change', 'median'),
        mean_direction_entropy_abs_change=('direction_entropy_abs_change', 'mean'))
else:
    sensitivity_summary = pd.DataFrame()

warnings = []
diagnostic_notes = []
full = summary.loc[summary['model'] == 'full'].iloc[0]
pretrained = summary.loc[summary['model'] == 'pretrained'].iloc[0]
equal_weight = summary.loc[summary['model'] == 'equal_weight'].iloc[0]
if not bool(full['hard_constraints_pass']): warnings.append('Full policy violates a numerical or portfolio constraint.')
if full['fraction_steps_with_short'] < 0.05: warnings.append('Full policy almost never takes an actual negative asset weight.')
if full['mean_cross_asset_weight_sd'] < 1e-3: warnings.append('Full policy is nearly equal-weight for all states.')
if full['fraction_at_gross_cap'] > 0.90:
    warnings.append('Full policy saturates the configured gross-leverage cap in more than 90% of training-path decisions.')
if full['mean_leverage_gate'] > 0.95:
    warnings.append('Full policy leverage gate averages above 0.95; dynamic leverage has collapsed to its upper boundary.')
if full['std_leverage_gate'] < 1e-3:
    warnings.append('Full policy leverage gate is effectively constant across training-prefix decisions.')
if full['gate_gross_mae'] > 1e-5:
    warnings.append('Full policy leverage gate does not match realised normalized gross exposure.')
if full['fraction_at_position_cap'] > WARN_POSITION_CAP_FRACTION:
    diagnostic_notes.append(
        'Full policy frequently approaches a valid individual position limit; '
        'reported diagnostically because cap equality is not a constraint violation.')
if full['median_turnover'] > 1.0:
    warnings.append('Full policy has median monthly turnover above 100% of portfolio notional.')
if full['fraction_concentrated_weight'] > 0.50:
    warnings.append('Full policy violates the configured single-asset position limit in over half of decisions.')
if full['mean_normalized_direction_entropy'] < MIN_MEAN_NORMALIZED_ENTROPY:
    warnings.append('Full policy mean normalized book entropy is below the preregistered diversification floor.')
if full['q05_normalized_direction_entropy'] < MIN_Q05_NORMALIZED_ENTROPY:
    warnings.append('Full policy lower-tail normalized book entropy is below the preregistered floor.')
if full['mean_effective_positions'] < MIN_MEAN_EFFECTIVE_POSITIONS:
    warnings.append('Full policy mean effective position count is below the preregistered floor.')
# In-sample performance is reported but must not become a tuning gate. The
# objective includes utility, CVaR, turnover and financing costs; raw terminal
# wealth is neither the complete objective nor evidence of generalisation.
if full['mean_episode_reward'] < pretrained['mean_episode_reward']:
    diagnostic_notes.append(
        'Full policy has lower mean in-sample objective reward than the pretrained policy.')
if full['mean_episode_reward'] < equal_weight['mean_episode_reward']:
    diagnostic_notes.append(
        'Full policy has lower mean in-sample objective reward than equal weight.')
if full['median_terminal_wealth'] > 2.0 * 100000.0:
    warnings.append('Very high in-sample wealth suggests memorisation; do not interpret this as generalisation.')
if full['apparent_expost_top_hit_rate'] > 2.0 / ACTION_DIM:
    warnings.append('Unique-decision in-sample long-asset selection is implausibly aligned with the ex-post best asset; treat as trajectory memorisation.')
if not sensitivity_summary.empty:
    full_sensitivity = sensitivity_summary[sensitivity_summary['model'] == 'full']
    if (full_sensitivity['mean_action_l1_change'] < 1e-4).all():
        warnings.append('Full policy is effectively insensitive to all tested state perturbations.')
    if (full_sensitivity['mean_leverage_gate_abs_change'] < 1e-4).all():
        warnings.append('Full policy leverage gate is effectively insensitive to all tested state perturbations.')
    vine_row = full_sensitivity[full_sensitivity['perturbation'] == 'zero_vine']
    holdings_row = full_sensitivity[full_sensitivity['perturbation'] == 'neutral_holdings']
    if VINE_OBSERVATION_MODE == 'full':
        if not vine_row.empty and float(vine_row['median_action_l1_change'].iloc[0]) < 0.05:
            warnings.append('Typical full-policy allocation is nearly insensitive to the NN-vine state.')
    else:
        # This is the defining negative-control property of the matched
        # ablation, not a behavioural defect.  Any response would prove that
        # vine information leaked through another state channel.
        if vine_row.empty or float(vine_row['median_action_l1_change'].iloc[0]) > 1e-8:
            warnings.append('No-vine policy changes when the already-zero vine channel is perturbed.')
    if not holdings_row.empty and float(holdings_row['median_action_l1_change'].iloc[0]) < 0.05 and full['median_turnover'] > 1.0:
        warnings.append('Full policy largely ignores previous holdings despite excessive realised turnover.')

pd.DataFrame(integrity_rows).to_csv(os.path.join(OUTPUT_DIR, 'checkpoint_integrity.csv'), index=False)
steps.to_csv(os.path.join(OUTPUT_DIR, 'policy_steps.csv'), index=False)
summary.to_csv(os.path.join(OUTPUT_DIR, 'policy_summary.csv'), index=False)
sensitivity.to_csv(os.path.join(OUTPUT_DIR, 'state_sensitivity_steps.csv'), index=False)
sensitivity_summary.to_csv(os.path.join(OUTPUT_DIR, 'state_sensitivity_summary.csv'), index=False)
with open(os.path.join(OUTPUT_DIR, 'sanity_report.json'), 'w') as stream:
    json.dump({
        'protocol': 'training-prefix episodes only; final 24 months never loaded',
        'seed': SEED, 'episodes': N_EPISODES, 'episode_length': EPISODE_LENGTH,
        'obs_dim': OBS_DIM, 'action_dim': ACTION_DIM, 'vine_dim': VINE_DIM,
        'vine_observation_mode': VINE_OBSERVATION_MODE,
        'vine_feature_mode': VINE_FEATURE_MODE,
        'cvar_observation_mode': CVAR_OBSERVATION_MODE,
        'cvar_reward_mode': CVAR_REWARD_MODE,
        'pretrain_data_mode': PRETRAIN_DATA_MODE,
        'rl_algorithm': RL_ALGORITHM,
        'policy_encoder': POLICY_ENCODER,
        'run_finetune': RUN_FINETUNE,
        'no_vine_signal_mask': NO_VINE_SIGNAL_MASK,
        'warnings': warnings, 'diagnostic_notes': diagnostic_notes,
        'publication_behavior_pass': len(warnings) == 0,
        'overall_pass': bool(all(row['architecture_match'] and row['all_checkpoint_tensors_finite']
                                 for row in integrity_rows) and
                             summary['hard_constraints_pass'].all() and
                             len(warnings) == 0)}, stream, indent=2)

print(summary.to_string(index=False))
print('Checkpoint integrity:')
print(pd.DataFrame(integrity_rows)[['model', 'sha256', 'size_bytes',
      'architecture_match', 'all_checkpoint_tensors_finite', 'update_count']].to_string(index=False))
if warnings:
    print('Sanity warnings:')
    for warning in warnings: print(' - ' + warning)
else:
    print('No hard or behavioural sanity warnings were triggered.')
if diagnostic_notes:
    print('Non-gating diagnostic notes:')
    for note in diagnostic_notes: print(' - ' + note)
print('No-holdout diagnostics written to ' + OUTPUT_DIR)
")

cat(sprintf(
  "\nNo-holdout sanity check complete: %d training-prefix trajectories, %d steps each.\n",
  length(finetune_returns), episode_length))
cat(sprintf("Outputs: %s\n", file.path(model_dir, "sanity_no_holdout")))
