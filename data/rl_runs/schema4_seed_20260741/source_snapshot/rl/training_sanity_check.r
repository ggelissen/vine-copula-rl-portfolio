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
initial_leverage_gate <- required_scalar(config$agent$initial_leverage_gate,
                                         "agent.initial_leverage_gate")
allocation_entropy_coef <- required_scalar(config$agent$entropy_coef,
                                            "agent.entropy_coef")
leverage_soft_target <- required_scalar(config$agent$leverage_soft_target,
                                        "agent.leverage_soft_target")
leverage_penalty_coef <- required_scalar(config$agent$leverage_penalty_coef,
                                         "agent.leverage_penalty_coef")
use_amp <- isTRUE(config$agent$use_amp)
expected_finetune_uses <- required_scalar(config$finetuning$episodes,
                                          "finetuning.episodes", "integer")

if (episode_length != 24L) stop("The sanity check expects the trained 24-step objective.")
if (!identical(utility_mode, "terminal_wealth_crra")) {
  stop("The sanity check supports only terminal_wealth_crra checkpoints.")
}
if (gross_leverage < abs(net_exposure)) stop("Gross leverage is below absolute net exposure.")
if (net_exposure <= 0) stop("Schema-4 rank-partition actions require positive net exposure.")
if (!file.exists(finetune_file)) {
  stop(sprintf("Fine-tuning episodes not found: %s\nRun this on the training machine after synthetic_returns.r.",
               finetune_file))
}

checkpoint_paths <- file.path(model_dir, c(
  pretrained = "td3_lstm_vine_pretrained.pt",
  full = "td3_lstm_vine_full.pt"))
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
  observation <- c(
    sanity_state$wealth / w0,
    sanity_state$last_returns * 100,
    sanity_state$last_vols * 100,
    sanity_state$last_cvar * 100,
    sanity_state$previous_action,
    sum(abs(sanity_state$previous_action)),
    sum(sanity_state$previous_action),
    sanity_state$vine_state)
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
  financing_cost <- (short_borrow_rate * short_notional +
                     cash_borrow_rate * cash_borrow_notional) / 12
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
  reward <- utility_increment - lambda * cvar

  sanity_state$previous_action <- weights
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
  SANITY_INITIAL_LEVERAGE_GATE = as.character(initial_leverage_gate),
  SANITY_ALLOCATION_ENTROPY_COEF = as.character(allocation_entropy_coef),
  SANITY_LEVERAGE_SOFT_TARGET = as.character(leverage_soft_target),
  SANITY_LEVERAGE_PENALTY_COEF = as.character(leverage_penalty_coef),
  SANITY_USE_AMP = tolower(as.character(use_amp)),
  SANITY_EPISODES = as.character(length(finetune_returns)),
  SANITY_EPISODE_LENGTH = as.character(episode_length),
  SANITY_SEQ_LEN = as.character(seq_len),
  SANITY_BASE_OBS_DIM = as.character(obs_dim - vine_dim)
)

if ("torch" %in% loadedNamespaces()) {
  stop("R torch/Lantern is loaded. Run this in a fresh Rscript process so Python PyTorch is isolated.")
}

py_run_string("
import os, json, hashlib, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
INITIAL_LEVERAGE_GATE = float(os.environ['SANITY_INITIAL_LEVERAGE_GATE'])
ALLOCATION_ENTROPY_COEF = float(os.environ['SANITY_ALLOCATION_ENTROPY_COEF'])
LEVERAGE_SOFT_TARGET = float(os.environ['SANITY_LEVERAGE_SOFT_TARGET'])
LEVERAGE_PENALTY_COEF = float(os.environ['SANITY_LEVERAGE_PENALTY_COEF'])
USE_AMP = os.environ['SANITY_USE_AMP'].lower() in ('1', 'true', 'yes')
N_EPISODES = int(os.environ['SANITY_EPISODES'])
EPISODE_LENGTH = int(os.environ['SANITY_EPISODE_LENGTH'])
SEQ_LEN = int(os.environ['SANITY_SEQ_LEN'])
FULL_SHORT_BUDGET = 0.5 * (GROSS_LEVERAGE - NET_EXPOSURE)
SHORT_SUPPORT_SIZE = (int(math.ceil(FULL_SHORT_BUDGET / MAX_SHORT_WEIGHT - 1e-12))
                      if FULL_SHORT_BUDGET > 0 else 0)
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

def capped_simplex(logits, cap):
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    lower = torch.full_like(shifted[..., :1], -50.0)
    upper = torch.full_like(shifted[..., :1], 50.0)
    for _ in range(30):
        midpoint = 0.5 * (lower + upper)
        candidate = torch.minimum(torch.exp(shifted - midpoint), cap)
        too_large = candidate.sum(dim=-1, keepdim=True) > 1.0
        lower = torch.where(too_large, midpoint, lower)
        upper = torch.where(too_large, upper, midpoint)
    probabilities = torch.minimum(
        torch.exp(shifted - 0.5 * (lower + upper)), cap)
    return probabilities / probabilities.sum(
        dim=-1, keepdim=True).clamp_min(1e-12)

def portfolio_books(raw_action):
    direction_logits = DIRECTION_LOGIT_BOUND * torch.tanh(
        raw_action[..., :ACTION_DIM] / DIRECTION_LOGIT_BOUND)
    leverage_gate = torch.sigmoid(raw_action[..., -1:])
    short_indices = torch.topk(
        direction_logits, k=SHORT_SUPPORT_SIZE, dim=-1, largest=False).indices
    short_mask = torch.zeros_like(direction_logits, dtype=torch.bool)
    short_mask.scatter_(-1, short_indices, True)
    long_logits = torch.where(
        ~short_mask, direction_logits, torch.full_like(direction_logits, -1e9))
    short_logits = torch.where(
        short_mask, -direction_logits, torch.full_like(direction_logits, -1e9))
    short_budget = FULL_SHORT_BUDGET * leverage_gate
    long_budget = NET_EXPOSURE + short_budget
    long_probs = capped_simplex(
        long_logits, torch.clamp(MAX_LONG_WEIGHT / long_budget.clamp_min(1e-12), max=1.0))
    short_probs = (capped_simplex(
        short_logits, torch.clamp(MAX_SHORT_WEIGHT / short_budget.clamp_min(1e-12), max=1.0))
        if SHORT_SUPPORT_SIZE > 0 else torch.zeros_like(long_probs))
    return long_probs, short_probs, leverage_gate, long_budget, short_budget

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
    architecture = checkpoint.get('architecture', {})
    expected = {
        'obs_dim': OBS_DIM, 'action_dim': ACTION_DIM,
        'actor_output_dim': ACTION_DIM + 1, 'hidden': HIDDEN,
        'num_layers': NUM_LAYERS, 'agent': 'td3',
        'state_normalization': 'layer_norm',
        'action_mode': 'rank_partition_leverage_gate_v4',
        'gross_leverage': GROSS_LEVERAGE, 'net_exposure': NET_EXPOSURE,
        'short_borrow_rate': SHORT_BORROW_RATE,
        'cash_borrow_rate': CASH_BORROW_RATE, 'utility_mode': UTILITY_MODE,
        'max_long_weight': MAX_LONG_WEIGHT,
        'max_short_weight': MAX_SHORT_WEIGHT,
        'direction_logit_bound': DIRECTION_LOGIT_BOUND,
        'initial_leverage_gate': INITIAL_LEVERAGE_GATE,
        'allocation_entropy_coef': ALLOCATION_ENTROPY_COEF,
        'leverage_soft_target': LEVERAGE_SOFT_TARGET,
        'leverage_penalty_coef': LEVERAGE_PENALTY_COEF,
        'short_support_size': SHORT_SUPPORT_SIZE,
        'use_amp': USE_AMP,
        'checkpoint_schema': 4}
    mismatches = {key: [architecture.get(key), value] for key, value in expected.items()
                  if architecture.get(key) != value}
    tensors = tensor_audit(checkpoint)
    all_finite = bool(tensors) and all(record[2] for record in tensors)
    actor_state = {key.replace('_orig_mod.', ''): value
                   for key, value in checkpoint['actor'].items()}
    actor = LSTMActor(); actor.load_state_dict(actor_state); actor.eval()
    target_state = {key.replace('_orig_mod.', ''): value
                    for key, value in checkpoint.get('actor_target', checkpoint['actor']).items()}
    target_gap = max(float((actor_state[key] - target_state[key]).abs().max().item())
                     for key in actor_state)
    integrity = {
        'model': label, 'path': path, 'sha256': sha256(path),
        'size_bytes': os.path.getsize(path), 'architecture_match': not mismatches,
        'architecture_mismatches': json.dumps(mismatches, sort_keys=True),
        'all_checkpoint_tensors_finite': all_finite,
        'tensor_count': len(tensors),
        'tensor_elements': sum(record[1] for record in tensors),
        'max_tensor_abs': max(record[3] for record in tensors) if tensors else math.nan,
        'actor_parameters': sum(parameter.numel() for parameter in actor.parameters()),
        'update_count': int(checkpoint.get('update_count', -1)),
        'actor_target_max_abs_gap': target_gap,
        'pytorch_version': torch.__version__}
    if mismatches or not all_finite:
        raise RuntimeError(f'Checkpoint {label} failed integrity checks: {integrity}')
    return actor, integrity

actors, integrity_rows = {}, []
for label, filename in [('pretrained', 'td3_lstm_vine_pretrained.pt'),
                        ('full', 'td3_lstm_vine_full.pt')]:
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
        return weights, float(gate.item()), float(entropy.item())

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
                leverage_gate, direction_entropy = math.nan, math.nan
            else:
                action, leverage_gate, direction_entropy = actor_action(
                    actor, state, include_diagnostics=True)
                perturbations = {}
                reversed_state = state[::-1].copy()
                perturbations['reverse_time'] = reversed_state
                zero_vine = state.copy(); zero_vine[:, BASE_OBS_DIM:] = 0
                perturbations['zero_vine'] = zero_vine
                zero_market = state.copy(); zero_market[:, 1:1 + 2 * ACTION_DIM] = 0
                perturbations['zero_returns_volatility'] = zero_market
                neutral_holdings = state.copy()
                holdings_start = 2 * ACTION_DIM + 2
                neutral_holdings[:, holdings_start:holdings_start + ACTION_DIM] = NET_EXPOSURE / ACTION_DIM
                neutral_holdings[:, holdings_start + ACTION_DIM] = abs(NET_EXPOSURE)
                neutral_holdings[:, holdings_start + ACTION_DIM + 1] = NET_EXPOSURE
                perturbations['neutral_holdings'] = neutral_holdings
                for perturbation, modified in perturbations.items():
                    changed_action = actor_action(actor, modified)
                    sensitivity_rows.append({
                        'model': model, 'episode': episode, 'step': step,
                        'perturbation': perturbation,
                        'action_l1_change': float(np.abs(action - changed_action).sum())})
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
                'direction_entropy': direction_entropy}
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
    required_numeric = frame.drop(
        columns=['leverage_gate', 'direction_entropy']).select_dtypes(
            include=[np.number]).to_numpy()
    all_finite = bool(np.isfinite(required_numeric).all())
    net_error = float(np.max(np.abs(frame['net_exposure'] - NET_EXPOSURE)))
    gross_violation = float(max(0.0, frame['gross_exposure'].max() - GROSS_LEVERAGE))
    weights_array = frame[weight_columns].to_numpy()
    long_weight_violation = float(max(0.0, weights_array.max() - MAX_LONG_WEIGHT))
    short_weight_violation = float(max(0.0, -MAX_SHORT_WEIGHT - weights_array.min()))
    returns_array = frame[return_columns].to_numpy()
    if model == 'equal_weight':
        top_hit_rate = bottom_hit_rate = weight_return_correlation = math.nan
    else:
        top_hit_rate = float(np.mean(np.argmax(weights_array, axis=1) == np.argmax(returns_array, axis=1)))
        bottom_hit_rate = float(np.mean(np.argmin(weights_array, axis=1) == np.argmin(returns_array, axis=1)))
        correlations = [np.corrcoef(weight, realised)[0, 1]
                        for weight, realised in zip(weights_array, returns_array)]
        weight_return_correlation = float(np.nanmean(correlations))
    summary_rows.append({
        'model': model, 'episodes': int(terminals.shape[0]),
        'steps': int(frame.shape[0]), 'all_values_finite': all_finite,
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
                                      (frame['wealth'] > 0).all())})
summary = pd.DataFrame(summary_rows)

if not sensitivity.empty:
    sensitivity_summary = sensitivity.groupby(['model', 'perturbation'], as_index=False).agg(
        mean_action_l1_change=('action_l1_change', 'mean'),
        median_action_l1_change=('action_l1_change', 'median'),
        q95_action_l1_change=('action_l1_change', lambda x: x.quantile(0.95)))
else:
    sensitivity_summary = pd.DataFrame()

warnings = []
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
if full['gate_gross_mae'] > 1e-5:
    warnings.append('Full policy leverage gate does not match realised normalized gross exposure.')
if full['fraction_at_position_cap'] > 0.75:
    warnings.append('Full policy binds an individual position limit in over 75% of decisions.')
if full['median_turnover'] > 1.0:
    warnings.append('Full policy has median monthly turnover above 100% of portfolio notional.')
if full['fraction_concentrated_weight'] > 0.50:
    warnings.append('Full policy violates the configured single-asset position limit in over half of decisions.')
if full['mean_terminal_wealth'] < pretrained['mean_terminal_wealth']:
    warnings.append('Historical fine-tuning reduced in-sample mean terminal wealth relative to pre-training.')
if full['mean_terminal_wealth'] < equal_weight['mean_terminal_wealth']:
    warnings.append('Full policy trails equal weight even on the trajectories used for fine-tuning.')
if full['median_terminal_wealth'] > 2.0 * 100000.0:
    warnings.append('Very high in-sample wealth suggests memorisation; do not interpret this as generalisation.')
if full['apparent_expost_top_hit_rate'] > 2.0 / ACTION_DIM:
    warnings.append('In-sample long-asset selection is implausibly aligned with the ex-post best asset; treat as trajectory memorisation.')
if not sensitivity_summary.empty:
    full_sensitivity = sensitivity_summary[sensitivity_summary['model'] == 'full']
    if (full_sensitivity['mean_action_l1_change'] < 1e-4).all():
        warnings.append('Full policy is effectively insensitive to all tested state perturbations.')
    vine_row = full_sensitivity[full_sensitivity['perturbation'] == 'zero_vine']
    holdings_row = full_sensitivity[full_sensitivity['perturbation'] == 'neutral_holdings']
    if not vine_row.empty and float(vine_row['median_action_l1_change'].iloc[0]) < 0.05:
        warnings.append('Typical full-policy allocation is nearly insensitive to the NN-vine state.')
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
        'warnings': warnings,
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
print('No-holdout diagnostics written to ' + OUTPUT_DIR)
")

cat(sprintf(
  "\nNo-holdout sanity check complete: %d training-prefix trajectories, %d steps each.\n",
  length(finetune_returns), episode_length))
cat(sprintf("Outputs: %s\n", file.path(model_dir, "sanity_no_holdout")))
