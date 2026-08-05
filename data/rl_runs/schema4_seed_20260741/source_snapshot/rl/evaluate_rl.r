# Historical out-of-sample evaluation for the episodic vine-RL policy.
# The benchmark strategies and RL policy must be scored on the same realised
# monthly returns; stochastic simulated evaluation is a separate robustness
# experiment and must not be pooled with this backtest.

suppressPackageStartupMessages({ library(reticulate); library(xts); library(ggplot2); library(data.table) })
source("benchmark_models/dynamic_vine_NN.r")
source("rl/rl_environment.r")
source("helper/time_split.r")
source("helper/load_data.r")

eval_model_dir <- Sys.getenv("EVAL_MODEL_DIR")
eval_seed <- as.integer(Sys.getenv("EVAL_SEED"))
eval_gamma <- as.numeric(Sys.getenv("EVAL_GAMMA")); eval_lambda <- as.numeric(Sys.getenv("EVAL_LAMBDA")); eval_kappa <- as.numeric(Sys.getenv("EVAL_KAPPA"))
L <- as.integer(Sys.getenv("L")); ref_col <- as.integer(Sys.getenv("REF_COL")); n_sim_cvar <- as.integer(Sys.getenv("N_SIM_CVAR")); seq_len <- as.integer(Sys.getenv("ENV_SEQ_LEN"))
holding_days <- as.integer(Sys.getenv("ENV_HOLDING_DAYS", "21"))
gross_leverage <- as.numeric(Sys.getenv("ENV_GROSS_LEVERAGE", "1.5"))
net_exposure <- as.numeric(Sys.getenv("ENV_NET_EXPOSURE", "1"))
short_borrow_rate <- as.numeric(Sys.getenv("ENV_SHORT_BORROW_RATE", "0.03"))
cash_borrow_rate <- as.numeric(Sys.getenv("ENV_CASH_BORROW_RATE", "0.02"))
utility_mode <- Sys.getenv("ENV_UTILITY_MODE", "terminal_wealth_crra")
vine_model <- Sys.getenv("VINE_MODEL", "nn_dynamic_t_vine")
nn_vine_epochs <- as.integer(Sys.getenv("NN_VINE_EPOCHS", "200"))
nn_vine_lr <- as.numeric(Sys.getenv("NN_VINE_LR", "0.001"))
nn_vine_patience <- as.integer(Sys.getenv("NN_VINE_PATIENCE", "20"))
nn_vine_model_dir <- Sys.getenv("NN_VINE_MODEL_DIR", "data/nn_vine_models")
hidden <- as.integer(Sys.getenv("HIDDEN", "128")); num_layers <- as.integer(Sys.getenv("NUM_LAYERS", "2"))
benchmark_file <- Sys.getenv("BENCHMARK_RESULTS_FILE", "data/benchmark_results.RData")
training_marginals_file <- Sys.getenv("TRAINING_MARGINALS_FILE", "data/training_marginal_results.RData")
if (!nzchar(eval_model_dir) || any(is.na(c(eval_seed, eval_gamma, eval_lambda, eval_kappa, L, ref_col, n_sim_cvar, seq_len, holding_days, hidden, num_layers)))) stop("Missing evaluation configuration.")

set.seed(eval_seed)
if (!file.exists(training_marginals_file)) stop(sprintf("Training-only marginal file not found: %s\nRun rl/synthetic_returns.r and retrain first.", training_marginals_file))
load(training_marginals_file)
returns <- load_returns()
T_eval <- as.integer(Sys.getenv("EVALUATION_PERIODS", "24"))
if (T_eval != 24L) stop("The locked historical evaluation must contain exactly 24 monthly holding periods.")
period_split <- split_monthly_periods(
  build_monthly_periods(returns, min_history = L), T_eval
)
validate_period_split(period_split, T_eval)
eval_periods <- period_split$evaluation
eval_dates <- eval_periods$decision_date
train_periods <- period_split$train
train_end <- tail(period_split$train$decision_idx, 1L)
if (!identical(vine_model, "nn_dynamic_t_vine")) stop("Evaluation supports only the NN-driven dynamic t-vine; rolling vines are disabled.")
source("helper/marginals.r")
nn_fit <- load_nn_dynamic_vine_fit(nn_vine_model_dir)
if (!exists("copula_innovation_u") || !exists("copula_monthly_log") ||
    !exists("serial_copulas")) {
  stop("Training marginal artifact predates the monthly serial-PIT vine. Regenerate synthetic data and retrain.")
}
if (as.integer(nn_fit$training_observations) != nrow(copula_innovation_u) ||
    as.integer(nn_fit$dynamic_edge_count) != length(asset_names) * (length(asset_names) - 1L) / 2L) {
  stop("Persisted NN vine does not match the locked training split/all-tree architecture. Regenerate it.")
}
if (nrow(train_periods) < seq_len) stop("Not enough pre-evaluation periods for the LSTM burn-in.")
burnin_periods <- tail(train_periods, seq_len)
context_dates <- c(burnin_periods$decision_date, eval_dates)

# Extend the training-only serial-PIT innovation sequence with realised returns
# that are available before each OOS decision.  The empirical marginal maps and
# serial-copula parameters remain frozen at the training cutoff; no OOS return
# is used before its holding period has completed.
all_monthly_periods <- build_monthly_periods(returns, min_history = 0L)
all_monthly_log <- do.call(rbind, lapply(seq_len(nrow(all_monthly_periods)), function(i) {
  log(as.numeric(realised_gross_for_period(
    returns, all_monthly_periods$decision_date[i],
    all_monthly_periods$holding_end_date[i])))
}))
colnames(all_monthly_log) <- asset_names
training_sorted_for_pit <- lapply(seq_along(asset_names), function(j)
  sort(copula_monthly_log[, j]))
raw_monthly_u <- vapply(seq_along(asset_names), function(j) {
  sorted_log <- training_sorted_for_pit[[j]]
  probabilities <- (seq_len(length(sorted_log)) - 0.5) / length(sorted_log)
  pmin(pmax(approx(sorted_log, probabilities, xout = all_monthly_log[, j],
                   rule = 2, ties = "ordered")$y, 1e-6), 1 - 1e-6)
}, numeric(nrow(all_monthly_log)))
colnames(raw_monthly_u) <- asset_names
serial_conditional_pit <- function(previous_u, current_u, model) {
  rho <- model$rho; nu <- model$nu
  previous_t <- qt(pmin(pmax(previous_u, 1e-8), 1 - 1e-8), df = nu)
  current_t <- qt(pmin(pmax(current_u, 1e-8), 1 - 1e-8), df = nu)
  conditional_scale <- sqrt((nu + previous_t^2) * (1 - rho^2) / (nu + 1))
  pt((current_t - rho * previous_t) / conditional_scale, df = nu + 1)
}
all_innovation_u <- matrix(NA_real_, nrow(raw_monthly_u) - 1L,
                           ncol(raw_monthly_u), dimnames = list(NULL, asset_names))
for (j in seq_along(asset_names)) {
  all_innovation_u[, j] <- serial_conditional_pit(
    head(raw_monthly_u[, j], -1L), tail(raw_monthly_u[, j], -1L),
    serial_copulas[[j]])
}
all_innovation_u <- pmin(pmax(all_innovation_u, 1e-6), 1 - 1e-6)
all_innovation_dates <- all_monthly_periods$holding_end_date[-1L]
all_nn_states <- derive_nn_states(all_innovation_u)
vine_seq_context <- build_nn_vine_sequence(
  nn_fit, all_innovation_u, all_nn_states$z, all_nn_states$sigma,
  context_dates, all_innovation_dates)
vine_seq_burnin <- vine_seq_context[seq.int(1L, seq_len)]
vine_seq_eval <- vine_seq_context[seq.int(seq_len + 1L, seq_len + T_eval)]
if (length(vine_seq_eval) != T_eval) stop("Could not build every NN-vine evaluation snapshot.")

# Each row 1 is the historical realised monthly gross return; remaining rows
# are only for the ex-ante CVaR feature and penalty.  This exactly aligns the
# realised wealth series with the common research protocol. Scenario marginals
# use the same training-only monthly empirical transform as pre-training.
training_gross <- do.call(rbind, lapply(seq_len(nrow(train_periods)), function(i) {
  as.numeric(realised_gross_for_period(
    returns, train_periods$decision_date[i], train_periods$holding_end_date[i]
  ))
}))
colnames(training_gross) <- asset_names
training_log <- log(training_gross)
historical_log_sorted <- lapply(seq_along(asset_names), function(j) sort(training_log[, j]))
simulate_evaluation_scenarios <- function(vine, n_draws, previous_log_returns) {
  u <- rvinecop(n_draws, vine, cores = 1L)
  previous_u <- matrix(vapply(seq_along(asset_names), function(j) {
    sorted_log <- historical_log_sorted[[j]]
    probabilities <- (seq_len(length(sorted_log)) - 0.5) / length(sorted_log)
    approx(sorted_log, probabilities, xout = previous_log_returns[j],
           rule = 2, ties = "ordered")$y
  }, numeric(1)), nrow = 1L)
  previous_u <- previous_u[rep(1L, n_draws), , drop = FALSE]
  for (j in seq_along(asset_names)) {
    rho <- serial_copulas[[j]]$rho; nu <- serial_copulas[[j]]$nu
    previous_t <- qt(pmin(pmax(previous_u[, j], 1e-8), 1 - 1e-8), df = nu)
    conditional_t <- rho * previous_t +
      sqrt((nu + previous_t^2) * (1 - rho^2) / (nu + 1)) *
      qt(pmin(pmax(u[, j], 1e-8), 1 - 1e-8), df = nu + 1)
    u[, j] <- pt(conditional_t, df = nu)
  }
  out <- matrix(NA_real_, nrow = n_draws, ncol = length(asset_names))
  for (j in seq_along(asset_names)) {
    sorted_log <- historical_log_sorted[[j]]
    probabilities <- (seq_len(length(sorted_log)) - 0.5) / length(sorted_log)
    out[, j] <- exp(approx(probabilities, sorted_log, xout = u[, j], rule = 2)$y)
  }
  out
}
previous_returns <- training_log[nrow(training_log), ]
eval_steps <- vector("list", T_eval)
for (t in seq_len(T_eval)) {
  actual_gross <- realised_gross_for_period(
    returns, eval_periods$decision_date[t], eval_periods$holding_end_date[t]
  )
  scenarios <- simulate_evaluation_scenarios(
    vine_seq_eval[[t]], n_sim_cvar, previous_returns
  )
  eval_steps[[t]] <- rbind(as.numeric(actual_gross), scenarios)
  previous_returns <- log(pmax(actual_gross, 1e-12))
}

burnin_returns <- lapply(seq_len(nrow(burnin_periods)), function(i) {
  as.numeric(realised_gross_for_period(returns, burnin_periods$decision_date[i],
                                       burnin_periods$holding_end_date[i]))
})
evaluation_episode <- list(
  burnin_returns = burnin_returns,
  burnin_vine_states = lapply(vine_seq_burnin, extract_vine_state),
  returns = eval_steps, vine_states = lapply(vine_seq_eval, extract_vine_state),
  vine_start = 1L, source = "historical_oos")

env_eval <- RLEnvironment$new(marginals, asset_names, vine = NULL, vine_sequence = vine_seq_eval,
  ref_col = ref_col, gamma = eval_gamma, lambda = eval_lambda, kappa = eval_kappa,
  T = T_eval, w0 = 100000, n_sim_cvar = n_sim_cvar, seq_len = seq_len, sim_cores = 1L, holding_days = holding_days,
  gross_leverage = gross_leverage, net_exposure = net_exposure,
  max_long_weight = as.numeric(Sys.getenv("ENV_MAX_LONG_WEIGHT", "0.60")),
  max_short_weight = as.numeric(Sys.getenv("ENV_MAX_SHORT_WEIGHT", "0.20")),
  short_borrow_rate = short_borrow_rate, cash_borrow_rate = cash_borrow_rate,
  utility_mode = utility_mode)
env_eval$set_precomputed_returns(list(evaluation_episode))
r_env_reset <- function() env_eval$reset(); r_env_step <- function(action) env_eval$step(action)
r_env_get_action_dim <- function() as.integer(env_eval$get_action_dim()); r_env_get_obs_dim <- function() as.integer(env_eval$get_obs_dim()); r_env_get_seq_len <- function() as.integer(env_eval$get_seq_len()); r_env_get_history <- function() env_eval$get_history()

py_run_string("
import os, math, numpy as np, pandas as pd, torch, torch.nn as nn
OUTPUT_DIR = os.environ['EVAL_MODEL_DIR']
HIDDEN, NUM_LAYERS = int(os.environ.get('HIDDEN', '128')), int(os.environ.get('NUM_LAYERS', '2'))
GROSS_LEVERAGE = float(os.environ.get('ENV_GROSS_LEVERAGE', '1.5'))
NET_EXPOSURE = float(os.environ.get('ENV_NET_EXPOSURE', '1.0'))
MAX_LONG_WEIGHT = float(os.environ.get('ENV_MAX_LONG_WEIGHT', '0.60'))
MAX_SHORT_WEIGHT = float(os.environ.get('ENV_MAX_SHORT_WEIGHT', '0.20'))
SHORT_BORROW_RATE = float(os.environ.get('ENV_SHORT_BORROW_RATE', '0.03'))
CASH_BORROW_RATE = float(os.environ.get('ENV_CASH_BORROW_RATE', '0.02'))
UTILITY_MODE = os.environ.get('ENV_UTILITY_MODE', 'terminal_wealth_crra')
DIRECTION_LOGIT_BOUND = float(os.environ.get('DIRECTION_LOGIT_BOUND', '1.0'))
INITIAL_LEVERAGE_GATE = float(os.environ.get('INITIAL_LEVERAGE_GATE', '0.10'))
ENTROPY_COEF = float(os.environ.get('ENTROPY_COEF', '0.005'))
LEVERAGE_SOFT_TARGET = float(os.environ.get('LEVERAGE_SOFT_TARGET', '0.80'))
LEVERAGE_PENALTY_COEF = float(os.environ.get('LEVERAGE_PENALTY_COEF', '0.25'))
USE_AMP = os.environ.get('USE_AMP', 'false').lower() in ('1', 'true', 'yes')
if NET_EXPOSURE <= 0 or GROSS_LEVERAGE < NET_EXPOSURE:
    raise RuntimeError('Schema-4 rank-partition actions require positive net exposure and gross >= net.')
FULL_SHORT_BUDGET = 0.5 * (GROSS_LEVERAGE - NET_EXPOSURE)
SHORT_SUPPORT_SIZE = (int(math.ceil(FULL_SHORT_BUDGET / MAX_SHORT_WEIGHT - 1e-12))
                      if FULL_SHORT_BUDGET > 0 else 0)

class LSTMActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden, num_layers):
        super().__init__()
        self.input_norm = nn.LayerNorm(obs_dim)
        self.lstm = nn.LSTM(obs_dim, hidden, num_layers, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden)
        self.fc = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, action_dim + 1))
    def forward(self, state):
        x, _ = self.lstm(self.input_norm(state))
        return self.fc(self.layernorm(x[:, -1, :]))

def load_actor(path, obs_dim, action_dim):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    architecture = checkpoint.get('architecture')
    expected = {'obs_dim': obs_dim, 'action_dim': action_dim,
                'actor_output_dim': action_dim + 1, 'hidden': HIDDEN, 'num_layers': NUM_LAYERS,
                'agent': 'td3', 'state_normalization': 'layer_norm',
                'action_mode': 'rank_partition_leverage_gate_v4', 'gross_leverage': GROSS_LEVERAGE,
                'net_exposure': NET_EXPOSURE, 'short_borrow_rate': SHORT_BORROW_RATE,
                'cash_borrow_rate': CASH_BORROW_RATE, 'utility_mode': UTILITY_MODE,
                'max_long_weight': MAX_LONG_WEIGHT,
                'max_short_weight': MAX_SHORT_WEIGHT,
                'direction_logit_bound': DIRECTION_LOGIT_BOUND,
                'initial_leverage_gate': INITIAL_LEVERAGE_GATE,
                'allocation_entropy_coef': ENTROPY_COEF,
                'leverage_soft_target': LEVERAGE_SOFT_TARGET,
                'leverage_penalty_coef': LEVERAGE_PENALTY_COEF,
                'short_support_size': SHORT_SUPPORT_SIZE,
                'use_amp': USE_AMP,
                'checkpoint_schema': 4}
    if architecture is None:
        raise RuntimeError(f'{path} predates architecture metadata; retrain after the state/reward correction.')
    mismatches = {k: (architecture.get(k), v) for k, v in expected.items() if architecture.get(k) != v}
    if mismatches:
        raise RuntimeError(f'Checkpoint architecture does not match evaluation environment: {mismatches}. Retrain the model.')
    actor = LSTMActor(obs_dim, action_dim, HIDDEN, NUM_LAYERS)
    state = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['actor'].items()}
    actor.load_state_dict(state); actor.eval()
    return actor

def portfolio_weights(raw_action):
    action_dim = raw_action.shape[-1] - 1
    direction_logits = DIRECTION_LOGIT_BOUND * torch.tanh(
        raw_action[..., :action_dim] / DIRECTION_LOGIT_BOUND)
    leverage_gate = torch.sigmoid(raw_action[..., -1:])
    short_indices = torch.topk(
        direction_logits, k=SHORT_SUPPORT_SIZE, dim=-1, largest=False).indices
    short_mask = torch.zeros_like(direction_logits, dtype=torch.bool)
    short_mask.scatter_(-1, short_indices, True)
    long_logits = torch.where(
        ~short_mask, direction_logits, torch.full_like(direction_logits, -1e9))
    short_logits = torch.where(
        short_mask, -direction_logits, torch.full_like(direction_logits, -1e9))
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
    short_budget = FULL_SHORT_BUDGET * leverage_gate
    long_budget = NET_EXPOSURE + short_budget
    long_probs = capped_simplex(
        long_logits, torch.clamp(MAX_LONG_WEIGHT / long_budget.clamp_min(1e-12), max=1.0))
    short_probs = (capped_simplex(
        short_logits, torch.clamp(MAX_SHORT_WEIGHT / short_budget.clamp_min(1e-12), max=1.0))
        if SHORT_SUPPORT_SIZE > 0 else torch.zeros_like(long_probs))
    return long_budget * long_probs - short_budget * short_probs

class HistoricalEnv:
    def reset(self):
        r.r_env_reset(); self.state = np.asarray(r.r_env_get_history(), dtype=np.float32); self.logs = []; return self.state
    def step(self, action):
        result = r.r_env_step(action.tolist())
        obs = np.asarray(result['observation'], dtype=np.float32)
        self.state = np.roll(self.state, -1, axis=0); self.state[-1] = obs
        info = dict(result['info'])
        row = {'step': len(self.logs) + 1, 'wealth': info['wealth'],
               'gross_return': info['portf_ret'] - 1.0,
               'net_return': info['net_portf_ret'] - 1.0,
               'cvar': info['cvar'], 'turnover': info['turnover'],
               'transaction_cost': info['transaction_cost'],
               'financing_cost': info['financing_cost'],
               'utility': info['utility'], 'reward': float(result['reward'])}
        realised_weights = np.asarray(info['weights'], dtype=float)
        row.update({f'w{i+1}': float(w) for i, w in enumerate(realised_weights)})
        self.logs.append(row); return bool(result['done'])

obs_dim, action_dim = int(r.r_env_get_obs_dim()), int(r.r_env_get_action_dim())
def run(path):
    actor, env = load_actor(path, obs_dim, action_dim), HistoricalEnv(); state = env.reset()
    with torch.no_grad():
        for _ in range(24):
            action = portfolio_weights(actor(torch.from_numpy(state).unsqueeze(0))).squeeze(0).numpy()
            if env.step(action): break
    return pd.DataFrame(env.logs)

run_name = os.path.basename(OUTPUT_DIR)
logs_pretrained, logs_full = run(os.path.join(OUTPUT_DIR, 'td3_lstm_vine_pretrained.pt')), run(os.path.join(OUTPUT_DIR, 'td3_lstm_vine_full.pt'))
logs_pretrained['model'] = 'pretrained'; logs_full['model'] = 'full'
all_logs = pd.concat([logs_pretrained, logs_full], ignore_index=True)
all_logs.to_csv(f'data/evaluation_logs_{run_name}.csv', index=False)
")

all_logs <- as.data.frame(py$all_logs)
run_name <- basename(eval_model_dir)
source("eval/ablation.r")
all_logs$decision_date <- rep(eval_periods$decision_date, 2L)
all_logs$holding_end_date <- rep(eval_periods$holding_end_date, 2L)
for (j in seq_along(asset_names)) names(all_logs)[names(all_logs) == paste0("w", j)] <- paste0("w_", asset_names[j])
rl_rows <- lapply(c("pretrained", "full"), function(model) {
  returns_model <- all_logs$net_return[all_logs$model == model]
  data.frame(Strategy = paste("RL", model),
             t(annualised_path_metrics(returns_model)), check.names = FALSE)
})
rl_df <- rbindlist(rl_rows, fill = TRUE)
comparison_table <- rl_df
save(comparison_table, file = paste0("data/evaluation_comparison_", run_name, ".RData"))
write.csv(comparison_table, paste0("data/evaluation_comparison_", run_name, ".csv"), row.names = FALSE)
write.csv(all_logs, paste0("data/evaluation_logs_", run_name, ".csv"), row.names = FALSE)
for (model in c("pretrained", "full")) {
  weight_columns <- c("decision_date", paste0("w_", asset_names))
  write.csv(all_logs[all_logs$model == model, weight_columns, drop = FALSE],
            paste0("data/weights_rl_", model, "_", run_name, ".csv"), row.names = FALSE)
}

plot_df <- rbindlist(lapply(c("pretrained", "full"), function(model) data.table(step = 0:T_eval, wealth = c(100000, all_logs$wealth[all_logs$model == model]), model = model)))
ggsave(paste0("figures/wealth_curves_rl_evaluation_", run_name, ".pdf"), ggplot(plot_df, aes(step, wealth, colour = model)) + geom_line(linewidth = 1) + theme_bw() + labs(title = "Historical out-of-sample RL wealth", x = "Month", y = "Wealth"), width = 9, height = 5)
weight_cols <- grep("^w_", names(all_logs), value = TRUE)
weights_long <- melt(as.data.table(all_logs), id.vars = c("model", "step"), measure.vars = weight_cols, variable.name = "asset", value.name = "weight")
ggsave(paste0("figures/weights_evolution_", run_name, ".pdf"), ggplot(weights_long, aes(step, weight, colour = asset)) + geom_line() + facet_wrap(~model) + theme_bw(), width = 9, height = 5)
cat("Historical evaluation complete. Results saved for", run_name, "\n")
