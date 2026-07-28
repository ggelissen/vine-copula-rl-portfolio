# ============================================================================
# evaluate_rl.r
# Evaluate trained RL agents (pre-trained and final) against benchmarks
# ============================================================================

library(reticulate)
library(xts)

source("rl/rl_environment.r")
source("helper/load_data.r")
source("helper/plotting.r")

load("data/marginal_results.RData")
load("data/vine_fit.RData")
returns <- load_returns()

# Helper: Print separator
print_sep <- function() {
  cat(paste0("\n", paste(rep("=", 60), collapse = ""), "\n"))
}


# Benchmark Results From benchmarks.r
if (file.exists("data/benchmark_results.RData")) {
  cat("Loading pre-computed benchmark results...\n")
  load("data/benchmark_results.RData")
  benchmark_results <- results
} else {
  cat("Benchmark results not found. Running benchmarks...\n")
  source("benchmarks.r")
  # Need to define rebal_dates first
  L <- 500
  all_dates <- index(returns)
  rebal_idx <- endpoints(returns, on = "months")
  rebal_idx <- rebal_idx[rebal_idx >= (L + 1)]
  rebal_dates <- index(returns)[rebal_idx + L - 1]
  rebal_dates <- tail(rebal_dates, 36)
  
  benchmark_results <- run_all_benchmarks(
    returns_xts  = returns,
    U            = U,
    marginals    = marginals,
    asset_names  = asset_names,
    rebal_dates  = rebal_dates,
    T_horizon    = 36,
    ref_col      = 7,
    L            = 500,
    w0           = 100000,
    gamma        = 2,
    n_sim        = 10000,
    save_plot    = "figures/wealth_curves_benchmarks.pdf"
  )
}
benchmark_metrics <- benchmark_results$metrics_table


# Define Evaluation Window
L <- 250
all_dates <- index(returns)
rebal_idx <- endpoints(returns, on = "months")
rebal_idx <- rebal_idx[rebal_idx >= (L + 1)]
rebal_dates <- index(returns)[rebal_idx]
eval_dates <- tail(rebal_dates, 24)  # 24 months for evaluation

vine_seq_eval <- build_vine_sequence(returns, U, eval_dates, L = L)


# Setup R Environment
env_eval <- RLEnvironment$new(
  marginals, asset_names,
  vine = NULL,
  vine_sequence = vine_seq_eval,
  ref_col = 7,
  gamma = 2,
  lambda = 0.1,
  kappa = 0.01,
  T = 24,                   
  w0 = 100000,
  n_sim_cvar = 10000,
  seq_len = 30
)

# Expose methods to Python via py$
py$r_env_reset <- function() env_eval$reset()
py$r_env_step <- function(action) env_eval$step(action)
py$r_env_get_action_dim <- function() env_eval$get_action_dim()
py$r_env_get_obs_dim <- function() env_eval$get_obs_dim()
py$r_env_get_seq_len <- function() env_eval$get_seq_len()
py$r_env_get_history <- function() env_eval$get_history()


# Python Code: Load and Evaluate Both Models
py_run_string("
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── LSTM Actor ──────────────────────────────────────────────────────────────
class LSTMActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, state_seq, hidden=None):
        out, hidden = self.lstm(state_seq, hidden)
        action = self.fc(out[:, -1, :])
        return action, hidden

# ── LSTM Critic ─────────────────────────────────────────────────────────────
class LSTMCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim + action_dim, hidden, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, state_seq, action_seq, hidden=None):
        if action_seq.dim() == 2:
            action_seq = action_seq.unsqueeze(1)
        x = torch.cat([state_seq, action_seq], dim=-1)
        out, hidden = self.lstm(x, hidden)
        q = self.fc(out[:, -1, :])
        return q, hidden

# ── DDPG Agent ──────────────────────────────────────────────────────────────
class DDPGAgent:
    def __init__(self, obs_dim, action_dim, hidden=128, num_layers=2,
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005):
        self.actor = LSTMActor(obs_dim, action_dim, hidden, num_layers)
        self.actor_target = LSTMActor(obs_dim, action_dim, hidden, num_layers)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = LSTMCritic(obs_dim, action_dim, hidden, num_layers)
        self.critic_target = LSTMCritic(obs_dim, action_dim, hidden, num_layers)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.tau = tau
        self.action_dim = int(action_dim)
        self.obs_dim = int(obs_dim)
    
    def select_action(self, state_seq, noise_scale=0.0):
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_seq).unsqueeze(0)
            action, _ = self.actor(state_tensor)
        self.actor.train()
        action = action.squeeze(0).numpy()
        if action.ndim > 1:
            action = action[0]
        if noise_scale > 0:
            action += noise_scale * np.random.randn(self.action_dim)
            action = np.clip(action, 0.0, 1.0)
            action = action / np.sum(action)  # Normalize to sum to 1
        return action
    
    def load(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])

# ── Evaluation Environment Wrapper ─────────────────────────────────────────
class EvalEnv:
    def __init__(self, reset_fn, step_fn, get_history_fn, action_dim, obs_dim, seq_len):
        self.reset_fn = reset_fn
        self.step_fn = step_fn
        self.get_history_fn = get_history_fn
        self.action_dim = int(action_dim)
        self.obs_dim = int(obs_dim)
        self.seq_len = int(seq_len)
        self.history = None
        self.wealth_history = []
    
    def reset(self):
        obs = self.reset_fn()
        self.history = self.get_history_fn()
        self.wealth_history = [100000]
        if len(self.history) == 0:
            self.history = np.zeros((self.seq_len, self.obs_dim), dtype=np.float32)
        elif self.history.shape[0] != self.seq_len:
            if self.history.shape[0] < self.seq_len:
                pad = np.zeros((self.seq_len - self.history.shape[0], self.obs_dim), dtype=np.float32)
                self.history = np.vstack([self.history, pad])
            else:
                self.history = self.history[-self.seq_len:]
        return np.array(self.history, dtype=np.float32)
    
    def step(self, action):
        res = self.step_fn(action.tolist())
        obs = np.array(res['observation'], dtype=np.float32)
        reward = float(res['reward'])
        done = bool(res['done'])
        info = dict(res['info'])
        self.wealth_history.append(info['wealth'])
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = obs
        return self.history.copy(), reward, done, info
    
    def get_history(self):
        return np.array(self.get_history_fn(), dtype=np.float32)
    
    def get_wealth_path(self):
        return self.wealth_history

def evaluate_model(agent_path, env, obs_dim, action_dim, seq_len, model_name):
    print(f'Evaluating {model_name}...')
    agent = DDPGAgent(obs_dim, action_dim)
    agent.load(agent_path)
    env.reset()
    state = env.history.copy()
    
    for t in range(24):
        action = agent.select_action(state, noise_scale=0.0)
        next_state, reward, done, info = env.step(action)
        state = next_state
    
    wealth_path = env.get_wealth_path()
    print(f'{model_name} complete. Final wealth: {wealth_path[-1]:.0f}')
    return wealth_path

# ── Set up evaluation ──────────────────────────────────────────────────────
# Get dimensions using the R functions exposed via py$
obs_dim = int(r_env_get_obs_dim())
action_dim = int(r_env_get_action_dim())
seq_len = int(r_env_get_seq_len())

# Create a single environment for evaluation
eval_env = EvalEnv(r_env_reset, r_env_step, r_env_get_history, action_dim, obs_dim, seq_len)

# Evaluate Pre-trained Model
wealth_pretrained = evaluate_model(
    'data/ddpg_lstm_vine_pretrained.pt',
    eval_env,
    obs_dim, action_dim, seq_len,
    'Pre-trained RL'
)

# Evaluate Final Model
wealth_final = evaluate_model(
    'data/ddpg_lstm_vine_full.pt',
    eval_env,
    obs_dim, action_dim, seq_len,
    'Final RL'
)
")

# Retrieve wealth paths
wealth_pretrained <- py$wealth_pretrained
wealth_pretrained <- as.numeric(wealth_pretrained)

wealth_final <- py$wealth_final
wealth_final <- as.numeric(wealth_final)

# Source compute_metrics function
source("benchmarks.r")

# Compute metrics for both models
rl_pretrained_metrics <- compute_metrics(wealth_pretrained, T_horizon = 24, w0 = 100000)
rl_final_metrics <- compute_metrics(wealth_final, T_horizon = 24, w0 = 100000)

cat("\n========== PRE-TRAINED RL MODEL PERFORMANCE ==========\n")
cat(sprintf("  Final wealth:   %.0f\n", rl_pretrained_metrics["final_wealth"]))
cat(sprintf("  Total return:   %.2f%%\n", rl_pretrained_metrics["total_return"]))
cat(sprintf("  Annual return:  %.2f%%\n", rl_pretrained_metrics["annual_return"]))
cat(sprintf("  Annual vol:     %.2f%%\n", rl_pretrained_metrics["annual_vol"]))
cat(sprintf("  Sharpe ratio:   %.3f\n", rl_pretrained_metrics["sharpe_ratio"]))
cat(sprintf("  Max drawdown:   %.2f%%\n", rl_pretrained_metrics["max_drawdown"]))

cat("\n========== FINAL RL MODEL PERFORMANCE ==========\n")
cat(sprintf("  Final wealth:   %.0f\n", rl_final_metrics["final_wealth"]))
cat(sprintf("  Total return:   %.2f%%\n", rl_final_metrics["total_return"]))
cat(sprintf("  Annual return:  %.2f%%\n", rl_final_metrics["annual_return"]))
cat(sprintf("  Annual vol:     %.2f%%\n", rl_final_metrics["annual_vol"]))
cat(sprintf("  Sharpe ratio:   %.3f\n", rl_final_metrics["sharpe_ratio"]))
cat(sprintf("  Max drawdown:   %.2f%%\n", rl_final_metrics["max_drawdown"]))


# Combine into Comparison Table
benchmark_df <- data.frame(
  Strategy = rownames(benchmark_metrics),
  benchmark_metrics,
  stringsAsFactors = FALSE
)

rl_pretrained_row <- data.frame(
  Strategy = "RL (Pre-trained)",
  final_wealth   = rl_pretrained_metrics["final_wealth"],
  total_return   = rl_pretrained_metrics["total_return"],
  annual_return  = rl_pretrained_metrics["annual_return"],
  annual_vol     = rl_pretrained_metrics["annual_vol"],
  sharpe_ratio   = rl_pretrained_metrics["sharpe_ratio"],
  max_drawdown   = rl_pretrained_metrics["max_drawdown"]
)

rl_final_row <- data.frame(
  Strategy = "RL (Full)",
  final_wealth   = rl_final_metrics["final_wealth"],
  total_return   = rl_final_metrics["total_return"],
  annual_return  = rl_final_metrics["annual_return"],
  annual_vol     = rl_final_metrics["annual_vol"],
  sharpe_ratio   = rl_final_metrics["sharpe_ratio"],
  max_drawdown   = rl_final_metrics["max_drawdown"]
)

comparison_table <- rbind(benchmark_df, rl_pretrained_row, rl_final_row)


# Print Final Comparison Table
print_sep()
cat("TABLE 11: OUT-OF-SAMPLE PERFORMANCE COMPARISON\n")
print_sep()
cat(sprintf("%-25s %12s %10s %10s %10s %10s %10s\n",
            "Strategy", "Final W.", "Return%", "Ann.Ret%", "Vol%", "Sharpe", "MaxDD%"))
print_sep()

for (i in 1:nrow(comparison_table)) {
  cat(sprintf("%-25s %12.0f %10.2f %10.2f %10.2f %10.3f %10.2f\n",
              comparison_table$Strategy[i],
              comparison_table$final_wealth[i],
              comparison_table$total_return[i],
              comparison_table$annual_return[i],
              comparison_table$annual_vol[i],
              comparison_table$sharpe_ratio[i],
              comparison_table$max_drawdown[i]))
}
print_sep()


# Save Results
save(
  list = c("comparison_table", "wealth_pretrained", "wealth_final", 
           "rl_pretrained_metrics", "rl_final_metrics", "benchmark_results"),
  file = "data/evaluation_results.RData"
)

cat("\n✓ Results saved to data/evaluation_results.RData\n")


# Plot Wealth Curves (All Models + Both RL)
wealth_paths <- list(
  "Empirical MV" = benchmark_results$empirical$wealth,
  "DCC-GARCH" = benchmark_results$dcc$wealth,
  "Static Vine MV" = benchmark_results$static$wealth,
  "Rolling Vine MV" = benchmark_results$rolling$wealth,
  "NN Vine MV" = benchmark_results$nn_mv$wealth,
  "Myopic EU" = benchmark_results$eu_single$wealth,
  "Multi-period EU" = benchmark_results$eu_multi$wealth,
  "NN Vine EU" = benchmark_results$nn_eu$wealth,
  "RL (Pre-trained)" = wealth_pretrained,
  "RL (Full)" = wealth_final
)

dates <- seq(as.Date("2019-01-31"), by = "month", length.out = 37)
pdf("figures/wealth_curves_full_comparison.pdf", width = 12, height = 8)
par(mar = c(5, 5, 4, 2))

all_wealth <- unlist(wealth_paths)
ylim <- c(min(all_wealth) * 0.95, max(all_wealth) * 1.05)

colors <- c(
  "grey70", "grey60", "grey50", "grey40", "grey30",  # Benchmarks
  "blue", "blue3", "blue4",                          # EU models
  "orange",                                         # Pre-trained RL
  "red"                                             # Full RL
)

plot(dates, wealth_paths[[1]], type = "l", col = colors[1],
     ylim = ylim, lwd = 1.5,
     xlab = "Date", ylab = "Wealth ($)",
     main = "Out-of-Sample Cumulative Wealth Paths")

for (i in 2:length(wealth_paths)) {
  lwd_val <- ifelse(i == length(wealth_paths), 3, 1.5)
  lines(dates, wealth_paths[[i]], col = colors[i], lwd = lwd_val)
}

legend_names <- names(wealth_paths)
legend_colors <- colors
legend_lwd <- c(rep(1.5, 8), 1.5, 3)

legend("topleft", legend = legend_names, 
       col = legend_colors, lwd = legend_lwd,
       cex = 0.7, ncol = 2, bty = "n")

dev.off()

cat("\n✓ Wealth curve plot saved to figures/wealth_curves_full_comparison.pdf\n")


# Summary Statistics
print_sep()
cat("SUMMARY: KEY FINDINGS\n")
print_sep()

# Check if RL models are in the table
rl_indices <- which(comparison_table$Strategy %in% c("RL (Pre-trained)", "RL (Full)"))
best_sharpe_idx <- which.max(comparison_table$sharpe_ratio)
best_sharpe_model <- comparison_table$Strategy[best_sharpe_idx]
best_sharpe_value <- comparison_table$sharpe_ratio[best_sharpe_idx]

cat(sprintf("✓ Best Sharpe ratio: %.3f (%s)\n", best_sharpe_value, best_sharpe_model))

best_dd_idx <- which.min(comparison_table$max_drawdown)
best_dd_model <- comparison_table$Strategy[best_dd_idx]
best_dd_value <- comparison_table$max_drawdown[best_dd_idx]

cat(sprintf("✓ Lowest max drawdown: %.2f%% (%s)\n", best_dd_value, best_dd_model))

# Ranking of RL models
for (rl_idx in rl_indices) {
  rl_sharpe <- comparison_table$sharpe_ratio[rl_idx]
  rl_name <- comparison_table$Strategy[rl_idx]
  rl_rank <- sum(comparison_table$sharpe_ratio > rl_sharpe) + 1
  cat(sprintf("✓ %s ranks %d/%d in Sharpe ratio\n", rl_name, rl_rank, nrow(comparison_table)))
}

print_sep()
cat("\n✓ Evaluation complete!\n")