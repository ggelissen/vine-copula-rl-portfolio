# ============================================================================
# evaluate_rl.r
# Evaluate trained RL agent against benchmarks
# ============================================================================

library(reticulate)
library(xts)

source("rl/rl_environment.r")
source("helper/load_data.r")
source("helper/plotting.r")

load("data/marginal_results.RData")
load("data/vine_fit.RData")
returns <- load_returns()


# Benchmark Results From benchmarks.r
if (file.exists("data/benchmark_results.RData")) {
  cat("Loading pre-computed benchmark results...\n")
  load("data/benchmark_results.RData")

  benchmark_results <- results
} else {
  cat("Benchmark results not found. Running benchmarks...\n")
  source("benchmarks.r")
  benchmark_results <- run_all_benchmarks(
    returns_xts  = returns,
    U            = U,
    marginals    = marginals,
    asset_names  = asset_names,
    rebal_dates  = tail(rebal_dates, 36),
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
L <- 500
all_dates <- index(returns)
rebal_idx <- endpoints(returns, on = "months")
rebal_idx <- rebal_idx[rebal_idx >= (L + 1)]
rebal_dates <- index(returns)[rebal_idx]
eval_dates <- tail(rebal_dates, 30)

vine_seq_eval <- build_vine_sequence(returns, U, eval_dates, L = L)


# Setup R Environment
env_eval <- RLEnvironment$new(
  marginals, asset_names,
  vine = NULL,
  vine_sequence = vine_seq_eval,
  dynamic = TRUE,
  ref_col = 7,
  gamma = 2,
  lambda = 1.0,
  kappa = 0.05,
  T = 36,                   
  w0 = 100000,
  n_sim_cvar = 10000,
  seq_len = 30
)

# Expose methods to Python
r_env_reset <- function() env_eval$reset()
r_env_step  <- function(action) env_eval$step(action)
r_env_get_action_dim <- function() env_eval$get_action_dim()
r_env_get_obs_dim   <- function() env_eval$get_obs_dim()
r_env_get_seq_len   <- function() env_eval$get_seq_len()
r_env_get_history   <- function() env_eval$get_history()


# Python Code: Load Full Model and Evaluate
# Note: The Python code here includes the LSTM Actor, Critic, DDPGAgent classes
# from train_rl.r. For brevity, I'm showing only the evaluation-specific code.
# In practice, you should source the class definitions from a separate Python file.

py_run_string("
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── LSTM Actor (same as in train_rl.r) ──────────────────────────────────────
class LSTMActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()
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
            state_tensor = torch.FloatTensor(state_seq).unsqueeze(0)  # (1, seq_len, obs_dim)
            action, _ = self.actor(state_tensor)
        self.actor.train()
        action = action.squeeze(0).numpy()
        if action.ndim > 1:
            action = action[0]
        if noise_scale > 0:
            action += noise_scale * np.random.randn(self.action_dim)
        return np.clip(action, -0.5, 1.0)
    
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
        # Ensure history has correct shape
        if len(self.history) == 0:
            self.history = np.zeros((self.seq_len, self.obs_dim), dtype=np.float32)
        elif self.history.shape[0] != self.seq_len:
            # Pad or trim to seq_len
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
        # Update history
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = obs
        return self.history.copy(), reward, done, info
    
    def get_history(self):
        return np.array(self.get_history_fn(), dtype=np.float32)
    
    def get_wealth_path(self):
        return self.wealth_history

# ── Set up evaluation ──────────────────────────────────────────────────────
obs_dim = get_obs_dim()
action_dim = get_action_dim()
seq_len = get_seq_len()

agent = DDPGAgent(obs_dim, action_dim)
agent.load('data/ddpg_lstm_vine_full.pt')
print('Full model loaded.')

env = EvalEnv(reset_fn, step_fn, get_history_fn, action_dim, obs_dim, seq_len)
state = env.reset()

for t in range(36):
    action = agent.select_action(state, noise_scale=0.0)
    next_state, reward, done, info = env.step(action)
    state = next_state

wealth_rl = env.get_wealth_path()
print(f'RL Evaluation complete. Final wealth: {wealth_rl[-1]:.0f}')
")




# Compute RL Metrics (Using Same Function as Benchmarks)
wealth_rl <- py$wealth_rl
wealth_rl <- as.numeric(wealth_rl)

source("benchmarks.r")
rl_metrics <- compute_metrics(wealth_rl, T_horizon = 36, w0 = 100000)

rl_metrics_named <- c(
  final_wealth   = rl_metrics["final_wealth"],
  total_return   = rl_metrics["total_return"],
  annual_return  = rl_metrics["annual_return"],
  annual_vol     = rl_metrics["annual_vol"],
  sharpe_ratio   = rl_metrics["sharpe_ratio"],
  max_drawdown   = rl_metrics["max_drawdown"]
)

cat("\n========== RL MODEL PERFORMANCE ==========\n")
cat(sprintf("  Final wealth:   %.0f\n", rl_metrics["final_wealth"]))
cat(sprintf("  Total return:   %.2f%%\n", rl_metrics["total_return"]))
cat(sprintf("  Annual return:  %.2f%%\n", rl_metrics["annual_return"]))
cat(sprintf("  Annual vol:     %.2f%%\n", rl_metrics["annual_vol"]))
cat(sprintf("  Sharpe ratio:   %.3f\n", rl_metrics["sharpe_ratio"]))
cat(sprintf("  Max drawdown:   %.2f%%\n", rl_metrics["max_drawdown"]))




# Combine into Comparison Table
benchmark_df <- data.frame(
  Strategy = rownames(benchmark_metrics),
  benchmark_metrics,
  stringsAsFactors = FALSE
)

rl_row <- data.frame(
  Strategy = "RL (DDPG+LSTM+Vine)",
  final_wealth   = rl_metrics["final_wealth"],
  total_return   = rl_metrics["total_return"],
  annual_return  = rl_metrics["annual_return"],
  annual_vol     = rl_metrics["annual_vol"],
  sharpe_ratio   = rl_metrics["sharpe_ratio"],
  max_drawdown   = rl_metrics["max_drawdown"]
)

comparison_table <- rbind(benchmark_df, rl_row)



# Print Final Comparison Table (for Table 11 in your paper)
cat("\n" + "="*70 + "\n")
cat("TABLE 11: OUT-OF-SAMPLE PERFORMANCE COMPARISON\n")
cat("="*70 + "\n")
cat(sprintf("%-25s %12s %10s %10s %10s %10s %10s\n",
            "Strategy", "Final W.", "Return%", "Ann.Ret%", "Vol%", "Sharpe", "MaxDD%"))
cat("-"*70 + "\n")

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
cat("="*70 + "\n")



# Save Results
save(
  list = c("comparison_table", "wealth_rl", "rl_metrics", "benchmark_results"),
  file = "data/evaluation_results.RData"
)

cat("\n✓ Results saved to data/evaluation_results.RData\n")



# Plot Wealth Curves (All Models)
wealth_paths <- list(
  "Empirical MV" = benchmark_results$empirical$wealth,
  "DCC-GARCH" = benchmark_results$dcc$wealth,
  "Static Vine MV" = benchmark_results$static$wealth,
  "Rolling Vine MV" = benchmark_results$rolling$wealth,
  "NN Vine MV" = benchmark_results$nn_mv$wealth,
  "Myopic EU" = benchmark_results$eu_single$wealth,
  "Multi-period EU" = benchmark_results$eu_multi$wealth,
  "NN Vine EU" = benchmark_results$nn_eu$wealth,
  "RL (Ours)" = wealth_rl
)

dates <- seq(as.Date("2019-01-31"), by = "month", length.out = 37)
pdf("figures/wealth_curves_full_comparison.pdf", width = 12, height = 8)
par(mar = c(5, 5, 4, 2))

all_wealth <- unlist(wealth_paths)
ylim <- c(min(all_wealth) * 0.95, max(all_wealth) * 1.05)
colors <- c(
  "grey60", "grey50", "grey40", "grey30", "grey20",  # Benchmarks
  "blue", "blue3", "blue4",                          # EU models
  "red"                                              # RL (ours)
)
plot(dates, wealth_paths[[1]], type = "l", col = colors[1],
     ylim = ylim, lwd = 1.5,
     xlab = "Date", ylab = "Wealth ($)",
     main = "Out-of-Sample Cumulative Wealth Paths")

for (i in 2:length(wealth_paths)) {
  lines(dates, wealth_paths[[i]], col = colors[i], lwd = ifelse(i == length(wealth_paths), 3, 1.5))
}

legend_names <- names(wealth_paths)
legend_colors <- colors
legend_lwd <- c(rep(1.5, 8), 3)

legend("topleft", legend = legend_names, 
       col = legend_colors, lwd = legend_lwd,
       cex = 0.8, ncol = 2, bty = "n")

dev.off()

cat("\n✓ Wealth curve plot saved to figures/wealth_curves_full_comparison.pdf\n")




# Summary Statistics
cat("\n" + "="*70 + "\n")
cat("SUMMARY: KEY FINDINGS\n")
cat("="*70 + "\n")

best_sharpe_idx <- which.max(comparison_table$sharpe_ratio)
best_sharpe_model <- comparison_table$Strategy[best_sharpe_idx]
best_sharpe_value <- comparison_table$sharpe_ratio[best_sharpe_idx]

cat(sprintf("✓ Best Sharpe ratio: %.3f (%s)\n", best_sharpe_value, best_sharpe_model))

best_dd_idx <- which.min(comparison_table$max_drawdown)
best_dd_model <- comparison_table$Strategy[best_dd_idx]
best_dd_value <- comparison_table$max_drawdown[best_dd_idx]

cat(sprintf("✓ Lowest max drawdown: %.2f%% (%s)\n", best_dd_value, best_dd_model))

rl_sharpe <- comparison_table$sharpe_ratio[comparison_table$Strategy == "RL (DDPG+LSTM+Vine)"]
rl_rank <- sum(comparison_table$sharpe_ratio > rl_sharpe) + 1

cat(sprintf("✓ RL model ranks %d/%d in Sharpe ratio\n", rl_rank, nrow(comparison_table)))

cat("="*70 + "\n")

cat("\n✓ Evaluation complete!\n")