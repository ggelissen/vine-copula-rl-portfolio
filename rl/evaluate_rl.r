# ============================================================================
# evaluate_rl.r
# Evaluate trained RL agent against benchmarks
# ============================================================================

library(reticulate)

source("rl/rl_environment.r")
source("benchmark_models/expected_utility_single.r")
load("data/marginal_results.RData")
load("data/vine_fit.RData")
returns <- load_returns()

# Rebuild the same vine sequence used for training
L <- 500
all_dates <- index(returns)
rebal_idx <- endpoints(returns, on = "months")
rebal_idx <- rebal_idx[rebal_idx >= (L + 1)]
rebal_dates <- index(returns)[rebal_idx]
vine_seq <- build_vine_sequence(returns, U, rebal_dates, L = L)

# Take the bull window (last 30 months) for evaluation
eval_dates <- tail(rebal_dates, 30)
# Subset vine_seq to match (last 30 entries)
vine_seq_eval <- tail(vine_seq, 30)

# Create evaluation environment
env_eval <- RLEnvironment$new(
  marginals, asset_names,
  vine = NULL, vine_sequence = vine_seq_eval, dynamic = TRUE,
  ref_col = 7, gamma = 2, T = 30, w0 = 100000
)

# Expose methods
r_env_reset <- function() env_eval$reset()
r_env_step  <- function(action) env_eval$step(action)
r_env_get_action_dim <- function() env_eval$get_action_dim()
r_env_get_obs_dim   <- function() env_eval$get_obs_dim()

# Load the DDPG agent class definition (same py_run_string as training, without training loop)
py_run_string("
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class LSTMActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(int(obs_dim), hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, int(action_dim)),
            nn.Tanh()
        )
    def forward(self, state_seq, hidden=None):
        out, hidden = self.lstm(state_seq, hidden)
        action = self.fc(out[:, -1, :])
        return action, hidden

class LSTMCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(int(obs_dim + action_dim), hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, state_seq, action_seq, hidden=None):
        x = torch.cat([state_seq, action_seq], dim=-1)
        out, hidden = self.lstm(x, hidden)
        q = self.fc(out[:, -1, :])
        return q, hidden

class DDPGAgent:
    def __init__(self, obs_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3,
                 gamma=0.99, tau=0.005):
        self.actor = LSTMActor(obs_dim, action_dim)
        self.actor_target = LSTMActor(obs_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = LSTMCritic(obs_dim, action_dim)
        self.critic_target = LSTMCritic(obs_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.tau = tau
        self.action_dim = int(action_dim)

    def select_action(self, state_seq, noise_scale=0.0):
        self.actor.eval()
        with torch.no_grad():
            action, _ = self.actor(state_seq)
        self.actor.train()
        action = action.squeeze(0).numpy()
        if action.ndim > 1:
            action = action[0]
        action += noise_scale * np.random.randn(self.action_dim)
        return np.clip(action, -0.5, 1.0)

def load_agent(agent, path):
    ckpt = torch.load(path)
    agent.actor.load_state_dict(ckpt['actor'])
    agent.critic.load_state_dict(ckpt['critic'])
    return agent
")

# Load trained agent
obs_dim <- as.integer(r_env_get_obs_dim())
act_dim <- as.integer(r_env_get_action_dim())

agent <- py$DDPGAgent(obs_dim, act_dim)
py$load_agent(agent, "data/ddpg_lstm_vine_agent.pt")
cat("Agent loaded.\n")

# Run evaluation
obs <- r_env_reset()
wealth_rl <- numeric(31)
wealth_rl[1] <- 100000
weights_rl <- vector("list", 30)

for (t in 1:30) {
  state_tensor <- py$torch$FloatTensor(obs)$unsqueeze(0L)$unsqueeze(0L)  # (1, 1, obs_dim)
  action <- agent$select_action(state_tensor, noise_scale = 0.0)
  step_res <- env_eval$step(action)
  wealth_rl[t + 1] <- step_res$info$wealth
  weights_rl[[t]] <- action
  obs <- step_res$observation
}

# Compute metrics
rets_rl <- diff(wealth_rl) / wealth_rl[1:30]
ann_ret <- (wealth_rl[31] / 100000)^(1 / 2.5) - 1   # 30 months = 2.5 years
ann_vol <- sd(rets_rl) * sqrt(12)
sharpe_rl <- ann_ret / ann_vol
max_dd <- max(1 - wealth_rl / cummax(wealth_rl))

cat(sprintf("\nRL Agent (DDPG + LSTM + Vine):\n"))
cat(sprintf("  Final wealth: %.0f\n", wealth_rl[31]))
cat(sprintf("  Annual return: %.2f%%\n", ann_ret * 100))
cat(sprintf("  Annual vol:    %.2f%%\n", ann_vol * 100))
cat(sprintf("  Sharpe ratio:  %.3f\n", sharpe_rl))
cat(sprintf("  Max drawdown:  %.2f%%\n", max_dd * 100))