# ============================================================================
# train_rl.r — DEBUG VERSION with File Logging
# ============================================================================
library(reticulate)
library(parallel)

# Helper: Print separator
print_sep <- function() {
  cat(paste0("\n", paste(rep("=", 60), collapse = ""), "\n"))
}

source("rl/rl_environment.r")
load("data/marginal_results.RData")
load("data/vine_fit.RData")
returns <- load_returns()

# Build real vine sequence
L <- 500
all_dates <- index(returns)
rebal_idx <- endpoints(returns, on = "months")
rebal_idx <- rebal_idx[rebal_idx >= (L + 1)]
rebal_dates <- index(returns)[rebal_idx]
vine_seq_real <- build_vine_sequence(returns, U, rebal_dates, L = L)
cat(sprintf("Real vine sequence length: %d\n", length(vine_seq_real)))

# Build static vine
cat("Generating synthetic data for pre-training...\n")
vine_static <- vinecop(
    U,
    var_types = rep("c", ncol(U)),
    structure = dvine_structure(1:ncol(U)),
    family_set = c("gaussian", "t", "clayton", "gumbel", "frank", "joe"),
    selcrit = "aic"
)

sim <- build_simulator(marginals, asset_names, ref_col = 7)
n_synth_paths <- 100000
T_synth <- 12
n_assets <- length(asset_names)
synth_returns_array <- array(0, dim = c(n_synth_paths, T_synth, n_assets))
vine_for_synth <- vine_static
for (path in 1:min(n_synth_paths, 100)) {  # Reduced for speed
  sim_result <- sim$simulate_returns(vine_for_synth, n_sim = T_synth)
  synth_returns_array[path, , ] <- log(sim_result$gross)  
}

# Environments
env_pretrain <- RLEnvironment$new(
  marginals, asset_names,
  vine = vine_static, vine_sequence = NULL,
  ref_col = 7, gamma = 2, lambda = 1.0, kappa = 0.05, 
  T = 12, w0 = 100000, n_sim_cvar = 10000, seq_len = 30
)

env_finetune <- RLEnvironment$new(
  marginals, asset_names,
  vine = NULL, vine_sequence = vine_seq_real,
  ref_col = 7, gamma = 2, lambda = 1.0, kappa = 0.05, 
  T = 12, w0 = 100000, n_sim_cvar = 10000, seq_len = 30
)

# Expose R functions to Python
py$r_env_pretrain_reset <- function() env_pretrain$reset()
py$r_env_pretrain_step <- function(action) env_pretrain$step(action)
py$r_env_pretrain_get_action_dim <- function() as.integer(env_pretrain$get_action_dim())
py$r_env_pretrain_get_obs_dim <- function() as.integer(env_pretrain$get_obs_dim())
py$r_env_pretrain_get_seq_len <- function() as.integer(env_pretrain$get_seq_len())
py$r_env_pretrain_get_history <- function() env_pretrain$get_history()

py$r_env_finetune_reset <- function() env_finetune$reset()
py$r_env_finetune_step <- function(action) env_finetune$step(action)
py$r_env_finetune_get_action_dim <- function() as.integer(env_finetune$get_action_dim())
py$r_env_finetune_get_obs_dim <- function() as.integer(env_finetune$get_obs_dim())
py$r_env_finetune_get_seq_len <- function() as.integer(env_finetune$get_seq_len())
py$r_env_finetune_get_history <- function() env_finetune$get_history()

# ============================================================================
# Python Code — with File Logging
# ============================================================================
py_run_string("
import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
import sys

# ================================================================
# Open log file for debug output
# ================================================================
log_file = open('debug_output.txt', 'w')
log_file.write('='*60 + '\\n')
log_file.write('PYTHON DEBUG LOG STARTED\\n')
log_file.write('='*60 + '\\n')
log_file.flush()
os.fsync(log_file.fileno())

def log_print(*args, **kwargs):
    import time
    timestamp = time.strftime('%H:%M:%S')
    msg = ' '.join(str(a) for a in args)
    log_file.write(f'[{timestamp}] {msg}\\n')
    log_file.flush()
    os.fsync(log_file.fileno())
    # Also try to print to console
    print(msg)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
        self.action_space = spaces.Box(low=-0.5, high=1.0, shape=(self.action_dim,), dtype=np.float32)
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
        self.lstm = nn.LSTM(int(obs_dim), hidden, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, int(action_dim)),
            nn.Tanh()
        )
    def forward(self, state_seq, hidden=None):
        out, hidden = self.lstm(state_seq, hidden)
        action = self.fc(out)
        return action, hidden

# ── LSTM Critic with Shape Assertions ─────────────────────────────────
class LSTMCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(int(obs_dim + action_dim), hidden, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, state_seq, action_seq, hidden=None):
        if action_seq.dim() == 2:
            # This is a fallback — should not happen after fixing the actor
            action_seq = action_seq.unsqueeze(1)
            log_print(f'  WARNING: action_seq had dim 2, unsqueezed to {action_seq.shape}')
        if state_seq.shape[1] != action_seq.shape[1]:
            error_msg = (
                f'Shape mismatch in critic: state_seq has seq_len {state_seq.shape[1]}, '
                f'action_seq has seq_len {action_seq.shape[1]}. '
                f'Both must have the same sequence length.'
            )
            log_print('ERROR: ' + error_msg)
            raise RuntimeError(error_msg)
        x = torch.cat([state_seq, action_seq], dim=-1)
        out, hidden = self.lstm(x, hidden)
        q = self.fc(out[:, -1, :])
        return q, hidden

# ── Replay Buffer ──────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state_seq, action, reward, next_state_seq, done):
        seq_len = state_seq.shape[0]
        # --- CRITICAL FIX: repeat action to match sequence length ---
        action_seq = np.tile(action, (seq_len, 1))  # (seq_len, action_dim)
        self.buffer.append((
            np.array(state_seq, dtype=np.float32),
            np.array(action_seq, dtype=np.float32),
            float(reward),
            np.array(next_state_seq, dtype=np.float32),
            float(done)
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.stack(states))
        actions = torch.FloatTensor(np.stack(actions))
        rewards = torch.FloatTensor(list(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.stack(next_states))
        dones = torch.FloatTensor(list(dones)).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# ── DDPG Agent ──────────────────────────────────────────────────────────
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
        self.update_count = 0

    def select_action(self, state_seq, noise_scale=0.1):
        self.actor.eval()
        with torch.no_grad():
            if state_seq.ndim == 2:
                state_tensor = torch.FloatTensor(state_seq).unsqueeze(0)
            else:
                state_tensor = torch.FloatTensor(state_seq)
            actions, _ = self.actor(state_tensor)
            action = actions[:, -1, :]  # Take the last time step's action
        self.actor.train()
        action = action.numpy()
        # if action.ndim > 1:
        #     action = action[0]
        action += noise_scale * np.random.randn(self.action_dim)
        action_clipped = np.clip(action, -0.5, 1.0)
        return action_clipped

    def update(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # Critic update
        with torch.no_grad():
            next_actions, _ = self.actor_target(next_states)  # (batch, seq_len, action_dim)
            target_q, _ = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q, _ = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Actor update
        pred_actions, _ = self.actor(states)  # (batch, seq_len, action_dim)
        actor_loss, _ = self.critic(states, pred_actions)
        actor_loss = -actor_loss.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        for target, source in zip(self.actor_target.parameters(), self.actor.parameters()):
            target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)
        for target, source in zip(self.critic_target.parameters(), self.critic.parameters()):
            target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)
        self.update_count += 1

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'update_count': self.update_count
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        self.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
        self.update_count = ckpt['update_count']

# ── Training Function ──────────────────────────────────────────────────
def train_stage(env, agent, episodes, batch_size=64, noise_scale=0.3,
                noise_decay=0.999, log_interval=1):
    log_print('='*60)
    log_print(f'TRAIN STAGE STARTED: episodes={episodes}, batch_size={batch_size}')
    log_print('='*60)
    replay_buffer = ReplayBuffer()
    episode_rewards = []
    for ep in range(episodes):
        state_seq = env.reset()
        episode_reward = 0
        current_noise = noise_scale * (noise_decay ** ep)
        for t in range(env.seq_len):
            action = agent.select_action(state_seq, noise_scale=current_noise)
            next_state_seq, reward, done, _ = env.step(action)
            episode_reward += reward
            replay_buffer.push(state_seq, action, reward, next_state_seq, done)
            agent.update(replay_buffer, batch_size)
            state_seq = next_state_seq
            if done:
                log_print(f'  DONE at step {t}')
                break
        episode_rewards.append(episode_reward)
        log_print(f'Episode {ep+1}  Reward: {episode_reward:8.2f}')
        if (ep + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            log_print(f'Episode {ep+1:6d}  AvgReward: {avg_reward:8.2f}  Noise: {current_noise:.4f}')
    log_print('TRAIN STAGE COMPLETE')
    log_file.flush()
    os.fsync(log_file.fileno())
    return episode_rewards

def create_env(reset_fn, step_fn, render_fn, get_history_fn, action_dim, obs_dim, seq_len):
    return VinePortfolioEnv(reset_fn, step_fn, render_fn, get_history_fn, action_dim, obs_dim, seq_len)

def create_agent(obs_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005):
    return DDPGAgent(obs_dim, action_dim, lr_actor=lr_actor, lr_critic=lr_critic,
                     gamma=gamma, tau=tau)

def save_agent(agent, path):
    agent.save(path)

def load_agent(agent, path):
    agent.load(path)

log_print('PYTHON: Framework ready.')
log_file.flush()
os.fsync(log_file.fileno())
")

# ============================================================================
# Run Training
# ============================================================================

# print_sep()
# cat("Stage 1: Pre-training on Synthetic Data\n")
# print_sep()

# py_run_string("
# log_print('='*60)
# log_print('STAGE 1: PRE-TRAINING')
# log_print('='*60)

# obs_dim = int(r_env_pretrain_get_obs_dim())
# action_dim = int(r_env_pretrain_get_action_dim())
# seq_len = int(r_env_pretrain_get_seq_len())

# env_pretrain = create_env(
#     reset_fn = r_env_pretrain_reset,
#     step_fn = r_env_pretrain_step,
#     render_fn = lambda: None,
#     get_history_fn = r_env_pretrain_get_history,
#     action_dim = action_dim,
#     obs_dim = obs_dim,
#     seq_len = seq_len
# )

# agent = create_agent(obs_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3)

# pretrain_rewards = train_stage(env_pretrain, agent, episodes=5000, batch_size=64,
#                                noise_scale=0.3, noise_decay=0.999, log_interval=1)
# save_agent(agent, 'data/ddpg_lstm_vine_pretrained.pt')
# log_print('Pre-training complete. Agent saved.')

# # log_file.close()
# ")

print_sep()
cat("Stage 2: Fine-tuning on Real Data\n")
print_sep()

py_run_string("
log_print('='*60)
log_print('STAGE 2: FINE-TUNING')
log_print('='*60)

env_finetune = create_env(
    reset_fn = r_env_finetune_reset,
    step_fn = r_env_finetune_step,
    render_fn = lambda: None,
    get_history_fn = r_env_finetune_get_history,
    action_dim = int(r_env_finetune_get_action_dim()),
    obs_dim = int(r_env_finetune_get_obs_dim()),
    seq_len = int(r_env_finetune_get_seq_len())
)

# Load pre-trained agent with lower learning rate
agent_finetune = create_agent(int(r_env_finetune_get_obs_dim()), int(r_env_finetune_get_action_dim()), 
                              lr_actor=1e-5, lr_critic=1e-4, gamma=0.99, tau=0.005)
load_agent(agent_finetune, 'data/ddpg_lstm_vine_pretrained.pt')
print('Loaded pre-trained agent. Starting fine-tuning...')
finetune_rewards = train_stage(env_finetune, agent_finetune, episodes=2000, batch_size=64,
                               noise_scale=0.1, noise_decay=0.999, log_interval=200)
save_agent(agent_finetune, 'data/ddpg_lstm_vine_full.pt')
print('Fine-tuning complete. Final agent saved.')
")

print_sep()
cat("TRAINING COMPLETE\n")
print_sep()