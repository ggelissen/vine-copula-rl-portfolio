# ============================================================================
# train_rl.r — DDPG + LSTM + Vine Copula State
# ============================================================================
library(reticulate)

# Source infrastructure
source("rl/rl_environment.r")
load("data/marginal_results.RData")
load("data/vine_fit.RData")
returns <- load_returns()

# Build vine sequence
L <- 500
all_dates <- index(returns)
rebal_idx <- endpoints(returns, on = "months")
rebal_idx <- rebal_idx[rebal_idx >= (L + 1)]
rebal_dates <- index(returns)[rebal_idx]
vine_seq <- build_vine_sequence(returns, U, rebal_dates, L = L)

# Create R environment
env_r <- RLEnvironment$new(
  marginals, asset_names,
  vine = NULL, vine_sequence = vine_seq, dynamic = TRUE,
  ref_col = 7, gamma = 2, T = 12, w0 = 100000
)

# ---- Expose R6 methods ----
r_env_reset <- function() env_r$reset()
r_env_step  <- function(action) env_r$step(action)
r_env_render <- function() env_r$render()
r_env_get_action_dim <- function() env_r$get_action_dim()
r_env_get_obs_dim   <- function() env_r$get_obs_dim()

# ---- Define Python Gym + DDPG + LSTM ----
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ── Gym Environment ──────────────────────────────────────
class VinePortfolioEnv(gym.Env):
    def __init__(self, reset_fn, step_fn, render_fn, action_dim, obs_dim):
        super().__init__()
        self.reset_fn = reset_fn
        self.step_fn = step_fn
        self.render_fn = render_fn
        self.action_space = spaces.Box(low=-0.5, high=1.0,
                                        shape=(int(action_dim),), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(int(obs_dim),), dtype=np.float32)
    def reset(self):
        return np.array(self.reset_fn(), dtype=np.float32)
    def step(self, action):
        res = self.step_fn(action)
        obs = np.array(res['observation'], dtype=np.float32)
        reward = float(res['reward'])
        done = bool(res['done'])
        info = dict(res['info'])
        return obs, reward, done, info
    def render(self, mode='human'):
        self.render_fn()

# ── LSTM Actor (policy network) ──────────────────────────
class LSTMActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(int(obs_dim), hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()   # output in (-1, 1), scaled to action bounds
        )

    def forward(self, state_seq, hidden=None):
        # state_seq: (batch, seq_len, obs_dim)
        out, hidden = self.lstm(state_seq, hidden)
        action = self.fc(out[:, -1, :])   # last time step
        return action, hidden

# ── LSTM Critic (Q‑network) ──────────────────────────────
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
        # Concatenate state and action along feature dimension
        x = torch.cat([state_seq, action_seq], dim=-1)
        out, hidden = self.lstm(x, hidden)
        q = self.fc(out[:, -1, :])
        return q, hidden

# ── Replay Buffer ────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Store as numpy arrays; state/next_state are (obs_dim,) vectors
        self.buffer.append((
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done)
        ))

    def sample(self, batch_size, seq_len):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Stack into batches: each is (batch, obs_dim) or (batch, action_dim)
        states      = torch.FloatTensor(np.stack(states))
        actions     = torch.FloatTensor(np.stack(actions))
        rewards     = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.stack(next_states))
        dones       = torch.FloatTensor(dones).unsqueeze(1)

        # Add sequence dimension: (batch, 1, dim) — the LSTM expects (batch, seq_len, dim)
        states      = states.unsqueeze(1)
        actions     = actions.unsqueeze(1)
        next_states = next_states.unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# ── DDPG Agent ───────────────────────────────────────────
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
        self.update_count = 0   # debug counter

    def select_action(self, state_seq, noise_scale=0.1):
        self.actor.eval()
        with torch.no_grad():
            action, _ = self.actor(state_seq)
        self.actor.train()
        action = action.squeeze(0).numpy()
        if action.ndim > 1:
            action = action[0]
        action += noise_scale * np.random.randn(self.action_dim)
        return np.clip(action, -0.5, 1.0)

    def update(self, replay_buffer, batch_size=64, seq_len=8):
        if len(replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, seq_len)

        # ---- DEBUG: print shapes ----
        if self.update_count == 0:
            print(f'DEBUG update     {self.update_count}')
            eprint(f'  states shape:      {states.shape}')
            print(f'  actions shape:     {actions.shape}')
            print(f'  rewards shape:     {rewards.shape}')
            print(f'  next_states shape: {next_states.shape}')
            print(f'  dones shape:       {dones.shape}')

        # Critic update
        with torch.no_grad():
            next_actions, _ = self.actor_target(next_states)
            if self.update_count == 0:
                print(f'  next_actions shape: {next_actions.shape}')
            target_q, _ = self.critic_target(next_states, next_actions)
            if self.update_count == 0:
                print(f'  target_q shape:     {target_q.shape}')
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q, _ = self.critic(states, actions)
        if self.update_count == 0:
            print(f'  current_q shape:    {current_q.shape}')

        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        pred_actions, _ = self.actor(states)
        actor_loss, _ = self.critic(states, pred_actions)
        actor_loss = -actor_loss.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update targets
        for target, source in zip(self.actor_target.parameters(), self.actor.parameters()):
            target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)
        for target, source in zip(self.critic_target.parameters(), self.critic.parameters()):
            target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)

        self.update_count += 1

# ── Training Loop ────────────────────────────────────────
def train(env, agent, total_steps=200000, batch_size=64, seq_len=1):
    replay_buffer = ReplayBuffer()
    episode_rewards = []
    episode_reward = 0
    obs = env.reset()
    obs_history = []

    for step in range(total_steps):
        obs_history.append(obs)
        if len(obs_history) > seq_len:
            obs_history = obs_history[-seq_len:]

        pad_len = seq_len - len(obs_history)
        seq = np.array(obs_history)
        if pad_len > 0:
            seq = np.pad(seq, ((pad_len, 0), (0, 0)), mode='constant')

        state_tensor = torch.FloatTensor(seq).unsqueeze(0)
        noise = max(0.3 * (0.99995**step), 0.02)
        action = agent.select_action(state_tensor, noise_scale=noise)
        next_obs, reward, done, _ = env.step(action)
        episode_reward += reward

        replay_buffer.push(obs, action, reward, next_obs, done)
        agent.update(replay_buffer, batch_size, seq_len)

        obs = next_obs
        if done:
            episode_rewards.append(episode_reward)
            obs = env.reset()
            obs_history = []
            episode_reward = 0

        if step % 5000 == 0 and len(episode_rewards) > 0:
            print('Step {:6d}  Episodes: {:4d}  Avg100 Reward: {:8.2f}'.format(
                  step, len(episode_rewards), np.mean(episode_rewards[-100:])))

    return agent

def save_agent(agent, path):
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
    }, path)

def load_agent(agent, path):
    ckpt = torch.load(path)
    agent.actor.load_state_dict(ckpt['actor'])
    agent.critic.load_state_dict(ckpt['critic'])
    return agent
")

# ---- Instantiate environment and agent ----
py_env <- py$VinePortfolioEnv(
  reset_fn = r_env_reset, step_fn = r_env_step, render_fn = r_env_render,
  action_dim = r_env_get_action_dim(), obs_dim = r_env_get_obs_dim()
)

agent <- py$DDPGAgent(as.integer(r_env_get_obs_dim()), as.integer(r_env_get_action_dim()))

cat("Training DDPG + LSTM agent on vine environment...\n")
py$train(py_env, agent, total_steps = 200000L)

py$save_agent(agent, "data/ddpg_lstm_vine_agent.pt")
cat("Agent saved.\n")