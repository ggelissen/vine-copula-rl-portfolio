# ============================================================================
# train_rl.r — DEBUG VERSION with File Logging
# ============================================================================
library(reticulate)
library(parallel)

# Runtime configuration.  These are deliberately environment variables so a
# scheduler job array can run independent, reproducible training replicas.
train_seed <- as.integer(Sys.getenv("TRAIN_SEED", "20260727"))
output_dir <- Sys.getenv("TRAIN_OUTPUT_DIR", "data/rl_runs/default")
smoke_test <- tolower(Sys.getenv("TRAIN_SMOKE_TEST", "false")) %in% c("1", "true", "yes")
default_cvar_sims <- if (smoke_test) "100" else "10000"
n_sim_cvar <- as.integer(Sys.getenv("N_SIM_CVAR", default_cvar_sims))
vine_sim_cores <- max(1L, as.integer(Sys.getenv("VINE_SIM_CORES", "1")))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(train_seed)
cat(sprintf("Run mode: %s | seed: %d | CVaR simulations/step: %d | vine simulation cores: %d\n",
            if (smoke_test) "SMOKE TEST" else "full", train_seed, n_sim_cvar, vine_sim_cores))

# Let the scheduler choose the visible GPU.  Do not set CUDA_VISIBLE_DEVICES
# here: Slurm/PBS use it to isolate one GPU per job.
if (Sys.getenv("TRAIN_DEVICE", "auto") == "cpu") {
  Sys.setenv(CUDA_VISIBLE_DEVICES = "")
}

# Helper: Print separator
print_sep <- function() {
  cat(paste0("\n", paste(rep("=", 60), collapse = ""), "\n"))
}

source("rl/rl_environment.r")
load("data/marginal_results.RData")
load("data/vine_fit.RData")
returns <- load_returns()

# Build real vine sequence
L <- 250
all_dates <- index(returns)
rebal_idx <- endpoints(returns, on = "months")
rebal_idx <- rebal_idx[rebal_idx >= (L + 1)]
rebal_dates <- index(returns)[rebal_idx]
vine_seq_real <- build_vine_sequence(returns, U, rebal_dates, L = L)
cat(sprintf("Real vine sequence length: %d\n", length(vine_seq_real)))

# Build static vine.  The pre-training environment samples from this vine at
# every step.  The former 100,000-path synthetic array was never consumed by
# the environment, so it only added startup time and memory use.
cat("Building static vine for pre-training...\n")
vine_static <- vinecop(
    U,
    var_types = rep("c", ncol(U)),
    structure = dvine_structure(1:ncol(U)),
    family_set = c("gaussian", "t", "clayton", "gumbel", "frank", "joe"),
    selcrit = "aic"
)

# Environments
env_pretrain <- RLEnvironment$new(
  marginals, asset_names,
  vine = vine_static, vine_sequence = NULL,
  ref_col = 7, gamma = 2, lambda = 0.1, kappa = 0.01, 
  T = 24, w0 = 100000, n_sim_cvar = n_sim_cvar, sim_cores = vine_sim_cores, seq_len = 30
)

env_finetune <- RLEnvironment$new(
  marginals, asset_names,
  vine = NULL, vine_sequence = vine_seq_real,
  ref_col = 7, gamma = 2, lambda = 0.1, kappa = 0.01, 
  T = 24, w0 = 100000, n_sim_cvar = n_sim_cvar, sim_cores = vine_sim_cores, seq_len = 30
)

# Expose R functions to Python
r_env_pretrain_reset <- function() env_pretrain$reset()
r_env_pretrain_step <- function(action) env_pretrain$step(action)
r_env_pretrain_get_action_dim <- function() as.integer(env_pretrain$get_action_dim())
r_env_pretrain_get_obs_dim <- function() as.integer(env_pretrain$get_obs_dim())
r_env_pretrain_get_seq_len <- function() as.integer(env_pretrain$get_seq_len())
r_env_pretrain_get_history <- function() env_pretrain$get_history()

r_env_finetune_reset <- function() env_finetune$reset()
r_env_finetune_step <- function(action) env_finetune$step(action)
r_env_finetune_get_action_dim <- function() as.integer(env_finetune$get_action_dim())
r_env_finetune_get_obs_dim <- function() as.integer(env_finetune$get_obs_dim())
r_env_finetune_get_seq_len <- function() as.integer(env_finetune$get_seq_len())
r_env_finetune_get_history <- function() env_finetune$get_history()

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

SEED = int(os.environ.get('TRAIN_SEED', '20260727'))
OUTPUT_DIR = os.environ.get('TRAIN_OUTPUT_DIR', 'data/rl_runs/default')
REQUESTED_DEVICE = os.environ.get('TRAIN_DEVICE', 'auto').lower()
SMOKE_TEST = os.environ.get('TRAIN_SMOKE_TEST', 'false').lower() in ('1', 'true', 'yes')
VERBOSE = os.environ.get('TRAIN_VERBOSE', 'false').lower() in ('1', 'true', 'yes')
PRETRAIN_EPISODES = int(os.environ.get('PRETRAIN_EPISODES', '3' if SMOKE_TEST else '500'))
FINETUNE_EPISODES = int(os.environ.get('FINETUNE_EPISODES', '2' if SMOKE_TEST else '200'))
PRETRAIN_BATCH_SIZE = int(os.environ.get('PRETRAIN_BATCH_SIZE', '16' if SMOKE_TEST else '128'))
FINETUNE_BATCH_SIZE = int(os.environ.get('FINETUNE_BATCH_SIZE', '16' if SMOKE_TEST else '32'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
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
        self.layernorm = nn.LayerNorm(hidden)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, int(action_dim))
        )
    def forward(self, state_seq, hidden=None):
        out, hidden = self.lstm(state_seq, hidden)
        out = self.layernorm(out)
        action = self.fc(out)
        return action, hidden

# ── LSTM Critic with Shape Assertions ─────────────────────────────────
class LSTMCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(int(obs_dim), hidden, num_layers, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden)
        self.fc = nn.Sequential(
            nn.Linear(hidden + int(action_dim), hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, state_seq, action, hidden=None):
        lstm_out, hidden = self.lstm(state_seq, hidden)
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
        # Keep replay memory in host RAM; transfer only each sampled batch.
        # R needs actions on the host too, so keeping the buffer on GPU would
        # waste VRAM and make the R/Python boundary slower.
        states = torch.from_numpy(np.stack(states)).to(device, non_blocking=True)
        actions = torch.from_numpy(np.stack(actions)).to(device, non_blocking=True)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.from_numpy(np.stack(next_states)).to(device, non_blocking=True)
        dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# ── DDPG Agent ──────────────────────────────────────────────────────────
class DDPGAgent:
    def __init__(self, obs_dim, action_dim, hidden=128, num_layers=2,
                 lr_actor=1e-2, lr_critic=1e-4, gamma=0.99, tau=0.001):
        self.actor = LSTMActor(obs_dim, action_dim, hidden, num_layers).to(device)
        self.actor_target = LSTMActor(obs_dim, action_dim, hidden, num_layers).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = LSTMCritic(obs_dim, action_dim, hidden, num_layers).to(device)
        self.critic_target = LSTMCritic(obs_dim, action_dim, hidden, num_layers).to(device)
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
                state_tensor = torch.from_numpy(np.asarray(state_seq, dtype=np.float32)).unsqueeze(0).to(device)
            else:
                state_tensor = torch.from_numpy(np.asarray(state_seq, dtype=np.float32)).to(device)
            logits, _ = self.actor(state_tensor)
            logits = logits[:, -1, :]
        self.actor.train()

        # logits = logits + noise_scale * torch.randn_like(logits)   # disables noise to logits for now
        probs = torch.softmax(logits, dim=-1)      
        # The R environment expects a host-resident numeric vector.
        action = probs.detach().cpu().numpy().flatten()
        
        if VERBOSE:
            log_print(f'Action: {action[:3]}...')
        return action

    def update(self, replay_buffer, batch_size=32):
        if len(replay_buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # Critic update
        with torch.no_grad():
          next_logits, _ = self.actor_target(next_states)
          next_logits = next_logits[:, -1, :]
          next_probs = torch.softmax(next_logits, dim=-1)
          target_q, _ = self.critic_target(next_states, next_probs)
          target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q, _ = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        grad_norm_before = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Actor update
        pred_logits, _ = self.actor(states)     
        pred_logits_last = pred_logits[:, -1, :]       
        pred_probs = torch.softmax(pred_logits_last, dim=-1)
        q_value, _ = self.critic(states, pred_probs)

        # Scale Q to increase gradient magnitude
        actor_loss = -q_value.mean()
        entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-8), dim=-1).mean()
        actor_loss = actor_loss - 0.01 * entropy  # Encourage exploration
        
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
        # Restore targets
        self.actor_target.load_state_dict(ckpt['actor'])      # or separately saved
        self.critic_target.load_state_dict(ckpt['critic'])
        # Optionally restore optimizers (with caution)
        self.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        self.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
        self.update_count = ckpt['update_count']
        # Reset learning rates after loading optimizer state
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.actor_optimizer.defaults['lr']
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = self.critic_optimizer.defaults['lr']

# ── Training Function ──────────────────────────────────────────────────
def train_stage(env, agent, episodes, batch_size=32, noise_scale=0.3,
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
        t = 0
        done = False
        while not done and t < 100:
            action = agent.select_action(state_seq, noise_scale=current_noise)
            next_state_seq, reward, done, _ = env.step(action)
            episode_reward += reward
            replay_buffer.push(state_seq, action, reward, next_state_seq, done)
            agent.update(replay_buffer, batch_size)

            # in train_stage, after update
            if VERBOSE and ep % 10 == 0 and len(replay_buffer) > batch_size:
              actor_grad_norm = 0.0
              for p in agent.actor.parameters():
                  if p.grad is not None:
                      actor_grad_norm += p.grad.norm().item() ** 2
              actor_grad_norm = actor_grad_norm ** 0.5
              
              critic_grad_norm = 0.0 
              for p in agent.critic.parameters():
                  if p.grad is not None:
                      critic_grad_norm += p.grad.norm().item() ** 2
              critic_grad_norm = critic_grad_norm ** 0.5

              with torch.no_grad():
                sample_states, _, _, _, _ = replay_buffer.sample(min(batch_size, len(replay_buffer)))
                logits, _ = agent.actor(sample_states)
                probs = torch.softmax(logits, dim=-1)
                q_vals, _ = agent.critic(sample_states, probs)
                mean_q = q_vals.mean().item()
              
              log_print(f'  Gradients - Actor: {actor_grad_norm:.6f}, Critic: {critic_grad_norm:.6f}')
              log_print(f'  Time: {t}')

            state_seq = next_state_seq
            t += 1

        episode_rewards.append(episode_reward)
        log_print(f'Episode {ep+1}  Reward: {episode_reward:8.2f}')
        if (ep + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            log_print(f'Episode {ep+1:6d}  AvgReward: {avg_reward:8.2f}  Noise: {current_noise:.4f}')
    log_print('TRAIN STAGE COMPLETE')
    log_file.flush()
    return episode_rewards

def create_env(reset_fn, step_fn, render_fn, get_history_fn, action_dim, obs_dim, seq_len):
    return VinePortfolioEnv(reset_fn, step_fn, render_fn, get_history_fn, action_dim, obs_dim, seq_len)

def create_agent(obs_dim, action_dim, lr_actor=1e-2, lr_critic=1e-4, gamma=0.99, tau=0.001):
    return DDPGAgent(obs_dim, action_dim, lr_actor=lr_actor, lr_critic=lr_critic,
                     gamma=gamma, tau=tau)

def save_agent(agent, path):
    agent.save(path)

def load_agent(agent, path):
    agent.load(path)

log_print('PYTHON: Framework ready.')
log_file.flush()
")

# ============================================================================
# Run Training
# ============================================================================

print_sep()
cat("Stage 1: Pre-training on Synthetic Data\n")
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

agent = create_agent(int(r.r_env_pretrain_get_obs_dim()), int(r.r_env_pretrain_get_action_dim()), 
                     lr_actor=1e-4, lr_critic=1e-4, gamma=1.0, tau=0.005)

pretrain_rewards = train_stage(env_pretrain, agent, episodes=PRETRAIN_EPISODES, batch_size=PRETRAIN_BATCH_SIZE,
                               noise_scale=0.05, noise_decay=0.999, log_interval=10)
save_agent(agent, os.path.join(OUTPUT_DIR, 'ddpg_lstm_vine_pretrained.pt'))
log_print('Pre-training complete. Agent saved.')

# log_file.close()
")

print_sep()
cat("Stage 2: Fine-tuning on Real Data\n")
print_sep()

py_run_string("
log_print('='*60)
log_print('STAGE 2: FINE-TUNING')
log_print('='*60)

env_finetune = create_env(
    reset_fn = r.r_env_finetune_reset,
    step_fn = r.r_env_finetune_step,
    render_fn = lambda: None,
    get_history_fn = r.r_env_finetune_get_history,
    action_dim = int(r.r_env_finetune_get_action_dim()),
    obs_dim = int(r.r_env_finetune_get_obs_dim()),
    seq_len = int(r.r_env_finetune_get_seq_len())
)

# Load pre-trained agent with lower learning rate
agent_finetune = create_agent(int(r.r_env_finetune_get_obs_dim()), int(r.r_env_finetune_get_action_dim()), 
                              lr_actor=1e-4, lr_critic=1e-4, gamma=1.0, tau=0.001)
load_agent(agent_finetune, os.path.join(OUTPUT_DIR, 'ddpg_lstm_vine_pretrained.pt'))
agent_finetune.actor_target.load_state_dict(agent_finetune.actor.state_dict())
agent_finetune.critic_target.load_state_dict(agent_finetune.critic.state_dict())
print('Loaded pre-trained agent. Starting fine-tuning...')
finetune_rewards = train_stage(env_finetune, agent_finetune, episodes=FINETUNE_EPISODES, batch_size=FINETUNE_BATCH_SIZE,
                               noise_scale=0.1, noise_decay=0.999, log_interval=10)
save_agent(agent_finetune, os.path.join(OUTPUT_DIR, 'ddpg_lstm_vine_full.pt'))
print('Fine-tuning complete. Final agent saved.')
")

print_sep()
cat("TRAINING COMPLETE\n")
print_sep()
