# ============================================================================
# train_rl.r — Training algorithm for DRL agent
# ============================================================================

library(reticulate)
library(parallel)
library(rvinecopulib)
library(zoo)

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
env_short_borrow_rate <- as.numeric(Sys.getenv("ENV_SHORT_BORROW_RATE", "0.03"))
env_cash_borrow_rate <- as.numeric(Sys.getenv("ENV_CASH_BORROW_RATE", "0.02"))
env_utility_mode <- Sys.getenv("ENV_UTILITY_MODE", "terminal_wealth_crra")
vine_model <- Sys.getenv("VINE_MODEL", "nn_dynamic_t_vine")
nn_vine_epochs <- as.integer(Sys.getenv("NN_VINE_EPOCHS", "200"))
nn_vine_lr <- as.numeric(Sys.getenv("NN_VINE_LR", "0.001"))
nn_vine_patience <- as.integer(Sys.getenv("NN_VINE_PATIENCE", "20"))

pretrain_episodes <- as.integer(Sys.getenv("PRETRAIN_EPISODES"))
pretrain_batch_size <- as.integer(Sys.getenv("PRETRAIN_BATCH_SIZE"))
pretrain_noise_scale <- as.numeric(Sys.getenv("PRETRAIN_NOISE_SCALE"))
pretrain_noise_decay <- as.numeric(Sys.getenv("PRETRAIN_NOISE_DECAY"))
pretrain_updates <- as.integer(Sys.getenv("PRETRAIN_UPDATES_PER_STEP"))

finetune_episodes <- as.integer(Sys.getenv("FINETUNE_EPISODES"))
finetune_batch_size <- as.integer(Sys.getenv("FINETUNE_BATCH_SIZE"))
finetune_noise_scale <- as.numeric(Sys.getenv("FINETUNE_NOISE_SCALE"))
finetune_noise_decay <- as.numeric(Sys.getenv("FINETUNE_NOISE_DECAY"))
finetune_updates <- as.integer(Sys.getenv("FINETUNE_UPDATES_PER_STEP"))
load_model_path <- Sys.getenv("LOAD_MODEL_PATH", "")

lr_actor <- as.numeric(Sys.getenv("LR_ACTOR"))
lr_critic <- as.numeric(Sys.getenv("LR_CRITIC"))
discount <- as.numeric(Sys.getenv("DISCOUNT"))
tau <- as.numeric(Sys.getenv("TAU"))
hidden <- as.integer(Sys.getenv("HIDDEN"))
num_layers <- as.integer(Sys.getenv("NUM_LAYERS"))
replay_capacity <- as.integer(Sys.getenv("REPLAY_CAPACITY"))
entropy_coef <- as.numeric(Sys.getenv("ENTROPY_COEF"))
grad_clip_norm <- as.numeric(Sys.getenv("GRAD_CLIP_NORM"))

# ---- Ensure required variables are set ----
if (any(is.na(c(train_seed, output_dir, device, n_sim_cvar, vine_sim_cores, L, ref_col,
                synthetic_file, env_gamma, env_lambda, env_kappa,
                env_T, env_w0, env_seq_len, env_holding_days, env_gross_leverage, env_net_exposure, pretrain_episodes, pretrain_batch_size,
                pretrain_noise_scale, pretrain_noise_decay, pretrain_updates,
                finetune_episodes, finetune_batch_size, finetune_noise_scale,
                finetune_noise_decay, finetune_updates, lr_actor, lr_critic,
                discount, tau, hidden, num_layers, replay_capacity,
                entropy_coef, grad_clip_norm)))) {
  stop("One or more required environment variables are not set. Check your launcher.")
}
if (!identical(vine_model, "nn_dynamic_t_vine")) stop("RL supports only VINE_MODEL=nn_dynamic_t_vine; rolling-window vines are intentionally disabled.")
if (!identical(env_utility_mode, "terminal_wealth_crra")) stop("Set ENV_UTILITY_MODE=terminal_wealth_crra for the multi-period CRRA objective.")
if (abs(discount - 1) > 1e-12) stop("DISCOUNT must be 1.0 when using telescoping terminal-wealth CRRA utility.")
if (evaluation_periods != 24L) stop("The final historical holdout must contain exactly 24 monthly holding periods.")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(train_seed)
source("helper/reproducibility.r")
write_run_manifest(output_dir, train_seed,
  config_file = Sys.getenv("CONFIG_FILE", "config/config.yaml"),
  data_files = c("data/portfolio_B_7assets_2013.csv", synthetic_file,
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
if (!file.exists(training_marginals_file)) stop(sprintf("Training-only marginal file not found: %s\nRun rl/synthetic_returns.r first.", training_marginals_file))
load(training_marginals_file)
returns <- load_returns()

# Reconstruct and validate the locked calendar; the NN vine itself was fitted
# once by synthetic_returns.r and is not redundantly re-estimated here.
period_split <- split_monthly_periods(
  build_monthly_periods(returns, min_history = L), evaluation_periods
)
validate_period_split(period_split, evaluation_periods)
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
metadata <- if (exists("metadata", envir = bundle)) get("metadata", envir = bundle) else NULL
if (is.null(metadata) || !isTRUE(metadata$diagnostics_passed) || !exists("train_end", envir = bundle, inherits = FALSE) || !identical(metadata$pretrain_realised_source, "synthetic_vine") || !identical(metadata$finetune_realised_source, "historical") || !identical(metadata$pretrain_vine_frequency, "monthly_marginal_transform") || !identical(metadata$pretrain_vine_model, "nn_dynamic_t_vine") || !identical(metadata$finetune_vine_model, "nn_dynamic_t_vine") || !identical(metadata$pretrain_vine_structure, "nn_dynamic_all_tree_dvine") || !identical(as.integer(metadata$dynamic_vine_edges), length(asset_names) * (length(asset_names) - 1L) / 2L) || !identical(as.integer(metadata$sequence_length), env_seq_len) || !identical(as.integer(metadata$reserved_evaluation_steps), evaluation_periods) || as.integer(train_end) != as.integer(get("train_end", envir = bundle, inherits = FALSE))) {
  stop("Training bundle does not satisfy the NN-vine synthetic-pretrain / historical-finetune protocol. Regenerate it with rl/synthetic_returns.r.")
}
cat(sprintf("Loaded %d pre-training and %d fine-tuning episodes.\n", length(pretrain_returns), length(finetune_returns)))
if (pretrain_episodes != length(pretrain_returns)) {
  stop("PRETRAIN_EPISODES must equal the generated episode count so all and only the synthetic data are used once.")
}
if (finetune_episodes < length(finetune_returns) || finetune_episodes %% length(finetune_returns) != 0L) {
  stop("FINETUNE_EPISODES must be a positive multiple of the historical trajectory count for balanced exposure.")
}

# ---- Create environments ----
env_pretrain <- RLEnvironment$new(
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
  short_borrow_rate = env_short_borrow_rate,
  cash_borrow_rate = env_cash_borrow_rate,
  utility_mode = env_utility_mode,
  episode_sampling = "sequential"
)
env_pretrain$set_precomputed_returns(pretrain_returns)

env_finetune <- RLEnvironment$new(
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
  short_borrow_rate = env_short_borrow_rate,
  cash_borrow_rate = env_cash_borrow_rate,
  utility_mode = env_utility_mode,
  episode_sampling = "sequential"
)
env_finetune$set_precomputed_returns(finetune_returns)

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
DISCOUNT = float(os.environ.get('DISCOUNT'))
TAU = float(os.environ.get('TAU'))
HIDDEN = int(os.environ.get('HIDDEN'))
NUM_LAYERS = int(os.environ.get('NUM_LAYERS'))
REPLAY_CAPACITY = int(os.environ.get('REPLAY_CAPACITY'))
ENTROPY_COEF = float(os.environ.get('ENTROPY_COEF'))
GRAD_CLIP_NORM = float(os.environ.get('GRAD_CLIP_NORM'))
POLICY_DELAY = int(os.environ.get('POLICY_DELAY', '2'))
TARGET_POLICY_NOISE = float(os.environ.get('TARGET_POLICY_NOISE', '0.2'))
TARGET_NOISE_CLIP = float(os.environ.get('TARGET_NOISE_CLIP', '0.5'))
RANDOM_EXPLORATION_STEPS = int(os.environ.get('RANDOM_EXPLORATION_STEPS', '1000'))
DETERMINISTIC_ALGORITHMS = os.environ.get('DETERMINISTIC_ALGORITHMS', 'true').lower() in ('1', 'true', 'yes')
LOAD_MODEL_PATH = os.environ.get('LOAD_MODEL_PATH', '')
PRETRAIN_NOISE_SCALE = float(os.environ.get('PRETRAIN_NOISE_SCALE'))
PRETRAIN_NOISE_DECAY = float(os.environ.get('PRETRAIN_NOISE_DECAY'))
PRETRAIN_UPDATES = int(os.environ.get('PRETRAIN_UPDATES_PER_STEP'))
FINETUNE_NOISE_SCALE = float(os.environ.get('FINETUNE_NOISE_SCALE'))
FINETUNE_NOISE_DECAY = float(os.environ.get('FINETUNE_NOISE_DECAY'))
FINETUNE_UPDATES = int(os.environ.get('FINETUNE_UPDATES_PER_STEP'))
GROSS_LEVERAGE = float(os.environ.get('ENV_GROSS_LEVERAGE'))
NET_EXPOSURE = float(os.environ.get('ENV_NET_EXPOSURE'))
SHORT_BORROW_RATE = float(os.environ.get('ENV_SHORT_BORROW_RATE', '0.03'))
CASH_BORROW_RATE = float(os.environ.get('ENV_CASH_BORROW_RATE', '0.02'))
UTILITY_MODE = os.environ.get('ENV_UTILITY_MODE')
if GROSS_LEVERAGE < abs(NET_EXPOSURE):
    raise RuntimeError('Gross leverage must be at least abs(net exposure).')

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
            nn.Linear(hidden, int(action_dim))
        )
    def forward(self, state_seq, hidden=None):
        out, hidden = self.lstm(self.input_norm(state_seq), hidden)
        out = self.layernorm(out)
        action = self.fc(out)
        return action, hidden

def portfolio_weights(logits):
    # Differentiable self-financing two-book projection. Separate long/short
    # budgets impose net exposure and the gross cap without clipping gradients.
    long_budget = 0.5 * (GROSS_LEVERAGE + NET_EXPOSURE)
    short_budget = 0.5 * (GROSS_LEVERAGE - NET_EXPOSURE)
    return long_budget * torch.softmax(logits, dim=-1) - short_budget * torch.softmax(-logits, dim=-1)

def allocation_entropy(logits):
    long_probs = torch.softmax(logits, dim=-1)
    short_probs = torch.softmax(-logits, dim=-1)
    return -0.5 * (torch.sum(long_probs * torch.log(long_probs + 1e-8), dim=-1) +
                   torch.sum(short_probs * torch.log(short_probs + 1e-8), dim=-1)).mean()

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
                 entropy_coef=0.0, grad_clip_norm=1.0):
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
        self.update_count = 0
        self.entropy_coef = entropy_coef
        self.grad_clip_norm = grad_clip_norm
        self.total_actions = 0
        self.scaler = torch.amp.GradScaler('cuda',enabled=(device.type == 'cuda'))


    def select_action(self, state_seq, noise_scale=0.0):
        self.actor.eval()
        with torch.no_grad():
            if state_seq.ndim == 2:
                state_tensor = torch.from_numpy(np.asarray(state_seq, dtype=np.float32)).unsqueeze(0).to(device)
            else:
                state_tensor = torch.from_numpy(np.asarray(state_seq, dtype=np.float32)).to(device)
            if self.total_actions < RANDOM_EXPLORATION_STEPS:
                logits = torch.randn((state_tensor.shape[0], self.action_dim), device=device)
            else:
                logits, _ = self.actor(state_tensor)
                logits = logits[:, -1, :]
        self.actor.train()
        # Exploration belongs in logits, before the long-short projection.
        if noise_scale > 0:
            logits = logits + torch.randn_like(logits) * noise_scale
        action = portfolio_weights(logits).detach().cpu().numpy().flatten()
        self.total_actions += 1
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
            smoothing = torch.randn_like(next_logits) * TARGET_POLICY_NOISE
            next_logits = next_logits + smoothing.clamp(-TARGET_NOISE_CLIP, TARGET_NOISE_CLIP)
            next_action = portfolio_weights(next_logits)
            target_q1, _ = self.critic_target(next_states, next_action)
            target_q2, _ = self.critic2_target(next_states, next_action)
            target_q = rewards + (1 - dones) * self.gamma * torch.minimum(target_q1, target_q2)
        
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            current_q, _ = self.critic(states, actions)
            current_q2, _ = self.critic2(states, actions)
            critic_loss = nn.MSELoss()(current_q, target_q) + nn.MSELoss()(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.critic_optimizer)
        self.scaler.unscale_(self.critic2_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.grad_clip_norm)
        self.scaler.step(self.critic_optimizer)
        self.scaler.step(self.critic2_optimizer)
        self.scaler.update()

        self.update_count += 1
        # Delayed policy and target updates are the defining TD3 correction for
        # critic over-estimation and rapidly moving targets.
        if self.update_count % POLICY_DELAY == 0:
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                pred_logits, _ = self.actor(states)
                pred_logits_last = pred_logits[:, -1, :]
                pred_action = portfolio_weights(pred_logits_last)
                q_value, _ = self.critic(states, pred_action)
                actor_loss = -q_value.mean()
                if self.entropy_coef != 0.0:
                    actor_loss = actor_loss - self.entropy_coef * allocation_entropy(pred_logits_last)

            self.actor_optimizer.zero_grad()
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_optimizer)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()

            for target, source in zip(self.actor_target.parameters(), self.actor.parameters()):
                target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)
            for target, source in zip(self.critic_target.parameters(), self.critic.parameters()):
                target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)
            for target, source in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)

    def save(self, path):
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
            'architecture': {'obs_dim': self.obs_dim, 'action_dim': self.action_dim,
                             'hidden': self.actor.lstm.hidden_size, 'num_layers': self.actor.lstm.num_layers,
                             'agent': 'td3', 'state_normalization': 'layer_norm',
                             'action_mode': 'long_short_two_book', 'gross_leverage': GROSS_LEVERAGE,
                             'net_exposure': NET_EXPOSURE, 'short_borrow_rate': SHORT_BORROW_RATE,
                             'cash_borrow_rate': CASH_BORROW_RATE, 'utility_mode': UTILITY_MODE}
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location='cpu')
        architecture = ckpt.get('architecture', {})
        expected = {'agent': 'td3', 'state_normalization': 'layer_norm',
                    'action_mode': 'long_short_two_book', 'gross_leverage': GROSS_LEVERAGE,
                    'net_exposure': NET_EXPOSURE, 'short_borrow_rate': SHORT_BORROW_RATE,
                    'cash_borrow_rate': CASH_BORROW_RATE, 'utility_mode': UTILITY_MODE}
        mismatches = {k: (architecture.get(k), v) for k, v in expected.items() if architecture.get(k) != v}
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
        # Reset learning rates after loading
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.actor_optimizer.defaults['lr']
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = self.critic_optimizer.defaults['lr']
        for param_group in self.critic2_optimizer.param_groups:
            param_group['lr'] = self.critic2_optimizer.defaults['lr']

# ── Training Function ──────────────────────────────────────────────────
def train_stage(env, agent, episodes, batch_size=32, noise_scale=0.0,
                noise_decay=0.999, log_interval=1, updates_per_step=3):
    log_print('='*60)
    log_print(f'TRAIN STAGE STARTED: episodes={episodes}, batch_size={batch_size}, updates_per_step={updates_per_step}')
    log_print('='*60)
    replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
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

            for _ in range(updates_per_step):
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
                actions = portfolio_weights(logits[:, -1, :])
                q_vals, _ = agent.critic(sample_states, actions)
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

def create_agent(obs_dim, action_dim, lr_actor=1e-4, lr_critic=1e-4, gamma=1.0, tau=0.005):
    return TD3Agent(obs_dim, action_dim, hidden=HIDDEN, num_layers=NUM_LAYERS,
                     lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma, tau=tau,
                     entropy_coef=ENTROPY_COEF, grad_clip_norm=GRAD_CLIP_NORM)

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

# ---- Pre-training ----
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

agent = create_agent(
    obs_dim = int(r.r_env_pretrain_get_obs_dim()),
    action_dim = int(r.r_env_pretrain_get_action_dim()),
    lr_actor = LR_ACTOR,
    lr_critic = LR_CRITIC,
    gamma = DISCOUNT,
    tau = TAU
)

pretrain_rewards = train_stage(
    env_pretrain, agent,
    episodes = PRETRAIN_EPISODES,
    batch_size = PRETRAIN_BATCH_SIZE,
    noise_scale = PRETRAIN_NOISE_SCALE,
    noise_decay = PRETRAIN_NOISE_DECAY,
    log_interval = 10,
    updates_per_step = PRETRAIN_UPDATES
)
save_agent(agent, os.path.join(OUTPUT_DIR, 'td3_lstm_vine_pretrained.pt'))
log_print('Pre-training complete. Agent saved.')
")

# ---- Fine-tuning ----
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

agent_finetune = create_agent(
    obs_dim = int(r.r_env_finetune_get_obs_dim()),
    action_dim = int(r.r_env_finetune_get_action_dim()),
    lr_actor = LR_ACTOR,
    lr_critic = LR_CRITIC,
    gamma = DISCOUNT,
    tau = TAU
)

# Load pre-trained model if specified
if LOAD_MODEL_PATH and os.path.exists(LOAD_MODEL_PATH):
    load_agent(agent_finetune, LOAD_MODEL_PATH)
    print(f'Loaded pre-trained agent from {LOAD_MODEL_PATH}')
else:
    default_path = os.path.join(OUTPUT_DIR, 'td3_lstm_vine_pretrained.pt')
    if os.path.exists(default_path):
        load_agent(agent_finetune, default_path)
        print(f'Loaded pre-trained agent from {default_path}')
    else:
        print('No pre-trained agent found; starting from scratch.')

# Sync target networks
agent_finetune.actor_target.load_state_dict(agent_finetune.actor.state_dict())
agent_finetune.critic_target.load_state_dict(agent_finetune.critic.state_dict())
agent_finetune.critic2_target.load_state_dict(agent_finetune.critic2.state_dict())
print('Loaded pre-trained agent. Starting fine-tuning...')

finetune_rewards = train_stage(
    env_finetune, agent_finetune,
    episodes = FINETUNE_EPISODES,
    batch_size = FINETUNE_BATCH_SIZE,
    noise_scale = FINETUNE_NOISE_SCALE,
    noise_decay = FINETUNE_NOISE_DECAY,
    log_interval = 10,
    updates_per_step = FINETUNE_UPDATES
)
save_agent(agent_finetune, os.path.join(OUTPUT_DIR, 'td3_lstm_vine_full.pt'))
print('Fine-tuning complete. Final agent saved.')
")

print_sep()
cat("TRAINING COMPLETE\n")
print_sep()
