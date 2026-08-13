"""Matched-architecture TD3/DDPG/SAC/PPO/A2C controls for portfolio experiments.

The consumed schema-5 TD3-LSTM implementation remains the reference. This
module supplies post-holdout controls under the same observation tensors,
differentiable long/short projection, constraints, rewards, and seeds. SAC and
PPO optimize entropy in the unconstrained latent action space; the economic
action is always the shared deterministic portfolio projection.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from rl.action_projection import portfolio_books


# Explicit synchronization marker.  The first v3 validation accidentally ran
# the corrected regression tests against the pre-fix baseline implementation.
BASELINE_IMPLEMENTATION_REVISION = "causal_retry_v3_20260812"


@dataclass(frozen=True)
class BaselineConfig:
    algorithm: str
    encoder: str
    obs_dim: int
    action_dim: int
    seq_len: int
    hidden: int
    num_layers: int
    lr_actor: float
    lr_critic: float
    gamma: float
    tau: float
    entropy_coef: float
    grad_clip_norm: float
    random_exploration_steps: int
    replay_capacity: int
    policy_delay: int
    target_policy_noise: float
    target_noise_clip: float
    diagnostic_interval: int
    direction_logit_bound: float
    projection_temperature: float
    net_exposure: float
    gross_leverage: float
    max_long_weight: float
    max_short_weight: float
    initial_leverage_gate: float
    leverage_soft_target: float
    leverage_penalty_coef: float
    short_borrow_rate: float
    cash_borrow_rate: float
    utility_mode: str
    vine_observation_mode: str
    vine_feature_mode: str
    cvar_observation_mode: str
    cvar_reward_mode: str
    pretrain_data_mode: str
    run_finetune: bool
    pretrain_behavior_gate_mode: str = "strict"
    use_amp: bool = False

    def validate(self) -> None:
        if self.algorithm not in {"td3", "ddpg", "sac", "ppo", "a2c"}:
            raise RuntimeError(f"Unsupported algorithm: {self.algorithm}")
        if self.encoder not in {"lstm", "mlp"}:
            raise RuntimeError(f"Unsupported encoder: {self.encoder}")
        if self.algorithm != "td3" and self.encoder != "lstm":
            raise RuntimeError("Algorithm controls must retain the LSTM encoder.")
        if self.obs_dim < 1 or self.action_dim < 2 or self.seq_len < 1:
            raise RuntimeError("Invalid policy dimensions.")
        if self.net_exposure <= 0 or self.gross_leverage < self.net_exposure:
            raise RuntimeError("Invalid long/short mandate.")
        if self.gamma != 1.0:
            raise RuntimeError("Multi-period telescoping CRRA requires gamma=1.")
        if self.pretrain_behavior_gate_mode not in {"strict", "report_only"}:
            raise RuntimeError("Invalid pre-training behavior-gate mode.")

    @property
    def full_short_budget(self) -> float:
        return 0.5 * (self.gross_leverage - self.net_exposure)

    @property
    def short_support_size(self) -> int:
        return (int(math.ceil(self.full_short_budget / self.max_short_weight - 1e-12))
                if self.full_short_budget > 0 else 0)

    def architecture(self, *, parameter_count: int,
                     capacity_target: int | None = None) -> dict[str, Any]:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "seq_len": self.seq_len,
            "actor_output_dim": self.action_dim + 1,
            "hidden": self.hidden,
            "num_layers": self.num_layers,
            "agent": self.algorithm,
            "rl_algorithm": self.algorithm,
            "policy_encoder": self.encoder,
            "state_normalization": "layer_norm",
            "action_mode": "interior_rank_partition_leverage_gate_v5",
            "gross_leverage": self.gross_leverage,
            "net_exposure": self.net_exposure,
            "short_borrow_rate": self.short_borrow_rate,
            "cash_borrow_rate": self.cash_borrow_rate,
            "utility_mode": self.utility_mode,
            "vine_observation_mode": self.vine_observation_mode,
            "vine_feature_mode": self.vine_feature_mode,
            "cvar_observation_mode": self.cvar_observation_mode,
            "cvar_reward_mode": self.cvar_reward_mode,
            "pretrain_data_mode": self.pretrain_data_mode,
            "run_finetune": self.run_finetune,
            "pretrain_behavior_gate_mode": self.pretrain_behavior_gate_mode,
            "max_long_weight": self.max_long_weight,
            "max_short_weight": self.max_short_weight,
            "direction_logit_bound": self.direction_logit_bound,
            "projection_temperature": self.projection_temperature,
            "initial_leverage_gate": self.initial_leverage_gate,
            "allocation_entropy_coef": self.entropy_coef,
            "leverage_soft_target": self.leverage_soft_target,
            "leverage_penalty_coef": self.leverage_penalty_coef,
            "short_support_size": self.short_support_size,
            "latent_entropy_objective": self.algorithm in {"sac", "ppo", "a2c"},
            "training_update_protocol": (
                "episodic_gae_clipped_surrogate" if self.algorithm == "ppo" else
                "episodic_n_step_advantage_actor_critic" if self.algorithm == "a2c"
                else "off_policy_replay"
            ),
            "parameter_count": parameter_count,
            "capacity_target_parameter_count": capacity_target,
            "use_amp": self.use_amp,
            "checkpoint_schema": 6,
        }


class SequenceEncoder(nn.Module):
    def __init__(self, config: BaselineConfig, mlp_width: int | None = None):
        super().__init__()
        self.kind = config.encoder
        self.input_norm = nn.LayerNorm(config.obs_dim)
        if self.kind == "lstm":
            self.network = nn.LSTM(
                config.obs_dim, config.hidden, config.num_layers, batch_first=True
            )
            self.output_norm = nn.LayerNorm(config.hidden)
            self.output_dim = config.hidden
        else:
            width = int(mlp_width or config.hidden)
            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(config.seq_len * config.obs_dim, width),
                nn.ReLU(),
                nn.Linear(width, config.hidden),
                nn.ReLU(),
            )
            self.output_norm = nn.LayerNorm(config.hidden)
            self.output_dim = config.hidden

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = self.input_norm(state)
        if self.kind == "lstm":
            # deepcopy-created target networks can lose cuDNN's contiguous
            # recurrent weight layout.  Re-establish it lazily to avoid a
            # costly repack on every forward call.
            self.network.flatten_parameters()
            sequence, _ = self.network(state)
            output = sequence[:, -1, :]
        else:
            output = self.network(state)
        return self.output_norm(output)


class DeterministicActor(nn.Module):
    def __init__(self, config: BaselineConfig, mlp_width: int | None = None):
        super().__init__()
        self.encoder = SequenceEncoder(config, mlp_width)
        self.head = nn.Sequential(
            nn.Linear(config.hidden, config.hidden), nn.ReLU(),
            nn.Linear(config.hidden, config.action_dim + 1),
        )
        with torch.no_grad():
            self.head[-1].weight[-1].zero_()
            self.head[-1].bias[-1].fill_(math.log(
                config.initial_leverage_gate / (1 - config.initial_leverage_gate)))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(state))


class GaussianActor(nn.Module):
    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.encoder = SequenceEncoder(config)
        self.mean = nn.Sequential(
            nn.Linear(config.hidden, config.hidden), nn.ReLU(),
            nn.Linear(config.hidden, config.action_dim + 1),
        )
        self.log_std = nn.Parameter(torch.full((config.action_dim + 1,), -1.5))
        with torch.no_grad():
            self.mean[-1].weight[-1].zero_()
            self.mean[-1].bias[-1].fill_(math.log(
                config.initial_leverage_gate / (1 - config.initial_leverage_gate)))

    def distribution(self, state: torch.Tensor) -> Normal:
        mean = self.mean(self.encoder(state))
        log_std = self.log_std.clamp(-5.0, 1.0).expand_as(mean)
        return Normal(mean, log_std.exp())

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        return self.distribution(state).mean

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(state)
        raw = distribution.rsample()
        return raw, distribution.log_prob(raw).sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, config: BaselineConfig, mlp_width: int | None = None):
        super().__init__()
        self.encoder = SequenceEncoder(config, mlp_width)
        self.head = nn.Sequential(
            nn.Linear(config.hidden + config.action_dim, config.hidden),
            nn.ReLU(), nn.Linear(config.hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((self.encoder(state), action), dim=-1))


class ValueNetwork(nn.Module):
    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.encoder = SequenceEncoder(config)
        self.head = nn.Sequential(
            nn.Linear(config.hidden, config.hidden), nn.ReLU(), nn.Linear(config.hidden, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(state))


def count_parameters(*modules: nn.Module) -> int:
    return sum(parameter.numel() for module in modules
               for parameter in module.parameters())


def matched_mlp_width(config: BaselineConfig) -> tuple[int, int]:
    reference = BaselineConfig(**{**config.__dict__, "encoder": "lstm"})
    target = count_parameters(
        DeterministicActor(reference), Critic(reference), Critic(reference))
    def candidate_count(width: int) -> int:
        return count_parameters(
            DeterministicActor(config, width), Critic(config, width),
            Critic(config, width))

    # Parameter count is monotone in the shared MLP width.  Search every
    # integer width around the crossing rather than a coarse grid: with the
    # publication dimensions (30 x 88 observations, hidden=64), the closest
    # valid width is 15 and the former step-of-four grid skipped it entirely.
    low, high = 1, 4096
    while low <= high:
        middle = (low + high) // 2
        if candidate_count(middle) < target:
            low = middle + 1
        else:
            high = middle - 1
    candidates = sorted({max(1, min(4096, value))
                         for value in (high - 1, high, low, low + 1)})
    scored = []
    for width in candidates:
        count = candidate_count(width)
        scored.append((width, abs(count - target), count))
    best = min(scored, key=lambda item: (item[1], item[0]))
    if best[1] / target > 0.05:
        raise RuntimeError("Could not parameter-match the feedforward TD3 within 5%.")
    return int(best[0]), target


class BaseAgent:
    on_policy = False

    def __init__(self, config: BaselineConfig, device: torch.device):
        config.validate()
        self.config = config
        self.device = device
        self.action_dim = config.action_dim
        self.obs_dim = config.obs_dim
        self.total_actions = 0
        self.update_count = 0
        self.last_action_diagnostics: dict[str, float] = {}
        self.capacity_target: int | None = None

    def _state(self, state: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.asarray(state, dtype=np.float32))
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def project(self, raw: torch.Tensor) -> torch.Tensor:
        long_probs, short_probs, _, long_budget, short_budget = portfolio_books(
            raw,
            direction_logit_bound=self.config.direction_logit_bound,
            projection_temperature=self.config.projection_temperature,
            net_exposure=self.config.net_exposure,
            full_short_budget=self.config.full_short_budget,
            max_long_weight=self.config.max_long_weight,
            max_short_weight=self.config.max_short_weight,
            short_support_size=self.config.short_support_size,
        )
        return long_budget * long_probs - short_budget * short_probs

    def leverage_regularization(self, raw: torch.Tensor) -> torch.Tensor:
        """Shared soft leverage prior used by every actor objective.

        The economic projection and hard constraints remain identical across
        algorithms.  This term only prevents an algorithm-specific omission
        from making maximum gross leverage an artificial default.
        """
        gate = torch.sigmoid(raw[..., -1])
        return self.config.leverage_penalty_coef * torch.relu(
            gate - self.config.leverage_soft_target).square().mean()

    def raw_deterministic_tensor(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _diagnostics(self, raw: torch.Tensor, weights: torch.Tensor) -> None:
        long_probs, short_probs, gate, _, short_budget = portfolio_books(
            raw,
            direction_logit_bound=self.config.direction_logit_bound,
            projection_temperature=self.config.projection_temperature,
            net_exposure=self.config.net_exposure,
            full_short_budget=self.config.full_short_budget,
            max_long_weight=self.config.max_long_weight,
            max_short_weight=self.config.max_short_weight,
            short_support_size=self.config.short_support_size,
        )
        short_active = (short_budget > 1e-10).to(long_probs.dtype).squeeze(-1)
        entropy = -0.5 * (
            (long_probs * torch.log(long_probs + 1e-8)).sum(dim=-1) +
            short_active * (short_probs * torch.log(short_probs + 1e-8)).sum(dim=-1)
        )
        denominator = max(self.config.gross_leverage - abs(self.config.net_exposure), 1e-12)
        effective = ((weights.abs().sum(dim=-1, keepdim=True) -
                      abs(self.config.net_exposure)) / denominator).clamp(0, 1)
        at_cap = torch.any(
            (weights >= self.config.max_long_weight - 1e-4) |
            (weights <= -self.config.max_short_weight + 1e-4), dim=-1)
        self.last_action_diagnostics = {
            "leverage_gate": float(gate.mean().detach().cpu()),
            "effective_leverage": float(effective.mean().detach().cpu()),
            "gate_gross_error": float(torch.abs(gate - effective).mean().detach().cpu()),
            "position_at_cap": float(at_cap.float().mean().detach().cpu()),
            "direction_entropy": float(entropy.mean().detach().cpu()),
        }

    def deterministic_action(self, state: np.ndarray) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            raw = self.raw_deterministic_tensor(self._state(state))
            weights = self.project(raw)
        self.actor.train()
        return weights.cpu().numpy().reshape(-1)

    def sync_targets(self) -> None:
        return None

    def record_outcome(self, reward: float, done: bool, next_state: np.ndarray) -> None:
        return None

    def finish_episode(self) -> dict[str, float] | None:
        return None

    def parameter_count(self) -> int:
        modules = [self.actor]
        for name in ("critic", "critic2", "value"):
            if hasattr(self, name):
                modules.append(getattr(self, name))
        return count_parameters(*modules)

    @staticmethod
    def _set_learning_rate(optimizer: optim.Optimizer, learning_rate: float) -> None:
        """Restore weights/moments while enforcing the current stage's LR.

        PyTorch optimizer state dictionaries contain the pretraining learning
        rate.  Without this reset, loading a pretrained checkpoint into a
        fine-tuning agent silently discards FINETUNE_LR_ACTOR/CRITIC.
        """
        for group in optimizer.param_groups:
            group["lr"] = float(learning_rate)

    def checkpoint(self) -> dict[str, Any]:
        raise NotImplementedError

    def validate_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        architecture = checkpoint.get("architecture")
        if not isinstance(architecture, dict):
            raise RuntimeError("Checkpoint lacks architecture metadata.")
        expected = {
            "obs_dim": self.config.obs_dim,
            "action_dim": self.config.action_dim,
            "seq_len": self.config.seq_len,
            "rl_algorithm": self.config.algorithm,
            "policy_encoder": self.config.encoder,
            "vine_feature_mode": self.config.vine_feature_mode,
            "cvar_observation_mode": self.config.cvar_observation_mode,
            "cvar_reward_mode": self.config.cvar_reward_mode,
            "pretrain_data_mode": self.config.pretrain_data_mode,
            "run_finetune": self.config.run_finetune,
            "pretrain_behavior_gate_mode": self.config.pretrain_behavior_gate_mode,
            "gross_leverage": self.config.gross_leverage,
            "net_exposure": self.config.net_exposure,
            "max_long_weight": self.config.max_long_weight,
            "max_short_weight": self.config.max_short_weight,
            "checkpoint_schema": 6,
        }
        mismatches = {field: (architecture.get(field), value)
                      for field, value in expected.items()
                      if architecture.get(field) != value}
        if mismatches:
            raise RuntimeError(f"Checkpoint architecture mismatch: {mismatches}")

    def save(self, path: str) -> None:
        payload = self.checkpoint()
        payload["update_count"] = self.update_count
        payload["total_actions"] = self.total_actions
        payload["architecture"] = self.config.architecture(
            parameter_count=self.parameter_count(), capacity_target=self.capacity_target)
        torch.save(payload, path)


class DeterministicAgent(BaseAgent):
    def __init__(self, config: BaselineConfig, device: torch.device):
        super().__init__(config, device)
        mlp_width = None
        if config.encoder == "mlp":
            mlp_width, self.capacity_target = matched_mlp_width(config)
        self.actor = DeterministicActor(config, mlp_width).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(config, mlp_width).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.twin = config.algorithm == "td3"
        if self.twin:
            self.critic2 = Critic(config, mlp_width).to(device)
            self.critic2_target = copy.deepcopy(self.critic2)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        critics = list(self.critic.parameters())
        if self.twin:
            critics += list(self.critic2.parameters())
        self.critic_optimizer = optim.Adam(critics, lr=config.lr_critic)

    def raw_deterministic_tensor(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state)

    def select_action(self, state: np.ndarray, noise_scale: float = 0) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            tensor = self._state(state)
            if self.total_actions < self.config.random_exploration_steps:
                raw = torch.randn((tensor.shape[0], self.action_dim + 1), device=self.device)
            else:
                raw = self.actor(tensor)
            if noise_scale:
                raw = raw + noise_scale * torch.randn_like(raw)
            weights = self.project(raw)
            self._diagnostics(raw, weights)
        self.actor.train()
        self.total_actions += 1
        return weights.cpu().numpy().reshape(-1)

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
            target_parameter.data.mul_(1 - self.config.tau).add_(
                source_parameter.data, alpha=self.config.tau)

    def sync_targets(self) -> None:
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        if self.twin:
            self.critic2_target.load_state_dict(self.critic2.state_dict())

    def update(self, replay_buffer: Any, batch_size: int = 32) -> dict[str, float] | None:
        if len(replay_buffer) < batch_size:
            return None
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        with torch.no_grad():
            next_raw = self.actor_target(next_states)
            if self.twin:
                noise = (torch.randn_like(next_raw) * self.config.target_policy_noise).clamp(
                    -self.config.target_noise_clip, self.config.target_noise_clip)
                next_raw = next_raw + noise
            next_action = self.project(next_raw)
            target_q = self.critic_target(next_states, next_action)
            if self.twin:
                target_q = torch.minimum(
                    target_q, self.critic2_target(next_states, next_action))
            target_q = rewards + (1 - dones) * self.config.gamma * target_q
        q1 = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(q1, target_q)
        q2 = None
        if self.twin:
            q2 = self.critic2(states, actions)
            critic_loss = critic_loss + nn.functional.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad = nn.utils.clip_grad_norm_(
            list(self.critic.parameters()) +
            (list(self.critic2.parameters()) if self.twin else []),
            self.config.grad_clip_norm)
        self.critic_optimizer.step()
        self.update_count += 1
        actor_loss_value = float("nan")
        if not self.twin or self.update_count % self.config.policy_delay == 0:
            raw = self.actor(states)
            long_probs, short_probs, _, long_budget, short_budget = portfolio_books(
                raw,
                direction_logit_bound=self.config.direction_logit_bound,
                projection_temperature=self.config.projection_temperature,
                net_exposure=self.config.net_exposure,
                full_short_budget=self.config.full_short_budget,
                max_long_weight=self.config.max_long_weight,
                max_short_weight=self.config.max_short_weight,
                short_support_size=self.config.short_support_size,
            )
            action = long_budget * long_probs - short_budget * short_probs
            actor_loss = -self.critic(states, action).mean()
            if self.config.entropy_coef:
                short_active = (short_budget > 1e-10).to(long_probs.dtype)
                direction_entropy = -0.5 * (
                    (long_probs * torch.log(long_probs + 1e-8)).sum(dim=-1) +
                    short_active.squeeze(-1) *
                    (short_probs * torch.log(short_probs + 1e-8)).sum(dim=-1)
                )
                actor_loss = actor_loss - self.config.entropy_coef * \
                    direction_entropy.mean()
            if self.config.leverage_penalty_coef:
                actor_loss = actor_loss + self.leverage_regularization(raw)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.detach().cpu())
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)
            if self.twin:
                self._soft_update(self.critic2_target, self.critic2)
        if self.update_count % self.config.diagnostic_interval:
            return None
        return {
            "update": self.update_count,
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": actor_loss_value,
            "q1_mean": float(q1.mean().detach().cpu()),
            "q2_mean": float(q2.mean().detach().cpu()) if q2 is not None else None,
            "target_q_mean": float(target_q.mean().detach().cpu()),
            "twin_q_gap": (float(torch.abs(q1 - q2).mean().detach().cpu())
                           if q2 is not None else None),
            "critic_grad_norm": float(critic_grad.detach().cpu()),
        }

    def checkpoint(self) -> dict[str, Any]:
        payload = {
            "actor": self.actor.state_dict(), "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(), "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
        if self.twin:
            payload.update({"critic2": self.critic2.state_dict(),
                            "critic2_target": self.critic2_target.state_dict()})
        return payload

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.validate_checkpoint(checkpoint)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if self.twin:
            self.critic2.load_state_dict(checkpoint["critic2"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self._set_learning_rate(self.actor_optimizer, self.config.lr_actor)
        self._set_learning_rate(self.critic_optimizer, self.config.lr_critic)
        self.update_count = int(checkpoint.get("update_count", 0))
        self.total_actions = int(checkpoint.get("total_actions", 0))
        self.sync_targets()


class SACAgent(BaseAgent):
    def __init__(self, config: BaselineConfig, device: torch.device):
        super().__init__(config, device)
        self.actor = GaussianActor(config).to(device)
        self.critic = Critic(config).to(device)
        self.critic2 = Critic(config).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_optimizer = optim.Adam(
            list(self.critic.parameters()) + list(self.critic2.parameters()),
            lr=config.lr_critic)
        self.log_alpha = torch.tensor(
            math.log(max(config.entropy_coef, 1e-4)), device=device,
            requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.lr_actor)
        self.target_entropy = -float(config.action_dim + 1)

    def raw_deterministic_tensor(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor.deterministic(state)

    def select_action(self, state: np.ndarray, noise_scale: float = 0) -> np.ndarray:
        with torch.no_grad():
            tensor = self._state(state)
            if self.total_actions < self.config.random_exploration_steps:
                raw = torch.randn((tensor.shape[0], self.action_dim + 1), device=self.device)
            else:
                raw, _ = self.actor.sample(tensor)
            weights = self.project(raw)
            self._diagnostics(raw, weights)
        self.total_actions += 1
        return weights.cpu().numpy().reshape(-1)

    def sync_targets(self) -> None:
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    def update(self, replay_buffer: Any, batch_size: int = 32) -> dict[str, float] | None:
        if len(replay_buffer) < batch_size:
            return None
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        alpha = self.log_alpha.exp()
        with torch.no_grad():
            next_raw, next_log_prob = self.actor.sample(next_states)
            next_action = self.project(next_raw)
            next_q = torch.minimum(
                self.critic_target(next_states, next_action),
                self.critic2_target(next_states, next_action)) - alpha * next_log_prob
            target = rewards + (1 - dones) * self.config.gamma * next_q
        q1, q2 = self.critic(states, actions), self.critic2(states, actions)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(); critic_loss.backward()
        critic_grad = nn.utils.clip_grad_norm_(
            list(self.critic.parameters()) + list(self.critic2.parameters()),
            self.config.grad_clip_norm)
        self.critic_optimizer.step()
        raw, log_prob = self.actor.sample(states)
        policy_action = self.project(raw)
        actor_loss = (alpha.detach() * log_prob - torch.minimum(
            self.critic(states, policy_action), self.critic2(states, policy_action))).mean()
        if self.config.leverage_penalty_coef:
            actor_loss = actor_loss + self.leverage_regularization(raw)
        self.actor_optimizer.zero_grad(); actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
        self.actor_optimizer.step()
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(); alpha_loss.backward(); self.alpha_optimizer.step()
        self.update_count += 1
        for target_network, source_network in ((self.critic_target, self.critic),
                                               (self.critic2_target, self.critic2)):
            for target_parameter, source_parameter in zip(
                    target_network.parameters(), source_network.parameters()):
                target_parameter.data.mul_(1 - self.config.tau).add_(
                    source_parameter.data, alpha=self.config.tau)
        if self.update_count % self.config.diagnostic_interval:
            return None
        return {"update": self.update_count,
                "critic_loss": float(critic_loss.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "q1_mean": float(q1.mean().detach().cpu()),
                "q2_mean": float(q2.mean().detach().cpu()),
                "target_q_mean": float(target.mean().detach().cpu()),
                "twin_q_gap": float(torch.abs(q1 - q2).mean().detach().cpu()),
                "critic_grad_norm": float(critic_grad.detach().cpu()),
                "entropy_temperature": float(alpha.detach().cpu())}

    def checkpoint(self) -> dict[str, Any]:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "critic2": self.critic2.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "critic2_target": self.critic2_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "log_alpha": self.log_alpha.detach(),
                "alpha_optimizer": self.alpha_optimizer.state_dict()}

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.validate_checkpoint(checkpoint)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.log_alpha.data.copy_(checkpoint["log_alpha"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        self._set_learning_rate(self.actor_optimizer, self.config.lr_actor)
        self._set_learning_rate(self.critic_optimizer, self.config.lr_critic)
        self._set_learning_rate(self.alpha_optimizer, self.config.lr_actor)
        self.update_count = int(checkpoint.get("update_count", 0))
        self.total_actions = int(checkpoint.get("total_actions", 0))
        self.sync_targets()


class PPOAgent(BaseAgent):
    on_policy = True

    def __init__(self, config: BaselineConfig, device: torch.device):
        super().__init__(config, device)
        self.actor = GaussianActor(config).to(device)
        self.value = ValueNetwork(config).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=config.lr_critic)
        self.rollout: list[dict[str, Any]] = []
        self.pending: dict[str, Any] | None = None
        self.clip_ratio = 0.2
        self.gae_lambda = 0.95
        self.epochs = 10 if config.algorithm == "ppo" else 1

    def raw_deterministic_tensor(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor.deterministic(state)

    def select_action(self, state: np.ndarray, noise_scale: float = 0) -> np.ndarray:
        tensor = self._state(state)
        with torch.no_grad():
            distribution = self.actor.distribution(tensor)
            raw = distribution.sample()
            log_prob = distribution.log_prob(raw).sum(dim=-1)
            value = self.value(tensor).squeeze(-1)
            weights = self.project(raw)
            self._diagnostics(raw, weights)
        self.pending = {"state": tensor.squeeze(0).cpu(), "raw": raw.squeeze(0).cpu(),
                        "log_prob": log_prob.squeeze(0).cpu(),
                        "value": value.squeeze(0).cpu()}
        self.total_actions += 1
        return weights.cpu().numpy().reshape(-1)

    def record_outcome(self, reward: float, done: bool, next_state: np.ndarray) -> None:
        if self.pending is None:
            raise RuntimeError("PPO outcome has no pending action.")
        self.pending.update({"reward": float(reward), "done": bool(done)})
        self.rollout.append(self.pending)
        self.pending = None

    def update(self, replay_buffer: Any, batch_size: int = 32) -> None:
        return None

    def finish_episode(self) -> dict[str, float] | None:
        if not self.rollout:
            return None
        rewards = torch.tensor([row["reward"] for row in self.rollout], device=self.device)
        # Arithmetic masks must be floating-point.  Newer PyTorch releases
        # correctly reject ``1 - bool_tensor``; making the dtype explicit also
        # keeps PPO/A2C return calculations consistent with replay agents.
        dones = torch.tensor([row["done"] for row in self.rollout],
                             dtype=torch.float32, device=self.device)
        old_values = torch.stack([row["value"] for row in self.rollout]).to(self.device)
        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0, device=self.device)
        next_value = torch.tensor(0.0, device=self.device)
        for index in reversed(range(len(self.rollout))):
            delta = rewards[index] + self.config.gamma * next_value * (1 - dones[index]) - old_values[index]
            gae = delta + self.config.gamma * self.gae_lambda * (1 - dones[index]) * gae
            advantages[index] = gae
            next_value = old_values[index]
        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / \
            advantages.std(unbiased=False).clamp_min(1e-8)
        states = torch.stack([row["state"] for row in self.rollout]).to(self.device)
        raws = torch.stack([row["raw"] for row in self.rollout]).to(self.device)
        old_log_probs = torch.stack([row["log_prob"] for row in self.rollout]).to(self.device)
        actor_loss = value_loss = entropy = torch.tensor(float("nan"), device=self.device)
        for _ in range(self.epochs):
            distribution = self.actor.distribution(states)
            log_prob = distribution.log_prob(raws).sum(dim=-1)
            entropy = distribution.entropy().sum(dim=-1).mean()
            if self.config.algorithm == "ppo":
                ratio = torch.exp(log_prob - old_log_probs)
                surrogate = torch.minimum(
                    ratio * advantages,
                    ratio.clamp(1 - self.clip_ratio, 1 + self.clip_ratio) * advantages)
                actor_loss = -surrogate.mean() - self.config.entropy_coef * entropy
            else:
                actor_loss = -(log_prob * advantages.detach()).mean() - \
                    self.config.entropy_coef * entropy
            if self.config.leverage_penalty_coef:
                # Stored actions are constants.  Penalize the current policy
                # mean so the regularizer has a gradient for PPO/A2C.
                actor_loss = actor_loss + self.leverage_regularization(
                    distribution.mean)
            self.actor_optimizer.zero_grad(); actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
            self.actor_optimizer.step()
            value_loss = nn.functional.mse_loss(self.value(states).squeeze(-1), returns)
            self.value_optimizer.zero_grad(); value_loss.backward()
            nn.utils.clip_grad_norm_(self.value.parameters(), self.config.grad_clip_norm)
            self.value_optimizer.step()
        self.rollout.clear()
        self.update_count += 1
        if self.update_count % max(1, self.config.diagnostic_interval // 24):
            return None
        return {"update": self.update_count,
                "critic_loss": float(value_loss.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "policy_entropy": float(entropy.detach().cpu())}

    def checkpoint(self) -> dict[str, Any]:
        return {"actor": self.actor.state_dict(), "value": self.value.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict()}

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.validate_checkpoint(checkpoint)
        self.actor.load_state_dict(checkpoint["actor"])
        self.value.load_state_dict(checkpoint["value"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        self._set_learning_rate(self.actor_optimizer, self.config.lr_actor)
        self._set_learning_rate(self.value_optimizer, self.config.lr_critic)
        self.update_count = int(checkpoint.get("update_count", 0))
        self.total_actions = int(checkpoint.get("total_actions", 0))


def build_agent(config: BaselineConfig, device: torch.device) -> BaseAgent:
    config.validate()
    if config.algorithm in {"td3", "ddpg"}:
        return DeterministicAgent(config, device)
    if config.algorithm == "sac":
        return SACAgent(config, device)
    return PPOAgent(config, device)
