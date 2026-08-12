from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from rl.recurrent_baselines import BaselineConfig, build_agent


def configuration(algorithm: str, encoder: str = "lstm") -> BaselineConfig:
    return BaselineConfig(
        algorithm=algorithm, encoder=encoder, obs_dim=18, action_dim=7,
        seq_len=6, hidden=32, num_layers=2, lr_actor=1e-4,
        lr_critic=1e-4, gamma=1.0, tau=0.005, entropy_coef=0.005,
        grad_clip_norm=1.0, random_exploration_steps=0,
        replay_capacity=100, policy_delay=2, target_policy_noise=0.2,
        target_noise_clip=0.5, diagnostic_interval=10,
        direction_logit_bound=1.0, projection_temperature=1.5,
        net_exposure=1.0, gross_leverage=1.5, max_long_weight=0.6,
        max_short_weight=0.2, initial_leverage_gate=0.1,
        leverage_soft_target=0.8, leverage_penalty_coef=0.25,
        short_borrow_rate=0.03, cash_borrow_rate=0.02,
        utility_mode="terminal_wealth_crra", vine_observation_mode="full",
        vine_feature_mode="full", cvar_observation_mode="full",
        cvar_reward_mode="full", pretrain_data_mode="vine_synthetic",
        run_finetune=True, use_amp=False)


@pytest.mark.parametrize("algorithm", ["ddpg", "sac", "ppo", "a2c"])
def test_algorithm_controls_respect_identical_constraints(algorithm: str) -> None:
    torch.manual_seed(12)
    agent = build_agent(configuration(algorithm), torch.device("cpu"))
    state = np.zeros((6, 18), dtype=np.float32)
    action = agent.select_action(state)
    assert action.shape == (7,)
    assert np.isfinite(action).all()
    assert abs(action.sum() - 1.0) <= 1e-5
    assert np.abs(action).sum() <= 1.5 + 1e-5
    assert action.max() <= 0.6 + 1e-5
    assert action.min() >= -0.2 - 1e-5


def test_feedforward_td3_is_parameter_matched() -> None:
    agent = build_agent(configuration("td3", "mlp"), torch.device("cpu"))
    assert agent.capacity_target is not None
    assert abs(agent.parameter_count() - agent.capacity_target) / agent.capacity_target <= 0.05


def test_feedforward_match_covers_publication_dimensions() -> None:
    base = configuration("td3", "mlp")
    publication = BaselineConfig(**{
        **base.__dict__, "obs_dim": 88, "seq_len": 30,
        "hidden": 64, "num_layers": 1,
    })
    agent = build_agent(publication, torch.device("cpu"))
    assert agent.capacity_target is not None
    assert abs(agent.parameter_count() - agent.capacity_target) / agent.capacity_target <= 0.05


@pytest.mark.parametrize("algorithm", ["ppo", "a2c"])
def test_on_policy_episode_uses_numeric_terminal_masks(algorithm: str) -> None:
    torch.manual_seed(17)
    agent = build_agent(configuration(algorithm), torch.device("cpu"))
    state = np.zeros((6, 18), dtype=np.float32)
    for index in range(3):
        agent.select_action(state)
        agent.record_outcome(0.01 * (index + 1), index == 2, state)
    diagnostics = agent.finish_episode()
    assert diagnostics is not None
    assert np.isfinite(diagnostics["actor_loss"])
    assert np.isfinite(diagnostics["critic_loss"])


def test_shared_leverage_regularizer_has_actor_gradient() -> None:
    agent = build_agent(configuration("sac"), torch.device("cpu"))
    raw = torch.zeros((4, 8), requires_grad=True)
    with torch.no_grad():
        raw[:, -1] = 5.0
    penalty = agent.leverage_regularization(raw)
    penalty.backward()
    assert penalty.item() > 0
    assert raw.grad is not None
    assert torch.all(raw.grad[:, -1] > 0)


@pytest.mark.parametrize("algorithm", ["ddpg", "sac", "ppo", "a2c"])
def test_loading_pretrained_state_preserves_finetuning_learning_rates(
        tmp_path, algorithm: str) -> None:
    pretrain = configuration(algorithm=algorithm, encoder="lstm")
    first = build_agent(pretrain, torch.device("cpu"))
    path = tmp_path / f"{algorithm}.pt"
    first.save(str(path))
    fine = BaselineConfig(**{
        **pretrain.__dict__, "lr_actor": 2e-5, "lr_critic": 1e-4,
    })
    restored = build_agent(fine, torch.device("cpu"))
    restored.load(str(path))
    assert {group["lr"] for group in restored.actor_optimizer.param_groups} == {2e-5}
    critic_optimizer = (restored.value_optimizer if algorithm in {"ppo", "a2c"}
                        else restored.critic_optimizer)
    assert {group["lr"] for group in critic_optimizer.param_groups} == {1e-4}
