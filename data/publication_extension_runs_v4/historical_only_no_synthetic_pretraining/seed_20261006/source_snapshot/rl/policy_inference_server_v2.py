#!/usr/bin/env python3
"""Isolated inference for schema-5 TD3 and schema-6 ablation checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


def runtime_value(name: str, default: str, cast=float):
    return cast(os.environ.get(name, default))


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def load_policy(checkpoint_path: Path, obs_dim: int, action_dim: int,
                seq_len: int, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, dict):
        raise RuntimeError("Checkpoint lacks mandatory architecture metadata.")
    schema = int(architecture.get("checkpoint_schema", 0))
    if architecture.get("obs_dim") != obs_dim or architecture.get("action_dim") != action_dim:
        raise RuntimeError("Checkpoint and evaluation environment dimensions differ.")
    if schema == 5:
        from rl.policy_inference_server import LSTMActor
        actor = LSTMActor(obs_dim, action_dim, int(architecture["hidden"]),
                          int(architecture["num_layers"]))
        actor.load_state_dict({key.replace("_orig_mod.", ""): value
                               for key, value in checkpoint["actor"].items()})
        raw_action = lambda state: actor(state)
    elif schema == 6:
        from rl.recurrent_baselines import (
            BaselineConfig, DeterministicActor, GaussianActor
        )
        encoder = str(architecture["policy_encoder"])
        algorithm = str(architecture["rl_algorithm"])
        config = BaselineConfig(
            algorithm=algorithm, encoder=encoder, obs_dim=obs_dim,
            action_dim=action_dim, seq_len=seq_len,
            hidden=int(architecture["hidden"]),
            num_layers=int(architecture["num_layers"]),
            lr_actor=1e-4, lr_critic=1e-4, gamma=1.0, tau=0.005,
            entropy_coef=float(architecture.get("allocation_entropy_coef", 0.0)),
            grad_clip_norm=1.0, random_exploration_steps=0,
            replay_capacity=1, policy_delay=2, target_policy_noise=0.2,
            target_noise_clip=0.5, diagnostic_interval=100,
            direction_logit_bound=float(architecture["direction_logit_bound"]),
            projection_temperature=float(architecture["projection_temperature"]),
            net_exposure=float(architecture["net_exposure"]),
            gross_leverage=float(architecture["gross_leverage"]),
            max_long_weight=float(architecture["max_long_weight"]),
            max_short_weight=float(architecture["max_short_weight"]),
            initial_leverage_gate=float(architecture["initial_leverage_gate"]),
            leverage_soft_target=float(architecture["leverage_soft_target"]),
            leverage_penalty_coef=float(architecture["leverage_penalty_coef"]),
            short_borrow_rate=float(architecture["short_borrow_rate"]),
            cash_borrow_rate=float(architecture["cash_borrow_rate"]),
            utility_mode=str(architecture["utility_mode"]),
            vine_observation_mode=str(architecture["vine_observation_mode"]),
            vine_feature_mode=str(architecture["vine_feature_mode"]),
            cvar_observation_mode=str(architecture["cvar_observation_mode"]),
            cvar_reward_mode=str(architecture["cvar_reward_mode"]),
            pretrain_data_mode=str(architecture["pretrain_data_mode"]),
            pretrain_behavior_gate_mode=str(
                architecture.get("pretrain_behavior_gate_mode", "strict")),
            run_finetune=bool(architecture["run_finetune"]),
            use_amp=bool(architecture.get("use_amp", False)))
        if encoder == "mlp":
            state = checkpoint["actor"]
            width = int(state["encoder.network.1.weight"].shape[0])
        else:
            width = None
        if algorithm in {"td3", "ddpg"}:
            actor = DeterministicActor(config, width)
            raw_action = lambda state: actor(state)
        else:
            actor = GaussianActor(config)
            raw_action = lambda state: actor.deterministic(state)
        actor.load_state_dict(checkpoint["actor"])
    else:
        raise RuntimeError(f"Unsupported checkpoint schema: {schema}")
    actor.to(device).eval()
    expected_modes = {
        "VINE_FEATURE_MODE": architecture.get(
            "vine_feature_mode", architecture.get("vine_observation_mode", "full")),
        "CVAR_OBSERVATION_MODE": architecture.get(
            "cvar_observation_mode", architecture.get("vine_observation_mode", "full")),
        "CVAR_REWARD_MODE": architecture.get("cvar_reward_mode", "full"),
    }
    for name, checkpoint_value in expected_modes.items():
        runtime = os.environ.get(name, str(checkpoint_value))
        if runtime != checkpoint_value:
            raise RuntimeError(f"{name}={runtime} disagrees with checkpoint {checkpoint_value}.")
    projection = {
        "direction_logit_bound": float(architecture["direction_logit_bound"]),
        "projection_temperature": float(architecture["projection_temperature"]),
        "net_exposure": float(architecture["net_exposure"]),
        "full_short_budget": 0.5 * (float(architecture["gross_leverage"]) -
                                    float(architecture["net_exposure"])),
        "max_long_weight": float(architecture["max_long_weight"]),
        "max_short_weight": float(architecture["max_short_weight"]),
        "short_support_size": int(architecture["short_support_size"]),
    }
    return raw_action, projection, architecture


def run_server(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from rl.action_projection import portfolio_books
    ipc = args.ipc_dir.resolve(); ipc.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    policy, projection, architecture = load_policy(
        args.checkpoint.resolve(), args.obs_dim, args.action_dim, args.seq_len, device)
    ready = {"python": sys.executable, "torch": torch.__version__,
             "checkpoint": str(args.checkpoint.resolve()), "obs_dim": args.obs_dim,
             "action_dim": args.action_dim,
             "checkpoint_schema": architecture["checkpoint_schema"],
             "rl_algorithm": architecture.get("rl_algorithm", "td3"),
             "policy_encoder": architecture.get("policy_encoder", "lstm"),
             "protocol": "file_ipc_isolated_libtorch_v2"}
    atomic_text(ipc / "READY.json", json.dumps(ready, sort_keys=True) + "\n")
    handled: set[str] = set()
    while True:
        if (ipc / "STOP").exists():
            atomic_text(ipc / "DONE", "ok\n"); return
        for request in sorted(ipc.glob("request_*.csv")):
            if request.name in handled:
                continue
            state = np.loadtxt(request, delimiter=",", dtype=np.float32, ndmin=2)
            if state.shape != (args.seq_len, args.obs_dim):
                raise RuntimeError(f"Invalid state shape {state.shape}.")
            with torch.no_grad():
                raw = policy(torch.from_numpy(state).unsqueeze(0)).squeeze(0)
                long_probs, short_probs, _, long_budget, short_budget = portfolio_books(
                    raw, **projection)
                action = (long_budget * long_probs - short_budget * short_probs).numpy()
            response = ipc / request.name.replace("request_", "response_")
            temporary = response.with_suffix(response.suffix + ".tmp")
            np.savetxt(temporary, action.reshape(1, -1), delimiter=",", fmt="%.17g")
            os.replace(temporary, response)
            handled.add(request.name); request.unlink(missing_ok=True)
        time.sleep(0.01)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--ipc-dir", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--obs-dim", required=True, type=int)
    parser.add_argument("--action-dim", required=True, type=int)
    parser.add_argument("--seq-len", required=True, type=int)
    args = parser.parse_args()
    try:
        run_server(args)
    except Exception:
        args.ipc_dir.mkdir(parents=True, exist_ok=True)
        atomic_text(args.ipc_dir / "ERROR.txt", traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
