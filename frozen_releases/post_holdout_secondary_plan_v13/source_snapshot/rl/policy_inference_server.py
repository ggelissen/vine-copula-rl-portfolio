#!/usr/bin/env python3
"""Isolated CPU policy inference server for the R historical evaluator.

R torch/Lantern and Python PyTorch bundle incompatible libtorch builds on some
systems.  This process boundary keeps them in separate address spaces while
preserving the exact checkpoint actor and action projection.
"""

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
import torch.nn as nn

try:
    from rl.checkpoint_attestation import resolve_architecture_mode
except ModuleNotFoundError:  # Direct execution as ``python rl/policy_inference_server.py``.
    from checkpoint_attestation import resolve_architecture_mode


def environment_float(name: str, default: str) -> float:
    return float(os.environ.get(name, default))


def environment_int(name: str, default: str) -> int:
    return int(os.environ.get(name, default))


def environment_bool(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes"}


class LSTMActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int, num_layers: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(obs_dim)
        self.lstm = nn.LSTM(obs_dim, hidden, num_layers, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, action_dim + 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(self.input_norm(state))
        return self.fc(self.layernorm(sequence[:, -1, :]))


def runtime_contract(obs_dim: int, action_dim: int) -> tuple[dict[str, object], dict[str, object]]:
    hidden = environment_int("HIDDEN", "128")
    num_layers = environment_int("NUM_LAYERS", "2")
    gross_leverage = environment_float("ENV_GROSS_LEVERAGE", "1.5")
    net_exposure = environment_float("ENV_NET_EXPOSURE", "1.0")
    max_long = environment_float("ENV_MAX_LONG_WEIGHT", "0.60")
    max_short = environment_float("ENV_MAX_SHORT_WEIGHT", "0.20")
    short_rate = environment_float("ENV_SHORT_BORROW_RATE", "0.03")
    cash_rate = environment_float("ENV_CASH_BORROW_RATE", "0.02")
    utility_mode = os.environ.get("ENV_UTILITY_MODE", "terminal_wealth_crra")
    vine_mode = os.environ.get("VINE_OBSERVATION_MODE", "full")
    direction_bound = environment_float("DIRECTION_LOGIT_BOUND", "1.0")
    temperature = environment_float("PROJECTION_TEMPERATURE", "1.5")
    initial_gate = environment_float("INITIAL_LEVERAGE_GATE", "0.10")
    entropy_coef = environment_float("ENTROPY_COEF", "0.005")
    leverage_target = environment_float("LEVERAGE_SOFT_TARGET", "0.80")
    leverage_penalty = environment_float("LEVERAGE_PENALTY_COEF", "0.25")
    use_amp = environment_bool("USE_AMP")
    if net_exposure <= 0 or gross_leverage < net_exposure:
        raise RuntimeError("Schema-5 actions require positive net exposure and gross >= net.")
    full_short_budget = 0.5 * (gross_leverage - net_exposure)
    short_support_size = (
        int(math.ceil(full_short_budget / max_short - 1e-12))
        if full_short_budget > 0 else 0
    )
    expected: dict[str, object] = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "actor_output_dim": action_dim + 1,
        "hidden": hidden,
        "num_layers": num_layers,
        "agent": "td3",
        "state_normalization": "layer_norm",
        "action_mode": "interior_rank_partition_leverage_gate_v5",
        "gross_leverage": gross_leverage,
        "net_exposure": net_exposure,
        "short_borrow_rate": short_rate,
        "cash_borrow_rate": cash_rate,
        "utility_mode": utility_mode,
        "vine_observation_mode": vine_mode,
        "max_long_weight": max_long,
        "max_short_weight": max_short,
        "direction_logit_bound": direction_bound,
        "projection_temperature": temperature,
        "initial_leverage_gate": initial_gate,
        "allocation_entropy_coef": entropy_coef,
        "leverage_soft_target": leverage_target,
        "leverage_penalty_coef": leverage_penalty,
        "short_support_size": short_support_size,
        "use_amp": use_amp,
        "checkpoint_schema": 5,
    }
    if vine_mode == "zero":
        expected["no_vine_signal_mask"] = "explicit_vine_and_scenario_cvar_v1"
    projection = {
        "direction_logit_bound": direction_bound,
        "projection_temperature": temperature,
        "net_exposure": net_exposure,
        "full_short_budget": full_short_budget,
        "max_long_weight": max_long,
        "max_short_weight": max_short,
        "short_support_size": short_support_size,
    }
    return expected, projection


def load_actor(
    checkpoint_path: Path, obs_dim: int, action_dim: int
) -> tuple[LSTMActor, dict[str, object], str]:
    expected, projection = runtime_contract(obs_dim, action_dim)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture")
    if architecture is None:
        raise RuntimeError("Checkpoint predates mandatory architecture metadata.")
    actual, mode_metadata_source = resolve_architecture_mode(
        checkpoint_path, dict(architecture), str(expected["vine_observation_mode"])
    )
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Checkpoint architecture mismatch: {mismatches}")
    actor = LSTMActor(
        obs_dim, action_dim, int(expected["hidden"]), int(expected["num_layers"])
    )
    state = {
        key.replace("_orig_mod.", ""): value
        for key, value in checkpoint["actor"].items()
    }
    actor.load_state_dict(state)
    actor.eval()
    return actor, projection, mode_metadata_source


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def run_server(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from rl.action_projection import portfolio_books

    ipc = args.ipc_dir.resolve()
    ipc.mkdir(parents=True, exist_ok=True)
    actor, projection, mode_metadata_source = load_actor(
        args.checkpoint.resolve(), args.obs_dim, args.action_dim
    )
    ready = {
        "python": sys.executable,
        "torch": torch.__version__,
        "checkpoint": str(args.checkpoint.resolve()),
        "obs_dim": args.obs_dim,
        "action_dim": args.action_dim,
        "protocol": "file_ipc_isolated_libtorch_v1",
        "mode_metadata_source": mode_metadata_source,
    }
    atomic_text(ipc / "READY.json", json.dumps(ready, sort_keys=True) + "\n")
    handled: set[str] = set()
    while True:
        if (ipc / "STOP").exists():
            atomic_text(ipc / "DONE", "ok\n")
            return
        for request in sorted(ipc.glob("request_*.csv")):
            if request.name in handled:
                continue
            state = np.loadtxt(request, delimiter=",", dtype=np.float32, ndmin=2)
            if state.shape != (args.seq_len, args.obs_dim):
                raise RuntimeError(
                    f"Invalid state shape {state.shape}; expected {(args.seq_len, args.obs_dim)}"
                )
            with torch.no_grad():
                raw_action = actor(torch.from_numpy(state).unsqueeze(0)).squeeze(0)
                long_probs, short_probs, _, long_budget, short_budget = portfolio_books(
                    raw_action, **projection
                )
                action = (long_budget * long_probs - short_budget * short_probs).numpy()
            response = ipc / request.name.replace("request_", "response_")
            temporary = response.with_suffix(response.suffix + ".tmp")
            np.savetxt(temporary, action.reshape(1, -1), delimiter=",", fmt="%.17g")
            os.replace(temporary, response)
            handled.add(request.name)
            request.unlink(missing_ok=True)
        time.sleep(0.01)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--ipc-dir", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--obs-dim", required=True, type=int)
    parser.add_argument("--action-dim", required=True, type=int)
    parser.add_argument("--seq-len", required=True, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_server(args)
    except Exception:
        args.ipc_dir.mkdir(parents=True, exist_ok=True)
        atomic_text(args.ipc_dir / "ERROR.txt", traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
