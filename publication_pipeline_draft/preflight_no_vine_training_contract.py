#!/usr/bin/env python3
"""Fail-closed source/config preflight for the clean no-vine retraining sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml


EXPECTED_SEEDS = list(range(20260841, 20260851))
EXPECTED_MODE = "zero"
EXPECTED_MASK = "explicit_vine_and_scenario_cvar_v1"


class PreflightError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(pattern: str, text: str, label: str, *, fixed: bool = False) -> None:
    matched = pattern in text if fixed else re.search(pattern, text, re.MULTILINE) is not None
    if not matched:
        raise PreflightError(f"Missing no-vine training contract: {label}")


def preflight(repo_root: Path, expected_sweep_root: Path) -> dict[str, object]:
    files = {
        "seed_specification": repo_root / "config/no_vine_ablation_seeds.yaml",
        "secondary_contract": repo_root / "publication_pipeline_draft/config/secondary_experiments_v1.json",
        "runner": repo_root / "rl/run_seed_sweep.r",
        "launcher": repo_root / "run_with_config.r",
        "trainer": repo_root / "rl/train_rl.r",
        "environment": repo_root / "rl/rl_environment.r",
        "manifest_writer": repo_root / "helper/reproducibility.r",
        "hpc_launcher": repo_root / "hpc/run_no_vine_4gpu.sh",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise PreflightError(f"Missing source files: {', '.join(missing)}")

    seed_spec = yaml.safe_load(files["seed_specification"].read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in seed_spec.get("seeds", [])]
    mode = str(seed_spec.get("vine_observation_mode", "")).strip().lower()
    if seeds != EXPECTED_SEEDS or mode != EXPECTED_MODE:
        raise PreflightError(
            f"Seed specification mismatch: mode={mode!r}, seeds={seeds!r}"
        )

    contract = json.loads(files["secondary_contract"].read_text(encoding="utf-8"))
    experiments = {
        item["experiment_id"]: item for item in contract.get("experiments", [])
    }
    experiment = experiments.get("no_vine_td3")
    if experiment is None:
        raise PreflightError("Secondary contract lacks no_vine_td3")
    declared_root = (repo_root / experiment["sweep_root"]).resolve()
    if declared_root != expected_sweep_root.resolve():
        raise PreflightError(
            f"Sweep-root mismatch: contract={declared_root}, launcher={expected_sweep_root.resolve()}"
        )
    if experiment.get("vine_observation_mode") != EXPECTED_MODE:
        raise PreflightError("Secondary contract does not declare zero mode")
    if experiment.get("signal_mask") != EXPECTED_MASK:
        raise PreflightError("Secondary contract signal mask is not frozen")

    text = {key: path.read_text(encoding="utf-8") for key, path in files.items()}
    require("VINE_OBSERVATION_MODE=", text["runner"], "runner exports child mode", fixed=True)
    require(r"env\s*=\s*environment", text["runner"], "runner passes system2 environment")
    require(
        r"Sys\.getenv\s*\(\s*[\"']VINE_OBSERVATION_MODE[\"']",
        text["trainer"],
        "trainer reads process mode",
    )
    require(
        r"vine_observation_mode\s*=\s*vine_observation_mode",
        text["trainer"],
        "trainer passes mode to RLEnvironment",
    )
    require("vine_observation_mode.txt", text["trainer"], "trainer writes redundant mode marker", fixed=True)
    require("'vine_observation_mode': VINE_OBSERVATION_MODE", text["trainer"], "checkpoint embeds mode", fixed=True)
    require("'no_vine_signal_mask': NO_VINE_SIGNAL_MASK", text["trainer"], "checkpoint embeds signal mask", fixed=True)
    require(
        r"no_vine_observation\s*<-\s*identical\s*\(\s*private\$vine_observation_mode\s*,\s*[\"']zero[\"']\s*\)",
        text["environment"],
        "environment activates zero mode",
    )
    require(
        r"vine_observation\s*<-\s*if\s*\(\s*no_vine_observation\s*\)",
        text["environment"],
        "environment zeros direct vine channel",
    )
    require(
        r"cvar_observation\s*<-\s*if\s*\(\s*no_vine_observation\s*\)\s*0",
        text["environment"],
        "environment zeros indirect CVaR channel",
    )
    require('"VINE_OBSERVATION_MODE"', text["manifest_writer"], "manifest records mode", fixed=True)
    require("parse(file='rl/train_rl.r')", text["hpc_launcher"], "HPC launcher parses trainer before launch", fixed=True)

    return {
        "schema_version": 1,
        "status": "clean_no_vine_retraining_contract_passed",
        "expected_mode": EXPECTED_MODE,
        "expected_signal_mask": EXPECTED_MASK,
        "expected_seeds": EXPECTED_SEEDS,
        "sweep_root": str(expected_sweep_root.resolve()),
        "source_sha256": {
            key: sha256_file(path) for key, path in sorted(files.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--sweep-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = preflight(args.repo_root.resolve(), args.sweep_root.resolve())
    except (PreflightError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"NO-VINE PREFLIGHT FAILURE: {error}") from error
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
