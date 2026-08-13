#!/usr/bin/env python3
"""Validate 130 causal policy-weight logs and build 13 investable ensembles.

The ensemble is an arithmetic mean of target weights at each decision date.
Returns and costs are deliberately not averaged; a common evaluator must score
the resulting target weights against the same realized asset-return panel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from publication_pipeline_draft.causal_analysis_contract import (
    CausalAnalysisContractError,
    load_contract,
    require,
)


class EnsembleError(CausalAnalysisContractError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Weight manifest not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"experiment_id", "seed", "path", "sha256"}
    require(bool(rows) and required <= set(rows[0]),
            f"Weight manifest requires columns: {sorted(required)}")
    return rows


def validate_weights(frame: pd.DataFrame, label: str, expected_periods: int,
                     economics: dict[str, Any]) -> tuple[list[str], pd.DataFrame]:
    date_columns = ["decision_date", "holding_end_date"]
    require(set(date_columns) <= set(frame.columns), f"{label} lacks date columns.")
    weight_columns = [name for name in frame.columns if name.startswith("w_")]
    require(bool(weight_columns), f"{label} has no w_ asset columns.")
    require(len(frame) == expected_periods, f"{label} does not have {expected_periods} periods.")
    require(not frame[date_columns].duplicated().any(), f"{label} duplicates a period.")
    frame = frame.copy()
    for name in date_columns:
        frame[name] = pd.to_datetime(frame[name], errors="raise")
    require((frame["holding_end_date"] > frame["decision_date"]).all(),
            f"{label} contains a non-positive holding period.")
    weights = frame[weight_columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    require(np.isfinite(weights).all(), f"{label} contains non-finite weights.")
    tolerance = float(economics["weight_tolerance"])
    require(np.max(np.abs(weights.sum(axis=1) - float(economics["net_exposure"]))) <= tolerance,
            f"{label} violates net exposure.")
    require(np.max(np.abs(weights).sum(axis=1)) <=
            float(economics["gross_leverage"]) + tolerance,
            f"{label} violates gross leverage.")
    require(weights.max() <= float(economics["max_long_weight"]) + tolerance,
            f"{label} violates the long position cap.")
    require(weights.min() >= -float(economics["max_short_weight"]) - tolerance,
            f"{label} violates the short position cap.")
    return weight_columns, frame.sort_values(date_columns).reset_index(drop=True)


def assemble(contract_path: Path, manifest_path: Path, repo_root: Path,
             output: Path) -> dict[str, Any]:
    contract = load_contract(contract_path)
    require(not output.exists(), f"Ensemble output already exists: {output}")
    rows = read_manifest(manifest_path)
    seeds = {int(value) for value in contract.raw["expected_seeds"]}
    experiments = set(contract.experiment_ids)
    keys = [(row["experiment_id"], int(row["seed"])) for row in rows]
    require(len(rows) == len(set(keys)) == 130, "Manifest must contain 130 unique jobs.")
    require(set(keys) == {(experiment, seed) for experiment in experiments for seed in seeds},
            "Weight manifest is not the exact 13-by-10 matched design.")

    expected_periods = int(contract.raw["sample"]["expected_periods"])
    economics = contract.raw["economics"]
    loaded: dict[tuple[str, int], pd.DataFrame] = {}
    inventory: list[dict[str, Any]] = []
    canonical_columns: list[str] | None = None
    canonical_dates: pd.DataFrame | None = None
    for row in rows:
        key = (row["experiment_id"], int(row["seed"]))
        path = (repo_root / row["path"]).resolve()
        require(path.is_file(), f"Policy weights not found: {path}")
        actual_hash = sha256(path)
        require(actual_hash == row["sha256"], f"Weight hash mismatch: {path}")
        columns, frame = validate_weights(pd.read_csv(path), f"{key}", expected_periods,
                                          economics)
        dates = frame[["decision_date", "holding_end_date"]]
        if canonical_columns is None:
            canonical_columns = columns
            canonical_dates = dates
        else:
            require(columns == canonical_columns, f"Asset order differs at {key}.")
            require(dates.equals(canonical_dates), f"Evaluation dates differ at {key}.")
        loaded[key] = frame
        inventory.append({"experiment_id": key[0], "seed": key[1],
                          "path": str(path.relative_to(repo_root)),
                          "sha256": actual_hash, "periods": len(frame)})
    assert canonical_columns is not None and canonical_dates is not None

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        ensemble_rows: list[dict[str, Any]] = []
        for experiment in sorted(experiments):
            frames = [loaded[(experiment, seed)] for seed in sorted(seeds)]
            stack = np.stack([frame[canonical_columns].to_numpy(float) for frame in frames])
            ensemble = canonical_dates.copy()
            ensemble[canonical_columns] = stack.mean(axis=0)
            validate_weights(ensemble, f"{experiment} ensemble", expected_periods, economics)
            filename = f"weights_{experiment}_ensemble.csv"
            path = temporary / filename
            ensemble.to_csv(path, index=False, date_format="%Y-%m-%d")
            ensemble_rows.append({
                "experiment_id": experiment,
                "strategy_id": f"{experiment}_ensemble",
                "strategy_level": "ensemble",
                "seed_count": len(seeds),
                "path": filename,
                "sha256": sha256(path),
                "ensemble_rule": economics["ensemble_construction"],
                "returns_averaged": False,
            })
        pd.DataFrame(inventory).sort_values(["experiment_id", "seed"]).to_csv(
            temporary / "individual_weight_inventory.csv", index=False)
        pd.DataFrame(ensemble_rows).to_csv(
            temporary / "ensemble_weight_manifest.csv", index=False)
        manifest = {
            "schema_version": 1,
            "status": "causal_weight_ensembles_complete",
            "analysis_contract_sha256": contract.sha256,
            "experiment_count": len(experiments),
            "seed_policy_count": len(rows),
            "ensemble_count": len(ensemble_rows),
            "asset_count": len(canonical_columns),
            "period_count": expected_periods,
            "ensemble_rule": economics["ensemble_construction"],
            "common_evaluator_required": True,
            "return_aggregation_used": False,
            "individual_weight_manifest_sha256": sha256(manifest_path),
            "confirmatory_claim_permitted": False,
        }
        (temporary / "causal_ensemble_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        checksum_lines = []
        for path in sorted(temporary.iterdir()):
            if path.is_file() and path.name != "CONTENTS.sha256":
                checksum_lines.append(f"{sha256(path)}  {path.name}")
        (temporary / "CONTENTS.sha256").write_text(
            "\n".join(checksum_lines) + "\n", encoding="ascii")
        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path(
        "publication_pipeline_draft/config/causal_analysis_contract_v1.json"))
    parser.add_argument("--weight-manifest", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = assemble(args.contract, args.weight_manifest, args.repo_root.resolve(),
                          args.output)
    except (CausalAnalysisContractError, OSError, ValueError) as error:
        print(f"CAUSAL ENSEMBLE FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
