#!/usr/bin/env python3
"""Export and validate the standardized 143-strategy causal period panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from publication_pipeline_draft.causal_analysis_contract import load_contract, require


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(contract_path: Path, common_output: Path, output: Path) -> dict[str, object]:
    contract = load_contract(contract_path)
    require(not output.exists(), f"Causal period panel already exists: {output}")
    scored_path = common_output / "raw/scored_monthly_panel.csv"
    manifest_path = common_output / "raw/validated_strategy_manifest.csv"
    require(scored_path.is_file() and manifest_path.is_file(),
            "Common evaluator output is incomplete.")
    scored = pd.read_csv(scored_path)
    metadata = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    causal_metadata = (
        "strategy_id", "experiment_id", "strategy_level", "seed",
        "protocol_eligibility_pass", "behavior_gate_pass",
        "behavior_gate_mode", "behavior_gate_failed_metrics",
        "operational_source",
    )
    require(set(causal_metadata) <=
            set(metadata.columns), "Validated manifest lacks causal identifiers.")
    require(len(metadata) == metadata["strategy_id"].nunique() == 143,
            "Validated manifest must describe exactly 143 strategies.")
    metadata = metadata[list(causal_metadata)]
    frame = scored.merge(metadata, on="strategy_id", how="left", validate="many_to_one")
    require(frame["experiment_id"].notna().all(), "A scored path lacks causal metadata.")
    weight_columns = [name for name in scored.columns if name.startswith("w_")]
    require(bool(weight_columns), "Scored panel lacks target weights.")
    weights = frame[weight_columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    economics = contract.raw["economics"]
    tolerance = float(economics["weight_tolerance"])
    gross = np.abs(weights).sum(axis=1)
    net = weights.sum(axis=1)
    long_violation = np.maximum(weights.max(axis=1) -
                                float(economics["max_long_weight"]), 0.0)
    short_violation = np.maximum(-float(economics["max_short_weight"]) -
                                 weights.min(axis=1), 0.0)
    frame["max_abs_weight"] = np.abs(weights).max(axis=1)
    frame["gross_constraint_violation"] = np.maximum(
        gross - float(economics["gross_leverage"]), 0.0)
    frame["net_constraint_violation"] = np.abs(
        net - float(economics["net_exposure"]))
    frame["position_constraint_violation"] = np.maximum(long_violation, short_violation)
    frame["complete"] = frame["is_complete_period"]
    frame["seed"] = frame["seed"].replace("", np.nan)
    required = contract.raw["required_period_columns"]
    require(set(required) <= set(frame.columns),
            f"Scored panel cannot supply: {sorted(set(required) - set(frame.columns))}")
    frame = frame[required].sort_values(
        ["experiment_id", "strategy_level", "strategy_id", "decision_date"])
    require(len(frame) == 143 * 24 and frame["strategy_id"].nunique() == 143,
            "Standardized panel must contain 143 complete 24-period paths.")
    for _, group in frame.groupby("strategy_id", sort=False):
        require(len(group) == 24, "Every causal strategy must have 24 periods.")
    require(frame[["gross_constraint_violation", "net_constraint_violation",
                   "position_constraint_violation"]].to_numpy(float).max() <= tolerance,
            "Common evaluator exposed a constraint violation.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, date_format="%Y-%m-%d")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    result = {
        "schema_version": 1, "status": "causal_period_panel_complete",
        "analysis_contract_sha256": contract.sha256,
        "common_scored_panel_sha256": sha256(scored_path),
        "causal_period_panel_sha256": sha256(output),
        "rows": len(frame), "strategies": 143, "experiments": 13,
        "periods_per_strategy": 24,
        "common_realized_returns_and_costs": True,
        "constraint_integrity_pass": True,
        "evidence_class": "post_holdout_explanatory",
        "confirmatory_claim_permitted": False,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path(
        "publication_pipeline_draft/config/causal_analysis_contract_v1.json"))
    parser.add_argument("--common-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = export(args.contract.resolve(), args.common_output.resolve(),
                        args.output.resolve())
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        print(f"CAUSAL PERIOD EXPORT FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
