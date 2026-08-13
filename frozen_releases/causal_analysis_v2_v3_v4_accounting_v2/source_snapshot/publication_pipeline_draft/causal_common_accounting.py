#!/usr/bin/env python3
"""Score 143 causal target-weight paths with the frozen common accounting code.

Unlike the general publication pipeline, this interface intentionally performs
no superiority test. Statistical decisions belong exclusively to the frozen
causal analysis contract and ``analyze_causal_results.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from publication_pipeline_draft.publication_pipeline import (
    Contract,
    ProtocolError,
    read_and_validate_weights,
    read_realized_panel,
    read_strategy_manifest,
    score_strategy,
    sha256_file,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def run(contract_path: Path, realized_path: Path, manifest_path: Path,
        output: Path) -> dict[str, Any]:
    require(not output.exists(), f"Common-accounting output already exists: {output}")
    contract = Contract.read(contract_path)
    require(contract.get("confirmatory_claim_permitted") is False,
            "Causal common accounting cannot authorize confirmation.")
    require(contract.get("evidence_class") == "post_holdout_explanatory",
            "Causal common accounting has the wrong evidence class.")
    require(contract["predeclared_ensembles"] == [],
            "Causal ensembles must already exist as explicit weight logs.")
    realized, assets = read_realized_panel(realized_path, contract)
    manifest = read_strategy_manifest(manifest_path, contract)
    require(len(manifest) == manifest["strategy_id"].nunique() == 143,
            "Common accounting requires exactly 143 explicit strategies.")
    require({"experiment_id", "strategy_level"} <= set(manifest.columns),
            "Strategy manifest lacks causal metadata.")
    weights: dict[str, pd.DataFrame] = {}
    hashes = [
        {"artifact": "evaluation_contract", "path": str(contract_path),
         "sha256": sha256_file(contract_path)},
        {"artifact": "realized_panel", "path": str(realized_path),
         "sha256": sha256_file(realized_path)},
        {"artifact": "strategy_manifest", "path": str(manifest_path),
         "sha256": sha256_file(manifest_path)},
    ]
    for _, row in manifest.iterrows():
        weight, digest = read_and_validate_weights(
            row, manifest_path, realized, assets, contract)
        weights[row["strategy_id"]] = weight
        hashes.append({"artifact": f"weights:{row['strategy_id']}",
                       "path": row["weight_log_path"], "sha256": digest})
    scored = pd.concat([
        score_strategy(strategy_id, weight, realized, assets, contract)
        for strategy_id, weight in weights.items()
    ], ignore_index=True)
    require(len(scored) == 143 * 24 and scored["strategy_id"].nunique() == 143,
            "Common accounting did not create 143 complete 24-period paths.")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        raw = temporary / "raw"; raw.mkdir()
        scored.to_csv(raw / "scored_monthly_panel.csv", index=False,
                      date_format="%Y-%m-%d")
        manifest.to_csv(raw / "validated_strategy_manifest.csv", index=False)
        pd.DataFrame(hashes).to_csv(raw / "input_hashes.csv", index=False)
        pd.DataFrame([
            {"check": "strategy_cardinality", "status": "pass",
             "detail": "130 seeds plus 13 explicit weight-space ensembles"},
            {"check": "common_realized_panel", "status": "pass",
             "detail": "Every strategy joined one-to-one to the same 24 periods"},
            {"check": "constraints", "status": "pass",
             "detail": "Net, gross, long, and short constraints validated"},
            {"check": "common_costs", "status": "pass",
             "detail": "Drifted turnover, transaction, and financing costs rescored"},
            {"check": "statistical_inference", "status": "not_run_here",
             "detail": "Only the frozen causal analyzer may test contrasts"},
        ]).to_csv(raw / "protocol_checks.csv", index=False)
        result = {
            "schema_version": 1, "status": "causal_common_accounting_complete",
            "evaluation_id": contract["evaluation_id"],
            "contract_sha256": sha256_file(contract_path),
            "realized_panel_sha256": sha256_file(realized_path),
            "strategy_manifest_sha256": sha256_file(manifest_path),
            "strategy_count": 143, "periods_per_strategy": 24,
            "scored_row_count": len(scored), "asset_count": len(assets),
            "common_realized_returns": True, "common_cost_accounting": True,
            "statistical_inference_performed": False,
            "evidence_class": "post_holdout_explanatory",
            "confirmatory_claim_permitted": False,
        }
        (temporary / "run_manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        lines = []
        for path in sorted(temporary.rglob("*")):
            if path.is_file() and path.name != "CONTENTS.sha256":
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                lines.append(f"{digest}  {path.relative_to(temporary).as_posix()}")
        (temporary / "CONTENTS.sha256").write_text(
            "\n".join(lines) + "\n", encoding="ascii")
        os.replace(temporary, output)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--realized", required=True, type=Path)
    parser.add_argument("--strategies", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run(args.contract.resolve(), args.realized.resolve(),
                     args.strategies.resolve(), args.output.resolve())
    except (ProtocolError, OSError, ValueError, KeyError) as error:
        print(f"CAUSAL COMMON ACCOUNTING FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
