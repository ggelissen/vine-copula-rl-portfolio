#!/usr/bin/env python3
"""Freeze a completed causal evaluation, standardized panel, and analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from publication_pipeline_draft.causal_analysis_contract import load_contract, require
from publication_pipeline_draft.freeze_causal_analysis_plan import (
    CausalAnalysisFreezeError,
    verify_causal_analysis_release,
    verify_contents,
)
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    repo, output = args.repo_root.resolve(), args.output.resolve()
    require(not output.exists(), f"Causal result release already exists: {output}")
    require(args.archive is None or not args.archive.resolve().exists(),
            "Causal result archive already exists.")
    release = verify_causal_analysis_release(args.analysis_release, repo)
    contract = load_contract(args.contract)
    require(release["analysis_contract_sha256"] == contract.sha256,
            "Analysis release and contract differ.")
    verify_contents(args.evaluation_interface)
    verify_contents(args.common_output)
    verify_contents(args.analysis_output)
    bindings = json.loads((args.evaluation_interface /
                           "causal_evaluation_bindings.json").read_text(encoding="utf-8"))
    analysis = json.loads((args.analysis_output /
                           "causal_analysis_manifest.json").read_text(encoding="utf-8"))
    period_manifest_path = args.period_panel.with_suffix(
        args.period_panel.suffix + ".manifest.json")
    panel = json.loads(period_manifest_path.read_text(encoding="utf-8"))
    common_manifest_path = args.common_output / "run_manifest.json"
    common = json.loads(common_manifest_path.read_text(encoding="utf-8"))
    require(bindings.get("analysis_contract_sha256") == contract.sha256 and
            analysis.get("analysis_contract_sha256") == contract.sha256 and
            panel.get("analysis_contract_sha256") == contract.sha256,
            "A causal artifact uses a different analysis contract.")
    require(analysis.get("status") == "causal_analysis_complete" and
            panel.get("status") == "causal_period_panel_complete",
            "Causal analysis or standardized panel is incomplete.")
    require(sha256(args.period_panel) == analysis.get("period_panel_sha256") ==
            panel.get("causal_period_panel_sha256"),
            "Standardized causal period panel hash differs.")
    require(int(analysis.get("strategy_count", -1)) == 143 and
            int(analysis.get("experiment_count", -1)) == 13,
            "Causal analysis cardinality differs from the contract.")
    require(common.get("contract_sha256") == sha256(
        args.evaluation_interface / "evaluation_contract.json"),
        "Common evaluator used a different accounting contract.")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        shutil.copytree(args.analysis_output, temporary / "analysis_results")
        shutil.copytree(args.evaluation_interface, temporary / "evaluation_interface")
        shutil.copytree(args.common_output, temporary / "common_accounting")
        shutil.copy2(args.period_panel, temporary / args.period_panel.name)
        shutil.copy2(period_manifest_path, temporary / period_manifest_path.name)
        result = {
            "schema_version": 1,
            "release_status": "frozen_post_holdout_causal_results",
            "analysis_id": contract.raw["analysis_id"],
            "analysis_contract_sha256": contract.sha256,
            "analysis_plan_release_contents_sha256": release["release_contents_sha256"],
            "causal_period_panel_sha256": sha256(args.period_panel),
            "strategy_count": 143, "experiment_count": 13,
            "primary_contrast_count": 8, "algorithm_contrast_count": 4,
            "all_preregistered_results_reported": True,
            "evidence_class": "post_holdout_explanatory",
            "confirmatory_claim_permitted": False,
            "claim_limit": "mechanism attribution on a previously consumed holdout",
        }
        (temporary / "causal_result_release_manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        lines = []
        for path in sorted(temporary.rglob("*")):
            if path.is_file() and path.name != "CONTENTS.sha256":
                lines.append(f"{sha256(path)}  {path.relative_to(temporary).as_posix()}")
        (temporary / "CONTENTS.sha256").write_text(
            "\n".join(lines) + "\n", encoding="ascii")
        os.replace(temporary, output)
        if args.archive is not None:
            from publication_pipeline_draft.freeze_training_release import deterministic_tar
            args.archive.parent.mkdir(parents=True, exist_ok=True)
            deterministic_tar(output, args.archive.resolve())
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path, default=Path(
        "publication_pipeline_draft/config/causal_analysis_contract_v1.json"))
    parser.add_argument("--analysis-release", required=True, type=Path)
    parser.add_argument("--evaluation-interface", required=True, type=Path)
    parser.add_argument("--common-output", required=True, type=Path)
    parser.add_argument("--period-panel", required=True, type=Path)
    parser.add_argument("--analysis-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.contract = (args.repo_root / args.contract).resolve()
    for name in ("analysis_release", "evaluation_interface", "common_output",
                 "period_panel", "analysis_output", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.archive is not None:
        args.archive = args.archive.resolve()
    try:
        result = freeze(args)
    except (CausalAnalysisFreezeError, OSError, ValueError, KeyError,
            json.JSONDecodeError) as error:
        print(f"CAUSAL RESULT FREEZE FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
