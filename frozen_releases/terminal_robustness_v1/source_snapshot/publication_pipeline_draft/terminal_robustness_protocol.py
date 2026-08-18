#!/usr/bin/env python3
"""Validate and freeze inputs for the terminal robustness campaign."""

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


class TerminalProtocolError(RuntimeError):
    pass


REQUIRED_PANEL_COLUMNS = {
    "strategy_id", "window_id", "decision_date", "holding_end_date",
    "is_complete_period", "gross_return", "net_return", "turnover",
    "transaction_cost", "financing_cost", "short_notional",
    "cash_borrow_notional", "gross_exposure", "net_exposure",
}

FROZEN_CODE = [
    "publication_pipeline_draft/terminal_robustness_protocol.py",
    "publication_pipeline_draft/run_terminal_robustness.py",
    "publication_pipeline_draft/verify_terminal_robustness.py",
    "publication_pipeline_draft/config/terminal_robustness_v1.json",
    "publication_pipeline_draft/tests/test_terminal_robustness_protocol.py",
    "publication_pipeline_draft/TERMINAL_ROBUSTNESS_RUNBOOK.md",
    "hpc/run_terminal_robustness_v1.sh",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TerminalProtocolError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value.get("schema_version") == 1, "Unsupported terminal contract schema.")
    require(value.get("analysis_id") == "terminal_robustness_and_reproducibility_v1",
            "Unexpected terminal analysis_id.")
    require(value.get("policy_retraining_permitted") is False and
            value.get("model_selection_permitted") is False,
            "Terminal campaign must prohibit retraining and model selection.")
    assets = value.get("asset_order", [])
    require(len(assets) == 7 and len(set(assets)) == 7,
            "Terminal contract requires seven distinct ordered assets.")
    sources = value.get("sources", [])
    source_ids = [item.get("source_id") for item in sources]
    require(len(sources) >= 3 and len(source_ids) == len(set(source_ids)),
            "Terminal evidence sources are missing or duplicated.")
    valid_classes = {
        "frozen_primary_evaluation", "post_holdout_explanatory",
        "retrospective_walk_forward",
    }
    for item in sources:
        require(item.get("evidence_class") in valid_classes,
                f"Invalid evidence class for {item.get('source_id')}.")
        require(bool(item.get("path")) and item.get("required") is True,
                "Every terminal input must be required and explicitly located.")
    economics = value.get("economics", {})
    require(float(economics.get("crra_gamma", 0)) > 0,
            "CRRA gamma must be positive.")
    for name in ("transaction_cost_bps_grid", "annual_short_borrow_percent_grid",
                 "annual_cash_borrow_percent_grid"):
        grid = economics.get(name, [])
        require(grid and all(float(item) >= 0 for item in grid) and
                list(grid) == sorted(set(grid)), f"Invalid frozen grid: {name}")
    inference = value.get("inference", {})
    require(int(inference.get("bootstrap_replications", 0)) >= 9999,
            "At least 9,999 bootstrap replications are required.")
    require(3 in inference.get("moving_block_lengths", []),
            "Registered block length three must remain in the sensitivity grid.")
    contrast_ids = [item.get("contrast_id") for item in value.get("contrasts", [])]
    require(contrast_ids and len(contrast_ids) == len(set(contrast_ids)),
            "Terminal contrasts are missing or duplicated.")
    for contrast in value["contrasts"]:
        require(contrast.get("candidate_source") in source_ids and
                contrast.get("comparator_source") in source_ids and
                bool(contrast.get("family")),
                f"Invalid contrast source/family: {contrast.get('contrast_id')}")
    require("No additional same-holdout policy training" in value.get("stop_rule", ""),
            "Terminal stop rule is absent.")
    return value


def csv_inventory(path: Path, assets: list[str]) -> tuple[int, int, int]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        header = set(reader.fieldnames or [])
        require(REQUIRED_PANEL_COLUMNS <= header,
                f"Scored panel has missing columns: {path}")
        require({f"w_{asset}" for asset in assets} <= header,
                f"Scored panel lacks ordered portfolio weights: {path}")
        rows = list(reader)
    require(bool(rows), f"Scored panel is empty: {path}")
    return len(rows), len({row["strategy_id"] for row in rows}), len({
        row["window_id"] for row in rows})


def write_contents(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "CONTENTS.sha256":
            lines.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "CONTENTS.sha256").write_text("\n".join(lines) + "\n", encoding="ascii")


def verify_release(release: Path) -> dict[str, Any]:
    contents = release / "CONTENTS.sha256"
    manifest_path = release / "terminal_robustness_release_manifest.json"
    require(contents.is_file() and manifest_path.is_file(),
            "Terminal robustness release is incomplete.")
    for line in contents.read_text(encoding="ascii").splitlines():
        if line.strip():
            expected, relative = line.split("  ", 1)
            target = release / relative.removeprefix("./")
            require(target.is_file() and sha256(target) == expected,
                    f"Terminal release hash mismatch: {relative}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("release_status") == "frozen_terminal_robustness_inputs",
            "Terminal release status is invalid.")
    return manifest


def freeze(repo: Path, contract_path: Path, output: Path) -> dict[str, Any]:
    require(not output.exists(), f"Terminal release already exists: {output}")
    contract = load_contract(contract_path)
    assets = list(contract["asset_order"])
    levels = repo / contract["daily_adjusted_levels"]
    require(levels.is_file(), f"Adjusted-level panel not found: {levels}")
    for relative in FROZEN_CODE:
        require((repo / relative).is_file(), f"Frozen analysis source is missing: {relative}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        snapshots = temporary / "input_snapshots"
        source_snapshot = temporary / "source_snapshot"
        snapshots.mkdir()
        source_snapshot.mkdir()
        inventory: list[dict[str, Any]] = []
        for item in contract["sources"]:
            original = repo / item["path"]
            require(original.is_file(), f"Required evidence panel not found: {original}")
            rows, strategies, windows = csv_inventory(original, assets)
            target = snapshots / f"{item['source_id']}.csv"
            shutil.copy2(original, target)
            inventory.append({
                "source_id": item["source_id"], "original_path": item["path"],
                "snapshot_path": target.relative_to(temporary).as_posix(),
                "evidence_class": item["evidence_class"],
                "claim_scope": item["claim_scope"], "rows": rows,
                "strategies": strategies, "windows": windows,
                "sha256": sha256(target),
            })
        levels_target = snapshots / "seven_asset_adjusted_levels.csv"
        shutil.copy2(levels, levels_target)
        shutil.copy2(contract_path, temporary / "terminal_robustness_contract.json")
        for relative in FROZEN_CODE:
            target = source_snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / relative, target)
        with (temporary / "source_inventory.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
            writer.writeheader(); writer.writerows(inventory)
        manifest = {
            "schema_version": 1,
            "release_status": "frozen_terminal_robustness_inputs",
            "analysis_id": contract["analysis_id"],
            "source_count": len(inventory),
            "source_rows": sum(item["rows"] for item in inventory),
            "source_strategies_before_namespacing": sum(
                item["strategies"] for item in inventory),
            "daily_adjusted_levels_sha256": sha256(levels_target),
            "contract_sha256": sha256(temporary / "terminal_robustness_contract.json"),
            "scientific_model_contract_changed": False,
            "policy_retraining_permitted": False,
            "model_selection_permitted": False,
            "confirmatory_claim_created": False,
            "next_action": "execute deterministic robustness campaign from snapshots only",
        }
        (temporary / "terminal_robustness_release_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_contents(temporary)
        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--contract", required=True, type=Path)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--repo-root", default=Path("."), type=Path)
    freeze_parser.add_argument("--contract", required=True, type=Path)
    freeze_parser.add_argument("--output", required=True, type=Path)
    verify = sub.add_parser("verify-release")
    verify.add_argument("--release", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            result = load_contract(args.contract.resolve())
            value = {"status": "contract_valid", "analysis_id": result["analysis_id"],
                     "sources": len(result["sources"]),
                     "contrasts": len(result["contrasts"])}
        elif args.command == "freeze":
            value = freeze(args.repo_root.resolve(), args.contract.resolve(),
                           args.output.resolve())
        else:
            value = verify_release(args.release.resolve())
    except (OSError, ValueError, json.JSONDecodeError, TerminalProtocolError) as error:
        print(f"TERMINAL ROBUSTNESS PROTOCOL FAILURE: {error}")
        return 1
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
