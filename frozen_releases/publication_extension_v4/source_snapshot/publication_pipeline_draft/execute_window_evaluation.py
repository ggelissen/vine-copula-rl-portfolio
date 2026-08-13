#!/usr/bin/env python3
"""Execute one immutable external-development evaluation and daily risk audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from publication_pipeline_draft.daily_mark_to_market import run as run_daily
from publication_pipeline_draft.freeze_training_release import deterministic_tar
from publication_pipeline_draft.publication_pipeline import ProtocolError, run_pipeline


class ExecutionError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_protocol(root: Path, realized: Path) -> dict[str, Any]:
    contents = root / "CONTENTS.sha256"
    manifest_path = root / "window_evaluation_manifest.json"
    if not contents.is_file() or not manifest_path.is_file():
        raise ExecutionError("Window evaluation protocol is incomplete.")
    for line in contents.read_text(encoding="ascii").splitlines():
        if line.strip():
            expected, relative = line.split("  ", 1)
            target = root / relative.removeprefix("./")
            if not target.is_file() or sha256(target) != expected:
                raise ExecutionError(f"Evaluation protocol hash mismatch: {target}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("release_status") != \
            "frozen_external_development_evaluation_contract" or \
            manifest.get("confirmatory_claim_permitted") is not False:
        raise ExecutionError("Protocol does not authorize development evaluation.")
    if not realized.is_file() or sha256(realized) != manifest["realized_panel_sha256"]:
        raise ExecutionError("Realized panel differs from the frozen protocol.")
    return manifest


def execute(protocol: Path, realized: Path, daily_returns: Path,
            return_manifest: Path, output: Path, bundle: Path | None) -> dict[str, Any]:
    if output.exists() or (bundle is not None and bundle.exists()):
        raise ExecutionError("Evaluation output/archive already exists.")
    protocol_manifest = verify_protocol(protocol, realized)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        contract = protocol / "evaluation_contract.json"
        strategies = protocol / "strategy_manifest.csv"
        common = temporary / "common_evaluator"
        run_pipeline(contract, realized, strategies, common)
        daily = temporary / "daily_mark_to_market"
        daily_manifest = run_daily(
            contract, realized, strategies, daily_returns, return_manifest, daily)
        manifest = {
            "schema_version": 1,
            "status": "external_development_window_evaluation_complete",
            "window_id": protocol_manifest["window_id"],
            "benchmark_count": protocol_manifest["benchmark_count"],
            "individual_policy_count": protocol_manifest["individual_policy_count"],
            "ensemble_count": protocol_manifest["ensemble_count"],
            "monthly_common_evaluator_complete": True,
            "daily_mark_to_market_complete": True,
            "all_daily_paths_reconciled": daily_manifest[
                "all_monthly_paths_reconciled"],
            "protocol_contents_sha256": sha256(protocol / "CONTENTS.sha256"),
            "realized_panel_sha256": sha256(realized),
            "daily_returns_sha256": sha256(daily_returns),
            "confirmatory_claim_permitted": False,
        }
        (temporary / "window_evaluation_execution_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        checksum = []
        for path in sorted(temporary.rglob("*")):
            if path.is_file() and path.name != "CONTENTS.sha256":
                checksum.append(
                    f"{sha256(path)}  {path.relative_to(temporary).as_posix()}")
        (temporary / "CONTENTS.sha256").write_text(
            "\n".join(checksum) + "\n", encoding="ascii")
        os.replace(temporary, output)
        if bundle is not None:
            bundle.parent.mkdir(parents=True, exist_ok=True)
            deterministic_tar(output, bundle)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--realized", required=True, type=Path)
    parser.add_argument("--daily-returns", required=True, type=Path)
    parser.add_argument("--return-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    try:
        result = execute(args.protocol.resolve(), args.realized.resolve(),
                         args.daily_returns.resolve(),
                         args.return_manifest.resolve(), args.output,
                         args.bundle)
    except (OSError, ValueError, json.JSONDecodeError, ProtocolError,
            ExecutionError) as error:
        print(f"WINDOW EVALUATION FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
