#!/usr/bin/env python3
"""Freeze the complete two-window focused robustness evidence package."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

class FocusedResultFreezeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FocusedResultFreezeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_tar(source: Path, bundle: Path) -> None:
    sidecar = bundle.with_suffix(bundle.suffix + ".sha256")
    require(not bundle.exists() and not sidecar.exists(),
            f"Bundle or checksum already exists: {bundle}")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{bundle.name}.", suffix=".tmp", dir=bundle.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                               mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w",
                                  format=tarfile.PAX_FORMAT) as handle:
                    for path in [source, *sorted(source.rglob("*"))]:
                        arcname = Path(source.name) / path.relative_to(source)
                        info = handle.gettarinfo(str(path), arcname.as_posix())
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mtime = 0
                        if info.isdir():
                            info.mode = 0o755
                        elif info.isreg():
                            info.mode = 0o644
                        if info.isreg():
                            with path.open("rb") as stream:
                                handle.addfile(info, stream)
                        else:
                            handle.addfile(info)
        digest = sha256(temporary)
        os.replace(temporary, bundle)
        sidecar.write_text(f"{digest}  {bundle.name}\n", encoding="ascii")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def verify_contents(root: Path) -> None:
    contents = root / "CONTENTS.sha256"
    require(contents.is_file(), f"Missing CONTENTS.sha256: {root}")
    for line in contents.read_text(encoding="ascii").splitlines():
        if line.strip():
            expected, relative = line.split("  ", 1)
            target = root / relative.removeprefix("./")
            require(target.is_file() and sha256(target) == expected,
                    f"Checksum mismatch: {target}")


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def copy_release(source: Path, destination: Path) -> None:
    verify_contents(source)
    shutil.copytree(source, destination)


def freeze(*, prospective_release: Path, panel_release: Path,
           window_release: Path, period_release: Path,
           contracts: list[Path], audits: list[Path], weights: list[Path],
           scores: list[Path], benchmarks: list[Path],
           combined_panel: Path, analysis: Path,
           statuses: list[Path], output: Path, archive: Path | None) -> dict[str, Any]:
    require(not output.exists(), f"Output already exists: {output}")
    require(len(contracts) == len(audits) == len(weights) == len(scores) ==
            len(benchmarks) == len(statuses) == 2,
            "Exactly two windows of evidence are required.")
    for root in [prospective_release, panel_release, window_release,
                 period_release, analysis, *contracts, *audits, *weights, *scores]:
        verify_contents(root)
    for root in benchmarks:
        require((root / "benchmark_manifest.csv").is_file() and
                (root / "solver_audit.csv").is_file(),
                f"Focused benchmark release is incomplete: {root}")
    combined_manifest = combined_panel.with_suffix(combined_panel.suffix +
                                                   ".manifest.json")
    combined = read_json(combined_manifest)
    require(combined.get("window_count") == 2 and
            combined.get("confirmatory_claim_permitted") is False and
            sha256(combined_panel) == combined.get("combined_panel_sha256"),
            "Combined period panel is not the frozen two-window result.")
    result_manifest = read_json(analysis / "focused_walk_forward_manifest.json")
    require(result_manifest.get("window_count") == 2 and
            result_manifest.get("confirmatory_claim_permitted") is False and
            result_manifest.get("period_panel_sha256") == sha256(combined_panel),
            "Focused analysis does not match the combined period panel.")

    audit_rows: list[dict[str, str]] = []
    for root in audits:
        manifest = read_json(root / "focused_sweep_audit_manifest.json")
        require(manifest.get("job_count") == 15 and
                manifest.get("all_checkpoint_tensors_finite") is True and
                manifest.get("all_checkpoint_metadata_match") is True,
                f"Focused audit is not eligible: {root}")
        with (root / "focused_checkpoint_audit.csv").open(
                newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == 15, f"Focused audit has the wrong size: {root}")
        audit_rows.extend(rows)
    require(len(audit_rows) == 30 and len({(row["window_id"],
            row["experiment_id"], row["seed"]) for row in audit_rows}) == 30,
            "Focused checkpoint inventory is not exactly 2 x 3 x 5.")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        releases = temporary / "releases"
        named = {
            "prospective_code": prospective_release,
            "seven_asset_panel": panel_release,
            "two_window_schedule": window_release,
            "evaluation_periods": period_release,
            "focused_analysis": analysis,
        }
        for number, root in enumerate(contracts, 1): named[f"window_{number}_contract"] = root
        for number, root in enumerate(audits, 1): named[f"window_{number}_audit"] = root
        for number, root in enumerate(weights, 1): named[f"window_{number}_weights"] = root
        for number, root in enumerate(scores, 1): named[f"window_{number}_score"] = root
        for label, source in named.items():
            copy_release(source, releases / label)
        for number, source in enumerate(benchmarks, 1):
            shutil.copytree(source, releases / f"window_{number}_benchmarks")
        combined_root = temporary / "combined"
        combined_root.mkdir()
        shutil.copy2(combined_panel, combined_root / combined_panel.name)
        shutil.copy2(combined_manifest, combined_root / combined_manifest.name)
        status_root = temporary / "sweep_status"
        status_root.mkdir()
        for number, status in enumerate(statuses, 1):
            require(status.is_file(), f"Missing sweep status: {status}")
            shutil.copy2(status, status_root / f"window_{number}_status.csv")

        checkpoint_root = temporary / "checkpoints"
        checkpoint_inventory: list[dict[str, Any]] = []
        for row in audit_rows:
            source = Path(row["checkpoint"])
            require(source.is_file() and sha256(source) == row["checkpoint_sha256"],
                    f"Checkpoint changed after audit: {source}")
            relative = (Path(row["window_id"]) / row["experiment_id"] /
                        f"seed_{row['seed']}" / source.name)
            destination = checkpoint_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            checkpoint_inventory.append({
                "window_id": row["window_id"],
                "experiment_id": row["experiment_id"],
                "seed": row["seed"],
                "file": destination.relative_to(temporary).as_posix(),
                "sha256": row["checkpoint_sha256"],
                "size_bytes": source.stat().st_size,
            })
        inventory_path = temporary / "checkpoint_inventory.csv"
        with inventory_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(checkpoint_inventory[0]))
            writer.writeheader(); writer.writerows(checkpoint_inventory)
        manifest = {
            "schema_version": 1,
            "status": "frozen_focused_retrospective_walk_forward_results",
            "window_count": 2, "experiment_count": 3,
            "seeds_per_experiment": 5, "checkpoint_count": 30,
            "financial_benchmark_count_per_window": 6,
            "periods_per_window": 24,
            "evidence_class": "retrospective_walk_forward",
            "contains_previously_consumed_holdout": True,
            "confirmatory_claim_permitted": False,
            "combined_panel_sha256": sha256(combined_panel),
            "analysis_manifest_sha256": sha256(
                analysis / "focused_walk_forward_manifest.json"),
            "scientific_note": (
                "Post-design same-market temporal robustness evidence. Seeds are "
                "optimization replicates, not independent market samples."
            ),
        }
        (temporary / "focused_result_release_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "READ_ONLY_RESULTS.txt").write_text(
            "Do not edit. Frozen retrospective robustness evidence.\n",
            encoding="utf-8")
        checksum = [f"{sha256(path)}  {path.relative_to(temporary).as_posix()}"
                    for path in sorted(temporary.rglob("*")) if path.is_file()
                    and path.name != "CONTENTS.sha256"]
        (temporary / "CONTENTS.sha256").write_text(
            "\n".join(checksum) + "\n", encoding="ascii")
        os.replace(temporary, output)
        if archive is not None:
            deterministic_tar(output, archive)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prospective-release", required=True, type=Path)
    parser.add_argument("--panel-release", required=True, type=Path)
    parser.add_argument("--window-release", required=True, type=Path)
    parser.add_argument("--period-release", required=True, type=Path)
    parser.add_argument("--contracts", required=True, nargs=2, type=Path)
    parser.add_argument("--audits", required=True, nargs=2, type=Path)
    parser.add_argument("--weights", required=True, nargs=2, type=Path)
    parser.add_argument("--scores", required=True, nargs=2, type=Path)
    parser.add_argument("--benchmarks", required=True, nargs=2, type=Path)
    parser.add_argument("--combined-panel", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--statuses", required=True, nargs=2, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    try:
        result = freeze(
            prospective_release=args.prospective_release.resolve(),
            panel_release=args.panel_release.resolve(),
            window_release=args.window_release.resolve(),
            period_release=args.period_release.resolve(),
            contracts=[value.resolve() for value in args.contracts],
            audits=[value.resolve() for value in args.audits],
            weights=[value.resolve() for value in args.weights],
            scores=[value.resolve() for value in args.scores],
            benchmarks=[value.resolve() for value in args.benchmarks],
            combined_panel=args.combined_panel.resolve(),
            analysis=args.analysis.resolve(),
            statuses=[value.resolve() for value in args.statuses],
            output=args.output.resolve(),
            archive=args.archive.resolve() if args.archive is not None else None,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError,
            FocusedResultFreezeError) as error:
        print(f"FOCUSED RESULT FREEZE FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
