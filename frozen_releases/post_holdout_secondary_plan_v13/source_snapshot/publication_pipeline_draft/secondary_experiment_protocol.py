"""Freeze and validate post-holdout secondary experiment plans.

This module never scores the consumed holdout and never fabricates an
ablation result.  It turns a human-readable experiment contract into an
immutable, checksummed execution plan and validates artifacts produced by the
existing training pipeline.

The completed ``locked_oos_v1`` result is already consumed.  Every experiment
managed here is therefore explanatory unless it is evaluated on a separately
declared, non-overlapping future/external sample.
"""

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
from typing import Any, Iterable

EVIDENCE_CLASS = "post_holdout_explanatory"
READY_STATES = {"ready_to_run", "ready_existing_artifacts"}
KNOWN_STATES = READY_STATES | {
    "blocked_requires_implementation",
    "blocked_requires_preregistration",
    "deferred_high_cost",
}


class ProtocolError(RuntimeError):
    """Raised when a secondary experiment would violate its frozen contract."""


def parse_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ProtocolError(f"{field} is not a valid boolean: {value!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name != "CONTENTS.sha256":
            rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "CONTENTS.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def deterministic_tar(
    source: Path,
    bundle: Path,
    *,
    root_name: str | None = None,
) -> None:
    """Create a deterministic gzip tar and publish bundle plus sidecar atomically."""
    sidecar = bundle.with_suffix(bundle.suffix + ".sha256")
    if bundle.exists() or sidecar.exists():
        raise ProtocolError(f"Bundle or sidecar already exists: {bundle}")
    archive_root = root_name or source.name
    if not archive_root or Path(archive_root).name != archive_root:
        raise ProtocolError(f"Archive root must be one stable path component: {archive_root!r}")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{bundle.name}.", suffix=".tmp", dir=bundle.parent
    )
    os.close(descriptor)
    temporary_bundle = Path(temporary_name)
    temporary_sidecar = Path(f"{temporary_name}.sha256")
    published_bundle = False
    try:
        with temporary_bundle.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
                ) as handle:
                    paths = [
                        source,
                        *sorted(source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()),
                    ]
                    for path in paths:
                        arcname = Path(archive_root) / path.relative_to(source)
                        info = handle.gettarinfo(str(path), arcname=arcname.as_posix())
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mtime = 0
                        if info.isdir():
                            info.mode = 0o755
                        elif info.isreg():
                            info.mode = 0o755 if info.mode & 0o111 else 0o644
                        if info.isreg():
                            with path.open("rb") as stream:
                                handle.addfile(info, stream)
                        else:
                            handle.addfile(info)
        temporary_sidecar.write_text(
            f"{sha256_file(temporary_bundle)}  {bundle.name}\n", encoding="utf-8"
        )
        os.replace(temporary_bundle, bundle)
        published_bundle = True
        os.replace(temporary_sidecar, sidecar)
    except Exception:
        temporary_bundle.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
        if published_bundle:
            bundle.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProtocolError(f"JSON file not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ProtocolError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProtocolError(f"Expected a JSON object: {path}")
    return value


def require_fields(value: dict[str, Any], fields: Iterable[str], label: str) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise ProtocolError(f"{label} is missing fields: {', '.join(missing)}")


def validate_contract(contract: dict[str, Any]) -> list[dict[str, Any]]:
    require_fields(
        contract,
        [
            "schema_version",
            "protocol_id",
            "evidence_class",
            "consumed_holdout",
            "main_result_immutable",
            "source_files",
            "experiments",
        ],
        "secondary experiment contract",
    )
    if int(contract["schema_version"]) != 1:
        raise ProtocolError("Only secondary experiment schema_version=1 is supported.")
    if contract["evidence_class"] != EVIDENCE_CLASS:
        raise ProtocolError(
            f"Secondary experiments on the consumed sample must use evidence_class={EVIDENCE_CLASS}."
        )
    if not parse_bool(contract["main_result_immutable"], "main_result_immutable"):
        raise ProtocolError("The completed main result must remain immutable.")
    holdout = contract["consumed_holdout"]
    if not isinstance(holdout, dict):
        raise ProtocolError("consumed_holdout must be an object.")
    require_fields(holdout, ["evaluation_id", "archive_sha256", "status"], "consumed_holdout")
    if holdout["status"] != "consumed" or len(str(holdout["archive_sha256"])) != 64:
        raise ProtocolError("The consumed holdout must have status=consumed and a SHA-256 digest.")
    if not isinstance(contract["source_files"], list) or not contract["source_files"]:
        raise ProtocolError("source_files must be a non-empty list.")
    experiments = contract["experiments"]
    if not isinstance(experiments, list) or not experiments:
        raise ProtocolError("experiments must be a non-empty list.")

    identifiers: set[str] = set()
    for index, experiment in enumerate(experiments):
        if not isinstance(experiment, dict):
            raise ProtocolError(f"Experiment {index} must be an object.")
        require_fields(
            experiment,
            [
                "experiment_id",
                "label",
                "status",
                "scientific_question",
                "evidence_class",
                "holdout_use",
                "expected_seeds",
            ],
            f"experiment {index}",
        )
        identifier = str(experiment["experiment_id"])
        if not identifier or identifier in identifiers:
            raise ProtocolError(f"Duplicate/empty experiment_id: {identifier!r}")
        identifiers.add(identifier)
        if experiment["status"] not in KNOWN_STATES:
            raise ProtocolError(f"Unknown status for {identifier}: {experiment['status']}")
        if experiment["evidence_class"] != EVIDENCE_CLASS:
            raise ProtocolError(f"{identifier} is not labelled post-holdout explanatory.")
        if experiment["holdout_use"] != "secondary_explanatory_only":
            raise ProtocolError(f"{identifier} has an invalid holdout_use declaration.")
        seeds = experiment["expected_seeds"]
        if not isinstance(seeds, list) or not all(isinstance(seed, int) for seed in seeds):
            raise ProtocolError(f"{identifier} expected_seeds must be an integer list.")
        if len(seeds) != len(set(seeds)):
            raise ProtocolError(f"{identifier} contains duplicate seeds.")
        if experiment["status"] in READY_STATES and not seeds:
            raise ProtocolError(f"Ready experiment {identifier} must declare seeds.")
        if identifier == "no_vine_td3":
            require_fields(
                experiment,
                [
                    "seed_specification", "sweep_root", "vine_observation_mode",
                    "signal_mask", "ablation_scope",
                ],
                identifier,
            )
            if experiment["vine_observation_mode"] != "zero":
                raise ProtocolError("no_vine_td3 must use vine_observation_mode=zero.")
            if experiment["signal_mask"] != "explicit_vine_and_scenario_cvar_v1":
                raise ProtocolError("no_vine_td3 must remove all policy-visible vine signals.")
            if experiment["ablation_scope"] != (
                "policy_visible_vine_state_only_reward_cvar_retained"
            ):
                raise ProtocolError(
                    "no_vine_td3 must disclose that vine-scenario CVaR remains in the reward."
                )
    return experiments


def source_inventory(repo_root: Path, relatives: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in relatives:
        relative = Path(str(raw)).as_posix()
        if relative in seen or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ProtocolError(f"Unsafe or duplicate source path: {raw}")
        seen.add(relative)
        path = repo_root / relative
        if not path.is_file():
            raise ProtocolError(f"Protocol source file is missing: {path}")
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def render_commands(contract: dict[str, Any]) -> str:
    experiments = {item["experiment_id"]: item for item in contract["experiments"]}
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "REPO_ROOT=\"${REPO_ROOT:-$(pwd -P)}\"",
        "RSCRIPT=\"${RSCRIPT:-Rscript}\"",
        "PYTHON=\"${PYTHON:-python3}\"",
        ": \"${TRAIN_PYTHON:?Set TRAIN_PYTHON to the GPU-enabled reticulate Python for training}\"",
        "export TRAIN_PYTHON",
        "V4_ARCHIVE=\"${V4_ARCHIVE:-locked_evaluation/main_oos_v4_operational_retry.tar.gz}\"",
        "V4_SIDECAR=\"${V4_SIDECAR:-locked_evaluation/main_oos_v4_operational_retry.tar.gz.sha256}\"",
        "PLAN_ROOT=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd -P)\"",
        "cd \"$REPO_ROOT\"",
        "export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC",
        "sha256sum -c \"$PLAN_ROOT/LIVE_SOURCE_CONTENTS.sha256\"",
        "",
        "# These runs are post-holdout explanatory. They cannot retroactively",
        "# convert locked_oos_v1 into a fresh confirmatory result.",
    ]
    no_vine = experiments.get("no_vine_td3")
    if no_vine and no_vine["status"] == "ready_to_run":
        root = _shell_quote(str(no_vine["sweep_root"]))
        expected = len(no_vine["expected_seeds"])
        lines.extend(
            [
                "",
                "# 1. Matched-capacity TD3 without policy-visible vine state (expensive).",
                "# Four independent one-GPU workers; defaults to 18 vine-simulation cores each.",
                f"NO_VINE_SWEEP_ROOT={root} RSCRIPT=\"$RSCRIPT\" PYTHON=\"$PYTHON\" \\",
                "  bash hpc/run_no_vine_4gpu.sh",
                "",
                "# 2. Fail-closed validation before aggregation.",
                "\"$PYTHON\" publication_pipeline_draft/secondary_experiment_protocol.py \\",
                "  validate-sweep \\",
                "  --contract publication_pipeline_draft/config/secondary_experiments_v1.json \\",
                "  --experiment no_vine_td3 \\",
                f"  --status {_shell_quote(str(no_vine['sweep_root']) + '/seed_sweep_status.csv')}",
                "",
                "# 3. Aggregate/freeze only after all expected gates pass.",
                "\"$PYTHON\" publication_pipeline_draft/diagnostic_artifacts.py \\",
                f"  --rl-runs {root} --expected-seeds {expected} \\",
                "  --output data/publication_no_vine_training_artifacts_10seeds",
                "tar -czf no_vine_training_artifacts_10seeds.tar.gz \\",
                "  -C data publication_no_vine_training_artifacts_10seeds",
                "\"$PYTHON\" publication_pipeline_draft/freeze_training_release.py \\",
                f"  --repo-root . --rl-runs {root} \\",
                "  --diagnostics-archive \"$REPO_ROOT/no_vine_training_artifacts_10seeds.tar.gz\" \\",
                f"  --expected-seeds {expected} \\",
                "  --output frozen_releases/no_vine_schema5_secondary_v1 \\",
                "  --bundle frozen_releases/no_vine_schema5_secondary_v1.tar.gz",
                "(cd frozen_releases && sha256sum -c no_vine_schema5_secondary_v1.tar.gz.sha256)",
            ]
        )
    pretrained = experiments.get("pretrained_only")
    if pretrained and pretrained["status"] == "ready_existing_artifacts":
        lines.extend(
            [
                "",
                "# 4. Validate that every frozen full seed contains the already-trained",
                "# pre-fine-tuning checkpoint. No training or holdout scoring occurs here.",
                "\"$PYTHON\" publication_pipeline_draft/secondary_experiment_protocol.py \\",
                "  validate-checkpoints \\",
                "  --contract publication_pipeline_draft/config/secondary_experiments_v1.json \\",
                "  --experiment pretrained_only \\",
                "  --training-release frozen_releases/training_schema5_v1",
            ]
        )
    lines.extend(
        [
            "",
            "# 5. Same-sample explanatory evaluation. This is never confirmatory.",
            ": \"${POLICY_PYTHON:?Set POLICY_PYTHON to the isolated CPU PyTorch interpreter}\"",
            "export POLICY_PYTHON",
            "\"$PYTHON\" publication_pipeline_draft/secondary_ablation_batch.py \\",
            "  --repo-root . \\",
            "  --contract publication_pipeline_draft/config/secondary_evaluation_contract_v1.json \\",
            "  --successful-archive \"$V4_ARCHIVE\" \\",
            "  --successful-sidecar \"$V4_SIDECAR\" \\",
            "  --evaluation-contract publication_pipeline_draft/config/evaluation_contract.json \\",
            "  --runtime-config config/config.yaml \\",
            "  --full-training-release frozen_releases/training_schema5_v1 \\",
            "  --no-vine-training-release frozen_releases/no_vine_schema5_secondary_v1 \\",
            "  --secondary-plan-release \"$PLAN_ROOT\" \\",
            "  --output secondary_evaluation/post_holdout_explanatory_ablation_v2_operational_retry \\",
            "  --bundle secondary_evaluation/post_holdout_explanatory_ablation_v2_operational_retry.tar.gz \\",
            "  --rscript \"$RSCRIPT\"",
            "(cd secondary_evaluation && \\",
            "  sha256sum -c post_holdout_explanatory_ablation_v2_operational_retry.tar.gz.sha256)",
            "",
            "echo 'Secondary explanatory batch complete. Do not present it as a fresh confirmatory result.'",
            "",
        ]
    )
    return "\n".join(lines)


def freeze_plan(
    repo_root: Path,
    contract_path: Path,
    output: Path,
    bundle: Path | None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    contract_path = contract_path.resolve()
    output = output.resolve()
    if output.exists():
        raise ProtocolError(f"Output already exists and will not be overwritten: {output}")
    if bundle is not None and (bundle.exists() or bundle.with_suffix(bundle.suffix + ".sha256").exists()):
        raise ProtocolError(f"Bundle or sidecar already exists: {bundle}")
    contract = load_json(contract_path)
    experiments = validate_contract(contract)
    inventory = source_inventory(repo_root, [str(item) for item in contract["source_files"]])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}_", dir=output.parent))
    try:
        normalized = json.dumps(contract, indent=2, sort_keys=True) + "\n"
        (temporary / "secondary_experiment_contract.json").write_text(normalized, encoding="utf-8")
        with (temporary / "source_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["path", "size_bytes", "sha256"])
            writer.writeheader()
            writer.writerows(inventory)
        snapshot = temporary / "source_snapshot"
        live_checksum_rows = []
        for row in inventory:
            relative = Path(str(row["path"]))
            source = repo_root / relative
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            live_checksum_rows.append(f"{row['sha256']}  {relative.as_posix()}")
        (temporary / "LIVE_SOURCE_CONTENTS.sha256").write_text(
            "\n".join(live_checksum_rows) + "\n", encoding="utf-8"
        )
        command_file = temporary / "EXECUTE_SECONDARY_EXPERIMENTS.sh"
        command_file.write_text(render_commands(contract), encoding="utf-8", newline="\n")
        command_file.chmod(0o755)
        (temporary / "EVIDENCE_CLASSIFICATION.md").write_text(
            "# Evidence classification\n\n"
            "This release was created after `locked_oos_v1` was consumed. All same-sample "
            "ablation comparisons are secondary/explanatory. They may explain mechanisms but "
            "cannot establish a fresh confirmatory superiority claim. The successful v4 main "
            "result and all failed operational attempts remain immutable.\n",
            encoding="utf-8",
        )
        source_payload = "\n".join(
            f"{row['sha256']}  {row['path']}" for row in inventory
        ).encode("utf-8")
        manifest = {
            "schema_version": 1,
            "release_status": "frozen_post_holdout_secondary_plan",
            "protocol_id": contract["protocol_id"],
            "evidence_class": EVIDENCE_CLASS,
            "consumed_holdout": contract["consumed_holdout"],
            "main_result_immutable": True,
            "experiment_count": len(experiments),
            "ready_experiment_ids": [
                item["experiment_id"] for item in experiments if item["status"] in READY_STATES
            ],
            "blocked_experiment_ids": [
                item["experiment_id"] for item in experiments if item["status"] not in READY_STATES
            ],
            "contract_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            "source_inventory_sha256": hashlib.sha256(source_payload).hexdigest(),
            "next_action": "execute only ready experiments; do not edit this frozen plan",
        }
        (temporary / "secondary_plan_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_checksums(temporary)
        os.replace(temporary, output)
        if bundle is not None:
            deterministic_tar(output, bundle.resolve())
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def experiment_by_id(contract: dict[str, Any], identifier: str) -> dict[str, Any]:
    validate_contract(contract)
    matches = [item for item in contract["experiments"] if item["experiment_id"] == identifier]
    if len(matches) != 1:
        raise ProtocolError(f"Experiment not found: {identifier}")
    return matches[0]


def read_status(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ProtocolError(f"Sweep status file not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ProtocolError("Sweep status is empty.")
    return rows


def validate_sweep(contract_path: Path, experiment_id: str, status_path: Path) -> dict[str, Any]:
    experiment = experiment_by_id(load_json(contract_path), experiment_id)
    if experiment_id != "no_vine_td3":
        raise ProtocolError("validate-sweep currently supports only no_vine_td3.")
    rows = read_status(status_path)
    required = {
        "seed",
        "training_status",
        "sanity_status",
        "no_holdout_gate_pass",
        "vine_observation_mode",
        "no_vine_signal_mask",
        "full_zero_vine_median_action_l1",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ProtocolError(f"Sweep status is missing columns: {', '.join(missing)}")
    expected = [int(seed) for seed in experiment["expected_seeds"]]
    actual = [int(row["seed"]) for row in rows]
    if len(rows) != len(expected) or len(set(actual)) != len(expected) or set(actual) != set(expected):
        raise ProtocolError(f"Sweep seeds do not match the frozen plan: {actual} != {expected}")
    failures: list[str] = []
    for row in rows:
        seed = row["seed"]
        if int(float(row["training_status"])) != 0 or int(float(row["sanity_status"])) != 0:
            failures.append(f"seed {seed}: non-zero training/sanity status")
        if not parse_bool(row["no_holdout_gate_pass"], "no_holdout_gate_pass"):
            failures.append(f"seed {seed}: no-holdout gate failed")
        if row["vine_observation_mode"] != "zero":
            failures.append(f"seed {seed}: vine mode is not zero")
        if row["no_vine_signal_mask"] != experiment["signal_mask"]:
            failures.append(f"seed {seed}: signal mask mismatch")
        try:
            invariance = abs(float(row["full_zero_vine_median_action_l1"]))
        except ValueError:
            invariance = float("inf")
        if not invariance <= 1e-8:
            failures.append(f"seed {seed}: zero-channel invariance={invariance:g}")
    if failures:
        raise ProtocolError("No-vine sweep validation failed:\n - " + "\n - ".join(failures))
    return {
        "status": "valid",
        "experiment_id": experiment_id,
        "seed_count": len(rows),
        "status_file_sha256": sha256_file(status_path),
        "evidence_class": EVIDENCE_CLASS,
    }


def merge_sweep_status(
    contract_path: Path,
    experiment_id: str,
    inputs: list[Path],
    output: Path,
) -> dict[str, Any]:
    if not inputs:
        raise ProtocolError("At least one worker status file is required.")
    if output.exists():
        raise ProtocolError(f"Merged status already exists and will not be overwritten: {output}")
    experiment = experiment_by_id(load_json(contract_path), experiment_id)
    expected = {int(seed) for seed in experiment["expected_seeds"]}
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for path in inputs:
        worker_rows = read_status(path)
        worker_fields = list(worker_rows[0])
        if fieldnames is None:
            fieldnames = worker_fields
        elif worker_fields != fieldnames:
            raise ProtocolError(f"Worker status columns/order differ: {path}")
        rows.extend(worker_rows)
    actual = [int(row["seed"]) for row in rows]
    if len(actual) != len(set(actual)):
        raise ProtocolError("Worker statuses contain duplicate seeds.")
    if set(actual) != expected:
        raise ProtocolError(
            f"Worker statuses do not cover the frozen seeds: {sorted(actual)} != {sorted(expected)}"
        )
    assert fieldnames is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise ProtocolError(f"Temporary merged status already exists: {temporary}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: int(row["seed"])))
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    result = validate_sweep(contract_path, experiment_id, output)
    result["worker_status_count"] = len(inputs)
    return result


def verify_contents(root: Path) -> None:
    checksums = root / "CONTENTS.sha256"
    if not checksums.is_file():
        raise ProtocolError(f"Training release lacks CONTENTS.sha256: {root}")
    for raw in checksums.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ProtocolError(f"Training release checksum mismatch: {relative}")


def validate_checkpoints(
    contract_path: Path, experiment_id: str, training_release: Path
) -> dict[str, Any]:
    experiment = experiment_by_id(load_json(contract_path), experiment_id)
    if experiment_id != "pretrained_only":
        raise ProtocolError("validate-checkpoints currently supports only pretrained_only.")
    root = training_release.resolve()
    if not root.is_dir():
        raise ProtocolError(f"Training release not found: {root}")
    verify_contents(root)
    manifest = load_json(root / "training_release_manifest.json")
    if manifest.get("release_status") != "frozen_pre_oos" or parse_bool(
        manifest.get("holdout_accessed_by_freezer", True),
        "holdout_accessed_by_freezer",
    ):
        raise ProtocolError("Training release is not a valid holdout-blind frozen release.")
    expected = [int(seed) for seed in experiment["expected_seeds"]]
    observed_directories: dict[int, Path] = {}
    for directory in sorted((root / "seeds").glob("seed_*")):
        if not directory.is_dir():
            continue
        try:
            seed = int(directory.name.removeprefix("seed_"))
        except ValueError as error:
            raise ProtocolError(f"Invalid seed directory: {directory.name}") from error
        if seed in observed_directories:
            raise ProtocolError(f"Duplicate seed directory: {seed}")
        observed_directories[seed] = directory
    if set(observed_directories) != set(expected):
        raise ProtocolError(
            "Training release seed directories do not exactly match the frozen contract: "
            f"expected={sorted(expected)}, observed={sorted(observed_directories)}"
        )
    rows: list[dict[str, Any]] = []
    for seed in expected:
        directory = observed_directories[seed]
        checkpoint = directory / "td3_lstm_vine_pretrained.pt"
        full = directory / "td3_lstm_vine_full.pt"
        if not checkpoint.is_file() or not full.is_file():
            raise ProtocolError(f"Seed {seed} lacks pretrained/full checkpoint pair.")
        rows.append(
            {
                "seed": seed,
                "pretrained_sha256": sha256_file(checkpoint),
                "full_sha256": sha256_file(full),
            }
        )
    if len({row["pretrained_sha256"] for row in rows}) != len(rows):
        raise ProtocolError("Pretrained checkpoint hashes are not unique across seeds.")
    return {
        "status": "valid",
        "experiment_id": experiment_id,
        "seed_count": len(rows),
        "training_release": str(root),
        "training_release_contents_sha256": sha256_file(root / "CONTENTS.sha256"),
        "evidence_class": EVIDENCE_CLASS,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="Freeze a checksummed secondary plan.")
    freeze.add_argument("--repo-root", type=Path, default=Path("."))
    freeze.add_argument("--contract", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--bundle", type=Path)

    sweep = subparsers.add_parser("validate-sweep", help="Validate a no-vine sweep status.")
    sweep.add_argument("--contract", type=Path, required=True)
    sweep.add_argument("--experiment", required=True)
    sweep.add_argument("--status", type=Path, required=True)

    merge = subparsers.add_parser(
        "merge-sweep-status", help="Merge disjoint worker statuses and validate them."
    )
    merge.add_argument("--contract", type=Path, required=True)
    merge.add_argument("--experiment", required=True)
    merge.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)

    checkpoints = subparsers.add_parser(
        "validate-checkpoints", help="Validate existing pretrained checkpoints."
    )
    checkpoints.add_argument("--contract", type=Path, required=True)
    checkpoints.add_argument("--experiment", required=True)
    checkpoints.add_argument("--training-release", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "freeze":
            result = freeze_plan(args.repo_root, args.contract, args.output, args.bundle)
        elif args.command == "validate-sweep":
            result = validate_sweep(args.contract, args.experiment, args.status)
        elif args.command == "merge-sweep-status":
            result = merge_sweep_status(
                args.contract, args.experiment, args.inputs, args.output
            )
        elif args.command == "validate-checkpoints":
            result = validate_checkpoints(args.contract, args.experiment, args.training_release)
        else:  # pragma: no cover
            raise ProtocolError(f"Unsupported command: {args.command}")
    except ProtocolError as error:
        print(f"PROTOCOL FAILURE: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
