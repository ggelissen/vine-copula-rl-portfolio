#!/usr/bin/env python3
"""Run an immutable, post-holdout explanatory ablation evaluation batch.

The consumed v4 holdout is never reconstructed from live market data here.
Realized returns, benchmark weights, and full-policy weights are copied
byte-for-byte from the successful v4 archive.  Only no-vine ``full`` and
existing full-model ``pretrained`` checkpoint weights are generated.  All
same-sample summaries are descriptive post-holdout evidence, never
confirmatory evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

try:  # package import in tests; script import at the command line
    from .freeze_training_release import deterministic_tar
    from .publication_pipeline import (
        Contract,
        KEYS,
        ProtocolError as EvaluationProtocolError,
        crra_utility,
        empirical_metrics,
        read_realized_panel,
        score_strategy,
        validate_weight_matrix,
    )
except ImportError:  # pragma: no cover
    from freeze_training_release import deterministic_tar
    from publication_pipeline import (
        Contract,
        KEYS,
        ProtocolError as EvaluationProtocolError,
        crra_utility,
        empirical_metrics,
        read_realized_panel,
        score_strategy,
        validate_weight_matrix,
    )


EVIDENCE_CLASS = "post_holdout_explanatory"
INTERPRETATION = "descriptive_same_sample_not_confirmatory"
SHA256_LENGTH = 64
Runner = Callable[[list[str], Path, dict[str, str], Path, str], float]


class SecondaryAblationError(RuntimeError):
    """Raised whenever the explanatory batch cannot prove input equivalence."""


@dataclass(frozen=True)
class SeedCheckpoint:
    seed: int
    directory: Path
    checkpoint: Path
    checkpoint_sha256: str
    model: str
    observation_mode: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    text = str(value).strip().lower()
    return len(text) == SHA256_LENGTH and all(char in "0123456789abcdef" for char in text)


def require_fields(value: dict[str, Any], fields: Iterable[str], label: str) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise SecondaryAblationError(f"{label} is missing fields: {', '.join(missing)}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SecondaryAblationError(f"JSON file not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SecondaryAblationError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise SecondaryAblationError(f"Expected a JSON object: {path}")
    return value


def parse_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise SecondaryAblationError(f"{label} is not boolean: {value!r}")


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    require_fields(
        contract,
        [
            "schema_version", "protocol_id", "evidence_class",
            "same_sample_interpretation", "confirmatory_claims_permitted",
            "successful_v4", "evaluation", "matched_design", "experiments", "outputs",
        ],
        "secondary ablation contract",
    )
    if int(contract["schema_version"]) != 1:
        raise SecondaryAblationError("Only secondary ablation schema_version=1 is supported.")
    if contract["evidence_class"] != EVIDENCE_CLASS:
        raise SecondaryAblationError(f"evidence_class must be {EVIDENCE_CLASS}.")
    if contract["same_sample_interpretation"] != INTERPRETATION:
        raise SecondaryAblationError(f"same_sample_interpretation must be {INTERPRETATION}.")
    if parse_bool(contract["confirmatory_claims_permitted"], "confirmatory_claims_permitted"):
        raise SecondaryAblationError("Confirmatory claims are forbidden on the consumed holdout.")

    successful = contract["successful_v4"]
    if not isinstance(successful, dict):
        raise SecondaryAblationError("successful_v4 must be an object.")
    require_fields(
        successful,
        ["archive_sha256", "evaluation_id", "required_artifacts"],
        "successful_v4",
    )
    if not is_sha256(successful["archive_sha256"]):
        raise SecondaryAblationError("successful_v4.archive_sha256 is invalid.")
    artifacts = successful["required_artifacts"]
    required_artifact_names = {
        "batch_manifest", "realized_panel", "strategy_manifest",
        "evaluation_run_manifest", "input_hashes", "scored_monthly_panel",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != required_artifact_names:
        raise SecondaryAblationError(
            "successful_v4.required_artifacts must declare exactly: "
            + ", ".join(sorted(required_artifact_names))
        )
    for label, relative in artifacts.items():
        safe_relative(str(relative), f"successful_v4.required_artifacts.{label}")

    evaluation = contract["evaluation"]
    if not isinstance(evaluation, dict):
        raise SecondaryAblationError("evaluation must be an object.")
    require_fields(
        evaluation,
        [
            "economics_fields", "expected_economics", "benchmark_strategy_ids",
            "archive_score_tolerance", "inference_replay_weight_tolerance",
            "weight_tolerance",
        ],
        "evaluation",
    )
    economics = list(evaluation["economics_fields"])
    if not economics or len(economics) != len(set(economics)):
        raise SecondaryAblationError("evaluation.economics_fields must be unique and non-empty.")
    if set(economics) != set(evaluation["expected_economics"]):
        raise SecondaryAblationError("expected_economics must cover every economics field exactly.")
    benchmark_ids = list(evaluation["benchmark_strategy_ids"])
    if not benchmark_ids or len(benchmark_ids) != len(set(benchmark_ids)):
        raise SecondaryAblationError("benchmark_strategy_ids must be unique and non-empty.")
    for name in [
        "archive_score_tolerance", "inference_replay_weight_tolerance", "weight_tolerance"
    ]:
        value = float(evaluation[name])
        if not np.isfinite(value) or value <= 0:
            raise SecondaryAblationError(f"evaluation.{name} must be finite and positive.")
    if float(evaluation["inference_replay_weight_tolerance"]) > float(
        evaluation["weight_tolerance"]
    ):
        raise SecondaryAblationError(
            "Inference replay tolerance cannot exceed the portfolio weight tolerance."
        )

    matched = contract["matched_design"]
    if not isinstance(matched, dict):
        raise SecondaryAblationError("matched_design must be an object.")
    require_fields(
        matched,
        [
            "required_equal_code_paths", "require_equal_actor_parameter_count",
            "require_equal_observation_and_action_dimensions",
            "require_equal_pretraining_update_count", "full_update_count_rule",
        ],
        "matched_design",
    )
    code_paths = matched["required_equal_code_paths"]
    if (
        not isinstance(code_paths, list)
        or not code_paths
        or len(code_paths) != len(set(code_paths))
    ):
        raise SecondaryAblationError(
            "matched_design.required_equal_code_paths must be unique and non-empty."
        )
    for path in code_paths:
        safe_relative(str(path), "matched_design.required_equal_code_paths")
    for field in [
        "require_equal_actor_parameter_count",
        "require_equal_observation_and_action_dimensions",
        "require_equal_pretraining_update_count",
    ]:
        if not parse_bool(matched[field], f"matched_design.{field}"):
            raise SecondaryAblationError(f"matched_design.{field} must be true.")
    if matched["full_update_count_rule"] != "at_least_pretraining_update_count":
        raise SecondaryAblationError("Unsupported full_update_count_rule.")

    experiments = contract["experiments"]
    if not isinstance(experiments, dict) or set(experiments) != {
        "full_reference", "no_vine", "pretrained_only"
    }:
        raise SecondaryAblationError(
            "experiments must contain exactly full_reference, no_vine, and pretrained_only."
        )
    expected = {
        "full_reference": ("successful_v4_archive", "full", "full"),
        "no_vine": ("no_vine_training_release", "full", "zero"),
        "pretrained_only": ("full_training_release", "pretrained", "full"),
    }
    for name, (source, model, mode) in expected.items():
        item = experiments[name]
        require_fields(
            item,
            [
                "evidence_class", "source", "checkpoint_model", "observation_mode",
                "expected_seeds", "ensemble_strategy_id",
            ],
            f"experiments.{name}",
        )
        if item["evidence_class"] != EVIDENCE_CLASS:
            raise SecondaryAblationError(f"{name} is not labelled {EVIDENCE_CLASS}.")
        if (item["source"], item["checkpoint_model"], item["observation_mode"]) != (
            source, model, mode
        ):
            raise SecondaryAblationError(f"{name} source/model/observation contract is invalid.")
        seeds = item["expected_seeds"]
        if not isinstance(seeds, list) or not seeds or not all(isinstance(seed, int) for seed in seeds):
            raise SecondaryAblationError(f"{name}.expected_seeds must be a non-empty integer list.")
        if len(seeds) != len(set(seeds)):
            raise SecondaryAblationError(f"{name}.expected_seeds contains duplicates.")
    if experiments["pretrained_only"]["expected_seeds"] != experiments["full_reference"]["expected_seeds"]:
        raise SecondaryAblationError("Pretrained and full reference seeds must be paired and identical.")
    if experiments["no_vine"].get("ablation_scope") != (
        "policy_visible_vine_state_only_reward_cvar_retained"
    ):
        raise SecondaryAblationError(
            "The state ablation must disclose that vine-scenario CVaR remains in the reward."
        )
    ensemble_ids = [item["ensemble_strategy_id"] for item in experiments.values()]
    if len(ensemble_ids) != len(set(ensemble_ids)):
        raise SecondaryAblationError("Ensemble strategy identifiers must be unique.")

    outputs = contract["outputs"]
    require_fields(outputs, ["release_status", "same_sample_tests"], "outputs")
    if outputs["release_status"] != "frozen_post_holdout_explanatory_ablation":
        raise SecondaryAblationError("Invalid explanatory release status.")
    if outputs["same_sample_tests"] != "descriptive_only_no_confirmatory_tests":
        raise SecondaryAblationError("Same-sample tests must be explicitly descriptive only.")
    return contract


def safe_relative(raw: str, label: str) -> PurePosixPath:
    normalized = raw.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if not normalized or pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise SecondaryAblationError(f"Unsafe {label}: {raw!r}")
    return pure


def parse_sidecar(sidecar: Path, archive: Path) -> str:
    if not sidecar.is_file() or not archive.is_file():
        raise SecondaryAblationError("Successful archive or SHA-256 sidecar is missing.")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if not fields or not is_sha256(fields[0]):
        raise SecondaryAblationError(f"Invalid archive sidecar: {sidecar}")
    expected = fields[0].lower()
    actual = sha256_file(archive)
    if expected != actual:
        raise SecondaryAblationError(
            f"Successful archive sidecar mismatch: expected {expected}, found {actual}."
        )
    return actual


def archive_index(archive: Path) -> tuple[str, dict[str, str]]:
    roots: set[str] = set()
    files: dict[str, str] = {}
    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            raw = member.name.replace("\\", "/").rstrip("/")
            if not raw:
                continue
            pure = safe_relative(raw, "archive member")
            roots.add(pure.parts[0])
            if member.issym() or member.islnk() or member.isdev():
                raise SecondaryAblationError(f"Unsafe archive member type: {member.name}")
            if member.isfile():
                if len(pure.parts) < 2:
                    raise SecondaryAblationError("Archive files must live below one top-level directory.")
                relative = PurePosixPath(*pure.parts[1:]).as_posix()
                if relative in files:
                    raise SecondaryAblationError(f"Duplicate archive path: {relative}")
                files[relative] = member.name
    if len(roots) != 1 or not files:
        raise SecondaryAblationError("Archive must contain exactly one non-empty top-level directory.")
    return next(iter(roots)), files


def archive_bytes(archive: Path, index: dict[str, str], relative: str) -> bytes:
    normalized = safe_relative(relative, "required archive artifact").as_posix()
    member_name = index.get(normalized)
    if member_name is None:
        raise SecondaryAblationError(f"Required v4 artifact is absent: {normalized}")
    with tarfile.open(archive, "r:*") as handle:
        stream = handle.extractfile(member_name)
        if stream is None:
            raise SecondaryAblationError(f"Could not read v4 artifact: {normalized}")
        return stream.read()


def csv_bytes(value: bytes, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(value))
    except Exception as error:
        raise SecondaryAblationError(f"Invalid CSV in {label}: {error}") from error


def json_bytes(value: bytes, label: str) -> dict[str, Any]:
    try:
        result = json.loads(value.decode("utf-8"))
    except Exception as error:
        raise SecondaryAblationError(f"Invalid JSON in {label}: {error}") from error
    if not isinstance(result, dict):
        raise SecondaryAblationError(f"Expected JSON object in {label}.")
    return result


def verify_contents(release: Path) -> dict[str, str]:
    contents = release / "CONTENTS.sha256"
    if not release.is_dir() or not contents.is_file():
        raise SecondaryAblationError(f"Frozen training release is incomplete: {release}")
    declared: dict[str, str] = {}
    for line in contents.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as error:
            raise SecondaryAblationError(f"Malformed CONTENTS.sha256 line in {release}: {line!r}") from error
        relative = safe_relative(relative, "release checksum path").as_posix()
        if relative in declared or not is_sha256(expected):
            raise SecondaryAblationError(f"Duplicate/invalid release checksum: {relative}")
        path = release / Path(relative)
        if not path.is_file() or sha256_file(path) != expected.lower():
            raise SecondaryAblationError(f"Frozen release checksum mismatch: {relative}")
        declared[relative] = expected.lower()
    actual = {
        path.relative_to(release).as_posix()
        for path in release.rglob("*")
        if path.is_file() and path.name != "CONTENTS.sha256"
    }
    if set(declared) != actual:
        raise SecondaryAblationError(
            f"Frozen release checksum coverage mismatch; undeclared={sorted(actual-set(declared))}, "
            f"missing={sorted(set(declared)-actual)}."
        )
    return declared


def validate_training_release(
    release: Path,
    expected_seeds: list[int],
    model: str,
    expected_mode: str,
    allow_legacy_full_mode: bool,
) -> list[SeedCheckpoint]:
    release = release.resolve()
    checksums = verify_contents(release)
    manifest = load_json(release / "training_release_manifest.json")
    holdout_accessed = parse_bool(
        manifest.get("holdout_accessed_by_freezer"), "holdout_accessed_by_freezer"
    )
    if expected_mode == "zero":
        valid_release = (
            manifest.get("release_status") ==
            "frozen_post_holdout_explanatory_training"
            and manifest.get("evidence_class") == "post_holdout_explanatory"
            and parse_bool(
                manifest.get("confirmatory_claims_permitted"),
                "confirmatory_claims_permitted",
            ) is False
            and not holdout_accessed
        )
        label = "post-holdout explanatory no-vine"
    else:
        valid_release = (
            manifest.get("release_status") == "frozen_pre_oos"
            and not holdout_accessed
        )
        label = "pre-OOS full-model"
    if not valid_release:
        raise SecondaryAblationError(
            f"Training release is not a valid frozen {label} release: {release}"
        )
    directories = sorted((release / "seeds").glob("seed_*"))
    observed: dict[int, Path] = {}
    for directory in directories:
        try:
            seed = int(directory.name.removeprefix("seed_"))
        except ValueError as error:
            raise SecondaryAblationError(f"Invalid seed directory: {directory}") from error
        if seed in observed:
            raise SecondaryAblationError(f"Duplicate seed directory: {seed}")
        observed[seed] = directory
    if set(observed) != set(expected_seeds):
        raise SecondaryAblationError(
            f"Frozen release seed mismatch: expected={sorted(expected_seeds)}, observed={sorted(observed)}"
        )
    rows: list[SeedCheckpoint] = []
    for seed in expected_seeds:
        directory = observed[seed]
        checkpoint = directory / f"td3_lstm_vine_{model}.pt"
        relative = checkpoint.relative_to(release).as_posix()
        if relative not in checksums:
            raise SecondaryAblationError(f"Checkpoint is not checksum-declared: {checkpoint}")
        mode_file = directory / "vine_observation_mode.txt"
        if mode_file.is_file():
            recorded_mode = mode_file.read_text(encoding="utf-8").strip()
            if recorded_mode != expected_mode:
                raise SecondaryAblationError(
                    f"Seed {seed} observation mode is {recorded_mode!r}; expected {expected_mode!r}."
                )
        elif not (expected_mode == "full" and allow_legacy_full_mode):
            raise SecondaryAblationError(f"Seed {seed} lacks vine_observation_mode={expected_mode} evidence.")
        rows.append(
            SeedCheckpoint(
                seed=seed,
                directory=directory,
                checkpoint=checkpoint,
                checkpoint_sha256=checksums[relative],
                model=model,
                observation_mode=expected_mode,
            )
        )
    return rows


def verify_secondary_plan_release(release: Path, repo_root: Path) -> dict[str, Any]:
    """Prove that execution uses the exact live bytes frozen in the plan."""
    release = release.resolve()
    verify_contents(release)
    manifest = load_json(release / "secondary_plan_manifest.json")
    if (
        manifest.get("release_status") != "frozen_post_holdout_secondary_plan"
        or manifest.get("evidence_class") != EVIDENCE_CLASS
        or manifest.get("main_result_immutable") is not True
    ):
        raise SecondaryAblationError("Secondary plan release is invalid or relabelled.")
    inventory_path = release / "source_inventory.csv"
    snapshot = release / "source_snapshot"
    if not inventory_path.is_file() or not snapshot.is_dir():
        raise SecondaryAblationError("Secondary plan lacks its source inventory/snapshot.")
    inventory = pd.read_csv(inventory_path)
    required = {"path", "sha256", "size_bytes"}
    if not required.issubset(inventory.columns) or inventory.empty:
        raise SecondaryAblationError("Secondary plan source inventory is malformed.")
    if inventory["path"].astype(str).duplicated().any():
        raise SecondaryAblationError("Secondary plan source inventory contains duplicates.")
    for row in inventory.to_dict("records"):
        relative = Path(safe_relative(str(row["path"]), "secondary plan source path"))
        expected = str(row["sha256"]).lower()
        live = repo_root / relative
        frozen = snapshot / relative
        if not is_sha256(expected) or not live.is_file() or not frozen.is_file():
            raise SecondaryAblationError(f"Secondary plan source is missing: {relative}")
        if sha256_file(live) != expected or sha256_file(frozen) != expected:
            raise SecondaryAblationError(
                f"Live/frozen source drift detected before ablation execution: {relative}"
            )
    return {
        "release": str(release),
        "contents_sha256": sha256_file(release / "CONTENTS.sha256"),
        "source_count": len(inventory),
        "source_inventory_sha256": manifest.get("source_inventory_sha256"),
    }


def _read_seed_protocol_rows(
    release: Path, seeds: list[int], expected_mode: str
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read frozen, checksum-covered architecture/budget evidence."""
    snapshot_inventory = release / "training_snapshot_inventory.csv"
    if not snapshot_inventory.is_file():
        raise SecondaryAblationError(
            f"Training release lacks training_snapshot_inventory.csv: {release}"
        )
    snapshot = pd.read_csv(snapshot_inventory)
    required_snapshot = {"artifact_kind", "normalized_path", "expected_md5"}
    if not required_snapshot.issubset(snapshot.columns):
        raise SecondaryAblationError("Training snapshot inventory is missing hash columns.")
    code = snapshot[snapshot["artifact_kind"].astype(str) == "code"]
    if code["normalized_path"].astype(str).duplicated().any():
        raise SecondaryAblationError("Training snapshot has duplicate code paths.")
    code_hashes = {
        str(row["normalized_path"]).replace("\\", "/"): str(row["expected_md5"]).lower()
        for row in code.to_dict("records")
    }

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        directory = release / "seeds" / f"seed_{seed}"
        integrity_path = directory / "sanity_no_holdout" / "checkpoint_integrity.csv"
        report_path = directory / "sanity_no_holdout" / "sanity_report.json"
        if not integrity_path.is_file() or not report_path.is_file():
            raise SecondaryAblationError(
                f"Seed {seed} lacks frozen architecture/budget diagnostics."
            )
        integrity = pd.read_csv(integrity_path)
        required_integrity = {
            "model", "architecture_match", "all_checkpoint_tensors_finite",
            "actor_parameters", "update_count",
        }
        if not required_integrity.issubset(integrity.columns):
            raise SecondaryAblationError(
                f"Seed {seed} checkpoint integrity lacks matched-design fields."
            )
        report = load_json(report_path)
        if str(report.get("vine_observation_mode", "")) != expected_mode:
            raise SecondaryAblationError(
                f"Seed {seed} sanity mode differs from {expected_mode!r}."
            )
        if not parse_bool(report.get("overall_pass", False), f"seed {seed} overall_pass"):
            raise SecondaryAblationError(f"Seed {seed} did not pass its frozen sanity gate.")
        model_rows: dict[str, dict[str, Any]] = {}
        for model in ["pretrained", "full"]:
            selected = integrity[integrity["model"].astype(str) == model]
            if len(selected) != 1:
                raise SecondaryAblationError(
                    f"Seed {seed} needs exactly one {model} checkpoint-integrity row."
                )
            record = selected.iloc[0].to_dict()
            if not parse_bool(record["architecture_match"], f"seed {seed} architecture_match"):
                raise SecondaryAblationError(f"Seed {seed} {model} architecture mismatch.")
            if not parse_bool(
                record["all_checkpoint_tensors_finite"],
                f"seed {seed} all_checkpoint_tensors_finite",
            ):
                raise SecondaryAblationError(f"Seed {seed} {model} checkpoint is non-finite.")
            model_rows[model] = record
        pretrained_updates = int(model_rows["pretrained"]["update_count"])
        full_updates = int(model_rows["full"]["update_count"])
        if pretrained_updates <= 0 or full_updates < pretrained_updates:
            raise SecondaryAblationError(
                f"Seed {seed} violates the declared training-update budget ordering."
            )
        rows.append(
            {
                "seed": seed,
                "vine_observation_mode": expected_mode,
                "obs_dim": int(report["obs_dim"]),
                "action_dim": int(report["action_dim"]),
                "vine_dim": int(report["vine_dim"]),
                "pretrained_actor_parameters": int(
                    model_rows["pretrained"]["actor_parameters"]
                ),
                "full_actor_parameters": int(model_rows["full"]["actor_parameters"]),
                "pretrained_update_count": pretrained_updates,
                "full_update_count": full_updates,
            }
        )
    return pd.DataFrame(rows), code_hashes


def verify_matched_training_design(
    full_release: Path,
    full_seeds: list[int],
    no_vine_release: Path,
    no_vine_seeds: list[int],
    matched_contract: dict[str, Any],
) -> pd.DataFrame:
    full, full_code = _read_seed_protocol_rows(full_release, full_seeds, "full")
    ablation, ablation_code = _read_seed_protocol_rows(
        no_vine_release, no_vine_seeds, "zero"
    )
    rows: list[dict[str, Any]] = []
    for path in matched_contract["required_equal_code_paths"]:
        full_hash = full_code.get(str(path))
        ablation_hash = ablation_code.get(str(path))
        passed = bool(full_hash and full_hash == ablation_hash)
        rows.append(
            {
                "check": f"equal_code:{path}",
                "full_value": full_hash or "missing",
                "ablation_value": ablation_hash or "missing",
                "status": "pass" if passed else "fail",
            }
        )
    for field in [
        "obs_dim", "action_dim", "vine_dim", "pretrained_actor_parameters",
        "full_actor_parameters", "pretrained_update_count",
    ]:
        full_values = sorted(set(int(value) for value in full[field]))
        ablation_values = sorted(set(int(value) for value in ablation[field]))
        passed = len(full_values) == 1 and full_values == ablation_values
        rows.append(
            {
                "check": f"matched_design:{field}",
                "full_value": json.dumps(full_values),
                "ablation_value": json.dumps(ablation_values),
                "status": "pass" if passed else "fail",
            }
        )
    report = pd.DataFrame(rows)
    if (report["status"] != "pass").any():
        failed = report.loc[report["status"] != "pass", "check"].tolist()
        raise SecondaryAblationError(
            "Matched-capacity/training-budget validation failed: " + ", ".join(failed)
        )
    report.insert(0, "same_sample_interpretation", INTERPRETATION)
    report.insert(0, "evidence_class", EVIDENCE_CLASS)
    return report


def hash_table_map(frame: pd.DataFrame) -> dict[str, str]:
    required = {"artifact", "sha256"}
    if not required.issubset(frame.columns):
        raise SecondaryAblationError("v4 input_hashes.csv lacks artifact/sha256 columns.")
    result: dict[str, str] = {}
    for row in frame.to_dict("records"):
        key = str(row["artifact"])
        digest = str(row["sha256"]).lower()
        generated_equal_weight = key == "weights:equal_weight" and digest == "generated"
        if key in result or (not is_sha256(digest) and not generated_equal_weight):
            raise SecondaryAblationError(f"Duplicate/invalid v4 artifact hash row: {key}")
        result[key] = digest
    return result


def read_weights_bytes(
    value: bytes,
    label: str,
    realized: pd.DataFrame,
    assets: list[str],
    contract: Contract,
) -> pd.DataFrame:
    frame = csv_bytes(value, label)
    return validate_weights(frame, label, realized, assets, contract)


def validate_weights(
    frame: pd.DataFrame,
    label: str,
    realized: pd.DataFrame,
    assets: list[str],
    contract: Contract,
) -> pd.DataFrame:
    frame = frame.copy()
    weight_columns = [f"w_{asset}" for asset in assets]
    missing = [column for column in KEYS + weight_columns if column not in frame.columns]
    if missing:
        raise SecondaryAblationError(f"{label} lacks columns: {', '.join(missing)}")
    for column in ["decision_date", "holding_end_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    frame["window_id"] = frame["window_id"].astype(str)
    realized = realized.copy()
    realized["window_id"] = realized["window_id"].astype(str)
    if frame[KEYS].duplicated().any():
        raise SecondaryAblationError(f"{label} contains duplicate period keys.")
    expected = realized[KEYS].copy()
    observed = frame[KEYS].copy()
    for column in ["decision_date", "holding_end_date"]:
        expected[column] = expected[column].astype(str)
        observed[column] = observed[column].astype(str)
    expected["window_id"] = expected["window_id"].astype(str)
    observed["window_id"] = observed["window_id"].astype(str)
    expected = expected.sort_values(KEYS, kind="stable").reset_index(drop=True)
    observed = observed.sort_values(KEYS, kind="stable").reset_index(drop=True)
    if len(observed) != len(expected) or not observed.equals(expected):
        raise SecondaryAblationError(f"{label} does not contain the exact consumed-holdout period keys.")
    normalized = realized[KEYS].merge(frame[KEYS + weight_columns], on=KEYS, validate="one_to_one")
    matrix = normalized[weight_columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    try:
        validate_weight_matrix(matrix, label, contract)
    except EvaluationProtocolError as error:
        raise SecondaryAblationError(str(error)) from error
    normalized[weight_columns] = matrix
    return normalized


def verify_successful_v4(
    archive: Path,
    sidecar: Path,
    contract_definition: dict[str, Any],
    evaluation_contract_path: Path,
    full_release_rows: list[SeedCheckpoint],
) -> dict[str, Any]:
    actual_archive_hash = parse_sidecar(sidecar, archive)
    expected_archive_hash = str(contract_definition["successful_v4"]["archive_sha256"]).lower()
    if actual_archive_hash != expected_archive_hash:
        raise SecondaryAblationError(
            f"Wrong successful archive: expected {expected_archive_hash}, found {actual_archive_hash}."
        )
    _, index = archive_index(archive)
    paths = contract_definition["successful_v4"]["required_artifacts"]
    raw = {label: archive_bytes(archive, index, relative) for label, relative in paths.items()}
    batch_manifest = json_bytes(raw["batch_manifest"], "v4 batch manifest")
    if batch_manifest.get("status") != "complete" or batch_manifest.get("holdout_accessed") is not True:
        raise SecondaryAblationError("The supplied v4 archive is not the completed consumed-holdout batch.")
    run_manifest = json_bytes(raw["evaluation_run_manifest"], "v4 evaluation run manifest")
    expected_id = contract_definition["successful_v4"]["evaluation_id"]
    if run_manifest.get("evaluation_id") != expected_id:
        raise SecondaryAblationError("v4 evaluation_id does not match the explanatory contract.")
    actual_evaluation_contract_hash = sha256_file(evaluation_contract_path)
    if run_manifest.get("contract_sha256") != actual_evaluation_contract_hash:
        raise SecondaryAblationError("Evaluation contract bytes do not match the successful v4 run.")
    if run_manifest.get("strategy_manifest_sha256") != sha256_bytes(raw["strategy_manifest"]):
        raise SecondaryAblationError("v4 strategy manifest bytes do not match its run manifest.")
    artifact_hashes = hash_table_map(csv_bytes(raw["input_hashes"], "v4 input hashes"))
    if artifact_hashes.get("evaluation_contract") != actual_evaluation_contract_hash:
        raise SecondaryAblationError("v4 artifact registry does not link the evaluation contract.")
    if artifact_hashes.get("strategy_manifest") != sha256_bytes(raw["strategy_manifest"]):
        raise SecondaryAblationError("v4 strategy manifest hash does not match its artifact registry.")
    if sha256_bytes(raw["realized_panel"]) != run_manifest.get("realized_panel_sha256"):
        raise SecondaryAblationError("v4 realized panel hash does not match its run manifest.")
    if artifact_hashes.get("realized_panel") != sha256_bytes(raw["realized_panel"]):
        raise SecondaryAblationError("v4 realized panel hash does not match its artifact registry.")

    strategy_manifest = csv_bytes(raw["strategy_manifest"], "v4 strategy manifest")
    required_manifest = {
        "strategy_id", "seed", "role", "weight_log_path", "weight_log_sha256",
        "checkpoint_sha256",
    }
    if not required_manifest.issubset(strategy_manifest.columns):
        raise SecondaryAblationError("v4 strategy manifest lacks required provenance columns.")
    if strategy_manifest["strategy_id"].astype(str).duplicated().any():
        raise SecondaryAblationError("v4 strategy manifest contains duplicate strategy IDs.")

    evaluation_contract = Contract.read(evaluation_contract_path)
    if evaluation_contract["evaluation_id"] != expected_id:
        raise SecondaryAblationError("Evaluation contract ID differs from v4.")
    expected_economics = contract_definition["evaluation"]["expected_economics"]
    for field in contract_definition["evaluation"]["economics_fields"]:
        if evaluation_contract.get(field) != expected_economics[field]:
            raise SecondaryAblationError(
                f"Evaluation economics mismatch for {field}: "
                f"expected {expected_economics[field]!r}, found {evaluation_contract.get(field)!r}."
            )

    with tempfile.TemporaryDirectory() as directory:
        realized_path = Path(directory) / "realized.csv"
        realized_path.write_bytes(raw["realized_panel"])
        try:
            realized, assets = read_realized_panel(realized_path, evaluation_contract)
        except EvaluationProtocolError as error:
            raise SecondaryAblationError(str(error)) from error

    benchmark_ids = contract_definition["evaluation"]["benchmark_strategy_ids"]
    expected_full_seeds = contract_definition["experiments"]["full_reference"]["expected_seeds"]
    selected: dict[str, dict[str, Any]] = {}
    copied_artifacts: dict[str, bytes] = {
        paths["realized_panel"]: raw["realized_panel"],
        paths["strategy_manifest"]: raw["strategy_manifest"],
        paths["evaluation_run_manifest"]: raw["evaluation_run_manifest"],
        paths["input_hashes"]: raw["input_hashes"],
        paths["scored_monthly_panel"]: raw["scored_monthly_panel"],
        paths["batch_manifest"]: raw["batch_manifest"],
    }

    release_full_hashes = {row.seed: row.checkpoint_sha256 for row in full_release_rows}
    for strategy_id in benchmark_ids:
        rows = strategy_manifest[strategy_manifest["strategy_id"].astype(str) == strategy_id]
        if len(rows) != 1:
            raise SecondaryAblationError(f"Expected one v4 benchmark strategy row: {strategy_id}")
        row = rows.iloc[0].to_dict()
        path = str(row["weight_log_path"])
        if path == "GENERATE_EQUAL_WEIGHT":
            selected[strategy_id] = {"row": row, "weights": None, "source_path": path}
            continue
        value = archive_bytes(archive, index, path)
        digest = sha256_bytes(value)
        if digest != str(row["weight_log_sha256"]).lower() or artifact_hashes.get(
            f"weights:{strategy_id}"
        ) != digest:
            raise SecondaryAblationError(f"v4 benchmark weight provenance mismatch: {strategy_id}")
        weights = read_weights_bytes(value, strategy_id, realized, assets, evaluation_contract)
        copied_artifacts[path] = value
        selected[strategy_id] = {"row": row, "weights": weights, "source_path": path}

    full_weights: dict[int, pd.DataFrame] = {}
    full_paths: dict[int, str] = {}
    for seed in expected_full_seeds:
        strategy_id = f"vine_td3_seed_{seed}"
        rows = strategy_manifest[strategy_manifest["strategy_id"].astype(str) == strategy_id]
        if len(rows) != 1:
            raise SecondaryAblationError(f"Expected one v4 full-policy row: {strategy_id}")
        row = rows.iloc[0].to_dict()
        if str(row["checkpoint_sha256"]).lower() != release_full_hashes.get(seed):
            raise SecondaryAblationError(f"Full checkpoint linkage mismatch for seed {seed}.")
        path = str(row["weight_log_path"])
        value = archive_bytes(archive, index, path)
        digest = sha256_bytes(value)
        if digest != str(row["weight_log_sha256"]).lower() or artifact_hashes.get(
            f"weights:{strategy_id}"
        ) != digest:
            raise SecondaryAblationError(f"v4 full weight provenance mismatch for seed {seed}.")
        full_weights[seed] = read_weights_bytes(
            value, strategy_id, realized, assets, evaluation_contract
        )
        full_paths[seed] = path
        copied_artifacts[path] = value
    return {
        "archive_sha256": actual_archive_hash,
        "archive_index": index,
        "realized": realized,
        "assets": assets,
        "evaluation_contract": evaluation_contract,
        "evaluation_contract_sha256": actual_evaluation_contract_hash,
        "strategy_manifest": strategy_manifest,
        "artifact_hashes": artifact_hashes,
        "archived_scored": csv_bytes(
            raw["scored_monthly_panel"], "v4 scored monthly panel"
        ),
        "benchmarks": selected,
        "full_weights": full_weights,
        "full_paths": full_paths,
        "copied_artifacts": copied_artifacts,
    }


def run_logged(command: list[str], cwd: Path, env: dict[str, str], logs: Path, label: str) -> float:
    start = time.monotonic()
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    elapsed = time.monotonic() - start
    (logs / f"{label}.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (logs / f"{label}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise SecondaryAblationError(
            f"Explanatory checkpoint evaluation {label} failed with exit code {result.returncode}."
        )
    return elapsed


def generate_weights(
    rows: list[SeedCheckpoint],
    group: str,
    model: str,
    mode: str,
    destination: Path,
    logs: Path,
    repo_root: Path,
    runtime_config: Path,
    rscript: str,
    evaluation_id: str,
    runner: Runner,
    realized: pd.DataFrame,
    assets: list[str],
    evaluation_contract: Contract,
) -> tuple[dict[int, pd.DataFrame], list[dict[str, Any]], dict[str, float]]:
    destination.mkdir(parents=True, exist_ok=False)
    result: dict[int, pd.DataFrame] = {}
    inventory: list[dict[str, Any]] = []
    timings: dict[str, float] = {}
    base_env = os.environ.copy()
    base_env.update({"LC_ALL": "C", "LANG": "C", "LANGUAGE": "C", "TZ": "UTC"})
    for item in rows:
        env = base_env.copy()
        env.update(
            {
                "EVAL_MODEL_DIR": str(item.directory),
                "EVAL_OUTPUT_DIR": str(destination),
                "EVAL_WEIGHTS_ONLY": "true",
                "EVAL_CHECKPOINT_MODELS": model,
                "VINE_OBSERVATION_MODE": mode,
                "EVAL_WINDOW_ID": evaluation_id,
            }
        )
        label = f"{group}_{item.seed}_{model}"
        timings[label] = runner(
            [
                rscript, "--vanilla", "evaluate_with_config.r",
                str(runtime_config), str(item.directory),
            ],
            repo_root,
            env,
            logs,
            label,
        )
        output = destination / f"weights_rl_{model}_{item.directory.name}.csv"
        if not output.is_file():
            raise SecondaryAblationError(f"Expected generated weight file is missing: {output}")
        frame = validate_weights(
            pd.read_csv(output), f"{group} seed {item.seed}", realized, assets, evaluation_contract
        )
        result[item.seed] = frame
        inventory.append(
            {
                "evidence_class": EVIDENCE_CLASS,
                "same_sample_interpretation": INTERPRETATION,
                "group": group,
                "seed": item.seed,
                "checkpoint_model": model,
                "observation_mode": mode,
                "checkpoint_sha256": item.checkpoint_sha256,
                "weight_path": output.name,
                "weight_sha256": sha256_file(output),
                "evaluation_seconds": timings[label],
            }
        )
    if set(result) != {item.seed for item in rows}:
        raise SecondaryAblationError(f"Incomplete generated weight set for {group}.")
    return result, inventory, timings


def verify_full_inference_replay(
    generated: dict[int, pd.DataFrame],
    archived: dict[int, pd.DataFrame],
    assets: list[str],
    tolerance: float,
) -> pd.DataFrame:
    """Fail if live inference/data code cannot reproduce every archived v4 seed."""
    if set(generated) != set(archived):
        raise SecondaryAblationError("Full-policy replay seed set differs from archived v4.")
    weight_columns = [f"w_{asset}" for asset in assets]
    rows: list[dict[str, Any]] = []
    for seed in sorted(archived):
        current = generated[seed]
        frozen = archived[seed]
        if not current[KEYS].reset_index(drop=True).equals(
            frozen[KEYS].reset_index(drop=True)
        ):
            raise SecondaryAblationError(
                f"Full-policy replay period keys differ for seed {seed}."
            )
        current_values = current[weight_columns].to_numpy(float)
        frozen_values = frozen[weight_columns].to_numpy(float)
        if not np.isfinite(current_values).all() or not np.isfinite(frozen_values).all():
            raise SecondaryAblationError(
                f"Full-policy replay contains non-finite weights for seed {seed}."
            )
        maximum = float(np.max(np.abs(current_values - frozen_values)))
        if maximum > tolerance:
            raise SecondaryAblationError(
                "Live policy/data machinery does not reproduce archived v4 full weights "
                f"for seed {seed}; max error={maximum:.6g}, tolerance={tolerance:.6g}."
            )
        rows.append(
            {
                "evidence_class": EVIDENCE_CLASS,
                "same_sample_interpretation": INTERPRETATION,
                "seed": seed,
                "rows_verified": len(current),
                "maximum_absolute_weight_error": maximum,
                "tolerance": tolerance,
                "status": "exact_within_tolerance",
            }
        )
    return pd.DataFrame(rows)


def mean_ensemble(
    members: dict[int, pd.DataFrame],
    strategy_id: str,
    assets: list[str],
    contract: Contract,
) -> pd.DataFrame:
    if not members:
        raise SecondaryAblationError(f"Cannot construct empty ensemble {strategy_id}.")
    weight_columns = [f"w_{asset}" for asset in assets]
    ordered = [members[seed] for seed in sorted(members)]
    reference = ordered[0][KEYS].reset_index(drop=True)
    arrays: list[np.ndarray] = []
    for frame in ordered:
        if not frame[KEYS].reset_index(drop=True).equals(reference):
            raise SecondaryAblationError(f"Ensemble member key mismatch for {strategy_id}.")
        arrays.append(frame[weight_columns].to_numpy(float))
    ensemble = reference.copy()
    ensemble[weight_columns] = np.stack(arrays).mean(axis=0)
    try:
        validate_weight_matrix(ensemble[weight_columns].to_numpy(float), strategy_id, contract)
    except EvaluationProtocolError as error:
        raise SecondaryAblationError(str(error)) from error
    return ensemble


def equal_weight(realized: pd.DataFrame, assets: list[str], contract: Contract) -> pd.DataFrame:
    frame = realized[KEYS].copy()
    value = float(contract["net_exposure"]) / len(assets)
    for asset in assets:
        frame[f"w_{asset}"] = value
    validate_weight_matrix(
        frame[[f"w_{asset}" for asset in assets]].to_numpy(float), "equal_weight", contract
    )
    return frame


def verify_archived_scores(
    recomputed: dict[str, pd.DataFrame],
    archived: pd.DataFrame,
    assets: list[str],
    tolerance: float,
) -> pd.DataFrame:
    columns = [
        "gross_return", "net_return", "turnover", "transaction_cost", "financing_cost",
        *[f"w_{asset}" for asset in assets],
    ]
    required = {"strategy_id", *KEYS, *columns}
    if not required.issubset(archived.columns):
        raise SecondaryAblationError("v4 scored panel lacks columns required for economic replay.")
    rows: list[dict[str, Any]] = []
    for strategy_id, frame in recomputed.items():
        frozen = archived[archived["strategy_id"].astype(str) == strategy_id].copy()
        if len(frozen) != len(frame):
            raise SecondaryAblationError(f"v4 scored row count mismatch for {strategy_id}.")
        current = frame.copy()
        for value in [current, frozen]:
            value["window_id"] = value["window_id"].astype(str)
            for column in ["decision_date", "holding_end_date"]:
                value[column] = pd.to_datetime(value[column], errors="raise")
        merged = current.merge(
            frozen[KEYS + columns], on=KEYS, how="inner", validate="one_to_one",
            suffixes=("_new", "_v4"),
        )
        if len(merged) != len(frame):
            raise SecondaryAblationError(f"v4 scored key mismatch for {strategy_id}.")
        maximum = 0.0
        for column in columns:
            current_values = pd.to_numeric(merged[f"{column}_new"]).to_numpy(float)
            archived_values = pd.to_numeric(merged[f"{column}_v4"]).to_numpy(float)
            if not np.isfinite(current_values).all() or not np.isfinite(archived_values).all():
                raise SecondaryAblationError(
                    f"Economic replay contains non-finite {column} values for {strategy_id}."
                )
            delta = np.max(np.abs(current_values - archived_values))
            maximum = max(maximum, float(delta))
        if maximum > tolerance:
            raise SecondaryAblationError(
                f"Economic replay differs from v4 for {strategy_id}; max error={maximum:.6g}."
            )
        rows.append(
            {
                "evidence_class": EVIDENCE_CLASS,
                "same_sample_interpretation": INTERPRETATION,
                "strategy_id": strategy_id,
                "rows_verified": len(frame),
                "maximum_absolute_error": maximum,
                "tolerance": tolerance,
                "status": "exact_within_tolerance",
            }
        )
    return pd.DataFrame(rows)


def descriptive_metrics(
    scored: dict[str, pd.DataFrame], contract: Contract
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy_id, frame in scored.items():
        for scope in ["locked_all", "complete_periods"]:
            scoped = frame if scope == "locked_all" else frame[frame["is_complete_period"]].copy()
            for window_id, group in scoped.groupby("window_id", sort=False):
                row: dict[str, Any] = {
                    "evidence_class": EVIDENCE_CLASS,
                    "same_sample_interpretation": INTERPRETATION,
                    "confirmatory_test": "not_performed",
                    "strategy_id": strategy_id,
                    "sample_scope": scope,
                    "window_id": window_id,
                }
                row.update(empirical_metrics(group, contract))
                rows.append(row)
    return pd.DataFrame(rows)


def explanatory_differences(
    scored: dict[str, pd.DataFrame],
    full_id: str,
    alternatives: list[str],
    contract: Contract,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    gamma = float(contract["crra_gamma"])
    for alternative in alternatives:
        for scope in ["locked_all", "complete_periods"]:
            full = scored[full_id]
            other = scored[alternative]
            if scope == "complete_periods":
                full = full[full["is_complete_period"]]
                other = other[other["is_complete_period"]]
            merged = full[KEYS + ["net_return"]].merge(
                other[KEYS + ["net_return"]], on=KEYS, validate="one_to_one",
                suffixes=("_full", "_alternative"),
            )
            full_returns = merged["net_return_full"].to_numpy(float)
            alternative_returns = merged["net_return_alternative"].to_numpy(float)
            utility_delta = crra_utility(full_returns, gamma) - crra_utility(
                alternative_returns, gamma
            )
            rows.append(
                {
                    "evidence_class": EVIDENCE_CLASS,
                    "same_sample_interpretation": INTERPRETATION,
                    "confirmatory_test": "not_performed",
                    "reference_strategy_id": full_id,
                    "alternative_strategy_id": alternative,
                    "sample_scope": scope,
                    "observations": len(merged),
                    "mean_net_return_difference_full_minus_alternative": float(
                        np.mean(full_returns - alternative_returns)
                    ),
                    "mean_crra_utility_difference_full_minus_alternative": float(
                        np.mean(utility_delta)
                    ),
                    "full_better_period_fraction": float(np.mean(full_returns > alternative_returns)),
                    "p_value": "not_computed",
                    "confidence_interval": "not_computed",
                    "scientific_claim": "mechanism_and_incremental_contribution_only",
                }
            )
    return pd.DataFrame(rows)


def write_checksums(root: Path) -> None:
    rows = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.name != "CONTENTS.sha256"
    ]
    (root / "CONTENTS.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _remove_bundle_pair(bundle: Path) -> None:
    bundle.unlink(missing_ok=True)
    bundle.with_suffix(bundle.suffix + ".sha256").unlink(missing_ok=True)


def _publish_tree(temporary: Path, output: Path, bundle: Path) -> None:
    deterministic_tar(temporary, bundle, root_name=output.name)
    try:
        os.replace(temporary, output)
    except Exception:
        _remove_bundle_pair(bundle)
        raise


def _preserve_failed_tree(temporary: Path, output: Path, bundle: Path) -> Exception | None:
    archive_error: Exception | None = None
    try:
        deterministic_tar(temporary, bundle, root_name=output.name)
    except Exception as error:
        archive_error = error
        _remove_bundle_pair(bundle)
    os.replace(temporary, output)
    return archive_error


def execute_secondary_ablation(
    *,
    repo_root: Path,
    contract_path: Path,
    successful_archive: Path,
    successful_sidecar: Path,
    evaluation_contract_path: Path,
    runtime_config: Path,
    full_training_release: Path,
    no_vine_training_release: Path,
    output: Path,
    bundle: Path,
    secondary_plan_release: Path | None = None,
    rscript: str = "Rscript",
    runner: Runner = run_logged,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    contract_path = contract_path.resolve()
    successful_archive = successful_archive.resolve()
    successful_sidecar = successful_sidecar.resolve()
    evaluation_contract_path = evaluation_contract_path.resolve()
    runtime_config = runtime_config.resolve()
    full_training_release = full_training_release.resolve()
    no_vine_training_release = no_vine_training_release.resolve()
    if secondary_plan_release is not None:
        secondary_plan_release = secondary_plan_release.resolve()
    output = output.resolve()
    bundle = bundle.resolve()
    sidecar = bundle.with_suffix(bundle.suffix + ".sha256")
    if output.exists() or bundle.exists() or sidecar.exists():
        raise SecondaryAblationError("Output, bundle, or sidecar already exists; refusing overwrite.")
    if EVIDENCE_CLASS not in output.name or EVIDENCE_CLASS not in bundle.name:
        raise SecondaryAblationError(
            f"Output and bundle names must include {EVIDENCE_CLASS} to prevent evidence relabelling."
        )
    if not repo_root.is_dir() or not runtime_config.is_file():
        raise SecondaryAblationError("Repository root or runtime configuration is missing.")
    if secondary_plan_release is None and runner is run_logged:
        raise SecondaryAblationError(
            "A frozen --secondary-plan-release is required for real execution."
        )
    plan_verification = (
        verify_secondary_plan_release(secondary_plan_release, repo_root)
        if secondary_plan_release is not None
        else {
            "release": "unit_test_injected_runner",
            "contents_sha256": "not_applicable",
            "source_count": 0,
            "source_inventory_sha256": "not_applicable",
        }
    )

    contract_definition = validate_contract(load_json(contract_path))
    full_experiment = contract_definition["experiments"]["full_reference"]
    no_vine_experiment = contract_definition["experiments"]["no_vine"]
    pretrained_experiment = contract_definition["experiments"]["pretrained_only"]
    full_release_full = validate_training_release(
        full_training_release,
        list(full_experiment["expected_seeds"]),
        "full", "full", bool(full_experiment.get("allow_legacy_missing_mode", False)),
    )
    full_release_pretrained = validate_training_release(
        full_training_release,
        list(pretrained_experiment["expected_seeds"]),
        "pretrained", "full", bool(pretrained_experiment.get("allow_legacy_missing_mode", False)),
    )
    no_vine_release_full = validate_training_release(
        no_vine_training_release,
        list(no_vine_experiment["expected_seeds"]),
        "full", "zero", False,
    )
    matched_design_report = verify_matched_training_design(
        full_training_release,
        list(full_experiment["expected_seeds"]),
        no_vine_training_release,
        list(no_vine_experiment["expected_seeds"]),
        contract_definition["matched_design"],
    )
    verified = verify_successful_v4(
        successful_archive,
        successful_sidecar,
        contract_definition,
        evaluation_contract_path,
        full_release_full,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}_", dir=output.parent))
    try:
        logs = temporary / "post_holdout_explanatory_command_logs"
        logs.mkdir()
        source_root = temporary / "post_holdout_explanatory_frozen_v4_inputs"
        source_inventory: list[dict[str, Any]] = []
        for relative, value in sorted(verified["copied_artifacts"].items()):
            target = source_root / Path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
            source_inventory.append(
                {
                    "evidence_class": EVIDENCE_CLASS,
                    "same_sample_interpretation": INTERPRETATION,
                    "source": "successful_v4_archive",
                    "path": relative,
                    "sha256": sha256_bytes(value),
                    "bytes_reused_without_modification": True,
                }
            )
        evaluation_contract_copy = source_root / "evaluation_contract.json"
        evaluation_contract_copy.write_bytes(evaluation_contract_path.read_bytes())
        source_inventory.append(
            {
                "evidence_class": EVIDENCE_CLASS,
                "same_sample_interpretation": INTERPRETATION,
                "source": "v4_hash_linked_evaluation_contract",
                "path": "evaluation_contract.json",
                "sha256": sha256_file(evaluation_contract_copy),
                "bytes_reused_without_modification": True,
            }
        )

        generated_root = temporary / "post_holdout_explanatory_generated_weights"
        generated_root.mkdir()
        full_replay_weights, full_replay_inventory, full_replay_timings = generate_weights(
            full_release_full,
            "full_v4_inference_replay",
            "full",
            "full",
            generated_root / "full_v4_inference_replay",
            logs,
            repo_root,
            runtime_config,
            rscript,
            str(contract_definition["successful_v4"]["evaluation_id"]),
            runner,
            verified["realized"],
            verified["assets"],
            verified["evaluation_contract"],
        )
        full_inference_replay = verify_full_inference_replay(
            full_replay_weights,
            verified["full_weights"],
            verified["assets"],
            float(
                contract_definition["evaluation"]["inference_replay_weight_tolerance"]
            ),
        )
        pretrained_weights, pretrained_inventory, pretrained_timings = generate_weights(
            full_release_pretrained,
            "pretrained_only",
            "pretrained",
            "full",
            generated_root / "pretrained_only",
            logs,
            repo_root,
            runtime_config,
            rscript,
            str(contract_definition["successful_v4"]["evaluation_id"]),
            runner,
            verified["realized"],
            verified["assets"],
            verified["evaluation_contract"],
        )
        no_vine_weights, no_vine_inventory, no_vine_timings = generate_weights(
            no_vine_release_full,
            "no_vine",
            "full",
            "zero",
            generated_root / "no_vine",
            logs,
            repo_root,
            runtime_config,
            rscript,
            str(contract_definition["successful_v4"]["evaluation_id"]),
            runner,
            verified["realized"],
            verified["assets"],
            verified["evaluation_contract"],
        )

        ensembles = {
            full_experiment["ensemble_strategy_id"]: mean_ensemble(
                verified["full_weights"],
                full_experiment["ensemble_strategy_id"],
                verified["assets"],
                verified["evaluation_contract"],
            ),
            no_vine_experiment["ensemble_strategy_id"]: mean_ensemble(
                no_vine_weights,
                no_vine_experiment["ensemble_strategy_id"],
                verified["assets"],
                verified["evaluation_contract"],
            ),
            pretrained_experiment["ensemble_strategy_id"]: mean_ensemble(
                pretrained_weights,
                pretrained_experiment["ensemble_strategy_id"],
                verified["assets"],
                verified["evaluation_contract"],
            ),
        }
        ensemble_dir = temporary / "post_holdout_explanatory_ensemble_weights"
        ensemble_inventory: list[dict[str, Any]] = []
        for strategy_id, frame in ensembles.items():
            labelled = frame.copy()
            labelled.insert(0, "same_sample_interpretation", INTERPRETATION)
            labelled.insert(0, "evidence_class", EVIDENCE_CLASS)
            path = ensemble_dir / f"post_holdout_explanatory_weights_{strategy_id}.csv"
            write_frame(path, labelled)
            ensemble_inventory.append(
                {
                    "evidence_class": EVIDENCE_CLASS,
                    "same_sample_interpretation": INTERPRETATION,
                    "strategy_id": strategy_id,
                    "member_count": (
                        len(verified["full_weights"])
                        if strategy_id == full_experiment["ensemble_strategy_id"]
                        else len(no_vine_weights)
                        if strategy_id == no_vine_experiment["ensemble_strategy_id"]
                        else len(pretrained_weights)
                    ),
                    "aggregation_rule": "arithmetic_mean_of_period_asset_weights",
                    "weight_sha256": sha256_file(path),
                }
            )

        benchmark_weights: dict[str, pd.DataFrame] = {}
        for strategy_id, value in verified["benchmarks"].items():
            benchmark_weights[strategy_id] = (
                equal_weight(
                    verified["realized"], verified["assets"], verified["evaluation_contract"]
                )
                if value["weights"] is None
                else value["weights"]
            )
        scored = {
            strategy_id: score_strategy(
                strategy_id,
                weights,
                verified["realized"],
                verified["assets"],
                verified["evaluation_contract"],
            )
            for strategy_id, weights in {**benchmark_weights, **ensembles}.items()
        }
        replay_ids = [
            *contract_definition["evaluation"]["benchmark_strategy_ids"],
            full_experiment["ensemble_strategy_id"],
        ]
        replay = verify_archived_scores(
            {strategy_id: scored[strategy_id] for strategy_id in replay_ids},
            verified["archived_scored"],
            verified["assets"],
            float(contract_definition["evaluation"]["archive_score_tolerance"]),
        )

        scored_panel = pd.concat(scored.values(), ignore_index=True)
        scored_panel.insert(0, "same_sample_interpretation", INTERPRETATION)
        scored_panel.insert(0, "evidence_class", EVIDENCE_CLASS)
        reports = temporary / "post_holdout_explanatory_reports"
        write_frame(reports / "post_holdout_explanatory_scored_panel.csv", scored_panel)
        write_frame(
            reports / "post_holdout_explanatory_descriptive_metrics.csv",
            descriptive_metrics(scored, verified["evaluation_contract"]),
        )
        full_id = full_experiment["ensemble_strategy_id"]
        comparison_ids = [
            no_vine_experiment["ensemble_strategy_id"],
            pretrained_experiment["ensemble_strategy_id"],
            *contract_definition["evaluation"]["benchmark_strategy_ids"],
        ]
        write_frame(
            reports / "post_holdout_explanatory_differences.csv",
            explanatory_differences(
                scored, full_id, comparison_ids, verified["evaluation_contract"]
            ),
        )
        write_frame(reports / "post_holdout_explanatory_v4_replay_verification.csv", replay)
        write_frame(
            reports / "post_holdout_explanatory_full_inference_replay.csv",
            full_inference_replay,
        )
        write_frame(
            reports / "post_holdout_explanatory_matched_design_verification.csv",
            matched_design_report,
        )
        write_frame(
            temporary / "post_holdout_explanatory_source_inventory.csv",
            pd.DataFrame(source_inventory),
        )
        write_frame(
            temporary / "post_holdout_explanatory_checkpoint_weight_inventory.csv",
            pd.DataFrame(full_replay_inventory + pretrained_inventory + no_vine_inventory),
        )
        write_frame(
            temporary / "post_holdout_explanatory_ensemble_inventory.csv",
            pd.DataFrame(ensemble_inventory),
        )

        command_source_inventory = []
        for path in [
            repo_root / "evaluate_with_config.r",
            repo_root / "rl" / "evaluate_rl.r",
            runtime_config,
            contract_path,
            Path(__file__).resolve(),
        ]:
            if not path.is_file():
                raise SecondaryAblationError(f"Execution source is missing: {path}")
            command_source_inventory.append(
                {
                    "evidence_class": EVIDENCE_CLASS,
                    "same_sample_interpretation": INTERPRETATION,
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            )
        write_frame(
            temporary / "post_holdout_explanatory_execution_source_inventory.csv",
            pd.DataFrame(command_source_inventory),
        )
        manifest = {
            "schema_version": 1,
            "release_status": contract_definition["outputs"]["release_status"],
            "evidence_class": EVIDENCE_CLASS,
            "same_sample_interpretation": INTERPRETATION,
            "same_sample_tests": "descriptive_only_no_confirmatory_tests",
            "confirmatory_claims_permitted": False,
            "holdout_status": "previously_consumed_by_successful_v4",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "successful_v4_archive_sha256": verified["archive_sha256"],
            "evaluation_contract_sha256": verified["evaluation_contract_sha256"],
            "runtime_config_sha256": sha256_file(runtime_config),
            "secondary_plan_release": plan_verification,
            "full_training_release": str(full_training_release),
            "no_vine_training_release": str(no_vine_training_release),
            "full_reference_seed_count": len(verified["full_weights"]),
            "pretrained_seed_count": len(pretrained_weights),
            "no_vine_seed_count": len(no_vine_weights),
            "benchmark_count": len(benchmark_weights),
            "ensemble_strategy_ids": list(ensembles),
            "policy_generation_environment": {
                "pretrained_only_EVAL_CHECKPOINT_MODELS": "pretrained",
                "no_vine_EVAL_CHECKPOINT_MODELS": "full",
                "full_v4_replay_EVAL_CHECKPOINT_MODELS": "full",
                "pretrained_only_VINE_OBSERVATION_MODE": "full",
                "no_vine_VINE_OBSERVATION_MODE": "zero",
                "EVAL_WEIGHTS_ONLY": "true",
            },
            "timings_seconds": {
                **full_replay_timings, **pretrained_timings, **no_vine_timings
            },
            "full_inference_replay_verified": bool(
                (full_inference_replay["status"] == "exact_within_tolerance").all()
            ),
            "matched_design_verified": bool(
                (matched_design_report["status"] == "pass").all()
            ),
            "economic_replay_verified": bool((replay["status"] == "exact_within_tolerance").all()),
            "scientific_limit": (
                "All comparisons reuse the consumed holdout and are explanatory only. "
                "Confirmatory claims require non-overlapping future or external data."
            ),
        }
        (temporary / "post_holdout_explanatory_release_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temporary / "READ_ONLY_RELEASE.txt").write_text(
            "POST-HOLDOUT EXPLANATORY RELEASE - READ ONLY\n"
            "The v4 holdout was already consumed. Same-sample outputs are descriptive only, "
            "not confirmatory.\n",
            encoding="utf-8",
        )
        write_checksums(temporary)
        _publish_tree(temporary, output, bundle)
        return manifest
    except Exception as error:
        failure = {
            "schema_version": 1,
            "release_status": "failed_post_holdout_explanatory_batch",
            "evidence_class": EVIDENCE_CLASS,
            "same_sample_interpretation": INTERPRETATION,
            "confirmatory_claims_permitted": False,
            "successful_v4_archive_accessed": True,
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "scientific_limit": (
                "This failed same-sample explanatory run is preserved as incident evidence; "
                "it cannot be relabelled as confirmatory evidence."
            ),
        }
        (temporary / "post_holdout_explanatory_release_manifest.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_checksums(temporary)
        archive_error = _preserve_failed_tree(temporary, output, bundle)
        if archive_error is not None:
            raise SecondaryAblationError(
                f"{error} Failure logs were preserved at {output}, but archive creation "
                f"also failed: {archive_error}"
            ) from error
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--successful-archive", required=True, type=Path)
    parser.add_argument("--successful-sidecar", required=True, type=Path)
    parser.add_argument("--evaluation-contract", required=True, type=Path)
    parser.add_argument("--runtime-config", required=True, type=Path)
    parser.add_argument("--full-training-release", required=True, type=Path)
    parser.add_argument("--no-vine-training-release", required=True, type=Path)
    parser.add_argument("--secondary-plan-release", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--rscript", default="Rscript")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = execute_secondary_ablation(
            repo_root=args.repo_root,
            contract_path=args.contract,
            successful_archive=args.successful_archive,
            successful_sidecar=args.successful_sidecar,
            evaluation_contract_path=args.evaluation_contract,
            runtime_config=args.runtime_config,
            full_training_release=args.full_training_release,
            no_vine_training_release=args.no_vine_training_release,
            secondary_plan_release=args.secondary_plan_release,
            output=args.output,
            bundle=args.bundle,
            rscript=args.rscript,
        )
    except (SecondaryAblationError, EvaluationProtocolError, OSError, ValueError) as error:
        print(f"POST-HOLDOUT EXPLANATORY BATCH FAILED: {error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
