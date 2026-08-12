#!/usr/bin/env python3
"""Freeze a gate-passing multi-seed training release before OOS access.

The command is deliberately holdout-blind.  It verifies the aggregate
diagnostic archive, the exact training source/data hashes, and the per-seed
checkpoint artifacts, then creates a new immutable-by-convention release
directory plus an optional tar.gz bundle.  It never reads realized holdout
returns and refuses to overwrite an existing output.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Iterable

import pandas as pd

try:  # package import in tests; script import in production CLI
    from .publication_pipeline import ProtocolError, parse_bool, sha256_file
except ImportError:  # pragma: no cover - exercised by direct CLI invocation
    from publication_pipeline import ProtocolError, parse_bool, sha256_file


ARCHIVE_TABLES = {
    "status": "tables/table_t01_seed_sweep_status.csv",
    "gates": "tables/table_t02_pretraining_gates.csv",
    "policies": "tables/table_t03_policy_sanity.csv",
    "sensitivity": "tables/table_t04_state_sensitivity.csv",
    "finetune_validation": "tables/table_t05_finetune_validation.csv",
    "checkpoints": "tables/table_t07_checkpoint_integrity.csv",
    "sanity": "tables/table_t08_sanity_reports.csv",
    "hash_consensus": "tables/table_t09_code_data_hash_consensus.csv",
    "artifact_inventory": "tables/table_t10_artifact_inventory.csv",
    "episodes": "raw/training_episode_metrics_all_seeds.csv",
    "updates": "raw/training_update_metrics_all_seeds.csv",
    "finetune_schedule": "raw/finetune_episode_schedule_all_seeds.csv",
}

SEED_ARTIFACTS = (
    "training_episode_metrics.csv",
    "training_update_metrics.csv",
    "pretraining_behavior_gate.csv",
    "pretraining_policy_diagnostics.csv",
    "pretraining_behavior_warnings.csv",
    "finetune_validation_metrics.csv",
    "finetune_episode_schedule.csv",
    "finetune_selection.txt",
    "data_hashes.csv",
    "code_hashes.csv",
    "run_manifest.rds",
    "td3_lstm_vine_pretrained.pt",
    "td3_lstm_vine_full.pt",
    "sanity_no_holdout/checkpoint_integrity.csv",
    "sanity_no_holdout/policy_summary.csv",
    "sanity_no_holdout/policy_steps.csv",
    "sanity_no_holdout/state_sensitivity_summary.csv",
    "sanity_no_holdout/state_sensitivity_steps.csv",
    "sanity_no_holdout/sanity_report.json",
)
OPTIONAL_SEED_ARTIFACTS = ("vine_observation_mode.txt",)


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # nosec B324 - compatibility with recorded R hashes
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_archive_members(archive: Path) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            name = member.name.replace("\\", "/")
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts:
                raise ProtocolError(f"Unsafe path in diagnostic archive: {member.name}")
            if member.issym() or member.islnk():
                raise ProtocolError(f"Links are not allowed in diagnostic archive: {member.name}")
            if member.isfile():
                members[name] = member
    return members


def archive_bytes(archive: Path, suffix: str) -> bytes:
    members = safe_archive_members(archive)
    matches = [name for name in members if name == suffix or name.endswith("/" + suffix)]
    if len(matches) != 1:
        raise ProtocolError(
            f"Expected exactly one {suffix} in {archive}; found {len(matches)}."
        )
    with tarfile.open(archive, "r:*") as handle:
        stream = handle.extractfile(members[matches[0]])
        if stream is None:
            raise ProtocolError(f"Could not read {matches[0]} from {archive}.")
        return stream.read()


def archive_csv(archive: Path, suffix: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(archive_bytes(archive, suffix)))


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ProtocolError(f"{label} is missing columns: {', '.join(missing)}")


def validate_diagnostic_archive(
    archive: Path, expected_seeds: int
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    if not archive.is_file():
        raise ProtocolError(
            f"Training diagnostic archive not found: {archive}. "
            f"Current working directory: {Path.cwd()}. "
            "Use an absolute --diagnostics-archive path and run `test -s PATH` first."
        )
    tables = {name: archive_csv(archive, suffix) for name, suffix in ARCHIVE_TABLES.items()}

    status = tables["status"]
    require_columns(
        status,
        ["seed", "training_status", "sanity_status", "no_holdout_gate_pass"],
        "seed sweep status",
    )
    if len(status) != expected_seeds or status["seed"].nunique() != expected_seeds:
        raise ProtocolError(
            f"Expected {expected_seeds} unique completed seeds; found {len(status)} rows "
            f"and {status['seed'].nunique()} unique seeds."
        )
    gate = pd.Series(
        [parse_bool(value, "no_holdout_gate_pass") for value in status["no_holdout_gate_pass"]]
    )
    if not ((status["training_status"] == 0) & (status["sanity_status"] == 0) & gate).all():
        raise ProtocolError("At least one seed failed training, sanity, or the no-holdout gate.")

    gates = tables["gates"]
    require_columns(gates, ["seed", "metric", "pass"], "pre-training gates")
    if gates["seed"].nunique() != expected_seeds or not all(
        parse_bool(value, "pretraining gate pass") for value in gates["pass"]
    ):
        raise ProtocolError("The aggregate pre-training gate table is incomplete or contains a failure.")

    policies = tables["policies"]
    require_columns(
        policies,
        ["seed", "model", "all_values_finite", "hard_constraints_pass"],
        "policy sanity",
    )
    trained = policies[policies["model"].isin(["pretrained", "full"])]
    expected_models = {(int(seed), model) for seed in status["seed"] for model in ["pretrained", "full"]}
    observed_models = set(zip(trained["seed"].astype(int), trained["model"].astype(str)))
    if observed_models != expected_models:
        raise ProtocolError("Policy sanity does not contain exactly pretrained/full for every seed.")
    if not all(parse_bool(value, "all_values_finite") for value in trained["all_values_finite"]):
        raise ProtocolError("A trained policy contains non-finite sanity-check values.")
    if not all(parse_bool(value, "hard_constraints_pass") for value in trained["hard_constraints_pass"]):
        raise ProtocolError("A trained policy failed the hard portfolio constraints.")

    sanity = tables["sanity"]
    require_columns(
        sanity,
        ["seed", "warning_count", "publication_behavior_pass", "overall_pass"],
        "sanity reports",
    )
    if sanity["seed"].nunique() != expected_seeds or (sanity["warning_count"] != 0).any():
        raise ProtocolError("Sanity reports are incomplete or contain publication warnings.")
    for field in ["publication_behavior_pass", "overall_pass"]:
        if not all(parse_bool(value, field) for value in sanity[field]):
            raise ProtocolError(f"At least one sanity report has {field}=false.")

    checkpoints = tables["checkpoints"]
    require_columns(
        checkpoints,
        ["seed", "model", "sha256", "architecture_match", "all_checkpoint_tensors_finite"],
        "checkpoint integrity",
    )
    observed_checkpoints = set(
        zip(checkpoints["seed"].astype(int), checkpoints["model"].astype(str))
    )
    if observed_checkpoints != expected_models:
        raise ProtocolError("Checkpoint table does not contain exactly two checkpoints per seed.")
    for field in ["architecture_match", "all_checkpoint_tensors_finite"]:
        if not all(parse_bool(value, field) for value in checkpoints[field]):
            raise ProtocolError(f"At least one checkpoint has {field}=false.")

    consensus = tables["hash_consensus"]
    require_columns(
        consensus,
        ["artifact_kind", "normalized_path", "seed_count", "distinct_hashes", "md5"],
        "code/data hash consensus",
    )
    if (consensus["seed_count"] != expected_seeds).any() or (consensus["distinct_hashes"] != 1).any():
        raise ProtocolError("Code/data hashes are not identical across every seed.")
    if not set(consensus["artifact_kind"]).issubset({"code", "data"}):
        raise ProtocolError("Unexpected artifact kind in code/data hash consensus.")

    episodes = tables["episodes"]
    require_columns(
        episodes,
        ["seed", "stage", "episode", "reward", "terminal_wealth", "mean_turnover",
         "mean_cvar", "mean_gross_exposure"],
        "training episode metrics",
    )
    updates = tables["updates"]
    require_columns(
        updates,
        ["seed", "stage", "update", "critic_loss", "actor_loss", "twin_q_gap",
         "critic_grad_norm", "critic2_grad_norm", "actor_grad_norm"],
        "training update metrics",
    )
    for label, frame, required_numeric in [
        (
            "training episode metrics", episodes,
            ["episode", "reward", "terminal_wealth", "mean_turnover", "mean_cvar",
             "mean_gross_exposure"],
        ),
        (
            "training update metrics", updates,
            ["update", "critic_loss", "actor_loss", "twin_q_gap", "critic_grad_norm",
             "critic2_grad_norm", "actor_grad_norm"],
        ),
    ]:
        numeric = frame[required_numeric].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any() or not (numeric.abs() < float("inf")).all().all():
            raise ProtocolError(f"{label} contains missing or non-finite required values.")
        counts = frame.groupby(["seed", "stage"]).size().unstack(fill_value=0)
        if counts.shape[0] != expected_seeds or any(counts[column].nunique() != 1 for column in counts):
            raise ProtocolError(f"{label} has inconsistent stage counts across seeds.")
        required_stages = {"pretrain", "finetune_selection", "finetune_refit_all"}
        if set(counts.columns) != required_stages or (counts <= 0).any().any():
            raise ProtocolError(
                f"{label} must contain positive, identical counts for exactly {sorted(required_stages)}."
            )

    schedule = tables["finetune_schedule"]
    require_columns(
        schedule, ["seed", "stage", "position", "original_episode"],
        "fine-tune episode schedule",
    )
    if schedule["seed"].nunique() != expected_seeds:
        raise ProtocolError("Fine-tune schedule is incomplete across seeds.")
    if schedule.duplicated(["seed", "stage", "position"]).any() or schedule.duplicated(
        ["seed", "stage", "original_episode"]
    ).any():
        raise ProtocolError("Fine-tune schedule contains duplicate positions or episodes.")
    schedule_counts = schedule.groupby(["seed", "stage"]).size().unstack(fill_value=0)
    if any(schedule_counts[column].nunique() != 1 for column in schedule_counts):
        raise ProtocolError("Fine-tune schedule stage counts differ across seeds.")
    required_schedule_stages = {"selection_fit", "all_history_refit"}
    if set(schedule_counts.columns) != required_schedule_stages or (schedule_counts <= 0).any().any():
        raise ProtocolError(
            "Fine-tune schedule must contain positive selection_fit and all_history_refit rows."
        )

    metrics: dict[str, object] = {
        "decision": "accepted_for_locked_evaluation",
        "expected_seeds": expected_seeds,
        "seed_min": int(status["seed"].min()),
        "seed_max": int(status["seed"].max()),
        "pretraining_gate_rows": int(len(gates)),
        "trained_policy_rows": int(len(trained)),
        "checkpoint_rows": int(len(checkpoints)),
        "code_data_consensus_rows": int(len(consensus)),
        "episode_rows": int(len(episodes)),
        "update_rows": int(len(updates)),
        "finetune_schedule_rows": int(len(schedule)),
    }
    for field in [
        "full_mean_reward",
        "full_mean_terminal_wealth",
        "finetune_reward_delta",
        "finetune_terminal_wealth_delta",
        "full_median_turnover",
        "full_mean_leverage_gate",
        "full_mean_normalized_entropy",
        "full_mean_effective_positions",
    ]:
        if field in status:
            values = pd.to_numeric(status[field], errors="raise")
            metrics[field + "_mean_across_seeds"] = float(values.mean())
            metrics[field + "_min_across_seeds"] = float(values.min())
            metrics[field + "_max_across_seeds"] = float(values.max())
    return tables, metrics


def resolve_seed_directory(base: Path, repo_root: Path, recorded: str, seed: int) -> Path:
    recorded_path = Path(recorded)
    candidates = []
    if recorded_path.is_absolute():
        candidates.append(recorded_path)
    candidates.extend([repo_root / recorded_path, base / recorded_path.name])
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    matches = [
        path for path in base.rglob(f"*{seed}*")
        if path.is_dir() and (path / "td3_lstm_vine_full.pt").is_file()
    ]
    if len(matches) == 1:
        return matches[0].resolve()
    raise ProtocolError(
        f"Could not uniquely locate seed {seed} under {base}; candidates found: {matches}"
    )


def verify_and_copy_seed_artifacts(
    tables: dict[str, pd.DataFrame], rl_runs: Path, repo_root: Path, destination: Path
) -> list[dict[str, object]]:
    status = tables["status"]
    checkpoints = tables["checkpoints"]
    inventory = tables["artifact_inventory"]
    records: list[dict[str, object]] = []
    for _, row in status.sort_values("seed").iterrows():
        seed = int(row["seed"])
        run = resolve_seed_directory(rl_runs, repo_root, str(row["output_dir"]), seed)
        seed_destination = destination / f"seed_{seed}"
        declared_inventory = inventory[inventory["seed"].astype(int) == seed]
        declared_by_artifact = {
            str(item["artifact"]).replace("\\", "/"): str(item["sha256"]).lower()
            for _, item in declared_inventory.iterrows()
        }
        checkpoint_by_model = {
            str(item["model"]): str(item["sha256"]).lower()
            for _, item in checkpoints[checkpoints["seed"].astype(int) == seed].iterrows()
        }
        for relative in SEED_ARTIFACTS:
            source = run / Path(relative)
            if not source.is_file():
                raise ProtocolError(f"Missing seed {seed} release artifact: {source}")
            actual_sha = sha256_file(source)
            normalized = relative.replace("\\", "/")
            declared = declared_by_artifact.get(normalized)
            if declared is not None and actual_sha.lower() != declared:
                raise ProtocolError(f"Aggregate inventory hash mismatch for seed {seed}: {relative}")
            if relative == "td3_lstm_vine_pretrained.pt":
                model = "pretrained"
            elif relative == "td3_lstm_vine_full.pt":
                model = "full"
            else:
                model = None
            if model is not None and actual_sha.lower() != checkpoint_by_model[model]:
                raise ProtocolError(f"Checkpoint hash mismatch for seed {seed}, model {model}.")
            target = seed_destination / Path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            records.append(
                {
                    "seed": seed,
                    "artifact": normalized,
                    "source_path": str(source),
                    "release_path": target.relative_to(destination.parent).as_posix(),
                    "size_bytes": source.stat().st_size,
                    "sha256": actual_sha,
                }
            )
        for relative in OPTIONAL_SEED_ARTIFACTS:
            source = run / Path(relative)
            if not source.is_file():
                continue
            actual_sha = sha256_file(source)
            target = seed_destination / Path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            records.append(
                {
                    "seed": seed,
                    "artifact": relative,
                    "source_path": str(source),
                    "release_path": target.relative_to(destination.parent).as_posix(),
                    "size_bytes": source.stat().st_size,
                    "sha256": actual_sha,
                }
            )
    return records


def verify_training_snapshot(
    consensus: pd.DataFrame,
    repo_root: Path,
    destination: Path,
    verify_data: bool,
    copy_data: bool,
    training_run_dirs: list[Path] | None = None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for _, row in consensus.sort_values(["artifact_kind", "normalized_path"]).iterrows():
        kind = str(row["artifact_kind"])
        relative = str(row["normalized_path"]).replace("\\", "/")
        expected_md5 = str(row["md5"]).lower()
        source = repo_root / Path(relative)
        source_origin = "live_repo"
        snapshot_copy_count = 0
        if kind == "code" and training_run_dirs:
            snapshot_candidates = [
                run / "source_snapshot" / Path(relative)
                for run in training_run_dirs
            ]
            existing_snapshots = [path for path in snapshot_candidates if path.is_file()]
            if existing_snapshots and len(existing_snapshots) != len(snapshot_candidates):
                raise ProtocolError(
                    f"Training source snapshot is incomplete for {relative}: "
                    f"found {len(existing_snapshots)}/{len(snapshot_candidates)} seed copies."
                )
            if existing_snapshots:
                mismatched = [
                    path for path in existing_snapshots
                    if md5_file(path).lower() != expected_md5
                ]
                if mismatched:
                    raise ProtocolError(
                        f"Per-seed training source snapshot hash mismatch for {relative}: "
                        + ", ".join(str(path) for path in mismatched)
                    )
                source = existing_snapshots[0]
                source_origin = "per_seed_training_snapshot_consensus"
                snapshot_copy_count = len(existing_snapshots)
        should_verify = kind == "code" or verify_data
        if should_verify:
            if not source.is_file():
                raise ProtocolError(f"Recorded {kind} artifact is missing from source root: {source}")
            actual_md5 = md5_file(source)
            if actual_md5.lower() != expected_md5:
                raise ProtocolError(
                    f"Recorded {kind} hash mismatch for {relative}: "
                    f"expected {expected_md5}, found {actual_md5}."
                )
        else:
            actual_md5 = "not_verified"

        copied = kind == "code" or (kind == "data" and copy_data)
        release_path = ""
        sha256 = ""
        size_bytes = source.stat().st_size if source.is_file() else 0
        if copied:
            if not source.is_file():
                raise ProtocolError(f"Cannot copy missing {kind} artifact: {source}")
            target_root = destination / ("source_snapshot" if kind == "code" else "training_data")
            target = target_root / Path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            release_path = target.relative_to(destination).as_posix()
            sha256 = sha256_file(target)
        records.append(
            {
                "artifact_kind": kind,
                "normalized_path": relative,
                "expected_md5": expected_md5,
                "verified_md5": actual_md5,
                "verified": bool(should_verify),
                "copied_into_release": copied,
                "release_path": release_path,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "source_origin": source_origin,
                "training_snapshot_copy_count": snapshot_copy_count,
            }
        )
    return records


def write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name == "CONTENTS.sha256":
            continue
        rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "CONTENTS.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def deterministic_tar(
    source: Path,
    bundle: Path,
    *,
    root_name: str | None = None,
) -> None:
    """Write a byte-reproducible gzip tar and checksum without partial outputs.

    ``tarfile.open(..., "w:gz")`` embeds the current time (and can embed the
    output filename) in the gzip header.  Building the gzip layer explicitly
    fixes both fields.  The archive is first written beside the destination
    and atomically renamed only after its checksum sidecar also exists.
    """
    sidecar = bundle.with_suffix(bundle.suffix + ".sha256")
    if bundle.exists() or sidecar.exists():
        raise ProtocolError(
            f"Bundle or checksum already exists and will not be overwritten: {bundle}"
        )
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


def freeze_training_release(
    repo_root: Path,
    rl_runs: Path,
    diagnostics_archive: Path,
    output: Path,
    expected_seeds: int = 20,
    verify_data: bool = True,
    copy_data: bool = False,
    bundle: Path | None = None,
    evidence_class: str = "pre_oos",
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    rl_runs = rl_runs.resolve()
    diagnostics_archive = diagnostics_archive.resolve()
    output = output.resolve()
    if output.exists():
        raise ProtocolError(f"Release output already exists and will not be overwritten: {output}")
    if not repo_root.is_dir() or not rl_runs.is_dir():
        raise ProtocolError("Both --repo-root and --rl-runs must be existing directories.")
    if evidence_class not in {"pre_oos", "post_holdout_explanatory"}:
        raise ProtocolError(f"Unsupported training-release evidence class: {evidence_class}")

    tables, acceptance = validate_diagnostic_archive(diagnostics_archive, expected_seeds)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}_", dir=str(output.parent)))
    try:
        audit_dir = temporary / "audit_tables"
        audit_dir.mkdir(parents=True)
        for name, suffix in ARCHIVE_TABLES.items():
            (audit_dir / Path(suffix).name).write_bytes(archive_bytes(diagnostics_archive, suffix))
        shutil.copy2(diagnostics_archive, temporary / diagnostics_archive.name)

        training_run_dirs = [
            resolve_seed_directory(
                rl_runs, repo_root, str(row["output_dir"]), int(row["seed"])
            )
            for _, row in tables["status"].sort_values("seed").iterrows()
        ]
        snapshot_inventory = verify_training_snapshot(
            tables["hash_consensus"], repo_root, temporary, verify_data,
            copy_data, training_run_dirs=training_run_dirs
        )
        seed_inventory = verify_and_copy_seed_artifacts(
            tables, rl_runs, repo_root, temporary / "seeds"
        )
        pd.DataFrame(snapshot_inventory).to_csv(
            temporary / "training_snapshot_inventory.csv", index=False
        )
        pd.DataFrame(seed_inventory).to_csv(
            temporary / "seed_artifact_inventory.csv", index=False
        )
        post_holdout = evidence_class == "post_holdout_explanatory"
        release_manifest = {
            "schema_version": 1,
            "release_status": (
                "frozen_post_holdout_explanatory_training"
                if post_holdout else "frozen_pre_oos"
            ),
            "evidence_class": evidence_class,
            "confirmatory_claims_permitted": not post_holdout,
            "holdout_accessed_by_freezer": False,
            "diagnostics_archive": diagnostics_archive.name,
            "diagnostics_archive_sha256": sha256_file(diagnostics_archive),
            "repo_root_used_for_verification": str(repo_root),
            "rl_runs_used_for_verification": str(rl_runs),
            "training_data_hashes_verified": verify_data,
            "training_data_copied": copy_data,
            "seed_artifact_count": len(seed_inventory),
            "snapshot_artifact_count": len(snapshot_inventory),
            "acceptance": acceptance,
            "next_gate": (
                "execute only the frozen same-sample explanatory ablation batch"
                if post_holdout else
                "freeze evaluation contract and benchmark implementations before one locked OOS batch"
            ),
        }
        (temporary / "training_release_manifest.json").write_text(
            json.dumps(release_manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        release_label = (
            "frozen post-holdout explanatory training release"
            if post_holdout else "frozen pre-OOS training release"
        )
        scientific_limit = (
            "This release may support explanatory ablations only; confirmatory claims are forbidden.\n"
            if post_holdout else ""
        )
        (temporary / "READ_ONLY_RELEASE.txt").write_text(
            f"This directory is the {release_label}. Do not edit files in place.\n"
            f"{scientific_limit}"
            "Any correction must create a new version with new hashes and an explicit rationale.\n",
            encoding="utf-8",
        )
        write_checksums(temporary)
        os.replace(temporary, output)
        if bundle is not None:
            deterministic_tar(output, bundle.resolve())
        return release_manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--rl-runs", required=True, type=Path)
    parser.add_argument("--diagnostics-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-seeds", type=int, default=20)
    parser.add_argument(
        "--skip-training-data-verification", action="store_true",
        help="Do not re-hash the large recorded training-data files (not recommended for the final release).",
    )
    parser.add_argument(
        "--copy-training-data", action="store_true",
        help="Copy large training-data files into the release in addition to verifying their hashes.",
    )
    parser.add_argument(
        "--bundle", type=Path,
        help="Optional new .tar.gz path for a portable release bundle and sidecar SHA-256 file.",
    )
    parser.add_argument(
        "--evidence-class",
        choices=("pre_oos", "post_holdout_explanatory"),
        default="pre_oos",
        help="Chronological/scientific classification of the frozen training run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = freeze_training_release(
        repo_root=args.repo_root,
        rl_runs=args.rl_runs,
        diagnostics_archive=args.diagnostics_archive,
        output=args.output,
        expected_seeds=args.expected_seeds,
        verify_data=not args.skip_training_data_verification,
        copy_data=args.copy_training_data,
        bundle=args.bundle,
        evidence_class=args.evidence_class,
    )
    print(json.dumps(manifest["acceptance"], indent=2, sort_keys=True))
    print(f"Frozen training release written to {args.output.resolve()}")
    if args.bundle is not None:
        print(f"Portable bundle written to {args.bundle.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
