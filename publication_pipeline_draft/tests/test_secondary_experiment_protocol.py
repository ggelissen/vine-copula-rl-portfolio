from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from publication_pipeline_draft.secondary_experiment_protocol import (
    EVIDENCE_CLASS,
    ProtocolError,
    freeze_plan,
    load_json,
    merge_sweep_status,
    validate_checkpoints,
    validate_contract,
    validate_sweep,
    write_checksums,
)
from publication_pipeline_draft.secondary_ablation_batch import (
    SecondaryAblationError,
    verify_secondary_plan_release,
)
from publication_pipeline_draft.preflight_no_vine_training_contract import (
    preflight as preflight_no_vine,
)
from rl.checkpoint_attestation import resolve_architecture_mode, sha256_file


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "publication_pipeline_draft/config/secondary_experiments_v1.json"


def test_live_secondary_contract_is_explicitly_post_holdout() -> None:
    contract = load_json(CONTRACT)
    experiments = validate_contract(contract)
    assert contract["evidence_class"] == EVIDENCE_CLASS
    assert contract["main_result_immutable"] is True
    assert contract["consumed_holdout"]["status"] == "consumed"
    assert contract["consumed_holdout"]["archive_sha256"] == (
        "770d2944f915d0ad21ae9af32e31d68d652fdb54e98939caeab45c327b4e5ea1"
    )
    assert {item["experiment_id"] for item in experiments} >= {
        "no_vine_td3",
        "pretrained_only",
        "no_cvar_signal_and_penalty",
        "feedforward_no_lstm",
    }


def test_four_gpu_launcher_requires_an_explicit_cuda_training_runtime() -> None:
    launcher = (ROOT / "hpc/run_no_vine_4gpu.sh").read_text(encoding="utf-8")
    assert "${TRAIN_PYTHON:?" in launcher
    assert "RETICULATE_PYTHON=\"$TRAIN_PYTHON\"" in launcher
    assert "env -u CONDA_PREFIX" in launcher
    assert "torch.cuda.device_count() < 4" in launcher
    assert "import gymnasium" in launcher
    requirements = (ROOT / "requirements-secondary-training.txt").read_text(
        encoding="utf-8"
    )
    assert "gymnasium==1.3.0" in requirements


def test_worker_shards_fail_closed_and_emit_the_ablation_schema() -> None:
    runner = (ROOT / "rl/run_seed_sweep.r").read_text(encoding="utf-8")
    assert '"vine_observation_mode", "no_vine_signal_mask"' in runner
    assert '"full_zero_vine_median_action_l1"' in runner
    assert "Internal sweep status schema error" in runner
    assert "Worker shard failed closed" in runner
    assert "SWEEP_REUSE_COMPLETED_TRAINING" in runner


def test_clean_no_vine_retraining_contract_preflight() -> None:
    contract = load_json(CONTRACT)
    experiment = next(
        item
        for item in contract["experiments"]
        if item["experiment_id"] == "no_vine_td3"
    )
    expected_root = ROOT / experiment["sweep_root"]
    result = preflight_no_vine(ROOT, expected_root)
    assert result["status"] == "clean_no_vine_retraining_contract_passed"
    assert result["expected_mode"] == "zero"
    assert len(result["expected_seeds"]) == 10
    assert contract["invalidated_predecessor"]["status"] == (
        "invalid_not_a_no_vine_ablation"
    )
    launcher = (ROOT / "hpc/run_no_vine_4gpu.sh").read_text(encoding="utf-8")
    assert "preflight_no_vine_training_contract.py" in launcher
    assert "--require-embedded" in launcher
    assert "no_vine_rl_runs_secondary_v3" in launcher


def test_mode_marker_recovery_is_attested_and_sanity_only() -> None:
    repair = (ROOT / "rl/repair_no_vine_mode_markers.r").read_text(
        encoding="utf-8"
    )
    verifier = (
        ROOT / "publication_pipeline_draft/verify_no_vine_training_evidence.py"
    ).read_text(encoding="utf-8")
    resume = (ROOT / "hpc/resume_no_vine_sanity_4gpu.sh").read_text(
        encoding="utf-8"
    )
    assert "scientific_model_or_checkpoint_changed = FALSE" in repair
    assert "legacy_field_absent_hash_verified_launcher_and_worker_log" in repair
    assert '"helper/reproducibility.r", "rl/run_seed_sweep.r"' in repair
    assert '"run_with_config.r"' in repair
    assert "Sweep vine observation mode: zero" in repair
    assert "Worker-log evidence is missing, duplicated, or inconsistent" in repair
    assert '"VINE_OBSERVATION_MODE" %in% names(environment)' in repair
    assert "launcher_preserves_or_passively_inherits_mode" in repair
    assert "launcher_direct_override" in repair
    assert "failed checks:" in repair
    assert "valid_embedded_no_vine_checkpoint_evidence" in verifier
    assert "valid_checkpoint_files_with_legacy_missing_mode_metadata" in verifier
    assert 'SWEEP_REUSE_COMPLETED_TRAINING=true' in resume
    assert "Training was not rerun" in resume
    attestation = (ROOT / "rl/checkpoint_attestation.py").read_text(
        encoding="utf-8"
    )
    sanity = (ROOT / "rl/training_sanity_check.r").read_text(encoding="utf-8")
    server = (ROOT / "rl/policy_inference_server.py").read_text(encoding="utf-8")
    assert "Checkpoint hash differs from the recovery attestation" in attestation
    assert "resolve_architecture_mode" in sanity
    assert "resolve_architecture_mode" in server


def test_legacy_zero_mode_requires_a_hash_bound_attestation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "td3_lstm_vine_full.pt"
    checkpoint.write_bytes(b"immutable learned checkpoint bytes")
    with pytest.raises(RuntimeError, match="lacks embedded metadata"):
        resolve_architecture_mode(checkpoint, {"checkpoint_schema": 5}, "zero")
    repair = {
        "repair_type": "post_hoc_missing_plaintext_mode_marker_reconstruction",
        "scientific_model_or_checkpoint_changed": False,
        "reconstructed_value": "zero",
        "checkpoint_evidence": [
            {
                "checkpoint": checkpoint.name,
                "sha256": sha256_file(checkpoint),
                "mode_metadata_status": (
                    "legacy_missing_requires_manifest_source_attestation"
                ),
            }
        ],
    }
    (tmp_path / "vine_observation_mode_repair.json").write_text(
        json.dumps(repair), encoding="utf-8"
    )
    architecture, source = resolve_architecture_mode(
        checkpoint, {"checkpoint_schema": 5}, "zero"
    )
    assert architecture["vine_observation_mode"] == "zero"
    assert architecture["no_vine_signal_mask"] == (
        "explicit_vine_and_scenario_cvar_v1"
    )
    assert source == "attested_legacy_zero_mode"
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="hash differs"):
        resolve_architecture_mode(checkpoint, {"checkpoint_schema": 5}, "zero")


def test_contract_rejects_confirmatory_relabelling() -> None:
    contract = load_json(CONTRACT)
    contract["evidence_class"] = "confirmatory"
    with pytest.raises(ProtocolError, match="post_holdout_explanatory"):
        validate_contract(contract)


def _write_status(path: Path, mutate: dict[str, str] | None = None) -> None:
    contract = load_json(CONTRACT)
    experiment = next(
        item for item in contract["experiments"] if item["experiment_id"] == "no_vine_td3"
    )
    fields = [
        "seed",
        "training_status",
        "sanity_status",
        "no_holdout_gate_pass",
        "vine_observation_mode",
        "no_vine_signal_mask",
        "full_zero_vine_median_action_l1",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for seed in experiment["expected_seeds"]:
            row = {
                "seed": str(seed),
                "training_status": "0",
                "sanity_status": "0",
                "no_holdout_gate_pass": "true",
                "vine_observation_mode": "zero",
                "no_vine_signal_mask": "explicit_vine_and_scenario_cvar_v1",
                "full_zero_vine_median_action_l1": "0",
            }
            if mutate and seed == experiment["expected_seeds"][0]:
                row.update(mutate)
            writer.writerow(row)


def test_validate_sweep_accepts_only_complete_negative_control(tmp_path: Path) -> None:
    status = tmp_path / "status.csv"
    _write_status(status)
    result = validate_sweep(CONTRACT, "no_vine_td3", status)
    assert result["status"] == "valid"
    assert result["seed_count"] == 10


def test_validate_sweep_rejects_failed_invariance(tmp_path: Path) -> None:
    status = tmp_path / "status.csv"
    _write_status(status, {"full_zero_vine_median_action_l1": "0.1"})
    with pytest.raises(ProtocolError, match="zero-channel invariance"):
        validate_sweep(CONTRACT, "no_vine_td3", status)


def test_merge_worker_status_is_disjoint_complete_and_sorted(tmp_path: Path) -> None:
    all_status = tmp_path / "all.csv"
    _write_status(all_status)
    rows = list(csv.DictReader(all_status.open(newline="", encoding="utf-8")))
    fields = list(rows[0])
    workers = []
    for worker_index in range(4):
        path = tmp_path / f"worker_{worker_index + 1}.csv"
        workers.append(path)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows[worker_index::4])
    output = tmp_path / "merged.csv"
    result = merge_sweep_status(CONTRACT, "no_vine_td3", workers, output)
    assert result["status"] == "valid"
    assert result["worker_status_count"] == 4
    merged = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    assert [int(row["seed"]) for row in merged] == sorted(int(row["seed"]) for row in rows)


def test_free_plan_is_fail_closed_and_checksummed(tmp_path: Path) -> None:
    output = tmp_path / "secondary_release"
    bundle = tmp_path / "secondary_release.tar.gz"
    manifest = freeze_plan(ROOT, CONTRACT, output, bundle)
    assert manifest["release_status"] == "frozen_post_holdout_secondary_plan"
    assert (output / "CONTENTS.sha256").is_file()
    assert (output / "LIVE_SOURCE_CONTENTS.sha256").is_file()
    assert (output / "source_snapshot/hpc/run_no_vine_4gpu.sh").is_file()
    assert (output / "EXECUTE_SECONDARY_EXPERIMENTS.sh").is_file()
    execution_script = (output / "EXECUTE_SECONDARY_EXPERIMENTS.sh").read_text(
        encoding="utf-8"
    )
    assert "${TRAIN_PYTHON:?" in execution_script
    assert 'sha256sum -c "$PLAN_ROOT/LIVE_SOURCE_CONTENTS.sha256"' in execution_script
    assert verify_secondary_plan_release(output, ROOT)["source_count"] > 0
    assert bundle.is_file()
    assert bundle.with_suffix(bundle.suffix + ".sha256").is_file()
    with pytest.raises(ProtocolError, match="will not be overwritten"):
        freeze_plan(ROOT, CONTRACT, output, None)

    frozen_source = output / "source_snapshot/hpc/run_no_vine_4gpu.sh"
    frozen_source.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(SecondaryAblationError, match="checksum mismatch"):
        verify_secondary_plan_release(output, ROOT)


def test_checkpoint_validation_rejects_extra_seed_and_accessed_release(
    tmp_path: Path,
) -> None:
    contract = load_json(CONTRACT)
    experiment = next(
        item
        for item in contract["experiments"]
        if item["experiment_id"] == "pretrained_only"
    )
    release = tmp_path / "training_release"
    for seed in experiment["expected_seeds"]:
        directory = release / "seeds" / f"seed_{seed}"
        directory.mkdir(parents=True)
        (directory / "td3_lstm_vine_pretrained.pt").write_bytes(f"p-{seed}".encode())
        (directory / "td3_lstm_vine_full.pt").write_bytes(f"f-{seed}".encode())
    manifest = release / "training_release_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "release_status": "frozen_pre_oos",
                "holdout_accessed_by_freezer": False,
            }
        ),
        encoding="utf-8",
    )
    write_checksums(release)
    assert validate_checkpoints(CONTRACT, "pretrained_only", release)["status"] == "valid"

    extra = release / "seeds" / "seed_99999999"
    extra.mkdir()
    (extra / "td3_lstm_vine_pretrained.pt").write_bytes(b"p-extra")
    (extra / "td3_lstm_vine_full.pt").write_bytes(b"f-extra")
    write_checksums(release)
    with pytest.raises(ProtocolError, match="do not exactly match"):
        validate_checkpoints(CONTRACT, "pretrained_only", release)

    for child in extra.iterdir():
        child.unlink()
    extra.rmdir()
    manifest.write_text(
        json.dumps(
            {
                "release_status": "frozen_pre_oos",
                "holdout_accessed_by_freezer": True,
            }
        ),
        encoding="utf-8",
    )
    write_checksums(release)
    with pytest.raises(ProtocolError, match="holdout-blind"):
        validate_checkpoints(CONTRACT, "pretrained_only", release)
