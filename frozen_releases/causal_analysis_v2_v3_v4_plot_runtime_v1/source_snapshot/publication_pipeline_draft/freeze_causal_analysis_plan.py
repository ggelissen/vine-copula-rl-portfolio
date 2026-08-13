#!/usr/bin/env python3
"""Freeze the outcome-blind causal evaluation and analysis plan.

The freezer binds the prospective analysis contract to the already frozen
publication-extension training release.  It never reads checkpoints, policy
weights, realised returns, or causal outcomes.
"""

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

from publication_pipeline_draft.causal_analysis_contract import (
    CausalAnalysisContractError,
    load_contract,
    materialize_plan,
)
from publication_pipeline_draft.extension_release import (
    ExtensionReleaseError,
    verify_frozen_extension_integrity,
)
class CausalAnalysisFreezeError(CausalAnalysisContractError):
    pass


SOURCES = (
    # Complete policy-replay runtime closure.  The first analysis-plan release
    # omitted these transitive R/Python dependencies, which allowed a legacy
    # per-run sanity requirement to surface only at replay time.
    "evaluate_with_config.r",
    "config/config.yaml",
    "rl/evaluate_rl.r",
    "rl/rl_environment.r",
    "rl/policy_inference_server.py",
    "rl/policy_inference_server_v2.py",
    "rl/recurrent_baselines.py",
    "rl/action_projection.py",
    "helper/load_data.r",
    "helper/time_split.r",
    "helper/marginals.r",
    "benchmark_models/dynamic_vine_NN.r",
    "publication_pipeline_draft/config/causal_analysis_contract_v1.json",
    "publication_pipeline_draft/config/causal_analysis_contract_v2.json",
    "publication_pipeline_draft/config/evaluation_contract.json",
    "publication_pipeline_draft/extension_release.py",
    "publication_pipeline_draft/causal_analysis_contract.py",
    "publication_pipeline_draft/audit_causal_sweep.py",
    "publication_pipeline_draft/merge_causal_operational_retry.py",
    "publication_pipeline_draft/merge_causal_three_revision_retry.py",
    "publication_pipeline_draft/generate_causal_policy_weights.py",
    "publication_pipeline_draft/assemble_causal_policy_ensembles.py",
    "publication_pipeline_draft/materialize_causal_evaluation.py",
    "publication_pipeline_draft/causal_common_accounting.py",
    "publication_pipeline_draft/export_causal_period_panel.py",
    "publication_pipeline_draft/analyze_causal_results.py",
    "publication_pipeline_draft/freeze_causal_analysis_plan.py",
    "publication_pipeline_draft/freeze_causal_results.py",
    "hpc/finalize_causal_evaluation_v4.sh",
    "publication_pipeline_draft/CAUSAL_ANALYSIS_RUNBOOK.md",
    "publication_pipeline_draft/PUBLICATION_EXTENSION_60_JOB_RETRY.md",
    "publication_pipeline_draft/PUBLICATION_EXTENSION_V4_GATE_RECOVERY.md",
    "publication_pipeline_draft/V4_RETRY29_EVIDENCE.md",
    "publication_pipeline_draft/publication_pipeline.py",
    "publication_pipeline_draft/tests/test_causal_analysis_framework.py",
    "publication_pipeline_draft/tests/test_causal_sweep_gate_policy.py",
    "publication_pipeline_draft/tests/test_merge_causal_operational_retry.py",
    "publication_pipeline_draft/tests/test_three_revision_retry_protocol.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_contents(root: Path) -> None:
    checksum = root / "CONTENTS.sha256"
    if not checksum.is_file():
        raise CausalAnalysisFreezeError(f"Release checksum is missing: {root}")
    for line in checksum.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as error:
            raise CausalAnalysisFreezeError(
                f"Malformed checksum line: {line}"
            ) from error
        target = root / relative.removeprefix("./")
        if not target.is_file() or sha256(target) != expected:
            raise CausalAnalysisFreezeError(f"Release checksum mismatch: {target}")


def verify_causal_analysis_release(release: Path, repo_root: Path) -> dict[str, Any]:
    """Verify internal integrity and exact agreement with live analysis sources."""
    release, repo_root = release.resolve(), repo_root.resolve()
    verify_contents(release)
    manifest_path = release / "causal_analysis_release_manifest.json"
    inventory_path = release / "source_inventory.csv"
    if not manifest_path.is_file() or not inventory_path.is_file():
        raise CausalAnalysisFreezeError("Causal analysis release is incomplete.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("release_status") not in {
            "frozen_pre_causal_evaluation_analysis_plan",
            "frozen_pre_causal_outcome_scoring_analysis_plan"}:
        raise CausalAnalysisFreezeError("Causal analysis release has the wrong status.")
    if manifest.get("causal_outcomes_accessed_by_freezer") is not False:
        raise CausalAnalysisFreezeError("Causal outcomes were accessed by the freezer.")
    with inventory_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != int(manifest.get("source_count", -1)):
        raise CausalAnalysisFreezeError("Causal release source count differs.")
    for row in rows:
        frozen = release / "source_snapshot" / row["path"]
        live = repo_root / row["path"]
        if not frozen.is_file() or sha256(frozen) != row["sha256"]:
            raise CausalAnalysisFreezeError(f"Frozen source mismatch: {row['path']}")
        if not live.is_file() or sha256(live) != row["sha256"]:
            raise CausalAnalysisFreezeError(f"Live analysis source drift: {row['path']}")
    result = dict(manifest)
    result["release_contents_sha256"] = sha256(release / "CONTENTS.sha256")
    result["release_path"] = str(release)
    return result


def freeze(repo_root: Path, extension_release: Path, contract_path: Path,
           output: Path, archive: Path | None,
           carried_extension_release: Path | None = None,
           operational_merge_manifest: Path | None = None,
           intermediate_extension_release: Path | None = None) -> dict[str, Any]:
    repo_root, output = repo_root.resolve(), output.resolve()
    if output.exists() or (archive is not None and archive.resolve().exists()):
        raise CausalAnalysisFreezeError("Analysis release/archive already exists.")
    # Training releases are historical immutable inputs to this later analysis
    # freeze. Requiring their entire source snapshots to equal the current live
    # repository would incorrectly forbid prospective post-training analysis
    # code and test additions. Verify the frozen v4 release internally, then
    # snapshot and hash the current analysis SOURCES independently below.
    extension = verify_frozen_extension_integrity(extension_release.resolve())
    carried_extension = None
    intermediate_extension = None
    merge_evidence = None
    if carried_extension_release is not None or operational_merge_manifest is not None:
        if carried_extension_release is None or operational_merge_manifest is None:
            raise CausalAnalysisFreezeError(
                "Mixed-revision analysis requires both the carried release and merge manifest.")
        carried_extension = verify_frozen_extension_integrity(
            carried_extension_release.resolve())
        merge_evidence = json.loads(operational_merge_manifest.read_text(
            encoding="utf-8"))
        if merge_evidence.get("status") not in {
                "complete_70_v2_plus_60_v3_operational_merge",
                "complete_70_v2_plus_31_v3_plus_29_v4_operational_merge"}:
            raise CausalAnalysisFreezeError("Operational merge evidence has the wrong status.")
        releases = merge_evidence.get("releases", {})
        if merge_evidence["status"] == \
                "complete_70_v2_plus_31_v3_plus_29_v4_operational_merge":
            if intermediate_extension_release is None:
                raise CausalAnalysisFreezeError(
                    "Three-revision analysis requires the intermediate v3 release.")
            intermediate_extension = verify_frozen_extension_integrity(
                intermediate_extension_release.resolve())
            if (releases.get("v2", {}).get("contents_sha256") !=
                    carried_extension["release_contents_sha256"] or
                    releases.get("v3", {}).get("contents_sha256") !=
                    intermediate_extension["release_contents_sha256"] or
                    releases.get("v4", {}).get("contents_sha256") !=
                    extension["release_contents_sha256"]):
                raise CausalAnalysisFreezeError(
                    "Operational merge is not bound to the supplied v2/v3/v4 releases.")
        elif (releases.get("original", {}).get("contents_sha256") !=
              carried_extension["release_contents_sha256"] or
              releases.get("retry", {}).get("contents_sha256") !=
              extension["release_contents_sha256"]):
            raise CausalAnalysisFreezeError(
                "Operational merge is not bound to the supplied v2/v3 releases.")
    contract = load_contract(contract_path.resolve())
    if contract.raw["confirmatory_claim_permitted"] is not False:
        raise CausalAnalysisFreezeError("Analysis contract improperly permits confirmation.")
    for relative in SOURCES:
        if not (repo_root / relative).is_file():
            raise CausalAnalysisFreezeError(f"Required analysis source missing: {relative}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        snapshot = temporary / "source_snapshot"
        inventory: list[dict[str, Any]] = []
        for relative in SOURCES:
            source = repo_root / relative
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            inventory.append({"path": relative, "sha256": sha256(destination),
                              "size_bytes": destination.stat().st_size})
        materialize_plan(contract_path.resolve(), temporary / "causal_contrast_plan.csv")
        with (temporary / "source_inventory.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
            writer.writeheader(); writer.writerows(inventory)
        pre_scoring_correction = contract.raw.get("analysis_status") == \
            "operational_correction_before_causal_outcome_scoring"
        manifest = {
            "schema_version": 1,
            "release_status": (
                "frozen_pre_causal_outcome_scoring_analysis_plan"
                if pre_scoring_correction else
                "frozen_pre_causal_evaluation_analysis_plan"),
            "analysis_id": contract.raw["analysis_id"],
            "analysis_contract_sha256": contract.sha256,
            "publication_extension_release_contents_sha256":
                extension["release_contents_sha256"],
            "carried_publication_extension_release_contents_sha256": (
                carried_extension["release_contents_sha256"]
                if carried_extension is not None else None),
            "intermediate_publication_extension_release_contents_sha256": (
                intermediate_extension["release_contents_sha256"]
                if intermediate_extension is not None else None),
            "operational_merge_manifest_sha256": (
                sha256(operational_merge_manifest)
                if operational_merge_manifest is not None else None),
            "mixed_revision_carry_forward": merge_evidence is not None,
            "v2_carried_count": (merge_evidence.get("v2_carried_count")
                                 if merge_evidence else 0),
            "v3_retry_count": (
                merge_evidence.get(
                    "v3_retry_count",
                    60 if merge_evidence.get("status") ==
                    "complete_70_v2_plus_31_v3_plus_29_v4_operational_merge"
                    else 0)
                if merge_evidence else 0),
            "v3_carried_count": (merge_evidence.get("v3_carried_count")
                                  if merge_evidence else 0),
            "v4_retry_count": (merge_evidence.get("v4_retry_count")
                                if merge_evidence else 0),
            "publication_extension_program_sha256": extension["program_sha256"],
            "source_count": len(inventory),
            "contrast_count": 12,
            "expected_policy_count": 130,
            "expected_ensemble_count": 13,
            "expected_scored_strategy_count": 143,
            "causal_outcomes_accessed_by_freezer": False,
            "operational_revision": contract.raw.get(
                "operational_revision",
                "replay_v2_centralized_checkpoint_audit_authorization"),
            "analysis_runtime_revision":
                "matplotlib_boxplot_explicit_tick_labels_v1",
            "analysis_runtime_revision_scientific_effect": "none",
            "causal_performance_outcomes_scored_before_runtime_revision": True,
            "runtime_revision_scope":
                "boxplot keyword compatibility and resumable immutable staging only",
            "operational_revision_reason": (
                contract.raw.get("revision_disclosure",
                    "Replay is authorized only through the centralized audit.")),
            "prior_failed_replay_exported_policy_count": 0,
            "prior_completed_replay_policy_count": (
                130 if contract.raw.get("causal_policy_replay_completed_before_revision")
                else 0),
            "causal_performance_outcomes_scored_before_freeze":
                contract.raw.get(
                    "causal_performance_outcomes_scored_before_revision", False),
            "sample_acceptance_contract_corrected": pre_scoring_correction,
            "estimand_or_scoring_sample_changed": False,
            "scientific_analysis_contract_changed": False,
            "consumed_main_holdout_declared": True,
            "evidence_class": "post_holdout_explanatory",
            "confirmatory_claim_permitted": False,
            "next_action": (
                "rebind audited weights and ensembles, then score causal outcomes once"
                if pre_scoring_correction else
                "audit training, generate weights, construct ensembles, and score once"),
        }
        (temporary / "causal_analysis_release_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "READ_ONLY_RELEASE.txt").write_text(
            ("Do not edit. This freezes the disclosed accounting correction "
             "before causal outcome scoring.\n" if pre_scoring_correction else
             "Do not edit. This freezes post-holdout explanatory analysis "
             "before causal evaluation.\n"),
            encoding="utf-8")
        lines = []
        for path in sorted(temporary.rglob("*")):
            if path.is_file() and path.name != "CONTENTS.sha256":
                lines.append(f"{sha256(path)}  {path.relative_to(temporary).as_posix()}")
        (temporary / "CONTENTS.sha256").write_text(
            "\n".join(lines) + "\n", encoding="ascii")
        os.replace(temporary, output)
        if archive is not None:
            from publication_pipeline_draft.freeze_training_release import deterministic_tar
            archive = archive.resolve()
            archive.parent.mkdir(parents=True, exist_ok=True)
            deterministic_tar(output, archive)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--extension-release", required=True, type=Path)
    parser.add_argument("--contract", type=Path, default=Path(
        "publication_pipeline_draft/config/causal_analysis_contract_v1.json"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--carried-extension-release", type=Path)
    parser.add_argument("--intermediate-extension-release", type=Path)
    parser.add_argument("--operational-merge-manifest", type=Path)
    args = parser.parse_args()
    try:
        result = freeze(args.repo_root, args.extension_release, args.contract,
                        args.output, args.archive, args.carried_extension_release,
                        args.operational_merge_manifest,
                        args.intermediate_extension_release)
    except (CausalAnalysisContractError, ExtensionReleaseError, OSError, ValueError) as error:
        print(f"CAUSAL ANALYSIS FREEZE FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
