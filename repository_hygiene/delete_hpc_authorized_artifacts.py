#!/usr/bin/env python3
"""Delete only artifacts authorized by a passing canonical-copy audit."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


CONFIRMATION = "DELETE_AUTHORIZED_GENERATED_ARTIFACTS"

DIRECTORIES = (
    ".pytest_cache",
    "publication_pipeline_draft/__pycache__",
    "publication_pipeline_draft/terminal_publication/__pycache__",
    "publication_pipeline_draft/tests/__pycache__",
    "publication_pipeline_draft/tikz_figures/__pycache__",
    "rl/__pycache__",
    "tmp",
    "analysis_outputs/causal_policy_weights_v2_v3_v4_retry1",
    "data/focused_original_7asset_runs_v1/retrospective_original_7asset_expanding_24m_v1_w01/failed_attempts",
    "publication_eval/static_vine_convergence_probe_retrospective_original_7asset_expanding_24m_v1_w01_v3",
    "publication_eval/vine_convergence_probe_retrospective_original_7asset_expanding_24m_v1_w01_v4",
    "analysis_outputs/SYNTHETIC_DOSE_RESPONSE_V1_INTERPRETATION_files",
    # Canonically archived training work trees.
    "data/rl_runs",
    "data/publication_training_artifacts_20seeds",
    "data/no_vine_rl_runs_secondary_v3",
    "data/publication_no_vine_training_artifacts_10seeds",
    "data/no_vine_rl_runs_4gpu",
    "data/masked_pretraining_control_runs_v1",
    "analysis_outputs/masked_pretraining_controls_v1_weights",
    "analysis_outputs/masked_pretraining_controls_v1_audit",
    "logs/masked_pretraining_controls_v1",
    "data/mixed_pretraining_runs_v1",
    "analysis_outputs/mixed_pretraining_response_v1_weights",
    "analysis_outputs/mixed_pretraining_response_v1_audit",
    "logs/mixed_pretraining_response_v1",
    # Causal replay/accounting intermediates embedded in the final release.
    "analysis_outputs/causal_analysis_results_v2_v3_v4_plot_runtime_v1",
    "analysis_outputs/causal_common_accounting_v2_v3_v4_accounting_v2",
    "analysis_outputs/causal_common_accounting_v2_v3_v4_plot_runtime_v1",
    "analysis_outputs/causal_evaluation_interface_v2_v3_v4_accounting_v2",
    "analysis_outputs/causal_evaluation_interface_v2_v3_v4_plot_runtime_v1",
    "analysis_outputs/causal_evaluation_interface_v2_v3_v4_replay_v2",
    "analysis_outputs/causal_policy_ensembles_v2_v3_v4_accounting_v2",
    "analysis_outputs/causal_policy_ensembles_v2_v3_v4_replay_v2",
    "analysis_outputs/causal_policy_weights_v2_v3_v4",
    "analysis_outputs/causal_policy_weights_v2_v3_v4_accounting_v2",
    "analysis_outputs/causal_sweep_audit_v2_v3_v4",
    "analysis_outputs/terminal_robustness_v1_cleanroom",
)

FILES = (
    "data/focused_original_7asset_runs_v1/retrospective_original_7asset_expanding_24m_v1_w01/training_data/synthetic_bundle_manifest.strict_v1_failed.json",
    "data/focused_original_7asset_runs_v1/retrospective_original_7asset_expanding_24m_v1_w01/training_data/synthetic_returns.strict_v1_failed.RData",
    "analysis_outputs/SYNTHETIC_DOSE_RESPONSE_V1_INTERPRETATION.html",
    "figures/synthetic_monthly_return_distributions.pdf",
    "figures/wealth_curves.pdf",
    "manuscript_revision_causal_v1/publication_mixed_pretraining_v1/figures/tikz/preview_mixed_pretraining_figure.aux",
    "manuscript_revision_causal_v1/publication_mixed_pretraining_v1/figures/tikz/preview_mixed_pretraining_figure.log",
    "manuscript_revision_causal_v1/publication_mixed_pretraining_v1/figures/tikz/preview_mixed_pretraining_figure.pdf",
    "manuscript_revision_causal_v1/publication_terminal_v1/figures/tikz/preview_terminal_figures.aux",
    "manuscript_revision_causal_v1/publication_terminal_v1/figures/tikz/preview_terminal_figures.fdb_latexmk",
    "manuscript_revision_causal_v1/publication_terminal_v1/figures/tikz/preview_terminal_figures.fls",
    "manuscript_revision_causal_v1/publication_terminal_v1/figures/tikz/preview_terminal_figures.log",
    "manuscript_revision_causal_v1/publication_terminal_v1/figures/tikz/preview_terminal_figures.pdf",
    "logs/causal_evaluation_v2_v3_v4_freeze_plan_retry1.log",
    "logs/causal_evaluation_v2_v3_v4_preflight_retry1.log",
    "logs/focused_walk_forward_v1_validation.log",
    "logs/focused_walk_forward_v1_validation_retry1.log",
    "logs/publication_extension_v4_validation_retry1.log",
    "logs/synthetic_dose_response_v1_freeze_retry1.log",
    "logs/no_vine_secondary_v3_preflight_v8.json",
    "analysis_outputs/causal_strategy_periods_v2_v3_v4_accounting_v2.csv",
    "analysis_outputs/causal_strategy_periods_v2_v3_v4_accounting_v2.csv.manifest.json",
    "analysis_outputs/causal_strategy_periods_v2_v3_v4_plot_runtime_v1.csv",
    "analysis_outputs/causal_strategy_periods_v2_v3_v4_plot_runtime_v1.csv.manifest.json",
    "analysis_outputs/terminal_robustness_v1_cleanroom_verification.json",
)

PROTECTED_PREFIXES = (
    "frozen_releases",
    "protocol_manifests",
    "provenance_environment_extension_v2",
    "provenance_environment_extension_v3_retry",
    "provenance_environment_extension_v4",
    "provenance_environment_v4",
    "benchmark_models",
    "config",
    "eval",
    "helper",
    "hpc",
    "rl",
    "tests",
)


def within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def protected(relative: Path) -> bool:
    value = relative.as_posix().rstrip("/")
    return any(value == prefix or value.startswith(prefix + "/") for prefix in PROTECTED_PREFIXES)


def collect_targets(root: Path) -> list[Path]:
    targets = [root / item for item in DIRECTORIES + FILES]
    logs = root / "logs"
    if logs.is_dir():
        targets.extend(logs.rglob("*.pid"))
        targets.extend(logs.glob("masked_pretraining_controls_v1*.log"))
        targets.extend(logs.glob("mixed_pretraining_response_v1*.log"))
    unique: dict[str, Path] = {}
    for path in targets:
        resolved = path.resolve()
        if not within(resolved, root) or resolved == root.resolve():
            raise RuntimeError(f"Unsafe cleanup target: {path}")
        relative = resolved.relative_to(root.resolve())
        # Exact cache directories below protected source roots are allowed.
        cache_exception = relative.name == "__pycache__"
        if protected(relative) and not cache_exception:
            raise RuntimeError(f"Cleanup target enters protected prefix: {relative}")
        unique[resolved.as_posix()] = resolved
    return sorted(unique.values(), key=lambda item: (len(item.parts), item.as_posix()), reverse=True)


def load_audit(root: Path, path: Path) -> dict[str, object]:
    audit_path = path if path.is_absolute() else root / path
    if not audit_path.is_file():
        raise RuntimeError(f"Canonical audit does not exist: {audit_path}")
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    if report.get("status") != "canonical_copy_audit_passed":
        raise RuntimeError("Canonical audit status is not canonical_copy_audit_passed.")
    required_groups = {"A", "C1", "C2", "C3", "C4", "C5", "C6", "C7"}
    if set(report.get("authorized_cleanup_groups", [])) != required_groups:
        raise RuntimeError("Canonical audit does not authorize the expected cleanup groups.")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("repository_hygiene/hpc_canonical_copy_audit_v1/canonical_copy_audit.json"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    arguments = parser.parse_args()

    root = arguments.repo_root.resolve()
    if not (root / "publication_pipeline_draft").is_dir():
        print(f"CLEANUP FAILURE: Not a repository root: {root}", file=sys.stderr)
        return 1
    try:
        audit = load_audit(root, arguments.audit)
        targets = collect_targets(root)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"CLEANUP FAILURE: {error}", file=sys.stderr)
        return 1

    existing = [path for path in targets if path.exists() or path.is_symlink()]
    print(f"Canonical audit status: {audit['status']}")
    print(f"Existing authorized targets: {len(existing)}")
    for path in existing:
        print(path.relative_to(root).as_posix())

    if not arguments.execute:
        print("DRY RUN ONLY. No files were deleted.")
        return 0
    if arguments.confirm != CONFIRMATION:
        print(
            f"CLEANUP FAILURE: --execute requires --confirm {CONFIRMATION}",
            file=sys.stderr,
        )
        return 1

    removed: list[str] = []
    for path in existing:
        relative = path.relative_to(root).as_posix()
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(relative)

    receipt = {
        "schema_version": 1,
        "status": "authorized_generated_artifacts_deleted",
        "canonical_audit": arguments.audit.as_posix(),
        "removed_count": len(removed),
        "removed": sorted(removed),
        "blocked_raw_roots_preserved": audit.get("blocked_raw_run_roots", []),
    }
    receipt_path = root / "repository_hygiene" / "hpc_cleanup_receipt_v1.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
