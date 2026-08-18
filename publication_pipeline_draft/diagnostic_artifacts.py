#!/usr/bin/env python3
"""Create paper-ready synthetic-data and training diagnostic artifacts.

This is a read-only adapter for generator/training outputs. It never opens the
locked holdout and refuses to replace missing diagnostics with fallback values.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from publication_pipeline import ProtocolError, parse_bool, save_figure, sha256_file


SYNTHETIC_FILES = [
    "fidelity_metrics.csv",
    "correlation_comparison.csv",
    "tail_dependence_comparison.csv",
    "temporal_dependence.csv",
    "summary_statistics.csv",
    "tail_risk.csv",
]


def require_columns(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [name for name in columns if name not in frame]
    if missing:
        raise ProtocolError(f"{source} is missing columns: {', '.join(missing)}")


def copy_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def synthetic_artifacts(source: Path, output: Path) -> dict[str, object]:
    paths = {name: source / name for name in SYNTHETIC_FILES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ProtocolError("Missing synthetic diagnostic files: " + ", ".join(missing))
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    fidelity = frames["fidelity_metrics.csv"]
    correlation = frames["correlation_comparison.csv"]
    tail = frames["tail_dependence_comparison.csv"]
    temporal = frames["temporal_dependence.csv"]
    require_columns(
        fidelity,
        ["asset", "historical_mean", "synthetic_mean", "historical_sd", "synthetic_sd",
         "historical_q05", "synthetic_q05", "historical_cvar05", "synthetic_cvar05", "pass_marginals"],
        paths["fidelity_metrics.csv"],
    )
    require_columns(
        correlation,
        ["asset_i", "asset_j", "historical_correlation", "synthetic_correlation", "pass_correlation"],
        paths["correlation_comparison.csv"],
    )
    require_columns(
        tail,
        ["asset_i", "asset_j", "historical_lower_tail", "synthetic_lower_tail",
         "historical_tail_events", "historical_joint_tail_events", "pass_lower_tail"],
        paths["tail_dependence_comparison.csv"],
    )
    require_columns(
        temporal,
        ["asset", "historical_lag1", "synthetic_lag1", "historical_squared_lag1",
         "synthetic_squared_lag1", "pass_temporal"],
        paths["temporal_dependence.csv"],
    )
    table_dir = output / "tables"
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        copy_table(frame, table_dir / f"synthetic_{name}")
    summary_rows = [
        ["marginal assets: strict target", len(fidelity),
         sum(parse_bool(x, "pass_marginals") for x in fidelity["pass_marginals"])],
        ["correlation pairs: strict target", len(correlation),
         sum(parse_bool(x, "pass_correlation") for x in correlation["pass_correlation"])],
        ["lower-tail pairs: interval compatibility", len(tail),
         sum(parse_bool(x, "pass_lower_tail") for x in tail["pass_lower_tail"])],
        ["temporal assets", len(temporal),
         sum(parse_bool(x, "pass_temporal") for x in temporal["pass_temporal"])],
    ]
    if "statistically_compatible" in fidelity:
        summary_rows.insert(1, [
            "marginal assets: sampling-aware gate", len(fidelity),
            sum(parse_bool(x, "statistically_compatible")
                for x in fidelity["statistically_compatible"]),
        ])
    if "statistically_compatible" in correlation:
        summary_rows.insert(3, [
            "correlation pairs: sampling-aware gate", len(correlation),
            sum(parse_bool(x, "statistically_compatible")
                for x in correlation["statistically_compatible"]),
        ])
    summary = pd.DataFrame(
        summary_rows,
        columns=["diagnostic_family", "tests", "passed"],
    )
    summary["pass_fraction"] = summary["passed"] / summary["tests"]
    copy_table(summary, table_dir / "table_s01_synthetic_gate_summary.csv")

    comparisons = [
        ("Mean", "historical_mean", "synthetic_mean"),
        ("Standard deviation", "historical_sd", "synthetic_sd"),
        ("5% quantile", "historical_q05", "synthetic_q05"),
        ("5% CVaR", "historical_cvar05", "synthetic_cvar05"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.2))
    for ax, (title, historical, synthetic) in zip(axes.ravel(), comparisons):
        x, y = fidelity[historical].to_numpy(float), fidelity[synthetic].to_numpy(float)
        low, high = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([low, high], [low, high], color="#777777", linestyle="--", linewidth=0.8)
        ax.scatter(x, y, color="#0072B2")
        for _, row in fidelity.iterrows():
            ax.annotate(row["asset"], (row[historical], row[synthetic]), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("Historical")
        ax.set_ylabel("Synthetic")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figure_dir / "figure_s01_marginal_fidelity")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    for ax, data, hist_col, synth_col, title in [
        (axes[0], correlation, "historical_correlation", "synthetic_correlation", "Pairwise correlation"),
        (axes[1], tail, "historical_lower_tail", "synthetic_lower_tail", "5% lower-tail co-exceedance"),
    ]:
        x, y = data[hist_col].to_numpy(float), data[synth_col].to_numpy(float)
        low, high = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([low, high], [low, high], color="#777777", linestyle="--", linewidth=0.8)
        ax.scatter(x, y, color="#D55E00", alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Historical")
        ax.set_ylabel("Synthetic")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figure_dir / "figure_s02_dependence_fidelity")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3))
    for ax, hist_col, synth_col, title in [
        (axes[0], "historical_lag1", "synthetic_lag1", "Lag-1 return dependence"),
        (axes[1], "historical_squared_lag1", "synthetic_squared_lag1", "Lag-1 squared-return dependence"),
    ]:
        x, y = temporal[hist_col].to_numpy(float), temporal[synth_col].to_numpy(float)
        low, high = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([low, high], [low, high], color="#777777", linestyle="--", linewidth=0.8)
        ax.scatter(x, y, color="#009E73")
        for _, row in temporal.iterrows():
            ax.annotate(row["asset"], (row[hist_col], row[synth_col]), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("Historical")
        ax.set_ylabel("Synthetic")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figure_dir / "figure_s03_temporal_fidelity")
    return {
        "files": {name: sha256_file(path) for name, path in paths.items()},
        "gate_summary": summary.to_dict(orient="records"),
    }


def read_seed_file(run_dirs: list[Path], filename: str) -> pd.DataFrame:
    rows = []
    for directory in run_dirs:
        path = directory / filename
        if not path.is_file():
            raise ProtocolError(f"Missing {filename} for seed directory {directory}")
        frame = pd.read_csv(path)
        try:
            seed = int(directory.name.removeprefix("seed_"))
        except ValueError as error:
            raise ProtocolError(f"Unexpected seed directory name: {directory.name}") from error
        frame.insert(0, "seed", seed)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def seed_from_directory(directory: Path) -> int:
    try:
        return int(directory.name.removeprefix("seed_"))
    except ValueError as error:
        raise ProtocolError(f"Unexpected seed directory name: {directory.name}") from error


def training_provenance_artifacts(
    run_dirs: list[Path], output: Path
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Collect and validate every artifact needed before holdout access."""
    table_dir = output / "tables"
    raw_dir = output / "raw"
    inventory_rows: list[dict[str, object]] = []
    sanity_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    checkpoint_rows: list[pd.DataFrame] = []
    validation_rows: list[pd.DataFrame] = []
    schedule_rows: list[pd.DataFrame] = []
    data_hash_rows: list[pd.DataFrame] = []
    code_hash_rows: list[pd.DataFrame] = []
    required_files = [
        "finetune_validation_metrics.csv",
        "finetune_episode_schedule.csv",
        "finetune_selection.txt",
        "data_hashes.csv",
        "code_hashes.csv",
        "run_manifest.rds",
        "td3_lstm_vine_pretrained.pt",
        "td3_lstm_vine_full.pt",
        "sanity_no_holdout/checkpoint_integrity.csv",
        "sanity_no_holdout/sanity_report.json",
    ]
    for directory in run_dirs:
        seed = seed_from_directory(directory)
        for relative in required_files:
            path = directory / relative
            if not path.is_file():
                raise ProtocolError(f"Missing publication artifact for seed {seed}: {path}")
            inventory_rows.append(
                {
                    "seed": seed,
                    "artifact": relative,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

        validation = pd.read_csv(directory / "finetune_validation_metrics.csv")
        validation.insert(0, "seed", seed)
        validation_rows.append(validation)
        schedule = pd.read_csv(directory / "finetune_episode_schedule.csv")
        schedule.insert(0, "seed", seed)
        schedule_rows.append(schedule)
        selection = (directory / "finetune_selection.txt").read_text(encoding="utf-8").strip()
        if not selection:
            raise ProtocolError(f"Empty fine-tune selection record for seed {seed}.")
        selection_rows.append({"seed": seed, "selection_record": selection})

        integrity_path = directory / "sanity_no_holdout/checkpoint_integrity.csv"
        integrity = pd.read_csv(integrity_path)
        require_columns(
            integrity,
            ["model", "path", "sha256", "architecture_match",
             "all_checkpoint_tensors_finite", "tensor_count", "tensor_elements"],
            integrity_path,
        )
        if set(integrity["model"]) != {"pretrained", "full"}:
            raise ProtocolError(f"Seed {seed} checkpoint integrity must contain pretrained and full rows.")
        architecture_ok = [parse_bool(x, "architecture_match") for x in integrity["architecture_match"]]
        tensors_ok = [parse_bool(x, "all_checkpoint_tensors_finite") for x in integrity["all_checkpoint_tensors_finite"]]
        if not all(architecture_ok) or not all(tensors_ok):
            raise ProtocolError(f"Seed {seed} contains an invalid checkpoint integrity row.")
        for _, checkpoint in integrity.iterrows():
            local_checkpoint = directory / Path(str(checkpoint["path"])).name
            actual = sha256_file(local_checkpoint)
            if actual.lower() != str(checkpoint["sha256"]).lower():
                raise ProtocolError(
                    f"Checkpoint hash mismatch for seed {seed}, model {checkpoint['model']}."
                )
        integrity.insert(0, "seed", seed)
        checkpoint_rows.append(integrity)

        report_path = directory / "sanity_no_holdout/sanity_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not bool(report.get("overall_pass")) or not bool(report.get("publication_behavior_pass")):
            raise ProtocolError(f"Seed {seed} sanity report does not pass both publication gates.")
        sanity_rows.append(
            {
                "seed": seed,
                "protocol": report.get("protocol", ""),
                "episodes": report.get("episodes", np.nan),
                "episode_length": report.get("episode_length", np.nan),
                "obs_dim": report.get("obs_dim", np.nan),
                "action_dim": report.get("action_dim", np.nan),
                "vine_dim": report.get("vine_dim", np.nan),
                "warning_count": len(report.get("warnings", [])),
                "warnings": " | ".join(report.get("warnings", [])),
                "diagnostic_notes": " | ".join(report.get("diagnostic_notes", [])),
                "publication_behavior_pass": bool(report.get("publication_behavior_pass")),
                "overall_pass": bool(report.get("overall_pass")),
            }
        )

        for kind, container in [("data", data_hash_rows), ("code", code_hash_rows)]:
            hash_path = directory / f"{kind}_hashes.csv"
            hashes = pd.read_csv(hash_path)
            require_columns(hashes, ["path", "md5"], hash_path)
            hashes.insert(0, "seed", seed)
            hashes.insert(1, "artifact_kind", kind)
            hashes["normalized_path"] = hashes["path"].map(
                lambda value: str(value).replace("\\", "/").split("/copula-portfolio-clean/")[-1]
            )
            container.append(hashes)

    frames = {
        "finetune_validation": pd.concat(validation_rows, ignore_index=True),
        "finetune_schedule": pd.concat(schedule_rows, ignore_index=True),
        "finetune_selection": pd.DataFrame(selection_rows),
        "checkpoint_integrity": pd.concat(checkpoint_rows, ignore_index=True),
        "sanity_reports": pd.DataFrame(sanity_rows),
        "data_hashes": pd.concat(data_hash_rows, ignore_index=True),
        "code_hashes": pd.concat(code_hash_rows, ignore_index=True),
    }
    combined_hashes = pd.concat([frames["data_hashes"], frames["code_hashes"]], ignore_index=True)
    consensus = (
        combined_hashes.groupby(["artifact_kind", "normalized_path"], as_index=False)
        .agg(seed_count=("seed", "nunique"), distinct_hashes=("md5", "nunique"), md5=("md5", "first"))
    )
    expected_seed_count = len(run_dirs)
    if (consensus["seed_count"] != expected_seed_count).any() or (consensus["distinct_hashes"] != 1).any():
        failures = consensus[
            (consensus["seed_count"] != expected_seed_count) | (consensus["distinct_hashes"] != 1)
        ]
        raise ProtocolError(
            "Code/data hashes are not identical across every seed: "
            + ", ".join(failures["normalized_path"].tolist())
        )
    frames["hash_consensus"] = consensus
    inventory = pd.DataFrame(inventory_rows)
    copy_table(frames["finetune_validation"], table_dir / "table_t05_finetune_validation.csv")
    copy_table(frames["finetune_selection"], table_dir / "table_t06_finetune_selection.csv")
    copy_table(frames["checkpoint_integrity"], table_dir / "table_t07_checkpoint_integrity.csv")
    copy_table(frames["sanity_reports"], table_dir / "table_t08_sanity_reports.csv")
    copy_table(consensus, table_dir / "table_t09_code_data_hash_consensus.csv")
    copy_table(inventory, table_dir / "table_t10_artifact_inventory.csv")
    copy_table(frames["finetune_schedule"], raw_dir / "finetune_episode_schedule_all_seeds.csv")
    copy_table(frames["data_hashes"], raw_dir / "data_hashes_all_seeds.csv")
    copy_table(frames["code_hashes"], raw_dir / "code_hashes_all_seeds.csv")
    return frames, inventory


def rolling_training_summary(episodes: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    required = ["seed", "stage", "episode", "reward", "terminal_wealth", "mean_turnover", "mean_cvar", "mean_gross_exposure"]
    require_columns(episodes, required, Path("training_episode_metrics.csv"))
    results = []
    for (seed, stage), group in episodes.groupby(["seed", "stage"], sort=False):
        group = group.sort_values("episode").copy()
        for metric in ["reward", "terminal_wealth", "mean_turnover", "mean_cvar", "mean_gross_exposure"]:
            group[f"rolling_{metric}"] = group[metric].rolling(window, min_periods=max(5, window // 5)).mean()
        results.append(group)
    return pd.concat(results, ignore_index=True)


def training_artifacts(rl_runs: Path, output: Path, expected_seeds: int | None) -> dict[str, object]:
    status_path = rl_runs / "seed_sweep_status.csv"
    if not status_path.is_file():
        raise ProtocolError(f"Missing seed sweep status: {status_path}")
    status = pd.read_csv(status_path)
    require_columns(status, ["seed", "output_dir", "training_status", "sanity_status", "no_holdout_gate_pass"], status_path)
    status = status.sort_values("seed").reset_index(drop=True)
    if expected_seeds is not None and len(status) != expected_seeds:
        raise ProtocolError(f"Expected {expected_seeds} completed seed rows; found {len(status)}.")
    gate_pass = pd.Series(
        [parse_bool(x, "no_holdout_gate_pass") for x in status["no_holdout_gate_pass"]],
        index=status.index,
    )
    if not ((status["training_status"] == 0) & (status["sanity_status"] == 0) & gate_pass).all():
        raise ProtocolError("At least one seed did not complete and pass the no-holdout gate.")
    run_dirs = []
    for path_text in status["output_dir"]:
        path = Path(path_text)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.is_dir():
            raise ProtocolError(f"Seed output directory not found: {path}")
        run_dirs.append(path)
    episodes = read_seed_file(run_dirs, "training_episode_metrics.csv")
    updates = read_seed_file(run_dirs, "training_update_metrics.csv")
    gates = read_seed_file(run_dirs, "pretraining_behavior_gate.csv")
    policies = read_seed_file(run_dirs, "sanity_no_holdout/policy_summary.csv")
    sensitivities = read_seed_file(run_dirs, "sanity_no_holdout/state_sensitivity_summary.csv")
    provenance, artifact_inventory = training_provenance_artifacts(run_dirs, output)
    rolling = rolling_training_summary(episodes)
    table_dir = output / "tables"
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [
        ("table_t01_seed_sweep_status.csv", status),
        ("table_t02_pretraining_gates.csv", gates),
        ("table_t03_policy_sanity.csv", policies),
        ("table_t04_state_sensitivity.csv", sensitivities),
    ]:
        copy_table(frame, table_dir / name)
    copy_table(episodes, output / "raw" / "training_episode_metrics_all_seeds.csv")
    copy_table(updates, output / "raw" / "training_update_metrics_all_seeds.csv")

    pretrain = rolling[rolling["stage"] == "pretrain"]
    metrics = ["rolling_reward", "rolling_terminal_wealth", "rolling_mean_turnover", "rolling_mean_gross_exposure"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    for ax, metric in zip(axes.ravel(), metrics):
        pivot = pretrain.pivot_table(index="episode", columns="seed", values=metric)
        center = pivot.median(axis=1)
        lower = pivot.quantile(0.10, axis=1)
        upper = pivot.quantile(0.90, axis=1)
        ax.plot(center.index, center, color="#0072B2", linewidth=1.3)
        ax.fill_between(center.index, lower, upper, color="#56B4E9", alpha=0.28)
        ax.set_title(metric.removeprefix("rolling_").replace("_", " "))
        ax.set_xlabel("Pre-training episode")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figure_dir / "figure_t01_pretraining_stability")

    require_columns(updates, ["stage", "update", "critic_loss", "actor_loss", "twin_q_gap", "actor_grad_norm", "critic_grad_norm"], Path("training_update_metrics.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    for ax, metric in zip(axes.ravel(), ["critic_loss", "actor_loss", "twin_q_gap", "actor_grad_norm"]):
        clean = updates.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])
        # The selection diagnostic and the final all-history refit are two
        # branches starting from the same pretrained checkpoint and therefore
        # reuse update numbers. Plot only the branch that produces the frozen
        # full checkpoint; otherwise a line connects/overlays mutually
        # exclusive trajectories and creates a false instability signal.
        final_path = clean[clean["stage"] != "finetune_selection"].copy()
        for seed, group in final_path.groupby("seed"):
            group = group.sort_values("update")
            ax.plot(group["update"], group[metric].rolling(10, min_periods=1).median(), alpha=0.18, linewidth=0.6)
        summary = final_path.groupby("update")[metric].median()
        ax.plot(summary.index, summary, color="#D55E00", linewidth=1.3)
        pretrain = final_path[final_path["stage"] == "pretrain"]
        if not pretrain.empty:
            ax.axvline(pretrain["update"].max(), color="#666666", linestyle="--", linewidth=0.8)
        if metric in {"critic_loss", "twin_q_gap", "actor_grad_norm"} and (summary > 0).all():
            ax.set_yscale("log")
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("Gradient update")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, figure_dir / "figure_t02_optimizer_diagnostics")
    return {
        "seed_count": len(status),
        "gate_pass_count": int(gate_pass.sum()),
        "checkpoint_count": int(len(provenance["checkpoint_integrity"])),
        "artifact_inventory_count": int(len(artifact_inventory)),
        "code_data_hash_consensus_count": int(len(provenance["hash_consensus"])),
        "status_sha256": sha256_file(status_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--synthetic-diagnostics", type=Path)
    parser.add_argument("--rl-runs", type=Path)
    parser.add_argument("--expected-seeds", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Output already exists and will not be overwritten: {args.output}")
    if args.synthetic_diagnostics is None and args.rl_runs is None:
        raise SystemExit("Provide --synthetic-diagnostics and/or --rl-runs.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output.name}_", dir=str(args.output.parent)))
    manifest: dict[str, object] = {}
    try:
        if args.synthetic_diagnostics is not None:
            manifest["synthetic"] = synthetic_artifacts(args.synthetic_diagnostics, temporary)
        if args.rl_runs is not None:
            manifest["training"] = training_artifacts(args.rl_runs, temporary, args.expected_seeds)
        (temporary / "diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Diagnostic artifacts written to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
