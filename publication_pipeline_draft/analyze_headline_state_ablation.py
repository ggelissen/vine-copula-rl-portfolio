#!/usr/bin/env python3
"""Rescore four matched mixed-curriculum input states for manuscript Table 8."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from publication_pipeline_draft.analyze_causal_results import (
    annualization, annualized_ce, circular_indices, crra_utility, holm_adjust,
    metrics,
)
from publication_pipeline_draft.analyze_synthetic_dose_response import load_weight
from publication_pipeline_draft.headline_state_ablation import (
    ARMS, HEADLINE, headline_evidence, load_design, verify_release,
)
from publication_pipeline_draft.publication_pipeline import (
    KEYS, Contract, read_realized_panel, score_strategy, validate_weight_matrix,
)
from publication_pipeline_draft.synthetic_dose_protocol import (
    DoseProtocolError, read_csv, require, sha256,
)

LABELS = {
    HEADLINE: "Compressed inputs (headline)",
    "scalar_scenario_cvar": "Scalar scenario-CVaR",
    "raw_vine_coordinates": "Raw vine coordinates",
    "full_state": "Full state",
}


def source_weights(repo: Path, design: dict[str, Any], new_manifest: Path,
                   realized: pd.DataFrame, assets: list[str],
                   evaluation: Contract) -> dict[tuple[str, int], pd.DataFrame]:
    release, headline_rows, _ = headline_evidence(repo, design)
    rows = read_csv(new_manifest)
    require(len(rows) == 30 and
            {(row["experiment_id"], int(row["seed"])) for row in rows} ==
            {(arm, seed) for arm in ARMS for seed in design["seeds"]},
            "New weight manifest must contain the exact 30 arm/seed keys.")
    weights = {}
    for row in rows:
        key = row["experiment_id"], int(row["seed"])
        weights[key] = load_weight(repo / row["path"], row["sha256"],
                                   realized, assets, evaluation)
    for row in headline_rows:
        seed = int(row["seed"])
        path = (release / "source_evidence/weight_evidence/weights/"
                "mixed_pretraining_plus_historical_finetuning" /
                f"seed_{seed}" / f"weights_rl_full_seed_{seed}.csv")
        weights[HEADLINE, seed] = load_weight(path, row["sha256"],
                                              realized, assets, evaluation)
    require(len(weights) == 40, "Exactly 40 policies are required.")
    return weights


def score_four(weights: dict[tuple[str, int], pd.DataFrame],
               realized: pd.DataFrame, assets: list[str],
               evaluation: Contract) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    columns = [f"w_{asset}" for asset in assets]
    paths = []
    ensembles = {}
    for (arm, seed), frame in sorted(weights.items()):
        scored = score_strategy(f"{arm}__seed_{seed}", frame,
                                realized, assets, evaluation)
        scored["arm_id"] = arm
        scored["level"] = "seed"
        scored["seed"] = seed
        paths.append(scored)
    for arm in (HEADLINE, *ARMS):
        members = [weights[arm, seed] for seed in sorted(
            seed for name, seed in weights if name == arm)]
        require(len(members) == 10, f"{arm} lacks ten seed policies.")
        ensemble = members[0][KEYS].copy()
        ensemble[columns] = np.stack(
            [frame[columns].to_numpy(float) for frame in members]).mean(axis=0)
        validate_weight_matrix(ensemble[columns].to_numpy(float),
                               f"{arm}_ensemble", evaluation)
        ensembles[arm] = ensemble
        scored = score_strategy(f"{arm}_ensemble", ensemble,
                                realized, assets, evaluation)
        scored["arm_id"] = arm
        scored["level"] = "ensemble"
        scored["seed"] = np.nan
        paths.append(scored)
    panel = pd.concat(paths, ignore_index=True)
    counts = panel.groupby("strategy_id").size()
    require(len(counts) == 44 and (counts == 24).all(),
            "Every policy and ensemble needs 24 locked periods.")
    require(set(panel["turnover_convention"]) == {"target_to_target_v1"},
            "Table 8 must use headline target-to-target cost accounting.")
    return panel, ensembles


def reference_check(repo: Path, design: dict[str, Any],
                    headline_metric: dict[str, Any]) -> dict[str, Any]:
    source = (repo / design["headline_evidence_release"] /
              "source_evidence/analysis_results/tables/"
              "mixed_pretraining_four_arm_metrics.csv")
    archived = [row for row in read_csv(source) if row["experiment_id"] ==
                "mixed_pretraining_plus_historical_finetuning"]
    require(len(archived) == 1, "Archived headline metric row is not unique.")
    declared = {
        "annualized_certainty_equivalent_return":
            design["evaluation"]["headline_annualized_ce"],
        "cagr": design["evaluation"]["headline_cagr"],
        "mean_monthly_turnover":
            design["evaluation"]["headline_mean_monthly_turnover"],
    }
    result = {}
    for name in ("total_return", "cagr", "annual_volatility", "sharpe_ratio",
                 "max_drawdown", "annualized_certainty_equivalent_return",
                 "mean_monthly_turnover", "mean_gross_exposure",
                 "total_transaction_cost", "total_financing_cost"):
        frozen = float(archived[0][name])
        observed = float(headline_metric[name])
        require((name not in declared or
                 abs(frozen - float(declared[name])) < 1e-12) and
                abs(observed - frozen) < 1e-10,
                f"Headline {name} failed exact Table 4/5 reconciliation: "
                f"rescored={observed}, archived={frozen}.")
        result[name] = {"archived": frozen, "rescored": observed}
    return result


def inference(headline: pd.DataFrame, alternative: pd.DataFrame,
              causal: dict[str, Any], seed: int) -> dict[str, float]:
    columns = ["decision_date", "holding_end_date"]
    left = headline.sort_values(columns).reset_index(drop=True)
    right = alternative.sort_values(columns).reset_index(drop=True)
    require(left[columns].equals(right[columns]) and len(left) == 22,
            "Input contrast does not share the exact 22-period calendar.")
    x = left["net_return"].to_numpy(float)
    y = right["net_return"].to_numpy(float)
    gamma = float(causal["economics"]["crra_gamma"])
    factor, _ = annualization(left, causal)
    utility = crra_utility(x, gamma) - crra_utility(y, gamma)
    observed_mean = float(utility.mean())
    effect = annualized_ce(x, gamma, factor) - annualized_ce(y, gamma, factor)
    replications = 9999
    rng = np.random.default_rng(seed)
    centered = utility - observed_mean
    null = np.empty(replications)
    effects = np.empty(replications)
    for index in range(replications):
        draw = circular_indices(rng, len(x), 3)
        null[index] = centered[draw].mean()
        effects[index] = (annualized_ce(x[draw], gamma, factor) -
                          annualized_ce(y[draw], gamma, factor))
    return {
        "annualized_ce_difference_headline_minus_alternative": effect,
        "mean_monthly_crra_utility_difference": observed_mean,
        "ci_lower_pointwise": float(np.quantile(effects, .025)),
        "ci_upper_pointwise": float(np.quantile(effects, .975)),
        "two_sided_p": float((1 + np.sum(np.abs(null) >= abs(observed_mean))) /
                             (replications + 1)),
    }


def latex_table(frame: pd.DataFrame) -> str:
    by_arm = frame.set_index("arm_id")
    lines = [
        "\\begin{table}[htb]", "\\centering",
        "\\caption{Policy inputs under the headline mixed-training curriculum: ten matched seeds per input state, 22 common complete periods, and target-to-target turnover. This post-holdout comparison is explanatory.}",
        "\\label{tab:information_comparison}", "\\small",
        "\\begin{tabularx}{\\textwidth}{Xrrrrrr}", "\\toprule",
        "Policy inputs & CAGR & Volatility & Sharpe & Max DD & CRRA CE & Turnover \\\\ ",
        "\\midrule",
    ]
    for arm in (HEADLINE, *ARMS):
        row = by_arm.loc[arm]
        label = "\\textbf{" + LABELS[arm] + "}" if arm == HEADLINE else LABELS[arm]
        lines.append(
            f"{label} & {100*row['cagr']:.2f}\\% & "
            f"{100*row['annual_volatility']:.2f}\\% & "
            f"{row['sharpe_ratio']:.3f} & "
            f"{100*row['max_drawdown']:.2f}\\% & "
            f"{100*row['annualized_certainty_equivalent_return']:.2f}\\% & "
            f"{row['mean_monthly_turnover']:.3f} \\\\ ")
    lines.extend(["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""])
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve()
    require(not args.output.exists(), "Headline-state results already exist.")
    design, base, digest = load_design(repo, args.contract)
    verify_release(repo, args.release, args.jobs, design, base, digest)
    audit_manifest_path = args.audit / "headline_state_audit_manifest.json"
    replay_manifest_path = args.weights / "headline_state_replay_manifest.json"
    new_manifest = args.weights / "headline_state_weight_manifest.csv"
    audit = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_manifest_path.read_text(encoding="utf-8"))
    require(audit["status"] == "headline_state_checkpoint_audit_passed" and
            audit["checkpoint_audit_sha256"] == sha256(
                args.audit / "headline_state_checkpoint_audit.csv") and
            replay["status"] == "headline_state_weight_replay_complete" and
            replay["weight_manifest_sha256"] == sha256(new_manifest) and
            replay["contract_sha256"] == digest,
            "New checkpoints or weight logs lack an exact passed audit.")
    require(sha256(args.realized) ==
            design["evaluation"]["realized_panel_sha256"],
            "The realized panel differs from Tables 4 and 5.")
    evaluation = Contract.read(args.evaluation_contract)
    require(evaluation.get("turnover_convention", "target_to_target_v1") ==
            "target_to_target_v1" and
            evaluation.get("financing_proration", "fixed_period_v1") ==
            "fixed_period_v1", "Economic accounting differs from the headline.")
    realized, assets = read_realized_panel(args.realized, evaluation)
    weights = source_weights(repo, design, new_manifest, realized, assets,
                             evaluation)
    scored, ensembles = score_four(weights, realized, assets, evaluation)
    complete = scored[scored["is_complete_period"].astype(bool)].copy()
    require(len(complete) == 44 * 22 and
            complete.groupby("strategy_id").size().eq(22).all(),
            "The common complete-period sample differs from Tables 4 and 5.")
    causal = json.loads(args.causal_contract.read_text(encoding="utf-8"))
    metric_rows = []
    for arm in (HEADLINE, *ARMS):
        group = complete[(complete["arm_id"] == arm) &
                         (complete["level"] == "ensemble")]
        require(len(group) == 22, f"Incomplete ensemble: {arm}")
        metric_rows.append({"arm_id": arm, "label": LABELS[arm],
                            **metrics(group, causal)})
    metric_frame = pd.DataFrame(metric_rows)
    reference = reference_check(repo, design, metric_rows[0])
    headline = complete[(complete["arm_id"] == HEADLINE) &
                        (complete["level"] == "ensemble")]
    contrasts = []
    for index, arm in enumerate(ARMS):
        alternative = complete[(complete["arm_id"] == arm) &
                               (complete["level"] == "ensemble")]
        contrasts.append({"candidate": HEADLINE, "alternative": arm,
                          **inference(headline, alternative, causal,
                                      design["inference"]["seed"] + 1009 * index)})
    for row, corrected in zip(contrasts, holm_adjust(
            [row["two_sided_p"] for row in contrasts])):
        row["holm_two_sided_p"] = corrected
        row["confirmatory_claim_permitted"] = False
    seed_rows = []
    for arm in ARMS:
        for seed in design["seeds"]:
            base_group = complete[(complete["arm_id"] == HEADLINE) &
                                  (complete["level"] == "seed") &
                                  (complete["seed"] == seed)]
            alt_group = complete[(complete["arm_id"] == arm) &
                                 (complete["level"] == "seed") &
                                 (complete["seed"] == seed)]
            require(len(base_group) == len(alt_group) == 22,
                    f"Matched seed pair missing: {arm}/{seed}")
            a, b = metrics(base_group, causal), metrics(alt_group, causal)
            seed_rows.append({"alternative": arm, "seed": seed,
                              "ce_headline_minus_alternative": (
                                  a["annualized_certainty_equivalent_return"] -
                                  b["annualized_certainty_equivalent_return"]),
                              "turnover_headline_minus_alternative": (
                                  a["mean_monthly_turnover"] -
                                  b["mean_monthly_turnover"])})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".headline-state-results.",
                                      dir=args.output.parent))
    try:
        tables = temporary / "tables"
        tables.mkdir()
        metric_frame.to_csv(tables / "headline_state_metrics.csv", index=False)
        pd.DataFrame(contrasts).to_csv(
            tables / "headline_state_contrasts.csv", index=False)
        pd.DataFrame(seed_rows).to_csv(
            tables / "headline_state_matched_seed_effects.csv", index=False)
        (tables / "table_08_matched_headline_inputs.tex").write_text(
            latex_table(metric_frame), encoding="utf-8")
        pd.DataFrame([{"metric": name, **values} for name, values in
                      reference.items()]).to_csv(
            tables / "headline_table_4_5_reconciliation.csv", index=False)
        complete.to_csv(temporary / "headline_state_scored_periods.csv", index=False)
        for arm, frame in ensembles.items():
            frame.to_csv(temporary / f"weights_{arm}_ensemble.csv", index=False)
        result = {
            "status": "matched_headline_state_ablation_complete",
            "schema_version": 1, "evidence_class": "post_holdout_explanatory",
            "confirmatory_claim_permitted": False,
            "arm_count": 4, "new_policy_count": 30,
            "reused_frozen_headline_policy_count": 10,
            "common_complete_periods": 22, "locked_periods": 24,
            "turnover_convention": "target_to_target_v1",
            "headline_table_4_5_reconciliation": reference,
            "contract_sha256": digest,
            "original_training_release_manifest_sha256": sha256(
                repo / design["original_training_release"] /
                "mixed_pretraining_release_manifest.json"),
            "new_weight_manifest_sha256": sha256(new_manifest),
            "realized_panel_sha256": sha256(args.realized),
            "scientific_note": "Same consumed holdout; post-holdout explanatory input comparison, not a new confirmatory test.",
        }
        (temporary / "headline_state_analysis_manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        (temporary / "CONTENTS.sha256").write_text("".join(
            f"{sha256(path)}  {path.relative_to(temporary).as_posix()}\n"
            for path in files), encoding="ascii")
        os.replace(temporary, args.output)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path, default=Path(
        "publication_pipeline_draft/config/headline_state_ablation_v1.json"))
    parser.add_argument("--release", type=Path, default=Path(
        "frozen_releases/headline_state_ablation_v1"))
    parser.add_argument("--jobs", type=Path, default=Path(
        "protocol_manifests/headline_state_jobs_v1.csv"))
    parser.add_argument("--audit", type=Path, default=Path(
        "analysis_outputs/headline_state_ablation_v1_audit"))
    parser.add_argument("--weights", type=Path, default=Path(
        "analysis_outputs/headline_state_ablation_v1_weights"))
    parser.add_argument("--realized", type=Path, default=Path(
        "locked_evaluation/main_oos_v4_operational_retry/inputs/realized_asset_gross.csv"))
    parser.add_argument("--evaluation-contract", type=Path, default=Path(
        "publication_pipeline_draft/config/evaluation_contract.json"))
    parser.add_argument("--causal-contract", type=Path, default=Path(
        "publication_pipeline_draft/config/causal_analysis_contract_v2.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "analysis_outputs/headline_state_ablation_v1_results"))
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    for name in ("contract", "release", "jobs", "audit", "weights", "realized",
                 "evaluation_contract", "causal_contract", "output"):
        setattr(args, name, (args.repo_root / getattr(args, name)).resolve())
    try:
        result = analyze(args)
    except (DoseProtocolError, OSError, ValueError, KeyError, IndexError,
            json.JSONDecodeError) as error:
        print(f"HEADLINE STATE ANALYSIS FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
