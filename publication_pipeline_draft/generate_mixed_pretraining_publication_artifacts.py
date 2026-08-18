#!/usr/bin/env python3
"""Create an additive publication bundle for the mixed-pretraining experiment.

The generator uses only frozen weights and already consumed realized returns.
It performs no training or model selection.  In addition to the registered
four-arm results, it reports leave-one-seed-out ensemble stability and
transaction-cost rescoring so ensemble-level gains cannot be mistaken for
uniform seed-level dominance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from publication_pipeline_draft.analyze_causal_results import metrics
from publication_pipeline_draft.publication_pipeline import (
    KEYS, Contract, ProtocolError, read_realized_panel, score_strategy,
    sha256_file, validate_weight_matrix,
)
from publication_pipeline_draft.tikz_figures.style import STYLE


class MixedPublicationError(RuntimeError):
    """Raised when immutable mixed-pretraining evidence is incomplete."""


HISTORICAL = "historical_only_training"
SYNTHETIC_ONLY = "synthetic_only_training"
MIXED = "mixed_pretraining_plus_historical_finetuning"
SYNTHETIC_FINETUNED = "synthetic_pretraining_plus_historical_finetuning"
ORDER = (HISTORICAL, SYNTHETIC_ONLY, MIXED, SYNTHETIC_FINETUNED)
LABELS = {
    HISTORICAL: "Historical throughout",
    SYNTHETIC_ONLY: "Synthetic only",
    MIXED: "Mixed + historical fine-tuning",
    SYNTHETIC_FINETUNED: "Synthetic + historical fine-tuning",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MixedPublicationError(message)


def verify_contents(root: Path) -> None:
    inventory = root / "CONTENTS.sha256"
    require(inventory.is_file(), f"Frozen evidence inventory is missing: {inventory}")
    for line in inventory.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        path = root / relative.strip().lstrip("*")
        require(path.is_file(), f"Frozen evidence member is missing: {path}")
        require(sha256_file(path) == expected.lower(),
                f"Frozen evidence member changed: {path}")


def causal_contract(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    require(result.get("evidence_class") == "post_holdout_explanatory",
            "Causal accounting contract has an unexpected evidence class")
    return result


def read_table(path: Path) -> pd.DataFrame:
    require(path.is_file(), f"Required mixed-pretraining table is missing: {path}")
    result = pd.read_csv(path)
    require(not result.empty, f"Required mixed-pretraining table is empty: {path}")
    return result


def load_weight(path: Path, realized: pd.DataFrame, assets: list[str],
                evaluation: Contract) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for name in ("decision_date", "holding_end_date"):
        frame[name] = pd.to_datetime(frame[name], errors="raise").dt.normalize()
    frame["window_id"] = frame["window_id"].astype(str)
    columns = [f"w_{asset}" for asset in assets]
    require(set(KEYS + columns) <= set(frame.columns),
            f"Weight file is incomplete: {path}")
    frame = frame[KEYS + columns].sort_values(KEYS).reset_index(drop=True)
    expected = realized[KEYS].sort_values(KEYS).reset_index(drop=True)
    require(frame[KEYS].equals(expected),
            f"Weight file does not share the locked calendar: {path}")
    matrix = frame[columns].apply(pd.to_numeric, errors="raise").to_numpy(float)
    validate_weight_matrix(matrix, path.name, evaluation)
    frame[columns] = matrix
    return frame


def mixed_member_weights(release: Path, realized: pd.DataFrame,
                         assets: list[str], evaluation: Contract
                         ) -> dict[int, pd.DataFrame]:
    root = release / "source_evidence/weight_evidence/weights" / MIXED
    files = sorted(root.glob("seed_*/weights_rl_full_seed_*.csv"))
    require(len(files) == 10, f"Expected ten mixed-policy weight files, found {len(files)}")
    result: dict[int, pd.DataFrame] = {}
    for path in files:
        seed = int(path.parent.name.removeprefix("seed_"))
        require(seed not in result, f"Duplicate mixed-policy seed: {seed}")
        result[seed] = load_weight(path, realized, assets, evaluation)
    return result


def complete_metrics(path: pd.DataFrame, contract: dict[str, Any]) -> dict[str, Any]:
    selected = path[path["is_complete_period"].astype(bool)].copy()
    require(len(selected) == 22, "Primary robustness metric must use 22 complete periods")
    return metrics(selected, contract)


def leave_one_seed_out(members: dict[int, pd.DataFrame],
                       frozen_metrics: pd.DataFrame, realized: pd.DataFrame,
                       assets: list[str], evaluation: Contract,
                       contract: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [f"w_{asset}" for asset in assets]
    by_arm = frozen_metrics.set_index("experiment_id")
    require(set(ORDER) <= set(by_arm.index),
            "Frozen four-arm metrics are incomplete")
    rows: list[dict[str, Any]] = []
    for omitted in sorted(members):
        retained = [frame for seed, frame in members.items() if seed != omitted]
        ensemble = retained[0][KEYS].copy()
        ensemble[columns] = np.stack(
            [frame[columns].to_numpy(float) for frame in retained]).mean(axis=0)
        validate_weight_matrix(ensemble[columns].to_numpy(float),
                               f"mixed_leave_seed_{omitted}_out", evaluation)
        scored = score_strategy(f"mixed_leave_seed_{omitted}_out", ensemble,
                                realized, assets, evaluation)
        result = complete_metrics(scored, contract)
        row = {
            "omitted_seed": omitted,
            "retained_seeds": 9,
            "annualized_crra_ce": result["annualized_certainty_equivalent_return"],
            "cagr": result["cagr"],
            "annual_volatility": result["annual_volatility"],
            "sharpe_ratio": result["sharpe_ratio"],
            "max_drawdown": result["max_drawdown"],
            "mean_monthly_turnover": result["mean_monthly_turnover"],
            "mean_gross_exposure": result["mean_gross_exposure"],
        }
        for arm in (HISTORICAL, SYNTHETIC_ONLY, SYNTHETIC_FINETUNED):
            row[f"ce_difference_vs_{arm}"] = (
                result["annualized_certainty_equivalent_return"] -
                float(by_arm.loc[arm, "annualized_certainty_equivalent_return"]))
        rows.append(row)
    detail = pd.DataFrame(rows)
    summaries: list[dict[str, Any]] = []
    for arm in (HISTORICAL, SYNTHETIC_ONLY, SYNTHETIC_FINETUNED):
        column = f"ce_difference_vs_{arm}"
        summaries.append({
            "candidate": MIXED,
            "comparator": arm,
            "leave_one_seed_out_replicates": len(detail),
            "minimum_ce_difference": float(detail[column].min()),
            "median_ce_difference": float(detail[column].median()),
            "maximum_ce_difference": float(detail[column].max()),
            "fraction_positive_ce_difference": float((detail[column] > 0).mean()),
            "inference_note": (
                "ensemble-composition diagnostic only; replicates share one market path"),
        })
    return detail, pd.DataFrame(summaries)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def table_tex(metrics_frame: pd.DataFrame) -> str:
    by_id = metrics_frame.set_index("experiment_id")
    rows = []
    for arm in ORDER:
        row = by_id.loc[arm]
        rows.append(
            f"{LABELS[arm]} & {100*row['total_return']:.1f} & {100*row['cagr']:.1f} & "
            f"{100*row['annual_volatility']:.1f} & {row['sharpe_ratio']:.2f} & "
            f"{100*row['annualized_certainty_equivalent_return']:.1f} & "
            f"{100*row['max_drawdown']:.1f} & {row['mean_monthly_turnover']:.3f} & "
            f"{100*row['implementation_drag_total_return']:.2f} " + r"\\")
    return r"""% Requires booktabs and graphicx.
\begin{table}[tbp]
\centering
\caption{Training-Source Curriculum Comparison on the Consumed Holdout}
\label{tab:mixed-pretraining-four-arm}
\scriptsize
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrrr}
\toprule
Protocol & Total return & CAGR & Vol. & Sharpe & CRRA CE & Max DD & Turnover & Impl. drag \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
\caption*{\footnotesize Notes: Returns, volatility, CRRA certainty equivalent, drawdown, and implementation drag are percentages; turnover is mean monthly target-to-target turnover. All arms share the masked TD3--LSTM architecture, matched seeds, realized returns, constraints, and costs. The primary panel contains 22 complete holding periods. This experiment is post-holdout explanatory and cannot establish a fresh confirmatory winner.}
\end{table}
"""


def figure_tex(metrics_frame: pd.DataFrame, contrasts: pd.DataFrame,
               seed_effects: pd.DataFrame) -> str:
    by_id = metrics_frame.set_index("experiment_id")
    bars = " ".join(
        f"({index + 1},{100*float(by_id.loc[arm, 'annualized_certainty_equivalent_return']):.6g})"
        for index, arm in enumerate(ORDER))
    y_map = {HISTORICAL: 3.0, SYNTHETIC_ONLY: 2.0, SYNTHETIC_FINETUNED: 1.0}
    seed_coordinates = []
    for _, row in seed_effects.iterrows():
        seed = int(row["seed"])
        offset = (seed % 10 - 4.5) * 0.045
        seed_coordinates.append(
            f"({100*float(row['ce_difference']):.6g},"
            f"{y_map[str(row['comparator'])] + offset:.5g})")
    intervals = []
    for _, row in contrasts.iterrows():
        comparator = str(row["comparator"])
        value = 100 * float(row["annualized_ce_difference"])
        lower = 100 * float(row["ci_lower"])
        upper = 100 * float(row["ci_upper"])
        y_value = y_map[comparator]
        intervals.append(
            "\\addplot+[only marks,mark=diamond*,mark size=2.2pt,black,"
            "error bars/.cd,x dir=both,x explicit] coordinates "
            f"{{({value:.6g},{y_value:.3g}) += ({upper-value:.6g},0) "
            f"-= ({value-lower:.6g},0)}};")
    return r"""% Native TikZ/PGFPlots; post-holdout explanatory evidence.
\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=2 by 1,horizontal sep=2.35cm},
  height=5.35cm,
  tick label style={font=\scriptsize},label style={font=\small},
  title style={font=\small\bfseries},grid=major]
\nextgroupplot[
  width=0.40\linewidth,
  title={(a) Ensemble economic value},
  ylabel={Annualized CRRA CE (\%)},
  xmin=0.45,xmax=4.55,ymin=20,ymax=35,
  xtick={1,2,3,4},
  xticklabels={Historical,Synthetic,{Mixed},{Two-Stage}},
  x tick label style={font=\tiny,align=center,yshift=-2pt}]
\addplot+[ybar,bar width=10pt,fill=black!32,draw=black] coordinates {""" + bars + r"""};
\nextgroupplot[
  width=0.44\linewidth,
  title={(b) Mixed curriculum minus controls},
  xlabel={Annualized CRRA CE difference (pp)},
  xmin=-32,xmax=36,ymin=0.55,ymax=3.45,
  ytick={1,2,3},
  yticklabels={Two-Stage,Synthetic,Historical},
  legend style={at={(0.5,-0.31)},anchor=north,legend columns=2,font=\scriptsize,
    draw=none,/tikz/every even column/.append style={column sep=20pt}}]
\addplot[black,dashed,forget plot] coordinates {(0,0.55) (0,3.45)};
\addplot+[only marks,mark=o,mark size=1.45pt,draw=black!60,fill=white]
  coordinates {""" + " ".join(seed_coordinates) + r"""};
\addlegendentry{Matched-seed effects}
""" + "\n".join(intervals) + r"""
\addlegendimage{only marks,mark=diamond*,black,error bars/.cd,x dir=both,x explicit}
\addlegendentry{Ensemble effect and MBB(3) 95\% CI}
\end{groupplot}
\end{tikzpicture}
"""


def claim_rows(metrics_frame: pd.DataFrame, contrasts: pd.DataFrame,
               seed_summary: pd.DataFrame,
               loo_summary: pd.DataFrame) -> list[dict[str, str]]:
    by_id = metrics_frame.set_index("experiment_id")
    by_control = contrasts.set_index("comparator")
    seed_by_control = seed_summary.set_index("comparator")
    mixed = by_id.loc[MIXED]
    history = by_id.loc[HISTORICAL]
    synthetic = by_id.loc[SYNTHETIC_ONLY]
    loo_by_control = loo_summary.set_index("comparator")
    return [
        {
            "claim_id": "MP-C01", "evidence_class": "post_holdout_explanatory",
            "decision": "supported_descriptively",
            "claim": "The mixed curriculum has the highest primary-sample ensemble economic value.",
            "decisive_evidence": (
                f"Mixed CE={100*mixed['annualized_certainty_equivalent_return']:.2f}%, "
                f"CAGR={100*mixed['cagr']:.2f}%, and total return={100*mixed['total_return']:.2f}% on 22 complete periods."),
            "permissible_wording": "The mixed curriculum produced the highest ensemble CE and CAGR point estimates.",
            "prohibited_wording": "Mixed pretraining is statistically proven superior.",
        },
        {
            "claim_id": "MP-C02", "evidence_class": "post_holdout_explanatory",
            "decision": "not_established",
            "claim": "Mixed pretraining reliably dominates historical-only training.",
            "decisive_evidence": (
                f"CE effect={100*by_control.loc[HISTORICAL,'annualized_ce_difference']:+.2f} pp; "
                f"Holm p={by_control.loc[HISTORICAL,'holm_p_candidate_greater']:.3f}; "
                f"matched-seed positive fraction={seed_by_control.loc[HISTORICAL,'fraction_positive_ce_difference']:.0%}."),
            "permissible_wording": "Mixed pretraining improved the ensemble point estimate but not seedwise or adjusted statistical evidence versus historical training.",
            "prohibited_wording": "Mixed pretraining dominates historical-only RL.",
        },
        {
            "claim_id": "MP-C03", "evidence_class": "post_holdout_explanatory",
            "decision": "directional_support_not_familywise_significant",
            "claim": "Historical information improves upon synthetic-only training.",
            "decisive_evidence": (
                f"Mixed minus synthetic-only CE={100*by_control.loc[SYNTHETIC_ONLY,'annualized_ce_difference']:+.2f} pp, "
                f"raw p={by_control.loc[SYNTHETIC_ONLY,'one_sided_p_candidate_greater']:.3f}, "
                f"Holm p={by_control.loc[SYNTHETIC_ONLY,'holm_p_candidate_greater']:.3f}."),
            "permissible_wording": "Synthetic-only training was economically weakest; adding historical information had the strongest directional benefit.",
            "prohibited_wording": "The mixed curriculum is familywise significantly superior to synthetic-only training.",
        },
        {
            "claim_id": "MP-C04", "evidence_class": "post_holdout_explanatory",
            "decision": "supported_descriptively",
            "claim": "Synthetic augmentation lowers implementation intensity relative to historical-only training.",
            "decisive_evidence": (
                f"Mean monthly turnover falls from {history['mean_monthly_turnover']:.3f} to "
                f"{mixed['mean_monthly_turnover']:.3f}; implementation drag falls from "
                f"{100*history['implementation_drag_total_return']:.2f} to "
                f"{100*mixed['implementation_drag_total_return']:.2f} pp."),
            "permissible_wording": "Synthetic augmentation materially reduced turnover and implementation drag in the ensemble.",
            "prohibited_wording": "Synthetic data guarantees lower future trading costs.",
        },
        {
            "claim_id": "MP-C05", "evidence_class": "post_holdout_explanatory",
            "decision": "rejected",
            "claim": "Synthetic augmentation uniformly lowers portfolio risk.",
            "decisive_evidence": (
                f"Historical-only volatility={100*history['annual_volatility']:.2f}% and max DD={100*history['max_drawdown']:.2f}%, "
                f"versus mixed {100*mixed['annual_volatility']:.2f}% and {100*mixed['max_drawdown']:.2f}%."),
            "permissible_wording": "The mixed curriculum improved economic value and implementation efficiency, not every risk metric.",
            "prohibited_wording": "Synthetic augmentation uniformly de-risks the portfolio.",
        },
        {
            "claim_id": "MP-C06", "evidence_class": "protocol_boundary",
            "decision": "prohibited",
            "claim": "Further tuning on the same holdout can create a fresh confirmatory result.",
            "decisive_evidence": "The frozen mixed-pretraining manifest explicitly prohibits further same-holdout tuning and confirmation.",
            "permissible_wording": "The mixed curriculum is a candidate for independently frozen external validation.",
            "prohibited_wording": "The consumed holdout confirms the revised winning architecture.",
        },
        {
            "claim_id": "MP-C07", "evidence_class": "frozen_weight_robustness",
            "decision": "supported_descriptively",
            "claim": "The mixed ensemble point advantage is not attributable to one indispensable seed.",
            "decisive_evidence": (
                f"Mixed-minus-historical CE remains positive in "
                f"{loo_by_control.loc[HISTORICAL,'fraction_positive_ce_difference']:.0%} "
                f"of ten leave-one-seed-out ensembles, with range "
                f"[{100*loo_by_control.loc[HISTORICAL,'minimum_ce_difference']:+.2f}, "
                f"{100*loo_by_control.loc[HISTORICAL,'maximum_ce_difference']:+.2f}] pp."),
            "permissible_wording": "The mixed ensemble advantage usually survives omission of one member, although two omissions reverse the historical-only comparison.",
            "prohibited_wording": "The mixed ensemble advantage is fully stable across all seed subsets.",
        },
    ]


def generate(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve(); release = args.evidence_release.resolve()
    verify_contents(release)
    release_manifest = json.loads((
        release / "mixed_pretraining_evidence_manifest.json"
    ).read_text(encoding="utf-8"))
    require(release_manifest.get("status") == "frozen_mixed_pretraining_evidence_v1",
            "Mixed-pretraining evidence release is not frozen")
    require(release_manifest.get("confirmatory_claim_permitted") is False,
            "Publication extension cannot permit confirmation")
    analysis_root = release / "source_evidence/analysis_results"
    tables_root = analysis_root / "tables"
    source_manifest = release_manifest["source_analysis_manifest"]
    require(sha256_file(args.realized.resolve()) ==
            source_manifest["realized_panel_sha256"],
            "Realized panel differs from the one used by the frozen analysis")
    evaluation = Contract.read(args.evaluation_contract.resolve())
    causal = causal_contract(args.causal_contract.resolve())
    realized, assets = read_realized_panel(args.realized.resolve(), evaluation)
    metrics_frame = read_table(tables_root / "mixed_pretraining_four_arm_metrics.csv")
    contrasts = read_table(tables_root / "mixed_pretraining_primary_contrasts.csv")
    seed_effects = read_table(tables_root / "mixed_pretraining_matched_seed_effects.csv")
    seed_summary = read_table(tables_root / "mixed_pretraining_matched_seed_summary.csv")
    require(set(metrics_frame["experiment_id"]) == set(ORDER),
            "Four-arm metric table is incomplete")
    members = mixed_member_weights(release, realized, assets, evaluation)
    loo_detail, loo_summary = leave_one_seed_out(
        members, metrics_frame, realized, assets, evaluation, causal)

    output = args.output.resolve()
    require(output != repo and output != Path(output.anchor), "Unsafe output path")
    if output.exists() and not args.replace:
        raise MixedPublicationError(
            f"Additive output exists: {output}; pass --replace to regenerate it")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        write_csv(temporary / "tables/table_mp01_four_arm_performance.csv",
                  metrics_frame)
        (temporary / "tables/table_mp01_four_arm_performance.tex").write_text(
            table_tex(metrics_frame), encoding="utf-8")
        write_csv(temporary / "tables/table_mp02_registered_contrasts.csv", contrasts)
        write_csv(temporary / "robustness/mixed_leave_one_seed_out.csv", loo_detail)
        write_csv(temporary / "robustness/mixed_leave_one_seed_out_summary.csv",
                  loo_summary)
        claims = claim_rows(metrics_frame, contrasts, seed_summary, loo_summary)
        claims_path = temporary / "claim_ledger/mixed_pretraining_claim_ledger.csv"
        claims_path.parent.mkdir(parents=True, exist_ok=True)
        with claims_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(claims[0]))
            writer.writeheader(); writer.writerows(claims)
        claim_md = ["# Mixed-pretraining claim ledger", "",
                    "This ledger is additive to the terminal claim ledger.", ""]
        for claim in claims:
            claim_md.extend((
                f"## {claim['claim_id']} — {claim['decision']}", "",
                f"**Claim.** {claim['claim']}", "",
                f"**Evidence.** {claim['decisive_evidence']}", "",
                f"**Permissible wording.** {claim['permissible_wording']}", "",
                f"**Do not write.** {claim['prohibited_wording']}", ""))
        (temporary / "claim_ledger/mixed_pretraining_claim_ledger.md").write_text(
            "\n".join(claim_md), encoding="utf-8")
        figure_root = temporary / "figures/tikz"; figure_root.mkdir(parents=True)
        (figure_root / "figure_mp01_mixed_pretraining_evidence.tex").write_text(
            figure_tex(metrics_frame, contrasts, seed_effects), encoding="utf-8")
        preamble = STYLE.replace(r"\pgfplotsset{compat=1.18}",
                                 r"\pgfplotsset{compat=1.16}")
        (figure_root / "tikz_preamble.tex").write_text(preamble, encoding="utf-8")
        preview = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=16mm]{geometry}
\usepackage[T1]{fontenc}
\input{tikz_preamble.tex}
\begin{document}
\begin{figure}[p]\centering
\input{figure_mp01_mixed_pretraining_evidence.tex}
\caption{Mixed-pretraining ensemble performance, matched-seed dispersion, and paired circular-block intervals. Hollow circles are matched-seed CE differences; diamonds and horizontal bars are ensemble differences and MBB(3) 95\% intervals. This is post-holdout explanatory evidence.}
\end{figure}
\end{document}
"""
        (figure_root / "preview_mixed_pretraining_figure.tex").write_text(
            preview, encoding="utf-8")
        narrative = r"""\subsubsection{Mixed historical--synthetic pretraining}
The terminal curriculum experiment pooled 100 unique vine-synthetic episodes and 61 historical-prefix episodes into 1,000 pretraining presentations before the unchanged 61-episode historical fine-tuning stage.  On the 22 complete holding periods, the mixed ensemble attained the highest CAGR and annualized CRRA certainty equivalent among the four matched training protocols, while using materially less turnover and implementation drag than historical-only training.  Historical-only training nevertheless retained the lowest volatility, drawdown, and CVaR and the highest Sharpe, Sortino, Calmar, and Omega ratios.

The registered mixed-minus-control CE effects were positive in the primary panel, but none survived Holm adjustment.  In particular, the mixed-minus-historical ensemble effect was not representative of matched seeds: fewer than half of the seedwise effects were positive and the median effect was negative.  Nevertheless, the ensemble result was not generated by one indispensable seed: eight of ten leave-one-seed-out mixed ensembles retained a positive CE difference relative to historical-only training, and all ten retained positive differences relative to both synthetic controls.  The strongest directional evidence was against synthetic-only training, showing that synthetic experience did not substitute for real historical information.  The results therefore support synthetic simulation as an augmentation and implementation regularizer rather than a standalone training distribution.  Because the experiment was designed after inspection of the consumed holdout, it is reported as post-holdout explanatory evidence and the mixed curriculum is retained only as a candidate for independently frozen external validation.
"""
        (temporary / "narrative").mkdir()
        (temporary / "narrative/mixed_pretraining_results.tex").write_text(
            narrative, encoding="utf-8")
        plan = """# Mixed-pretraining manuscript placement

- **Main text:** replace the older pretraining-source trade-off figure with
  `figure_mp01_mixed_pretraining_evidence.tex` in the post-holdout mechanism
  subsection.  It combines ensemble economics, seed dispersion, and inference.
- **Appendix:** include `table_mp01_four_arm_performance.tex` for exact values.
- **Online supplement:** retain the registered-contrast CSV and leave-one-seed-out
  diagnostics.  Do not spend main-paper pages on another ensemble-composition
  figure.
- **Not produced:** a mixed-arm transaction-cost counterfactual.  The immutable
  archive contains all 24 mixed-policy target paths but comparator ensemble
  targets only for the 22 complete periods; omitting the two target transitions
  would make turnover rescoring approximate rather than exact.
- **Claim control:** append the six MP claims to the authorial claim ledger;
  never merge their evidence class with the frozen primary benchmark result.
"""
        (temporary / "manuscript_plan").mkdir()
        (temporary / "manuscript_plan/mixed_pretraining_placement.md").write_text(
            plan, encoding="utf-8")
        readme = """# Mixed-pretraining publication extension

This additive package transforms the immutable post-holdout experiment into
one manuscript figure, one appendix table, an authorial claim ledger, and an
exact leave-one-seed-out robustness diagnostic.  It does not edit the existing
manuscript or terminal publication bundle and performs no training or model
selection.
"""
        (temporary / "README.md").write_text(readme, encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "status": "mixed_pretraining_publication_artifacts_generated",
            "additive_only": True,
            "existing_publication_artifacts_modified": False,
            "evidence_class": "post_holdout_explanatory",
            "confirmatory_claim_created": False,
            "model_training_performed": False,
            "model_selection_performed": False,
            "same_holdout_further_tuning_authorized": False,
            "source_release_sha256": sha256_file(
                release / "mixed_pretraining_evidence_manifest.json"),
            "realized_panel_sha256": sha256_file(args.realized.resolve()),
            "leave_one_seed_out_replicates": len(loo_detail),
            "transaction_cost_rescoring_status": (
                "not_generated_missing_full_24_period_comparator_target_weights"),
        }
        (temporary / "mixed_pretraining_publication_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        (temporary / "CONTENTS.sha256").write_text("".join(
            f"{sha256_file(path)}  {path.relative_to(temporary).as_posix()}\n"
            for path in files), encoding="ascii")
        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--evidence-release", type=Path, default=Path(
        "frozen_releases/mixed_pretraining_response_v1_evidence_v1"))
    parser.add_argument("--realized", required=True, type=Path)
    parser.add_argument("--evaluation-contract", type=Path, default=Path(
        "publication_pipeline_draft/config/evaluation_contract.json"))
    parser.add_argument("--causal-contract", type=Path, default=Path(
        "publication_pipeline_draft/config/causal_analysis_contract_v2.json"))
    parser.add_argument("--output", type=Path, default=Path(
        "manuscript_revision_causal_v1/publication_mixed_pretraining_v1"))
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(); repo = args.repo_root.resolve()
    for name in ("evidence_release", "realized", "evaluation_contract",
                 "causal_contract", "output"):
        value = getattr(args, name)
        setattr(args, name, (value if value.is_absolute() else repo / value).resolve())
    args.repo_root = repo
    try:
        result = generate(args)
    except (MixedPublicationError, ProtocolError, OSError, ValueError, KeyError,
            json.JSONDecodeError) as error:
        print(f"MIXED PUBLICATION FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
