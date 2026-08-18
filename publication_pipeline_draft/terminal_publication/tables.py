from __future__ import annotations

import statistics
from typing import Any

from .common import (
    CONTRAST_LABELS, EVIDENCE_LABELS, STRATEGY_LABELS,
    PublicationContext, finite, format_p, require, tex, unique_row)


PRIMARY_ORDER = (
    "equal_weight", "shrinkage_mean_variance", "dcc_garch", "static_vine",
    "rolling_vine", "dynamic_nn_vine", "vine_td3_ensemble",
)

KEY_CONTRASTS = (
    "primary_vs_equal_weight",
    "primary_vs_static_vine",
    "primary_vs_dynamic_nn_vine",
    "raw_vine_state_contribution",
    "focused_joint_visible_dependence_contribution",
    "historical_vs_original_bootstrap",
    "concentrated_synthetic_vs_historical_masked",
    "concentrated_synthetic_vs_bootstrap_masked",
)

PRETRAINING = (
    (
        "masked_pretraining_controls",
        "masked_historical_prefix_1000_presentations_ensemble",
        "masked_historical_prefix_1000_presentations__seed_",
    ),
    (
        "synthetic_presentations",
        "synthetic_100_unique_1000_presentations_no_policy_visible_dependence_ensemble",
        "synthetic_100_unique_1000_presentations_no_policy_visible_dependence__seed_",
    ),
    (
        "masked_pretraining_controls",
        "masked_moving_block_bootstrap_1000_presentations_ensemble",
        "masked_moving_block_bootstrap_1000_presentations__seed_",
    ),
)


def _primary_rows(context: PublicationContext) -> list[dict[str, Any]]:
    economic = context.rows("primary_economic_metrics.csv")
    daily = context.rows("daily_tail_risk_metrics.csv")
    result: list[dict[str, Any]] = []
    for strategy in PRIMARY_ORDER:
        common = dict(scope="complete_periods", source_id="frozen_primary_oos",
                      strategy_id=strategy, window_id="locked_oos_v1")
        monthly = unique_row(economic, **common)
        path = unique_row(daily, **common)
        result.append({
            "strategy_id": strategy,
            "strategy": STRATEGY_LABELS[strategy].replace("--", "-"),
            "cagr_percent": 100 * finite(monthly["cagr"]),
            "monthly_volatility_percent": 100 * finite(monthly["annualized_volatility"]),
            "daily_volatility_percent": 100 * finite(path["annualized_daily_volatility"]),
            "monthly_max_drawdown_percent": 100 * finite(monthly["max_drawdown"]),
            "daily_max_drawdown_percent": 100 * finite(path["daily_path_max_drawdown"]),
            "daily_cvar_95_percent": 100 * finite(path["daily_cvar_95_loss"]),
            "annual_crra_ce_percent": 100 * finite(monthly["annualized_certainty_equivalent_return"]),
            "daily_observations": int(path["daily_observations"]),
        })
    return result


def write_primary(context: PublicationContext) -> None:
    inputs = [context.input("primary_economic_metrics.csv"),
              context.input("daily_tail_risk_metrics.csv")]
    rows = _primary_rows(context)
    context.write_csv(
        "tables/table_r01_final_primary_performance_daily_risk.csv", rows,
        artifact_type="publication_table_csv",
        title="Final primary performance and daily risk",
        evidence_class="frozen_primary_evaluation", inputs=inputs)
    lines = []
    for row in rows:
        label = tex(row["strategy"])
        if row["strategy_id"] == "vine_td3_ensemble":
            label = rf"\textbf{{{label}}}"
        lines.append(
            f"{label} & {row['cagr_percent']:.2f} & "
            f"{row['monthly_volatility_percent']:.2f} & {row['daily_volatility_percent']:.2f} & "
            f"{row['monthly_max_drawdown_percent']:.2f} & {row['daily_max_drawdown_percent']:.2f} & "
            f"{row['daily_cvar_95_percent']:.2f} & {row['annual_crra_ce_percent']:.2f} "
            + r"\\")
    body = r"""% Requires booktabs and graphicx.
\begin{table}[tbp]
\centering
\caption{Frozen Primary Performance with Intramonth Risk Reconstruction}
\label{tab:terminal-primary-daily-risk}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrr}
\toprule
Strategy & CAGR & Monthly vol. & Daily vol. & Monthly DD & Daily DD & Daily CVaR$_{95}$ & CRRA CE \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}%
}
\caption*{\footnotesize Notes: All entries are percentages and use the 22 complete locked periods. Daily metrics reconstruct the exact within-period mark-to-market path. CVaR$_{95}$ is descriptive and is based on 22 tail days from 431 observations. Rankings do not establish superiority.}
\end{table}
"""
    context.write_text(
        "tables/table_r01_final_primary_performance_daily_risk.tex", body,
        artifact_type="publication_table_tex",
        title="Final primary performance and daily risk",
        evidence_class="frozen_primary_evaluation", inputs=inputs)


def _contrast_rows(context: PublicationContext) -> list[dict[str, Any]]:
    source = context.rows("registered_contrast_robustness_summary.csv")
    result = []
    for row in source:
        result.append({
            "contrast_id": row["contrast_id"],
            "contrast": CONTRAST_LABELS[row["contrast_id"]],
            "evidence_class": EVIDENCE_LABELS[row["family"]],
            "annual_crra_ce_effect_pp": 100 * finite(row["annualized_ce_difference"]),
            "moving_block_3_ci_lower_pp": 100 * finite(row["registered_moving_block_3_ci_lower"]),
            "moving_block_3_ci_upper_pp": 100 * finite(row["registered_moving_block_3_ci_upper"]),
            "holm_p_positive_direction": finite(row["registered_moving_block_3_holm_p"]),
            "specifications_positive_holm_0_05": int(row["specifications_positive_holm_0_05"]),
            "resampling_specifications": int(row["resampling_specifications"]),
            "leave_one_out_fraction_positive": finite(row["leave_one_out_fraction_positive"]),
            "break_even_transaction_cost_bps": row["break_even_transaction_cost_bps"],
            "break_even_status": row["break_even_status"],
        })
    return result


def _contrast_tex(rows: list[dict[str, Any]], caption: str, label: str) -> str:
    lines = []
    for row in rows:
        lines.append(
            f"{tex(row['contrast'])} & {tex(row['evidence_class'])} & "
            f"{row['annual_crra_ce_effect_pp']:+.2f} & "
            f"[{row['moving_block_3_ci_lower_pp']:.2f}, {row['moving_block_3_ci_upper_pp']:.2f}] & "
            f"{format_p(row['holm_p_positive_direction'])} & "
            f"{100*row['leave_one_out_fraction_positive']:.0f}\\% " + r"\\")
    return r"""% Requires booktabs and graphicx.
\begin{table}[tbp]
\centering
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrr}
\toprule
Contrast (candidate minus comparator) & Evidence class & CE effect (pp) & MBB(3) 95\% CI & Holm $p_{+}$ & LOO positive \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}%
}
\caption*{\footnotesize Notes: $p_{+}$ tests the preregistered positive-effect alternative. Consequently, robust adverse-direction effects have $p_{+}$ near one. LOO reports the fraction of leave-one-period-out effects above zero. Frozen, post-holdout, and retrospective evidence must not be pooled into a confirmatory claim.}
\end{table}
"""


def write_contrasts(context: PublicationContext) -> None:
    input_path = context.input("registered_contrast_robustness_summary.csv")
    rows = _contrast_rows(context)
    context.write_csv(
        "tables/table_a01_full_registered_contrasts.csv", rows,
        artifact_type="publication_table_csv",
        title="Full registered terminal robustness contrasts",
        evidence_class="mixed_evidence_classes", inputs=[input_path])
    context.write_text(
        "tables/table_a01_full_registered_contrasts.tex",
        _contrast_tex(rows, "Full Registered Terminal-Robustness Contrasts",
                      "tab:terminal-full-contrasts"),
        artifact_type="publication_table_tex",
        title="Full registered terminal robustness contrasts",
        evidence_class="mixed_evidence_classes", inputs=[input_path])
    by_id = {row["contrast_id"]: row for row in rows}
    key_rows = [by_id[item] for item in KEY_CONTRASTS]
    context.write_csv(
        "tables/table_r02_key_registered_contrasts.csv", key_rows,
        artifact_type="publication_table_csv",
        title="Key registered terminal robustness contrasts",
        evidence_class="mixed_evidence_classes", inputs=[input_path])
    context.write_text(
        "tables/table_r02_key_registered_contrasts.tex",
        _contrast_tex(key_rows, "Key Registered Effects Across Evidence Layers",
                      "tab:terminal-key-contrasts"),
        artifact_type="publication_table_tex",
        title="Key registered terminal robustness contrasts",
        evidence_class="mixed_evidence_classes", inputs=[input_path])


def _pretraining_rows(context: PublicationContext) -> list[dict[str, Any]]:
    economic = context.rows("primary_economic_metrics.csv")
    daily = context.rows("daily_tail_risk_metrics.csv")
    result = []
    for source_id, ensemble, seed_prefix in PRETRAINING:
        monthly = unique_row(
            economic, scope="complete_periods", source_id=source_id,
            strategy_id=ensemble, window_id="locked_oos_v1")
        path = unique_row(
            daily, scope="complete_periods", source_id=source_id,
            strategy_id=ensemble, window_id="locked_oos_v1")
        seed_values = [
            100 * finite(row["annualized_certainty_equivalent_return"])
            for row in economic
            if row["scope"] == "complete_periods"
            and row["source_id"] == source_id
            and row["strategy_id"].startswith(seed_prefix)
        ]
        require(len(seed_values) == 10,
                f"Expected ten seed metrics for {ensemble}, found {len(seed_values)}")
        result.append({
            "strategy_id": ensemble,
            "pretraining_source": STRATEGY_LABELS[ensemble],
            "ensemble_crra_ce_percent": 100 * finite(monthly["annualized_certainty_equivalent_return"]),
            "daily_volatility_percent": 100 * finite(path["annualized_daily_volatility"]),
            "daily_max_drawdown_percent": 100 * finite(path["daily_path_max_drawdown"]),
            "daily_cvar_95_percent": 100 * finite(path["daily_cvar_95_loss"]),
            "seed_ce_standard_deviation_pp": statistics.stdev(seed_values),
            "seed_ce_minimum_percent": min(seed_values),
            "seed_ce_median_percent": statistics.median(seed_values),
            "seed_ce_maximum_percent": max(seed_values),
        })
    return result


def write_pretraining(context: PublicationContext) -> None:
    inputs = [context.input("primary_economic_metrics.csv"),
              context.input("daily_tail_risk_metrics.csv")]
    rows = _pretraining_rows(context)
    context.write_csv(
        "tables/table_r03_pretraining_source_tradeoff.csv", rows,
        artifact_type="publication_table_csv",
        title="Matched pretraining source trade-off",
        evidence_class="post_holdout_explanatory", inputs=inputs)
    lines = [
        f"{tex(row['pretraining_source'])} & {row['ensemble_crra_ce_percent']:.2f} & "
        f"{row['daily_volatility_percent']:.2f} & {row['daily_max_drawdown_percent']:.2f} & "
        f"{row['daily_cvar_95_percent']:.2f} & {row['seed_ce_standard_deviation_pp']:.2f} & "
        f"[{row['seed_ce_minimum_percent']:.2f}, {row['seed_ce_maximum_percent']:.2f}] " + r"\\"
        for row in rows
    ]
    body = r"""% Requires booktabs.
\begin{table}[tbp]
\centering
\caption{Matched Pretraining Sources Under the Masked-State Architecture}
\label{tab:terminal-pretraining-tradeoff}
\small
\begin{tabular}{lrrrrrr}
\toprule
Source & Ensemble CE & Daily vol. & Daily DD & Daily CVaR$_{95}$ & Seed CE SD & Seed CE range \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\caption*{\footnotesize Notes: All entries except the source name are percentage points or percentages. Each source uses ten matched seeds and 1,000 pretraining presentations. These results are post-holdout explanatory.}
\end{table}
"""
    context.write_text(
        "tables/table_r03_pretraining_source_tradeoff.tex", body,
        artifact_type="publication_table_tex",
        title="Matched pretraining source trade-off",
        evidence_class="post_holdout_explanatory", inputs=inputs)


def write_friction(context: PublicationContext) -> None:
    input_path = context.input("break_even_costs.csv")
    source = context.rows("break_even_costs.csv")
    rows = []
    for row in source:
        rows.append({
            "contrast_id": row["contrast_id"],
            "contrast": CONTRAST_LABELS[row["contrast_id"]],
            "ce_effect_at_zero_cost_pp": 100 * finite(row["ce_difference_at_zero_transaction_cost"]),
            "ce_effect_at_500_bps_pp": 100 * finite(row["ce_difference_at_search_limit"]),
            "break_even_transaction_cost_bps": (
                "" if not row["break_even_transaction_cost_bps"]
                else finite(row["break_even_transaction_cost_bps"])),
            "status": row["break_even_status"],
            "crossing_direction": row["crossing_direction"],
        })
    context.write_csv(
        "tables/table_a02_friction_break_even.csv", rows,
        artifact_type="publication_table_csv",
        title="Transaction-cost break-even diagnostics",
        evidence_class="robustness_rescoring", inputs=[input_path])
    lines = []
    for row in rows:
        crossing = (f"{row['break_even_transaction_cost_bps']:.1f}"
                    if isinstance(row["break_even_transaction_cost_bps"], float)
                    else "No crossing")
        lines.append(
            f"{tex(row['contrast'])} & {row['ce_effect_at_zero_cost_pp']:+.2f} & "
            f"{row['ce_effect_at_500_bps_pp']:+.2f} & {crossing} & "
            f"{tex(row['crossing_direction'].replace('_', ' '))} " + r"\\")
    body = r"""% Requires booktabs.
\begin{table}[tbp]
\centering
\caption{Transaction-Cost Break-Even Robustness}
\label{tab:terminal-friction-crossings}
\small
\begin{tabular}{lrrrr}
\toprule
Contrast & CE at 0 bps (pp) & CE at 500 bps (pp) & Crossing (bps) & Direction \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\caption*{\footnotesize Notes: Frozen weights are deterministically rescored; no policy is retrained. Financing rates are fixed at 3\% for shorts and 2\% for cash borrowing. A missing crossing means the sign is unchanged through 500 bps.}
\end{table}
"""
    context.write_text(
        "tables/table_a02_friction_break_even.tex", body,
        artifact_type="publication_table_tex",
        title="Transaction-cost break-even diagnostics",
        evidence_class="robustness_rescoring", inputs=[input_path])


def write_evidence_ledger(context: PublicationContext) -> None:
    input_path = context.input("evidence_ledger.csv")
    source = context.rows("evidence_ledger.csv")
    rows = [{
        "source_id": row["source_id"],
        "evidence_class": row["evidence_class"],
        "claim_scope": row["claim_scope"],
        "strategies": int(row["strategies"]),
        "windows": int(row["windows"]),
        "period_rows": int(row["period_rows"]),
        "complete_period_rows": int(row["complete_period_rows"]),
        "first_decision_date": row["first_decision_date"],
        "last_holding_end_date": row["last_holding_end_date"],
        "input_sha256": row["input_sha256"],
    } for row in source]
    context.write_csv(
        "tables/table_a03_evidence_ledger.csv", rows,
        artifact_type="publication_table_csv",
        title="Evidence-class and sample ledger",
        evidence_class="protocol_provenance", inputs=[input_path])
    lines = [
        f"{tex(row['source_id'])} & {tex(row['evidence_class'])} & "
        f"{row['strategies']} & {row['windows']} & {row['complete_period_rows']} & "
        f"{row['first_decision_date']}--{row['last_holding_end_date']} " + r"\\"
        for row in rows
    ]
    body = r"""% Requires booktabs and graphicx.
\begin{table}[tbp]
\centering
\caption{Evidence Classes and Audited Sample Boundaries}
\label{tab:terminal-evidence-ledger}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrr}
\toprule
Source & Evidence class & Strategies & Windows & Complete rows & Decision-to-holding dates \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}%
}
\caption*{\footnotesize Notes: The frozen primary, post-holdout explanatory, and retrospective evidence layers are kept separate throughout estimation and interpretation. Hashes are available in the accompanying CSV and immutable campaign manifest.}
\end{table}
"""
    context.write_text(
        "tables/table_a03_evidence_ledger.tex", body,
        artifact_type="publication_table_tex",
        title="Evidence-class and sample ledger",
        evidence_class="protocol_provenance", inputs=[input_path])


def generate(context: PublicationContext) -> None:
    write_primary(context)
    write_contrasts(context)
    write_pretraining(context)
    write_friction(context)
    write_evidence_ledger(context)
