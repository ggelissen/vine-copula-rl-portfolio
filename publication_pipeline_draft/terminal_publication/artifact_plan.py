from __future__ import annotations

from typing import Any

from .common import PublicationContext


PLAN: list[dict[str, Any]] = [
    {"order": 1, "artifact": "figure_m01_lstm_td3_architecture.tex", "origin": "existing", "decision": "main_text", "section": "Methodology: policy architecture", "role": "One compact architecture figure", "duplication_rule": "Keep; do not also include the full training diagram here"},
    {"order": 2, "artifact": "figure_m02_training_strategy.tex", "origin": "existing", "decision": "appendix", "section": "Appendix: training protocol", "role": "Detailed leakage-safe training flow", "duplication_rule": "Move out of main text"},
    {"order": 3, "artifact": "table_r01_final_primary_performance_daily_risk.tex", "origin": "new_terminal", "decision": "main_text", "section": "Results: frozen primary evaluation", "role": "Definitive primary economic and daily-risk table", "duplication_rule": "Supersedes the old primary-performance table"},
    {"order": 4, "artifact": "figure_01_wealth_drawdown.tex", "origin": "existing", "decision": "main_text", "section": "Results: frozen primary evaluation", "role": "Economic path and drawdown", "duplication_rule": "Keep as the only primary wealth figure"},
    {"order": 5, "artifact": "figure_r02_intramonth_risk.tex", "origin": "new_terminal", "decision": "main_text", "section": "Results: daily downside risk", "role": "Shows risk hidden by monthly endpoints", "duplication_rule": "Do not retain the old risk-return scatter in main text"},
    {"order": 6, "artifact": "figure_r01_terminal_contrast_forest.tex", "origin": "new_terminal", "decision": "main_text", "section": "Results: inference and mechanism synthesis", "role": "Single cross-evidence effect summary", "duplication_rule": "Supersedes figures 06 and 11 in the main text"},
    {"order": 7, "artifact": "figure_16_focused_walk_forward.tex", "origin": "existing", "decision": "main_text", "section": "Results: retrospective robustness", "role": "Two nonoverlapping focused windows", "duplication_rule": "Keep; label retrospective"},
    {"order": 8, "artifact": "figure_r04_pretraining_tradeoff.tex", "origin": "new_terminal", "decision": "main_text", "section": "Results: pretraining-source controls", "role": "Performance, seed stability, and tail-risk trade-off", "duplication_rule": "Use instead of separate dose/presentation plots"},
    {"order": 9, "artifact": "table_r02_key_registered_contrasts.tex", "origin": "new_terminal", "decision": "omit", "section": "Author convenience only", "role": "Compact numerical counterpart to the terminal forest", "duplication_rule": "The full appendix table already contains these rows"},
    {"order": 10, "artifact": "table_r03_pretraining_source_tradeoff.tex", "origin": "new_terminal", "decision": "appendix", "section": "Appendix: pretraining controls", "role": "Exact values underlying the trade-off plot", "duplication_rule": "Do not place beside the same figure in main text"},
    {"order": 11, "artifact": "figure_r03_friction_surface.tex", "origin": "new_terminal", "decision": "appendix", "section": "Appendix: implementation robustness", "role": "Frozen-weight cost sensitivity", "duplication_rule": "Supersedes existing implementation-intensity plot for cost claims"},
    {"order": 12, "artifact": "figure_r05_resampling_stability.tex", "origin": "new_terminal", "decision": "appendix", "section": "Appendix: inferential robustness", "role": "CI sign across all nine block specifications", "duplication_rule": "Keep full numerical contrast table adjacent"},
    {"order": 13, "artifact": "table_a01_full_registered_contrasts.tex", "origin": "new_terminal", "decision": "appendix", "section": "Appendix: inferential robustness", "role": "All eleven registered effects", "duplication_rule": "Full table only; key table may be omitted if page pressure is severe"},
    {"order": 14, "artifact": "table_a02_friction_break_even.tex", "origin": "new_terminal", "decision": "online_supplement", "section": "Supplement: implementation", "role": "Exact crossing diagnostics", "duplication_rule": "Figure R03 is sufficient in the PDF"},
    {"order": 15, "artifact": "table_a03_evidence_ledger.tex", "origin": "new_terminal", "decision": "appendix", "section": "Appendix: protocol provenance", "role": "Evidence-class and sample-boundary audit", "duplication_rule": "One compact provenance table"},
    {"order": 16, "artifact": "figure_s01_marginal_fidelity.tex", "origin": "existing", "decision": "appendix", "section": "Appendix: synthetic-data diagnostics", "role": "Marginal fidelity", "duplication_rule": "Combine S01--S03 across at most two pages"},
    {"order": 17, "artifact": "figure_s02_dependence_fidelity.tex", "origin": "existing", "decision": "appendix", "section": "Appendix: synthetic-data diagnostics", "role": "Dependence fidelity", "duplication_rule": "Combine S01--S03 across at most two pages"},
    {"order": 18, "artifact": "figure_s03_temporal_fidelity.tex", "origin": "existing", "decision": "appendix", "section": "Appendix: synthetic-data diagnostics", "role": "Temporal fidelity", "duplication_rule": "Combine S01--S03 across at most two pages"},
    {"order": 19, "artifact": "figure_05_seed_robustness.tex", "origin": "existing", "decision": "appendix", "section": "Appendix: optimization uncertainty", "role": "Primary 20-seed dispersion", "duplication_rule": "Retain because ensemble-only reporting masks instability"},
    {"order": 20, "artifact": "figure_08_ensemble_cancellation.tex", "origin": "existing", "decision": "appendix", "section": "Appendix: ensemble mechanism", "role": "Exposure and turnover cancellation", "duplication_rule": "One ensemble-mechanism plot is sufficient"},
    {"order": 21, "artifact": "figure_03_allocation_heatmap.tex", "origin": "existing", "decision": "appendix_optional", "section": "Appendix: portfolio composition", "role": "Allocation interpretability", "duplication_rule": "Include only if final PDF remains below 40 pages"},
    {"order": 22, "artifact": "figure_02_risk_return_utility.tex", "origin": "existing", "decision": "omit", "section": "None", "role": "Redundant risk-return scatter", "duplication_rule": "Table R01 and figure R02 now provide stronger evidence"},
    {"order": 23, "artifact": "figure_04_implementation.tex", "origin": "existing", "decision": "omit", "section": "None", "role": "Earlier implementation summary", "duplication_rule": "Superseded by terminal friction analysis"},
    {"order": 24, "artifact": "figure_06_primary_inference.tex", "origin": "existing", "decision": "omit", "section": "None", "role": "Primary-only forest", "duplication_rule": "Superseded by terminal contrast forest"},
    {"order": 25, "artifact": "figure_07_monthly_excess.tex", "origin": "existing", "decision": "omit", "section": "None", "role": "Monthly excess-return detail", "duplication_rule": "Low incremental value per page"},
    {"order": 26, "artifact": "figure_09_ensemble_size.tex", "origin": "existing", "decision": "online_supplement", "section": "Supplement: exploratory ensemble size", "role": "Post-holdout k sensitivity", "duplication_rule": "Do not spend PDF pages on unselected k"},
    {"order": 27, "artifact": "figure_10_drift_accounting.tex", "origin": "existing", "decision": "online_supplement", "section": "Supplement: accounting audit", "role": "Target versus drift accounting", "duplication_rule": "Describe reconciliation in text"},
    {"order": 28, "artifact": "figure_11_causal_forest.tex", "origin": "existing", "decision": "omit", "section": "None", "role": "Earlier causal forest", "duplication_rule": "Superseded by terminal contrast forest"},
    {"order": 29, "artifact": "figure_12_causal_turnover_performance.tex", "origin": "existing", "decision": "online_supplement", "section": "Supplement: causal diagnostics", "role": "Exploratory performance-turnover plane", "duplication_rule": "Not needed for core claims"},
    {"order": 30, "artifact": "figure_13_causal_seed_effects.tex", "origin": "existing", "decision": "omit", "section": "None", "role": "Earlier causal seed effects", "duplication_rule": "Pretraining figure and seed appendix cover instability"},
    {"order": 31, "artifact": "figure_14_causal_wealth.tex", "origin": "existing", "decision": "omit", "section": "None", "role": "Post-holdout causal wealth", "duplication_rule": "Avoid a second same-holdout wealth plot"},
    {"order": 32, "artifact": "figure_15_compressed_benchmark_reconciliation.tex", "origin": "existing", "decision": "online_supplement", "section": "Supplement: benchmark reconciliation", "role": "Operational audit", "duplication_rule": "Report pass status in prose"},
    {"order": 33, "artifact": "figure_t01_pretraining_stability.tex", "origin": "existing", "decision": "online_supplement", "section": "Supplement: training diagnostics", "role": "Training trajectories", "duplication_rule": "Do not confuse in-sample training diagnostics with OOS evidence"},
    {"order": 34, "artifact": "figure_t02_optimizer_diagnostics.tex", "origin": "existing", "decision": "online_supplement", "section": "Supplement: training diagnostics", "role": "Optimizer trajectories", "duplication_rule": "Do not confuse in-sample training diagnostics with OOS evidence"},
]


def generate(context: PublicationContext) -> None:
    context.write_csv(
        "manuscript_plan/manuscript_artifact_plan.csv", PLAN,
        artifact_type="manuscript_inclusion_plan_csv",
        title="Strict manuscript artifact inclusion plan",
        evidence_class="authorial_synthesis")
    lines = [
        "# Manuscript artifact plan", "",
        "The target is a sub-40-page manuscript. Main-text artifacts are restricted "
        "to one methodology diagram, one definitive primary table, and five empirical figures.", "",
        "| Order | Artifact | Decision | Placement | Role |",
        "|---:|---|---|---|---|",
    ]
    for row in PLAN:
        lines.append(
            f"| {row['order']} | `{row['artifact']}` | **{row['decision']}** | "
            f"{row['section']} | {row['role']} |")
    lines.extend((
        "", "## Strict page-budget rule", "",
        "If the compiled manuscript exceeds 40 pages, remove the optional allocation "
        "heatmap first, then move the exact pretraining table to the online supplement. "
        "Do not remove the evidence ledger, terminal forest, focused walk-forward figure, "
        "or the pretraining trade-off figure.",
    ))
    context.write_text(
        "manuscript_plan/manuscript_artifact_plan.md", "\n".join(lines),
        artifact_type="manuscript_inclusion_plan_markdown",
        title="Strict manuscript artifact inclusion plan",
        evidence_class="authorial_synthesis")
