from __future__ import annotations

from typing import Any

from .common import PublicationContext, finite, format_p, unique_row


def generate(context: PublicationContext) -> None:
    contrasts = {row["contrast_id"]: row
                 for row in context.rows("registered_contrast_robustness_summary.csv")}
    reality = context.rows("white_reality_checks.csv")
    primary_reality = unique_row(
        reality, family="frozen_primary_benchmarks", method="moving_block",
        block_length="3")
    terminal_reality = unique_row(
        reality, family="post_holdout_terminal_pretraining_controls",
        method="moving_block", block_length="3")

    def effect(identifier: str) -> float:
        return 100 * finite(contrasts[identifier]["annualized_ce_difference"])

    def interval(identifier: str) -> str:
        row = contrasts[identifier]
        return (f"[{100*finite(row['registered_moving_block_3_ci_lower']):.2f}, "
                f"{100*finite(row['registered_moving_block_3_ci_upper']):.2f}] pp")

    claims: list[dict[str, Any]] = [
        {
            "claim_id": "TR-C01",
            "claim": "The frozen NN-vine TD3 ensemble is economically competitive with strong benchmarks.",
            "evidence_class": "frozen_primary_evaluation",
            "decision": "supported_descriptively",
            "decisive_evidence": "CRRA CE 27.96%; fourth of seven strategies and above equal weight, shrinkage MV, rolling vine, and DCC-GARCH.",
            "permissible_wording": "Competitive risk-adjusted performance with comparatively low volatility and implementation drag.",
            "prohibited_wording": "Best-performing or universally superior portfolio optimizer.",
            "manuscript_location": "Results: frozen primary evaluation",
        },
        {
            "claim_id": "TR-C02",
            "claim": "The frozen TD3 ensemble has a statistically established advantage over the benchmark family.",
            "evidence_class": "frozen_primary_evaluation",
            "decision": "rejected",
            "decisive_evidence": f"White reality-check p={finite(primary_reality['white_reality_check_p']):.3f}; no registered positive contrast is significant.",
            "permissible_wording": "No benchmark-family superiority is established on the short locked path.",
            "prohibited_wording": "Statistically proven benchmark superiority or dominance.",
            "manuscript_location": "Results: paired inference",
        },
        {
            "claim_id": "TR-C03",
            "claim": "Direct raw vine-state injection adds value to the recurrent policy.",
            "evidence_class": "post_holdout_explanatory",
            "decision": "opposite_direction_evidence",
            "decisive_evidence": f"Effect {effect('raw_vine_state_contribution'):+.2f} pp; MBB(3) CI {interval('raw_vine_state_contribution')}; negative in every leave-one-out sample.",
            "permissible_wording": "Raw policy-visible vine features were harmful in the causal path and not beneficial in focused windows.",
            "prohibited_wording": "The policy exploits a proven incremental raw-vine signal.",
            "manuscript_location": "Results: mechanism analysis",
        },
        {
            "claim_id": "TR-C04",
            "claim": "Masking policy-visible dependence improves robustness across retrospective windows.",
            "evidence_class": "retrospective_walk_forward",
            "decision": "supported_retrospectively_not_confirmatory",
            "decisive_evidence": f"Full minus masked effect {effect('focused_joint_visible_dependence_contribution'):+.2f} pp; MBB(3) CI {interval('focused_joint_visible_dependence_contribution')}; leave-one-out fraction positive 0%.",
            "permissible_wording": "The masked-state architecture was the strongest descriptive policy in both focused windows.",
            "prohibited_wording": "A new confirmatory masked-state winner was selected.",
            "manuscript_location": "Results: focused walk-forward robustness",
        },
        {
            "claim_id": "TR-C05",
            "claim": "Concentrated vine-synthetic pretraining outperforms matched historical-prefix training.",
            "evidence_class": "post_holdout_explanatory",
            "decision": "not_established",
            "decisive_evidence": f"Effect {effect('concentrated_synthetic_vs_historical_masked'):+.2f} pp; MBB(3) CI {interval('concentrated_synthetic_vs_historical_masked')}.",
            "permissible_wording": "Synthetic and historical masked ensembles were close in CE, with different stability and cost profiles.",
            "prohibited_wording": "Synthetic pretraining beats historical training.",
            "manuscript_location": "Results: pretraining-source controls",
        },
        {
            "claim_id": "TR-C06",
            "claim": "Concentrated vine-synthetic pretraining outperforms a moving-block bootstrap control.",
            "evidence_class": "post_holdout_explanatory",
            "decision": "promising_not_established",
            "decisive_evidence": f"Effect {effect('concentrated_synthetic_vs_bootstrap_masked'):+.2f} pp; MBB(3) CI {interval('concentrated_synthetic_vs_bootstrap_masked')}; terminal-family reality-check p={finite(terminal_reality['white_reality_check_p']):.3f}.",
            "permissible_wording": "The vine generator has a positive but statistically unresolved point advantage over naive temporal bootstrap pretraining.",
            "prohibited_wording": "The vine generator is proven superior to bootstrap simulation.",
            "manuscript_location": "Results: pretraining-source controls",
        },
        {
            "claim_id": "TR-C07",
            "claim": "Historical-only training dominates the original full-state moving-block-bootstrap control on this path.",
            "evidence_class": "post_holdout_explanatory",
            "decision": "supported_within_explanatory_sample",
            "decisive_evidence": f"Effect {effect('historical_vs_original_bootstrap'):+.2f} pp; MBB(3) CI {interval('historical_vs_original_bootstrap')}; significant under all nine resampling specifications.",
            "permissible_wording": "Historical training strongly outperformed the matched original bootstrap control in the consumed holdout.",
            "prohibited_wording": "Historical training is universally optimal.",
            "manuscript_location": "Results: pretraining-source controls",
        },
        {
            "claim_id": "TR-C08",
            "claim": "Synthetic or resampled pretraining regularizes optimization-seed variability.",
            "evidence_class": "post_holdout_explanatory",
            "decision": "supported_descriptively",
            "decisive_evidence": "Seed CE SD: historical 15.65 pp, concentrated vine synthetic 7.61 pp, moving-block bootstrap 4.47 pp.",
            "permissible_wording": "Simulation-based pretraining reduced seed dispersion and improved worst-seed stability, at a possible cost to ensemble mean performance.",
            "prohibited_wording": "Synthetic data guarantees better future returns or lower realized tail risk.",
            "manuscript_location": "Results: pretraining stability",
        },
        {
            "claim_id": "TR-C09",
            "claim": "Monthly reporting materially understates intramonth drawdown.",
            "evidence_class": "frozen_primary_daily_audit",
            "decision": "supported_descriptively",
            "decisive_evidence": "TD3 drawdown rises from 6.64% monthly to 13.45% on the reconstructed daily path; analogous ratios are near two for major benchmarks.",
            "permissible_wording": "Daily reconstruction approximately doubles observed drawdown relative to monthly endpoints.",
            "prohibited_wording": "Monthly maximum drawdown fully characterizes path risk.",
            "manuscript_location": "Results: daily downside risk",
        },
        {
            "claim_id": "TR-C10",
            "claim": "Lower turnover gives TD3 an implementation-cost advantage over direct vine optimizers.",
            "evidence_class": "frozen_weight_robustness_rescoring",
            "decision": "supported_descriptively",
            "decisive_evidence": "TD3 crosses dynamic NN-vine at 12.1 bps and static vine at 23.2 bps; no policy is retrained.",
            "permissible_wording": "The RL ensemble becomes preferable to direct vine optimizers under moderate transaction costs.",
            "prohibited_wording": "The cost analysis proves future net outperformance.",
            "manuscript_location": "Appendix: implementation robustness",
        },
        {
            "claim_id": "TR-C11",
            "claim": "The campaign provides precise 99% daily-tail estimates.",
            "evidence_class": "frozen_primary_daily_audit",
            "decision": "rejected_due_to_event_count",
            "decisive_evidence": "Only five 99% tail observations occur in each 431-day locked path.",
            "permissible_wording": "The 99% VaR/CVaR estimates are descriptive sensitivity diagnostics.",
            "prohibited_wording": "Precise or statistically established 99% tail dominance.",
            "manuscript_location": "Appendix: daily tail-risk diagnostics",
        },
        {
            "claim_id": "TR-C12",
            "claim": "Further same-holdout tuning can create a fresh confirmatory result.",
            "evidence_class": "protocol_boundary",
            "decision": "prohibited",
            "decisive_evidence": "The terminal contract forbids additional same-holdout training or model selection.",
            "permissible_wording": "Future confirmation requires independently frozen external or newly arriving data.",
            "prohibited_wording": "A revised model retested on these 24 months is independently confirmed.",
            "manuscript_location": "Limitations and future validation",
        },
    ]
    inputs = [context.input("registered_contrast_robustness_summary.csv"),
              context.input("white_reality_checks.csv")]
    context.write_csv(
        "claim_ledger/terminal_claim_ledger.csv", claims,
        artifact_type="claim_ledger_csv",
        title="Terminal evidence and claim ledger",
        evidence_class="mixed_evidence_classes", inputs=inputs)
    markdown = [
        "# Terminal evidence and claim ledger", "",
        "This is an author-control document, not a manuscript table. It prevents "
        "frozen, post-holdout, and retrospective evidence from being conflated.", "",
    ]
    for claim in claims:
        markdown.extend((
            f"## {claim['claim_id']} — {claim['decision']}", "",
            f"**Claim.** {claim['claim']}", "",
            f"**Evidence class.** `{claim['evidence_class']}`", "",
            f"**Decisive evidence.** {claim['decisive_evidence']}", "",
            f"**Permissible wording.** {claim['permissible_wording']}", "",
            f"**Do not write.** {claim['prohibited_wording']}", "",
            f"**Placement.** {claim['manuscript_location']}", "",
        ))
    context.write_text(
        "claim_ledger/terminal_claim_ledger.md", "\n".join(markdown),
        artifact_type="claim_ledger_markdown",
        title="Terminal evidence and claim ledger",
        evidence_class="mixed_evidence_classes", inputs=inputs)
