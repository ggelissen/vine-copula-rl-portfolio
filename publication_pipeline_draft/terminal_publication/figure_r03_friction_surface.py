from __future__ import annotations

from .common import PublicationContext, STRATEGY_LABELS, finite


PRIMARY = (
    ("frozen_primary_oos", "vine_td3_ensemble", "pubNavy,ultra thick"),
    ("frozen_primary_oos", "equal_weight", "pubGray,densely dotted,thick"),
    ("frozen_primary_oos", "static_vine", "pubGold,dashed,thick"),
    ("frozen_primary_oos", "dynamic_nn_vine", "pubPurple,dashdotted,thick"),
)

PRETRAINING = (
    ("masked_pretraining_controls", "masked_historical_prefix_1000_presentations_ensemble", "pubNavy,ultra thick"),
    ("synthetic_presentations", "synthetic_100_unique_1000_presentations_no_policy_visible_dependence_ensemble", "pubTeal,dashed,thick"),
    ("masked_pretraining_controls", "masked_moving_block_bootstrap_1000_presentations_ensemble", "pubGold,dashdotted,thick"),
)


def _plots(rows: list[dict[str, str]], specifications: tuple[tuple[str, str, str], ...]) -> str:
    result = []
    for source_id, strategy, style in specifications:
        selected = sorted([
            row for row in rows
            if row["scope"] == "complete_periods"
            and row["source_id"] == source_id
            and row["strategy_id"] == strategy
            and row["annual_short_borrow_percent"] == "3"
            and row["annual_cash_borrow_percent"] == "2"
        ], key=lambda row: finite(row["transaction_cost_bps_one_way"]))
        coordinates = " ".join(
            f"({finite(row['transaction_cost_bps_one_way']):.7g},"
            f"{100*finite(row['annualized_certainty_equivalent_return']):.7g})"
            for row in selected)
        result.append(
            f"\\addplot[{style},mark=*] coordinates {{{coordinates}}}; "
            f"\\addlegendentry{{{STRATEGY_LABELS[strategy]}}}")
    return "\n".join(result)


def generate(context: PublicationContext) -> None:
    input_path = context.input("friction_surface.csv")
    rows = context.rows(input_path.name)
    body = r"""\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=2 by 1,horizontal sep=1.05cm},
  publication axis,
  width=0.47\linewidth,
  height=0.39\linewidth,
  xlabel={One-way transaction cost (bps)},
  xmin=-1, xmax=51,
  xtick={0,10,25,50},
  legend columns=2,
  legend style={font=\tiny,at={(0.5,-0.23)},anchor=north,
                cells={anchor=west},column sep=3pt,row sep=1pt},
]
\nextgroupplot[title={(a) Frozen primary strategies},ylabel={Annual CRRA CE (\%)},
               ymin=20,ymax=33]
""" + _plots(rows, PRIMARY) + r"""
\addplot[pubSlate!70,thin,densely dotted] coordinates {(12.064,20) (12.064,33)};
\addplot[pubSlate!70,thin,densely dashed] coordinates {(23.198,20) (23.198,33)};
\nextgroupplot[title={(b) Matched pretraining sources},ymin=18,ymax=35]
""" + _plots(rows, PRETRAINING) + r"""
\addplot[pubSlate!70,thin,densely dashed] coordinates {(43.114,18) (43.114,35)};
\end{groupplot}
\end{tikzpicture}
"""
    context.write_figure(
        "figure_r03_friction_surface.tex", body,
        title="Frozen-weight transaction-cost robustness",
        evidence_class="robustness_rescoring_no_retraining",
        inputs=[input_path])
