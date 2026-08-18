from __future__ import annotations

from .common import FigureContext, archive_rows, number, tex


MEMBER = "analysis_results/causal_primary_contrasts.csv"

SHORT_LABELS = {
    "Direct NN-vine state contribution": "Direct raw vine state",
    "Scenario-CVaR observation contribution": "Scenario-CVaR observation",
    "Joint policy-visible dependence contribution": "Joint visible dependence",
    "CVaR reward-shaping contribution": "CVaR reward shaping",
    "Synthetic NN-vine pretraining contribution": "Synthetic pre-training",
    "NN-vine generator versus temporal bootstrap": "NN-vine vs block bootstrap",
    "Recurrent encoder contribution": "Recurrent encoder",
    "Historical fine-tuning contribution": "Historical fine-tuning",
}


def generate(context: FigureContext) -> None:
    data = archive_rows(context.causal_archive, MEMBER)
    plots, labels, bounds = [], [], []
    for index, row in enumerate(data, start=1):
        estimate = 100 * number(row["annualized_ce_difference"])
        lower = 100 * number(row["annualized_ce_ci_lower"])
        upper = 100 * number(row["annualized_ce_ci_upper"])
        bounds.extend((lower, upper))
        decision = row["contract_decision"]
        color = "pubRose" if decision == "opposite_direction_evidence" else "pubNavy"
        plots.append(
            f"\\addplot[{color}, thick, mark=|, mark options={{solid,scale=1.2}}] "
            f"coordinates {{({lower:.10g},{index}) ({upper:.10g},{index})}};\n"
            f"\\addplot[publication point, {color}, fill={color}] coordinates "
            f"{{({estimate:.10g},{index})}};")
        short = SHORT_LABELS.get(row["label"], row["label"])
        labels.append(tex(short))
    body = r"""\begin{tikzpicture}
\begin{axis}[
  publication axis,
  width=0.84\linewidth,
  height=0.49\linewidth,
  xlabel={Annual CRRA CE effect (percentage points)}, ylabel={},
  ytick={""" + ",".join(str(i) for i in range(1, len(data)+1)) + r"""},
  yticklabels={""" + ",".join(labels) + r"""},
  y dir=reverse,
  ymin=0.85, ymax=""" + f"{len(data)+0.15}" + r""",
  enlarge y limits=false,
  title={Matched causal component effects: full reference minus alternative},
  xmin=""" + f"{min(bounds)-0.08*(max(bounds)-min(bounds)):.8g}" + r""",
  xmax=""" + f"{max(bounds)+0.06*(max(bounds)-min(bounds)):.8g}" + r""",
]
\addplot[pubSlate, thin] coordinates {(0,0) (0,""" + f"{len(data)+1}" + r""")};
""" + "\n".join(plots) + r"""
\end{axis}
\end{tikzpicture}
"""
    context.write(
        "figure_11_causal_forest.tex", body,
        title="Causal component-effect forest",
        evidence_class="post_holdout_explanatory",
        inputs=[context.causal_archive, MEMBER])
