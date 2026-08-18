from __future__ import annotations

from collections import defaultdict

from .common import FigureContext, archive_rows, cumulative_wealth, date_coordinates, number


MEMBER = "causal_strategy_periods_v2_v3_v4_plot_runtime_v1.csv"
SELECTED = {
    "full_vine_state_and_cvar_observation": ("Full raw state", "pubSlate, dashed, thick"),
    "zero_vine_features_keep_cvar_observation": ("Compressed scenario-CVaR", "pubNavy, ultra thick"),
    "zero_vine_features_and_cvar_observation": ("No visible dependence", "pubRose, dashdotted, thick"),
    "historical_only_no_synthetic_pretraining": ("Historical-only diagnostic", "pubGold, densely dashed, thick"),
}


def generate(context: FigureContext) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in archive_rows(context.causal_archive, MEMBER):
        if row["strategy_level"] == "ensemble" and row["experiment_id"] in SELECTED:
            grouped[row["experiment_id"]].append(row)
    plots = []
    for experiment, (label, style) in SELECTED.items():
        values = sorted(grouped[experiment], key=lambda row: row["holding_end_date"])
        wealth = cumulative_wealth(number(row["net_return"]) for row in values)
        coords = date_coordinates((row["holding_end_date"], value)
                                  for row, value in zip(values, wealth))
        plots.append(f"\\addplot[{style}] coordinates {{{coords}}};\n\\addlegendentry{{{label}}}")
    body = r"""\begin{tikzpicture}
\begin{axis}[
  publication axis,
  width=0.88\linewidth,
  height=0.34\linewidth,
  date coordinates in=x,
  xticklabel={\year--\month},
  xticklabel style={rotate=30, anchor=north east},
  xlabel={Holding-period end}, ylabel={Net wealth index},
  title={Causal representation ensembles on the common realized path},
  legend pos=north west,
]
""" + "\n".join(plots) + r"""
\end{axis}
\end{tikzpicture}
"""
    context.write(
        "figure_14_causal_wealth.tex", body,
        title="Causal ensemble wealth",
        evidence_class="post_holdout_explanatory",
        inputs=[context.causal_archive, MEMBER])
