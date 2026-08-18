from __future__ import annotations

from collections import defaultdict

from .common import FigureContext, boolean, date_coordinates, number, rows


def generate(context: FigureContext) -> None:
    source = context.main_root / "raw/scored_monthly_panel.csv"
    wanted = {"vine_td3_ensemble", "equal_weight", "dcc_garch",
              "static_vine", "dynamic_nn_vine"}
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows(source):
        if row["strategy_id"] in wanted and boolean(row["is_complete_period"]):
            grouped[row["strategy_id"]][row["holding_end_date"]] = number(
                row["net_return"])
    reference = grouped["equal_weight"]
    styles = {
        "vine_td3_ensemble": "pubNavy, ultra thick",
        "dcc_garch": "pubTeal, dashed, thick",
        "static_vine": "pubGold, dashdotted, thick",
        "dynamic_nn_vine": "pubPurple, densely dashed, thick",
    }
    labels = {
        "vine_td3_ensemble": "NN-vine TD3", "dcc_garch": "DCC--GARCH",
        "static_vine": "Static vine", "dynamic_nn_vine": "Dynamic NN-vine"}
    plots = []
    for strategy, style in styles.items():
        values = sorted(grouped[strategy].items())
        plots.append(
            f"\\addplot[{style}, mark=none] coordinates "
            f"{{{date_coordinates((d, 100*(v-reference[d])) for d,v in values)}}};\n"
            f"\\addlegendentry{{{labels[strategy]}}}")
    body = r"""\begin{tikzpicture}
\begin{axis}[
  publication axis,
  width=0.88\linewidth,
  height=0.34\linewidth,
  date coordinates in=x,
  xticklabel={\year--\month},
  xticklabel style={rotate=30, anchor=north east},
  xlabel={Holding-period end}, ylabel={Excess return (pp)},
  title={Month-by-month relative performance},
  xmin=2024-08-01, xmax=2026-06-30,
  enlarge x limits=false,
  legend style={at={(0.98,0.04)}, anchor=south east, legend columns=2},
]
\addplot[pubGray, thin] coordinates {(2024-08-01,0) (2026-06-30,0)};
""" + "\n".join(plots) + r"""
\end{axis}
\end{tikzpicture}
"""
    context.write(
        "figure_07_monthly_excess.tex", body,
        title="Monthly excess returns",
        evidence_class="frozen_confirmatory_primary_evaluation",
        inputs=[source])
