from __future__ import annotations

from .common import FigureContext, number, rows, tex


def generate(context: FigureContext) -> None:
    source = context.main_root / "tables/table_10_primary_effects.csv"
    data = rows(source)
    plots, ticklabels, bounds = [], [], []
    for index, row in enumerate(data, start=1):
        estimate = 100 * number(row["mean_utility_difference"])
        lower = 100 * number(row["bootstrap_ci_lower"])
        upper = 100 * number(row["bootstrap_ci_upper"])
        bounds.extend((lower, upper))
        plots.append(
            f"\\addplot[pubNavy, thick, mark=|, mark options={{solid,scale=1.2}}] "
            f"coordinates {{({lower:.10g},{index}) ({upper:.10g},{index})}};\n"
            f"\\addplot[publication point, pubNavy, fill=pubNavy] coordinates "
            f"{{({estimate:.10g},{index})}};")
        short = (row["benchmark_label"]
                 .replace("Constrained shrinkage mean-variance", "Shrinkage MV")
                 .replace("Dynamic NN-vine optimizer without RL", "Dynamic NN-vine")
                 .replace(" optimizer", ""))
        ticklabels.append(tex(short))
    body = r"""\begin{tikzpicture}
\begin{axis}[
  publication axis,
  width=0.86\linewidth,
  height=0.40\linewidth,
  xlabel={Mean monthly CRRA utility difference (percentage points)},
  ylabel={},
  ytick={""" + ",".join(str(i) for i in range(1, len(data)+1)) + r"""},
  yticklabels={""" + ",".join(ticklabels) + r"""},
  y dir=reverse,
  xmin=""" + f"{min(bounds)-0.15*(max(bounds)-min(bounds)):.8g}" + r""",
  xmax=""" + f"{max(bounds)+0.08*(max(bounds)-min(bounds)):.8g}" + r""",
  title={NN-vine TD3 ensemble minus each frozen benchmark},
]
\addplot[pubSlate, thin] coordinates {(0,0.5) (0,""" + f"{len(data)+0.5}" + r""")};
""" + "\n".join(plots) + r"""
\end{axis}
\end{tikzpicture}
"""
    context.write(
        "figure_06_primary_inference.tex", body,
        title="Primary paired utility inference",
        evidence_class="frozen_confirmatory_primary_evaluation",
        inputs=[source])
