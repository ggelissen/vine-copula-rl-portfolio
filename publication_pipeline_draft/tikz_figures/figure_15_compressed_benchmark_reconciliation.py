from __future__ import annotations

from .common import FigureContext, STRATEGY_LABELS, number, rows, tex


def generate(context: FigureContext) -> None:
    source = context.reconciliation_root / "paired_crra_inference.csv"
    data = rows(source)
    plots, ticks, bounds = [], [], []
    for y, row in enumerate(data, start=1):
        estimate = 100 * number(row["annualized_crra_ce_difference"])
        lower = 100 * number(row["annualized_crra_ce_ci_lower"])
        upper = 100 * number(row["annualized_crra_ce_ci_upper"])
        bounds.extend((lower, upper))
        plots.append(
            f"\\addplot[pubNavy, thick, mark=|, mark options={{solid,scale=1.2}}] "
            f"coordinates {{({lower:.10g},{y}) ({upper:.10g},{y})}};\n"
            f"\\addplot[publication point, pubNavy, fill=pubNavy] coordinates "
            f"{{({estimate:.10g},{y})}};")
        strategy = row["benchmark_strategy_id"]
        ticks.append(tex(STRATEGY_LABELS.get(strategy, strategy.replace("_", " "))))
    body = r"""\begin{tikzpicture}
\begin{axis}[
  publication axis,
  width=0.84\linewidth,
  height=0.39\linewidth,
  xlabel={Annual CRRA CE difference (percentage points)}, ylabel={},
  ytick={""" + ",".join(str(i) for i in range(1, len(data)+1)) + r"""},
  yticklabels={""" + ",".join(ticks) + r"""},
  y dir=reverse,
  title={Compressed scenario-CVaR policy minus frozen benchmarks},
  xmin=""" + f"{min(bounds)-0.07*(max(bounds)-min(bounds)):.8g}" + r""",
  xmax=""" + f"{max(bounds)+0.06*(max(bounds)-min(bounds)):.8g}" + r""",
]
\addplot[pubSlate, thin] coordinates {(0,0.5) (0,""" + f"{len(data)+0.5}" + r""")};
""" + "\n".join(plots) + r"""
\end{axis}
\end{tikzpicture}
"""
    context.write(
        "figure_15_compressed_benchmark_reconciliation.tex", body,
        title="Compressed-vine benchmark reconciliation",
        evidence_class="post_holdout_exploratory",
        inputs=[source])
