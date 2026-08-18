from __future__ import annotations

from .common import FigureContext, STRATEGY_LABELS, number, rows, tex


def generate(context: FigureContext) -> None:
    source = context.main_root / "tables/table_01_oos_performance.csv"
    data = rows(source)
    marks = {
        "equal_weight": "square*", "shrinkage_mean_variance": "triangle*",
        "dcc_garch": "diamond*", "static_vine": "pentagon*",
        "rolling_vine": "x", "dynamic_nn_vine": "star",
        "vine_td3_ensemble": "*",
    }
    points = []
    for index, row in enumerate(data, start=1):
        strategy = row["strategy_id"]
        color = "pubNavy" if strategy == "vine_td3_ensemble" else "pubSlate"
        size = "3.6pt" if strategy == "vine_td3_ensemble" else "2.7pt"
        label = STRATEGY_LABELS[strategy]
        points.append(
            f"\\addplot[only marks, mark={marks[strategy]}, mark size={size}, {color}] "
            f"coordinates {{({100*number(row['annual_volatility']):.8g},"
            f"{100*number(row['cagr']):.8g})}};\n"
            f"\\addlegendentry{{{index} {tex(label)} "
            f"[CE {100*number(row['annualized_certainty_equivalent_return']):.1f}\\%]}}\n"
            f"\\node[font=\\scriptsize\\bfseries, text={color}, anchor=south west, "
            f"inner sep=3.3pt] at "
            f"(axis cs:{100*number(row['annual_volatility']):.8g},"
            f"{100*number(row['cagr']):.8g}) {{{index}}};")
    body = r"""\begin{tikzpicture}
\begin{axis}[
  publication axis,
  width=0.62\linewidth,
  height=0.40\linewidth,
  xlabel={Annual volatility (\%)},
  ylabel={CAGR (\%)},
  title={Return--risk frontier with CRRA certainty equivalents},
  enlarge x limits=0.08,
  enlarge y limits=0.12,
  legend style={at={(1.04,0.50)}, anchor=west, legend columns=1,
    column sep=7pt, row sep=1pt, /tikz/every even column/.append style={column sep=7pt}},
]
""" + "\n".join(points) + r"""
\end{axis}
\end{tikzpicture}
"""
    context.write(
        "figure_02_risk_return_utility.tex", body,
        title="Risk-return-utility frontier",
        evidence_class="frozen_confirmatory_primary_evaluation",
        inputs=[source])
