from __future__ import annotations

from .common import FigureContext, archive_rows, number, tex


MEMBER = "analysis_results/causal_strategy_metrics.csv"
SELECTED = {
    "full_vine_state_and_cvar_observation": ("Full state", "pubSlate", "square*"),
    "zero_vine_features_keep_cvar_observation": ("Compressed CVaR", "pubNavy", "*"),
    "zero_vine_features_and_cvar_observation": ("No visible dependence", "pubRose", "triangle*"),
    "historical_only_no_synthetic_pretraining": ("Historical only", "pubGold", "diamond*"),
    "moving_block_bootstrap_pretraining": ("Moving-block bootstrap", "pubTeal", "pentagon*"),
}


def generate(context: FigureContext) -> None:
    data = [row for row in archive_rows(context.causal_archive, MEMBER)
            if row["strategy_level"] == "ensemble" and
            row["experiment_id"] in SELECTED]
    plots = []
    xs, ys = [], []
    for index, row in enumerate(data, start=1):
        label, color, mark = SELECTED[row["experiment_id"]]
        x = number(row["mean_monthly_turnover"])
        y = 100 * number(row["cagr"])
        xs.append(x)
        ys.append(y)
        plots.append(
            f"\\addplot[only marks, mark={mark}, mark size=3pt, {color}] coordinates {{({x:.10g},{y:.10g})}};\n"
            f"\\addlegendentry{{{index} {tex(label)}}}\n"
            f"\\node[font=\\scriptsize\\bfseries, text={color}, anchor=south west, "
            f"inner sep=3.2pt] at (axis cs:{x:.10g},{y:.10g}) {{{index}}};")
    body = r"""\begin{tikzpicture}
\begin{axis}[
  publication axis,
  width=0.62\linewidth,
  height=0.36\linewidth,
  xlabel={Mean monthly turnover}, ylabel={CAGR (\%)},
  title={Economic location of the principal causal representations},
  xmin=""" + f"{min(xs)-0.10*(max(xs)-min(xs)):.8g}" + r""",
  xmax=""" + f"{max(xs)+0.10*(max(xs)-min(xs)):.8g}" + r""",
  ymin=""" + f"{min(ys)-0.14*(max(ys)-min(ys)):.8g}" + r""",
  ymax=""" + f"{max(ys)+0.14*(max(ys)-min(ys)):.8g}" + r""",
  legend style={at={(1.04,0.50)}, anchor=west, legend columns=1,
    column sep=7pt, row sep=1pt},
]
""" + "\n".join(plots) + r"""
\end{axis}
\end{tikzpicture}
"""
    context.write(
        "figure_12_causal_turnover_performance.tex", body,
        title="Causal CAGR-turnover comparison",
        evidence_class="post_holdout_explanatory",
        inputs=[context.causal_archive, MEMBER])
