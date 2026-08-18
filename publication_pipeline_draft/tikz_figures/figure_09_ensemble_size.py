from __future__ import annotations

from .common import FigureContext, coordinates, number, rows


def generate(context: FigureContext) -> None:
    source = context.ensemble_root / "tables/exploratory_k_seed_summary.csv"
    data = rows(source)
    panels = []
    for panel_number, (metric, label, scale) in enumerate((
        ("sharpe_ratio", "Sharpe ratio", 1.0),
        ("mean_monthly_turnover", "Monthly turnover", 1.0),
    ), start=1):
        low = [(int(row["k"]), scale * number(row[f"{metric}_bootstrap_q05"])) for row in data]
        med = [(int(row["k"]), scale * number(row[f"{metric}_bootstrap_median"])) for row in data]
        high = [(int(row["k"]), scale * number(row[f"{metric}_bootstrap_q95"])) for row in data]
        panels.append(
            r"\nextgroupplot[title={" + label + r"}, xlabel={Ensemble size $k$}]" + "\n"
            f"\\addplot[name path=low{panel_number}, draw=none] coordinates {{" + coordinates(low) + r"};" + "\n"
            f"\\addplot[name path=high{panel_number}, draw=none] coordinates {{" + coordinates(high) + r"};" + "\n"
            f"\\addplot[pubNavy!14] fill between[of=low{panel_number} and high{panel_number}];" + "\n"
            r"\addplot[pubNavy, ultra thick, mark=*] coordinates {" + coordinates(med) + r"};")
    body = r"""\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=2 by 1, vertical sep=0.75cm},
  publication axis,
  width=0.47\linewidth, height=0.31\linewidth,
  xtick={1,2,3,5,10,15,20},
]
""" + "\n".join(panels) + r"""
\end{groupplot}
\end{tikzpicture}
"""
    context.write(
        "figure_09_ensemble_size.tex", body,
        title="Ensemble-size sensitivity",
        evidence_class="post_holdout_exploratory",
        inputs=[source])
