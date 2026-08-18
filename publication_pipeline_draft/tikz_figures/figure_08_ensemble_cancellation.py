from __future__ import annotations

from .common import FigureContext, number, rows


def generate(context: FigureContext) -> None:
    source = context.ensemble_root / "tables/explanatory_ensemble_mechanism_summary.csv"
    row = rows(source)[0]
    metrics = [
        ("Gross exposure", number(row["mean_seed_gross_exposure"]),
         number(row["ensemble_gross_exposure"])),
        ("Short notional", number(row["mean_seed_short_notional"]),
         number(row["ensemble_short_notional"])),
        ("Monthly turnover", number(row["mean_seed_turnover"]),
         number(row["ensemble_turnover"])),
    ]
    labels = ",".join(label for label, _, _ in metrics)
    plots = []
    for y, (_, seed_value, ensemble_value) in enumerate(metrics, 1):
        reduction = 100 * (seed_value - ensemble_value) / seed_value
        plots.append(
            f"\\draw[pubGray, thick] (axis cs:{seed_value:.10g},{y}) -- "
            f"(axis cs:{ensemble_value:.10g},{y});\n"
            f"\\addplot[only marks, mark=*, mark size=2.4pt, pubSlate] "
            f"coordinates {{({seed_value:.10g},{y})}};\n"
            f"\\addplot[only marks, mark=diamond*, mark size=2.8pt, pubNavy] "
            f"coordinates {{({ensemble_value:.10g},{y})}};\n"
            f"\\node[font=\\scriptsize, text=pubNavy, anchor=south, inner sep=2pt] "
            f"at (axis cs:{(seed_value + ensemble_value)/2:.10g},{y}) "
            f"{{{reduction:.1f}\\% lower}};")
    body = r"""\begin{tikzpicture}
\begin{axis}[
  publication axis,
  width=0.82\linewidth, height=0.34\linewidth,
  xmin=0, ylabel={}, xlabel={Average exposure or turnover},
  ytick={1,2,3}, yticklabels={""" + labels + r"""}, y dir=reverse,
  legend style={at={(0.98,0.04)}, anchor=south east, legend columns=1},
  title={Weight-space ensembling cancels risky disagreement},
]
""" + "\n".join(plots) + r"""
\addlegendimage{only marks, mark=*, mark size=2.4pt, pubSlate}
\addlegendentry{Mean individual seed}
\addlegendimage{only marks, mark=diamond*, mark size=2.8pt, pubNavy}
\addlegendentry{Weight ensemble}
\end{axis}
\end{tikzpicture}
"""
    context.write(
        "figure_08_ensemble_cancellation.tex", body,
        title="Ensemble exposure cancellation",
        evidence_class="post_holdout_explanatory",
        inputs=[source])
