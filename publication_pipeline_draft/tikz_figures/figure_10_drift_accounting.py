from __future__ import annotations

from .common import FigureContext, number, rows


PANELS = (
    ("reported_turnover", "drift_aware_turnover",
     "Monthly turnover", "Turnover"),
    ("reported_transaction_cost_bps", "drift_aware_transaction_cost_bps",
     "Transaction cost (bps)", "Trading cost"),
)


def generate(context: FigureContext) -> None:
    source = context.ensemble_root / "tables/exploratory_drift_turnover_summary.csv"
    data = rows(source)
    labels = ["Mean seed", "Ensemble"]
    panels = []
    for reported_key, drifted_key, xlabel, title in PANELS:
        plots = []
        for y, row in enumerate(data, 1):
            reported = number(row[reported_key])
            drifted = number(row[drifted_key])
            delta = drifted - reported
            plots.append(
                f"\\draw[pubGray, line width=0.8pt] (axis cs:{reported:.10g},{y}) -- "
                f"(axis cs:{drifted:.10g},{y});\n"
                f"\\addplot[only marks, mark=*, mark size=3pt, pubSlate] "
                f"coordinates {{({reported:.10g},{y})}};\n"
                f"\\addplot[only marks, mark=diamond*, mark size=3.3pt, pubNavy] "
                f"coordinates {{({drifted:.10g},{y})}};\n"
                f"\\node[font=\\tiny, anchor=south, inner sep=2pt, text=pubNavy] "
                f"at (axis cs:{(reported+drifted)/2:.10g},{y}) "
                f"{{+$\\Delta$ {delta:.3f}}};")
        panels.append(
            f"\\nextgroupplot[title={{{title}}}, xlabel={{{xlabel}}}, "
            f"ytick={{1,2}}, yticklabels={{{','.join(labels)}}}, y dir=reverse, "
            "ymin=0.55, ymax=2.45, grid=none, xlabel style={yshift=-1.5mm}]\n" +
            "\n".join(plots))
    body = r"""\begin{tikzpicture}
\begin{groupplot}[
 group style={group size=2 by 1, horizontal sep=2cm},
 publication axis, width=0.42\linewidth, height=0.25\linewidth,
]
""" + "\n".join(panels) + r"""
\end{groupplot}
\matrix[draw=none, font=\scriptsize, column sep=3pt, row sep=2pt,
 anchor=east]
 at ($(current bounding box.east)+(-1mm,0)$) {
 \node[draw=pubSlate, fill=pubSlate, circle, inner sep=1.2pt] {}; &
 \node[anchor=west] {Target-to-target}; \\
 \node[draw=pubNavy, fill=pubNavy, diamond, inner sep=1.2pt] {}; &
 \node[anchor=west] {Drift-aware}; \\
};
\end{tikzpicture}
"""
    context.write(
        "figure_10_drift_accounting.tex", body,
        title="Drift-aware turnover and cost accounting",
        evidence_class="post_holdout_exploratory",
        inputs=[source])
