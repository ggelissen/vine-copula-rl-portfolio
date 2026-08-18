from __future__ import annotations

from .common import FigureContext, coordinates, number, rows, tex


PANELS = (
    ("Mean", "historical_mean", "synthetic_mean"),
    ("Standard deviation", "historical_sd", "synthetic_sd"),
    (r"5\% quantile", "historical_q05", "synthetic_q05"),
    (r"5\% CVaR", "historical_cvar05", "synthetic_cvar05"),
)


def generate(context: FigureContext) -> None:
    source = context.synthetic_root / "fidelity_metrics.csv"
    if not source.is_file():
        context.skip("figure_s01_marginal_fidelity.tex",
                     f"Synthetic diagnostic input is unavailable: {source}")
        return
    data = rows(source)
    panels = []
    for title, historical, synthetic in PANELS:
        values = [(number(row[historical]), number(row[synthetic])) for row in data]
        endpoints = [value for pair in values for value in pair]
        span = max(endpoints) - min(endpoints)
        margin = max(span * 0.12, 0.001)
        low, high = min(endpoints) - margin, max(endpoints) + margin
        labels = "\n".join(
            f"\\node[font=\\tiny, anchor=south west, inner sep=2.5pt, text=pubSlate] at "
            f"(axis cs:{x:.10g},{y:.10g}) {{{tex(row['asset'])}}};"
            for row, (x, y) in zip(data, values))
        panels.append(
            f"\\nextgroupplot[title={{{title}}}, xmin={low:.10g}, xmax={high:.10g}, "
            f"ymin={low:.10g}, ymax={high:.10g}, xlabel={{Historical}}, ylabel={{Synthetic}}]\n"
            f"\\addplot[pubSlate, dashed, thin] coordinates "
            f"{{({low:.10g},{low:.10g}) ({high:.10g},{high:.10g})}};\n"
            f"\\addplot[publication point, pubNavy, fill=pubNavy] coordinates "
            "{" + coordinates(values) + "};\n" + labels)
    passed = sum(str(row.get("pass_marginals", "")).lower() == "true" for row in data)
    body = r"""\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=2 by 2, horizontal sep=1.75cm, vertical sep=2cm},
  publication axis, width=0.48\linewidth, height=0.3\linewidth,
  tick label style={font=\tiny, text=pubInk},
]
""" + "\n".join(panels) + r"""
\end{groupplot}
\end{tikzpicture}
"""
    context.write("figure_s01_marginal_fidelity.tex", body,
                  title="Synthetic marginal fidelity",
                  evidence_class="training_prefix_synthetic_diagnostic",
                  inputs=[source])
