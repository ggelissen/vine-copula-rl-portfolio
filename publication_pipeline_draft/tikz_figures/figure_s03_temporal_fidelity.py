from __future__ import annotations

from .common import FigureContext, coordinates, number, rows, tex


def generate(context: FigureContext) -> None:
    source = context.synthetic_root / "temporal_dependence.csv"
    if not source.is_file():
        context.skip("figure_s03_temporal_fidelity.tex",
                     f"Temporal diagnostic input is unavailable: {source}")
        return
    data = rows(source)
    panels = []
    for historical, synthetic, title in (
        ("historical_lag1", "synthetic_lag1", "Lag-1 return dependence"),
        ("historical_squared_lag1", "synthetic_squared_lag1",
         "Lag-1 squared-return dependence"),
    ):
        values = [(number(row[historical]), number(row[synthetic])) for row in data]
        endpoints = [value for pair in values for value in pair]
        margin = max((max(endpoints) - min(endpoints)) * 0.12, 0.02)
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
            f"\\addplot[publication point, pubPurple, fill=pubPurple] coordinates "
            "{" + coordinates(values) + "};\n" + labels)
    passed = sum(str(row.get("pass_temporal", "")).lower() == "true" for row in data)
    body = r"""\begin{tikzpicture}
\begin{groupplot}[
 group style={group size=2 by 1, horizontal sep=1.75cm}, publication axis,
 width=0.45\linewidth, height=0.34\linewidth,
 tick label style={font=\tiny, text=pubInk},
]
""" + "\n".join(panels) + r"""
\end{groupplot}
\end{tikzpicture}
"""
    context.write("figure_s03_temporal_fidelity.tex", body,
                  title="Synthetic temporal fidelity",
                  evidence_class="training_prefix_synthetic_diagnostic",
                  inputs=[source])
