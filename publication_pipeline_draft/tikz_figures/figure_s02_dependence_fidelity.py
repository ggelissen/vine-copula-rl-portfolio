from __future__ import annotations

from .common import FigureContext, coordinates, number, rows


def generate(context: FigureContext) -> None:
    correlation_path = context.synthetic_root / "correlation_comparison.csv"
    tail_path = context.synthetic_root / "tail_dependence_comparison.csv"
    if not correlation_path.is_file() or not tail_path.is_file():
        context.skip("figure_s02_dependence_fidelity.tex",
                     "Correlation and tail-dependence diagnostic inputs are unavailable.")
        return
    panels = []
    notes = []
    for source, x_name, y_name, gate, title in (
        (correlation_path, "historical_correlation", "synthetic_correlation",
         "pass_correlation", "Pairwise correlation"),
        (tail_path, "historical_lower_tail", "synthetic_lower_tail",
         "pass_lower_tail", r"5\% lower-tail co-exceedance"),
    ):
        data = rows(source)
        passed = [row for row in data if str(row.get(gate, "")).lower() == "true"]
        failed = [row for row in data if row not in passed]
        values = [(number(row[x_name]), number(row[y_name])) for row in data]
        endpoints = [value for pair in values for value in pair]
        margin = max((max(endpoints) - min(endpoints)) * 0.10, 0.02)
        low, high = min(endpoints) - margin, max(endpoints) + margin
        panels.append(
            f"\\nextgroupplot[title={{{title}}}, xmin={low:.10g}, xmax={high:.10g}, "
            f"ymin={low:.10g}, ymax={high:.10g}, xlabel={{Historical}}, ylabel={{Synthetic}}]\n"
            f"\\addplot[pubSlate, dashed, thin] coordinates "
            f"{{({low:.10g},{low:.10g}) ({high:.10g},{high:.10g})}};\n"
            f"\\addplot[publication point, pubTeal, fill=pubTeal] coordinates "
            "{" + coordinates((number(row[x_name]), number(row[y_name])) for row in passed) + "};\n"
            f"\\addplot[publication point, pubRose, fill=pubRose, mark=triangle*] coordinates "
            "{" + coordinates((number(row[x_name]), number(row[y_name])) for row in failed) + "};")
        notes.append(f"{len(passed)}/{len(data)}")
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
    context.write("figure_s02_dependence_fidelity.tex", body,
                  title="Synthetic dependence fidelity",
                  evidence_class="training_prefix_synthetic_diagnostic",
                  inputs=[correlation_path, tail_path])
