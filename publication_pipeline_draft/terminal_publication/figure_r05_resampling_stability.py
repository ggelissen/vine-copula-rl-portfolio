from __future__ import annotations

from .common import CONTRAST_LABELS, PublicationContext, finite, tex
from .tables import KEY_CONTRASTS


SPECS = (
    ("moving_block", "1", "M1"),
    ("moving_block", "2", "M2"),
    ("moving_block", "3", "M3"),
    ("moving_block", "4", "M4"),
    ("moving_block", "6", "M6"),
    ("stationary", "2", "S2"),
    ("stationary", "3", "S3"),
    ("stationary", "6", "S6"),
    ("stationary", "12", "S12"),
)


def generate(context: PublicationContext) -> None:
    input_path = context.input("resampling_robustness.csv")
    rows = context.rows(input_path.name)
    lookup = {(row["contrast_id"], row["method"], row["block_length"]): row
              for row in rows}
    cells = []
    for y, contrast_id in enumerate(KEY_CONTRASTS):
        for x, (method, length, _) in enumerate(SPECS):
            row = lookup[(contrast_id, method, length)]
            lower = finite(row["ci_lower"])
            upper = finite(row["ci_upper"])
            if lower > 0:
                fill, symbol = "pubTeal!72", "+"
            elif upper < 0:
                fill, symbol = "pubRose!72", "$-$"
            else:
                fill, symbol = "pubGray!28", "0"
            cells.append(
                f"\\node[draw=white,line width=0.6pt,fill={fill},minimum width=7.2mm,"
                f"minimum height=5.5mm,font=\\scriptsize] at ({x},{-y}) {{{symbol}}};")
    xlabels = [
        f"\\node[font=\\scriptsize,anchor=south] at ({x},0.58) {{{label}}};"
        for x, (_, _, label) in enumerate(SPECS)
    ]
    ylabels = [
        f"\\node[font=\\scriptsize,anchor=east] at (-0.65,{-y}) "
        f"{{{tex(CONTRAST_LABELS[contrast_id])}}};"
        for y, contrast_id in enumerate(KEY_CONTRASTS)
    ]
    body = r"""\begin{tikzpicture}[x=7.2mm,y=5.5mm]
""" + "\n".join(cells + xlabels + ylabels) + r"""
\node[draw=none,fill=pubTeal!72,minimum width=4mm,minimum height=3mm] at (0,-8.55) {};
\node[font=\scriptsize,anchor=west] at (0.35,-8.55) {Above zero};
\node[draw=none,fill=pubGray!28,minimum width=4mm,minimum height=3mm] at (3.5,-8.55) {};
\node[font=\scriptsize,anchor=west] at (3.85,-8.55) {Crosses zero};
\node[draw=none,fill=pubRose!72,minimum width=4mm,minimum height=3mm] at (7.0,-8.55) {};
\node[font=\scriptsize,anchor=west] at (7.35,-8.55) {Below zero};
\end{tikzpicture}
"""
    context.write_figure(
        "figure_r05_resampling_stability.tex", body,
        title="Sign stability across block-bootstrap specifications",
        evidence_class="mixed_evidence_classes_robustness",
        inputs=[input_path])
