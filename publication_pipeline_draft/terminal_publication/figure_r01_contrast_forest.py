from __future__ import annotations

from .common import (
    CONTRAST_LABELS, PublicationContext, finite, tex)
from .tables import KEY_CONTRASTS


COLORS = {
    "frozen_primary_benchmarks": "pubNavy",
    "post_holdout_causal_components": "pubGold",
    "retrospective_focused_mechanisms": "pubRose",
    "post_holdout_pretraining_sources": "pubGold",
    "post_holdout_terminal_pretraining_controls": "pubGold",
}


def generate(context: PublicationContext) -> None:
    input_path = context.input("registered_contrast_robustness_summary.csv")
    source = {row["contrast_id"]: row for row in context.rows(input_path.name)}
    plots, labels, bounds = [], [], []
    for index, contrast_id in enumerate(KEY_CONTRASTS, start=1):
        row = source[contrast_id]
        estimate = 100 * finite(row["annualized_ce_difference"])
        lower = 100 * finite(row["registered_moving_block_3_ci_lower"])
        upper = 100 * finite(row["registered_moving_block_3_ci_upper"])
        color = COLORS[row["family"]]
        bounds.extend((lower, upper))
        labels.append(tex(CONTRAST_LABELS[contrast_id]))
        plots.append(
            f"\\addplot[{color}, thick, mark=|, mark options={{solid,scale=1.15}}] "
            f"coordinates {{({lower:.8g},{index}) ({upper:.8g},{index})}};\n"
            f"\\addplot[publication point,{color},fill={color}] coordinates "
            f"{{({estimate:.8g},{index})}};")
    padding = 0.05 * (max(bounds) - min(bounds))
    body = r"""\begin{tikzpicture}
\begin{axis}[
  publication axis,
  % PGFPlots' width excludes the long y tick labels.  Reserve enough of the
  % text block for those labels so the complete TikZ bounding box fits.
  width=0.72\linewidth,
  height=0.50\linewidth,
  xmin=""" + f"{min(bounds)-padding:.8g}" + r""", xmax=""" + f"{max(bounds)+padding:.8g}" + r""",
  ymin=0.55, ymax=""" + f"{len(KEY_CONTRASTS)+0.45}" + r""",
  enlarge y limits=false,
  y dir=reverse,
  xlabel={Annual CRRA certainty-equivalent effect (percentage points)},
  ylabel={},
  ytick={""" + ",".join(str(i) for i in range(1, len(KEY_CONTRASTS)+1)) + r"""},
  yticklabels={""" + ",".join(labels) + r"""},
  tick label style={font=\scriptsize},
  legend columns=3,
  legend style={at={(0.5,1.02)},anchor=south,cells={anchor=center},column sep=4pt},
]
\addplot[pubSlate,thin] coordinates {(0,0.55) (0,""" + f"{len(KEY_CONTRASTS)+0.45}" + r""")};
""" + "\n".join(plots) + r"""
\addplot[pubNavy,thick] coordinates {(nan,nan)}; \addlegendentry{Frozen primary}
\addplot[pubGold,thick] coordinates {(nan,nan)}; \addlegendentry{Post-holdout}
\addplot[pubRose,thick] coordinates {(nan,nan)}; \addlegendentry{Retrospective}
\end{axis}
\end{tikzpicture}
"""
    # Replace PGFPlots' conventional legend-only NaNs with finite off-canvas
    # coordinates because the publication validator forbids non-finite tokens.
    body = body.replace("(nan,nan)", f"({min(bounds)-1000:.8g},0)")
    context.write_figure(
        "figure_r01_terminal_contrast_forest.tex", body,
        title="Terminal registered-effect forest",
        evidence_class="mixed_evidence_classes",
        inputs=[input_path])
