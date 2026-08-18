from __future__ import annotations

from collections import defaultdict

from .common import FigureContext, archive_rows, number, tex


MEMBER = "analysis_results/causal_seed_pair_effects.csv"
CONTRASTS = [
    "Direct NN-vine state contribution",
    "Joint policy-visible dependence contribution",
    "Synthetic NN-vine pretraining contribution",
]


def generate(context: FigureContext) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in archive_rows(context.causal_archive, MEMBER):
        if row["label"] in CONTRASTS:
            grouped[row["label"]].append(row)
    plots, ticks = [], []
    for y, label in enumerate(CONTRASTS, start=1):
        ticks.append(tex(label.replace("Direct NN-vine state contribution", "Raw vine state")
                         .replace("Joint policy-visible dependence contribution", "Joint visible dependence")
                         .replace("Synthetic NN-vine pretraining contribution", "Synthetic pre-training")))
        effects = sorted(100 * number(row["paired_annualized_ce_difference"])
                         for row in grouped[label])
        for effect in effects:
            plots.append(
                f"\\addplot[only marks, mark=*, mark size=1.5pt, pubNavy!58] "
                f"coordinates {{({effect:.10g},{y})}};")
        middle = len(effects) // 2
        median = effects[middle] if len(effects) % 2 else (
            effects[middle - 1] + effects[middle]) / 2
        plots.append(
            f"\\addplot[only marks, mark=diamond*, mark size=3pt, pubRose] "
            f"coordinates {{({median:.10g},{y})}};")
    body = r"""\begin{tikzpicture}
\begin{axis}[
  publication axis,
  width=0.84\linewidth,
  height=0.27\linewidth,
  xlabel={Paired annual CRRA CE effect (percentage points)}, ylabel={},
  ytick={1,2,3}, yticklabels={""" + ",".join(ticks) + r"""},
  y dir=reverse,
  ymin=0.85, ymax=3.15,
  enlarge y limits=false,
  title={Matched-seed heterogeneity in key mechanism effects},
]
\addplot[pubSlate, thin] coordinates {(0,0.5) (0,3.5)};
""" + "\n".join(plots) + r"""
\end{axis}
\end{tikzpicture}
"""
    context.write(
        "figure_13_causal_seed_effects.tex", body,
        title="Causal matched-seed heterogeneity",
        evidence_class="post_holdout_explanatory",
        inputs=[context.causal_archive, MEMBER])
