from __future__ import annotations

import statistics

from .common import PublicationContext, STRATEGY_LABELS, finite, require, unique_row
from .tables import PRETRAINING


COLORS = ("pubNavy", "pubTeal", "pubGold")
SHORT_LABELS = ("Historical", "Vine synthetic", "Block bootstrap")
STABILITY_ANCHORS = ("south east", "north west", "north west")
REALIZED_ANCHORS = ("south west", "north east", "north west")


def generate(context: PublicationContext) -> None:
    monthly_path = context.input("primary_economic_metrics.csv")
    daily_path = context.input("daily_tail_risk_metrics.csv")
    monthly = context.rows(monthly_path.name)
    daily = context.rows(daily_path.name)
    stability, realized = [], []
    for ((source_id, ensemble, prefix), color, label, stability_anchor,
         realized_anchor) in zip(
            PRETRAINING, COLORS, SHORT_LABELS, STABILITY_ANCHORS,
            REALIZED_ANCHORS, strict=True):
        m = unique_row(monthly, scope="complete_periods", source_id=source_id,
                       strategy_id=ensemble, window_id="locked_oos_v1")
        d = unique_row(daily, scope="complete_periods", source_id=source_id,
                       strategy_id=ensemble, window_id="locked_oos_v1")
        seeds = [100 * finite(row["annualized_certainty_equivalent_return"])
                 for row in monthly
                 if row["source_id"] == source_id
                 and row["strategy_id"].startswith(prefix)
                 and row["scope"] == "complete_periods"]
        require(len(seeds) == 10, f"Expected ten seeds for {ensemble}")
        ce = 100 * finite(m["annualized_certainty_equivalent_return"])
        seed_sd = statistics.stdev(seeds)
        cvar = 100 * finite(d["daily_cvar_95_loss"])
        stability_shift = "-2pt" if stability_anchor.endswith("east") else "2pt"
        realized_shift = "-2pt" if realized_anchor.endswith("east") else "2pt"
        stability.append(
            f"\\addplot[publication point,{color},fill={color},mark size=3pt] "
            f"coordinates {{({seed_sd:.7g},{ce:.7g})}} "
            f"node[font=\\scriptsize,anchor={stability_anchor},xshift={stability_shift},"
            f"text=pubInk] {{{label}}};")
        realized.append(
            f"\\addplot[publication point,{color},fill={color},mark size=3pt] "
            f"coordinates {{({cvar:.7g},{ce:.7g})}} "
            f"node[font=\\scriptsize,anchor={realized_anchor},xshift={realized_shift},"
            f"text=pubInk] {{{label}}};")
    body = r"""\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=2 by 1,horizontal sep=0.95cm},
  publication axis,
  width=0.41\linewidth,
  height=0.39\linewidth,
]
\nextgroupplot[
  title={(a) Optimization stability},
  xlabel={Across-seed CE standard deviation (pp)},
  ylabel={Ensemble annual CRRA CE (\%)},
  xmin=3.2,xmax=17.5,ymin=21.5,ymax=34.0,
]
""" + "\n".join(stability) + r"""
\nextgroupplot[
  title={(b) Realized tail-risk trade-off},
  xlabel={Daily CVaR$_{95}$ loss (\%)},
  xmin=1.75,xmax=2.75,ymin=21.5,ymax=34.0,
]
""" + "\n".join(realized) + r"""
\end{groupplot}
\end{tikzpicture}
"""
    context.write_figure(
        "figure_r04_pretraining_tradeoff.tex", body,
        title="Pretraining performance, stability, and tail-risk trade-off",
        evidence_class="post_holdout_explanatory",
        inputs=[monthly_path, daily_path])
