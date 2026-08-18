from __future__ import annotations

from collections import defaultdict

from .common import (FigureContext, STRATEGY_LABELS, STRATEGY_STYLES,
                     boolean, date_coordinates, number, rows, tex)


def generate(context: FigureContext) -> None:
    source = context.main_root / "raw/scored_monthly_panel.csv"
    data = [row for row in rows(source)
            if row["strategy_id"] in STRATEGY_LABELS and
            boolean(row["is_complete_period"])]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in data:
        grouped[row["strategy_id"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: row["holding_end_date"])
    require_ids = set(STRATEGY_LABELS)
    if set(grouped) != require_ids:
        raise ValueError("The main wealth panel does not contain seven strategies.")

    wealth_plots, drawdown_plots, legends = [], [], []
    for strategy in STRATEGY_LABELS:
        values = grouped[strategy]
        style = STRATEGY_STYLES[strategy]
        wealth_plots.append(
        f"\\addplot[{style}] coordinates {{{date_coordinates((r['holding_end_date'], number(r['wealth'])) for r in values)}}};")
        drawdown_plots.append(
            f"\\addplot[{style}] coordinates {{{date_coordinates((r['holding_end_date'], 100 * number(r['drawdown'])) for r in values)}}};")
        legends.append(tex(STRATEGY_LABELS[strategy]))
    body = r"""\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=1 by 2, vertical sep=1.35cm,
    x descriptions at=edge bottom},
  publication axis,
  width=\linewidth,
  date coordinates in=x,
  xticklabel={\year--\month},
  xticklabel style={rotate=30, anchor=north east},
  xmin=2024-08-01, xmax=2026-06-30,
]
\nextgroupplot[
  height=0.32\linewidth,
  ylabel={Wealth index},
  scaled y ticks=false,
  ytick={100000,120000,140000,160000},
  yticklabels={1,1.2,1.4,1.6},
  extra description/.code={\node[font=\scriptsize, anchor=south west]
    at (rel axis cs:0,1.01) {$\times10^{5}$};},
  title={(a) Common-path net wealth},
  legend columns=4,
  legend style={column sep=10pt, row sep=1pt},
  legend to name=wealthlegend,
]
""" + "\n".join(wealth_plots) + "\n\\legend{" + ",".join(legends) + r"""}
\nextgroupplot[
  height=0.23\linewidth,
  ylabel={Drawdown (\%)}, xlabel={Holding-period end},
  title={(b) Drawdown from running peak},
  ymax=0,
]
""" + "\n".join(drawdown_plots) + r"""
\end{groupplot}
\node[anchor=north] at ($(current bounding box.south)+(0,-2mm)$)
  {\pgfplotslegendfromname{wealthlegend}};
\end{tikzpicture}
"""
    context.write(
        "figure_01_wealth_drawdown.tex", body,
        title="Locked OOS wealth and drawdown",
        evidence_class="frozen_confirmatory_primary_evaluation",
        inputs=[source])
