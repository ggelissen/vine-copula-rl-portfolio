from __future__ import annotations

from .common import FigureContext, number, rows, tex


SHORT = {
    "equal_weight": "Equal weight",
    "shrinkage_mean_variance": "Shrinkage MV",
    "dcc_garch": "DCC--GARCH",
    "static_vine": "Static vine",
    "rolling_vine": "Rolling vine",
    "dynamic_nn_vine": "Dynamic NN-vine",
    "vine_td3_ensemble": "NN-vine TD3",
}


def generate(context: FigureContext) -> None:
    source = context.main_root / "tables/table_04_economic_implementation.csv"
    data = rows(source)
    labels = [tex(SHORT.get(row["strategy_id"], row["label"])) for row in data]
    turnover = " ".join(
        f"({number(row['mean_monthly_turnover']):.10g},{i})"
        for i, row in enumerate(data, 1))
    drag = " ".join(
        f"({100*number(row['implementation_drag_total_return']):.10g},{i})"
        for i, row in enumerate(data, 1))
    body = r"""\begin{tikzpicture}
\begin{groupplot}[
 group style={group size=2 by 1, horizontal sep=0.70cm,
   group name=implementation},
 publication axis, height=0.43\linewidth,
 ytick={""" + ",".join(str(i) for i in range(1, len(data) + 1)) + r"""},
 y dir=reverse, ymin=0.5, ymax=""" + f"{len(data)+0.5}" + r""",
]
\nextgroupplot[
 width=0.56\linewidth,
 title={Trading intensity}, xlabel={Mean monthly turnover},
 yticklabels={""" + ",".join(labels) + r"""},
]
\addplot[xbar, bar width=5pt, fill=pubNavy!70, draw=pubNavy]
 coordinates {""" + turnover + r"""};
\nextgroupplot[
 width=0.30\linewidth,
 title={Realized cost drag}, xlabel={Total-return drag (pp)},
 yticklabels=\empty,
]
\addplot[only marks, mark=diamond*, mark size=2.7pt, pubRose]
 coordinates {""" + drag + r"""};
\end{groupplot}
\end{tikzpicture}
"""
    context.write(
        "figure_04_implementation.tex", body,
        title="Turnover and implementation drag",
        evidence_class="frozen_confirmatory_primary_evaluation",
        inputs=[source])
