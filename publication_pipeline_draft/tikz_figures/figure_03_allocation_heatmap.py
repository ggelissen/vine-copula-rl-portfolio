from __future__ import annotations

from .common import FigureContext, boolean, number, rows, tex


def generate(context: FigureContext) -> None:
    source = context.main_root / "raw/scored_monthly_panel.csv"
    data = [row for row in rows(source)
            if row["strategy_id"] == "vine_td3_ensemble" and
            boolean(row["is_complete_period"])]
    assets = ["SP500", "NASDAQ", "DOW", "SSE50", "DIVIDEND", "CHINEXT", "GOLD"]
    entries = []
    # PGFPlots matrix plots use row-major scanlines: x changes fastest.
    for y, asset in enumerate(assets, start=1):
        for x, row in enumerate(data, start=1):
            entries.append(f"{x} {y} {number(row['w_' + asset]):.10g}")
    labels = ",".join(tex(asset) for asset in assets)
    tick_indices = list(range(1, len(data) + 1, 3))
    if tick_indices[-1] != len(data):
        tick_indices.append(len(data))
    tick_labels = [data[index - 1]["holding_end_date"][:7]
                   for index in tick_indices]
    body = r"""\begin{tikzpicture}
\begin{axis}[
  publication axis,
  width=0.88\linewidth,
  height=0.36\linewidth,
  title={NN-vine TD3 ensemble target allocations},
  xlabel={Evaluation month}, ylabel={},
  xmin=0.5, xmax=""" + f"{len(data)+0.5}, ymin=0.5, ymax={len(assets)+0.5},\n" + r"""  xtick={""" + ",".join(str(value) for value in tick_indices) + r"""},
  xticklabels={""" + ",".join(tick_labels) + r"""},
  xticklabel style={rotate=25, anchor=north east, font=\scriptsize},
  ytick={1,2,3,4,5,6,7},
  yticklabels={""" + labels + r"""},
  colormap={diverging}{rgb255(0cm)=(150,18,42)
    rgb255(0.25cm)=(218,112,126)
    rgb255(0.625cm)=(247,247,248)
    rgb255(1cm)=(37,100,158)
    rgb255(1.25cm)=(10,42,78)},
  point meta min=-0.2, point meta max=0.6,
  colorbar,
  colorbar style={width=1.8mm, ylabel={Weight},
    ylabel style={font=\scriptsize}, tick label style={font=\scriptsize},
    ytick={-0.2,0,0.2,0.4,0.6}},
]
\addplot[matrix plot*, mesh/cols=""" + str(len(data)) + r""", point meta=explicit]
 table[meta index=2] {
x y meta
""" + "\n".join(entries) + r"""
};
\end{axis}
\end{tikzpicture}
"""
    context.write(
        "figure_03_allocation_heatmap.tex", body,
        title="Ensemble allocation heatmap",
        evidence_class="frozen_confirmatory_primary_evaluation",
        inputs=[source])
