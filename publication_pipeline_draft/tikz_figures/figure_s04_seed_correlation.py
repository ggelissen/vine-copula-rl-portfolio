from __future__ import annotations

from .common import FigureContext, number, rows


def generate(context: FigureContext) -> None:
    source = context.ensemble_root / (
        "tables/explanatory_pairwise_weight_correlations.csv")
    if not source.is_file():
        context.skip("figure_s04_seed_correlation.tex",
                     f"Seed-correlation input is unavailable: {source}")
        return
    data = rows(source)
    seed_columns = [name for name in data[0] if name.startswith("vine_td3_seed_")]
    if not seed_columns:
        context.skip("figure_s04_seed_correlation.tex",
                     "No seed columns occur in the weight-correlation matrix.")
        return
    entries = []
    for y, row in enumerate(data, start=1):
        for x, column in enumerate(seed_columns, start=1):
            entries.append(f"{x} {y} {number(row[column]):.10g}")
    selected = [1, 5, 10, 15, 20]
    selected = [index for index in selected if index <= len(seed_columns)]
    ticks = ",".join(str(index) for index in selected)
    labels = ",".join(seed_columns[index - 1].removeprefix("vine_td3_seed_")
                      for index in selected)
    body = r"""\begin{tikzpicture}
\begin{axis}[
  publication axis, width=0.62\linewidth, height=0.50\linewidth,
  title={Pairwise correlation of complete target-weight paths},
  xlabel={Training seed}, ylabel={Training seed},
  xmin=0.5, xmax=""" + f"{len(seed_columns)+0.5}, ymin=0.5, ymax={len(seed_columns)+0.5},\n" + r"""  xtick={""" + ticks + r"""}, ytick={""" + ticks + r"""},
  xticklabels={""" + labels + r"""}, yticklabels={""" + labels + r"""},
  y dir=reverse,
  colormap={correlationmap}{rgb255(0cm)=(150,18,42)
    rgb255(0.5cm)=(247,247,248)
    rgb255(1cm)=(10,42,78)},
  point meta min=-1, point meta max=1,
  colorbar, colorbar style={width=1.8mm, ylabel={Correlation},
    ylabel style={font=\scriptsize}, tick label style={font=\scriptsize},
    ytick={-1,-0.5,0,0.5,1}},
]
\addplot[matrix plot*, mesh/cols=""" + str(len(seed_columns)) + r""", point meta=explicit]
 table[meta index=2] {
x y meta
""" + "\n".join(entries) + r"""
};
\end{axis}
\end{tikzpicture}
"""
    context.write("figure_s04_seed_correlation.tex", body,
                  title="Pairwise seed weight-path correlation",
                  evidence_class="post_holdout_explanatory",
                  inputs=[source])
