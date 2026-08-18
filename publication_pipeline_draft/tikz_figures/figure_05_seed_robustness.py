from __future__ import annotations

from .common import FigureContext, number, percentile, rows, tex


METRICS = {
    "cagr": ("CAGR (%)", 100.0, ""),
    "sharpe_ratio": ("Sharpe", 1.0, ""),
    "max_drawdown": ("Max DD (%)", 100.0, ""),
    "mean_monthly_turnover": ("Turnover", 1.0, ""),
}


def generate(context: FigureContext) -> None:
    source = context.ensemble_root / "tables/explanatory_seed_metrics.csv"
    if not source.is_file():
        context.skip(
            "figure_05_seed_robustness.tex",
            f"Optional seed-level evaluation input is unavailable: {source}")
        return
    data = [row for row in rows(source)
            if row["strategy_id"].startswith("vine_td3_seed_")]
    if len(data) != 20:
        raise ValueError(f"Expected 20 seeds, received {len(data)}.")
    panels = []
    for metric, (title, scale, unit) in METRICS.items():
        values = [scale * number(row[metric]) for row in data]
        q05, q25, median, q75, q95 = (
            percentile(values, p) for p in (0.05, 0.25, 0.50, 0.75, 0.95))
        spread = max(q95 - q05, abs(median) * 0.1, 0.01)
        ymin, ymax = q05 - 0.20 * spread, q95 + 0.20 * spread
        points = " ".join(
            f"({1 + ((i % 5)-2)*0.018:.4g},{value:.10g})"
            for i, value in enumerate(sorted(values)))
        panels.append(
            r"\nextgroupplot[" + "\n"
            f" title={{{tex(title)}}}, ymin={ymin:.8g}, ymax={ymax:.8g},\n"
            r" xmin=0.70, xmax=1.30, xtick=\empty, grid=none," + "\n"
            f" ylabel={{{unit}}}]\n"
            f"\\draw[pubSlate, thick] (axis cs:1,{q05:.10g}) -- "
            f"(axis cs:1,{q95:.10g});\n"
            f"\\draw[pubSlate, thick] (axis cs:0.94,{q05:.10g}) -- "
            f"(axis cs:1.06,{q05:.10g});\n"
            f"\\draw[pubSlate, thick] (axis cs:0.94,{q95:.10g}) -- "
            f"(axis cs:1.06,{q95:.10g});\n"
            f"\\filldraw[fill=pubNavy!18, draw=pubNavy, thick] "
            f"(axis cs:0.88,{q25:.10g}) rectangle (axis cs:1.12,{q75:.10g});\n"
            f"\\draw[pubRose, ultra thick] (axis cs:0.87,{median:.10g}) -- "
            f"(axis cs:1.13,{median:.10g});\n"
            r"\addplot[only marks, mark=*, mark size=1.0pt, pubSlate!55, opacity=0.62] "
            f"coordinates {{{points}}};")
    body = r"""\begin{tikzpicture}
\begin{groupplot}[
 group style={group size=4 by 1, horizontal sep=1.1cm},
 publication axis, width=0.23\linewidth, height=0.31\linewidth,
]
""" + "\n".join(panels) + r"""
\end{groupplot}
\end{tikzpicture}
"""
    context.write(
        "figure_05_seed_robustness.tex", body,
        title="Seed robustness distributions",
        evidence_class="frozen_confirmatory_primary_evaluation",
        inputs=[source])
