#!/usr/bin/env python3
"""Regenerate the primary wealth figure from audited complete-period returns."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PANEL = (
    ROOT.parent
    / "analysis_work/terminal_robustness_v1_review/analysis_outputs"
    / "terminal_robustness_v1/tables/normalized_monthly_evidence_panel.csv"
)
OUTPUT = ROOT / "figures/tikz/figure_01_wealth_drawdown.tex"

STRATEGIES = [
    ("equal_weight", "Equal weight", "pubGray, densely dotted, mark=none"),
    ("shrinkage_mean_variance", "Shrinkage MV", "pubSlate, dash pattern=on 2pt off 1.3pt, mark=none"),
    ("dcc_garch", "DCC--GARCH", "pubTeal, dashdotted, mark=none"),
    ("static_vine", "Static vine", "pubGold, dashed, mark=none"),
    ("rolling_vine", "Rolling vine", "pubRose, densely dashed, mark=none"),
    ("dynamic_nn_vine", "Dynamic NN-vine", "pubPurple, dash pattern=on 5pt off 1.5pt, mark=none"),
    ("vine_td3_ensemble", "NN-vine TD3", "pubNavy, ultra thick, mark=none"),
]


def fmt(value: float) -> str:
    return f"{value:.10g}"


def coordinates(points: list[tuple[str, float]]) -> str:
    return " ".join(f"({date},{fmt(value)})" for date, value in points)


def main() -> None:
    with PANEL.open(encoding="utf-8", newline="") as stream:
        source = list(csv.DictReader(stream))

    paths: dict[str, list[tuple[str, float, float]]] = {}
    for strategy_id, _, _ in STRATEGIES:
        rows = [
            row for row in source
            if row["source_id"] == "frozen_primary_oos"
            and row["strategy_id"] == strategy_id
            and row["is_complete_period"].strip().lower() == "true"
        ]
        rows.sort(key=lambda row: row["holding_end_date"])
        if len(rows) != 22:
            raise RuntimeError(f"{strategy_id}: expected 22 complete periods, found {len(rows)}")
        wealth = 100000.0
        peak = wealth
        path: list[tuple[str, float, float]] = []
        for row in rows:
            wealth *= 1.0 + float(row["net_return"])
            peak = max(peak, wealth)
            path.append((row["holding_end_date"], wealth, 100.0 * (wealth / peak - 1.0)))
        paths[strategy_id] = path

    lines = [
        "% AUTO-GENERATED FROM TERMINAL COMMON ACCOUNTING. DO NOT EDIT BY HAND.",
        "% Evidence class: frozen_confirmatory_primary_evaluation",
        "\\begin{tikzpicture}",
        "\\begin{groupplot}[",
        "  group style={group size=1 by 2, vertical sep=1.35cm,",
        "    x descriptions at=edge bottom},",
        "  publication axis,",
        "  width=\\linewidth,",
        "  date coordinates in=x,",
        "  xticklabel={\\year--\\month},",
        "  xticklabel style={rotate=30, anchor=north east},",
        "  xmin=2024-08-01, xmax=2026-06-30,",
        "]",
        "\\nextgroupplot[",
        "  height=0.32\\linewidth,",
        "  ylabel={Wealth index},",
        "  scaled y ticks=false,",
        "  ytick={100000,120000,140000,160000},",
        "  yticklabels={1,1.2,1.4,1.6},",
        "  extra description/.code={\\node[font=\\scriptsize, anchor=south west]",
        "    at (rel axis cs:0,1.01) {$\\times10^{5}$};},",
        "  title={(a) Complete-period net wealth},",
        "  legend columns=4,",
        "  legend style={column sep=10pt, row sep=1pt},",
        "  legend to name=wealthlegend,",
        "]",
    ]
    for strategy_id, _, style in STRATEGIES:
        points = [(date, wealth) for date, wealth, _ in paths[strategy_id]]
        lines.append(f"\\addplot[{style}] coordinates {{{coordinates(points)}}};")
    lines.append("\\legend{" + ",".join(label for _, label, _ in STRATEGIES) + "}")
    lines.extend([
        "\\nextgroupplot[",
        "  height=0.23\\linewidth,",
        "  ylabel={Drawdown (\\%)}, xlabel={Holding-period end},",
        "  title={(b) Drawdown from running peak},",
        "  ymax=0,",
        "]",
    ])
    for strategy_id, _, style in STRATEGIES:
        points = [(date, drawdown) for date, _, drawdown in paths[strategy_id]]
        lines.append(f"\\addplot[{style}] coordinates {{{coordinates(points)}}};")
    lines.extend([
        "\\end{groupplot}",
        "\\node[anchor=north] at ($(current bounding box.south)+(0,-2mm)$)",
        "  {\\pgfplotslegendfromname{wealthlegend}};",
        "\\end{tikzpicture}",
        "",
    ])
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")

    td3 = paths["vine_td3_ensemble"]
    terminal = td3[-1][1]
    max_drawdown = -min(point[2] for point in td3)
    if abs(terminal - 159668.8711) > 0.1 or abs(max_drawdown - 6.64) > 0.02:
        raise RuntimeError(
            f"terminal audit mismatch: wealth={terminal:.4f}, drawdown={max_drawdown:.4f}%"
        )
    print(f"Wrote {OUTPUT}; TD3 wealth={terminal:.4f}, monthly MDD={max_drawdown:.4f}%")


if __name__ == "__main__":
    main()
