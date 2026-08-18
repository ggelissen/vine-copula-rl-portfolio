from __future__ import annotations

from .common import FigureContext, number, rows, tex


def generate(context: FigureContext) -> None:
    source = context.focused_root / "focused_walk_forward_contrasts.csv"
    window_source = (
        context.focused_root / "focused_walk_forward_window_effects.csv")
    if not source.is_file():
        context.skip(
            "figure_16_focused_walk_forward.tex",
            "Focused walk-forward results do not exist yet; no placeholder data were generated.")
        return
    if not window_source.is_file():
        context.skip(
            "figure_16_focused_walk_forward.tex",
            "Focused per-window effects are missing; regenerate the focused analysis.")
        return

    data = rows(source)
    required = {
        "label", "annualized_ce_difference",
        "annualized_ce_ci_lower", "annualized_ce_ci_upper",
    }
    if not required <= set(data[0]):
        context.skip(
            "figure_16_focused_walk_forward.tex",
            "Focused result schema is not compatible with the final figure interface.")
        return
    window_data = [
        row for row in rows(window_source)
        if row["comparison_family"] == "mechanism"
    ]
    windows = sorted({row["window_id"] for row in window_data})
    if len(windows) != 2:
        context.skip(
            "figure_16_focused_walk_forward.tex",
            "Focused figure requires exactly two non-overlapping windows.")
        return

    window_styles = [
        "pubTeal, mark=*, mark options={solid,fill=pubTeal}",
        "pubGold, mark=triangle*, mark options={solid,fill=pubGold}",
    ]
    plots: list[str] = []
    labels: list[str] = []
    for y, row in enumerate(data, start=1):
        estimate = 100 * number(row["annualized_ce_difference"])
        lower = 100 * number(row["annualized_ce_ci_lower"])
        upper = 100 * number(row["annualized_ce_ci_upper"])
        plots.append(
            f"\\addplot[pubNavy, thick, mark=|, "
            f"mark options={{solid,scale=1.2}}, forget plot] "
            f"coordinates {{({lower:.10g},{y}) ({upper:.10g},{y})}};\n"
            f"\\addplot[publication point, pubNavy, fill=pubNavy, forget plot] "
            f"coordinates {{({estimate:.10g},{y})}};")
        for window_index, window_id in enumerate(windows):
            matches = [
                item for item in window_data
                if item["window_id"] == window_id and
                item["label"] == row["label"]
            ]
            if len(matches) != 1:
                context.skip(
                    "figure_16_focused_walk_forward.tex",
                    f"Missing focused effect for {row['label']} in {window_id}.")
                return
            effect = 100 * number(
                matches[0][
                    "difference_annualized_certainty_equivalent_return"])
            offset = -0.13 if window_index == 0 else 0.13
            plots.append(
                f"\\addplot[{window_styles[window_index]}, forget plot] "
                f"coordinates {{({effect:.10g},{y + offset:.3g})}};")
        labels.append("{" + tex(row["label"]) + "}")

    body = r"""\begin{tikzpicture}
\begin{axis}[
  publication axis,
  width=0.88\linewidth,
  height=0.38\linewidth,
  xlabel={Annual CRRA CE effect (percentage points)}, ylabel={},
  ytick={""" + ",".join(str(i) for i in range(1, len(data) + 1)) + r"""},
  yticklabels={""" + ",".join(labels) + r"""}, y dir=reverse,
  ymin=0.45, ymax=""" + f"{len(data) + 0.55}" + r""",
  legend style={at={(0.99,0.03)},anchor=south east,draw=none,
    fill=none,legend columns=3,
    /tikz/every even column/.append style={column sep=0.7em}},
  title={Focused two-window mechanism robustness},
]
\addplot[pubSlate, thin, forget plot] coordinates {(0,0.5) (0,""" + (
        f"{len(data) + 0.5}") + r""")};
\addlegendimage{publication point,pubNavy,fill=pubNavy}
\addlegendentry{Pooled effect with 95\% CI}
\addlegendimage{pubTeal,mark=*,mark options={solid,fill=pubTeal}}
\addlegendentry{Window 1}
\addlegendimage{pubGold,mark=triangle*,mark options={solid,fill=pubGold}}
\addlegendentry{Window 2}
""" + "\n".join(plots) + r"""
\end{axis}
\end{tikzpicture}
"""
    context.write(
        "figure_16_focused_walk_forward.tex", body,
        title="Focused walk-forward mechanism robustness",
        evidence_class="retrospective_walk_forward",
        inputs=[source, window_source])
