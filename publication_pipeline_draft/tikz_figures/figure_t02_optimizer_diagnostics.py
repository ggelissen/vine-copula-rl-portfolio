from __future__ import annotations

from collections import defaultdict

from .common import (
    FigureContext, FigureDataError, coordinates, number, percentile, require,
    training_rows)


METRICS = (
    ("critic_loss", "Critic loss"),
    ("actor_loss", "Actor loss"),
    ("twin_q_gap", "Twin-Q gap"),
    ("actor_grad_norm", "Actor gradient norm"),
)


def rolling(values: list[tuple[int, float]], window_size: int = 10) -> list[tuple[int, float]]:
    result: list[tuple[int, float]] = []
    history: list[float] = []
    for update, value in values:
        history.append(value)
        window = history[-window_size:]
        result.append((update, sum(window) / len(window)))
    return result


def generate(context: FigureContext) -> None:
    try:
        raw, inputs = training_rows(
            context, "training_update_metrics_all_seeds.csv")
    except Exception as error:
        if context.training_path is not None:
            raise FigureDataError(
                "Explicit training diagnostics could not produce T02: "
                f"{error}") from error
        context.skip("figure_t02_optimizer_diagnostics.tex", str(error))
        return
    data = [row for row in raw if row.get("stage") == "pretrain"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in data:
        grouped[row.get("seed", "single_run")].append(row)
    require(len(grouped) == 20,
            f"T02 requires exactly 20 pre-training seeds; found {len(grouped)}.")
    for group in grouped.values():
        group.sort(key=lambda row: int(float(row["update"])))

    panels = []
    for metric, title in METRICS:
        seed_series: list[list[tuple[int, float]]] = []
        by_update: dict[int, list[float]] = defaultdict(list)
        for group in grouped.values():
            series = rolling([(int(float(row["update"])), number(row[metric]))
                              for row in group])
            seed_series.append(series)
            for update, value in series:
                by_update[update].append(value)
        median = [(update, percentile(values, 0.50))
                  for update, values in sorted(by_update.items())]
        individual = [
            r"\addplot[pubRose, opacity=0.18, line width=0.35pt] coordinates {" +
            coordinates(series) + "};" for series in seed_series]
        panels.append(
            f"\\nextgroupplot[title={{{title}}}, xlabel={{Gradient update}}, scaled y ticks=true]\n" +
            "\n".join(individual) + "\n" +
            r"\addplot[pubRose, ultra thick] coordinates {" +
            coordinates(median) + "};")
    body = r"""\begin{tikzpicture}
\begin{groupplot}[
 group style={group size=2 by 2, horizontal sep=1.35cm, vertical sep=2.25cm},
 publication axis, width=0.49\linewidth, height=0.3\linewidth,
 title style={font=\footnotesize\bfseries, text=pubInk, yshift=1.2mm},
]
""" + "\n".join(panels) + r"""
\end{groupplot}
\end{tikzpicture}
"""
    context.write("figure_t02_optimizer_diagnostics.tex", body,
                  title="Optimizer diagnostics across 20 seeds",
                  evidence_class="training_prefix_diagnostic",
                  inputs=inputs)
