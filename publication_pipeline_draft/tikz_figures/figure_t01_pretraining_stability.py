from __future__ import annotations

from collections import defaultdict

from .common import (
    FigureContext, FigureDataError, coordinates, number, percentile, require,
    training_rows)


METRICS = (
    ("reward", "Rolling reward"),
    ("terminal_wealth", "Rolling terminal wealth"),
    ("mean_turnover", "Rolling turnover"),
    ("mean_gross_exposure", "Rolling gross exposure"),
)


def rolling_series(group: list[dict[str, str]], metric: str,
                   window_size: int = 50) -> list[tuple[int, float]]:
    history: list[float] = []
    result: list[tuple[int, float]] = []
    for row in group:
        history.append(number(row[metric]))
        window = history[-window_size:]
        if len(window) >= 10:
            result.append((int(float(row["episode"])), sum(window) / len(window)))
    return result


def generate(context: FigureContext) -> None:
    try:
        raw, inputs = training_rows(
            context, "training_episode_metrics_all_seeds.csv")
    except Exception as error:
        if context.training_path is not None:
            raise FigureDataError(
                "Explicit training diagnostics could not produce T01: "
                f"{error}") from error
        context.skip("figure_t01_pretraining_stability.tex", str(error))
        return
    data = [row for row in raw if row.get("stage") == "pretrain"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in data:
        grouped[row.get("seed", "single_run")].append(row)
    require(len(grouped) == 20,
            f"T01 requires exactly 20 pre-training seeds; found {len(grouped)}.")
    for group in grouped.values():
        group.sort(key=lambda row: int(float(row["episode"])))

    panels = []
    for metric, title in METRICS:
        seed_series = [rolling_series(group, metric) for group in grouped.values()]
        seed_series = [series for series in seed_series if series]
        by_episode: dict[int, list[float]] = defaultdict(list)
        for series in seed_series:
            for episode, value in series:
                by_episode[episode].append(value)
        summary = [(episode, percentile(values, 0.50))
                   for episode, values in sorted(by_episode.items())]
        stride = max(1, len(summary) // 220)
        individual = []
        for series in seed_series:
            sampled = series[::max(1, len(series) // 220)]
            individual.append(
                r"\addplot[pubNavy, opacity=0.20, line width=0.35pt] coordinates {" +
                coordinates(sampled) + "};")
        panels.append(
            f"\\nextgroupplot[title={{{title}}}, xlabel={{Pre-training episode}}]\n" +
            "\n".join(individual) + "\n" +
            r"\addplot[pubNavy, ultra thick] coordinates {" +
            coordinates(summary[::stride]) + "};")

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
    context.write("figure_t01_pretraining_stability.tex", body,
                  title="Pre-training stability across 20 seeds",
                  evidence_class="training_prefix_diagnostic",
                  inputs=inputs)
