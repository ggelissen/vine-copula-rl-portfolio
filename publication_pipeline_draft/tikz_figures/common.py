from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import tarfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable


class FigureDataError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FigureDataError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Required figure input is missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        result = list(csv.DictReader(stream))
    require(bool(result), f"Figure input is empty: {path}")
    return result


def archive_rows(archive: Path, suffix: str) -> list[dict[str, str]]:
    require(archive.is_file(), f"Required evidence archive is missing: {archive}")
    with tarfile.open(archive, "r:gz") as bundle:
        matches = [item for item in bundle.getmembers()
                   if item.isfile() and item.name.endswith(suffix)]
        require(len(matches) == 1,
                f"Archive member is absent or duplicated: {suffix}")
        handle = bundle.extractfile(matches[0])
        require(handle is not None, f"Could not read archive member: {suffix}")
        text = handle.read().decode("utf-8-sig")
    result = list(csv.DictReader(io.StringIO(text)))
    require(bool(result), f"Archive CSV is empty: {suffix}")
    return result


def training_rows(context: "FigureContext", filename: str) -> tuple[list[dict[str, str]], list[Path | str]]:
    """Read an aggregated training diagnostic from a directory or tar archive.

    Accepting the immutable 20-seed archive directly prevents an accidental
    fallback to a nearby single-seed CSV with the same base name.
    """
    root = context.training_root
    if root.is_file():
        require(tarfile.is_tarfile(root),
                f"Training diagnostics must be a directory or tar archive: {root}")
        member = f"raw/{filename}"
        return archive_rows(root, member), [root, member]
    candidates = (root / "raw" / filename, root / filename)
    source = next((path for path in candidates if path.is_file()), candidates[0])
    require(source.is_file(), f"Aggregated training diagnostic is missing: {source}")
    return rows(source), [source]


def number(value: str | float | int | None) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise FigureDataError(f"Expected finite number, received {value!r}") from error
    require(math.isfinite(result), f"Expected finite number, received {value!r}")
    return result


def boolean(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def tex(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def coordinate(x: Any, y: float) -> str:
    return f"({x},{y:.10g})"


def coordinates(points: Iterable[tuple[Any, float]]) -> str:
    return " ".join(coordinate(x, y) for x, y in points)


def date_coordinates(points: Iterable[tuple[str, float]]) -> str:
    return " ".join(f"({x},{y:.10g})" for x, y in points)


def percentile(values: list[float], probability: float) -> float:
    require(bool(values), "Cannot calculate a percentile of an empty sample.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def cumulative_wealth(returns: Iterable[float], initial: float = 1.0) -> list[float]:
    wealth: list[float] = []
    current = initial
    for value in returns:
        require(value > -1, "A net return is not above -100%.")
        current *= 1 + value
        wealth.append(current)
    return wealth


def drawdowns(wealth: Iterable[float]) -> list[float]:
    result: list[float] = []
    peak = 1.0
    for value in wealth:
        peak = max(peak, value)
        result.append(value / peak - 1)
    return result


@dataclass
class FigureContext:
    repo: Path
    output: Path
    main_path: Path | None = None
    ensemble_path: Path | None = None
    causal_path: Path | None = None
    reconciliation_path: Path | None = None
    synthetic_path: Path | None = None
    training_path: Path | None = None
    focused_path: Path | None = None
    generated: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)

    @property
    def main_root(self) -> Path:
        return self.main_path or self.repo / (
            "analysis_outputs/oos_v4_verified_770d2944/"
            "main_oos_v4_operational_retry/publication_results")

    @property
    def ensemble_root(self) -> Path:
        return self.ensemble_path or self.repo / (
            "analysis_outputs/oos_v4_verified_770d2944/"
            "post_holdout_ensemble_mechanism_v2")

    @property
    def causal_archive(self) -> Path:
        return self.causal_path or self.repo / (
            "frozen_releases/final_evidence/"
            "causal_results_v2_v3_v4_plot_runtime_v1.tar.gz")

    @property
    def reconciliation_root(self) -> Path:
        return self.reconciliation_path or self.repo / (
            "analysis_outputs/post_hoc_compressed_vine_benchmark_reconciliation_v1")

    @property
    def synthetic_root(self) -> Path:
        return self.synthetic_path or self.repo / "data/synthetic_diagnostics"

    @property
    def training_root(self) -> Path:
        return self.training_path or self.repo / "analysis_outputs/training_diagnostics"

    @property
    def focused_root(self) -> Path:
        return self.focused_path or self.repo / (
            "analysis_outputs/focused_walk_forward_mechanisms_v1")

    def write(self, filename: str, body: str, *, title: str,
              evidence_class: str, inputs: Iterable[Path | str]) -> Path:
        target = self.output / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        normalized_inputs: list[dict[str, str]] = []
        for item in inputs:
            if isinstance(item, Path):
                resolved = item.resolve()
                try:
                    display = resolved.relative_to(self.repo.resolve()).as_posix()
                except ValueError:
                    # Preserve provenance without embedding workstation-specific
                    # home-directory paths in a public manuscript artifact.
                    display = f"external_input/{resolved.name}"
                normalized_inputs.append({
                    "path": display, "sha256": sha256(item)})
            else:
                normalized_inputs.append({"archive_member": item})
        header = (
            "% AUTO-GENERATED. DO NOT EDIT BY HAND.\n"
            f"% Figure: {title}\n"
            f"% Evidence class: {evidence_class}\n"
            "% Regenerate with publication_pipeline_draft/generate_publication_tikz.py.\n"
        )
        target.write_text(header + body.rstrip() + "\n", encoding="utf-8")
        self.generated.append({
            "file": filename, "title": title,
            "evidence_class": evidence_class,
            "sha256": sha256(target), "inputs": normalized_inputs,
        })
        return target

    def skip(self, figure: str, reason: str) -> None:
        self.skipped.append({"figure": figure, "reason": reason})


STRATEGY_LABELS = {
    "equal_weight": "Equal weight",
    "shrinkage_mean_variance": "Shrinkage MV",
    "dcc_garch": "DCC--GARCH",
    "static_vine": "Static vine",
    "rolling_vine": "Rolling vine",
    "dynamic_nn_vine": "Dynamic NN-vine",
    "vine_td3_ensemble": "NN-vine TD3",
}


STRATEGY_STYLES = {
    "equal_weight": "pubGray, densely dotted, mark=none",
    "shrinkage_mean_variance": "pubSlate, dash pattern=on 2pt off 1.3pt, mark=none",
    "dcc_garch": "pubTeal, dashdotted, mark=none",
    "static_vine": "pubGold, dashed, mark=none",
    "rolling_vine": "pubRose, densely dashed, mark=none",
    "dynamic_nn_vine": "pubPurple, dash pattern=on 5pt off 1.5pt, mark=none",
    "vine_td3_ensemble": "pubNavy, ultra thick, mark=none",
}


def write_manifest(context: FigureContext) -> Path:
    manifest = {
        "schema_version": 1,
        "status": "publication_tikz_generated",
        "generator": "publication_pipeline_draft/generate_publication_tikz.py",
        "figure_count": len(context.generated),
        "skipped_count": len(context.skipped),
        "generated": context.generated,
        "skipped": context.skipped,
        "scientific_note": (
            "TikZ files are presentation transforms of frozen inputs; they do not "
            "recompute or alter statistical results."),
    }
    path = context.output / "figure_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    contents = [path, *(context.output / item["file"]
                        for item in context.generated)]
    style = context.output / "tikz_preamble.tex"
    if style.is_file():
        contents.append(style)
    for auxiliary_name in ("README.md", "preview_all_figures.tex"):
        auxiliary = context.output / auxiliary_name
        if auxiliary.is_file():
            contents.append(auxiliary)
    checksum = context.output / "CONTENTS.sha256"
    checksum.write_text("\n".join(
        f"{sha256(item)}  {item.relative_to(context.output).as_posix()}"
        for item in sorted(contents)
    ) + "\n", encoding="ascii")
    return path
