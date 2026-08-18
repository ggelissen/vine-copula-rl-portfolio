from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


class TerminalPublicationError(RuntimeError):
    """Raised when the immutable terminal evidence cannot be transformed safely."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TerminalPublicationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Required terminal publication input is missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        result = list(csv.DictReader(stream))
    require(bool(result), f"Terminal publication input is empty: {path}")
    return result


def finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TerminalPublicationError(f"Expected a finite number, received {value!r}") from error
    require(math.isfinite(result), f"Expected a finite number, received {value!r}")
    return result


def tex(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character)
                   for character in str(value))


def pct(value: Any, digits: int = 2, *, signed: bool = False) -> str:
    number = 100 * finite(value)
    format_spec = f"+.{digits}f" if signed else f".{digits}f"
    return f"{number:{format_spec}}\\%"


def pp(value: Any, digits: int = 2, *, signed: bool = True) -> str:
    number = 100 * finite(value)
    format_spec = f"+.{digits}f" if signed else f".{digits}f"
    return f"{number:{format_spec}}"


def format_p(value: Any) -> str:
    number = finite(value)
    if number < 0.001:
        return "$<0.001$"
    return f"{number:.3f}"


def verify_contents(root: Path) -> None:
    checksum = root / "CONTENTS.sha256"
    require(checksum.is_file(), f"Terminal results checksum is missing: {checksum}")
    for line in checksum.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        target = root / relative.strip().lstrip("*")
        require(target.is_file(), f"Terminal results member is missing: {target}")
        require(sha256(target) == expected.lower(),
                f"Terminal results checksum mismatch: {target}")


STRATEGY_LABELS = {
    "equal_weight": "Equal weight",
    "shrinkage_mean_variance": "Shrinkage MV",
    "dcc_garch": "DCC--GARCH",
    "static_vine": "Static vine",
    "rolling_vine": "Rolling vine",
    "dynamic_nn_vine": "Dynamic NN-vine",
    "vine_td3_ensemble": "NN-vine TD3",
    "masked_historical_prefix_1000_presentations_ensemble": "Historical prefix",
    "masked_moving_block_bootstrap_1000_presentations_ensemble": "Moving-block bootstrap",
    "synthetic_100_unique_1000_presentations_no_policy_visible_dependence_ensemble": "Concentrated vine synthetic",
}


CONTRAST_LABELS = {
    "primary_vs_equal_weight": "TD3 vs equal weight",
    "primary_vs_static_vine": "TD3 vs static vine",
    "primary_vs_dynamic_nn_vine": "TD3 vs dynamic NN-vine",
    "raw_vine_state_contribution": "Raw vine state",
    "joint_visible_dependence_contribution": "Joint visible dependence",
    "focused_raw_vine_state_contribution": "Focused raw vine state",
    "focused_joint_visible_dependence_contribution": "Focused visible dependence",
    "historical_vs_original_bootstrap": "Historical vs original bootstrap",
    "concentrated_synthetic_vs_full_state": "Masked vs full-state synthetic",
    "concentrated_synthetic_vs_historical_masked": "Synthetic vs historical",
    "concentrated_synthetic_vs_bootstrap_masked": "Synthetic vs bootstrap",
}


EVIDENCE_LABELS = {
    "frozen_primary_benchmarks": "Frozen primary",
    "post_holdout_causal_components": "Post-holdout",
    "retrospective_focused_mechanisms": "Retrospective",
    "post_holdout_pretraining_sources": "Post-holdout",
    "post_holdout_presentation_mechanism": "Post-holdout",
    "post_holdout_terminal_pretraining_controls": "Post-holdout",
}


@dataclass
class PublicationContext:
    repo: Path
    terminal_root: Path
    output: Path
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tables_root(self) -> Path:
        return self.terminal_root / "tables"

    def input(self, name: str) -> Path:
        return self.tables_root / name

    def rows(self, name: str) -> list[dict[str, str]]:
        return csv_rows(self.input(name))

    def register(self, path: Path, *, artifact_type: str, title: str,
                 evidence_class: str, inputs: Iterable[Path] = ()) -> None:
        self.artifacts.append({
            "file": path.relative_to(self.output).as_posix(),
            "artifact_type": artifact_type,
            "title": title,
            "evidence_class": evidence_class,
            "sha256": sha256(path),
            "inputs": [
                {"path": item.relative_to(self.repo).as_posix()
                 if item.is_relative_to(self.repo) else f"external_input/{item.name}",
                 "sha256": sha256(item)}
                for item in inputs
            ],
        })

    def write_text(self, relative: str, body: str, *, artifact_type: str,
                   title: str, evidence_class: str,
                   inputs: Iterable[Path] = ()) -> Path:
        target = self.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body.rstrip() + "\n", encoding="utf-8")
        self.register(target, artifact_type=artifact_type, title=title,
                      evidence_class=evidence_class, inputs=inputs)
        return target

    def write_csv(self, relative: str, rows: list[dict[str, Any]], *,
                  artifact_type: str, title: str, evidence_class: str,
                  inputs: Iterable[Path] = ()) -> Path:
        require(bool(rows), f"Refusing to write empty publication table: {relative}")
        target = self.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.register(target, artifact_type=artifact_type, title=title,
                      evidence_class=evidence_class, inputs=inputs)
        return target

    def write_figure(self, filename: str, body: str, *, title: str,
                     evidence_class: str, inputs: Iterable[Path]) -> Path:
        header = (
            "% AUTO-GENERATED. DO NOT EDIT BY HAND.\n"
            f"% Figure: {title}\n"
            f"% Evidence class: {evidence_class}\n"
            "% Regenerate with generate_terminal_publication_artifacts.py.\n"
        )
        return self.write_text(
            f"figures/tikz/{filename}", header + body,
            artifact_type="tikz_figure", title=title,
            evidence_class=evidence_class, inputs=inputs)


def unique_row(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = [row for row in rows
               if all(row.get(key) == value for key, value in criteria.items())]
    require(len(matches) == 1,
            f"Expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def write_manifest(context: PublicationContext, terminal_manifest: dict[str, Any]) -> Path:
    manifest = {
        "schema_version": 1,
        "status": "terminal_publication_artifacts_generated",
        "additive_only": True,
        "existing_publication_artifacts_modified": False,
        "source_analysis_id": terminal_manifest["analysis_id"],
        "source_contract_sha256": terminal_manifest["contract_sha256"],
        "source_release_contents_sha256": terminal_manifest["release_contents_sha256"],
        "confirmatory_claim_created": False,
        "artifact_count": len(context.artifacts),
        "artifacts": context.artifacts,
    }
    target = context.output / "publication_artifact_manifest.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    files = [target, *(context.output / item["file"] for item in context.artifacts)]
    checksum = context.output / "CONTENTS.sha256"
    checksum.write_text("\n".join(
        f"{sha256(path)}  {path.relative_to(context.output).as_posix()}"
        for path in sorted(set(files))
    ) + "\n", encoding="ascii")
    return target
