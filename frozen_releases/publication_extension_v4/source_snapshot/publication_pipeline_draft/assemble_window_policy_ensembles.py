#!/usr/bin/env python3
"""Validate 50 policy logs and build five preregistered seed ensembles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path


class EnsembleError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream); rows = list(reader)
        return list(reader.fieldnames or []), rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise EnsembleError(f"Output already exists: {args.output}")
        _, inventory = read(args.inventory)
        if len(inventory) != 50:
            raise EnsembleError("Exactly 50 policy logs are required.")
        grouped: dict[str, list[tuple[dict[str, str], Path]]] = defaultdict(list)
        canonical_keys = canonical_weights = None
        weight_columns: list[str] = []
        normalized: list[dict[str, str]] = []
        for item in inventory:
            path = Path(item["weight_file"])
            if not path.is_file() or sha256(path) != item["sha256"]:
                raise EnsembleError(f"Policy log hash mismatch: {path}")
            fields, rows = read(path)
            weights = [name for name in fields if name.startswith("w_")]
            keys = [name for name in fields if name in {
                "window_id", "decision_date", "holding_end_date"}]
            if len(rows) != 24 or len(weights) < 2 or len(keys) != 3:
                raise EnsembleError(f"Invalid policy log schema: {path}")
            current_keys = [[row[name] for name in keys] for row in rows]
            if canonical_keys is None:
                canonical_keys, weight_columns = current_keys, weights
            if current_keys != canonical_keys or weights != weight_columns:
                raise EnsembleError("Policy dates or asset ordering differ.")
            for row in rows:
                vector = [float(row[name]) for name in weights]
                if (abs(sum(vector) - 1) > 1e-6 or
                        sum(abs(value) for value in vector) > 1.5 + 1e-6 or
                        max(vector) > 0.6 + 1e-6 or min(vector) < -0.2 - 1e-6):
                    raise EnsembleError(f"Policy constraint failure: {path}")
            grouped[item["algorithm"]].append((item, path))
            normalized.append({"strategy_id": f"{item['algorithm']}_seed_{item['seed']}",
                               "strategy_class": "individual_rl_seed",
                               "algorithm": item["algorithm"], "seed": item["seed"],
                               "weight_file": str(path), "sha256": item["sha256"]})
        if set(grouped) != {"td3", "ddpg", "sac", "ppo", "a2c"} or \
                any(len(value) != 10 for value in grouped.values()):
            raise EnsembleError("Algorithms do not have ten matched seeds each.")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.",
                                          dir=args.output.parent))
        try:
            for algorithm, members in sorted(grouped.items()):
                member_rows = [read(path)[1] for _, path in members]
                ensemble_rows = []
                for index in range(24):
                    row = {name: canonical_keys[index][position]
                           for position, name in enumerate(
                               ("window_id", "decision_date", "holding_end_date"))}
                    for name in weight_columns:
                        row[name] = sum(float(rows[index][name])
                                        for rows in member_rows) / len(member_rows)
                    ensemble_rows.append(row)
                path = temporary / f"weights_{algorithm}_ensemble10.csv"
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=[
                        "window_id", "decision_date", "holding_end_date",
                        *weight_columns])
                    writer.writeheader(); writer.writerows(ensemble_rows)
                final_path = args.output.resolve() / path.name
                normalized.append({"strategy_id": f"{algorithm}_ensemble10",
                                   "strategy_class": "preregistered_rl_ensemble",
                                   "algorithm": algorithm, "seed": "",
                                   "weight_file": str(final_path), "sha256": sha256(path)})
            inventory_path = temporary / "rl_strategy_inventory.csv"
            with inventory_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(normalized[0]))
                writer.writeheader(); writer.writerows(normalized)
            manifest = {"schema_version": 1,
                        "status": "validated_policy_ensembles",
                        "individual_policy_count": 50,
                        "ensemble_count": 5,
                        "ensemble_rule": "arithmetic_mean_target_weights",
                        "inventory_sha256": sha256(inventory_path),
                        "confirmatory_claim_permitted": False}
            (temporary / "ensemble_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            os.replace(temporary, args.output)
            print(json.dumps(manifest, indent=2, sort_keys=True)); return 0
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True); raise
    except (OSError, ValueError, KeyError, EnsembleError) as error:
        print(f"WINDOW ENSEMBLE FAILURE: {error}"); return 1


if __name__ == "__main__":
    raise SystemExit(main())
