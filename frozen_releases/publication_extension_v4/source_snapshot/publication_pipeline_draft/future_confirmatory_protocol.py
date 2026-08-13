#!/usr/bin/env python3
"""Validate and execute a preregistered future confirmatory evaluation.

This module is intentionally separate from the consumed v4 evaluator.  It is a
small orchestration and audit layer for *new* non-overlapping external or
walk-forward test periods.  It never trains, selects, or tunes a model.  Model
selection, code, configuration, data snapshots, package locks, and checkpoints
must already be frozen and independently hashed before the first test command.

The executor creates an access ledger atomically, preserves both successful and
failed runs, refuses to overwrite outputs, and validates the economic accounting
and solver audits emitted by each test command.  A failed test execution is
evidence; changing the output path is not permission to retry the same sample.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONTAINER_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
PLACEHOLDER_RE = re.compile(r"^\{(artifact|output):([^{}]+)\}$")
ARTIFACT_CATEGORIES = ("code", "config", "data", "model", "environment")
DATA_ROLES = ("train", "validation", "test")
REQUIRED_ECONOMIC_RULES = {
    "turnover_definition": "sum_abs_target_minus_pretrade_drifted_weights",
    "pretrade_weight_drift": "post_return_self_financing",
    "initial_position": "equal_weight",
    "financing_proration": "actual_calendar_days_over_day_count_basis",
    "annualization_convention": "actual_elapsed_years_v1",
    "net_return_accounting": "log_gross_minus_transaction_and_financing_costs",
}


class ProtocolError(RuntimeError):
    """A fail-closed confirmatory-protocol violation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"JSON file not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"Cannot parse JSON file {path}: {error}") from error
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def parse_date(value: Any, field: str) -> dt.date:
    require(isinstance(value, str), f"{field} must be an ISO date string.")
    try:
        return dt.date.fromisoformat(value)
    except ValueError as error:
        raise ProtocolError(f"{field} is not a valid ISO date: {value}") from error


def parse_utc_timestamp(value: Any, field: str) -> dt.datetime:
    require(isinstance(value, str), f"{field} must be an ISO timestamp.")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ProtocolError(f"{field} is not a valid ISO timestamp: {value}") from error
    require(parsed.tzinfo is not None, f"{field} must include a UTC offset.")
    require(parsed.utcoffset() == dt.timedelta(0), f"{field} must be UTC.")
    return parsed


def finite_number(value: Any, field: str, minimum: float | None = None) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{field} must be numeric.")
    result = float(value)
    require(math.isfinite(result), f"{field} must be finite.")
    if minimum is not None:
        require(result >= minimum, f"{field} must be at least {minimum}.")
    return result


def safe_relative_path(value: Any, field: str) -> PurePosixPath:
    require(isinstance(value, str) and value.strip(), f"{field} must be a relative path.")
    require("\\" not in value, f"{field} must use portable '/' separators.")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts,
            f"{field} must remain below the declared root: {value}")
    return path


def materialize(root: Path, relative: PurePosixPath) -> Path:
    return root.joinpath(*relative.parts)


def require_keys(mapping: dict[str, Any], keys: Iterable[str], field: str) -> None:
    missing = [key for key in keys if key not in mapping]
    require(not missing, f"{field} is missing fields: {', '.join(missing)}")


def validate_consumed_holdouts(contract: dict[str, Any]) -> dt.date:
    rows = contract.get("consumed_holdouts")
    require(isinstance(rows, list) and rows,
            "consumed_holdouts must contain every previously accessed holdout.")
    seen: set[str] = set()
    maximum_end: dt.date | None = None
    for index, row in enumerate(rows):
        field = f"consumed_holdouts[{index}]"
        require(isinstance(row, dict), f"{field} must be an object.")
        require_keys(row, ("holdout_id", "start", "end", "result_sha256"), field)
        holdout_id = row["holdout_id"]
        require(isinstance(holdout_id, str) and holdout_id and holdout_id not in seen,
                f"{field}.holdout_id must be unique and non-empty.")
        seen.add(holdout_id)
        start = parse_date(row["start"], f"{field}.start")
        end = parse_date(row["end"], f"{field}.end")
        require(start < end, f"{field} must have start < end.")
        require(isinstance(row["result_sha256"], str)
                and SHA256_RE.fullmatch(row["result_sha256"]) is not None,
                f"{field}.result_sha256 must be a lowercase SHA-256 digest.")
        maximum_end = end if maximum_end is None else max(maximum_end, end)
    assert maximum_end is not None
    return maximum_end


def validate_assets_and_economics(contract: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    assets = contract.get("assets")
    require(isinstance(assets, list) and len(assets) >= 2,
            "assets must be a fixed ordered list with at least two entries.")
    require(all(isinstance(asset, str) and asset for asset in assets),
            "Every asset identifier must be a non-empty string.")
    require(len(set(assets)) == len(assets), "Asset identifiers must be unique.")

    economics = contract.get("economics")
    require(isinstance(economics, dict), "economics must be an object.")
    require_keys(
        economics,
        (
            "net_exposure", "gross_leverage", "max_long_weight",
            "max_short_weight", "turnover_cost", "annual_short_borrow_rate",
            "annual_cash_borrow_rate", "day_count_basis", "weight_tolerance",
            "metric_tolerance", *REQUIRED_ECONOMIC_RULES.keys(),
        ),
        "economics",
    )
    net = finite_number(economics["net_exposure"], "economics.net_exposure")
    gross = finite_number(economics["gross_leverage"], "economics.gross_leverage", 0)
    require(gross >= abs(net), "gross_leverage must be at least abs(net_exposure).")
    finite_number(economics["max_long_weight"], "economics.max_long_weight", 0)
    finite_number(economics["max_short_weight"], "economics.max_short_weight", 0)
    for name in ("turnover_cost", "annual_short_borrow_rate", "annual_cash_borrow_rate"):
        finite_number(economics[name], f"economics.{name}", 0)
    basis = int(finite_number(economics["day_count_basis"], "economics.day_count_basis", 1))
    require(basis in (360, 365, 366), "day_count_basis must be explicitly 360, 365, or 366.")
    finite_number(economics["weight_tolerance"], "economics.weight_tolerance", 0)
    finite_number(economics["metric_tolerance"], "economics.metric_tolerance", 0)
    for name, expected in REQUIRED_ECONOMIC_RULES.items():
        require(economics.get(name) == expected,
                f"economics.{name} must equal '{expected}'.")
    return list(assets), economics


def validate_selection(contract: dict[str, Any]) -> dict[str, Any]:
    selection = contract.get("selection")
    require(isinstance(selection, dict), "selection must be an object.")
    require_keys(
        selection,
        (
            "primary_strategy_id", "primary_benchmark_id", "primary_outcome",
            "alpha", "multiplicity_control", "selection_data_scope",
            "test_data_role", "test_feedback_policy", "hyperparameter_lock_artifact_id",
            "selection_manifest_artifact_id", "selection_locked_before_all_tests",
        ),
        "selection",
    )
    require(selection["selection_data_scope"] == "train_and_validation_only",
            "Model selection may use only train and validation data.")
    require(selection["test_data_role"] == "evaluation_only",
            "Test data must have evaluation_only role.")
    require(
        selection["test_feedback_policy"]
        == "state_updates_only_no_hyperparameter_or_model_selection",
        "Earlier test observations may update state only, never selection or hyperparameters.",
    )
    require(selection["selection_locked_before_all_tests"] is True,
            "Selection must be locked before every confirmatory test window.")
    alpha = finite_number(selection["alpha"], "selection.alpha")
    require(0 < alpha < 0.5, "selection.alpha must lie between 0 and 0.5.")
    for name in ("primary_strategy_id", "primary_benchmark_id", "primary_outcome",
                 "multiplicity_control", "hyperparameter_lock_artifact_id",
                 "selection_manifest_artifact_id"):
        require(isinstance(selection[name], str) and selection[name],
                f"selection.{name} must be non-empty.")
    return selection


def validate_windows(contract: dict[str, Any], consumed_end: dt.date) -> list[dict[str, Any]]:
    windows = contract.get("windows")
    require(isinstance(windows, list) and len(windows) >= 2,
            "At least two non-overlapping future test windows are required.")
    seen: set[str] = set()
    intervals: list[tuple[dt.date, dt.date, str]] = []
    forbidden_overrides = {"assets", "economics", "costs", "hyperparameters"}
    for index, window in enumerate(windows):
        field = f"windows[{index}]"
        require(isinstance(window, dict), f"{field} must be an object.")
        require_keys(
            window,
            (
                "window_id", "train_start", "train_end", "validation_start",
                "validation_end", "test_start", "test_end", "periods",
                "train_data_artifact_id", "validation_data_artifact_id",
                "test_data_artifact_id", "model_artifact_id",
            ),
            field,
        )
        require(not forbidden_overrides.intersection(window),
                f"{field} cannot override fixed assets, economics, costs, or hyperparameters.")
        window_id = window["window_id"]
        require(isinstance(window_id, str) and window_id and window_id not in seen,
                f"{field}.window_id must be unique and non-empty.")
        seen.add(window_id)
        train_start = parse_date(window["train_start"], f"{field}.train_start")
        train_end = parse_date(window["train_end"], f"{field}.train_end")
        validation_start = parse_date(window["validation_start"], f"{field}.validation_start")
        validation_end = parse_date(window["validation_end"], f"{field}.validation_end")
        test_start = parse_date(window["test_start"], f"{field}.test_start")
        test_end = parse_date(window["test_end"], f"{field}.test_end")
        require(train_start < train_end < validation_start <= validation_end <= test_start < test_end,
                f"{field} violates train < validation < test chronology.")
        require(test_start >= consumed_end,
                f"{field} overlaps the consumed holdout ending {consumed_end}.")
        periods = window["periods"]
        require(isinstance(periods, list) and periods,
                f"{field}.periods must preregister every decision and holding-end date.")
        previous_end: dt.date | None = None
        for period_index, period in enumerate(periods):
            period_field = f"{field}.periods[{period_index}]"
            require(isinstance(period, dict), f"{period_field} must be an object.")
            require_keys(period, ("decision_date", "holding_end_date"), period_field)
            decision = parse_date(period["decision_date"], f"{period_field}.decision_date")
            holding_end = parse_date(period["holding_end_date"], f"{period_field}.holding_end_date")
            require(decision < holding_end, f"{period_field} must have decision < holding_end.")
            require(test_start <= decision and holding_end <= test_end,
                    f"{period_field} falls outside its fixed test window.")
            if previous_end is None:
                require(decision == test_start,
                        f"{period_field} must begin exactly at test_start.")
            else:
                require(decision == previous_end,
                        f"{period_field} must continue from the previous holding_end_date.")
            previous_end = holding_end
        require(previous_end == test_end,
                f"{field}.periods must end exactly at test_end.")
        intervals.append((test_start, test_end, window_id))
        for name in ("train_data_artifact_id", "validation_data_artifact_id",
                     "test_data_artifact_id", "model_artifact_id"):
            require(isinstance(window[name], str) and window[name],
                    f"{field}.{name} must be non-empty.")

    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        require(current[0] >= previous[1],
                f"Future test windows overlap: {previous[2]} and {current[2]}.")
    return windows


def validate_artifact_schema(contract: dict[str, Any]) -> dict[str, tuple[str, dict[str, Any]]]:
    artifacts = contract.get("artifacts")
    require(isinstance(artifacts, dict), "artifacts must be an object.")
    require(set(artifacts) == set(ARTIFACT_CATEGORIES),
            "artifacts must have separate code, config, data, model, and environment inventories.")
    index: dict[str, tuple[str, dict[str, Any]]] = {}
    paths: dict[str, str] = {}
    for category in ARTIFACT_CATEGORIES:
        rows = artifacts[category]
        require(isinstance(rows, list) and rows,
                f"artifacts.{category} must be a non-empty list.")
        for row_index, row in enumerate(rows):
            field = f"artifacts.{category}[{row_index}]"
            require(isinstance(row, dict), f"{field} must be an object.")
            require_keys(row, ("artifact_id", "path", "sha256"), field)
            artifact_id = row["artifact_id"]
            require(isinstance(artifact_id, str) and artifact_id and artifact_id not in index,
                    f"{field}.artifact_id must be globally unique and non-empty.")
            path = safe_relative_path(row["path"], f"{field}.path")
            path_text = path.as_posix()
            require(path_text not in paths,
                    f"Artifact path appears in both {paths.get(path_text)} and {category}: {path_text}")
            digest = row["sha256"]
            require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
                    f"{field}.sha256 must be a lowercase SHA-256 digest.")
            index[artifact_id] = (category, row)
            paths[path_text] = category
            if category == "data":
                require(row.get("role") in DATA_ROLES,
                        f"{field}.role must be train, validation, or test.")
                require_keys(row, ("window_id", "date_start", "date_end", "format"), field)
                parse_date(row["date_start"], f"{field}.date_start")
                parse_date(row["date_end"], f"{field}.date_end")
                require(row["format"] == "csv", f"{field}.format must be csv.")
            elif category == "model":
                require_keys(row, ("window_id", "trained_through", "selected_through"), field)
                parse_date(row["trained_through"], f"{field}.trained_through")
                parse_date(row["selected_through"], f"{field}.selected_through")
            elif category == "environment":
                require(row.get("kind") in ("package_lock", "runtime_attestation"),
                        f"{field}.kind must be package_lock or runtime_attestation.")
    return index


def validate_window_artifact_links(
    windows: list[dict[str, Any]],
    artifact_index: dict[str, tuple[str, dict[str, Any]]],
) -> None:
    role_to_field = {
        "train": "train_data_artifact_id",
        "validation": "validation_data_artifact_id",
        "test": "test_data_artifact_id",
    }
    for window in windows:
        window_id = window["window_id"]
        for role, field_name in role_to_field.items():
            artifact_id = window[field_name]
            require(artifact_id in artifact_index, f"Unknown artifact id: {artifact_id}")
            category, row = artifact_index[artifact_id]
            require(category == "data" and row["role"] == role,
                    f"{artifact_id} must be a {role} data artifact.")
            require(row["window_id"] == window_id,
                    f"{artifact_id} is assigned to the wrong window.")
            require(row["date_start"] == window[f"{role}_start"]
                    and row["date_end"] == window[f"{role}_end"],
                    f"{artifact_id} dates do not match preregistered {role} dates.")
        model_id = window["model_artifact_id"]
        require(model_id in artifact_index, f"Unknown model artifact id: {model_id}")
        category, model = artifact_index[model_id]
        require(category == "model" and model["window_id"] == window_id,
                f"{model_id} must be a model artifact for {window_id}.")
        trained = parse_date(model["trained_through"], f"model {model_id}.trained_through")
        selected = parse_date(model["selected_through"], f"model {model_id}.selected_through")
        validation_end = parse_date(window["validation_end"], f"{window_id}.validation_end")
        test_start = parse_date(window["test_start"], f"{window_id}.test_start")
        require(trained <= validation_end and selected <= validation_end and selected < test_start,
                f"{model_id} was trained or selected using test-period information.")


def validate_environment_schema(
    contract: dict[str, Any], artifact_index: dict[str, tuple[str, dict[str, Any]]]
) -> dict[str, Any]:
    environment = contract.get("environment_lock")
    require(isinstance(environment, dict), "environment_lock must be an object.")
    require_keys(
        environment,
        (
            "python_version", "r_version", "platform", "container_image_digest",
            "runtime_attestation_artifact_id", "package_lock_artifact_ids",
            "require_exact_versions", "allow_unpinned_packages",
            "network_access_during_test", "enforce_orchestrator_python_version",
        ),
        "environment_lock",
    )
    require(environment["require_exact_versions"] is True,
            "Exact package/runtime versions are mandatory.")
    require(environment["allow_unpinned_packages"] is False,
            "Unpinned packages are forbidden.")
    require(environment["network_access_during_test"] == "disabled",
            "Network access must be disabled during locked test execution.")
    require(environment["enforce_orchestrator_python_version"] is True,
            "The orchestrator Python version must be enforced.")
    require(isinstance(environment["container_image_digest"], str)
            and CONTAINER_DIGEST_RE.fullmatch(environment["container_image_digest"]) is not None,
            "container_image_digest must be a sha256:<digest> image identifier.")
    attestation_id = environment["runtime_attestation_artifact_id"]
    require(attestation_id in artifact_index,
            "runtime_attestation_artifact_id is unknown.")
    category, row = artifact_index[attestation_id]
    require(category == "environment" and row.get("kind") == "runtime_attestation",
            "runtime_attestation_artifact_id must identify an environment attestation.")
    locks = environment["package_lock_artifact_ids"]
    require(isinstance(locks, list) and len(locks) >= 2 and len(set(locks)) == len(locks),
            "At least two distinct package locks (Python and R) are required.")
    for artifact_id in locks:
        require(artifact_id in artifact_index, f"Unknown package lock: {artifact_id}")
        category, row = artifact_index[artifact_id]
        require(category == "environment" and row.get("kind") == "package_lock",
                f"{artifact_id} must be an environment package_lock artifact.")
    return environment


def validate_solver_schema(contract: dict[str, Any]) -> dict[str, Any]:
    solver = contract.get("solver")
    require(isinstance(solver, dict), "solver must be an object.")
    require_keys(
        solver,
        ("methods", "accepted_convergence_codes", "explicitly_rejected_codes",
         "require_every_method_period", "allow_fallback_weights"),
        "solver",
    )
    methods = solver["methods"]
    require(isinstance(methods, list) and methods and len(set(methods)) == len(methods),
            "solver.methods must be a unique non-empty list.")
    accepted = solver["accepted_convergence_codes"]
    rejected = solver["explicitly_rejected_codes"]
    require(accepted == [1, 2, 3, 4],
            "Only NLopt success codes 1, 2, 3, and 4 are accepted.")
    require(isinstance(rejected, list) and 5 in rejected,
            "NLopt code 5 (MAXEVAL_REACHED) must be explicitly rejected.")
    require(not set(accepted).intersection(rejected),
            "Accepted and rejected solver codes overlap.")
    require(solver["require_every_method_period"] is True,
            "Every solver method/period must have an audit row.")
    require(solver["allow_fallback_weights"] is False,
            "Fallback or stale weights are forbidden after solver failure.")
    return solver


def validate_execution_schema(
    contract: dict[str, Any], windows: list[dict[str, Any]],
    artifact_index: dict[str, tuple[str, dict[str, Any]]], selection: dict[str, Any]
) -> dict[str, Any]:
    execution = contract.get("execution")
    require(isinstance(execution, dict), "execution must be an object.")
    require_keys(
        execution,
        ("access_ledger_path", "test_access_limit", "preserve_failed_runs",
         "working_directory", "commands"),
        "execution",
    )
    safe_relative_path(execution["access_ledger_path"], "execution.access_ledger_path")
    safe_relative_path(execution["working_directory"], "execution.working_directory")
    require(execution["test_access_limit"] == 1,
            "The confirmatory test may be accessed exactly once.")
    require(execution["preserve_failed_runs"] is True,
            "Failed locked runs must be preserved.")
    commands = execution["commands"]
    require(isinstance(commands, list) and commands, "execution.commands must be non-empty.")
    window_ids = {window["window_id"] for window in windows}
    command_windows: list[str] = []
    command_ids: set[str] = set()
    artifact_ids = set(artifact_index)
    for index, command in enumerate(commands):
        field = f"execution.commands[{index}]"
        require(isinstance(command, dict), f"{field} must be an object.")
        require_keys(
            command,
            ("command_id", "window_id", "phase", "argv", "input_artifact_ids",
             "expected_output_relpaths", "evaluation_panel_relpath", "solver_audit_relpath"),
            field,
        )
        command_id = command["command_id"]
        require(isinstance(command_id, str) and command_id and command_id not in command_ids,
                f"{field}.command_id must be unique and non-empty.")
        command_ids.add(command_id)
        window_id = command["window_id"]
        require(window_id in window_ids, f"{field}.window_id is unknown.")
        command_windows.append(window_id)
        require(command["phase"] == "test",
                f"{field}.phase must be test; training/tuning is not executable here.")
        argv = command["argv"]
        require(isinstance(argv, list) and argv and all(isinstance(item, str) for item in argv),
                f"{field}.argv must be a non-empty string list.")
        for argument in argv:
            placeholder = PLACEHOLDER_RE.fullmatch(argument)
            if placeholder:
                kind, value = placeholder.groups()
                if kind == "artifact":
                    require(value in artifact_ids, f"{field}.argv references unknown artifact {value}.")
                else:
                    safe_relative_path(value, f"{field}.argv output placeholder")
            else:
                require("{" not in argument and "}" not in argument,
                        f"{field}.argv contains a malformed placeholder: {argument}")
                path_like = ("/" in argument or "\\" in argument
                             or PurePosixPath(argument).suffix.lower()
                             in {".csv", ".json", ".pt", ".r", ".py", ".yaml", ".yml", ".lock"})
                require(not path_like and not os.path.isabs(argument),
                        f"{field}.argv path inputs/outputs must use declared placeholders: {argument}")
        inputs = command["input_artifact_ids"]
        require(isinstance(inputs, list) and inputs and len(inputs) == len(set(inputs)),
                f"{field}.input_artifact_ids must be a unique non-empty list.")
        require(set(inputs).issubset(artifact_ids), f"{field} has unknown input artifacts.")
        referenced = {match.group(2) for item in argv
                      if (match := PLACEHOLDER_RE.fullmatch(item)) and match.group(1) == "artifact"}
        require(referenced == set(inputs),
                f"{field}.input_artifact_ids must exactly match argv artifact placeholders.")
        for artifact_id in inputs:
            category, row = artifact_index[artifact_id]
            require(not (category == "data" and row.get("role") in ("train", "validation")),
                    f"{field} exposes {row.get('role')} data during test execution: {artifact_id}")
        window = next(item for item in windows if item["window_id"] == window_id)
        required_inputs = {
            window["test_data_artifact_id"], window["model_artifact_id"],
            selection["hyperparameter_lock_artifact_id"],
            selection["selection_manifest_artifact_id"],
        }
        require(required_inputs.issubset(set(inputs)),
                f"{field} must consume the frozen test data, model, hyperparameters, and selection manifest.")
        expected = command["expected_output_relpaths"]
        require(isinstance(expected, list) and expected and len(expected) == len(set(expected)),
                f"{field}.expected_output_relpaths must be a unique non-empty list.")
        expected_paths = {safe_relative_path(value, f"{field}.expected_output_relpaths").as_posix()
                          for value in expected}
        panel = safe_relative_path(command["evaluation_panel_relpath"],
                                   f"{field}.evaluation_panel_relpath").as_posix()
        audit = safe_relative_path(command["solver_audit_relpath"],
                                   f"{field}.solver_audit_relpath").as_posix()
        require(panel in expected_paths and audit in expected_paths,
                f"{field} must declare its evaluation panel and solver audit as outputs.")
    require(sorted(command_windows) == sorted(window_ids),
            "Exactly one locked test command is required for every future window.")
    return execution


def validate_contract_schema(contract: dict[str, Any]) -> dict[str, Any]:
    require_keys(
        contract,
        (
            "schema_version", "protocol_id", "protocol_status", "preregistered_utc",
            "consumed_holdouts", "assets", "economics", "selection", "windows",
            "artifacts", "environment_lock", "solver", "execution", "claims",
        ),
        "contract",
    )
    require(contract["schema_version"] == 1, "Only future confirmatory schema_version=1 is supported.")
    require(isinstance(contract["protocol_id"], str) and contract["protocol_id"],
            "protocol_id must be non-empty.")
    require(contract["protocol_status"] == "preregistered",
            "protocol_status must be preregistered before any test access.")
    parse_utc_timestamp(contract["preregistered_utc"], "preregistered_utc")
    consumed_end = validate_consumed_holdouts(contract)
    assets, economics = validate_assets_and_economics(contract)
    selection = validate_selection(contract)
    windows = validate_windows(contract, consumed_end)
    artifact_index = validate_artifact_schema(contract)
    validate_window_artifact_links(windows, artifact_index)
    environment = validate_environment_schema(contract, artifact_index)
    solver = validate_solver_schema(contract)
    execution = validate_execution_schema(
        contract, windows, artifact_index, selection
    )
    claims = contract["claims"]
    require(isinstance(claims, dict), "claims must be an object.")
    require_keys(claims, ("minimum_test_periods_per_window", "pooling_rule",
                          "tail_metrics_descriptive_below_events"), "claims")
    minimum_periods = int(finite_number(
        claims["minimum_test_periods_per_window"],
        "claims.minimum_test_periods_per_window", 2,
    ))
    for window in windows:
        require(len(window["periods"]) >= minimum_periods,
                f"{window['window_id']} has fewer than the preregistered minimum test periods.")
    require(claims["pooling_rule"] == "paired_non_overlapping_window_returns",
            "Confirmatory pooling must use paired non-overlapping window returns.")
    finite_number(claims["tail_metrics_descriptive_below_events"],
                  "claims.tail_metrics_descriptive_below_events", 1)
    return {
        "protocol_id": contract["protocol_id"],
        "contract_sha256": canonical_json_sha256(contract),
        "asset_count": len(assets),
        "window_count": len(windows),
        "future_test_start": min(window["test_start"] for window in windows),
        "future_test_end": max(window["test_end"] for window in windows),
        "artifact_count": len(artifact_index),
        "economics": economics,
        "environment_lock": environment,
        "solver": solver,
        "execution": execution,
        "artifact_index": artifact_index,
    }


def validate_environment_attestation(
    contract: dict[str, Any], artifact_index: dict[str, tuple[str, dict[str, Any]]],
    repo_root: Path,
) -> None:
    environment = contract["environment_lock"]
    attestation_id = environment["runtime_attestation_artifact_id"]
    _, row = artifact_index[attestation_id]
    attestation = load_json(materialize(
        repo_root, safe_relative_path(row["path"], f"artifact {attestation_id}.path")
    ))
    for field in ("python_version", "r_version", "platform", "container_image_digest"):
        require(attestation.get(field) == environment[field],
                f"Runtime attestation disagrees with environment_lock.{field}.")
    require(platform.python_version() == environment["python_version"],
            f"Orchestrator Python {platform.python_version()} != locked {environment['python_version']}.")


def validate_artifact_hashes(
    artifact_index: dict[str, tuple[str, dict[str, Any]]], repo_root: Path
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for artifact_id, (category, row) in sorted(artifact_index.items()):
        relative = safe_relative_path(row["path"], f"artifact {artifact_id}.path")
        path = materialize(repo_root, relative)
        require(path.is_file(), f"Frozen {category} artifact is missing: {path}")
        actual = sha256_file(path)
        require(actual == row["sha256"],
                f"Frozen {category} artifact hash mismatch: {artifact_id}")
        inventory.append({
            "artifact_id": artifact_id, "category": category,
            "path": relative.as_posix(), "sha256": actual,
            "size_bytes": path.stat().st_size,
        })
    return inventory


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    require(path.is_file(), f"CSV file not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def as_float(value: str, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ProtocolError(f"{field} is not numeric: {value}") from error
    require(math.isfinite(result), f"{field} must be finite.")
    return result


def window_period_keys(window: dict[str, Any]) -> list[tuple[str, str]]:
    return [(period["decision_date"], period["holding_end_date"])
            for period in window["periods"]]


def validate_test_data_csv(path: Path, assets: list[str], window: dict[str, Any]) -> None:
    header, rows = read_csv_rows(path)
    required = ["window_id", "decision_date", "holding_end_date", *[f"g_{a}" for a in assets]]
    require(all(name in header for name in required),
            f"Test data {path} lacks fixed keys/assets: {required}")
    require(len(rows) == len(window["periods"]),
            f"Test data row count does not match {window['window_id']}.")
    keys = [(row["decision_date"], row["holding_end_date"]) for row in rows]
    require(keys == window_period_keys(window),
            f"Test data dates do not match preregistered dates for {window['window_id']}.")
    for row_index, row in enumerate(rows):
        require(row["window_id"] == window["window_id"],
                f"Test data row {row_index} has the wrong window_id.")
        for asset in assets:
            require(as_float(row[f"g_{asset}"], f"test data row {row_index} g_{asset}") > 0,
                    "Realized asset gross returns must be positive.")


def validate_input_data(contract: dict[str, Any], repo_root: Path) -> None:
    assets = list(contract["assets"])
    artifact_index = validate_artifact_schema(contract)
    for window in contract["windows"]:
        artifact_id = window["test_data_artifact_id"]
        _, row = artifact_index[artifact_id]
        path = materialize(repo_root, safe_relative_path(row["path"], f"artifact {artifact_id}.path"))
        validate_test_data_csv(path, assets, window)


def validate_solver_audit(path: Path, contract: dict[str, Any], window: dict[str, Any]) -> None:
    header, rows = read_csv_rows(path)
    required = {"method", "decision_date", "convergence"}
    require(required.issubset(header), f"Solver audit lacks columns: {sorted(required)}")
    methods = contract["solver"]["methods"]
    accepted = set(contract["solver"]["accepted_convergence_codes"])
    rejected = set(contract["solver"]["explicitly_rejected_codes"])
    expected_dates = [period["decision_date"] for period in window["periods"]]
    observed: set[tuple[str, str]] = set()
    for row_index, row in enumerate(rows):
        method = row["method"]
        if method not in methods:
            continue
        key = (method, row["decision_date"])
        require(key not in observed, f"Duplicate solver audit row: {key}")
        observed.add(key)
        try:
            numeric = float(row["convergence"])
            code = int(numeric)
        except (TypeError, ValueError) as error:
            raise ProtocolError(f"Missing solver convergence code in row {row_index}.") from error
        require(float(code) == numeric, f"Non-integer convergence code in row {row_index}.")
        require(code not in rejected, f"Rejected solver convergence code {code} in {key}.")
        require(code in accepted,
                f"Solver convergence code {code} is not a declared success in {key}.")
    expected = {(method, date) for method in methods for date in expected_dates}
    require(observed == expected,
            f"Solver audit is incomplete for {window['window_id']}; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}")


def close(actual: float, expected: float, tolerance: float, field: str) -> None:
    require(abs(actual - expected) <= tolerance,
            f"{field}: observed {actual:.12g}, expected {expected:.12g}, tolerance {tolerance:.3g}")


def validate_evaluation_panel(path: Path, contract: dict[str, Any], window: dict[str, Any]) -> None:
    header, rows = read_csv_rows(path)
    assets = list(contract["assets"])
    economics = contract["economics"]
    target_columns = [f"target_w_{asset}" for asset in assets]
    pretrade_columns = [f"pretrade_w_{asset}" for asset in assets]
    gross_columns = [f"g_{asset}" for asset in assets]
    required = {
        "strategy_id", "window_id", "decision_date", "holding_end_date",
        "calendar_days", "gross_return", "net_return", "turnover",
        "transaction_cost", "financing_cost", "short_notional",
        "cash_borrow_notional", "gross_exposure", "net_exposure",
        *target_columns, *pretrade_columns, *gross_columns,
    }
    require(required.issubset(header),
            f"Evaluation panel lacks audit columns: {sorted(required - set(header))}")
    require(rows, f"Evaluation panel is empty: {path}")
    periods = window_period_keys(window)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        require(row["window_id"] == window["window_id"],
                "Evaluation panel contains an undeclared window_id.")
        grouped.setdefault(row["strategy_id"], []).append(row)
    tolerance = float(economics["metric_tolerance"])
    weight_tolerance = float(economics["weight_tolerance"])
    initial = [float(economics["net_exposure"]) / len(assets)] * len(assets)
    for strategy, strategy_rows in grouped.items():
        strategy_rows.sort(key=lambda row: (row["decision_date"], row["holding_end_date"]))
        observed_keys = [(row["decision_date"], row["holding_end_date"])
                         for row in strategy_rows]
        require(observed_keys == periods,
                f"Strategy {strategy} does not cover exactly the preregistered periods.")
        expected_pretrade = initial
        for row_index, row in enumerate(strategy_rows):
            decision = parse_date(row["decision_date"], "panel decision_date")
            holding_end = parse_date(row["holding_end_date"], "panel holding_end_date")
            calendar_days = int(as_float(row["calendar_days"], "calendar_days"))
            require(calendar_days == (holding_end - decision).days,
                    f"{strategy} row {row_index} calendar_days is not the actual interval length.")
            target = [as_float(row[name], name) for name in target_columns]
            pretrade = [as_float(row[name], name) for name in pretrade_columns]
            asset_gross = [as_float(row[name], name) for name in gross_columns]
            require(all(value > 0 for value in asset_gross),
                    "Asset gross returns must be positive.")
            for asset_index, (actual, expected) in enumerate(zip(pretrade, expected_pretrade)):
                close(actual, expected, weight_tolerance,
                      f"{strategy} row {row_index} drifted pretrade {assets[asset_index]}")
            net_exposure = sum(target)
            gross_exposure = sum(abs(value) for value in target)
            short_notional = sum(max(-value, 0.0) for value in target)
            cash_notional = max(net_exposure - 1.0, 0.0)
            close(net_exposure, float(economics["net_exposure"]), weight_tolerance,
                  f"{strategy} row {row_index} target net exposure")
            require(gross_exposure <= float(economics["gross_leverage"]) + weight_tolerance,
                    f"{strategy} row {row_index} exceeds gross leverage.")
            require(max(target) <= float(economics["max_long_weight"]) + weight_tolerance,
                    f"{strategy} row {row_index} exceeds the long cap.")
            require(min(target) >= -float(economics["max_short_weight"]) - weight_tolerance,
                    f"{strategy} row {row_index} exceeds the short cap.")
            close(as_float(row["net_exposure"], "net_exposure"), net_exposure, tolerance,
                  f"{strategy} row {row_index} net exposure log")
            close(as_float(row["gross_exposure"], "gross_exposure"), gross_exposure, tolerance,
                  f"{strategy} row {row_index} gross exposure log")
            close(as_float(row["short_notional"], "short_notional"), short_notional, tolerance,
                  f"{strategy} row {row_index} short notional log")
            close(as_float(row["cash_borrow_notional"], "cash_borrow_notional"), cash_notional,
                  tolerance, f"{strategy} row {row_index} cash notional log")
            turnover = sum(abs(target_value - pretrade_value)
                           for target_value, pretrade_value in zip(target, pretrade))
            transaction_cost = float(economics["turnover_cost"]) * turnover
            financing_cost = calendar_days / float(economics["day_count_basis"]) * (
                float(economics["annual_short_borrow_rate"]) * short_notional
                + float(economics["annual_cash_borrow_rate"]) * cash_notional
            )
            gross_return = sum(weight * gross for weight, gross in zip(target, asset_gross)) - 1.0
            require(1.0 + gross_return > 0,
                    f"{strategy} row {row_index} has non-positive portfolio gross.")
            net_return = math.exp(math.log1p(gross_return)
                                  - transaction_cost - financing_cost) - 1.0
            close(as_float(row["turnover"], "turnover"), turnover, tolerance,
                  f"{strategy} row {row_index} drift-aware turnover")
            close(as_float(row["transaction_cost"], "transaction_cost"), transaction_cost,
                  tolerance, f"{strategy} row {row_index} transaction cost")
            close(as_float(row["financing_cost"], "financing_cost"), financing_cost,
                  tolerance, f"{strategy} row {row_index} prorated financing cost")
            close(as_float(row["gross_return"], "gross_return"), gross_return, tolerance,
                  f"{strategy} row {row_index} gross return")
            close(as_float(row["net_return"], "net_return"), net_return, tolerance,
                  f"{strategy} row {row_index} net return")
            denominator = 1.0 + gross_return
            expected_pretrade = [weight * gross / denominator
                                 for weight, gross in zip(target, asset_gross)]


def validate_full_contract(contract: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    report = validate_contract_schema(contract)
    artifact_index = report["artifact_index"]
    inventory = validate_artifact_hashes(artifact_index, repo_root)
    validate_environment_attestation(contract, artifact_index, repo_root)
    validate_input_data(contract, repo_root)
    report["verified_artifacts"] = inventory
    return report


def render_command(
    argv: list[str], artifact_index: dict[str, tuple[str, dict[str, Any]]],
    repo_root: Path, temporary: Path,
) -> list[str]:
    rendered: list[str] = []
    for argument in argv:
        match = PLACEHOLDER_RE.fullmatch(argument)
        if not match:
            rendered.append(argument)
            continue
        kind, value = match.groups()
        if kind == "artifact":
            _, row = artifact_index[value]
            rendered.append(str(materialize(
                repo_root, safe_relative_path(row["path"], f"artifact {value}.path")
            ).resolve()))
        else:
            destination = materialize(temporary, safe_relative_path(value, "output placeholder"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            rendered.append(str(destination.resolve()))
    return rendered


def deterministic_tar(
    source: Path,
    bundle: Path,
    *,
    root_name: str | None = None,
) -> None:
    sidecar = bundle.with_suffix(bundle.suffix + ".sha256")
    require(not bundle.exists() and not sidecar.exists(),
            f"Immutable bundle or checksum already exists: {bundle}")
    archive_root = root_name or source.name
    require(bool(archive_root) and Path(archive_root).name == archive_root,
            f"Archive root must be one stable path component: {archive_root!r}")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{bundle.name}.", suffix=".tmp", dir=bundle.parent
    )
    os.close(descriptor)
    temporary_bundle = Path(temporary_name)
    temporary_sidecar = Path(f"{temporary_name}.sha256")
    published_bundle = False
    try:
        with temporary_bundle.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
                ) as archive:
                    entries = [
                        source,
                        *sorted(source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()),
                    ]
                    for entry in entries:
                        arcname = Path(archive_root) / entry.relative_to(source)
                        info = archive.gettarinfo(str(entry), arcname=arcname.as_posix())
                        info.mtime = 0
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        if info.isdir():
                            info.mode = 0o755
                        elif info.isreg():
                            info.mode = 0o755 if info.mode & 0o111 else 0o644
                        if info.isreg():
                            with entry.open("rb") as stream:
                                archive.addfile(info, stream)
                        else:
                            archive.addfile(info)
        temporary_sidecar.write_text(
            f"{sha256_file(temporary_bundle)}  {bundle.name}\n", encoding="utf-8"
        )
        os.replace(temporary_bundle, bundle)
        published_bundle = True
        os.replace(temporary_sidecar, sidecar)
    except Exception:
        temporary_bundle.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
        if published_bundle:
            bundle.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise


def _remove_bundle_pair(bundle: Path) -> None:
    bundle.unlink(missing_ok=True)
    bundle.with_suffix(bundle.suffix + ".sha256").unlink(missing_ok=True)


def _publish_tree(temporary: Path, output: Path, bundle: Path) -> None:
    deterministic_tar(temporary, bundle, root_name=output.name)
    try:
        os.replace(temporary, output)
    except Exception:
        _remove_bundle_pair(bundle)
        raise


def _preserve_failed_tree(temporary: Path, output: Path, bundle: Path) -> Exception | None:
    archive_error: Exception | None = None
    try:
        deterministic_tar(temporary, bundle, root_name=output.name)
    except Exception as error:
        archive_error = error
        _remove_bundle_pair(bundle)
    os.replace(temporary, output)
    return archive_error


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_access_ledger(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as error:
        raise ProtocolError(
            f"Confirmatory test access ledger already exists; rerun forbidden: {path}"
        ) from error


def execute_contract(
    contract_path: Path, repo_root: Path, output: Path, bundle: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output = output.resolve()
    bundle = bundle.resolve()
    contract = load_json(contract_path.resolve())
    schema_report = validate_contract_schema(contract)
    require(not output.exists(), f"Immutable output already exists: {output}")
    require(not bundle.exists() and not bundle.with_suffix(bundle.suffix + ".sha256").exists(),
            f"Immutable bundle or checksum already exists: {bundle}")
    execution = contract["execution"]
    ledger = materialize(
        repo_root, safe_relative_path(execution["access_ledger_path"],
                                      "execution.access_ledger_path")
    )
    require(not ledger.exists(), f"Confirmatory test has already been accessed: {ledger}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}_", dir=output.parent))
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": contract["protocol_id"],
        "contract_sha256": schema_report["contract_sha256"],
        "status": "started",
        "test_access_started": True,
        "started_utc": started,
        "commands": [],
    }
    try:
        create_access_ledger(ledger, {
            "protocol_id": contract["protocol_id"],
            "contract_sha256": schema_report["contract_sha256"],
            "test_access_started_utc": started,
            "declared_output": str(output),
            "scientific_rule": "No retry or tuning on these test observations.",
        })
        write_json(temporary / "ACCESS_STARTED.json", load_json(ledger))
        (temporary / "preregistered_contract.json").write_text(
            contract_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        full_report = validate_full_contract(contract, repo_root)
        serializable_report = {key: value for key, value in full_report.items()
                               if key not in {"artifact_index", "economics",
                                              "environment_lock", "solver", "execution"}}
        write_json(temporary / "pre_execution_validation.json", serializable_report)
        write_json(temporary / "input_hash_inventory.json", full_report["verified_artifacts"])
        artifact_index = full_report["artifact_index"]
        working_directory = materialize(
            repo_root, safe_relative_path(execution["working_directory"],
                                          "execution.working_directory")
        ).resolve()
        require(working_directory.is_dir(), f"Working directory not found: {working_directory}")
        logs = temporary / "command_logs"
        logs.mkdir()
        locked_env = os.environ.copy()
        locked_env.update({
            "LC_ALL": "C", "LANG": "C", "LANGUAGE": "C", "TZ": "UTC",
            "PYTHONNOUSERSITE": "1", "PIP_NO_INDEX": "1",
            "CONFIRMATORY_NETWORK_ACCESS": "disabled",
            "CONFIRMATORY_PROTOCOL_ID": contract["protocol_id"],
            "CONFIRMATORY_CONTRACT_SHA256": schema_report["contract_sha256"],
        })
        for command in execution["commands"]:
            command_id = command["command_id"]
            argv = render_command(command["argv"], artifact_index, repo_root, temporary)
            command_env = locked_env.copy()
            command_env["CONFIRMATORY_WINDOW_ID"] = command["window_id"]
            start = time.monotonic()
            result = subprocess.run(
                argv, cwd=working_directory, env=command_env,
                text=True, capture_output=True, check=False,
            )
            elapsed = time.monotonic() - start
            (logs / f"{command_id}.stdout.txt").write_text(result.stdout, encoding="utf-8")
            (logs / f"{command_id}.stderr.txt").write_text(result.stderr, encoding="utf-8")
            command_record = {
                "command_id": command_id, "window_id": command["window_id"],
                "returncode": result.returncode, "elapsed_seconds": elapsed,
            }
            manifest["commands"].append(command_record)
            require(result.returncode == 0,
                    f"Locked command {command_id} failed with exit code {result.returncode}.")
            for relative in command["expected_output_relpaths"]:
                expected = materialize(temporary, safe_relative_path(
                    relative, f"{command_id}.expected_output_relpaths"
                ))
                require(expected.is_file(), f"Locked command output is missing: {expected}")
            window = next(item for item in contract["windows"]
                          if item["window_id"] == command["window_id"])
            validate_evaluation_panel(
                materialize(temporary, safe_relative_path(
                    command["evaluation_panel_relpath"],
                    f"{command_id}.evaluation_panel_relpath",
                )), contract, window,
            )
            validate_solver_audit(
                materialize(temporary, safe_relative_path(
                    command["solver_audit_relpath"],
                    f"{command_id}.solver_audit_relpath",
                )), contract, window,
            )
        manifest.update({
            "status": "complete",
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "scientific_interpretation": (
                "Locked future confirmatory execution completed. Statistical conclusions "
                "must still follow the preregistered claims and multiplicity rules."
            ),
        })
        write_json(temporary / "future_confirmatory_manifest.json", manifest)
        _publish_tree(temporary, output, bundle)
        return manifest
    except Exception as error:
        manifest.update({
            "status": "failed",
            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "scientific_interpretation": (
                "The accessed test run failed and is preserved. Do not tune or retry on "
                "these observations; declare a genuinely new future sample."
            ),
        })
        write_json(temporary / "future_confirmatory_manifest.json", manifest)
        archive_error = _preserve_failed_tree(temporary, output, bundle)
        message = str(error)
        if archive_error is not None:
            message += (
                f" Failure logs were preserved at {output}, but archive creation also failed: "
                f"{archive_error}"
            )
        if isinstance(error, ProtocolError):
            if archive_error is None:
                raise
            raise ProtocolError(message) from error
        raise ProtocolError(message) from error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser(
        "validate", help="Validate preregistration schema without opening test artifacts."
    )
    validate.add_argument("--contract", type=Path, required=True)
    execute = subparsers.add_parser(
        "execute", help="Open test artifacts once and execute the immutable locked batch."
    )
    execute.add_argument("--contract", type=Path, required=True)
    execute.add_argument("--repo-root", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--bundle", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "validate":
            report = validate_contract_schema(load_json(args.contract.resolve()))
            printable = {key: value for key, value in report.items()
                         if key not in {"artifact_index", "economics", "environment_lock",
                                        "solver", "execution"}}
            print(json.dumps(printable, indent=2, sort_keys=True))
        else:
            print(json.dumps(execute_contract(
                args.contract, args.repo_root, args.output, args.bundle
            ), indent=2, sort_keys=True))
    except ProtocolError as error:
        print(f"FUTURE CONFIRMATORY PROTOCOL FAILURE: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
