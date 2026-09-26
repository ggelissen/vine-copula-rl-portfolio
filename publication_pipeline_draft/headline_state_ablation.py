#!/usr/bin/env python3
"""Frozen, post-holdout input ablation around the mixed-curriculum headline.

Only three information masks are newly trained. The compressed arm is read
from the immutable mixed-pretraining evidence release; no checkpoint is
relabelled or retrained to manufacture the headline comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from publication_pipeline_draft.audit_mixed_pretraining_comparison import (
    finite_tensors, gate_failures,
)
from publication_pipeline_draft.mixed_pretraining_protocol import (
    JOB_ENV_FIELDS, load_contract as load_base, validate_bundle,
)
from publication_pipeline_draft.run_mixed_pretraining_sweep import run_job
from publication_pipeline_draft.synthetic_dose_protocol import (
    DoseProtocolError, read_csv, require, sha256, write_archive,
)

ARMS = ("scalar_scenario_cvar", "raw_vine_coordinates", "full_state")
HEADLINE = "compressed_inputs"
TRAINING_SOURCES = (
    "run_with_config.r", "config/config.yaml", "helper/reproducibility.r",
    "helper/load_data.r", "helper/time_split.r", "helper/marginals.r",
    "benchmark_models/dynamic_vine_NN.r", "rl/rl_environment.r",
    "rl/train_rl.r", "rl/training_sanity_check.r", "rl/action_projection.py",
    "rl/recurrent_baselines.py",
)
SOURCES = (*TRAINING_SOURCES, "evaluate_with_config.r", "rl/evaluate_rl.r",
           "rl/policy_inference_server_v2.py",
           "publication_pipeline_draft/headline_state_ablation.py",
           "publication_pipeline_draft/analyze_headline_state_ablation.py",
           "publication_pipeline_draft/config/headline_state_ablation_v1.json",
           "publication_pipeline_draft/tests/test_headline_state_ablation.py",
           "publication_pipeline_draft/HEADLINE_STATE_ABLATION_V1_RUNBOOK.md",
           "hpc/run_headline_state_ablation_v1.sh")


def load_design(repo: Path, path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    design = json.loads(path.read_text(encoding="utf-8"))
    require(design["schema_version"] == 1 and
            design["experiment_id"] == "post_holdout_headline_mixed_policy_inputs_v1" and
            design["evidence_class"] == "post_holdout_explanatory" and
            design["confirmatory_claim_permitted"] is False and
            design["consumed_holdout_reused"] is True and
            "explicit post-holdout protocol deviation" in
            design["protocol_deviation_disclosure"],
            "Post-holdout evidence boundary changed.")
    base_path = repo / design["base_contract"]
    base, base_sha = load_base(base_path)
    require(base_sha == design["base_contract_sha256"],
            "Headline mixed-curriculum contract changed.")
    require(tuple(map(int, design["seeds"])) == tuple(base["seeds"]),
            "State ablation must reuse exactly the ten headline seed IDs.")
    require(design["headline_arm"] == HEADLINE and
            [arm["arm_id"] for arm in design["new_arms"]] == list(ARMS),
            "The four-state comparison changed.")
    require([(a["VINE_OBSERVATION_MODE"], a["VINE_FEATURE_MODE"],
              a["CVAR_OBSERVATION_MODE"]) for a in design["new_arms"]] == [
                ("full", "zero", "full"), ("full", "full", "zero"),
                ("full", "full", "full")],
            "Input masks must differ only in the two policy-visible channels.")
    matched = design["matched_training"]
    require(tuple(matched[key] for key in (
        "unique_synthetic_paths", "unique_historical_paths",
        "pretrain_presentations", "synthetic_presentations",
        "historical_presentations", "historical_finetune_episodes",
        "episode_length", "policy_observation_dimension")) ==
            (100, 61, 1000, 621, 379, 61, 24, 88) and
            matched["pretrain_data_mode"] == "mixed_historical_synthetic" and
            matched["cvar_reward_mode"] == "full",
            "Matched training geometry changed.")
    ev = design["evaluation"]
    require(ev["window_id"] == "locked_oos_v1" and
            ev["complete_periods"] == 22 and ev["locked_periods"] == 24 and
            ev["turnover_convention"] == "target_to_target_v1" and
            ev["ensemble_rule"] == "average_target_weights_then_rescore_costs",
            "Evaluation or ensemble accounting changed.")
    require(design["failure_policy"] ==
            "all_30_new_and_10_frozen_headline_policies_required_no_seed_substitution",
            "No-substitution rule changed.")
    return design, base, sha256(path)


def original_training_source_check(repo: Path, design: dict[str, Any]) -> str:
    release = repo / design["original_training_release"]
    inventory = release / "source_inventory.csv"
    contents = release / "CONTENTS.sha256"
    manifest_path = release / "mixed_pretraining_release_manifest.json"
    require(inventory.is_file() and contents.is_file() and manifest_path.is_file(),
            "Original frozen mixed-training release is missing. Restore its original tar.gz before training.")
    for line in contents.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        target = release / relative
        require(target.is_file() and sha256(target) == expected,
                f"Original training source release changed: {relative}")
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(frozen["release_status"] ==
            "frozen_post_holdout_mixed_pretraining_response_v1" and
            frozen["contract_sha256"] == design["base_contract_sha256"],
            "Original source release is not the headline mixed training release.")
    by_path = {row["path"]: row["sha256"] for row in read_csv(inventory)}
    for relative in TRAINING_SOURCES:
        snapshot = release / "source_snapshot" / relative
        require(relative in by_path and snapshot.is_file() and
                sha256(snapshot) == by_path[relative],
                f"Original frozen training source is incomplete: {relative}")
    return sha256(inventory)


def headline_evidence(repo: Path, design: dict[str, Any]) -> tuple[Path, list[dict[str, str]], str]:
    release = repo / design["headline_evidence_release"]
    evidence = release / "mixed_pretraining_evidence_manifest.json"
    inventory = (release / "source_evidence/weight_evidence/"
                 "mixed_pretraining_comparison_weight_manifest.csv")
    require(evidence.is_file() and inventory.is_file(),
            "Frozen headline weight evidence is missing.")
    manifest = json.loads(evidence.read_text(encoding="utf-8"))
    require(manifest["status"] == "frozen_mixed_pretraining_evidence_v1" and
            manifest["source_analysis_manifest"]["comparison_weight_manifest_sha256"]
            == sha256(inventory), "Headline weight manifest is not frozen evidence.")
    rows = [row for row in read_csv(inventory) if row["experiment_id"] ==
            "mixed_pretraining_plus_historical_finetuning"]
    require(len(rows) == 10 and {int(row["seed"]) for row in rows} ==
            set(design["seeds"]), "Frozen headline lacks ten unique seed weights.")
    for row in rows:
        weight = (release / "source_evidence/weight_evidence/weights/"
                  "mixed_pretraining_plus_historical_finetuning" /
                  f"seed_{row['seed']}" / f"weights_rl_full_seed_{row['seed']}.csv")
        require(weight.is_file() and sha256(weight) == row["sha256"],
                f"Frozen headline weight file changed: seed {row['seed']}")
    return release, rows, sha256(inventory)


def jobs_for(repo: Path, design: dict[str, Any], base: dict[str, Any],
             digest: str, bundle_sha256: str = "") -> list[dict[str, str]]:
    settings = {key: str(value) for key, value in base["base_model"].items()}
    settings["SYNTHETIC_RETURNS_FILE"] = base["bundle"]
    rows = []
    for arm in design["new_arms"]:
        for seed in design["seeds"]:
            env = dict(settings)
            env.update({key: arm[key] for key in (
                "VINE_OBSERVATION_MODE", "VINE_FEATURE_MODE",
                "CVAR_OBSERVATION_MODE")})
            env["CHECKPOINT_PREFIX"] = "td3_lstm_headline_state_v1"
            rows.append({
                "experiment_id": arm["arm_id"], "seed": str(seed),
                "output_dir": (Path("data/headline_state_ablation_runs_v1") /
                               arm["arm_id"] / f"seed_{seed}").as_posix(),
                "contract_sha256": digest, **env,
                "bundle_sha256": bundle_sha256,
            })
    require(len(rows) == 30 and all(set(JOB_ENV_FIELDS) <= set(row) for row in rows),
            "Expected three complete ten-seed training arms.")
    return rows


def prepare(args: argparse.Namespace, repo: Path, design: dict[str, Any],
            base: dict[str, Any], digest: str) -> dict[str, Any]:
    require(not args.output.exists() and not args.jobs.exists(),
            "Jobs or frozen release already exists.")
    runtime = repo / design["training_source_runtime"]
    require(not runtime.exists(), "Isolated original-source training runtime exists.")
    bundle = validate_bundle(repo, base)
    source_inventory_sha = original_training_source_check(repo, design)
    original_manifest = json.loads((repo / design["original_training_release"] /
                                    "mixed_pretraining_release_manifest.json").read_text(
                                        encoding="utf-8"))
    require(original_manifest["training_bundle_sha256"] == bundle["sha256"],
            "Original headline training release used a different mixed bundle.")
    _, _, headline_sha = headline_evidence(repo, design)
    rows = jobs_for(repo, design, base, digest, bundle["sha256"])
    require(args.runtime.is_file(), "Training runtime evidence is missing.")
    for relative in SOURCES:
        require((repo / relative).is_file(), f"Missing source: {relative}")
    # Execute the exact frozen original training source, not a later checkout.
    # Later edits to the marginal and vine helpers must not confound input masks.
    source = repo / design["original_training_release"] / "source_snapshot"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".headline-state-runtime.",
                                    dir=runtime.parent))
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        (staging / "data").symlink_to(repo / "data", target_is_directory=True)
        os.replace(staging, runtime)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    args.jobs.parent.mkdir(parents=True, exist_ok=True)
    with args.jobs.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".headline-state-freeze.",
                                      dir=args.output.parent))
    try:
        inventory = []
        for relative in SOURCES:
            destination = temporary / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / relative, destination)
            inventory.append({"path": relative, "sha256": sha256(destination)})
        with (temporary / "source_inventory.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
            writer.writeheader(); writer.writerows(inventory)
        shutil.copy2(args.jobs, temporary / "headline_state_jobs.csv")
        manifest = {
            "status": "frozen_post_holdout_headline_state_ablation_v1",
            "schema_version": 1, "evidence_class": "post_holdout_explanatory",
            "confirmatory_claim_permitted": False,
            "job_count": 30, "experiment_count": 3, "seeds_per_experiment": 10,
            "contract_sha256": digest, "jobs_sha256": sha256(args.jobs),
            "bundle_sha256": bundle["sha256"],
            "original_training_source_inventory_sha256": source_inventory_sha,
            "original_training_release_manifest_sha256": sha256(
                repo / design["original_training_release"] /
                "mixed_pretraining_release_manifest.json"),
            "training_source_runtime": design["training_source_runtime"],
            "headline_weight_manifest_sha256": headline_sha,
            "training_runtime_sha256": sha256(args.runtime),
        }
        (temporary / "headline_state_release_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        (temporary / "CONTENTS.sha256").write_text("".join(
            f"{sha256(path)}  {path.relative_to(temporary).as_posix()}\n"
            for path in files), encoding="ascii")
        os.replace(temporary, args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    if args.archive:
        write_archive(args.output, args.archive)
    return manifest


def verify_release(repo: Path, release: Path, jobs: Path,
                   design: dict[str, Any], base: dict[str, Any],
                   digest: str) -> dict[str, Any]:
    manifest_path = release / "headline_state_release_manifest.json"
    inventory_path = release / "source_inventory.csv"
    contents = release / "CONTENTS.sha256"
    require(all(path.is_file() for path in (manifest_path, inventory_path, contents)),
            "Frozen headline-state release is incomplete.")
    for line in contents.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        target = release / relative
        require(target.is_file() and sha256(target) == expected,
                f"Frozen release content changed: {relative}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["status"] ==
            "frozen_post_holdout_headline_state_ablation_v1" and
            manifest["contract_sha256"] == digest and
            manifest["jobs_sha256"] == sha256(jobs) and
            manifest["bundle_sha256"] == validate_bundle(repo, base)["sha256"] and
            manifest["original_training_source_inventory_sha256"] ==
            original_training_source_check(repo, design) and
            manifest["training_source_runtime"] ==
            design["training_source_runtime"] and
            manifest["training_runtime_sha256"] == sha256(
                repo / "protocol_manifests/training_python_runtime.json") and
            manifest["headline_weight_manifest_sha256"] ==
            headline_evidence(repo, design)[2],
            "Frozen release inputs differ from current exact evidence.")
    expected_rows = jobs_for(repo, design, base, digest,
                             manifest["bundle_sha256"])
    require(read_csv(jobs) == expected_rows, "Job rows differ from the frozen design.")
    runtime = repo / design["training_source_runtime"]
    frozen_inventory = read_csv(repo / design["original_training_release"] /
                                "source_inventory.csv")
    require((runtime / "data").resolve() == (repo / "data").resolve(),
            "Training data link does not point to the registered repository data.")
    for item in frozen_inventory:
        require(sha256(runtime / item["path"]) == item["sha256"],
                f"Isolated original-source runtime changed: {item['path']}")
    for item in read_csv(inventory_path):
        require(sha256(repo / item["path"]) == item["sha256"],
                f"Live source drifted after freeze: {item['path']}")
    return manifest


def train(args: argparse.Namespace, repo: Path, design: dict[str, Any],
          base: dict[str, Any], digest: str) -> dict[str, Any]:
    verify_release(repo, args.release, args.jobs, design, base, digest)
    require(not args.status.exists() and not args.log_root.exists(),
            "Training status or logs already exist; refusing overwrite.")
    require(args.train_python.is_file() and args.rscript.is_file(),
            "Training Python or Rscript does not exist.")
    gpus = [int(value) for value in args.gpus.split(",")]
    require(gpus and len(gpus) == len(set(gpus)) and min(gpus) >= 0 and
            args.cpu_cores >= len(gpus), "Invalid GPU or CPU allocation.")
    rows = read_csv(args.jobs)
    require(all(not (repo / row["output_dir"]).exists() for row in rows),
            "A job output directory already exists; never silently resume or replace.")
    args.log_root.mkdir(parents=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    # run_job resolves data/ through the runtime's symlink and executes from
    # the original frozen source snapshot. All job output paths still resolve
    # into the main repository's data/ tree.
    args.repo_root = repo / design["training_source_runtime"]
    args.config = (args.repo_root / "config/config.yaml").resolve()
    locks = {gpu: threading.Lock() for gpu in gpus}

    def assigned(pair: tuple[int, dict[str, str]]) -> dict[str, Any]:
        index, row = pair
        gpu = gpus[index % len(gpus)]
        with locks[gpu]:
            result = run_job(row, gpu, args, args.log_root,
                             max(1, args.cpu_cores // len(gpus)))
            print(json.dumps(result, sort_keys=True), flush=True)
            return result

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        results = [future.result() for future in as_completed(
            [pool.submit(assigned, pair) for pair in enumerate(rows)])]
    results.sort(key=lambda row: (row["experiment_id"], row["seed"]))
    with args.status.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    passed = sum(str(row["passed"]).lower() == "true" for row in results)
    return {"status": "complete" if passed == 30 else "failed",
            "jobs": 30, "passed": passed, "status_file": str(args.status)}


def audit(args: argparse.Namespace, repo: Path, design: dict[str, Any],
          base: dict[str, Any], digest: str) -> dict[str, Any]:
    verify_release(repo, args.release, args.jobs, design, base, digest)
    require(not args.output.exists(), "Checkpoint audit already exists.")
    jobs, statuses = read_csv(args.jobs), read_csv(args.status)
    job_keys = {(row["experiment_id"], int(row["seed"])) for row in jobs}
    status_keys = {(row["experiment_id"], int(row["seed"])) for row in statuses}
    require(len(job_keys) == len(status_keys) == 30 and job_keys == status_keys and
            all(str(row["passed"]).lower() in ("true", "1") and
                int(row["exit_code"]) == 0 for row in statuses),
            "Thirty exact jobs must pass before checkpoint audit.")
    try:
        import torch
    except ModuleNotFoundError as error:
        raise DoseProtocolError("GPU-training Python with PyTorch is required.") from error
    records = []
    for row in jobs:
        seed, arm = int(row["seed"]), row["experiment_id"]
        run = (repo / row["output_dir"]).resolve()
        checkpoint = run / f"{row['CHECKPOINT_PREFIX']}_full.pt"
        pretrain = run / f"{row['CHECKPOINT_PREFIX']}_pretrained.pt"
        gate = run / "pretraining_behavior_gate.csv"
        require(all(path.is_file() for path in (checkpoint, pretrain, gate)),
                f"Incomplete checkpoint files: {arm}/{seed}")
        failures = gate_failures(gate)
        full = torch.load(checkpoint, map_location="cpu", weights_only=True)
        before = torch.load(pretrain, map_location="cpu", weights_only=True)
        require(finite_tensors(full, torch) and finite_tensors(before, torch),
                f"Non-finite checkpoint tensor: {arm}/{seed}")
        expected_architecture = {
            "rl_algorithm": "td3", "policy_encoder": "lstm",
            "vine_observation_mode": row["VINE_OBSERVATION_MODE"],
            "vine_feature_mode": row["VINE_FEATURE_MODE"],
            "cvar_observation_mode": row["CVAR_OBSERVATION_MODE"],
            "cvar_reward_mode": "full",
            "pretrain_data_mode": "mixed_historical_synthetic",
            "pretrain_behavior_gate_mode": "report_only", "run_finetune": True,
        }
        require(all(full.get("architecture", {}).get(key) == value for key, value
                    in expected_architecture.items()) and
                int(before.get("total_actions", -1)) == 24000 and
                int(full.get("total_actions", -1)) == 25464,
                f"Training metadata/action count mismatch: {arm}/{seed}")
        records.append({
            "arm_id": arm, "seed": seed, "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "model_dir": str(run), "checkpoint_prefix": row["CHECKPOINT_PREFIX"],
            "all_tensors_finite": True, "behavior_gate_mode": "report_only",
            "behavior_gate_pass": not failures,
            "behavior_gate_failed_metrics": ";".join(failures),
            **{field: row[field] for field in JOB_ENV_FIELDS},
        })
    records.sort(key=lambda row: (row["arm_id"], row["seed"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".headline-state-audit.",
                                      dir=args.output.parent))
    try:
        table = temporary / "headline_state_checkpoint_audit.csv"
        with table.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
        manifest = {
            "status": "headline_state_checkpoint_audit_passed",
            "schema_version": 1, "job_count": 30, "experiment_count": 3,
            "seeds_per_experiment": 10, "all_checkpoint_tensors_finite": True,
            "all_checkpoint_metadata_match": True,
            "confirmatory_claim_permitted": False,
            "contract_sha256": digest, "jobs_sha256": sha256(args.jobs),
            "status_sha256": sha256(args.status),
            "checkpoint_audit_sha256": sha256(table),
        }
        (temporary / "headline_state_audit_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "CONTENTS.sha256").write_text("".join(
            f"{sha256(path)}  {path.name}\n" for path in sorted(temporary.iterdir())
            if path.is_file()), encoding="ascii")
        os.replace(temporary, args.output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def replay(args: argparse.Namespace, repo: Path, design: dict[str, Any],
           base: dict[str, Any], digest: str) -> dict[str, Any]:
    verify_release(repo, args.release, args.jobs, design, base, digest)
    require(not args.output.exists(), "Weight replay already exists.")
    manifest_path = args.audit / "headline_state_audit_manifest.json"
    table = args.audit / "headline_state_checkpoint_audit.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_csv(table)
    require(manifest["status"] == "headline_state_checkpoint_audit_passed" and
            manifest["contract_sha256"] == digest and
            manifest["checkpoint_audit_sha256"] == sha256(table) and
            len(rows) == 30 and args.policy_python.is_file() and
            args.rscript.is_file() and args.workers >= 1,
            "Replay requires exact passed audit and valid runtimes.")
    for row in rows:
        require(sha256(Path(row["checkpoint"])) == row["checkpoint_sha256"],
                f"Audited checkpoint changed: {row['arm_id']}/{row['seed']}")
    args.output.mkdir(parents=True)
    logs = args.output / "command_logs"
    logs.mkdir()

    runtime = repo / design["training_source_runtime"]

    def one(row: dict[str, str]) -> dict[str, Any]:
        arm, seed = row["arm_id"], int(row["seed"])
        destination = args.output / "weights" / arm / f"seed_{seed}"
        destination.mkdir(parents=True)
        environment = os.environ.copy()
        environment.update({field: row[field] for field in JOB_ENV_FIELDS})
        environment.update({
            "EVAL_MODEL_DIR": row["model_dir"],
            "EVAL_OUTPUT_DIR": str(destination.resolve()),
            "EVAL_WEIGHTS_ONLY": "true", "EVAL_CHECKPOINT_MODELS": "full",
            "EVAL_CHECKPOINT_PREFIX": row["checkpoint_prefix"],
            "EVAL_WINDOW_ID": "locked_oos_v1",
            "EVAL_GATE_AUTHORIZATION": "headline_state_checkpoint_audit_v1",
            "EVAL_HEADLINE_STATE_AUDIT_MANIFEST": str(manifest_path.resolve()),
            "EVAL_HEADLINE_STATE_CHECKPOINT_AUDIT": str(table.resolve()),
            "EVAL_HEADLINE_STATE_CHECKPOINT_SHA256": row["checkpoint_sha256"],
            "POLICY_INFERENCE_SERVER": "rl/policy_inference_server_v2.py",
            "POLICY_PYTHON": str(args.policy_python),
            "LC_ALL": "C", "LANG": "C", "LANGUAGE": "C", "TZ": "UTC",
        })
        label = f"{arm}__{seed}"
        stdout, stderr = logs / f"{label}.stdout.txt", logs / f"{label}.stderr.txt"
        # The new authorization launcher is the only live R file. All model,
        # scenario, and policy-inference source files come from the original
        # mixed experiment's frozen source snapshot.
        command = [str(args.rscript), "--vanilla",
                   str(repo / "evaluate_with_config.r"),
                   str(runtime / "config/config.yaml"), row["model_dir"]]
        with stdout.open("wb") as out, stderr.open("wb") as err:
            result = subprocess.run(command, cwd=runtime, env=environment,
                                    stdout=out, stderr=err, check=False)
        path = destination / f"weights_rl_full_seed_{seed}.csv"
        if result.returncode or not path.is_file():
            tail = stderr.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
            raise DoseProtocolError(f"Replay failed for {label}: {' | '.join(tail)}")
        return {"experiment_id": arm, "seed": seed,
                "path": str(path.relative_to(repo)), "sha256": sha256(path),
                "checkpoint_sha256": row["checkpoint_sha256"], "rows": 24}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        inventory = [future.result() for future in as_completed(
            [pool.submit(one, row) for row in rows])]
    inventory.sort(key=lambda row: (row["experiment_id"], row["seed"]))
    inventory_path = args.output / "headline_state_weight_manifest.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader(); writer.writerows(inventory)
    result = {"status": "headline_state_weight_replay_complete",
              "schema_version": 1, "policy_count": 30,
              "evidence_class": "post_holdout_explanatory",
              "confirmatory_claim_permitted": False,
              "contract_sha256": digest, "checkpoint_audit_sha256": sha256(table),
              "weight_manifest_sha256": sha256(inventory_path)}
    (args.output / "headline_state_replay_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "prepare", "train", "audit", "replay"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path, default=Path(
        "publication_pipeline_draft/config/headline_state_ablation_v1.json"))
    parser.add_argument("--jobs", type=Path, default=Path(
        "protocol_manifests/headline_state_jobs_v1.csv"))
    parser.add_argument("--release", type=Path, default=Path(
        "frozen_releases/headline_state_ablation_v1"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--runtime", type=Path, default=Path(
        "protocol_manifests/training_python_runtime.json"))
    parser.add_argument("--train-python", type=Path)
    parser.add_argument("--policy-python", type=Path)
    parser.add_argument("--rscript", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--cpu-cores", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    for key in ("contract", "jobs", "release", "runtime", "config", "output",
                "archive", "train_python", "policy_python", "rscript",
                "log_root", "status", "audit"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, (repo / value).resolve())
    try:
        design, base, digest = load_design(repo, args.contract)
        if args.command == "validate":
            result = {"status": "contract_valid", "contract_sha256": digest,
                      "new_policies": 30, "reused_headline_policies": 10}
        elif args.command == "prepare":
            require(args.output is not None, "prepare requires --output.")
            result = prepare(args, repo, design, base, digest)
        elif args.command == "train":
            require(all(getattr(args, key) is not None for key in
                        ("status", "log_root", "train_python", "rscript")),
                    "train requires status, log-root, train-python, rscript.")
            result = train(args, repo, design, base, digest)
        elif args.command == "audit":
            require(args.output is not None and args.status is not None,
                    "audit requires --output and --status.")
            result = audit(args, repo, design, base, digest)
        else:
            require(all(getattr(args, key) is not None for key in
                        ("output", "audit", "policy_python", "rscript")),
                    "replay requires output, audit, policy-python, rscript.")
            result = replay(args, repo, design, base, digest)
    except (DoseProtocolError, OSError, ValueError, KeyError, TypeError,
            json.JSONDecodeError) as error:
        print(f"HEADLINE STATE ABLATION FAILURE: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
