#!/usr/bin/env bash
set -euo pipefail

# Read-only pre-freeze validation. Required executable paths are supplied by
# the caller so R-Lantern and Python-PyTorch remain process isolated.
: "${PYTHON:?Set PYTHON to the pinned orchestration Python}"
: "${RSCRIPT:?Set RSCRIPT to the pinned Rscript}"
: "${TRAIN_PYTHON:?Set TRAIN_PYTHON to the pinned CUDA training Python}"
: "${POLICY_PYTHON:?Set POLICY_PYTHON to the isolated policy-inference Python}"

export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC

"$PYTHON" -m compileall -q publication_pipeline_draft rl
"$PYTHON" -m pytest -q publication_pipeline_draft/tests
"$RSCRIPT" --vanilla tests/run_tests.r
"$RSCRIPT" --vanilla tests/test_publication_benchmarks.r
"$RSCRIPT" --vanilla tests/test_extended_publication_benchmarks.r
POLICY_PYTHON="$POLICY_PYTHON" \
  "$RSCRIPT" --vanilla tests/test_policy_process_isolation.r

"$TRAIN_PYTHON" - <<'PY'
import gymnasium
import numpy as np
import torch
from rl.recurrent_baselines import (
    BASELINE_IMPLEMENTATION_REVISION,
    BaselineConfig,
    build_agent,
)

assert BASELINE_IMPLEMENTATION_REVISION == "causal_retry_v3_20260812"

print({
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "gymnasium": gymnasium.__version__,
})
assert torch.cuda.is_available()
for algorithm in ("ddpg", "sac", "ppo", "a2c"):
    config = BaselineConfig(
        algorithm=algorithm, encoder="lstm", obs_dim=24, action_dim=7,
        seq_len=4, hidden=16, num_layers=1,
        lr_actor=3e-4, lr_critic=1e-3, gamma=1.0, tau=0.005,
        entropy_coef=0.005, grad_clip_norm=1.0,
        random_exploration_steps=10, replay_capacity=1000, policy_delay=2,
        target_policy_noise=0.1, target_noise_clip=0.2,
        diagnostic_interval=10, direction_logit_bound=4.0,
        projection_temperature=0.5,
        gross_leverage=1.5,
        net_exposure=1.0, max_long_weight=0.6, max_short_weight=0.2,
        initial_leverage_gate=0.5, leverage_soft_target=0.8,
        leverage_penalty_coef=0.25, short_borrow_rate=0.03,
        cash_borrow_rate=0.02, utility_mode="terminal_wealth_crra",
        vine_observation_mode="full", vine_feature_mode="full",
        cvar_observation_mode="full", cvar_reward_mode="full",
        pretrain_data_mode="vine_synthetic", run_finetune=True,
    )
    agent = build_agent(config, torch.device("cuda"))
    assert agent.config.algorithm == algorithm
    if algorithm in {"ppo", "a2c"}:
        state = np.zeros((4, 24), dtype=np.float32)
        for index in range(3):
            agent.select_action(state)
            agent.record_outcome(0.01 * (index + 1), index == 2, state)
        diagnostics = agent.finish_episode()
        assert diagnostics and np.isfinite(diagnostics["actor_loss"])

publication_mlp = BaselineConfig(**{
    **config.__dict__, "algorithm": "td3", "encoder": "mlp",
    "obs_dim": 88, "seq_len": 30, "hidden": 64, "num_layers": 1,
    "lr_actor": 3e-5, "lr_critic": 1e-4,
})
mlp = build_agent(publication_mlp, torch.device("cuda"))
assert abs(mlp.parameter_count() - mlp.capacity_target) / mlp.capacity_target <= 0.05
print("Modern RL construction smoke passed.")
PY

temporary="$(mktemp -d /tmp/publication-extension-v2.XXXXXX)"
trap 'rm -rf -- "$temporary"' EXIT
"$PYTHON" publication_pipeline_draft/publication_research_program.py validate \
  --program publication_pipeline_draft/config/publication_research_program_v2.json
"$PYTHON" publication_pipeline_draft/publication_research_program.py jobs \
  --program publication_pipeline_draft/config/publication_research_program_v2.json \
  --output "$temporary/program_jobs.csv"
"$PYTHON" publication_pipeline_draft/causal_ablation_protocol.py \
  --output-root data/publication_extension_runs_v3 \
  --output "$temporary/causal_jobs.csv"
"$PYTHON" - "$temporary" <<'PY'
import csv
import json
import sys
from pathlib import Path
from publication_pipeline_draft.freeze_publication_extension import SOURCES

root = Path(sys.argv[1])
with (root / "program_jobs.csv").open(newline="") as handle:
    program = list(csv.DictReader(handle))
with (root / "causal_jobs.csv").open(newline="") as handle:
    causal = list(csv.DictReader(handle))
assert len(program) == 560
assert len(causal) == 130
assert len({(row["experiment_id"], row["seed"]) for row in causal}) == 130
assert len(SOURCES) == len(set(SOURCES))
missing = [source for source in SOURCES if not Path(source).is_file()]
assert not missing, missing
universe = json.loads(Path(
    "publication_pipeline_draft/config/scalability_universe_v1.json"
).read_text())
assert len(universe["asset_order"]) == 40
print({"program_jobs": 560, "causal_jobs": 130,
       "frozen_sources": len(SOURCES), "scalability_assets": 40})
PY

echo "Publication extension v2 validation passed."
