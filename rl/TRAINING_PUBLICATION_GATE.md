# RL training publication gate

## Why the 20260741 checkpoint is rejected

The checkpoint is numerically intact, but the no-holdout diagnostic is not
evidence of generalisation. On the 61 overlapping historical trajectories used
for fine-tuning, the full policy produced median terminal wealth of 335,645,
used approximately 1.5x gross exposure and a 25% short book at every step, and
had median monthly turnover of 2.20. It selected the ex-post best asset in
45.5% of duplicated decisions and the ex-post worst asset in 33.4%, versus a
seven-asset naive rate of 14.3%. Typical allocation sensitivity to the vine
state and previous holdings fell after fine-tuning. This is severe in-sample
trajectory memorisation, not a publication result.

The subsequent schema-2 rerun is also rejected.  Its optimiser and checkpoint
tensors remained finite, but the last 100 synthetic episodes averaged 1.4996x
gross exposure, placed approximately 124.9% in one asset, and used the gross
cap on 84.8% of decisions.  The historical refit then used the cap on 95.2%
of decisions.  The actor's raw leverage output was saturated even for zero and
random probe states.  This is action-boundary collapse, not evidence that
dynamic leverage was learned.

The schema-3 pre-training run is numerically healthier but is rejected for a
different structural reason. Its last 100 episodes had a raw leverage gate of
0.9994 while mean realised gross exposure was only 1.2976x and gross-cap use
was zero. Independent long and short books were allowed to overlap and cancel,
so the gate did not identify realised leverage. The Euclidean capped-simplex
map also behaved as sparsemax and drove the maximum asset weight to roughly
59.85% against a 60% limit. Lowering the gate threshold would hide both faults.

The schema-4 run fixed the leverage identity and remained numerically stable,
but its clipped capped-softmax increasingly selected an exact long-position
boundary. Its former `fraction_at_position_cap` gate was not statistically
well-founded: tightening a valid cap can mechanically increase this fraction,
whereas loosening the cap can make the gate pass while permitting more
concentration. Schema 5 therefore removes the clipping corner and tests
diversification directly with entropy and effective positions.

## Corrected training protocol

1. All generated synthetic episodes are used once for TD3 pre-training.
2. The pretrained checkpoint persists its action counter, optimiser state and
   independent AMP scaler states. Random warm-up therefore does not restart at
   historical fine-tuning.
3. Historical fine-tuning uses one preregistered pass.  The last 24 months of
   the *training prefix* remain a purged diagnostic, but a single realised path
   no longer adaptively selects among up to eight pass counts.  The separately
   locked final 24 months remain unread.
4. A fresh copy of the pretrained agent is refit once using every historical
   trajectory in a deterministic seeded permutation.
5. Fine-tuning uses a lower learning rate, less exploration and one update per
   environment step. These settings reduce adaptation capacity; their final
   choice still requires multi-seed validation and sensitivity reporting.
6. Checkpoint schema 5 uses one cross-sectional asset-score vector and an
   explicit leverage gate. The two lowest-ranked assets form a disjoint short
   support and the remaining assets form the long support. A smooth interior
   logistic capped-simplex map with temperature 1.5 enforces 60% maximum long
   and 20% maximum short weights while retaining useful gradients near the
   limits. Disjoint books make realised gross exposure
   exactly `abs(net) + gate * (gross_cap - abs(net))`, so the gate is identifiable.
   A conservative initialization, allocation-entropy regularizer and soft
   penalty above 80% of the admissible leverage range discourage a constant
   boundary policy while retaining the 1.5x hard cap for state-dependent use.
7. Episode and sampled-update diagnostics are written as structured CSV files.
   Non-finite actions, rewards, actor losses or critic losses abort training.
8. `pretraining_behavior_gate.csv` is fail-closed and is computed from the
   final actor without exploration noise on a fixed held-in synthetic slice.
   Fine-tuning is not allowed when preregistered leverage, gate/gross identity,
   gross-cap, turnover, true position-limit, normalized book-entropy or
   effective-position requirements fail. Exact contact with a valid hard
   position limit is retained in a warning file, not misclassified as a
   constraint violation or proof of policy collapse.
9. Full float32 is the publication default.  Mixed precision remains an
   explicit benchmark option, not a silent source of skipped actor updates.

## Required artifacts after every run

- `debug_output.txt`
- `training_episode_metrics.csv`
- `training_update_metrics.csv`
- `pretraining_behavior_gate.csv`
- `pretraining_policy_diagnostics.csv`
- `pretraining_behavior_warnings.csv`
- `finetune_validation_metrics.csv`
- `finetune_selection.txt`
- `finetune_episode_schedule.csv`
- `td3_lstm_vine_pretrained.pt`
- `td3_lstm_vine_full.pt`
- `run_manifest.rds`, code hashes and data hashes
- `source_snapshot/` containing the exact code/configuration bytes used by the run
- the complete `sanity_no_holdout/` directory

The no-holdout sanity report fails its overall gate whenever it emits a
structural behavioural warning. It rechecks post-fine-tuning diversification,
effective holdings, dynamic leverage and state sensitivity. Overlapping
historical trajectories are collapsed to unique calendar decisions for
ex-post alignment diagnostics, avoiding pseudo-replication. In-sample reward
and wealth comparisons are reported as non-gating diagnostics: using them to
select a policy would reward training-set overfitting. Numerical constraint
compliance alone remains insufficient.

After one corrected seed has been inspected, run the preregistered sequential
replications with:

```bash
LC_ALL=C LANG=C LANGUAGE=C TZ=UTC \
Rscript --vanilla rl/run_seed_sweep.r config/config.yaml
```

This runner trains and sanity-checks every seed and writes
`data/rl_runs/seed_sweep_status.csv`, including pretrained-to-full reward and
wealth deltas plus final-policy leverage, diversification, turnover and state-
sensitivity diagnostics. It intentionally never opens the final OOS
evaluation.

## Publication claims that remain prohibited

Do not claim superiority from one seed, from the overlapping in-sample sanity
paths, or from repeatedly inspecting the final 24-month holdout. The minimum
research package is:

- a preregistered configuration and primary benchmark;
- at least 20 independent training seeds with uncertainty across seeds;
- purged/embargoed expanding-window development folds;
- one like-for-like realised-return evaluation contract for every method;
- turnover, leverage, borrow and transaction costs applied identically;
- paired utility tests with multiplicity correction and a backtest-overfitting
  assessment across the tested configurations;
- ablations that are genuinely rerun, never hard-coded or simulated fallbacks.

Methodological anchors include Fujimoto, van Hoof and Meger (2018),
"Addressing Function Approximation Error in Actor-Critic Methods" (TD3),
Henderson et al. (2018), "Deep Reinforcement Learning that Matters"
(multi-seed reproducibility), and Bailey et al. (2016), "The Probability of
Backtest Overfitting" (selection bias in financial backtests).
