# Publication extension v4 retry-29 training evidence

This note records the outcome-blind training diagnostics inspected before any
v4 causal checkpoint was replayed on the consumed 24-month evaluation panel.
It does not report portfolio returns or causal contrasts.

## Preserved inputs

- `publication_extension_v4_retry29_diagnostics.tar.gz`:
  `842c346905e38b82fce17a95b55125f0627e8d0b6fb8d956cdc8a258d9f3b003`
- `publication_extension_v4_retry29.launch.log`:
  `d243fbb60141a06b5b0ed0a60f7ac52a6ca1cce63a03292426f735c99145b5d7`
- `causal_sweep_status_v4_retry29.csv`:
  `f4e915270ecee1c2cfe1edc5174a15394c89035f75bf95d3e6d59c936c8a30ea`

## Operational findings

- The status contains exactly 29 unique experiment/seed rows.
- All 29 rows have exit code zero, a final checkpoint, a behavior-gate file,
  and `passed=true`.
- The slice comprises 10 historical-only controls, 10 moving-block-bootstrap
  controls, 2 capacity-matched feedforward controls, and 7 recurrent SAC
  controls.
- The diagnostics archive contains 29 stdout and 29 stderr files. All stderr
  files are empty; every stdout reaches `TRAINING COMPLETE`, completes the
  fixed one-pass historical fine-tune, and explicitly reports the economic
  pretraining diagnostic under the frozen `report_only` rule.
- No traceback, non-finite diagnostic, CUDA out-of-memory condition, hard
  constraint failure, or silent fallback was found.

## V3-to-v4 reproducibility check

For every one of the exact 29 v3-failed keys, the numeric failed-gate string in
the v3 stderr log was compared with the v4 report-only diagnostic string after
removing only the enforcement wording. All 29 pairs match exactly. This is
consistent with the frozen-source audit: v4 changed the gate branch, not the
model settings, seed, pretraining data, or pretraining trajectory.

The 29 controls did not pass every economic behavior diagnostic: 22 reported
turnover above the preregistered threshold and 7 SAC controls reported low
directional entropy and/or too few effective positions. They remain in the
causal design under the prospective intent-to-train/report-only rule to avoid
selecting ablation controls on favorable policy behavior. The final causal
tables must disclose these warnings separately from the fail-closed eligibility
checks for finite tensors, exposure constraints, and position constraints.

