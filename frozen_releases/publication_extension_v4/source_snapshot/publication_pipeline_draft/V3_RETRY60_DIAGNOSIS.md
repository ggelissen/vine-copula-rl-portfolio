# V3 retry-60 diagnostic determination

## Evidence identity

- `causal_sweep_status_v3_retry60.csv`: `012a91226cb23962112ba0b51250b4de862c649f300f2a500bf69b73b05e0053`
- `publication_extension_v3_retry60.launch.log`: `82ea1dce3cbca32065e1af1d51f616107261eae6b11580408b60fd8fddb21946`
- `publication_extension_v3_retry.tar.gz`: `a24ed9240dcbfb7f029085612c4b7feda84d7645c2b3f3b4a2c8cb41ceb409e2`
- `publication_extension_v3_retry60_diagnostics.tar.gz`: `4ea50204e8d9489ea622e2da90a9f886c8f53583b3ad8c63e5c037ca5530d727`

## Result

Thirty-one of 60 jobs completed: A2C 10/10, PPO 10/10, capacity-matched
feed-forward TD3 8/10, SAC 3/10, historical-prefix pretraining 0/10, and
moving-block-bootstrap pretraining 0/10.

The completed policies were internally coherent on the deterministic gate
slice.  A2C averaged turnover 0.0688, normalized direction entropy 0.9892, and
5.0375 effective positions.  PPO averaged 0.1541, 0.9727, and 4.7592,
respectively.  The eight completed feed-forward policies averaged turnover
0.8935, entropy 0.8532, and 4.0763 effective positions.  The three completed
SAC policies averaged turnover 0.3719, entropy 0.7416, and 3.0003 effective
positions.  All completed runs reported zero gross-gate error and zero maximum
position-limit violation at displayed precision.

All 29 failures were finite economic behavior diagnostics.  No log reported a
non-finite diagnostic, `gate_gross_mae` failure, or
`max_position_limit_violation` failure.  The historical-prefix control failed
only mean turnover (mean 1.232052; range 1.03411--1.47379).  The moving-block
control failed only mean turnover (mean 1.368474; range 1.12975--1.58481).
Two feed-forward seeds marginally exceeded the turnover threshold (mean
1.02045; range 1.00564--1.03526).  Seven SAC seeds were too concentrated; six
reported mean normalized direction entropy below 0.7 (mean 0.668595; range
0.636699--0.696237), and all seven reported effective positions below 2.5
(mean 2.227529; range 2.11072--2.36528).

## Root cause and disposition

The frozen v3 trainer had SHA-256
`0dd04dae25d57fc84a48937555d9d29bed713ef8dac5b6443fb5581caa46a2fa`
and did not read or attest `PRETRAIN_BEHAVIOR_GATE_MODE`.  The v3 job contract
registered `report_only`, but the trainer therefore used the legacy strict
default and terminated before historical fine-tuning.  The failure is an
operational release-wiring defect, not evidence of training crashes or hard
constraint violations.

V3 is preserved as disclosed diagnostic evidence.  Its 31 strict-path
successes may be retained because every economic and structural gate passed and
the source diff is confined to the report-only branch and its metadata.  V4
reruns the exact 29 v3 failures with the same seeds and scientific settings
under fail-closed report-only wiring.  The final mixed-revision panel combines
70 strict-gate v2 successes, 31 strict-path v3 successes, and 29 completed v4
retries through a dedicated three-release merger.  Economic diagnostic failures remain reportable
outcomes and are not converted into passes; only their use as a differential
seed-selection barrier is removed.
