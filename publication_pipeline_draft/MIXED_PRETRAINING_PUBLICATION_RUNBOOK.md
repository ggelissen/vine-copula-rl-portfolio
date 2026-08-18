# Mixed-pretraining evidence freeze and publication extension

This workflow is terminal and additive. It registers the completed four-arm
experiment, performs frozen-weight diagnostics, and generates one manuscript
figure plus one appendix table. It performs no training or model selection.
The archive does not retain all 24 target-weight transitions for every reused
comparator, so no approximate transaction-cost counterfactual is emitted.

## HPC workflow

```bash
cd /gabirel/copula-portfolio-clean
export LC_ALL=C LANG=C LANGUAGE=C TZ=UTC
PYTHON=/gabirel/miniforge3/bin/python3

bash hpc/run_mixed_pretraining_response_v1.sh freeze-evidence \
  | tee logs/mixed_pretraining_response_v1_freeze_evidence.log

bash hpc/run_mixed_pretraining_response_v1.sh publication \
  | tee logs/mixed_pretraining_response_v1_publication.log

(
  cd frozen_releases/mixed_pretraining_response_v1_evidence_v1
  sha256sum -c CONTENTS.sha256
)

(
  cd manuscript_revision_causal_v1/publication_mixed_pretraining_v1
  sha256sum -c CONTENTS.sha256
)

"$PYTHON" -m pytest -q \
  publication_pipeline_draft/tests/test_mixed_pretraining_publication_artifacts.py \
  publication_pipeline_draft/tests/test_mixed_pretraining_protocol.py
```

The existing archive names are assumed:

- `mixed_pretraining_response_v1_final.tar.gz`
- `mixed_pretraining_response_v1_checkpoints.tar.gz`
- their corresponding `.sha256` sidecars.

## Local PowerShell workflow

Use a Python environment containing `numpy`, `pandas`, and `matplotlib`:

```powershell
cd C:\Users\gabri\Downloads\copula-based_dynamic_portfolio_selection
$PYTHON = "C:\Users\gabri\anaconda3\python.exe"

& $PYTHON -m publication_pipeline_draft.freeze_mixed_pretraining_evidence `
  --final-archive C:\Users\gabri\Downloads\mixed_pretraining_response_v1_final.tar.gz `
  --final-checksum C:\Users\gabri\Downloads\mixed_pretraining_response_v1_final.tar.gz.sha256 `
  --checkpoint-archive C:\Users\gabri\Downloads\mixed_pretraining_response_v1_checkpoints.tar.gz `
  --checkpoint-checksum C:\Users\gabri\Downloads\mixed_pretraining_response_v1_checkpoints.tar.gz.sha256 `
  --output frozen_releases\mixed_pretraining_response_v1_evidence_v1

& $PYTHON -m publication_pipeline_draft.generate_mixed_pretraining_publication_artifacts `
  --repo-root . `
  --evidence-release frozen_releases\mixed_pretraining_response_v1_evidence_v1 `
  --realized analysis_outputs\oos_v4_verified_770d2944\main_oos_v4_operational_retry\inputs\realized_asset_gross.csv `
  --output manuscript_revision_causal_v1\publication_mixed_pretraining_v1
```

## Manuscript placement

- Main text: use `figure_mp01_mixed_pretraining_evidence.tex` in the
  post-holdout pretraining-mechanism subsection. It replaces, rather than
  supplements, the older pretraining-source trade-off figure.
- Appendix: use `table_mp01_four_arm_performance.tex` for exact values.
- Online supplement: retain the exact leave-one-seed-out CSVs.
- Author control: append the six MP claims to the claim ledger. Do not merge
  their evidence class with the frozen primary benchmark evaluation.
