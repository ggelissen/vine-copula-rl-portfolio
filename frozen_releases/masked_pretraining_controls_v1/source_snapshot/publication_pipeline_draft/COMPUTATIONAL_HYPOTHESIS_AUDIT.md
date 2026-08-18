# Computational hypothesis audit and terminal decision

This audit separates questions that the existing evidence answers from claims
that cannot be repaired by additional optimization seeds on the same realized
market path.

| Research question | Evidence status before the terminal controls | Decision |
|---|---|---|
| Are NN-vine simulations marginally, cross-sectionally, temporally, and tail-risk compatible with the training prefix? | Addressed by the frozen synthetic diagnostics. | No new generator tuning on the evaluation period. |
| Is recurrent TD3 training numerically stable and constraint compliant across seeds? | Addressed by checkpoint, behavior-gate, no-holdout, and 20-seed audits. | Report ensemble and individual-seed dispersion. |
| Does the frozen full model dominate financial benchmarks? | Answered negatively as a universal claim: economic performance is competitive, but predeclared superiority is not statistically established. | Preserve the mixed confirmatory result. |
| Which policy-visible dependence representation is useful? | Addressed by causal masking and focused two-window analysis; raw high-dimensional state is not uniformly beneficial and compressed/masked representations are more competitive. | Do not search additional state variants on the consumed holdout. |
| Is performance sensitive to synthetic path count and presentation count? | Addressed by 100/100 and 100-unique/1,000-presentation experiments. The interaction is nonlinear. | Do not add an ex post dose grid. |
| Does the selected 100-path/1,000-presentation masked policy benefit specifically from NN-vine pretraining? | **Unresolved:** prior historical and moving-block controls used a different full-state architecture. | Train the two matched masked controls in this release. |
| Is superiority externally valid or genuinely confirmatory? | Unresolved. Ten seeds do not create independent market histories, and the holdout is consumed. | Requires a future period or independent panel; not another same-sample HPC sweep. |

The terminal masked-control study is therefore the highest-value feasible GPU
experiment. It isolates pretraining source while holding architecture,
interaction budget, seeds, fine-tuning, reward, constraints, costs, and realized
returns fixed. Regardless of outcome, the same-holdout neural-training program
stops after this experiment.

Remaining computations are deterministic post-processing of frozen weights:
common accounting, paired block-bootstrap inference, cost and leverage
sensitivity where not already available, publication tables, and figures.
