# Project evidence synthesis and compute decision

## Executive conclusion

The current evidence supports a **mixed but scientifically useful result**, not
a proof that the original full architecture is universally superior.

The frozen 20-seed NN-vine LSTM-TD3 ensemble is economically competitive and
risk efficient.  On the 22 complete locked periods it earned a 29.08% CAGR,
13.19% annual volatility, 2.02 Sharpe ratio, 6.64% maximum drawdown, and 28.05%
annual CRRA certainty equivalent.  It had the highest Sharpe ratio and lowest
volatility of the seven primary strategies, and its 59.67% total return was
close to the static-vine (64.50%), dynamic-NN-vine (61.44%), and DCC-GARCH
(58.79%) alternatives.  Its predeclared CRRA advantage over equal weight was
positive but not significant (paired block-bootstrap p=0.2347; interval crossed
zero), and the White reality-check p-value was 0.5142.  The primary superiority
hypothesis therefore failed.

The causal study does not imply that vine modelling is useless.  Its strongest
result is more specific: passing all raw NN-vine parameters into the policy
reduced performance relative to retaining only the vine-derived scenario-CVaR
observation.  The latter is still a vine-plus-RL model: the NN-vine remains the
scenario and tail-risk engine, the CVaR reward remains active, and synthetic
pretraining still uses NN-vine paths.  It removes only the 63-dimensional raw
vine state.

That compressed model earned 69.41% total return, 33.21% CAGR, 13.67% annual
volatility, 2.19 Sharpe, 6.72% maximum drawdown, 32.08% annual CRRA certainty
equivalent, and 0.393 mean monthly turnover.  Among the frozen six financial
benchmarks and the causal representations compared here, it had the highest
Sharpe and CRRA CE.  A same-calendar post-hoc reconciliation found positive
CRRA CE differences against all six benchmarks, ranging from +2.62 percentage
points versus static vine to +13.60 points versus rolling vine.  None survived
Holm correction.  Because this candidate was selected after inspecting the
consumed holdout, these comparisons are exploratory and cannot restore a fresh
confirmatory claim.

The most defensible paper is therefore about a **decision-aligned dynamic-vine
risk representation**: a dynamic NN-vine can be valuable as a compressed
scenario-risk engine, while raw high-dimensional dependence-state expansion
can overload the controller.  The current paper can claim competitive economic
performance, strong ensemble-level risk control, and an informative negative
causal result.  It cannot claim statistically proven universal superiority.

## Evidence layers

### 1. Frozen primary benchmark evaluation

- The original 20-seed ensemble passed all portfolio constraints.
- It dominated the primary table on Sharpe and volatility, and was competitive
  on return and CRRA CE.
- Its CRRA CE ranked behind static vine and dynamic NN-vine but ahead of DCC,
  equal weight, shrinkage mean-variance, and rolling vine.
- All paired superiority intervals included zero after time-series-aware
  inference and multiplicity control.
- The correct conclusion is economic competitiveness and risk efficiency, not
  statistical superiority.

### 2. Seed robustness and the ensemble mechanism

Individual policies were materially less stable than the ensemble.  Median
seed CAGR was 25.95%, median Sharpe 1.63, and median monthly turnover 0.539.
Only 45% of seeds beat DCC-GARCH on CAGR and 40% beat static vine; no seed beat
equal weight on maximum drawdown or realized CVaR.  The ensemble is therefore
not representative of a typical single trained policy.

This is not automatically a defect.  The ensemble is the preregistered
deployable strategy.  Weight averaging cancelled 94.50% of incremental gross
and short exposure, lowering gross exposure from a 1.402 mean across seeds to
1.022 and turnover from 0.575 to 0.317.  It saved implementation costs and
improved terminal wealth.  The paper must present policy ensembling as an
important mechanism rather than implying that every neural policy is safe.

### 3. Causal component analysis

No preregistered positive component effect survived Holm correction.  Direct
raw NN-vine state and synthetic NN-vine pretraining had intervals entirely in
the adverse direction.  Scenario-CVaR, joint dependence information, CVaR
reward shaping, recurrence, fine-tuning, and TD3's advantage over other RL
algorithms were not established.

The component table answers "does the reference full model beat each matched
alternative?"; it does not rank the alternatives against financial benchmarks.
The new post-hoc reconciliation fills that descriptive gap without changing
the inferential status of the consumed holdout.

### 4. What the causal alternatives actually imply

- `zero_vine_features_keep_cvar_observation` is a compressed vine-plus-RL
  policy, not a no-vine policy.  It is the strongest mechanism candidate.
- `zero_vine_features_and_cvar_observation` removes policy-visible dependence
  signals but still retains the vine generator and CVaR reward.  The compressed
  candidate beat this ensemble by 9.71 annual CRRA CE percentage points;
  a post-hoc paired block-bootstrap interval was [4.88, 16.33] points.  Eight
  of ten matched seed effects were positive.  This is strong explanatory
  support for the scalar scenario-CVaR signal, but it remains post-holdout.
- `historical_only_no_synthetic_pretraining` was the best observed ensemble,
  but all ten policies triggered the registered turnover warning and its mean
  monthly turnover was 0.614.  The common scorer already charged costs, so the
  result cannot be dismissed; it identifies synthetic-to-real domain shift as
  a priority limitation requiring independent validation.
- The broad five-algorithm comparison found no established TD3 advantage.
  Further algorithm sweeps have low expected scientific value.

## Five-day HPC plan

The remaining compute should answer one mechanism question well.  It should not
try every possible ablation.

### Day 1: preserve evidence and freeze the focused contract

1. Freeze one canonical 130-checkpoint causal training release before deleting
   the v2/v3/v4 run trees.  The frozen result archive preserves weights and
   results, but not enough checkpoint material to replay every policy.
2. Run the full server validation suite after the focused-CE correction.
3. Freeze the existing two-window, three-representation, five-matched-seed
   focused protocol.  Do not alter variants, seeds, windows, or hyperparameters
   after inspecting results.

### Days 1-3: run only the focused 30-policy walk-forward study

Use two deterministic, non-overlapping 24-month windows and exactly:

1. full raw vine state plus scenario-CVaR;
2. compressed scenario-CVaR only;
3. no policy-visible vine dependence observation.

Five matched seeds per variant and window give 30 trainings.  The design tests
whether the raw-state penalty and the compressed-CVaR advantage persist through
time.  It is retrospective robustness evidence, not fresh confirmation.

### Day 3: add benchmarks without additional neural training

Generate the same six causal financial benchmarks in both windows and score
them through common drifted-turnover and financing accounting.  This is
essential: the mechanism comparison and the economic benchmark comparison must
share dates, assets, costs, and realized returns.  The benchmark extension is
CPU work and must not expand the RL training matrix.

### Day 4: audit, infer, freeze

- Audit all 30 checkpoints and all constraint/accounting invariants.
- Report each window separately and a window-stratified pooled result.
- Compare compressed CVaR with full and no-visible-dependence variants as the
  two frozen mechanism contrasts.
- Compare the compressed ensemble descriptively with all financial benchmarks;
  apply a clearly declared multiplicity family.
- Freeze weights, scored panels, checkpoints (or a checkpoint release), code,
  environment manifests, tables, and logs.

### Day 5: operational reserve and CPU-only robustness

Reserve the day for failed operational jobs only.  If no retries are needed,
compute fixed-weight cost sensitivity (0/10/25/50 bps and short-borrow
0/3/6/10%), ensemble-size sensitivity, and figures from frozen paths.  Do not
train new algorithms or choose a new architecture from observed results.

## Stop/go rule after the focused study

- If compressed scenario-CVaR beats no-visible-dependence in both windows and
  remains economically competitive with the financial benchmarks, frame it as
  stable retrospective mechanism evidence and make it the recommended future
  architecture.
- If the sign differs across windows, report regime dependence and keep the
  original full model as the frozen primary result; do not run more variants.
- If it loses in both windows, reject the compressed-mechanism interpretation
  and emphasize the competitive but unproven primary ensemble result.

Regardless of outcome, do not call the two-window study confirmatory.  A new
market panel, a future unseen period, or a truly untouched dataset is required
for a new superiority claim.

## Three-week non-HPC plan

### Week 1: evidence-locked manuscript rewrite

- Rewrite the abstract, contribution, results, and conclusion around competitive
  risk-adjusted performance and decision-aligned compression.
- Separate confirmatory primary results, post-holdout causal evidence, and
  retrospective walk-forward evidence in every table caption.
- Explain the ensemble mechanism and seed instability explicitly.

### Week 2: publication tables and figures

- Main performance and inference table against all benchmarks.
- Seed distribution and ensemble-cancellation figure.
- Causal component forest plot, CAGR-turnover plot, and focused two-window
  mechanism plot.
- Cost/ensemble-size sensitivity from frozen weights.
- Reproducibility and compute table, including solver and constraint audits.

### Week 3: statistical and editorial audit

- Cross-check every manuscript number against a frozen CSV/JSON hash.
- Verify no seed is treated as an independent market history.
- Compile LaTeX in a clean environment and inspect every page.
- Write limitations: 22 complete monthly observations, consumed holdout,
  post-hoc compressed-model selection, synthetic domain gap, and external
  validation still required.
- Prepare anonymized code and a separate artifact release; keep large generated
  data and checkpoints in Git LFS or an archival repository, not ordinary Git.

## Explicitly deferred work

- another 130-policy causal sweep;
- more RL algorithms;
- broad hyperparameter sensitivity;
- 40-asset scalability training;
- a new "best" model selected on the consumed holdout;
- claims of confirmatory superiority from the retrospective windows.

These may be future-paper work.  They are not needed to complete a coherent,
honest paper with the remaining compute budget.
