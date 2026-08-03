# Prepare the data protocol used by the two-stage RL experiment.
# Stage 1: synthetic vine-copula episodes only.
# Stage 2: realised historical episodes, excluding the final evaluation horizon.
# Run from the project root: Rscript --vanilla rl/synthetic_returns.r config/config.yaml

suppressPackageStartupMessages({ library(data.table); library(ggplot2); library(rvinecopulib); library(yaml) })
qsave_portable <- function(object, file) {
  if (requireNamespace("qs", quietly = TRUE)) qs::qsave(object, file, preset = "fast")
  else saveRDS(object, file = file, compress = FALSE)
}
source("rl/rl_environment.r")
source("helper/time_split.r")

env_or <- function(name, default, fun = identity) {
  value <- Sys.getenv(name, unset = "")
  if (!nzchar(value)) return(default)
  tryCatch(fun(value), error = function(e) default)
}

args <- commandArgs(trailingOnly = TRUE)
config_file <- if (length(args)) args[1L] else "config/config.yaml"
if (!file.exists(config_file)) stop(sprintf("Config file not found: %s", config_file))
config <- yaml.load_file(config_file)
seed <- env_or("TRAIN_SEED", config$general$seed, as.integer)
output_file <- env_or("SYNTHETIC_RETURNS_FILE", "data/synthetic_returns.RData")
pretrain_file <- env_or("PRETRAIN_RETURNS_FILE", config$vine$pretrain_returns_file)
finetune_file <- env_or("FINETUNE_RETURNS_FILE", config$vine$finetune_returns_file)
training_marginals_file <- env_or("TRAINING_MARGINALS_FILE", config$vine$training_marginals_file)
diag_dir <- env_or("SYNTHETIC_DIAGNOSTICS_DIR", "data/synthetic_diagnostics")
distribution_figure <- env_or("SYNTHETIC_DISTRIBUTION_FIGURE", "figures/synthetic_monthly_return_distributions.pdf")
n_sim_cvar <- env_or("N_SIM_CVAR", config$vine$n_sim_cvar, as.integer)
sim_cores <- env_or("VINE_SIM_CORES", config$vine$sim_cores, as.integer)
available_physical_cores <- parallel::detectCores(logical = FALSE)
if (is.finite(available_physical_cores)) sim_cores <- min(sim_cores, available_physical_cores)
L <- env_or("L", config$vine$L, as.integer); ref_col <- env_or("REF_COL", config$vine$ref_col, as.integer)
pretrain_episodes <- env_or("PRETRAIN_EPISODES", config$pretraining$episodes, as.integer)
episode_length <- env_or("ENV_T", config$environment$T, as.integer)
sequence_length <- env_or("ENV_SEQ_LEN", config$environment$seq_len, as.integer)
evaluation_periods <- env_or("EVALUATION_PERIODS", config$evaluation$periods, as.integer)
vine_model <- if (is.null(config$vine$model)) "nn_dynamic_t_vine" else config$vine$model
nn_vine_epochs <- env_or("NN_VINE_EPOCHS", if (is.null(config$vine$nn_epochs)) 200L else config$vine$nn_epochs, as.integer)
nn_vine_lr <- env_or("NN_VINE_LR", if (is.null(config$vine$nn_learning_rate)) 1e-3 else config$vine$nn_learning_rate, as.numeric)
nn_vine_patience <- env_or("NN_VINE_PATIENCE", if (is.null(config$vine$nn_patience)) 20L else config$vine$nn_patience, as.integer)
nn_vine_model_dir <- env_or("NN_VINE_MODEL_DIR", config$vine$nn_model_dir)

if (any(c(n_sim_cvar, sim_cores, L, pretrain_episodes, episode_length, sequence_length, evaluation_periods) < 1L)) stop("Simulation settings must be positive.")
if (evaluation_periods != 24L) stop("Publication protocol locks the historical OOS holdout to exactly 24 monthly holding periods.")
for (directory in unique(c(dirname(output_file), dirname(pretrain_file), dirname(finetune_file), dirname(training_marginals_file), dirname(distribution_figure), diag_dir))) dir.create(directory, recursive = TRUE, showWarnings = FALSE)
set.seed(seed)

returns <- load_returns()
period_split <- split_monthly_periods(build_monthly_periods(returns, min_history = L), evaluation_periods)
validate_period_split(period_split, evaluation_periods)
train_periods <- period_split$train; eval_periods <- period_split$evaluation
train_dates <- train_periods$decision_date; eval_dates <- eval_periods$decision_date
train_end <- tail(train_periods$decision_idx, 1L)

# Refit all marginal models on the training prefix.  The full-sample RData is
# never used for the RL state/generator after this point.
source("helper/marginals.r")
marginals <- fit_marginals_training(returns[seq_len(train_end), ])
U <- training_pseudo_observations(returns, marginals)
asset_names <- colnames(returns)
training_cutoff <- tail(train_dates, 1L)
save(marginals, U, asset_names, training_cutoff, train_end, file = training_marginals_file)
if (!identical(vine_model, "nn_dynamic_t_vine")) stop("Synthetic generation supports only the NN-driven dynamic t-vine; rolling vines are disabled.")
nn_states <- filter_training_marginals(returns, marginals)
nn_fit <- fit_nn_dynamic_vine(U[seq_len(train_end), , drop = FALSE],
  nn_states$z[seq_len(train_end), , drop = FALSE], nn_states$sigma[seq_len(train_end), , drop = FALSE],
  epochs = nn_vine_epochs, lr = nn_vine_lr, patience = nn_vine_patience)
save_nn_dynamic_vine_fit(nn_fit, nn_vine_model_dir)
vine_seq_train <- build_nn_vine_sequence(nn_fit, U, nn_states$z, nn_states$sigma, train_dates, index(returns))
if (length(vine_seq_train) != length(train_dates)) stop("Failed to create every training NN-vine snapshot.")
sim <- build_simulator(marginals, asset_names, ref_col = ref_col)

# These are the actual monthly holding-period outcomes available for fine
# tuning.  They are also the calibration target for synthetic pre-training.
all_train_gross <- do.call(rbind, lapply(seq_len(nrow(train_periods)), function(i) {
  as.numeric(realised_gross_for_period(returns, train_periods$decision_date[i], train_periods$holding_end_date[i]))
}))
colnames(all_train_gross) <- asset_names
historical_log_sorted <- lapply(seq_len(ncol(all_train_gross)), function(j) sort(log(all_train_gross[, j])))
names(historical_log_sorted) <- asset_names
historical_log_returns <- log(all_train_gross)
monthly_ar1 <- vapply(seq_along(asset_names), function(j) {
  estimate <- cor(head(historical_log_returns[, j], -1L),
                  tail(historical_log_returns[, j], -1L), use = "complete.obs")
  if (!is.finite(estimate)) estimate <- 0
  pmax(pmin(estimate, 0.5), -0.5)
}, numeric(1))
names(monthly_ar1) <- asset_names

# The NN-driven t-vine snapshots above generate both pre-training and
# fine-tuning scenarios. The monthly empirical transform below calibrates their
# marginal scale and tails without replacing their dependence model.
pretrain_vine <- vine_seq_train[[1L]]

# Sample an NN-vine snapshot, then map each marginal rank to the empirical
# in-sample MONTHLY return distribution. This preserves dynamic cross-sectional
# rank dependence while calibrating marginal scale, drift, and tails to the
# training holding period.
simulate_calibrated_monthly_gross <- function(vine, n_draws,
                                               previous_log_returns = NULL) {
  u <- rvinecop(n_draws, vine, cores = sim_cores)
  if (!is.null(previous_log_returns)) {
    previous_latent <- vapply(seq_along(asset_names), function(j) {
      sorted_log <- historical_log_sorted[[asset_names[j]]]
      probability <- (findInterval(previous_log_returns[j], sorted_log) + 0.5) /
        (length(sorted_log) + 1)
      qnorm(pmin(pmax(probability, 1e-6), 1 - 1e-6))
    }, numeric(1))
    latent <- sweep(qnorm(pmin(pmax(u, 1e-6), 1 - 1e-6)), 2L,
                    sqrt(1 - monthly_ar1^2), "*")
    latent <- sweep(latent, 2L, monthly_ar1 * previous_latent, "+")
    u <- pnorm(latent)
  }
  out <- matrix(NA_real_, nrow = n_draws, ncol = length(asset_names), dimnames = list(NULL, asset_names))
  for (j in seq_along(asset_names)) {
    sorted_log <- historical_log_sorted[[asset_names[j]]]
    probabilities <- seq_len(length(sorted_log)) / (length(sorted_log) + 1)
    out[, j] <- exp(approx(probabilities, sorted_log, xout = u[, j], rule = 2)$y)
  }
  out
}

# A stationary Gaussian AR copula supplies the monthly serial dependence that
# an LSTM is expected to learn. Its innovations retain the contemporaneous
# NN-vine dependence; the empirical transform restores each monthly marginal.
simulate_calibrated_monthly_path <- function(vines, n_draws) {
  innovations <- lapply(vines, function(vine) {
    qnorm(pmin(pmax(rvinecop(n_draws, vine, cores = sim_cores), 1e-6), 1 - 1e-6))
  })
  latent_previous <- NULL
  lapply(seq_along(innovations), function(t) {
    latent <- if (is.null(latent_previous)) innovations[[t]] else
      sweep(latent_previous, 2L, monthly_ar1, "*") +
      sweep(innovations[[t]], 2L, sqrt(1 - monthly_ar1^2), "*")
    latent_previous <<- latent
    u <- pnorm(latent)
    out <- matrix(NA_real_, nrow = n_draws, ncol = length(asset_names),
                  dimnames = list(NULL, asset_names))
    for (j in seq_along(asset_names)) {
      sorted_log <- historical_log_sorted[[asset_names[j]]]
      probabilities <- seq_len(length(sorted_log)) / (length(sorted_log) + 1)
      out[, j] <- exp(approx(probabilities, sorted_log, xout = u[, j], rule = 2)$y)
    }
    out
  })
}

# Synthetic data is used only in pre-training. Every realised/scenario return
# is newly simulated; no realised historical outcome enters this object. Each
# synthetic episode follows a causal sequence of NN-vine dependence snapshots,
# so the LSTM sees time-varying dependence regimes without a rolling refit.
generate_synthetic_pretrain <- function(n_episodes) {
  total_length <- sequence_length + episode_length
  n_starts <- length(vine_seq_train) - total_length + 1L
  if (n_starts < 1L) stop("Not enough NN-vine snapshots for one synthetic episode.")
  episodes <- vector("list", n_episodes)
  pb <- txtProgressBar(min = 0, max = n_episodes, style = 3); on.exit(close(pb), add = TRUE)
  for (ep in seq_len(n_episodes)) {
    vine_start <- sample.int(n_starts, 1L)
    episode_vines <- vine_seq_train[vine_start:(vine_start + total_length - 1L)]
    steps <- simulate_calibrated_monthly_path(episode_vines, n_sim_cvar + 1L)
    all_states <- lapply(episode_vines, extract_vine_state)
    action_idx <- (sequence_length + 1L):total_length
    episodes[[ep]] <- list(
      # Burn-in is state context only; retaining 2,000 unused CVaR rows per
      # burn-in month inflated the bundle by several gigabytes.
      burnin_returns = lapply(steps[seq_len(sequence_length)], function(x) x[1L, ]),
      burnin_vine_states = all_states[seq_len(sequence_length)],
      returns = steps[action_idx], vine_states = all_states[action_idx],
      vine_start = vine_start + sequence_length, source = "synthetic_nn_vine")
    setTxtProgressBar(pb, ep)
  }
  episodes
}

# Fine-tuning sees only realised historical returns.  The scenario rows are
# ex-ante simulations used for CVaR, while row one is the actual return that
# updates wealth and reward.  Overlapping episodes are intentional: they make
# every permissible historical 24-month trajectory available to replay.
generate_historical_finetune <- function() {
  total_length <- sequence_length + episode_length
  n_start <- length(train_dates) - total_length + 1L
  if (n_start < 1L) stop("Training history is shorter than one fine-tuning episode.")
  episodes <- vector("list", n_start)
  for (start in seq_len(n_start)) {
    action_start <- start + sequence_length
    steps <- vector("list", episode_length)
    for (step in seq_len(episode_length)) {
      index <- action_start + step - 1L
      scenarios <- simulate_calibrated_monthly_gross(
        vine_seq_train[[index]], n_sim_cvar,
        previous_log_returns = log(all_train_gross[index - 1L, ]))
      steps[[step]] <- rbind(all_train_gross[index, ], scenarios)
    }
    burn_idx <- start:(action_start - 1L)
    action_idx <- action_start:(action_start + episode_length - 1L)
    episodes[[start]] <- list(
      burnin_returns = lapply(burn_idx, function(i) all_train_gross[i, ]),
      burnin_vine_states = lapply(vine_seq_train[burn_idx], extract_vine_state),
      returns = steps, vine_states = lapply(vine_seq_train[action_idx], extract_vine_state),
      vine_start = action_start, source = "historical_realised")
  }
  episodes
}

cat(sprintf("Generating %d synthetic NN-vine pre-training episodes.\n", pretrain_episodes))
pretrain_returns <- generate_synthetic_pretrain(pretrain_episodes)
cat("Preparing realised historical fine-tuning episodes (no synthetic realised returns).\n")
finetune_returns <- generate_historical_finetune()
qsave_portable(pretrain_returns, pretrain_file)
qsave_portable(finetune_returns, finetune_file)

extract_realised <- function(episodes) {
  out <- do.call(rbind, unlist(lapply(episodes, function(ep) lapply(ep$returns, function(x) as.numeric(x[1L, ]))), recursive = FALSE))
  colnames(out) <- asset_names
  out
}
synthetic_monthly <- extract_realised(pretrain_returns)

safe_skew <- function(x) { s <- sd(x); if (!is.finite(s) || s == 0) NA_real_ else mean((x - mean(x))^3) / s^3 }
safe_kurt <- function(x) { s <- sd(x); if (!is.finite(s) || s == 0) NA_real_ else mean((x - mean(x))^4) / s^4 - 3 }
stats <- function(x, regime) data.table(regime = regime, asset = colnames(x), observations = nrow(x), mean = colMeans(x), sd = apply(x, 2, sd), q05 = apply(x, 2, quantile, .05), q95 = apply(x, 2, quantile, .95), skew = apply(x, 2, safe_skew), excess_kurtosis = apply(x, 2, safe_kurt))
tail_risk <- function(x, regime) data.table(regime = regime, asset = colnames(x), var_05 = apply(x, 2, quantile, .05), cvar_05 = vapply(seq_len(ncol(x)), function(j) { q <- quantile(x[, j], .05); mean(x[x[, j] <= q, j]) }, numeric(1)))
historical_diagnostic_returns <- log(all_train_gross)
synthetic_diagnostic_returns <- log(synthetic_monthly)
summary_stats <- rbindlist(list(stats(historical_diagnostic_returns, "historical_finetune"), stats(synthetic_diagnostic_returns, "synthetic_pretrain")))
tail_metrics <- rbindlist(list(tail_risk(historical_diagnostic_returns, "historical_finetune"), tail_risk(synthetic_diagnostic_returns, "synthetic_pretrain")))

fidelity <- merge(summary_stats[regime == "historical_finetune", .(asset, historical_mean = mean, historical_sd = sd, historical_q05 = q05)], summary_stats[regime == "synthetic_pretrain", .(asset, synthetic_mean = mean, synthetic_sd = sd, synthetic_q05 = q05)], by = "asset")
fidelity <- merge(fidelity, tail_metrics[regime == "historical_finetune", .(asset, historical_cvar05 = cvar_05)], by = "asset")
fidelity <- merge(fidelity, tail_metrics[regime == "synthetic_pretrain", .(asset, synthetic_cvar05 = cvar_05)], by = "asset")
fidelity[, `:=`(
  mean_standardised_error = (synthetic_mean - historical_mean) / historical_sd,
  sd_relative_error = (synthetic_sd - historical_sd) / historical_sd,
  q05_standardised_error = (synthetic_q05 - historical_q05) / historical_sd,
  cvar05_standardised_error = (synthetic_cvar05 - historical_cvar05) / historical_sd)]
fidelity[, pass_marginals := abs(mean_standardised_error) <= 0.25 &
  abs(sd_relative_error) <= 0.10 & abs(q05_standardised_error) <= 0.25 &
  abs(cvar05_standardised_error) <= 0.25]

cor_history <- cor(historical_diagnostic_returns)
cor_synthetic <- cor(synthetic_diagnostic_returns)
block_correlation_interval <- function(x, i, j, replications = 999L, level = .95) {
  n <- nrow(x); block_length <- max(2L, ceiling(n^(1 / 3)))
  estimates <- replicate(replications, {
    starts <- sample.int(n, ceiling(n / block_length), replace = TRUE)
    index <- unlist(lapply(starts, function(start)
      ((start - 1L + seq_len(block_length) - 1L) %% n) + 1L))[seq_len(n)]
    cor(x[index, i], x[index, j])
  })
  unname(quantile(estimates, c((1 - level) / 2, (1 + level) / 2),
                  type = 8, na.rm = TRUE))
}
correlation_comparison <- data.table(asset_i = character(), asset_j = character(), historical_correlation = numeric(), synthetic_correlation = numeric(), absolute_error = numeric(), historical_ci_lower = numeric(), historical_ci_upper = numeric())
for (i in seq_len(ncol(all_train_gross) - 1L)) for (j in (i + 1L):ncol(all_train_gross)) {
  historical_ci <- block_correlation_interval(historical_diagnostic_returns, i, j)
  correlation_comparison <- rbind(correlation_comparison, data.table(asset_i = asset_names[i], asset_j = asset_names[j], historical_correlation = cor_history[i, j], synthetic_correlation = cor_synthetic[i, j], absolute_error = abs(cor_history[i, j] - cor_synthetic[i, j]), historical_ci_lower = historical_ci[1L], historical_ci_upper = historical_ci[2L]))
}
# The fixed 0.10 target is retained as a strict fidelity target.  The interval
# flag is the sampling-aware diagnostic and should be reported alongside it.
correlation_comparison[, `:=`(pass_correlation = absolute_error <= 0.10,
  statistically_compatible = synthetic_correlation >= historical_ci_lower & synthetic_correlation <= historical_ci_upper)]

# This is a finite-threshold (5%) lower-tail *co-exceedance* probability,
# P(asset_j in its 5% tail | asset_i in its 5% tail), not the asymptotic tail
# dependence coefficient.  With 115 historical months there are only about six
# conditioning observations.  An observed zero therefore has a wide exact
# binomial interval and must not be interpreted as proof of independence.
tail_coexceedance <- function(x, i, j, probability = .05) {
  i_tail <- x[, i] <= quantile(x[, i], probability)
  j_tail <- x[, j] <= quantile(x[, j], probability)
  n_tail <- sum(i_tail); joint_tail <- sum(i_tail & j_tail)
  if (!n_tail) stop("No observations in empirical tail; cannot assess tail co-exceedance.")
  list(rate = joint_tail / n_tail, n_tail = n_tail, joint_tail = joint_tail)
}

tail_dependence <- data.table(asset_i = character(), asset_j = character(), historical_lower_tail = numeric(), synthetic_lower_tail = numeric(), absolute_error = numeric(), historical_tail_events = integer(), historical_joint_tail_events = integer(), historical_ci_lower = numeric(), historical_ci_upper = numeric(), synthetic_tail_events = integer(), synthetic_joint_tail_events = integer(), synthetic_ci_lower = numeric(), synthetic_ci_upper = numeric())
for (i in seq_len(ncol(all_train_gross) - 1L)) for (j in (i + 1L):ncol(all_train_gross)) {
  historical_tail <- tail_coexceedance(all_train_gross, i, j)
  synthetic_tail <- tail_coexceedance(synthetic_monthly, i, j)
  historical_ci <- binom.test(historical_tail$joint_tail, historical_tail$n_tail)$conf.int
  synthetic_ci <- binom.test(synthetic_tail$joint_tail, synthetic_tail$n_tail)$conf.int
  tail_dependence <- rbind(tail_dependence, data.table(asset_i = asset_names[i], asset_j = asset_names[j], historical_lower_tail = historical_tail$rate, synthetic_lower_tail = synthetic_tail$rate, absolute_error = abs(historical_tail$rate - synthetic_tail$rate), historical_tail_events = historical_tail$n_tail, historical_joint_tail_events = historical_tail$joint_tail, historical_ci_lower = historical_ci[1L], historical_ci_upper = historical_ci[2L], synthetic_tail_events = synthetic_tail$n_tail, synthetic_joint_tail_events = synthetic_tail$joint_tail, synthetic_ci_lower = synthetic_ci[1L], synthetic_ci_upper = synthetic_ci[2L]))
}
# A generated sample is compatible when its 95% interval overlaps the exact
# historical interval.  This is the meaningful pass criterion at this sample
# size; the old absolute-error threshold was not statistically interpretable.
tail_dependence[, `:=`(pass_lower_tail = synthetic_ci_upper >= historical_ci_lower & synthetic_ci_lower <= historical_ci_upper,
  statistic = "5pct_conditional_coexceedance")]

# Sequential fidelity matters for an LSTM. NN-vine dependence changes across
# each synthetic path, but simulated marginal shocks remain conditionally IID;
# this table quantifies temporal features that historical fine-tuning must still
# learn rather than silently treating the generator as a full time-series model.
historical_log <- log(all_train_gross)
synthetic_lag_pairs <- function(episodes) {
  previous <- list(); following <- list()
  for (ep in episodes) {
    x <- log(do.call(rbind, lapply(ep$returns, function(draw) draw[1L, ])))
    previous[[length(previous) + 1L]] <- x[-nrow(x), , drop = FALSE]
    following[[length(following) + 1L]] <- x[-1L, , drop = FALSE]
  }
  list(previous = do.call(rbind, previous), following = do.call(rbind, following))
}
synthetic_pairs <- synthetic_lag_pairs(pretrain_returns)
temporal_dependence <- rbindlist(lapply(seq_along(asset_names), function(j) {
  data.table(asset = asset_names[j],
    historical_lag1 = cor(historical_log[-nrow(historical_log), j], historical_log[-1L, j]),
    synthetic_lag1 = cor(synthetic_pairs$previous[, j], synthetic_pairs$following[, j]),
    historical_squared_lag1 = cor(historical_log[-nrow(historical_log), j]^2, historical_log[-1L, j]^2),
    synthetic_squared_lag1 = cor(synthetic_pairs$previous[, j]^2, synthetic_pairs$following[, j]^2))
}))
temporal_dependence[, `:=`(lag1_absolute_error = abs(historical_lag1 - synthetic_lag1), squared_lag1_absolute_error = abs(historical_squared_lag1 - synthetic_squared_lag1))]
temporal_tolerance <- 2 / sqrt(nrow(historical_log))
temporal_dependence[, pass_temporal := lag1_absolute_error <= temporal_tolerance &
  squared_lag1_absolute_error <= temporal_tolerance]

# Do not compound unrelated synthetic episodes into a fictitious century-long
# wealth path.  Portfolio fidelity is reported as one-period distributional
# agreement and per-episode terminal multiples instead.
equal_weight <- function(x) rowMeans(x)
portfolio_metrics <- rbindlist(list(
  data.table(regime = "historical_finetune", mean_gross = mean(equal_weight(all_train_gross)), sd_gross = sd(equal_weight(all_train_gross)), q05_gross = quantile(equal_weight(all_train_gross), .05), worst_gross = min(equal_weight(all_train_gross))),
  data.table(regime = "synthetic_pretrain", mean_gross = mean(equal_weight(synthetic_monthly)), sd_gross = sd(equal_weight(synthetic_monthly)), q05_gross = quantile(equal_weight(synthetic_monthly), .05), worst_gross = min(equal_weight(synthetic_monthly)))
))
episode_terminal <- vapply(pretrain_returns, function(ep) prod(rowMeans(do.call(rbind, lapply(ep$returns, function(x) x[1L, ])))), numeric(1))
episode_metrics <- data.table(mean_terminal_multiple = mean(episode_terminal), median_terminal_multiple = median(episode_terminal), p05_terminal_multiple = quantile(episode_terminal, .05), p95_terminal_multiple = quantile(episode_terminal, .95))

write.csv(summary_stats, file.path(diag_dir, "summary_statistics.csv"), row.names = FALSE)
write.csv(tail_metrics, file.path(diag_dir, "tail_risk.csv"), row.names = FALSE)
write.csv(portfolio_metrics, file.path(diag_dir, "portfolio_metrics.csv"), row.names = FALSE)
write.csv(fidelity, file.path(diag_dir, "fidelity_metrics.csv"), row.names = FALSE)
write.csv(correlation_comparison, file.path(diag_dir, "correlation_comparison.csv"), row.names = FALSE)
write.csv(tail_dependence, file.path(diag_dir, "tail_dependence_comparison.csv"), row.names = FALSE)
write.csv(temporal_dependence, file.path(diag_dir, "temporal_dependence.csv"), row.names = FALSE)
write.csv(episode_metrics, file.path(diag_dir, "synthetic_episode_metrics.csv"), row.names = FALSE)

plot_data <- rbindlist(list(cbind(data.table(regime = "historical_finetune"), as.data.table(all_train_gross)), cbind(data.table(regime = "synthetic_pretrain"), as.data.table(synthetic_monthly))), fill = TRUE)
plot_long <- melt(plot_data, id.vars = "regime", variable.name = "asset", value.name = "gross_return")
ggsave(distribution_figure, ggplot(plot_long, aes(gross_return, colour = regime)) + geom_density() + facet_wrap(~asset, scales = "free") + theme_bw(), width = 12, height = 8)

metadata <- list(seed = seed, n_sim_cvar = n_sim_cvar, L = L, ref_col = ref_col,
  episode_length = episode_length, sequence_length = sequence_length,
  pretrain_episodes = pretrain_episodes,
  historical_finetune_episodes = length(finetune_returns), train_vine_count = length(vine_seq_train),
  reserved_evaluation_steps = evaluation_periods, pretrain_realised_source = "synthetic_vine",
  finetune_realised_source = "historical", marginal_fit_source = "training_prefix_only",
  pretrain_vine_frequency = "monthly_marginal_transform",
  pretrain_vine_structure = "nn_dynamic_all_tree_dvine",
  pretrain_vine_model = "nn_dynamic_t_vine", finetune_vine_model = "nn_dynamic_t_vine",
  pretrain_vine_families = "t", monthly_ar1 = monthly_ar1,
  vine_order = nn_fit$order, vine_truncation_level = nn_fit$truncation_level,
  dynamic_vine_edges = nn_fit$dynamic_edge_count,
  nn_vine_model_dir = nn_vine_model_dir,
  diagnostics_passed = all(fidelity$pass_marginals) &
    all(correlation_comparison$statistically_compatible) &
    all(tail_dependence$pass_lower_tail) & all(temporal_dependence$pass_temporal),
  vine_truncation_validation = nn_fit$truncation_validation,
  training_cutoff = training_cutoff,
  source_data_md5 = attr(returns, "source_md5"),
  evaluation_start = min(eval_periods$decision_date),
  evaluation_end = max(eval_periods$holding_end_date),
  split_rule = "synthetic_pretrain; historical_train_prefix_finetune; final_24_monthly_holding_periods_evaluation_only",
  asset_names = asset_names, generated_at = Sys.time())
save(pretrain_returns, finetune_returns, pretrain_vine, metadata, train_end,
     training_cutoff, summary_stats, tail_metrics, portfolio_metrics, fidelity,
     correlation_comparison, tail_dependence, temporal_dependence,
     episode_metrics, file = output_file)
cat(sprintf("\nSaved bundle: %s\nPre-training: %d synthetic episodes. Fine-tuning: %d historical episodes.\n", output_file, length(pretrain_returns), length(finetune_returns)))
cat(sprintf("Marginal fidelity passed for %d/%d assets; strict correlation target passed for %d/%d pairs; lower-tail co-exceedance was statistically compatible for %d/%d pairs. Inspect %s before training.\n", sum(fidelity$pass_marginals), nrow(fidelity), sum(correlation_comparison$pass_correlation), nrow(correlation_comparison), sum(tail_dependence$pass_lower_tail), nrow(tail_dependence), diag_dir))
if (!isTRUE(metadata$diagnostics_passed)) {
  stop(sprintf(paste0("Synthetic-data gate failed: marginals %d/%d; correlation compatibility %d/%d; ",
                      "lower-tail compatibility %d/%d; temporal fidelity %d/%d. ",
                      "Diagnostics were saved, but RL training is intentionally blocked."),
               sum(fidelity$pass_marginals), nrow(fidelity),
               sum(correlation_comparison$statistically_compatible), nrow(correlation_comparison),
               sum(tail_dependence$pass_lower_tail), nrow(tail_dependence),
               sum(temporal_dependence$pass_temporal), nrow(temporal_dependence)))
}
