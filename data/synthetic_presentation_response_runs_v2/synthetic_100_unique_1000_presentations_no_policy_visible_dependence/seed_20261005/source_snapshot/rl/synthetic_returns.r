# Prepare the data protocol used by the two-stage RL experiment.
# Stage 1: synthetic vine-copula episodes only.
# Stage 2: realised historical episodes, excluding the final evaluation horizon.
# Run from the project root: Rscript --vanilla rl/synthetic_returns.r config/config.yaml

suppressPackageStartupMessages({ library(data.table); library(ggplot2); library(rvinecopulib); library(yaml); library(jsonlite) })
qsave_portable <- function(object, file) {
  if (requireNamespace("qs", quietly = TRUE)) qs::qsave(object, file, preset = "fast")
  else saveRDS(object, file = file, compress = FALSE)
}
source("benchmark_models/dynamic_vine_NN.r")
source("rl/rl_environment.r")
source("helper/time_split.r")
source("helper/synthetic_fidelity.r")

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
bundle_manifest_file <- env_or(
  "SYNTHETIC_BUNDLE_MANIFEST", paste0(output_file, ".manifest.json"))
immutable_output <- tolower(env_or("IMMUTABLE_SYNTHETIC_OUTPUT", "false")) %in%
  c("1", "true", "yes")
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
vine_truncation_level <- env_or(
  "VINE_TRUNCATION_LEVEL",
  if (is.null(config$vine$truncation_level)) 0L else config$vine$truncation_level,
  as.integer)

if (any(c(n_sim_cvar, sim_cores, L, pretrain_episodes, episode_length, sequence_length, evaluation_periods) < 1L)) stop("Simulation settings must be positive.")
if (immutable_output && any(file.exists(c(
    output_file, bundle_manifest_file, pretrain_file, finetune_file,
    training_marginals_file)))) {
  stop("Immutable synthetic output path already exists; choose a new window/version.")
}
if (evaluation_periods != 24L) stop("Publication protocol locks the historical OOS holdout to exactly 24 monthly holding periods.")
for (directory in unique(c(dirname(output_file), dirname(bundle_manifest_file),
    dirname(pretrain_file), dirname(finetune_file),
    dirname(training_marginals_file), dirname(distribution_figure), diag_dir))) {
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
}
set.seed(seed)

returns <- load_returns()
validate_return_model_contract(returns, ref_col, vine_truncation_level)
period_split <- split_monthly_periods(build_monthly_periods(returns, min_history = L), evaluation_periods)
validate_period_split(period_split, evaluation_periods)
validate_return_evaluation_contract(returns, period_split, evaluation_periods)
train_periods <- period_split$train; eval_periods <- period_split$evaluation
train_dates <- train_periods$decision_date; eval_dates <- eval_periods$decision_date
train_end <- tail(train_periods$decision_idx, 1L)
asset_names <- colnames(returns)

# Holding-period returns are the modelling frequency of the generator and RL
# environment.  A copula fitted to daily ranks and then relabelled with monthly
# quantiles does not represent a monthly copula.  Use every pre-holdout monthly
# outcome for dependence estimation, including the early months that precede
# the L-day RL state warm-up; none of the final 24 holding periods enter here.
copula_period_split <- split_monthly_periods(
  build_monthly_periods(returns, min_history = 0L), evaluation_periods)
validate_period_split(copula_period_split, evaluation_periods)
copula_train_periods <- copula_period_split$train
copula_monthly_log <- do.call(rbind, lapply(seq_len(nrow(copula_train_periods)), function(i) {
  log(as.numeric(realised_gross_for_period(
    returns, copula_train_periods$decision_date[i],
    copula_train_periods$holding_end_date[i])))
}))
colnames(copula_monthly_log) <- asset_names
rank_pseudo_observations <- function(x) {
  x <- as.matrix(x); n <- nrow(x)
  out <- apply(x, 2L, function(column)
    (rank(column, ties.method = "average") - 0.5) / n)
  dimnames(out) <- dimnames(x)
  out
}
copula_monthly_u <- rank_pseudo_observations(copula_monthly_log)

# These are the realised monthly outcomes available for fine-tuning and for
# estimating the serial marginal copulas.  They exclude the final 24 holding
# periods by construction.
all_train_gross <- do.call(rbind, lapply(seq_len(nrow(train_periods)), function(i) {
  as.numeric(realised_gross_for_period(
    returns, train_periods$decision_date[i], train_periods$holding_end_date[i]))
}))
colnames(all_train_gross) <- asset_names
historical_log_returns <- log(all_train_gross)
historical_log_sorted <- lapply(seq_len(ncol(all_train_gross)), function(j)
  sort(historical_log_returns[, j]))
names(historical_log_sorted) <- asset_names

# Estimate marginal serial copulas before the cross-sectional vine.  The vine
# must be fitted to one-step conditional PIT innovations; fitting it to raw
# monthly ranks and subsequently applying serial filters attenuates the target
# contemporaneous dependence and double-counts the marginal dynamics.
serial_u <- rank_pseudo_observations(historical_log_returns)
serial_copulas <- lapply(seq_along(asset_names), function(j) {
  pair <- cbind(head(serial_u[, j], -1L), tail(serial_u[, j], -1L))
  fit <- tryCatch(
    bicop(pair, family_set = "t", selcrit = "bic"),
    error = function(e) NULL)
  if (is.null(fit) || length(fit$parameters) < 2L ||
      any(!is.finite(fit$parameters[1:2]))) {
    rho <- cor(qnorm(pair[, 1L]), qnorm(pair[, 2L]))
    if (!is.finite(rho)) rho <- 0
    return(list(rho = pmax(pmin(rho, 0.95), -0.95), nu = 10,
                fallback = TRUE))
  }
  list(rho = pmax(pmin(as.numeric(fit$parameters[1L]), 0.95), -0.95),
       nu = pmax(as.numeric(fit$parameters[2L]), 2.05), fallback = FALSE)
})
names(serial_copulas) <- asset_names

serial_conditional_pit <- function(previous_u, current_u, model) {
  rho <- model$rho; nu <- model$nu
  previous_t <- qt(pmin(pmax(previous_u, 1e-8), 1 - 1e-8), df = nu)
  current_t <- qt(pmin(pmax(current_u, 1e-8), 1 - 1e-8), df = nu)
  conditional_scale <- sqrt((nu + previous_t^2) * (1 - rho^2) / (nu + 1))
  pt((current_t - rho * previous_t) / conditional_scale, df = nu + 1)
}

copula_innovation_u <- matrix(
  NA_real_, nrow = nrow(copula_monthly_u) - 1L,
  ncol = ncol(copula_monthly_u),
  dimnames = list(NULL, asset_names))
for (j in seq_along(asset_names)) {
  copula_innovation_u[, j] <- serial_conditional_pit(
    head(copula_monthly_u[, j], -1L), tail(copula_monthly_u[, j], -1L),
    serial_copulas[[j]])
}
copula_innovation_u <- pmin(pmax(copula_innovation_u, 1e-6), 1 - 1e-6)
copula_innovation_dates <- copula_train_periods$holding_end_date[-1L]

# Refit all marginal models on the training prefix.  The full-sample RData is
# never used for the RL state/generator after this point.
source("helper/marginals.r")
marginals <- fit_marginals_training(returns[seq_len(train_end), ])
U <- training_pseudo_observations(returns, marginals)
training_cutoff <- tail(train_dates, 1L)
save(marginals, U, copula_monthly_u, copula_innovation_u,
     copula_monthly_log, serial_copulas, asset_names,
     training_cutoff, train_end, file = training_marginals_file)
if (!identical(vine_model, "nn_dynamic_t_vine")) stop("Synthetic generation supports only the NN-driven dynamic t-vine; rolling vines are disabled.")
monthly_nn_states <- derive_nn_states(copula_innovation_u)
nn_fit <- fit_nn_dynamic_vine(copula_innovation_u,
  monthly_nn_states$z, monthly_nn_states$sigma,
  epochs = nn_vine_epochs, lr = nn_vine_lr, patience = nn_vine_patience,
  truncation_level = vine_truncation_level)
sim <- build_simulator(marginals, asset_names, ref_col = ref_col)

serial_transition <- function(innovation_u, previous_u = NULL) {
  innovation_u <- pmin(pmax(as.matrix(innovation_u), 1e-8), 1 - 1e-8)
  if (is.null(previous_u)) return(innovation_u)
  previous_u <- as.matrix(previous_u)
  if (nrow(previous_u) == 1L && nrow(innovation_u) > 1L)
    previous_u <- previous_u[rep(1L, nrow(innovation_u)), , drop = FALSE]
  if (!identical(dim(previous_u), dim(innovation_u)))
    stop("Serial-copula state and innovations must have identical dimensions.")
  out <- innovation_u
  for (j in seq_along(asset_names)) {
    rho <- serial_copulas[[j]]$rho; nu <- serial_copulas[[j]]$nu
    previous_t <- qt(pmin(pmax(previous_u[, j], 1e-8), 1 - 1e-8), df = nu)
    conditional_t <- rho * previous_t +
      sqrt((nu + previous_t^2) * (1 - rho^2) / (nu + 1)) *
      qt(innovation_u[, j], df = nu + 1)
    out[, j] <- pt(conditional_t, df = nu)
  }
  dimnames(out) <- dimnames(innovation_u)
  out
}

uniforms_to_monthly_gross <- function(u) {
  u <- as.matrix(u)
  out <- matrix(NA_real_, nrow = nrow(u), ncol = length(asset_names),
                dimnames = list(NULL, asset_names))
  for (j in seq_along(asset_names)) {
    sorted_log <- historical_log_sorted[[asset_names[j]]]
    # Midpoint plotting positions make this a centred, continuous empirical
    # quantile map.  i/(n+1) assigns excessive probability below the first
    # interpolation knot and inflated the CVaR of assets with an isolated
    # minimum (notably DIVIDEND).
    probabilities <- (seq_len(length(sorted_log)) - 0.5) / length(sorted_log)
    out[, j] <- exp(approx(probabilities, sorted_log, xout = u[, j], rule = 2)$y)
  }
  out
}

logs_to_empirical_u <- function(previous_log_returns) {
  matrix(vapply(seq_along(asset_names), function(j) {
    sorted_log <- historical_log_sorted[[asset_names[j]]]
    probabilities <- (seq_len(length(sorted_log)) - 0.5) / length(sorted_log)
    approx(sorted_log, probabilities, xout = previous_log_returns[j],
           rule = 2, ties = "ordered")$y
  }, numeric(1)), nrow = 1L, dimnames = list(NULL, asset_names))
}

# One-parameter simulated method of moments (SMM) calibration.  A fitted
# finite-sample t D-vine followed by seven different serial filters can
# systematically attenuate unconditional Pearson dependence.  Pair-specific
# corrections would badly overfit 21 moments; a single Fisher-z multiplier is
# parsimonious, preserves every edge's sign and dynamic ordering, and is fitted
# exclusively on the training prefix.
scale_static_t_vine <- function(vine, dependence_scale) {
  pair_copulas <- lapply(vine$pair_copulas, function(tree) {
    lapply(tree, function(pc) {
      if (!pc$family %in% c("student", "t"))
        stop("Dependence calibration requires t-copula edges.")
      rho <- tanh(dependence_scale * atanh(as.numeric(pc$parameters[1L])))
      bicop_dist("t", rotation = 0L,
                 parameters = c(pmax(pmin(rho, 0.995), -0.995),
                                as.numeric(pc$parameters[2L])))
    })
  })
  vinecop_dist(pair_copulas = pair_copulas, structure = vine$structure)
}

calibration_target <- cor(historical_log_returns)
calibration_pairs <- upper.tri(calibration_target)
calibration_draws <- max(2000L, min(5000L, pretrain_episodes * 5L))
calibration_periods <- episode_length
dependence_objective <- function(dependence_scale) {
  set.seed(seed + 104729L)
  vine <- scale_static_t_vine(nn_fit$backbone, dependence_scale)
  previous_u <- NULL; simulated <- vector("list", calibration_periods)
  for (tt in seq_len(calibration_periods)) {
    innovation_u <- rvinecop(calibration_draws, vine, cores = sim_cores)
    current_u <- serial_transition(innovation_u, previous_u)
    previous_u <- current_u
    simulated[[tt]] <- log(uniforms_to_monthly_gross(current_u))
  }
  simulated_correlation <- cor(do.call(rbind, simulated))
  mean((simulated_correlation[calibration_pairs] -
          calibration_target[calibration_pairs])^2)
}
dependence_optim <- optimize(dependence_objective, interval = c(0.75, 1.75),
                             tol = 0.01)
nn_fit$dependence_scale <- as.numeric(dependence_optim$minimum)
nn_fit$dependence_smm_loss <- as.numeric(dependence_optim$objective)
cat(sprintf("Training-only SMM dependence scale: %.4f (moment loss %.6g)\n",
            nn_fit$dependence_scale, nn_fit$dependence_smm_loss))
save_nn_dynamic_vine_fit(nn_fit, nn_vine_model_dir)
vine_seq_train <- build_nn_vine_sequence(
  nn_fit, copula_innovation_u, monthly_nn_states$z, monthly_nn_states$sigma,
  train_dates, copula_innovation_dates)
if (length(vine_seq_train) != length(train_dates))
  stop("Failed to create every training NN-vine snapshot.")

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
  if (!is.null(previous_log_returns))
    u <- serial_transition(u, logs_to_empirical_u(previous_log_returns))
  uniforms_to_monthly_gross(u)
}

# Fitted marginal Student-t Markov copulas supply the monthly serial and tail
# persistence that an LSTM is expected to learn. Their conditional innovations
# retain the contemporaneous NN-vine dependence; the empirical transform
# restores each monthly marginal.
simulate_calibrated_monthly_path <- function(vines, n_draws) {
  previous_u <- NULL
  lapply(vines, function(vine) {
    innovation_u <- rvinecop(n_draws, vine, cores = sim_cores)
    u <- serial_transition(innovation_u, previous_u)
    previous_u <<- u
    uniforms_to_monthly_gross(u)
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
      holding_year_fractions = rep(1 / 12, episode_length),
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
      holding_year_fractions = as.numeric(
        train_periods$holding_end_date[action_idx] -
          train_periods$decision_date[action_idx]) / 365,
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
fidelity <- apply_sampling_aware_marginal_gate(
  fidelity, historical_diagnostic_returns, asset_names, seed)

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
correlation_comparison[, pass_correlation := absolute_error <= 0.10]
synthetic_episode_log_returns <- lapply(pretrain_returns, function(ep)
  log(do.call(rbind, lapply(ep$returns, function(draw)
    as.numeric(draw[1L, ])))))
correlation_comparison <- apply_sampling_aware_correlation_gate(
  correlation_comparison, synthetic_episode_log_returns, asset_names, seed)

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
  pretrain_vine_frequency = "monthly_holding_period_ranks",
  cross_sectional_fit_input = "monthly_one_step_serial_conditional_pit",
  pretrain_vine_structure = if (nn_fit$truncation_level == length(asset_names) - 1L)
    "nn_dynamic_all_tree_dvine" else "nn_dynamic_truncated_dvine",
  pretrain_vine_model = "nn_dynamic_t_vine", finetune_vine_model = "nn_dynamic_t_vine",
  pretrain_vine_families = "t", serial_copulas = serial_copulas,
  vine_order = nn_fit$order, vine_truncation_level = nn_fit$truncation_level,
  dynamic_vine_edges = nn_fit$dynamic_edge_count,
  dependence_scale = nn_fit$dependence_scale,
  dependence_smm_loss = nn_fit$dependence_smm_loss,
  nn_vine_model_dir = nn_vine_model_dir,
  diagnostics_passed = all(fidelity$statistically_compatible) &
    all(correlation_comparison$statistically_compatible) &
    all(tail_dependence$pass_lower_tail) & all(temporal_dependence$pass_temporal),
  diagnostic_gate_protocol = "sampling_aware_guardrailed_v2",
  vine_truncation_validation = nn_fit$truncation_validation,
  training_cutoff = training_cutoff,
  source_data_md5 = attr(returns, "source_md5"),
  source_data_sha256 = attr(returns, "source_sha256"),
  source_data_kind = attr(returns, "source_kind"),
  source_manifest_sha256 = attr(returns, "source_manifest_sha256"),
  panel_id = attr(returns, "panel_id"), window_id = attr(returns, "window_id"),
  evaluation_start = min(eval_periods$decision_date),
  evaluation_end = max(eval_periods$holding_end_date),
  split_rule = "synthetic_pretrain; historical_train_prefix_finetune; final_24_monthly_holding_periods_evaluation_only",
  asset_names = asset_names, generated_at = Sys.time())
save(pretrain_returns, finetune_returns, pretrain_vine, metadata, train_end,
     training_cutoff, summary_stats, tail_metrics, portfolio_metrics, fidelity,
     correlation_comparison, tail_dependence, temporal_dependence,
     episode_metrics, file = output_file)
bundle_manifest <- list(
  schema_version = 1L,
  release_status = if (immutable_output)
    "generated_immutable_training_bundle" else "generated_training_bundle",
  bundle_file = normalizePath(output_file, winslash = "/", mustWork = TRUE),
  bundle_sha256 = sha256_file(output_file),
  source_data_sha256 = attr(returns, "source_sha256"),
  source_manifest_sha256 = attr(returns, "source_manifest_sha256"),
  panel_id = attr(returns, "panel_id"), window_id = attr(returns, "window_id"),
  asset_names = asset_names, asset_count = length(asset_names),
  reference_asset_index_1based = ref_col,
  vine_truncation_level = nn_fit$truncation_level,
  dynamic_vine_edges = nn_fit$dynamic_edge_count,
  pretrain_episodes = length(pretrain_returns),
  finetune_episodes = length(finetune_returns),
  reserved_evaluation_periods = evaluation_periods,
  diagnostics_passed = isTRUE(metadata$diagnostics_passed),
  diagnostic_gate_protocol = metadata$diagnostic_gate_protocol,
  confirmatory_claim_permitted = FALSE)
jsonlite::write_json(bundle_manifest, bundle_manifest_file,
                     auto_unbox = TRUE, pretty = TRUE, null = "null")
cat(sprintf("\nSaved bundle: %s\nPre-training: %d synthetic episodes. Fine-tuning: %d historical episodes.\n", output_file, length(pretrain_returns), length(finetune_returns)))
cat(sprintf(paste0("Marginal strict target passed for %d/%d assets and sampling-aware compatibility for %d/%d; ",
                   "strict correlation target passed for %d/%d pairs and sampling-aware compatibility for %d/%d; ",
                   "lower-tail co-exceedance was statistically compatible for %d/%d pairs. Inspect %s before training.\n"),
            sum(fidelity$pass_marginals), nrow(fidelity),
            sum(fidelity$statistically_compatible), nrow(fidelity),
            sum(correlation_comparison$pass_correlation), nrow(correlation_comparison),
            sum(correlation_comparison$statistically_compatible), nrow(correlation_comparison),
            sum(tail_dependence$pass_lower_tail), nrow(tail_dependence), diag_dir))
if (!isTRUE(metadata$diagnostics_passed)) {
  stop(sprintf(paste0("Synthetic-data gate failed: marginal compatibility %d/%d; correlation compatibility %d/%d; ",
                      "lower-tail compatibility %d/%d; temporal fidelity %d/%d. ",
                      "Diagnostics were saved, but RL training is intentionally blocked."),
               sum(fidelity$statistically_compatible), nrow(fidelity),
               sum(correlation_comparison$statistically_compatible), nrow(correlation_comparison),
               sum(tail_dependence$pass_lower_tail), nrow(tail_dependence),
               sum(temporal_dependence$pass_temporal), nrow(temporal_dependence)))
}
