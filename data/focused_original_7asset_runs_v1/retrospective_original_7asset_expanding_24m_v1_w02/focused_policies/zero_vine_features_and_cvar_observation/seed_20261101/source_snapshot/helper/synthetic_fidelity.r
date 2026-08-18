# Sampling-aware synthetic-data fidelity diagnostics.
#
# Strict absolute-error targets remain visible as descriptive targets. The
# training gate additionally recognises finite historical samples and Monte
# Carlo uncertainty, while retaining conservative two-times-threshold
# guardrails so wide intervals cannot excuse economically implausible output.

synthetic_tail_metrics <- function(x) {
  q05 <- unname(quantile(x, 0.05, type = 7, na.rm = TRUE))
  c(mean = mean(x), sd = sd(x), q05 = q05,
    cvar05 = mean(x[x <= q05]))
}

circular_block_indices <- function(n, block_length) {
  starts <- sample.int(n, ceiling(n / block_length), replace = TRUE)
  unlist(lapply(starts, function(start)
    ((start - 1L + seq_len(block_length) - 1L) %% n) + 1L),
    use.names = FALSE)[seq_len(n)]
}

historical_metric_intervals <- function(x, replications = 999L,
                                        level = 0.95, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  n <- length(x)
  if (n < 30L) stop("Marginal fidelity intervals require at least 30 observations.")
  block_length <- max(2L, ceiling(n^(1 / 3)))
  estimates <- replicate(replications, synthetic_tail_metrics(
    x[circular_block_indices(n, block_length)]))
  bounds <- t(apply(estimates, 1L, quantile,
                    probs = c((1 - level) / 2, (1 + level) / 2),
                    type = 8, na.rm = TRUE))
  colnames(bounds) <- c("lower", "upper")
  bounds
}

apply_sampling_aware_marginal_gate <- function(
    fidelity, historical_log_returns, asset_names, seed,
    replications = 999L) {
  columns <- c("mean", "sd", "q05", "cvar05")
  for (column in columns) {
    fidelity[[paste0("historical_", column, "_ci_lower")]] <- NA_real_
    fidelity[[paste0("historical_", column, "_ci_upper")]] <- NA_real_
  }
  for (j in seq_along(asset_names)) {
    interval <- historical_metric_intervals(
      historical_log_returns[, j], replications = replications,
      seed = seed + 1009L * j)
    row <- which(fidelity$asset == asset_names[j])
    if (length(row) != 1L) stop("Marginal fidelity asset mapping is not one-to-one.")
    for (column in columns) {
      fidelity[[paste0("historical_", column, "_ci_lower")]][row] <-
        interval[column, "lower"]
      fidelity[[paste0("historical_", column, "_ci_upper")]][row] <-
        interval[column, "upper"]
    }
  }
  interval_compatible <-
    fidelity$synthetic_mean >= fidelity$historical_mean_ci_lower &
    fidelity$synthetic_mean <= fidelity$historical_mean_ci_upper &
    fidelity$synthetic_sd >= fidelity$historical_sd_ci_lower &
    fidelity$synthetic_sd <= fidelity$historical_sd_ci_upper &
    fidelity$synthetic_q05 >= fidelity$historical_q05_ci_lower &
    fidelity$synthetic_q05 <= fidelity$historical_q05_ci_upper &
    fidelity$synthetic_cvar05 >= fidelity$historical_cvar05_ci_lower &
    fidelity$synthetic_cvar05 <= fidelity$historical_cvar05_ci_upper
  guardrail <- abs(fidelity$mean_standardised_error) <= 0.50 &
    abs(fidelity$sd_relative_error) <= 0.20 &
    abs(fidelity$q05_standardised_error) <= 0.50 &
    abs(fidelity$cvar05_standardised_error) <= 0.50
  fidelity$marginal_interval_compatible <- interval_compatible
  fidelity$marginal_guardrail_pass <- guardrail
  fidelity$statistically_compatible <- interval_compatible & guardrail
  fidelity
}

episode_cluster_correlation_intervals <- function(
    episode_log_returns, pair_i, pair_j, replications = 499L,
    level = 0.95, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  episode_count <- length(episode_log_returns)
  if (episode_count < 30L) stop("Correlation intervals require at least 30 episodes.")
  dimension <- ncol(episode_log_returns[[1L]])
  episode_n <- vapply(episode_log_returns, nrow, integer(1))
  episode_sum <- do.call(rbind, lapply(episode_log_returns, colSums))
  episode_cross <- do.call(cbind, lapply(episode_log_returns, function(x)
    as.vector(crossprod(x))))
  estimates <- matrix(NA_real_, nrow = length(pair_i), ncol = replications)
  for (replication in seq_len(replications)) {
    index <- sample.int(episode_count, episode_count, replace = TRUE)
    n <- sum(episode_n[index])
    total <- colSums(episode_sum[index, , drop = FALSE])
    cross <- matrix(rowSums(episode_cross[, index, drop = FALSE]),
                    nrow = dimension, ncol = dimension)
    covariance <- (cross - tcrossprod(total) / n) / (n - 1)
    correlation <- cov2cor(covariance)
    estimates[, replication] <- correlation[cbind(pair_i, pair_j)]
  }
  bounds <- t(apply(estimates, 1L, quantile,
                    probs = c((1 - level) / 2, (1 + level) / 2),
                    type = 8, na.rm = TRUE))
  colnames(bounds) <- c("lower", "upper")
  bounds
}

apply_sampling_aware_correlation_gate <- function(
    comparison, episode_log_returns, asset_names, seed,
    replications = 499L) {
  pair_i <- match(comparison$asset_i, asset_names)
  pair_j <- match(comparison$asset_j, asset_names)
  if (anyNA(c(pair_i, pair_j))) stop("Correlation pair asset mapping failed.")
  intervals <- episode_cluster_correlation_intervals(
    episode_log_returns, pair_i, pair_j, replications = replications,
    seed = seed + 65537L)
  comparison$synthetic_ci_lower <- intervals[, "lower"]
  comparison$synthetic_ci_upper <- intervals[, "upper"]
  comparison$intervals_overlap <-
    comparison$synthetic_ci_upper >= comparison$historical_ci_lower &
    comparison$synthetic_ci_lower <= comparison$historical_ci_upper
  comparison$correlation_guardrail_pass <- comparison$absolute_error <= 0.25
  comparison$statistically_compatible <- comparison$intervals_overlap &
    comparison$correlation_guardrail_pass
  comparison
}
