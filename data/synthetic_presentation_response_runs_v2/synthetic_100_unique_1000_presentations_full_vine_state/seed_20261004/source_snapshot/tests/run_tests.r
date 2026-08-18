#!/usr/bin/env Rscript
options(warn = 2)
source("helper/load_data.r")
source("helper/time_split.r")
source("eval/ablation.r")
source("eval/statistical_tests.r")
source("eval/research_protocol.r")
source("benchmark_models/dynamic_vine_NN.r")
source("rl/rl_environment.r")
source("helper/synthetic_fidelity.r")

assert_close <- function(x, y, tolerance = 1e-8) {
  stopifnot(length(x) == length(y), max(abs(x - y)) <= tolerance)
}

returns <- load_returns()
split <- split_monthly_periods(build_monthly_periods(returns, 250L), 24L)
validate_period_split(split, 24L)
stopifnot(nrow(split$evaluation) == 24L)
# All fast tests use the final 24 periods of the training prefix.  Merely
# testing code must not consume or summarize the locked OOS outcomes.
development_periods <- tail(split$train, 24L)
development_returns <- returns[paste0("/", max(split$train$holding_end_date))]
gross <- do.call(rbind, lapply(seq_len(nrow(development_periods)), function(i) {
  realised_gross_for_period(development_returns,
                            development_periods$decision_date[i],
                            development_periods$holding_end_date[i])
}))
stopifnot(all(is.finite(gross)), all(gross > 0))

for (trial in seq_len(100L)) {
  raw <- rnorm(ncol(returns))
  w <- project_long_short_weights(raw, 1, 1.5)
  assert_close(sum(w), 1)
  stopifnot(sum(abs(w)) <= 1.5 + 1e-8)
}
increments <- c(crra_utility(1.02, 2) - crra_utility(1, 2),
                crra_utility(1.02 * 0.98, 2) - crra_utility(1.02, 2))
assert_close(sum(increments), crra_utility(1.02 * 0.98, 2) - crra_utility(1, 2))

set.seed(42)
U_test <- matrix(runif(120 * 4, 0.01, 0.99), 120, 4)
order_test <- select_dvine_order(U_test)
vine_test <- vinecop(U_test, structure = dvine_structure(order_test),
                     family_set = "t", selcrit = "bic")
edge_data <- compute_dvine_edge_data(U_test, vine_test)
pieces <- unlist(lapply(seq_along(edge_data), function(tree) {
  lapply(seq_along(edge_data[[tree]]), function(edge) {
    log(dbicop(edge_data[[tree]][[edge]],
               vine_test$pair_copulas[[tree]][[edge]]))
  })
}), recursive = FALSE)
assert_close(Reduce("+", pieces), log(dvinecop(U_test, vine_test)), 1e-7)

neutral <- matrix(1 / ncol(returns), nrow = 24, ncol = ncol(returns))
weight_log <- data.frame(decision_date = development_periods$decision_date)
names(neutral) <- NULL
for (j in seq_len(ncol(returns))) {
  weight_log[[paste0("w_", colnames(returns)[j])]] <- neutral[, j]
}
weights <- validate_weight_log(weight_log, development_periods,
                               colnames(returns), 1, 1.5)
scored <- score_weight_log(weights, gross, development_periods,
                           colnames(returns))
stopifnot(nrow(scored) == 24L, all(is.finite(scored$net_return)))

series_a <- data.frame(date = scored$holding_end_date, net_return = scored$net_return)
series_b <- transform(series_a, net_return = net_return - 0.001)
paired <- pairwise_utility_tests(list(a = series_a, b = series_b))
stopifnot(nrow(paired) == 1L, paired$mean_difference > 0,
          is.finite(paired$p_value_holm))

# Sampling-aware synthetic fidelity retains strict descriptive targets while
# accounting for finite historical samples and clustered simulation episodes.
set.seed(20260814)
historical_test <- matrix(rt(71L * 2L, df = 6), ncol = 2L,
                          dimnames = list(NULL, c("A", "B")))
metric_a <- synthetic_tail_metrics(historical_test[, 1L])
metric_b <- synthetic_tail_metrics(historical_test[, 2L])
fidelity_test <- data.table::data.table(
  asset = c("A", "B"),
  synthetic_mean = c(metric_a["mean"], metric_b["mean"]),
  synthetic_sd = c(metric_a["sd"], metric_b["sd"]),
  synthetic_q05 = c(metric_a["q05"], metric_b["q05"]),
  synthetic_cvar05 = c(metric_a["cvar05"], metric_b["cvar05"]),
  mean_standardised_error = 0, sd_relative_error = 0,
  q05_standardised_error = 0, cvar05_standardised_error = 0,
  pass_marginals = TRUE)
fidelity_test <- apply_sampling_aware_marginal_gate(
  fidelity_test, historical_test, c("A", "B"), 17L, replications = 99L)
stopifnot(all(fidelity_test$statistically_compatible),
          all(fidelity_test$marginal_guardrail_pass))

episode_test <- lapply(seq_len(40L), function(index) {
  first <- rnorm(12L)
  cbind(A = first, B = 0.5 * first + rnorm(12L, sd = 0.5))
})
synthetic_test <- do.call(rbind, episode_test)
rho_test <- cor(synthetic_test)[1L, 2L]
correlation_test <- data.table::data.table(
  asset_i = "A", asset_j = "B", historical_correlation = rho_test,
  synthetic_correlation = rho_test, absolute_error = 0,
  historical_ci_lower = -1, historical_ci_upper = 1,
  pass_correlation = TRUE)
correlation_test <- apply_sampling_aware_correlation_gate(
  correlation_test, episode_test, c("A", "B"), 19L, replications = 49L)
stopifnot(correlation_test$statistically_compatible,
          correlation_test$correlation_guardrail_pass,
          is.finite(correlation_test$synthetic_ci_lower),
          is.finite(correlation_test$synthetic_ci_upper))

cat("All fast research-protocol tests passed.\n")
