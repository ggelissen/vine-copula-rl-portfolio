#!/usr/bin/env Rscript
options(warn = 2)
source("helper/load_data.r")
source("helper/time_split.r")
source("eval/ablation.r")
source("eval/statistical_tests.r")
source("eval/research_protocol.r")
source("benchmark_models/dynamic_vine_NN.r")
source("rl/rl_environment.r")

assert_close <- function(x, y, tolerance = 1e-8) {
  stopifnot(length(x) == length(y), max(abs(x - y)) <= tolerance)
}

returns <- load_returns()
split <- split_monthly_periods(build_monthly_periods(returns, 250L), 24L)
validate_period_split(split, 24L)
stopifnot(nrow(split$evaluation) == 24L)
gross <- do.call(rbind, lapply(seq_len(nrow(split$evaluation)), function(i) {
  realised_gross_for_period(returns, split$evaluation$decision_date[i],
                            split$evaluation$holding_end_date[i])
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
weight_log <- data.frame(decision_date = split$evaluation$decision_date)
names(neutral) <- NULL
for (j in seq_len(ncol(returns))) {
  weight_log[[paste0("w_", colnames(returns)[j])]] <- neutral[, j]
}
weights <- validate_weight_log(weight_log, split$evaluation, colnames(returns), 1, 1.5)
scored <- score_weight_log(weights, gross, split$evaluation, colnames(returns))
stopifnot(nrow(scored) == 24L, all(is.finite(scored$net_return)))

series_a <- data.frame(date = scored$holding_end_date, net_return = scored$net_return)
series_b <- transform(series_a, net_return = net_return - 0.001)
paired <- pairwise_utility_tests(list(a = series_a, b = series_b))
stopifnot(nrow(paired) == 1L, paired$mean_difference > 0,
          is.finite(paired$p_value_holm))

cat("All fast research-protocol tests passed.\n")

