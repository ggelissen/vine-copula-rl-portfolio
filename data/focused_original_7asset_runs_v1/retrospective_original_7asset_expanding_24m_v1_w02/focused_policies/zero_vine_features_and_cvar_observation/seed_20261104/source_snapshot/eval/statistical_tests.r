# Leakage-safe inference for matched out-of-sample strategy returns.
# No code is executed on source. Tests operate on utilities/returns observed on
# identical dates; they are not mislabeled forecast-accuracy DM tests.

newey_west_mean_test <- function(difference, lag = NULL,
                                  alternative = c("two.sided", "greater", "less")) {
  alternative <- match.arg(alternative)
  x <- as.numeric(difference)
  x <- x[is.finite(x)]
  n <- length(x)
  if (n < 8L) stop("At least eight paired observations are required.")
  if (is.null(lag)) lag <- floor(4 * (n / 100)^(2 / 9))
  lag <- max(0L, min(as.integer(lag), n - 2L))
  centred <- x - mean(x)
  gamma0 <- sum(centred^2) / n
  long_run_variance <- gamma0
  if (lag > 0L) {
    for (ell in seq_len(lag)) {
      covariance <- sum(centred[(ell + 1L):n] * centred[seq_len(n - ell)]) / n
      long_run_variance <- long_run_variance +
        2 * (1 - ell / (lag + 1)) * covariance
    }
  }
  se <- sqrt(pmax(long_run_variance, 0) / n)
  statistic <- mean(x) / se
  p <- switch(alternative,
    two.sided = 2 * pnorm(-abs(statistic)),
    greater = pnorm(statistic, lower.tail = FALSE),
    less = pnorm(statistic)
  )
  c(mean_difference = mean(x), standard_error = se,
    statistic = statistic, p_value = p, lag = lag, observations = n)
}

crra_period_utility <- function(simple_return, gamma = 2) {
  gross <- 1 + as.numeric(simple_return)
  if (any(!is.finite(gross)) || any(gross <= 0)) {
    stop("CRRA utility requires finite positive gross portfolio returns.")
  }
  if (gamma == 1) log(gross) else (gross^(1 - gamma) - 1) / (1 - gamma)
}

moving_block_indices <- function(n, block_length) {
  starts <- sample.int(n, ceiling(n / block_length), replace = TRUE)
  unlist(lapply(starts, function(s) ((s - 1L + seq_len(block_length) - 1L) %% n) + 1L))[
    seq_len(n)]
}

paired_block_bootstrap <- function(difference, block_length = NULL,
                                   replications = 4999L, seed = 20260741L) {
  x <- as.numeric(difference); n <- length(x)
  if (n < 8L || any(!is.finite(x))) stop("Invalid paired difference series.")
  if (is.null(block_length)) block_length <- max(2L, ceiling(n^(1 / 3)))
  set.seed(seed)
  boot <- replicate(replications, mean(x[moving_block_indices(n, block_length)]))
  c(mean_difference = mean(x),
    ci_lower = unname(quantile(boot, 0.025, type = 8)),
    ci_upper = unname(quantile(boot, 0.975, type = 8)),
    p_greater = (1 + sum((boot - mean(x)) >= mean(x))) / (replications + 1),
    block_length = block_length, replications = replications)
}

align_strategy_returns <- function(strategy_returns) {
  if (!is.list(strategy_returns) || length(strategy_returns) < 2L ||
      is.null(names(strategy_returns))) stop("Supply a named list of strategy data frames.")
  required <- c("date", "net_return")
  cleaned <- lapply(strategy_returns, function(x) {
    if (length(setdiff(required, names(x)))) stop("Each strategy needs date and net_return.")
    out <- data.frame(date = as.Date(x$date), net_return = as.numeric(x$net_return))
    if (anyNA(out$date) || anyDuplicated(out$date) || any(!is.finite(out$net_return))) {
      stop("Invalid strategy return data.")
    }
    out[order(out$date), ]
  })
  dates <- cleaned[[1L]]$date
  if (!all(vapply(cleaned, function(x) identical(x$date, dates), logical(1)))) {
    stop("All strategies must be evaluated on identical realised dates.")
  }
  list(dates = dates, returns = do.call(cbind, lapply(cleaned, `[[`, "net_return")))
}

pairwise_utility_tests <- function(strategy_returns, gamma = 2, lag = NULL) {
  aligned <- align_strategy_returns(strategy_returns)
  names_list <- names(strategy_returns)
  utilities <- apply(aligned$returns, 2, crra_period_utility, gamma = gamma)
  pairs <- combn(seq_along(names_list), 2L)
  out <- do.call(rbind, lapply(seq_len(ncol(pairs)), function(k) {
    i <- pairs[1L, k]; j <- pairs[2L, k]
    test <- newey_west_mean_test(utilities[, i] - utilities[, j], lag, "two.sided")
    data.frame(strategy_a = names_list[i], strategy_b = names_list[j],
               t(test), row.names = NULL)
  }))
  out$p_value_holm <- p.adjust(out$p_value, method = "holm")
  out
}

# White-style Reality Check using a circular moving-block bootstrap. Candidate
# utility gains are jointly compared with one benchmark, controlling selection
# over many tried strategies.
reality_check <- function(strategy_returns, benchmark, gamma = 2,
                          block_length = NULL, replications = 9999L,
                          seed = 20260741L) {
  aligned <- align_strategy_returns(strategy_returns)
  strategy_names <- names(strategy_returns)
  benchmark_index <- match(benchmark, strategy_names)
  if (is.na(benchmark_index)) stop("benchmark is not in strategy_returns.")
  candidate_index <- setdiff(seq_along(strategy_names), benchmark_index)
  utility <- apply(aligned$returns, 2, crra_period_utility, gamma = gamma)
  differentials <- sweep(utility[, candidate_index, drop = FALSE], 1L,
                         utility[, benchmark_index], "-")
  n <- nrow(differentials)
  if (is.null(block_length)) block_length <- max(2L, ceiling(n^(1 / 3)))
  observed <- sqrt(n) * max(colMeans(differentials))
  centred <- sweep(differentials, 2L, colMeans(differentials), "-")
  set.seed(seed)
  bootstrap_max <- replicate(replications, {
    index <- moving_block_indices(n, block_length)
    sqrt(n) * max(colMeans(centred[index, , drop = FALSE]))
  })
  list(
    benchmark = benchmark,
    best_candidate = strategy_names[candidate_index[which.max(colMeans(differentials))]],
    observed_statistic = observed,
    p_value = (1 + sum(bootstrap_max >= observed)) / (replications + 1),
    mean_utility_differentials = setNames(colMeans(differentials),
                                          strategy_names[candidate_index]),
    block_length = block_length, replications = replications
  )
}

