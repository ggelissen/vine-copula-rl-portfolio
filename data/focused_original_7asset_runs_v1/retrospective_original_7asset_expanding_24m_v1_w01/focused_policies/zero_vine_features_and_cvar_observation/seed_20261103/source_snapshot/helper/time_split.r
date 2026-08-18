# Calendar-safe monthly decision/holding-period construction.
# A decision at month-end t is evaluated on (t, next month-end].  The final
# price observation is therefore a holding-period end, never a decision date.

build_monthly_periods <- function(returns_xts, min_history = 0L) {
  if (!xts::is.xts(returns_xts) || nrow(returns_xts) < 2L) stop("returns_xts must be a non-empty xts object.")
  endpoints_all <- xts::endpoints(returns_xts, on = "months")
  endpoints_all <- endpoints_all[endpoints_all > 0L]
  if (tail(endpoints_all, 1L) != nrow(returns_xts)) endpoints_all <- c(endpoints_all, nrow(returns_xts))
  if (length(endpoints_all) < 2L) stop("At least two monthly endpoints are required.")

  decision_idx <- head(endpoints_all, -1L)
  holding_end_idx <- tail(endpoints_all, -1L)
  keep <- decision_idx >= as.integer(min_history)
  decision_idx <- decision_idx[keep]; holding_end_idx <- holding_end_idx[keep]
  if (!length(decision_idx)) stop("No monthly periods remain after applying min_history.")

  data.frame(
    period_id = seq_along(decision_idx),
    decision_idx = decision_idx,
    holding_end_idx = holding_end_idx,
    decision_date = as.Date(zoo::index(returns_xts)[decision_idx]),
    holding_end_date = as.Date(zoo::index(returns_xts)[holding_end_idx])
  )
}

split_monthly_periods <- function(periods, evaluation_periods = 24L) {
  evaluation_periods <- as.integer(evaluation_periods)
  if (evaluation_periods < 1L || nrow(periods) <= evaluation_periods) stop("Not enough periods for the requested evaluation holdout.")
  split_at <- nrow(periods) - evaluation_periods
  list(train = periods[seq_len(split_at), , drop = FALSE],
       evaluation = periods[(split_at + 1L):nrow(periods), , drop = FALSE])
}

realised_gross_for_period <- function(returns_xts, decision_date, holding_end_date) {
  interval <- returns_xts[paste0(as.Date(decision_date) + 1, "/", as.Date(holding_end_date))]
  if (!nrow(interval)) stop(sprintf("Empty holding period: (%s, %s].", decision_date, holding_end_date))
  gross <- exp(colSums(interval))
  if (any(!is.finite(gross)) || any(gross <= 0)) stop("Invalid realised gross return.")
  gross
}

validate_period_split <- function(split, expected_evaluation_periods = 24L) {
  train <- split$train; evaluation <- split$evaluation
  stopifnot(nrow(evaluation) == as.integer(expected_evaluation_periods))
  stopifnot(max(train$holding_end_date) <= min(evaluation$decision_date))
  stopifnot(all(evaluation$holding_end_date > evaluation$decision_date))
  invisible(TRUE)
}
