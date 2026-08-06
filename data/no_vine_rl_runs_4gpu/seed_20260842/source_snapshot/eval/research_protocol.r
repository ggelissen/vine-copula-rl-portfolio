# Publication-grade common evaluation contract.
# Strategies supply only ex-ante weights. This evaluator applies the same
# realised asset returns, leverage limits, transaction costs and borrow costs
# to every method.

suppressPackageStartupMessages(library(data.table))
source("helper/time_split.r")
source("eval/ablation.r")
source("eval/statistical_tests.r")

validate_weight_log <- function(x, periods, asset_names, net_exposure,
                                gross_leverage, tolerance = 1e-6) {
  x <- as.data.table(x)
  weight_columns <- paste0("w_", asset_names)
  required <- c("decision_date", weight_columns)
  missing <- setdiff(required, names(x))
  if (length(missing)) stop("Weight log is missing: ", paste(missing, collapse = ", "))
  x[, decision_date := as.Date(decision_date)]
  if (nrow(x) != nrow(periods) ||
      !identical(x$decision_date, as.Date(periods$decision_date))) {
    stop("Weight log is not aligned to the locked evaluation periods.")
  }
  weights <- as.matrix(x[, ..weight_columns])
  storage.mode(weights) <- "double"
  if (any(!is.finite(weights))) stop("Non-finite portfolio weight.")
  net <- rowSums(weights); gross <- rowSums(abs(weights))
  if (any(abs(net - net_exposure) > tolerance)) {
    stop("A strategy violates the common net-exposure constraint.")
  }
  if (any(gross > gross_leverage + tolerance)) {
    stop("A strategy violates the common gross-leverage constraint.")
  }
  weights
}

score_weight_log <- function(weights, asset_gross, periods, asset_names,
                             initial_wealth = 100000, net_exposure = 1,
                             gross_leverage = 1.5, turnover_cost = 0.001,
                             short_borrow_rate = 0.03,
                             cash_borrow_rate = 0.02) {
  if (!identical(dim(weights), dim(asset_gross))) stop("Weight/return dimensions differ.")
  previous <- rep(net_exposure / ncol(weights), ncol(weights))
  rows <- vector("list", nrow(weights))
  wealth <- initial_wealth
  for (t in seq_len(nrow(weights))) {
    w <- weights[t, ]
    turnover <- sum(abs(w - previous))
    transaction_cost <- turnover_cost * turnover
    short_notional <- sum(pmax(-w, 0))
    cash_borrow_notional <- pmax(sum(w) - 1, 0)
    financing_cost <- (short_borrow_rate * short_notional +
                       cash_borrow_rate * cash_borrow_notional) / 12
    gross_portfolio_return <- 1 + sum(w * (asset_gross[t, ] - 1))
    net_gross <- gross_portfolio_return * exp(-transaction_cost - financing_cost)
    if (!is.finite(net_gross) || net_gross <= 0) {
      stop(sprintf("Portfolio insolvency/non-positive wealth at evaluation step %d.", t))
    }
    wealth <- wealth * net_gross
    rows[[t]] <- data.table(
      decision_date = as.Date(periods$decision_date[t]),
      holding_end_date = as.Date(periods$holding_end_date[t]),
      gross_return = gross_portfolio_return - 1,
      net_return = net_gross - 1, turnover = turnover,
      transaction_cost = transaction_cost, financing_cost = financing_cost,
      short_notional = short_notional, wealth = wealth
    )
    previous <- w
  }
  rbindlist(rows)
}

run_research_evaluation <- function(returns_xts, strategy_weight_logs,
                                    evaluation_periods = 24L,
                                    min_history = 250L,
                                    initial_wealth = 100000,
                                    net_exposure = 1,
                                    gross_leverage = 1.5,
                                    turnover_cost = 0.001,
                                    short_borrow_rate = 0.03,
                                    cash_borrow_rate = 0.02,
                                    gamma = 2,
                                    output_dir = "data/research_evaluation") {
  if (!is.list(strategy_weight_logs) || length(strategy_weight_logs) < 2L ||
      is.null(names(strategy_weight_logs))) {
    stop("Provide at least two named strategy weight logs.")
  }
  split <- split_monthly_periods(
    build_monthly_periods(returns_xts, min_history), evaluation_periods
  )
  validate_period_split(split, evaluation_periods)
  periods <- split$evaluation
  asset_names <- colnames(returns_xts)
  asset_gross <- do.call(rbind, lapply(seq_len(nrow(periods)), function(t) {
    as.numeric(realised_gross_for_period(
      returns_xts, periods$decision_date[t], periods$holding_end_date[t]
    ))
  }))
  colnames(asset_gross) <- asset_names

  scored <- lapply(names(strategy_weight_logs), function(name) {
    log <- strategy_weight_logs[[name]]
    if (is.character(log) && length(log) == 1L) log <- fread(log)
    weights <- validate_weight_log(log, periods, asset_names, net_exposure,
                                   gross_leverage)
    score_weight_log(weights, asset_gross, periods, asset_names, initial_wealth,
                     net_exposure, gross_leverage, turnover_cost,
                     short_borrow_rate, cash_borrow_rate)
  })
  names(scored) <- names(strategy_weight_logs)

  metrics <- rbindlist(lapply(names(scored), function(name) {
    as.data.table(as.list(annualised_path_metrics(scored[[name]]$net_return)))[
      , strategy := name]
  }), fill = TRUE)
  inference_input <- lapply(scored, function(x) {
    data.frame(date = x$holding_end_date, net_return = x$net_return)
  })
  pairwise <- pairwise_utility_tests(inference_input, gamma = gamma)
  # The first named method is the pre-declared benchmark, never selected after
  # looking at performance.
  reality <- reality_check(inference_input, benchmark = names(scored)[1L],
                           gamma = gamma)

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  fwrite(metrics, file.path(output_dir, "metrics.csv"))
  fwrite(pairwise, file.path(output_dir, "pairwise_hac_utility_tests.csv"))
  for (name in names(scored)) {
    fwrite(scored[[name]], file.path(output_dir, paste0("returns_", name, ".csv")))
  }
  saveRDS(list(periods = periods, metrics = metrics, pairwise = pairwise,
               reality_check = reality, contract = list(
                 evaluation_periods = evaluation_periods,
                 min_history = min_history, initial_wealth = initial_wealth,
                 net_exposure = net_exposure, gross_leverage = gross_leverage,
                 turnover_cost = turnover_cost,
                 short_borrow_rate = short_borrow_rate,
                 cash_borrow_rate = cash_borrow_rate, gamma = gamma)),
          file.path(output_dir, "evaluation_bundle.rds"))
  list(periods = periods, returns = scored, metrics = metrics,
       pairwise = pairwise, reality_check = reality)
}

