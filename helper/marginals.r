# ====================================================================
# marginals.r
# Process raw asset returns by fitting to ARMA-GARCH.
# Extract standardised residuals and transform to Unif() for copulas.
# ====================================================================

suppressPackageStartupMessages({
  library(rugarch)
  library(xts)
})

RUN_TESTS <- FALSE

source("helper/load_data.r")


# Fit marginal for a single asset
fit_marginal <- function(return_series, asset_name = "asset",
                         allow_misspecified = FALSE) {
  # Let the marginal absorb asymmetric/leverage volatility before the copula
  # is fitted.  BIC selects among parsimonious GARCH variants; residual
  # whiteness determines whether a BIC winner is acceptable.
  fit_candidate <- function(variance_model, ar_order, ma_order, distribution,
                            garch_p, garch_q) {
    tryCatch({
      spec <- ugarchspec(
        mean.model = list(armaOrder = c(ar_order, ma_order), include.mean = TRUE),
        variance.model = list(model = variance_model,
                              garchOrder = c(garch_p, garch_q)),
        distribution.model = distribution
      )
      fit <- ugarchfit(spec = spec, data = return_series, solver = "hybrid")
      if (fit@fit$convergence != 0) stop("non-converged fit")
      z <- as.numeric(residuals(fit, standardize = TRUE))
      information_criteria <- as.numeric(infocriteria(fit))
      information_names <- tolower(names(infocriteria(fit)))
      bic_index <- grep("bayes|bic", information_names)
      bic_value <- if (length(bic_index)) information_criteria[bic_index[1L]] else if (length(information_criteria) >= 2L) information_criteria[2L] else Inf
      if (!length(bic_value) || !is.finite(bic_value)) bic_value <- Inf
      diagnostic_lags <- unique(pmin(c(5L, 10L, 20L), max(1L, floor(length(z) / 5))))
      lb_z <- vapply(diagnostic_lags, function(lag) {
        Box.test(z, lag = lag, type = "Ljung-Box", fitdf = ar_order)$p.value
      }, numeric(1))
      lb_z2 <- vapply(diagnostic_lags, function(lag) {
        Box.test(z^2, lag = lag, type = "Ljung-Box")$p.value
      }, numeric(1))
      list(model = variance_model, ar_order = ar_order, ma_order = ma_order,
        garch_p = garch_p, garch_q = garch_q,
        distribution = distribution, fit = fit, z = z,
        bic = as.numeric(bic_value),
        lb_z = min(lb_z), lb_z2 = min(lb_z2),
        diagnostic_lags = diagnostic_lags,
        lb_z_by_lag = lb_z, lb_z2_by_lag = lb_z2)
    }, error = function(e) NULL)
  }
  specification_grid <- expand.grid(
    variance_model = c("sGARCH", "gjrGARCH", "eGARCH"),
    ar_order = c(0L, 1L), ma_order = 0L,
    distribution = c("std", "sstd"), garch_p = 1L, garch_q = 1L,
    stringsAsFactors = FALSE
  )
  fit_grid <- function(grid) lapply(seq_len(nrow(grid)), function(i) {
    fit_candidate(grid$variance_model[i], grid$ar_order[i], grid$ma_order[i],
                  grid$distribution[i], grid$garch_p[i], grid$garch_q[i])
  })
  candidates <- Filter(Negate(is.null), fit_grid(specification_grid))
  acceptable <- vapply(candidates, function(x) x$lb_z >= .01 && x$lb_z2 >= .01, logical(1))
  if (!any(acceptable)) {
    # A causal two-component EWMA is a parsimonious long/short-horizon
    # volatility fallback for series whose volatility persistence is not
    # absorbed by a one-component GARCH. Parameters are selected by Gaussian
    # quasi-likelihood; diagnostics remain a gate, never the objective.
    component_candidates <- list()
    for (ar_order in 0:1) {
      if (ar_order == 1L) {
        mean_fit <- lm(return_series[-1L] ~ return_series[-length(return_series)])
        mu <- unname(coef(mean_fit)[1L]); ar1 <- unname(coef(mean_fit)[2L])
        innovations <- c(0, return_series[-1L] - mu -
                            ar1 * return_series[-length(return_series)])
      } else {
        mu <- mean(return_series); ar1 <- 0
        innovations <- return_series - mu
      }
      initial_variance <- var(innovations)
      for (lambda_short in c(0.70, 0.80, 0.85, 0.90, 0.94))
        for (lambda_long in c(0.98, 0.99, 0.995, 0.998))
          for (short_weight in c(0.10, 0.25, 0.50, 0.75, 0.90)) {
            short_variance <- long_variance <- rep(initial_variance, length(innovations))
            for (t in 2:length(innovations)) {
              short_variance[t] <- lambda_short * short_variance[t - 1L] +
                (1 - lambda_short) * innovations[t - 1L]^2
              long_variance[t] <- lambda_long * long_variance[t - 1L] +
                (1 - lambda_long) * innovations[t - 1L]^2
            }
            conditional_variance <- short_weight * short_variance +
              (1 - short_weight) * long_variance
            conditional_sigma <- sqrt(pmax(conditional_variance, 1e-12))
            z <- innovations / conditional_sigma
            scale_adjustment <- sqrt(mean(z^2))
            conditional_sigma <- conditional_sigma * scale_adjustment
            z <- innovations / conditional_sigma
            diagnostic_lags <- c(5L, 10L, 20L)
            lb_z <- vapply(diagnostic_lags, function(lag)
              Box.test(z, lag = lag, type = "Ljung-Box", fitdf = ar_order)$p.value,
              numeric(1))
            lb_z2 <- vapply(diagnostic_lags, function(lag)
              Box.test(z^2, lag = lag, type = "Ljung-Box")$p.value, numeric(1))
            log_likelihood <- sum(dnorm(z, log = TRUE) - log(conditional_sigma))
            parameter_count <- 5L + ar_order
            bic <- (-2 * log_likelihood + parameter_count * log(length(z))) / length(z)
            component_candidates[[length(component_candidates) + 1L]] <- list(
              model = "componentEWMA", ar_order = ar_order, ma_order = 0L,
              garch_p = NA_integer_, garch_q = NA_integer_, distribution = "empirical",
              fit = NULL, z = z, sigma = conditional_sigma, bic = bic,
              lb_z = min(lb_z), lb_z2 = min(lb_z2),
              diagnostic_lags = diagnostic_lags, lb_z_by_lag = lb_z,
              lb_z2_by_lag = lb_z2, mu = mu, ar1 = ar1,
              lambda_short = lambda_short, lambda_long = lambda_long,
              short_weight = short_weight, initial_variance = initial_variance,
              scale_adjustment = scale_adjustment)
          }
    }
    candidates <- c(candidates, component_candidates)
  }
  candidates <- Filter(Negate(is.null), candidates)
  if (!length(candidates)) stop(sprintf("All AR-GARCH candidates failed for %s.", asset_name))
  # A 1% diagnostic gate across fixed lags is conservative without treating a
  # single noisy 5% test as a model-selection oracle.
  acceptable <- vapply(candidates, function(x) x$lb_z >= .01 && x$lb_z2 >= .01, logical(1))
  if (!any(acceptable) && !isTRUE(allow_misspecified)) {
    stop(sprintf(paste0("No marginal specification passed residual diagnostics for %s. ",
                        "Do not fit the copula/RL model until the marginal grid is revised."),
                 asset_name))
  }
  selection_pool <- if (any(acceptable)) candidates[acceptable] else candidates
  bic_values <- vapply(selection_pool, function(candidate) {
    value <- as.numeric(candidate$bic)
    if (!length(value) || !is.finite(value[1L])) Inf else value[1L]
  }, numeric(1))
  best_index <- if (length(bic_values) && any(is.finite(bic_values))) which.min(bic_values) else 1L
  fit_summary <- selection_pool[[best_index]]
  fit <- fit_summary$fit

  # Extract residuals and transform to Unif(0,1)
  z <- fit_summary$z
  z_sorted <- sort(z)  # store for later
  u <- rank(z) / (length(z) + 1)                                         # use empirical CDF (ranks)
  # u <- pdist(distribution = "sstd", q = z, mu = 0, sigma = 1,          # instead of parametric skewed-t
  #            skew = coef(fit)["skew"], shape = coef(fit)["shape"])     # since skewed-t gave imperfect results.

  cat(sprintf("\n=== %s (%s(%s,%s) ARMA(%d,%d) %s) ===\n", asset_name,
              fit_summary$model, fit_summary$garch_p, fit_summary$garch_q,
              fit_summary$ar_order, fit_summary$ma_order, fit_summary$distribution))
  cat(sprintf("Mean(z)=%.4f  Var(z)=%.4f\n", mean(z), var(z)))
  cat(sprintf("LB p-val (z): %.4f   LB p-val (z^2): %.4f\n",
              fit_summary$lb_z, fit_summary$lb_z2))
  if (!any(acceptable)) cat("Warning: explicitly permitted misspecified marginal.\n")
  
  return(list(fit = fit, z = z, u = u, z_sorted = z_sorted,
              variance_model = fit_summary$model, ar_order = fit_summary$ar_order,
              ma_order = fit_summary$ma_order, garch_order = c(fit_summary$garch_p, fit_summary$garch_q),
              distribution = fit_summary$distribution, bic = fit_summary$bic,
              lb_z_pvalue = fit_summary$lb_z, lb_z2_pvalue = fit_summary$lb_z2,
              marginal_type = if (fit_summary$model == "componentEWMA") "component_ewma" else "rugarch",
              mu_ar = if (fit_summary$model == "componentEWMA") fit_summary$mu else NULL,
              ar1 = if (fit_summary$model == "componentEWMA") fit_summary$ar1 else NULL,
              sigma = if (fit_summary$model == "componentEWMA") fit_summary$sigma else NULL,
              component_parameters = if (fit_summary$model == "componentEWMA")
                list(lambda_short = fit_summary$lambda_short,
                     lambda_long = fit_summary$lambda_long,
                     short_weight = fit_summary$short_weight,
                     initial_variance = fit_summary$initial_variance,
                     scale_adjustment = fit_summary$scale_adjustment) else NULL))
}

# Fit marginals on a training prefix only, then filter any later observations
# with fixed training parameters.  Pseudo-observations use the training
# empirical residual CDF, so the evaluation period cannot alter a rank.
fit_marginals_training <- function(returns_train) {
  names <- colnames(returns_train)
  models <- vector("list", length(names)); names(models) <- names
  for (i in seq_along(names)) models[[i]] <- fit_marginal(as.numeric(returns_train[, i]), names[i])
  models
}

filter_training_marginals <- function(returns_all, marginals_train) {
  asset_names <- colnames(returns_all)
  z_matrix <- sigma_matrix <- matrix(NA_real_, nrow = nrow(returns_all),
    ncol = length(asset_names), dimnames = list(NULL, asset_names))
  for (i in seq_along(asset_names)) {
    model <- marginals_train[[asset_names[i]]]
    if (identical(model$marginal_type, "component_ewma")) {
      y <- as.numeric(returns_all[, i])
      innovations <- c(y[1L] - model$mu_ar,
        y[-1L] - model$mu_ar - model$ar1 * y[-length(y)])
      parameters <- model$component_parameters
      short_variance <- long_variance <- rep(parameters$initial_variance, length(y))
      for (t in 2:length(y)) {
        short_variance[t] <- parameters$lambda_short * short_variance[t - 1L] +
          (1 - parameters$lambda_short) * innovations[t - 1L]^2
        long_variance[t] <- parameters$lambda_long * long_variance[t - 1L] +
          (1 - parameters$lambda_long) * innovations[t - 1L]^2
      }
      conditional_sigma <- sqrt(parameters$short_weight * short_variance +
        (1 - parameters$short_weight) * long_variance)
      conditional_sigma <- conditional_sigma * parameters$scale_adjustment
      z <- innovations / pmax(conditional_sigma, 1e-12)
    } else {
      spec <- getspec(model$fit)
      setfixed(spec) <- as.list(coef(model$fit))
      filtered <- ugarchfilter(spec, data = as.numeric(returns_all[, i]))
      z <- as.numeric(residuals(filtered, standardize = TRUE))
      conditional_sigma <- as.numeric(sigma(filtered))
    }
    z_matrix[, i] <- z
    sigma_matrix[, i] <- conditional_sigma
  }
  if (any(!is.finite(z_matrix)) || any(!is.finite(sigma_matrix)) || any(sigma_matrix <= 0)) {
    stop("Fixed-parameter causal marginal filtering produced invalid states.")
  }
  list(z = z_matrix, sigma = sigma_matrix)
}

training_pseudo_observations <- function(returns_all, marginals_train) {
  asset_names <- colnames(returns_all)
  states <- filter_training_marginals(returns_all, marginals_train)
  U <- matrix(NA_real_, nrow = nrow(returns_all), ncol = length(asset_names), dimnames = list(NULL, asset_names))
  for (i in seq_along(asset_names)) {
    model <- marginals_train[[asset_names[i]]]
    # Mid-ranks avoid exact 0/1 inputs, which are invalid for vine fitting.
    U[, i] <- (findInterval(states$z[, i], model$z_sorted, rightmost.closed = TRUE) + 0.5) /
      (length(model$z_sorted) + 1)
  }
  pmin(pmax(U, 1e-6), 1 - 1e-6)
}

# =========================================================================================

if (RUN_TESTS) {
  raw_returns <- load_returns()
  asset_names <- colnames(raw_returns)
  # Call function to fit all marginals
  marginals <- list()
  U <- NULL

  for (i in seq_along(asset_names)) {
    name <- asset_names[i]
    return <- as.numeric(raw_returns[,i])
    marginals[[name]] <- fit_marginal(return, name)
    
    if (is.null(U)) {
      U <- marginals[[name]]$u
    } else {
      U <- cbind(U, marginals[[name]]$u)
    }
  }

  colnames(U) <- asset_names
  save(marginals, U, asset_names, file = "data/marginal_results.RData")
  print("Data saved to marginal_results.RData")
}

#################################
# Note (07/07/2026):
# All assets pass the Ljung‑Box test at lag 10 except DIVIDEND 
# (p=0.024 at lag 10 for squared residuals). As a robustness 
# check I also use lag 5 where DIVIDEND marginally fails, and 
# the main results are qualitatively unchanged.
#################################
