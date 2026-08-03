# ============================================================
# load_data.r
# Data loading and preprocessing for asset pricing data
# ============================================================

library(xts)
library(zoo)
library(moments)

# Load raw log returns
load_returns <- function(filepath = "data/portfolio_B_7assets_2013.csv") {
  if (!file.exists(filepath)) stop("Price file not found: ", filepath)
  prices_df <- read.csv(filepath, check.names = FALSE)
  if (ncol(prices_df) < 2L || nrow(prices_df) < 3L) stop("Price file is empty or malformed.")
  dates <- as.Date(prices_df[[1L]])
  if (anyNA(dates) || anyDuplicated(dates) || is.unsorted(dates, strictly = TRUE)) {
    stop("Dates must be valid, unique, and strictly increasing; input is never silently reordered.")
  }
  values <- as.matrix(prices_df[, -1L, drop = FALSE])
  storage.mode(values) <- "double"
  if (any(!is.finite(values)) || any(values <= 0)) stop("Prices must be finite and strictly positive.")
  if (anyDuplicated(colnames(values))) stop("Asset names must be unique.")
  prices <- xts(values, order.by = dates)
  returns <- na.omit(diff(log(prices)))
  if (nrow(returns) != nrow(prices) - 1L || any(!is.finite(returns))) {
    stop("Log-return construction produced missing/non-finite observations.")
  }
  attr(returns, "source_file") <- normalizePath(filepath, winslash = "/", mustWork = TRUE)
  attr(returns, "source_md5") <- unname(tools::md5sum(filepath))
  return(returns)
}

# Build compatible returns list
preprocess_returns <- function(returns, ref_col = 7, L = 60,             
                               T = 12, nfreq = "monthly", end_date = NULL) {
  
  # Convert log returns to simple returns
  simple_returns <- exp(returns)
  asset_names <- colnames(simple_returns)
  d_total <- ncol(simple_returns)
  
  cat(sprintf("Input data: %d rows, %d columns\n", nrow(simple_returns), d_total))
  cat(sprintf("Columns: %s\n", paste(asset_names, collapse = ", ")))
  cat(sprintf("ref_col = %d (asset: %s)\n\n", ref_col, asset_names[ref_col]))
  
  risk_cols <- setdiff(1:d_total, ref_col)
  
  if (is.null(end_date)) {
    end_date <- index(simple_returns)[nrow(simple_returns)]
  }
  
  # Aggregate data if needed
  if (nfreq == "weekly") {
    ep <- endpoints(simple_returns, on = "weeks")
    simple_returns <- period.apply(simple_returns, INDEX = ep, FUN = function(x) apply(x, 2, prod))
  } else if (nfreq == "monthly") {
    ep <- endpoints(simple_returns, on = "months")
    simple_returns <- period.apply(simple_returns, INDEX = ep, FUN = function(x) apply(x, 2, prod))
  }
  
  cat(sprintf("After aggregation (%s): %d rows, %d columns\n", nfreq, nrow(simple_returns), ncol(simple_returns)))
  cat(sprintf("Columns: %s\n\n", paste(colnames(simple_returns), collapse = ", ")))
  
  simple_returns <- simple_returns[paste0("/", end_date)]
  
  # Number of available observations
  N_total <- nrow(simple_returns)
  if (N_total < L + T) {
    stop(sprintf("Not enough data: need at least %d observations, have %d.\nTry reducing L (currently %d) or T (currently %d).", 
                 L + T, N_total, L, T))
  }
  
  # Build the returns list
  returns_list <- vector("list", T)
  dates_used <- index(simple_returns)[(L + 1):(L + T)]
  
  for (t in 1:T) {
    current_idx <- L + t
    window_start <- current_idx - L
    window_end   <- current_idx - 1
    
    window_data <- as.matrix(simple_returns[window_start:window_end, ])
    
    if (t == 1) {
      cat(sprintf("Window dimensions: %d rows × %d columns\n", nrow(window_data), ncol(window_data)))
      cat(sprintf("ref_col = %d, risk_cols = %s\n", ref_col, paste(risk_cols, collapse = ", ")))
    }
    
    r0 <- window_data[, ref_col]
    e  <- window_data[, risk_cols, drop = FALSE]
    
    returns_list[[t]] <- cbind(r0, e)
  }
  
  return(list(
    returns_list = returns_list,
    risk_cols    = risk_cols,
    ref_col      = ref_col,
    dates        = dates_used,
    asset_names  = asset_names,
    T            = T,
    L            = L
  ))
}



# Descriptive statistics
compute_descriptive_stats <- function(returns) {
  
  stats <- data.frame(
    Mean     = colMeans(returns) * 100,        # convert to %
    Std_Dev  = apply(returns, 2, sd) * 100,    # convert to %
    Skewness = apply(returns, 2, skewness),
    Kurtosis = apply(returns, 2, kurtosis),
    Min      = apply(returns, 2, min) * 100,
    Max      = apply(returns, 2, max) * 100,
    JB_Stat  = apply(returns, 2, function(x) {
      n <- length(x)
      s <- skewness(x)
      k <- kurtosis(x)
      n * (s^2 / 6 + k^2 / 24)
    }),
    JB_pval  = apply(returns, 2, function(x) {
      n <- length(x)
      s <- skewness(x)
      k <- kurtosis(x)
      jb <- n * (s^2 / 6 + k^2 / 24)
      1 - pchisq(jb, df = 2)
    })
  )
  
  # Add Ljung–Box on squared returns (20 lags)
  stats$LB_pval <- apply(returns, 2, function(x) {
    Box.test(x^2, lag = 20, type = "Ljung-Box")$p.value
  })
  
  rownames(stats) <- colnames(returns)
  round(stats, 4)
}

# ========================================================================

#returns <- load_returns()
#desc_stats <- compute_descriptive_stats(returns)
