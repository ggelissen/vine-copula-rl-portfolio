# ============================================================
# load_data.r
# Data loading and preprocessing for asset pricing data
# ============================================================

suppressPackageStartupMessages({
  library(xts)
  library(zoo)
  library(moments)
})

# Hash and manifest helpers are deliberately kept next to the loader.  Every
# R entry point calls load_returns(), so this is the single fail-closed boundary
# that prevents a seven-asset checkpoint from being silently trained/evaluated
# on a different panel or on data beyond a frozen walk-forward window.
sha256_file <- function(path) {
  if (requireNamespace("digest", quietly = TRUE)) {
    return(digest::digest(file = path, algo = "sha256", serialize = FALSE))
  }
  executable <- Sys.which("sha256sum")
  if (nzchar(executable)) {
    output <- system2(executable, shQuote(normalizePath(
      path, winslash = "/", mustWork = TRUE)), stdout = TRUE, stderr = TRUE)
    command_status <- attr(output, "status")
    if (is.null(command_status)) command_status <- 0L
    if (!length(output) || command_status != 0L) {
      stop("sha256sum failed for: ", path)
    }
    return(strsplit(output[1L], "[[:space:]]+")[[1L]][1L])
  }
  stop("Package 'digest' or executable 'sha256sum' is required for SHA-256 verification.")
}

read_return_input_manifest <- function(path) {
  if (!nzchar(path) || !file.exists(path)) {
    stop("A valid RETURNS_DATA_MANIFEST is required for daily_log_returns input.")
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required to read the frozen return-data contract.")
  }
  manifest <- jsonlite::fromJSON(path, simplifyVector = TRUE)
  if (!identical(as.integer(manifest$schema_version), 1L) ||
      !identical(manifest$release_status,
                 "frozen_window_return_input_no_confirmation")) {
    stop("Return-data manifest is not a supported frozen window input release.")
  }
  if (!identical(manifest$confirmatory_claim_permitted, FALSE)) {
    stop("Development return input may not authorize a confirmatory claim.")
  }
  manifest
}

# Load either adjusted total-return levels (legacy/default) or a canonical
# frozen daily log-return panel.  Environment defaults preserve every existing
# seven-asset run; external-panel jobs must opt in explicitly.
load_returns <- function(
    filepath = Sys.getenv("RETURNS_DATA_FILE",
                          "data/portfolio_B_7assets_2013.csv"),
    input_kind = Sys.getenv("RETURNS_DATA_KIND", "adjusted_levels"),
    manifest_file = Sys.getenv("RETURNS_DATA_MANIFEST", "")) {
  input_kind <- tolower(trimws(input_kind))
  if (!input_kind %in% c("adjusted_levels", "daily_log_returns")) {
    stop("RETURNS_DATA_KIND must be adjusted_levels or daily_log_returns.")
  }
  if (!file.exists(filepath)) stop("Return data file not found: ", filepath)
  input_df <- read.csv(filepath, check.names = FALSE)
  if (ncol(input_df) < 2L || nrow(input_df) < 3L) {
    stop("Return data file is empty or malformed.")
  }
  dates <- as.Date(input_df[[1L]])
  if (anyNA(dates) || anyDuplicated(dates) || is.unsorted(dates, strictly = TRUE)) {
    stop("Dates must be valid, unique, and strictly increasing; input is never silently reordered.")
  }
  values <- as.matrix(input_df[, -1L, drop = FALSE])
  storage.mode(values) <- "double"
  if (any(!is.finite(values))) stop("Return input must be complete and finite.")
  if (anyDuplicated(colnames(values)) || any(!nzchar(colnames(values)))) {
    stop("Asset names must be non-empty and unique.")
  }

  manifest <- NULL
  if (identical(input_kind, "adjusted_levels")) {
    if (any(values <= 0)) stop("Adjusted levels must be strictly positive.")
    prices <- xts(values, order.by = dates)
    returns <- na.omit(diff(log(prices)))
    if (nrow(returns) != nrow(prices) - 1L || any(!is.finite(returns))) {
      stop("Log-return construction produced missing/non-finite observations.")
    }
  } else {
    manifest <- read_return_input_manifest(manifest_file)
    asset_order <- as.character(unlist(manifest$asset_order, use.names = FALSE))
    if (!identical(colnames(values), asset_order)) {
      stop("Daily return columns/order do not match the frozen manifest.")
    }
    if (!identical(as.integer(manifest$return_rows), nrow(values))) {
      stop("Daily return row count does not match the frozen manifest.")
    }
    if (!identical(as.character(manifest$date_start), as.character(min(dates))) ||
        !identical(as.character(manifest$date_end), as.character(max(dates)))) {
      stop("Daily return date bounds do not match the frozen manifest.")
    }
    actual_hash <- sha256_file(filepath)
    if (!identical(tolower(actual_hash),
                   tolower(as.character(manifest$return_file_sha256)))) {
      stop("Daily return file hash does not match the frozen manifest.")
    }
    returns <- xts(values, order.by = dates)
  }
  attr(returns, "source_file") <- normalizePath(filepath, winslash = "/", mustWork = TRUE)
  attr(returns, "source_md5") <- unname(tools::md5sum(filepath))
  attr(returns, "source_sha256") <- sha256_file(filepath)
  attr(returns, "source_kind") <- input_kind
  if (!is.null(manifest)) {
    attr(returns, "source_manifest") <- normalizePath(
      manifest_file, winslash = "/", mustWork = TRUE)
    attr(returns, "source_manifest_sha256") <- sha256_file(manifest_file)
    attr(returns, "panel_id") <- as.character(manifest$panel_id)
    attr(returns, "window_id") <- as.character(manifest$window_id)
    attr(returns, "expected_evaluation_start") <-
      as.Date(manifest$expected_evaluation_start)
    attr(returns, "expected_evaluation_end") <-
      as.Date(manifest$expected_evaluation_end)
    attr(returns, "expected_evaluation_periods") <-
      as.integer(manifest$expected_evaluation_periods)
    attr(returns, "reference_asset_index_1based") <-
      as.integer(manifest$reference_asset_index_1based)
    attr(returns, "vine_truncation_level") <-
      as.integer(manifest$vine_truncation_level)
  }
  return(returns)
}

validate_return_model_contract <- function(returns, ref_col,
                                           vine_truncation_level = 0L) {
  if (!identical(attr(returns, "source_kind"), "daily_log_returns")) {
    return(invisible(TRUE))
  }
  d <- ncol(returns)
  requested <- suppressWarnings(as.integer(vine_truncation_level))
  if (length(requested) != 1L || is.na(requested) || requested < 0L) {
    stop("Invalid requested vine truncation level.")
  }
  active <- if (requested == 0L) d - 1L else requested
  if (!identical(as.integer(ref_col),
                 attr(returns, "reference_asset_index_1based")) ||
      !identical(active, attr(returns, "vine_truncation_level"))) {
    stop("Reference asset or vine truncation disagrees with the frozen window contract.")
  }
  invisible(TRUE)
}

validate_return_evaluation_contract <- function(returns, period_split,
                                                evaluation_periods) {
  if (!identical(attr(returns, "source_kind"), "daily_log_returns")) {
    return(invisible(TRUE))
  }
  expected_periods <- attr(returns, "expected_evaluation_periods")
  expected_start <- attr(returns, "expected_evaluation_start")
  expected_end <- attr(returns, "expected_evaluation_end")
  if (length(expected_periods) != 1L || is.na(expected_periods) ||
      expected_periods != as.integer(evaluation_periods) ||
      nrow(period_split$evaluation) != expected_periods ||
      min(period_split$evaluation$decision_date) != expected_start ||
      max(period_split$evaluation$holding_end_date) != expected_end) {
    stop("Computed evaluation split does not match the frozen window contract.")
  }
  invisible(TRUE)
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
