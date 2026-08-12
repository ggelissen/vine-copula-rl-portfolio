# ============================================================================
# rl_environment.R
# RL environment for vine‑copula portfolio selection
# ============================================================================

suppressPackageStartupMessages({
  library(R6)
  library(rvinecopulib)
})

source("helper/load_data.r")
source("benchmark_models/expected_utility_single.r")   # VineReturnSimulator, build_simulator, crra_utility
# Marginals and vines are supplied explicitly to RLEnvironment$new().
# Neural-vine fitting/prediction is intentionally not sourced here.  Training
# consumes precomputed returns and vine states; loading R torch/Lantern in the
# same process as reticulate's Python PyTorch causes incompatible LibTorch
# symbols to collide on Linux.

# Helper: extract all vine parameters (Kendall's tau + tail dependence)
extract_vine_state <- function(vine) {
  if (is.null(vine)) return(numeric(0))
  
  n_trees <- length(vine$pair_copulas)
  all_features <- c()
  
  for (tree_idx in seq_len(n_trees)) {
    pc_list <- vine$pair_copulas[[tree_idx]]
    tree_features <- numeric(length(pc_list) * 3)
    
    for (i in seq_along(pc_list)) {
      pc <- pc_list[[i]]
      fam <- pc$family
      params <- pc$parameters
      rot <- if (is.null(pc$rotation)) 0 else pc$rotation
      
      # Kendall's tau
      tau <- tryCatch(
        as.numeric(par_to_ktau(fam, params, rotation = rot)),
        error = function(e) NA
      )
      
      # Tail dependence
      lt <- 0; ut <- 0
      if (fam == "t") {
        rho <- params[1]; nu <- params[2]
        lt <- 2 * pt(-sqrt((nu+1)*(1-rho)/(1+rho)), df = nu+1)
        ut <- lt
      } else if (fam == "gumbel") {
        theta <- params[1]
        if (rot == 0 || rot == 90) ut <- 2 - 2^(1/theta)
        if (rot == 180 || rot == 270) lt <- 2 - 2^(1/theta)
      } else if (fam == "clayton") {
        theta <- params[1]
        if (rot == 0 || rot == 90) lt <- 2^(-1/theta)
        if (rot == 180 || rot == 270) ut <- 2^(-1/theta)
      } else if (fam == "joe") {
        theta <- params[1]
        if (rot == 0 || rot == 90) ut <- 2 - 2^(1/theta)
        if (rot == 180 || rot == 270) lt <- 2 - 2^(1/theta)
      }
      # Gaussian and Frank: no tail dependence
      
      idx <- (i-1)*3 + 1
      tree_features[idx]   <- ifelse(is.na(tau), 0, tau)
      tree_features[idx+1] <- lt
      tree_features[idx+2] <- ut
    }
    all_features <- c(all_features, tree_features)
  }
  all_features
}


# Legacy benchmark helper only. The RL/synthetic/evaluation pipeline uses
# build_nn_vine_sequence() and never calls this rolling estimator.
build_rolling_vine_sequence_legacy <- function(returns_xts, U, rebal_dates, L = 500) {
  vine_seq <- vector("list", length(rebal_dates))
  T_max <- nrow(returns_xts)
  
  for (t in seq_along(rebal_dates)) {
    window_end   <- which(index(returns_xts) == rebal_dates[t])
    if (window_end < L || window_end > T_max) next
    
    window_start <- window_end - L + 1
    U_window <- U[window_start:window_end, ]
    
    vine_seq[[t]] <- vinecop(
      U_window,
      var_types  = rep("c", ncol(U_window)),
      structure  = dvine_structure(1:ncol(U_window)),
      family_set = c("gaussian","t","clayton","gumbel","frank","joe"),
      selcrit    = "aic"
    )
  }
  
  vine_seq[!sapply(vine_seq, is.null)]
}

# The fitted marginals are daily, whereas this project rebalances monthly.
# Preserve the copula draw while aggregating daily innovations to a monthly
# holding-period log return.  Only the first row is an episode's realised
# draw, so it receives the AR carry-over; all remaining rows are CVaR scenarios.
simulate_monthly_gross <- function(simulator, marginals, vine, n_sim,
                                   holding_days = 21L, cores = 1L,
                                   previous_monthly_log_returns = NULL) {
  raw <- simulator$simulate_returns(vine, n_sim = n_sim, cores = cores, prev_returns = NULL)$log
  asset_names <- simulator$asset_names
  # build_simulator() enriches a local copy of marginals, so those derived
  # fields are not guaranteed to exist in the caller's loaded RData object.
  # Derive them from the fitted AR-GARCH coefficients when absent.
  marginal_moments <- function(model) {
    if (identical(model$marginal_type, "component_ewma")) {
      mu_uncond <- if (abs(model$ar1) < 1) model$mu_ar / (1 - model$ar1) else model$mu_ar
      return(c(mu_uncond = mu_uncond, ar1 = model$ar1,
               sigma_uncond = sqrt(mean(model$sigma^2))))
    }
    cfit <- model$fit@fit$coef
    mu_ar <- as.numeric(cfit["mu"])
    ar1 <- if ("ar1" %in% names(cfit)) as.numeric(cfit["ar1"]) else 0
    omega <- as.numeric(cfit["omega"]); alpha <- as.numeric(cfit["alpha1"]); beta <- as.numeric(cfit["beta1"])
    mu_uncond <- if (!is.null(model$mu_uncond) && length(model$mu_uncond) == 1L) as.numeric(model$mu_uncond) else if (abs(ar1) < 1) mu_ar / (1 - ar1) else mean(model$z, na.rm = TRUE)
    sigma_uncond <- if (!is.null(model$sigma_uncond) && length(model$sigma_uncond) == 1L) as.numeric(model$sigma_uncond) else if (is.finite(alpha + beta) && alpha + beta < 1) sqrt(omega / (1 - alpha - beta)) else sd(model$z, na.rm = TRUE)
    c(mu_uncond = mu_uncond, ar1 = ar1, sigma_uncond = sigma_uncond)
  }
  moments <- lapply(asset_names, function(name) marginal_moments(marginals[[name]]))
  mu <- vapply(moments, `[[`, numeric(1), "mu_uncond")
  monthly_log <- sweep(raw, 2, mu, "-") * sqrt(holding_days) + rep(holding_days * mu, each = nrow(raw))
  if (!is.null(previous_monthly_log_returns) && length(previous_monthly_log_returns) == length(asset_names)) {
    ar1 <- vapply(moments, `[[`, numeric(1), "ar1")
    monthly_log[1L, ] <- monthly_log[1L, ] + ar1 * previous_monthly_log_returns
  }
  colnames(monthly_log) <- asset_names
  exp(monthly_log)
}


# CRRA Utility Function
crra_utility <- function(wealth, gamma) {
  if (gamma == 1) {
    return(log(wealth))
  } else {
    return((wealth^(1-gamma) - 1) / (1 - gamma))
  }
}

# Sanitize externally supplied portfolio weights without restoring the old
# long-only simplex.  The actor produces valid self-financing long-short
# weights; this is a defensive projection for numerical noise and API callers.
project_long_short_weights <- function(weights, net_exposure, gross_leverage) {
  weights <- as.numeric(weights)
  weights[!is.finite(weights)] <- 0
  d <- length(weights)
  if (!d) stop("Portfolio must contain at least one asset.")
  base <- rep(net_exposure / d, d)
  centred <- weights - mean(weights)
  candidate <- base + centred
  gross <- sum(abs(candidate))
  if (gross > gross_leverage + 1e-10 && sum(abs(centred)) > 0) {
    # Bisection preserves net exposure while shrinking only active long/short
    # tilts until the gross-leverage limit binds.
    lower <- 0; upper <- 1
    for (iter in seq_len(50L)) {
      middle <- (lower + upper) / 2
      if (sum(abs(base + middle * centred)) > gross_leverage) upper <- middle else lower <- middle
    }
    candidate <- base + lower * centred
  }
  candidate
}


# ============================================================================
# RLEnvironment class – fully optimised
# ============================================================================
RLEnvironment <- R6Class(
  "RLEnvironment",
  public = list(
    
    # Constructor
    initialize = function(marginals, asset_names, 
                          vine = NULL, vine_sequence = NULL, 
                          ref_col = 7, 
                          gamma = 2,           # CRRA risk aversion
                          lambda = 1.0,        # CVaR penalty
                          kappa = 0.05,        # Transaction cost penalty
                          T = 12,              # Episode length
                          w0 = 100000, 
                          n_sim_cvar = 10000,
                          sim_cores = 1L,
                           seq_len = 30,
                           holding_days = 21L,
                           gross_leverage = 1.5,
                           net_exposure = 1.0,
                           max_long_weight = 0.60,
                           max_short_weight = 0.20,
                           short_borrow_rate = 0.03,
                           cash_borrow_rate = 0.02,
                           utility_mode = c("terminal_wealth_crra", "one_period_crra"),
                           vine_observation_mode = c("full", "zero"),
                           episode_sampling = c("random", "sequential")) {

      private$simulator <- build_simulator(marginals, asset_names, ref_col)
      private$marginals <- marginals
      private$ref_col <- ref_col
      private$gamma <- gamma
      private$lambda <- lambda
      private$kappa <- kappa
      private$T <- T
      private$w0 <- w0
      private$n_sim_cvar <- n_sim_cvar
      private$sim_cores <- max(1L, as.integer(sim_cores))
      private$seq_len <- seq_len
      private$holding_days <- as.integer(holding_days)
      private$gross_leverage <- as.numeric(gross_leverage)
      private$net_exposure <- as.numeric(net_exposure)
      private$max_long_weight <- as.numeric(max_long_weight)
      private$max_short_weight <- as.numeric(max_short_weight)
      private$short_borrow_rate <- as.numeric(short_borrow_rate)
      private$cash_borrow_rate <- as.numeric(cash_borrow_rate)
      private$utility_mode <- match.arg(utility_mode)
      # The no-vine ablation deliberately preserves the observation dimension
      # and actor parameter count.  It removes policy-visible dependence
      # signals (the explicit vine vector and vine-scenario CVaR below),
      # preventing information leakage and capacity from confounding the
      # ablation.  The common risk reward remains unchanged.
      private$vine_observation_mode <- match.arg(vine_observation_mode)
      private$episode_sampling <- match.arg(episode_sampling)
      if (!is.finite(private$gross_leverage) || !is.finite(private$net_exposure) || private$gross_leverage < abs(private$net_exposure)) {
        stop("gross_leverage must be finite and at least abs(net_exposure).")
      }
      if (!is.finite(private$max_long_weight) ||
          !is.finite(private$max_short_weight) ||
          private$max_long_weight <= 0 || private$max_short_weight < 0) {
        stop("Position limits must be finite and non-negative.")
      }
      if (any(!is.finite(c(private$short_borrow_rate, private$cash_borrow_rate))) ||
          any(c(private$short_borrow_rate, private$cash_borrow_rate) < 0)) {
        stop("Borrow rates must be finite, non-negative annual rates.")
      }
      private$obs_history <- list()
      
      # Vine setup
      if (!is.null(vine)) {
        private$vine_static <- vine
        private$vine_current <- vine
      }
      
      if (!is.null(vine_sequence) && length(vine_sequence) > 0) {
        private$vine_sequence <- vine_sequence
        private$vine_seq_len <- length(vine_sequence)
        private$vine_seq_idx <- 1
        private$dynamic <- TRUE
      } else {
        private$dynamic <- FALSE
      }
      
      # Compute dimensions – action_dim is now full number of assets
      d <- length(asset_names)
      private$action_dim <- d

      # Extract AR(1) coefficients from marginals
      private$mu_ar <- numeric(d)
      private$ar1 <- numeric(d)
      for (i in seq_along(asset_names)) {
        name <- asset_names[i]
        model <- marginals[[name]]
        if (identical(model$marginal_type, "component_ewma")) {
          private$mu_ar[i] <- model$mu_ar
          private$ar1[i] <- model$ar1
        } else {
          cfit <- model$fit@fit$coef
          private$mu_ar[i] <- cfit["mu"]
          private$ar1[i] <- if ("ar1" %in% names(cfit)) cfit["ar1"] else 0
        }
      }
      
      # Get vine features dimension
      vine_for_dim <- private$vine_current
      if (is.null(vine_for_dim) && private$vine_seq_len > 0) {
        vine_for_dim <- private$vine_sequence[[1]]
      }
      
      if (!is.null(vine_for_dim)) {
        n_edges <- sum(sapply(vine_for_dim$pair_copulas, length))
      } else {
        n_edges <- 0
      }
      
      # Include the dependence regime and the current holdings.  The old
      # implementation computed vine_state but never exposed it to the policy,
      # and made turnover depend on an unobserved previous_action.
      private$vine_dim <- length(extract_vine_state(vine_for_dim))
      # Monthly state only. Daily AR-GARCH means are intentionally excluded:
      # applying daily AR coefficients to last month's return mixed horizons.
      private$obs_dim <- 1 + d + d + 1 + d + 2 + private$vine_dim
      
      # Store asset names for reference
      private$asset_names <- asset_names
      
      # Pre‑computed returns storage (empty initially)
      private$precomputed_returns <- NULL
      private$precomputed_idx <- 1
    },
    
    # Set pre‑computed returns.
    # - If dynamic is TRUE and length(returns_list) == vine_seq_len, the list
    #   is indexed by vine index; otherwise it is indexed by step (sequential).
    set_precomputed_returns = function(returns_list) {
      if (!is.list(returns_list) || !length(returns_list)) {
        stop("precomputed returns must be a non-empty list of episodes")
      }
      private$precomputed_returns <- returns_list
      private$precomputed_idx <- 0L
    },
    
    # Reset
    reset = function() {
      private$wealth <- private$w0
      private$t <- 0
      private$done <- FALSE
      private$obs_history <- list()
      # Start from a neutral self-financing allocation, not fictitious cash.
      private$previous_action <- rep(private$net_exposure / private$action_dim,
                                     private$action_dim)
      private$last_turnover <- 0
      private$precomputed_vine_states <- NULL
      
      # Initial returns and volatilities
      private$last_returns <- rep(0, length(private$asset_names))
      private$last_vols <- rep(0.01, length(private$asset_names))
      private$last_var <- 0.01^2
      private$vol_history <- NULL   # will be re-initialised on first step
      
      # Reset pre‑computed index to 1 (for sequential mode)
      # Select one complete pre-generated episode.  Resetting a flat index to
      # one here made every episode replay the same first 24 return matrices.
      if (!is.null(private$precomputed_returns)) {
        if (private$episode_sampling == "sequential") {
          private$precomputed_idx <- (private$precomputed_idx %% length(private$precomputed_returns)) + 1L
        } else {
          private$precomputed_idx <- sample.int(length(private$precomputed_returns), 1L)
        }
        private$precomputed_step <- 1L
        episode <- private$precomputed_returns[[private$precomputed_idx]]
        if (is.list(episode) && is.list(episode$vine_states) && length(episode$vine_states)) {
          private$precomputed_vine_states <- episode$vine_states
        }
      }

      # Initialize vine
      if (private$dynamic && private$vine_seq_len > 0) {
        private$vine_seq_idx <- sample(private$vine_seq_len, 1)
        if (!is.null(private$precomputed_returns)) {
          episode <- private$precomputed_returns[[private$precomputed_idx]]
          if (is.list(episode) && !is.null(episode$vine_start)) private$vine_seq_idx <- as.integer(episode$vine_start)
        }
        private$vine_seq_idx <- max(1L, min(private$vine_seq_idx, private$vine_seq_len))
        private$vine_current <- private$vine_sequence[[private$vine_seq_idx]]
      } else if (!is.null(private$vine_static)) {
        private$vine_current <- private$vine_static
      }
      
      private$vine_state <- extract_vine_state(private$vine_current)
      if (!is.null(private$precomputed_vine_states)) {
        private$vine_state <- as.numeric(private$precomputed_vine_states[[1L]])
      }
      
      # Initial CVaR
      private$last_cvar <- 0
      
      # Construct the recurrent history from information strictly preceding
      # the first action. With seq_len > T, repeated padding otherwise means
      # the policy never sees one fully genuine sequence.
      if (!is.null(private$precomputed_returns)) {
        episode <- private$precomputed_returns[[private$precomputed_idx]]
        if (!is.list(episode$burnin_returns) ||
            length(episode$burnin_returns) != private$seq_len) {
          stop(sprintf("Episode must contain exactly %d causal burn-in returns. Regenerate the data bundle.",
                       private$seq_len))
        }
        burnin_states <- episode$burnin_vine_states
        if (!is.null(burnin_states) && length(burnin_states) != private$seq_len) {
          stop("burnin_vine_states must have the same length as burnin_returns.")
        }
        for (i in seq_len(private$seq_len)) {
          burn <- episode$burnin_returns[[i]]
          gross <- if (is.matrix(burn)) burn[1L, ] else as.numeric(burn)
          if (length(gross) != private$action_dim || any(!is.finite(gross)) || any(gross <= 0)) {
            stop("Invalid gross return in episode burn-in.")
          }
          if (!is.null(burnin_states)) private$vine_state <- as.numeric(burnin_states[[i]])
          private$advance_market_state(gross)
          private$obs_history[[i]] <- self$get_obs()
        }
        if (!is.null(private$precomputed_vine_states)) {
          private$vine_state <- as.numeric(private$precomputed_vine_states[[1L]])
          private$obs_history[[private$seq_len]] <- self$get_obs()
        }
      } else {
        obs <- self$get_obs()
        private$obs_history <- replicate(private$seq_len, obs, simplify = FALSE)
      }

      return(self$get_obs())
    },
    
    # Step
    step = function(action) {
      if (private$done) stop("Episode finished. Call reset().")
      
      # Handle different input types for action
      if (is.list(action)) {
        action_vec <- as.numeric(unlist(action))
      } else if (is.array(action) || is.matrix(action)) {
        action_vec <- as.numeric(as.vector(action))
      } else {
        action_vec <- as.numeric(action)
      }
      
      # Ensure correct length
      if (length(action_vec) > private$action_dim) {
        action_vec <- action_vec[1:private$action_dim]
      }
      # Ensure no NA
      if (any(is.na(action_vec))) {
        action_vec[is.na(action_vec)] <- 0
      }
      
      # ---- 1. Get returns (realized + CVaR scenarios) ----
      if (!is.null(private$precomputed_returns)) {
        # Episodes are selected at reset, so each rollout consumes a coherent
        # return path rather than replaying the first path in the bundle.
        episode <- private$precomputed_returns[[private$precomputed_idx]]
        if (is.list(episode) && !is.null(episode$returns)) {
          if (private$precomputed_step > length(episode$returns)) stop("precomputed episode is shorter than T")
          ret_mat <- episode$returns[[private$precomputed_step]]
          private$precomputed_step <- private$precomputed_step + 1L
          R <- ret_mat[1, ]
          R_many <- ret_mat[-1, , drop = FALSE]
        } else {
        # Determine indexing mode
        if (private$dynamic && length(private$precomputed_returns) == private$vine_seq_len) {
          # Vine‑indexed: use current vine_seq_idx
          ret_mat <- private$precomputed_returns[[private$vine_seq_idx]]
        } else {
          # Sequential: use precomputed_idx and increment
          ret_mat <- private$precomputed_returns[[private$precomputed_step]]
          private$precomputed_step <- private$precomputed_step + 1L
        }
        R <- ret_mat[1, ]                           # first row = realized
        R_many <- ret_mat[-1, , drop = FALSE]       # rest = CVaR scenarios
        }
      } else {
        # Simulate on the fly
        n_sim <- private$n_sim_cvar
        prev_ret <- if (private$t == 0) NULL else private$last_returns
        sim_all <- simulate_monthly_gross(private$simulator, private$marginals, private$vine_current,
          n_sim = n_sim + 1L, holding_days = private$holding_days, cores = private$sim_cores,
          previous_monthly_log_returns = prev_ret)
        R <- sim_all[1, ]
        R_many <- sim_all[-1, , drop = FALSE]
      }
      
      # ---- 2. Compute portfolio return ----
      if (length(action_vec) != private$action_dim || any(!is.finite(action_vec))) {
        stop("Action has the wrong dimension or contains non-finite values")
      }
      action_vec <- project_long_short_weights(action_vec, private$net_exposure, private$gross_leverage)
      if (max(action_vec) > private$max_long_weight + 1e-6 ||
          min(action_vec) < -private$max_short_weight - 1e-6) {
        stop("Action violates the configured single-asset long/short limit.")
      }
      # Include the implicit cash account.  For net exposure one this equals
      # sum(weights * gross_returns); it remains correct for other net targets.
      portf_ret <- 1 + sum(action_vec * (R - 1))
      private$last_returns <- log(pmax(R, 1e-12))

      # Trading and financing costs are part of both realised wealth and the
      # forward loss distribution used by CVaR.
      turnover <- sum(abs(action_vec - private$previous_action))
      transaction_cost <- private$kappa * turnover
      short_notional <- sum(pmax(-action_vec, 0))
      cash_borrow_notional <- pmax(sum(action_vec) - 1, 0)
      financing_cost <- (private$short_borrow_rate * short_notional +
                         private$cash_borrow_rate * cash_borrow_notional) / 12
      cost_multiplier <- exp(-transaction_cost - financing_cost)
      
      # ---- 3. Compute CVaR ----
      portf_ret_many <- (1 + R_many %*% action_vec - sum(action_vec)) * cost_multiplier
      alpha <- 0.95
      losses <- 1 - portf_ret_many
      sorted_losses <- sort(losses, decreasing = TRUE)
      n_sim <- nrow(R_many)
      cvar_idx <- max(1L, ceiling((1 - alpha) * n_sim))
      cvar <- mean(sorted_losses[1:cvar_idx])
      private$last_cvar <- cvar
      
      # ---- 4. Compute turnover ----
      private$previous_action <- action_vec
      private$last_turnover <- turnover
      
      # ---- 5. Compute reward (dense log‑return) ----
      net_portf_ret <- pmax(portf_ret * cost_multiplier, 1e-12)
      wealth_before <- private$wealth
      private$wealth <- private$wealth * net_portf_ret
      # Terminal-wealth CRRA is a multi-period objective because wealth carries
      # all prior portfolio decisions.  Its utility increments telescope to
      # U(W_T/W_0) - U(1), while retaining dense learning signal.
      terminal_utility <- crra_utility(private$wealth / private$w0, private$gamma)
      previous_terminal_utility <- crra_utility(wealth_before / private$w0, private$gamma)
      period_utility <- if (private$gamma == 1) log(net_portf_ret) else (net_portf_ret^(1 - private$gamma) - 1) / (1 - private$gamma)
      utility_increment <- if (private$utility_mode == "terminal_wealth_crra") terminal_utility - previous_terminal_utility else period_utility
      reward <- utility_increment - private$lambda * cvar
      
      # ---- 6. Advance time ----
      private$t <- private$t + 1
      if (private$t >= private$T) private$done <- TRUE
      
      # ---- 7. Update vine (if dynamic) ----
      if (!is.null(private$precomputed_vine_states)) {
        next_state_index <- min(private$precomputed_step, length(private$precomputed_vine_states))
        private$vine_state <- as.numeric(private$precomputed_vine_states[[next_state_index]])
      } else if (private$dynamic && private$vine_seq_len > 0) {
        private$vine_seq_idx <- private$vine_seq_idx + 1
        if (private$vine_seq_idx > private$vine_seq_len) {
          private$vine_seq_idx <- 1
        }
        private$vine_current <- private$vine_sequence[[private$vine_seq_idx]]
        private$vine_state <- extract_vine_state(private$vine_current)
      }
      
      # ---- 8. Update volatilities (EWMA) ----
      private$advance_market_state(R)
      private$last_vols <- sqrt(apply(private$vol_history, 2, mean)) * sqrt(12)   # monthly → annual
      
      # ---- 9. Get observation ----
      obs <- self$get_obs()
      
      # ---- 10. Update history for LSTM ----
      private$obs_history <- c(private$obs_history[-1], list(obs))
      
      return(list(
        observation = obs,
        reward = reward,
        done = private$done,
        info = list(
          wealth = private$wealth,
          portf_ret = portf_ret,
          net_portf_ret = net_portf_ret,
          cvar = cvar,
          turnover = turnover,
          transaction_cost = transaction_cost,
          financing_cost = financing_cost,
          short_notional = short_notional,
          utility = terminal_utility,
          utility_increment = utility_increment,
          gross_exposure = sum(abs(action_vec)),
          net_exposure = sum(action_vec),
          weights = action_vec
        )
      ))
    },
    
    # Get Observation (single vector)
    get_obs = function() {
      no_vine_observation <- identical(private$vine_observation_mode, "zero")
      vine_observation <- if (no_vine_observation) {
        numeric(private$vine_dim)
      } else {
        private$vine_state
      }
      # last_cvar is computed from vine-simulated joint scenarios.  Leaving it
      # visible would let the nominal no-vine actor recover a dependence-regime
      # signal through an indirect channel.
      cvar_observation <- if (no_vine_observation) 0 else private$last_cvar * 100
      obs <- c(
        private$wealth / private$w0,
        private$last_returns * 100,
        private$last_vols * 100,
        cvar_observation,
        private$previous_action,
        sum(abs(private$previous_action)),
        sum(private$previous_action),
        vine_observation
      )
      obs[!is.finite(obs)] <- 0
      return(as.numeric(obs))
    },
    
    # Get observation history (for LSTM input)
    get_history = function() {
      if (length(private$obs_history) == 0) {
        return(matrix(0, nrow = private$seq_len, ncol = private$obs_dim))
      }
      obs_list <- private$obs_history
      hist_matrix <- do.call(rbind, obs_list)
      return(hist_matrix)
    },
    
    # Utility methods
    render = function() {
      cat(sprintf("t=%d/%d | Wealth: %.0f | CVaR: %.4f | Turnover: %.4f\n",
                  private$t, private$T, private$wealth, 
                  private$last_cvar, private$last_turnover))
    },
    
    get_action_dim = function() private$action_dim,
    get_obs_dim   = function() private$obs_dim,
    get_seq_len   = function() private$seq_len
  ),
  
  private = list(
    # Core components
    simulator = NULL,
    marginals = NULL,
    asset_names = NULL,
    ref_col = NULL,
    
    # Hyperparameters
    gamma = NULL,
    lambda = NULL,
    kappa = NULL,
    T = NULL,
    w0 = NULL,
    n_sim_cvar = NULL,
    sim_cores = NULL,
    seq_len = NULL,
    holding_days = NULL,
    gross_leverage = NULL,
    net_exposure = NULL,
    max_long_weight = NULL,
    max_short_weight = NULL,
    short_borrow_rate = NULL,
    cash_borrow_rate = NULL,
    utility_mode = NULL,
    vine_observation_mode = NULL,
    episode_sampling = NULL,
    
    # Vine
    vine_static = NULL,
    vine_current = NULL,
    vine_sequence = NULL,
    vine_seq_len = 0,
    vine_seq_idx = 1,
    dynamic = FALSE,
    vine_state = NULL,
    
    # Pre‑computed returns
    precomputed_returns = NULL,
    precomputed_idx = NULL,
    precomputed_step = NULL,
    precomputed_vine_states = NULL,
    
    # State
    wealth = NULL,
    t = NULL,
    done = NULL,
    last_returns = NULL,
    last_vols = NULL,
    last_var = NULL,
    last_cvar = NULL,
    last_turnover = NULL,
    previous_action = NULL,
    obs_history = list(),
    vol_history = NULL,
    mu_ar = NULL,
    ar1 = NULL,

    advance_market_state = function(gross_returns) {
      private$last_returns <- log(pmax(as.numeric(gross_returns), 1e-12))
      if (is.null(private$vol_history)) {
        initial_variance <- pmax(private$last_returns^2, 1e-6)
        private$vol_history <- matrix(rep(initial_variance, each = 20L),
                                      nrow = 20L,
                                      ncol = length(private$asset_names))
      }
      new_var <- 0.97 * private$last_var + 0.03 * private$last_returns^2
      private$vol_history <- rbind(private$vol_history[-1, , drop = FALSE], new_var)
      private$last_var <- new_var
      private$last_vols <- sqrt(colMeans(private$vol_history)) * sqrt(12)
      invisible(NULL)
    },
    
    # Dimensions
    obs_dim = NULL,
    action_dim = NULL,
    vine_dim = NULL
  )
)

if (identical(Sys.getenv("VERBOSE_R_STARTUP", "0"), "1"))
  cat("RLEnvironment (full framework) loaded.\n")
