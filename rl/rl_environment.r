# ============================================================================
# rl_environment.R
# RL environment for vine‑copula portfolio selection
# ============================================================================

library(R6)
library(rvinecopulib)

source("helper/load_data.r")
source("benchmark_models/expected_utility_single.r")   # VineReturnSimulator, build_simulator, crra_utility
load("data/marginal_results.RData")   # marginals, asset_names
load("data/vine_fit.RData")           # vine_fit (static D‑vine)

# Helper: extract first‑tree parameters and tail dependence
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
      rot <- pc$rotation %||% 0
      
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



build_vine_sequence <- function(returns_xts, U, rebal_dates, L = 500) {
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


# CRRA Utility Function
crra_utility <- function(wealth, gamma) {
  if (gamma == 1) {
    return(log(wealth))
  } else {
    return((wealth^(1-gamma) - 1) / (1 - gamma))
  }
}



# RL Environment class
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
                          seq_len = 30) {      # LSTM lookback window
      
      private$simulator <- build_simulator(marginals, asset_names, ref_col)
      private$ref_col <- ref_col
      private$gamma <- gamma
      private$lambda <- lambda
      private$kappa <- kappa
      private$T <- T
      private$w0 <- w0
      private$n_sim_cvar <- n_sim_cvar
      private$seq_len <- seq_len
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
      
      # Compute dimensions
      d <- length(asset_names)
      private$action_dim <- d - 1
      
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
      
      # obs_dim = wealth(1) + returns(d) + volatilities(d) + vine_features(n_edges*3) + CVaR(1)
      private$obs_dim <- 1 + d + d + n_edges * 3 + 1
      
      # Store asset names for reference
      private$asset_names <- asset_names
    },
    
    # Reset
    reset = function() {
      private$wealth <- private$w0
      private$t <- 0
      private$done <- FALSE
      private$obs_history <- list()
      private$previous_action <- rep(0, private$action_dim)
      
      # Initial returns and volatilities
      private$last_returns <- rep(0, length(private$asset_names))
      private$last_vols <- rep(0.01, length(private$asset_names))
      
      # Initialize vine
      if (private$dynamic && private$vine_seq_len > 0) {
        private$vine_seq_idx <- sample(private$vine_seq_len, 1)
        private$vine_current <- private$vine_sequence[[private$vine_seq_idx]]
      } else if (!is.null(private$vine_static)) {
        private$vine_current <- private$vine_static
      }
      
      private$vine_state <- extract_vine_state(private$vine_current)
      
      # Initial CVaR
      private$last_cvar <- 0
      
      # Initial observation (prepend zeros for history)
      obs <- self$get_obs()
      for (i in 1:private$seq_len) {
        private$obs_history[[i]] <- obs
      }
      
      return(obs)
    },
    
    # Step
    step = function(action) {
      if (private$done) stop("Episode finished. Call reset().")
      
      # Handle different input types for action
      if (is.list(action)) {
        # Unlist and convert to numeric
        action_vec <- as.numeric(unlist(action))
      } else if (is.array(action) || is.matrix(action)) {
        # Convert array/matrix to vector
        action_vec <- as.numeric(as.vector(action))
      } else {
        action_vec <- as.numeric(action)
      }

      # Ensure we have the right length
      if (length(action_vec) > private$action_dim) {
        action_vec <- action_vec[1:private$action_dim]
      }
      
      # Ensure no NA values
      if (any(is.na(action_vec))) {
        action_vec[is.na(action_vec)] <- 0
      }
      
      # 1. Simulate one-step returns from current vine
      sim <- private$simulator$simulate_returns(private$vine_current, n_sim = 1)
      R <- sim$gross[1, ]
      R_ref <- R[private$ref_col]
      R_risk <- R[-private$ref_col]
      
      # 2. Compute portfolio return
      portf_ret <- R_ref + sum((R_risk - R_ref) * action_vec)
      private$wealth <- private$wealth * portf_ret
      private$last_returns <- log(R)
      
      # 3. Compute CVaR from vine
      # Simulate many returns to compute CVaR
      n_sim <- private$n_sim_cvar
      sim_many <- private$simulator$simulate_returns(private$vine_current, n_sim = n_sim)
      R_many <- sim_many$gross
      
      # Compute portfolio returns for all simulations
      weights_full <- c(action_vec, 1 - sum(action_vec)) 
      portf_ret_many <- R_many %*% weights_full
      
      # CVaR at 95%
      alpha <- 0.95
      sorted_ret <- sort(portf_ret_many)
      cvar_idx <- floor((1 - alpha) * n_sim)
      cvar <- -mean(sorted_ret[1:cvar_idx])
      private$last_cvar <- cvar
      
      # 4. Compute turnover
      turnover <- sum(abs(action_vec - private$previous_action))
      private$previous_action <- action_vec
      
      # 5. Compute reward
      # Utility component
      utility <- crra_utility(portf_ret, private$gamma)
      reward <- utility - private$lambda * cvar - private$kappa * turnover
      
      # 6. Advance time
      private$t <- private$t + 1
      if (private$t >= private$T) private$done <- TRUE
      
      # 7. Update vine
      if (private$dynamic && private$vine_seq_len > 0) {
        private$vine_seq_idx <- private$vine_seq_idx + 1
        if (private$vine_seq_idx > private$vine_seq_len) {
          private$vine_seq_idx <- 1
        }
        private$vine_current <- private$vine_sequence[[private$vine_seq_idx]]
        private$vine_state <- extract_vine_state(private$vine_current)
      }
      
      # 8. Update volatilities (rolling estimate)
      # Simple approximation: EWMA
      if (is.null(private$vol_history)) {
        private$vol_history <- matrix(0.01, nrow = 20, ncol = length(private$asset_names))
      }
      new_vol <- 0.94 * private$last_vols + 0.06 * (private$last_returns^2)
      private$vol_history <- rbind(private$vol_history[-1, ], new_vol)
      private$last_vols <- sqrt(apply(private$vol_history, 2, mean)) * sqrt(252)
      
      # 9. Get observation
      obs <- self$get_obs()
      
      # 10. Update history for LSTM
      private$obs_history <- c(private$obs_history[-1], list(obs))
      
      return(list(
        observation = obs,
        reward = reward,
        done = private$done,
        info = list(
          wealth = private$wealth,
          portf_ret = portf_ret,
          cvar = cvar,
          turnover = turnover,
          utility = utility
        )
      ))
    },
    
    # Get Observation (with LSTM window)
    get_obs = function() {
      # Flatten the history into a single vector
      # Each history element is: wealth + returns + vols + vine_state + cvar
      # We return the most recent observation as the current state
      obs <- c(
        private$wealth / private$w0,
        private$last_returns * 100,
        private$last_vols * 100,
        private$vine_state,
        private$last_cvar
      )
      return(obs)
    },
    
    # Get observation history (for LSTM input)
    get_history = function() {
      # Returns the full history as a matrix: (seq_len, obs_dim)
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
                  private$last_cvar, 
                  sum(abs(private$previous_action - private$previous_action))))
    },
    
    get_action_dim = function() private$action_dim,
    get_obs_dim   = function() private$obs_dim,
    get_seq_len   = function() private$seq_len
  ),
  
  private = list(
    # Core components
    simulator = NULL,
    asset_names = NULL,
    ref_col = NULL,
    
    # Hyperparameters
    gamma = NULL,
    lambda = NULL,
    kappa = NULL,
    T = NULL,
    w0 = NULL,
    n_sim_cvar = NULL,
    seq_len = NULL,
    
    # Vine
    vine_static = NULL,
    vine_current = NULL,
    vine_sequence = NULL,
    vine_seq_len = 0,
    vine_seq_idx = 1,
    dynamic = FALSE,
    vine_state = NULL,
    
    # State
    wealth = NULL,
    t = NULL,
    done = NULL,
    last_returns = NULL,
    last_vols = NULL,
    last_cvar = NULL,
    previous_action = NULL,
    obs_history = list(),
    vol_history = NULL,
    
    # Dimensions
    obs_dim = NULL,
    action_dim = NULL
  )
)

cat("RLEnvironment (full framework) loaded.\n")


# ================================================================================================

# # Set up rebalancing dates (full range)
# L <- 500
# all_dates <- index(returns)
# rebal_dates <- endpoints(returns[L:nrow(returns)], on = "months")
# rebal_dates <- index(returns)[rebal_dates + L - 1]

# # Build vine sequence (only a subset if needed, e.g., 36 months)
# vine_seq <- build_vine_sequence(returns, U, rebal_dates, L = 500)

# env_dyn <- RLEnvironment$new(
#   marginals, asset_names,
#   vine = NULL,               
#   vine_sequence = vine_seq,
#   dynamic = TRUE,
#   ref_col = 7, gamma = 2, T = 12, w0 = 100000
# )