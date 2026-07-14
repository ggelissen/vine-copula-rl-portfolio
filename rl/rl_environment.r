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
  pc_list <- vine$pair_copulas[[1]]
  n_edges <- length(pc_list)
  
  state <- numeric(n_edges * 3)  # tau, lower tail dep, upper tail dep
  idx <- 1
  for (i in seq_len(n_edges)) {
    pc <- pc_list[[i]]
    fam <- pc$family
    params <- pc$parameters
    rot <- pc$rotation
    
    # Kendall's tau using rvinecopulib's function (handles rotation)
    tau <- tryCatch(
      as.numeric(par_to_ktau(fam, params, rotation = rot)),
      error = function(e) NA
    )
    
    # Tail dependence (lower, upper)
    lt <- 0; ut <- 0
    if (fam == "t") {
      rho <- params[1]; nu <- params[2]
      lt <- 2 * pt(-sqrt((nu+1)*(1-rho)/(1+rho)), df = nu+1)
      ut <- lt
    } else if (fam == "gumbel") {
      theta <- params[1]
      if (rot == 0) {
        ut <- 2 - 2^(1/theta)
      } else if (rot == 180) {
        lt <- 2 - 2^(1/theta)
      } else if (rot == 90) {
        ut <- 2 - 2^(1/theta) 
      } else if (rot == 270) {
        lt <- 2 - 2^(1/theta)
      }
    } else if (fam == "clayton") {
      theta <- params[1]
      if (rot == 0) {
        lt <- 2^(-1/theta)
      } else if (rot == 180) {
        ut <- 2^(-1/theta) 
      } else if (rot == 90) {
        lt <- 2^(-1/theta)
      } else if (rot == 270) {
        ut <- 2^(-1/theta)
      }
    } else if (fam == "joe") {
      theta <- params[1]
      if (rot == 0) {
        ut <- 2 - 2^(1/theta)
      } else if (rot == 180) {
        lt <- 2 - 2^(1/theta)
      } else if (rot == 90) {
        ut <- 2 - 2^(1/theta)
      } else if (rot == 270) {
        lt <- 2 - 2^(1/theta)
      }
    }
    # Gaussian and Frank have no tail dependence (lt=0, ut=0)
    
    state[idx]   <- tau
    state[idx+1] <- lt
    state[idx+2] <- ut
    idx <- idx + 3
  }
  state
}


build_vine_sequence <- function(returns_xts, U, rebal_dates, L = 500) {
  vine_seq <- vector("list", length(rebal_dates))
  T_max <- nrow(returns_xts)
  
  for (t in seq_along(rebal_dates)) {
    window_end   <- which(index(returns_xts) == rebal_dates[t])
    
    # Skip if not enough history or window extends past data
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
  
  # Remove empty slots
  vine_seq <- vine_seq[!sapply(vine_seq, is.null)]
  vine_seq
}




# RL Environment class
RLEnvironment <- R6Class(
  "RLEnvironment",
  public = list(
    initialize = function(marginals, asset_names, vine = NULL,
                          ref_col = 7, gamma = 2, T = 12, w0 = 100000,
                          n_sim = 1,
                          vine_sequence = NULL, 
                          dynamic = FALSE) {
      private$simulator <- build_simulator(marginals, asset_names, ref_col)
      private$ref_col <- ref_col
      private$gamma <- gamma
      private$T <- T
      private$w0 <- w0
      private$n_sim <- n_sim
      private$dynamic <- dynamic
      
      # Static vine (fallback if no sequence)
      if (!is.null(vine)) {
        private$vine_static <- vine
        private$vine_current <- vine
      }
      
      # Dynamic vine sequence
      private$vine_sequence <- vine_sequence
      private$vine_seq_len <- if (!is.null(vine_sequence)) length(vine_sequence) else 0
      private$vine_seq_idx <- 1
      
      # Pre‑compute state dimensions
      d <- length(asset_names)
      private$action_dim <- d - 1   # risky assets only
      
      # Observation dim: wealth + d returns + d volatilities + vine_state (n_edges*3)
      # We'll compute n_edges dynamically from the first vine
      if (!is.null(vine)) {
        n_edges <- length(vine$pair_copulas[[1]])
      } else if (!is.null(vine_sequence) && length(vine_sequence) > 0) {
        n_edges <- length(vine_sequence[[1]]$pair_copulas[[1]])
      } else {
        stop("Either vine or vine_sequence must be provided.")
      }
      private$obs_dim <- 1 + d + d + n_edges * 3
    },
    
    reset = function() {
      private$wealth <- private$w0
      private$t <- 0
      private$last_returns <- rep(0, length(asset_names))
      
      # Initial volatilities (unconditional)
      vols <- numeric(length(asset_names))
      for (i in seq_along(asset_names)) {
        name <- asset_names[i]
        vols[i] <- private$simulator$marginals[[name]]$sigma_uncond
      }
      private$last_vols <- vols
      
      # Choose starting vine
      if (private$dynamic && private$vine_seq_len > 0) {
        # Random start point in the sequence
        private$vine_seq_idx <- sample(private$vine_seq_len, 1)
        private$vine_current <- private$vine_sequence[[private$vine_seq_idx]]
      } else if (!is.null(private$vine_static)) {
        private$vine_current <- private$vine_static
      }
      
      private$vine_state <- extract_vine_state(private$vine_current)
      private$done <- FALSE
      self$get_obs()
    },
    
    step = function(action) {
      if (private$done) stop("Episode finished. Call reset().")
      
      # Simulate one‑step returns from current vine
      sim <- private$simulator$simulate_returns(private$vine_current, n_sim = 1)
      R <- sim$gross[1, ]
      R_ref <- R[private$ref_col]
      R_risk <- R[-private$ref_col]
      
      portf_ret <- R_ref + sum((R_risk - R_ref) * action)
      private$wealth <- private$wealth * portf_ret
      private$last_returns <- log(R)
      
      # Advance time
      private$t <- private$t + 1
      
      # Update vine if dynamic
      if (private$dynamic && private$vine_seq_len > 0) {
        # Move to next vine in sequence (circular)
        private$vine_seq_idx <- private$vine_seq_idx + 1
        if (private$vine_seq_idx > private$vine_seq_len) {
          private$vine_seq_idx <- 1 
        }
        private$vine_current <- private$vine_sequence[[private$vine_seq_idx]]
        private$vine_state <- extract_vine_state(private$vine_current)
      }
      
      if (private$t >= private$T) private$done <- TRUE
      
      #reward <- crra_utility(private$wealth, private$gamma)
      reward = log(portf_ret) * 100.0 
      obs <- self$get_obs()
      list(observation = obs, reward = reward, done = private$done, info = list(wealth = private$wealth))
    },
    
    get_obs = function() {
    c(private$wealth / 100000,
      private$last_returns * 100,
      private$last_vols * 100,
      private$vine_state)
    },
    
    render = function() {
      cat(sprintf("t=%d/%d  wealth=%.2f\n", private$t, private$T, private$wealth))
    },
    
    get_action_dim = function() private$action_dim,
    get_obs_dim   = function() private$obs_dim
  ),
  
  private = list(
    simulator = NULL,
    vine_static = NULL,
    vine_current = NULL,
    ref_col = NULL,
    gamma = NULL,
    T = NULL,
    w0 = NULL,
    n_sim = NULL,
    dynamic = FALSE,
    vine_sequence = NULL,
    vine_seq_len = 0,
    vine_seq_idx = 1,
    wealth = NULL,
    t = NULL,
    last_returns = NULL,
    last_vols = NULL,
    done = NULL,
    vine_state = NULL,
    obs_dim = NULL,
    action_dim = NULL
  )
)


# ================================================================================================

# Set up rebalancing dates (full range)
L <- 500
all_dates <- index(returns)
rebal_dates <- endpoints(returns[L:nrow(returns)], on = "months")
rebal_dates <- index(returns)[rebal_dates + L - 1]

# Build vine sequence (only a subset if needed, e.g., 36 months)
vine_seq <- build_vine_sequence(returns, U, rebal_dates, L = 500)

env_dyn <- RLEnvironment$new(
  marginals, asset_names,
  vine = NULL,               
  vine_sequence = vine_seq,
  dynamic = TRUE,
  ref_col = 7, gamma = 2, T = 12, w0 = 100000
)