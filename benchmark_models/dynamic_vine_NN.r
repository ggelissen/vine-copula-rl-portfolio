# ==============================================================================
# dynamic_vine_NN.r
# Fully dynamic all-tree t D-vine with neural correlation forecasts.
# Degrees of freedom are fixed per edge for identifiability; tail dependence
# remains time-varying through rho. Dynamic family switching is not claimed.
# ==============================================================================

library(rvinecopulib)
library(torch)
library(copula)

RUN_TESTS <- FALSE

prepare_nn_data <- function(u, z, sigma, nu_fix, lookback = 1,
                            rho_base = NULL) {
  T <- nrow(u)

  # Build the network features: lagged u, z, sigma, and nu
  rho_stat <- if (is.null(rho_base)) cor(qt(u[,1], df = nu_fix), qt(u[,2], df = nu_fix)) else rho_base
  theta_lag <- rep(rho_stat, T)
  score_lag <- rep(0, T)

  # Pre-compute the score for all t
  x <- qt(u, df = nu_fix)
  for (t in 2:T) {
    rho_prev <- theta_lag[t-1]
    score_lag[t] <- (nu_fix + 2) / nu_fix * (rho_prev * (x[t-1,1]^2 + x[t-1,2]^2)
                     - (1 + rho_prev^2) * x[t-1,1] * x[t-1,2]) / ((1 - rho_prev^2)
                    * (nu_fix * (1 - rho_prev^2) + (x[t-1,1]^2 + x[t-1,2]^2 - 2 * 
                    rho_prev * x[t-1,1] * x[t-1,2])))
  }

  # Build matrix of features
  features <- cbind(u_lag1 = c(NA, u[-T, 1]), u_lag2 = c(NA, u[-T, 2]),
                    z_lag1 = c(NA, z[-T, 1]), z_lag2 = c(NA, z[-T, 2]),
                    log_sigma_lag1 = c(NA, log(pmax(sigma[-T, 1], 1e-8))),
                    log_sigma_lag2 = c(NA, log(pmax(sigma[-T, 2], 1e-8))),
                    theta_lag = theta_lag, score_lag = score_lag
  )

  features <- features[-1, , drop = FALSE] # Remove first row with NA
  y <- u[-1, , drop = FALSE]               # same for the uniforms

  list(features = torch_tensor(features, dtype = torch_float()),
       y = torch_tensor(y, dtype = torch_float()))
}

prepare_nn_feature_next <- function(u, z, sigma, nu_fix, rho_base) {
  u <- as.matrix(u); z <- as.matrix(z); sigma <- as.matrix(sigma)
  last <- nrow(u)
  x <- qt(u[last, ], df = nu_fix)
  rho <- pmax(pmin(rho_base, 0.995), -0.995)
  score <- (nu_fix + 2) / nu_fix *
    (rho * sum(x^2) - (1 + rho^2) * prod(x)) /
    ((1 - rho^2) * (nu_fix * (1 - rho^2) + sum(x^2) - 2 * rho * prod(x)))
  feature <- matrix(c(u[last, 1], u[last, 2], z[last, 1], z[last, 2],
                      log(pmax(sigma[last, 1], 1e-8)),
                      log(pmax(sigma[last, 2], 1e-8)), rho, score), nrow = 1L)
  torch_tensor(feature, dtype = torch_float())
}


# Define the neural network model 
nn_model <- nn_module(
  "nn_model", 
  # Inputs: 8, Hidden layers: 16, 8, Output: 1 (version 1)
  initialize = function(input_dim = 8, hidden1 = 16, hidden2 = 8,
                        rho_base = 0, dynamic_scale = 0.25) {
    self$fc1 <- nn_linear(input_dim, hidden1)
    self$fc2 <- nn_linear(hidden1, hidden2)
    self$fc3 <- nn_linear(hidden2, 1)
    self$rho_base <- as.numeric(pmax(pmin(rho_base, 0.99), -0.99))
    self$dynamic_scale <- as.numeric(dynamic_scale)
    with_no_grad({
      self$fc3$weight$zero_()
      self$fc3$bias$zero_()
    })
  },
  forward = function(x) {
    x <- torch_tanh(self$fc1(x))
    x <- torch_tanh(self$fc2(x))
    rho <- torch_tanh(atanh(self$rho_base) + self$dynamic_scale * self$fc3(x))
    rho
  }
)


# Train the neural network model
train_nn_model <- function(u, nu_fix, z, sigma, epochs = 500, lr = 1e-3,
                           patience = 50, validation_fraction = 0.2,
                           rho_base = 0) {
  # Prepare the data
  data <- prepare_nn_data(u, z, sigma, nu_fix, rho_base = rho_base)

  model <- nn_model(rho_base = rho_base)
  optim <- optim_adam(model$parameters, lr = lr)
  
  n <- data$features$size(1)
  train_n <- floor(n * (1 - validation_fraction))
  if (train_n < 20L || n - train_n < 5L) stop("Too few observations for chronological NN validation.")
  train_features <- data$features[1:train_n, ]
  train_y <- data$y[1:train_n, ]
  val_features <- data$features[(train_n + 1L):n, ]
  val_y <- data$y[(train_n + 1L):n, ]

  copula_loss <- function(rho, y) {
    x <- qt(as.matrix(y), df = nu_fix)
    x1 <- x[, 1]; x2 <- x[, 2]
    denom <- 1 - rho^2
    quad <- x1^2 + x2^2 - 2 * rho * x1 * x2
    -mean(lgamma((nu_fix + 2) / 2) + lgamma(nu_fix / 2) -
      0.5 * log(denom) - (nu_fix + 2) / 2 *
      log(1 + quad / (nu_fix * denom)))
  }

  best_loss <- Inf; wait <- 0; best_state <- NULL

  for (epoch in 1:epochs) {
    # Set model to training mode
    model$train()
    optim$zero_grad()

    rho <- model(train_features)$squeeze()
    loss <- copula_loss(rho, train_y)

    # Backpropagation and optimization
    loss$backward()
    optim$step()

    if (epoch %% 10 == 0) {
      #cat(sprintf("Epoch %d: Loss = %.6f\n", epoch, loss$item()))
    }

    model$eval()
    val_loss <- with_no_grad({
      copula_loss(model(val_features)$squeeze(), val_y)$item()
    })
    # Chronological validation, never the optimisation loss, controls stopping.
    if (is.finite(val_loss) && val_loss < best_loss - 1e-7) {
      best_loss <- val_loss
      wait <- 0
      best_state <- lapply(model$state_dict(), function(x) x$clone())
    } else {
      wait <- wait + 1
      if (wait >= patience) {
        #cat("Early stopping...\n")
        break
      }
    }
  }
  if (is.null(best_state)) stop("NN training failed to produce a finite validation likelihood.")
  model$load_state_dict(best_state)
  model$eval()
  model
}

# For seven assets the exact search is only 7! = 5040 paths. The order is
# selected on training pseudo-observations by adjacent absolute Kendall tau;
# no alphabetical/arbitrary asset order is imposed.
select_dvine_order <- function(U) {
  U <- as.matrix(U); d <- ncol(U)
  tau <- abs(cor(U, method = "kendall", use = "pairwise.complete.obs"))
  best_score <- -Inf; best <- NULL
  visit <- function(prefix, remaining) {
    if (!length(remaining)) {
      score <- sum(tau[cbind(head(prefix, -1L), tail(prefix, -1L))])
      if (score > best_score) { best_score <<- score; best <<- prefix }
      return(invisible(NULL))
    }
    for (candidate in remaining) visit(c(prefix, candidate), setdiff(remaining, candidate))
  }
  visit(integer(), seq_len(d))
  as.integer(best)
}

validate_truncation <- function(U, order, train_fraction = 0.8) {
  n_train <- floor(nrow(U) * train_fraction)
  if (n_train < 30L || nrow(U) - n_train < 10L) stop("Too few observations for vine truncation validation.")
  train <- U[seq_len(n_train), , drop = FALSE]
  validation <- U[(n_train + 1L):nrow(U), , drop = FALSE]
  structure <- dvine_structure(order)
  truncated <- vinecop(train, structure = structure, family_set = "t",
                       trunc_lvl = 1L, selcrit = "bic")
  full <- vinecop(train, structure = structure, family_set = "t", selcrit = "bic")
  delta <- log(pmax(dvinecop(validation, full), 1e-300)) -
           log(pmax(dvinecop(validation, truncated), 1e-300))
  se <- sd(delta) / sqrt(length(delta))
  list(mean_full_minus_truncated = mean(delta), standard_error = se,
       z = if (is.finite(se) && se > 0) mean(delta) / se else NA_real_,
       validation_observations = length(delta))
}


# Predict the copula parameter (rho) using the trained neural network model
predict_rho_nn <- function(model, features) {
  model$eval()
  with_no_grad({
    rho <- model(features)$squeeze()
    as.numeric(rho)
  })
}


# Extract z and sigma from marginals
extract_marginal_states <- function(marginals, U_matrix, returns_xts) {
  d <- length(marginals)
  n <- nrow(U_matrix)
  
  z_mat <- matrix(0, n, d)
  sigma_mat <- matrix(0, n, d)
  
  for (i in seq_len(d)) {
    name <- names(marginals)[i]
    model <- marginals[[name]]
    
    if (identical(model$marginal_type, "component_ewma")) {
      z_mat[, i] <- as.numeric(model$z)
      sigma_mat[, i] <- as.numeric(model$sigma)
    } else {
      z_mat[, i] <- as.numeric(model$fit@fit$residuals / model$fit@fit$sigma)
      sigma_mat[, i] <- as.numeric(model$fit@fit$sigma)
    }
  }
  
  colnames(z_mat) <- names(marginals)
  colnames(sigma_mat) <- names(marginals)
  
  list(z = z_mat, sigma = sigma_mat)
}



# Train NN for all t‑copula edges
train_tree1_edges_legacy <- function(U, vine_fit, asset_names, z, sigma, epochs = 200, lr = 1e-3, patience = 20) {
  # Extract first‑tree edges
  tree1_pcs <- vine_fit$pair_copulas[[1]]
  order <- vine_fit$structure$order
  d <- length(order)
  tree1_edges <- cbind(order[1:(d-1)], order[2:d])
  
  nn_models <- vector("list", nrow(tree1_edges))
  
  for (k in seq_len(nrow(tree1_edges))) {
    vars <- tree1_edges[k, ]
    fam  <- tree1_pcs[[k]]$family
    
    #cat(sprintf("\n=== Edge %d-%d (%s) ===\n", vars[1], vars[2], fam))
    
    if (fam %in% c("student", "t")) {
      nu_fix <- as.numeric(tree1_pcs[[k]]$parameters[2])
      u_edge <- U[, vars]
      
      model <- train_nn_model(u_edge, nu_fix, z[, vars, drop = FALSE], sigma[, vars, drop = FALSE],
        epochs = epochs, lr = lr, patience = patience)
      nn_models[[k]] <- list(model = model, family = "t", nu = nu_fix, vars = vars)
      
      #cat(sprintf("Edge %d-%d trained.\n", vars[1], vars[2]))
    } else {
      # Non‑t edges: keep static
      nn_models[[k]] <- list(
        model = NULL,
        family = fam,
        params = tree1_pcs[[k]]$parameters,
        rotation = tree1_pcs[[k]]$rotation,
        vars = vars
      )
      #cat(sprintf("Edge %d-%d (%s): kept static.\n", vars[1], vars[2], fam))
    }
  }
  
  nn_models
}


# Predict all first‑tree parameters from trained NNs
predict_tree1_params_legacy <- function(nn_models, U_window, z_window = NULL, sigma_window = NULL) {
  n_edges <- length(nn_models)
  pc_list <- vector("list", n_edges)
  
  for (k in seq_len(n_edges)) {
    nm <- nn_models[[k]]
    if (nm$family == "t" && !is.null(nm$model)) {
      # Extract the uniforms for this edge
      u_edge <- U_window[, nm$vars, drop = FALSE]
      
      # Use real z and sigma if provided, otherwise dummy
      if (!is.null(z_window) && !is.null(sigma_window)) {
        z_edge     <- z_window[, nm$vars, drop = FALSE]
        sigma_edge <- sigma_window[, nm$vars, drop = FALSE]
      } else {
        z_edge     <- matrix(0, nrow(u_edge), 2)
        sigma_edge <- matrix(1, nrow(u_edge), 2)
      }
      
      # Prepare features and predict current rho
      features <- prepare_nn_data(u_edge, z_edge, sigma_edge, nm$nu)
      rho_now <- tail(predict_rho_nn(nm$model, features$features), 1)
      rho_now <- pmax(pmin(rho_now, 0.999), -0.999)
      
      pc_list[[k]] <- bicop_dist(
        family     = "t",
        rotation   = 0,
        parameters = c(rho_now, nm$nu)
      )
    } else {
      pc_list[[k]] <- bicop_dist(
        family     = nm$family,
        rotation   = ifelse(is.null(nm$rotation), 0, nm$rotation),
        parameters = nm$params
      )
    }
  }
  
  pc_list
}


# Build vine copula from NN‑predicted first tree and static higher trees
build_tree1_nn_vine_legacy <- function(nn_models, full_vine, U_window, z_window = NULL, sigma_window = NULL) {
  # Predict first‑tree pair copulas
  pc_first <- predict_tree1_params_legacy(nn_models, U_window, z_window, sigma_window)
  
  # Get higher‑tree pair copulas from the full‑sample static vine
  n_trees <- length(full_vine$pair_copulas)
  all_pcs <- vector("list", n_trees)
  all_pcs[[1]] <- pc_first
  
  if (n_trees > 1) {
    for (tree in 2:n_trees) {
      all_pcs[[tree]] <- full_vine$pair_copulas[[tree]]
    }
  }
  
  # The structure must match a d‑dimensional vine
  d <- ncol(U_window)
  # Build the structure matrix for a D‑vine
  struct <- dvine_structure(1:d)
  
  vinecop_dist(
    pair_copulas = all_pcs,
    structure = full_vine$structure
  )
}


# ------------------------------------------------------------------------------
# Fully dynamic all-tree implementation. These definitions intentionally replace
# the earlier tree-1 prototype above while keeping old checkpoints loadable only
# through explicit migration code.
# ------------------------------------------------------------------------------

compute_dvine_edge_data <- function(U, vine) {
  U <- as.matrix(U); order <- as.integer(vine$structure$order); d <- length(order)
  ordered <- U[, order, drop = FALSE]
  left <- right <- matrix(vector("list", d * d), d, d)
  for (i in seq_len(d)) left[[i, i]] <- right[[i, i]] <- ordered[, i]
  edge_data <- vector("list", d - 1L)
  for (tree in seq_len(d - 1L)) {
    edge_data[[tree]] <- vector("list", d - tree)
    for (i in seq_len(d - tree)) {
      j <- i + tree
      a <- if (tree == 1L) ordered[, i] else left[[i, j - 1L]]
      b <- if (tree == 1L) ordered[, j] else right[[i + 1L, j]]
      edge_data[[tree]][[i]] <- cbind(a, b)
      pc <- vine$pair_copulas[[tree]][[i]]
      left[[i, j]] <- hbicop(cbind(a, b), cond_var = 2, pc)
      right[[i, j]] <- hbicop(cbind(a, b), cond_var = 1, pc)
    }
  }
  edge_data
}

train_all_edges <- function(U, vine_fit, asset_names, z, sigma,
                            epochs = 200, lr = 1e-3, patience = 20) {
  order <- as.integer(vine_fit$structure$order); d <- length(order)
  conditional_data <- compute_dvine_edge_data(U, vine_fit)
  nn_models <- vector("list", d - 1L)
  for (tree in seq_len(d - 1L)) {
    nn_models[[tree]] <- vector("list", d - tree)
    for (i in seq_len(d - tree)) {
      vars <- c(order[i], order[i + tree])
      pc <- vine_fit$pair_copulas[[tree]][[i]]
      if (!pc$family %in% c("student", "t")) {
        stop("Fully dynamic implementation requires t-copula edges.")
      }
      nu_fix <- as.numeric(pc$parameters[2])
      model <- train_nn_model(conditional_data[[tree]][[i]], nu_fix,
        z[, vars, drop = FALSE], sigma[, vars, drop = FALSE],
        epochs = epochs, lr = lr, patience = patience,
        rho_base = as.numeric(pc$parameters[1]))
      nn_models[[tree]][[i]] <- list(
        model = model, family = "t", nu = nu_fix, vars = vars,
        tree = tree, position = i, static_rho = as.numeric(pc$parameters[1])
      )
    }
  }
  nn_models
}

build_nn_vine <- function(nn_models, full_vine, U_window,
                          z_window = NULL, sigma_window = NULL) {
  U_window <- as.matrix(U_window)
  order <- as.integer(full_vine$structure$order); d <- length(order)
  if (length(nn_models) != d - 1L) stop("NN model does not cover every vine tree.")
  ordered <- U_window[, order, drop = FALSE]
  left <- right <- matrix(vector("list", d * d), d, d)
  for (i in seq_len(d)) left[[i, i]] <- right[[i, i]] <- ordered[, i]
  all_pcs <- vector("list", d - 1L)
  for (tree in seq_len(d - 1L)) {
    if (length(nn_models[[tree]]) != d - tree) stop("NN model misses a conditional edge.")
    all_pcs[[tree]] <- vector("list", d - tree)
    for (i in seq_len(d - tree)) {
      j <- i + tree
      a <- if (tree == 1L) ordered[, i] else left[[i, j - 1L]]
      b <- if (tree == 1L) ordered[, j] else right[[i + 1L, j]]
      nm <- nn_models[[tree]][[i]]
      vars <- nm$vars
      z_edge <- if (is.null(z_window)) matrix(0, nrow(U_window), 2L) else
        z_window[, vars, drop = FALSE]
      sigma_edge <- if (is.null(sigma_window)) matrix(1, nrow(U_window), 2L) else
        sigma_window[, vars, drop = FALSE]
      last_feature <- prepare_nn_feature_next(cbind(a, b), z_edge, sigma_edge,
                                              nm$nu, nm$static_rho)
      rho_now <- predict_rho_nn(nm$model, last_feature)
      rho_now <- pmax(pmin(rho_now, 0.995), -0.995)
      pc <- bicop_dist("t", 0, c(rho_now, nm$nu))
      all_pcs[[tree]][[i]] <- pc
      left[[i, j]] <- hbicop(cbind(a, b), cond_var = 2, pc)
      right[[i, j]] <- hbicop(cbind(a, b), cond_var = 1, pc)
    }
  }
  vinecop_dist(pair_copulas = all_pcs, structure = full_vine$structure)
}

# Build causal NN inputs from pseudo-observations. This avoids assuming that a
# fitted in-sample GARCH object can provide states beyond its fitting range.
derive_nn_states <- function(U, ewma_lambda = 0.94) {
  U <- as.matrix(U)
  if (nrow(U) < 3L || any(!is.finite(U))) stop("U must contain at least three finite observations.")
  z <- qnorm(pmin(pmax(U, 1e-6), 1 - 1e-6))
  sigma <- matrix(1, nrow = nrow(z), ncol = ncol(z), dimnames = dimnames(z))
  for (t in 2:nrow(z)) sigma[t, ] <- sqrt(ewma_lambda * sigma[t - 1L, ]^2 + (1 - ewma_lambda) * z[t - 1L, ]^2)
  list(z = z, sigma = sigma)
}

# Fit once, then let every D-vine edge correlation vary through time.
# This intentionally contains no rolling-window copula re-estimation.
fit_nn_dynamic_vine <- function(U, z = NULL, sigma = NULL, epochs = 200L,
                                lr = 1e-3, patience = 20L, seed = 20260741L,
                                enforce_truncation_gate = TRUE) {
  U <- as.matrix(U)
  if (nrow(U) < 30L || ncol(U) < 2L) stop("NN dynamic vine requires at least 30 observations and two assets.")
  if (is.null(z) || is.null(sigma)) {
    states <- derive_nn_states(U)
    z <- states$z; sigma <- states$sigma
  }
  if (!identical(dim(U), dim(z)) || !identical(dim(U), dim(sigma))) stop("U, z, and sigma must have identical dimensions.")
  set.seed(seed)
  torch_manual_seed(as.integer(seed))
  order <- select_dvine_order(U)
  truncation_validation <- validate_truncation(U, order)
  # Keep the truncation comparison as evidence for/against higher trees. The
  # production model retains all trees and makes all of them dynamic.
  backbone <- vinecop(U, var_types = rep("c", ncol(U)),
    structure = dvine_structure(order), family_set = "t",
    selcrit = "bic")
  nn_models <- train_all_edges(U, backbone, colnames(U), z, sigma,
    epochs = as.integer(epochs), lr = lr, patience = as.integer(patience))
  list(backbone = backbone, nn_models = nn_models, order = order,
       truncation_level = Inf, dynamic_edge_count = ncol(U) * (ncol(U) - 1L) / 2L,
       truncation_validation = truncation_validation,
       model = "nn_dynamic_all_tree_t_vine", training_observations = nrow(U), seed = seed)
}

# Snapshots are causal: each uses only observations through its decision date.
build_nn_vine_sequence <- function(nn_fit, U, z, sigma, rebal_dates, all_dates) {
  if (is.null(nn_fit$backbone) || is.null(nn_fit$nn_models)) stop("Invalid NN-vine fit.")
  U <- as.matrix(U); z <- as.matrix(z); sigma <- as.matrix(sigma)
  if (!identical(dim(U), dim(z)) || !identical(dim(U), dim(sigma))) stop("U, z, and sigma must have identical dimensions.")
  end_indices <- match(as.Date(rebal_dates), as.Date(all_dates))
  if (anyNA(end_indices) || any(end_indices < 2L)) stop("Every rebalance date needs at least two prior observations.")
  lapply(end_indices, function(end_index) {
    history <- seq_len(end_index)
    build_nn_vine(nn_fit$nn_models, nn_fit$backbone,
      U[history, , drop = FALSE], z[history, , drop = FALSE], sigma[history, , drop = FALSE])
  })
}

# Obsolete flat tree-1 cache format.
save_tree1_nn_models_legacy <- function(nn_models, dir_path) {
  if (!dir.exists(dir_path)) dir.create(dir_path, recursive = TRUE)
  
  for (k in seq_along(nn_models)) {
    nm <- nn_models[[k]]
    file_prefix <- file.path(dir_path, paste0("edge_", k))
    
    # Save metadata (R list without torch objects)
    meta <- list(
      family   = nm$family,
      nu       = ifelse(!is.null(nm$nu), nm$nu, NA),
      vars     = nm$vars,
      params   = nm$params,
      rotation = nm$rotation
    )
    saveRDS(meta, file = paste0(file_prefix, "_meta.rds"))
    
    # Save torch model separately if it exists
    if (!is.null(nm$model)) {
      torch_save(nm$model, paste0(file_prefix, "_model.pt"))
    }
  }
  #cat(sprintf("Saved %d NN models to %s\n", length(nn_models), dir_path))
}

# Obsolete flat tree-1 cache format.
load_tree1_nn_models_legacy <- function(dir_path) {
  files <- list.files(dir_path, pattern = "_meta\\.rds$")
  n_edges <- length(files)
  nn_models <- vector("list", n_edges)
  
  for (k in seq_len(n_edges)) {
    file_prefix <- file.path(dir_path, paste0("edge_", k))
    meta <- readRDS(paste0(file_prefix, "_meta.rds"))
    
    # Load torch model if it exists
    model <- NULL
    model_file <- paste0(file_prefix, "_model.pt")
    if (file.exists(model_file)) {
      model <- torch_load(model_file)
    }
    
    nn_models[[k]] <- list(
      model    = model,
      family   = meta$family,
      nu       = meta$nu,
      vars     = meta$vars,
      params   = meta$params,
      rotation = meta$rotation
    )
  }
  #cat(sprintf("Loaded %d NN models from %s\n", length(nn_models), dir_path))
  nn_models
}



# Versioned persistence for the nested all-tree NN model.
save_nn_models <- function(nn_models, dir_path) {
  dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
  manifest <- lapply(seq_along(nn_models), function(tree) {
    lapply(seq_along(nn_models[[tree]]), function(position) {
      nm <- nn_models[[tree]][[position]]
      model_file <- sprintf("tree_%02d_edge_%02d.pt", tree, position)
      torch_save(nm$model, file.path(dir_path, model_file))
      list(tree = tree, position = position, family = nm$family, nu = nm$nu,
           vars = nm$vars, static_rho = nm$static_rho, model_file = model_file)
    })
  })
  saveRDS(list(schema_version = 2L, trees = manifest),
          file.path(dir_path, "manifest.rds"))
  invisible(dir_path)
}

load_nn_models <- function(dir_path) {
  manifest_file <- file.path(dir_path, "manifest.rds")
  if (!file.exists(manifest_file)) {
    stop("Obsolete/incomplete NN cache: all-tree manifest.rds is required.")
  }
  manifest <- readRDS(manifest_file)
  if (!identical(manifest$schema_version, 2L)) stop("Unsupported NN cache schema.")
  lapply(manifest$trees, function(tree_models) {
    lapply(tree_models, function(meta) {
      model_path <- file.path(dir_path, meta$model_file)
      if (!file.exists(model_path)) stop("Missing NN edge model: ", model_path)
      c(meta[setdiff(names(meta), "model_file")],
        list(model = torch_load(model_path)))
    })
  })
}

save_nn_dynamic_vine_fit <- function(nn_fit, dir_path) {
  if (is.null(nn_fit$nn_models) || is.null(nn_fit$backbone)) stop("Invalid NN-vine fit.")
  save_nn_models(nn_fit$nn_models, dir_path)
  metadata <- nn_fit
  metadata$nn_models <- NULL
  saveRDS(metadata, file.path(dir_path, "fit_metadata.rds"))
  invisible(dir_path)
}

load_nn_dynamic_vine_fit <- function(dir_path) {
  metadata_file <- file.path(dir_path, "fit_metadata.rds")
  if (!file.exists(metadata_file)) stop("Missing NN-vine fit metadata: ", metadata_file)
  fit <- readRDS(metadata_file)
  fit$nn_models <- load_nn_models(dir_path)
  if (!identical(fit$model, "nn_dynamic_all_tree_t_vine")) {
    stop("Obsolete NN-vine fit; regenerate the synthetic-data bundle.")
  }
  fit
}

# ===========================================================================================

if (RUN_TESTS) {
  library(torch)
  source("helper/load_data.r")
  load("data/marginal_results.RData")
  load("data/vine_fit.RData")

  # Retrieve data for edge 1-2.
  u12 <- U[, c(1, 2)]                                 
  nu_fix <- as.numeric(vine_fit$pair_copulas[[1]][[1]]$parameters[2])

  # Split data into train / validation
  train_frac <- 0.8
  train_n <- floor(nrow(u12) * train_frac)
  u_train <- u12[1:train_n, ]
  u_val   <- u12[(train_n + 1):nrow(u12), ]

  cat(sprintf("Training on %d obs, validating on %d obs\n", train_n, nrow(u_val)))

  # Train the NN
  model <- train_nn_model(u_train, nu_fix, epochs = 200, lr = 1e-3, patience = 20)

  # Predict rho on full sample
  marginal_states <- extract_marginal_states(marginals, U, returns)
  z <- marginal_states$z[, vars]
  sigma <- marginal_states$sigma[, vars]

  full_data <- prepare_nn_data(u12, z, sigma, nu_fix)
  rho_pred  <- predict_rho_nn(model, full_data$features)

  # Static benchmark
  x1 <- qt(u12[, 1], df = nu_fix)
  x2 <- qt(u12[, 2], df = nu_fix)
  static_rho <- cor(x1, x2)

  cat(sprintf("Static rho:  %.4f\n", static_rho))
  cat(sprintf("NN mean rho: %.4f (sd = %.4f)\n", mean(rho_pred), sd(rho_pred)))

  # Plot the predicted rho over time
  plot(rho_pred, type = 'l', col = 'blue', ylim = range(c(rho_pred, static_rho)),
      main = "NN Predicted Rho (Edge 1-2)", xlab = "Time", ylab = expression(rho))
  abline(h = static_rho, col = 'red', lty = 2)
  legend("topright", c("NN rho", "Static rho"), col = c("blue", "red"), lty = c(1, 2))

  # Run out-of-sample log-likelihood on validation set
  val_features <- prepare_nn_data(u_val, z_dummy[-(1:train_n), , drop = FALSE], 
                                  sigma_dummy[-(1:train_n), , drop = FALSE], nu_fix)
rho_val <- predict_rho_nn(model, val_features$features)

  # NN log-likelihood
  ll_nn <- 0
  for (t in seq_along(rho_val)) {
    cop_t <- tCopula(param = rho_val[t], dim = 2, dispstr = "un", df = nu_fix, df.fixed = TRUE)
    ll_nn <- ll_nn + log(dCopula(u_val[t, , drop = FALSE], cop_t))
  }

  # Static log-likelihood
  cop_static <- tCopula(param = static_rho, dim = 2, dispstr = "un", df = nu_fix, df.fixed = TRUE)
  ll_stat <- sum(log(dCopula(u_val, cop_static)))

  cat(sprintf("Validation log-lik (NN):     %.2f\n", ll_nn))
  cat(sprintf("Validation log-lik (static): %.2f\n", ll_stat))
  cat(sprintf("Improvement: %.2f\n", ll_nn - ll_stat))
}
