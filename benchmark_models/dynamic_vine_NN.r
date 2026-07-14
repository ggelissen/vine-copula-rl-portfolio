# ==============================================================================
# dynamic_vine_NN.r
# Dynamic vine copula with neural network copula (family+parameters) selection
# ==============================================================================

library(rvinecopulib)
library(torch)
library(copula)

RUN_TESTS <- FALSE

prepare_nn_data <- function(u, z, sigma, nu_fix, lookback = 1) {
  T <- nrow(u)

  # Build the network features: lagged u, z, sigma, and nu
  rho_stat <- cor(qt(u[,1], df = nu_fix), qt(u[,2], df = nu_fix))
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
                    sigma_lag1 = c(NA, sigma[-T, 1]), sigma_lag2 = c(NA, sigma[-T, 2]),
                    theta_lag = theta_lag, score_lag = score_lag
  )

  features <- features[-1, , drop = FALSE] # Remove first row with NA
  y <- u[-1, , drop = FALSE]               # same for the uniforms

  list(features = torch_tensor(features, dtype = torch_float()),
       y = torch_tensor(y, dtype = torch_float()))
}


# Define the neural network model 
nn_model <- nn_module(
  "nn_model", 
  # Inputs: 8, Hidden layers: 16, 8, Output: 1 (version 1)
  initialize = function(input_dim = 8, hidden1 = 16, hidden2 = 8) {
    self$fc1 <- nn_linear(input_dim, hidden1)
    self$fc2 <- nn_linear(hidden1, hidden2)
    self$fc3 <- nn_linear(hidden2, 1)
  },
  forward = function(x) {
    x <- torch_tanh(self$fc1(x))
    x <- torch_tanh(self$fc2(x))
    rho <- torch_tanh(self$fc3(x))  # output in (-1, 1)
    rho
  }
)


# Train the neural network model
train_nn_model <- function(u, nu_fix, z, sigma, epochs = 500, lr = 1e-3, patience = 50) {
  # Prepare the data
  data <- prepare_nn_data(u, z, sigma, nu_fix)

  model <- nn_model()
  optim <- optim_adam(model$parameters, lr = lr)
  
  best_loss <- Inf; wait <- 0; best_model <- NULL

  for (epoch in 1:epochs) {
    # Set model to training mode
    model$train()
    optim$zero_grad()

    rho <- model(data$features)$squeeze()

    # Compute log-likelihood for t-copula
    x <- qt(as.matrix(data$y), df = nu_fix)
    x1 <- x[,1]; x2 <- x[,2]
    rho2 <- rho^2
    denom <- 1 - rho2
    quad <- x1^2 + x2^2 - 2 * rho * x1 * x2
    log_dens <- lgamma((nu_fix + 2)/2) + lgamma(nu_fix/2) -
      0.5 * log(denom) - (nu_fix + 2)/2 * log(1 + quad / (nu_fix * denom))
    loss <- -mean(log_dens)

    # Backpropagation and optimization
    loss$backward()
    optim$step()

    if (epoch %% 10 == 0) {
      #cat(sprintf("Epoch %d: Loss = %.6f\n", epoch, loss$item()))
    }

    # Early stopping based on validation loss
    if (loss$item() < best_loss) {
      best_loss <- loss$item()
      wait <- 0
      best_model <- model$clone()
    } else {
      wait <- wait + 1
      if (wait >= patience) {
        #cat("Early stopping...\n")
        break
      }
    }
  }
  best_model
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
    
    # Extract standardised residuals from the fitted uGARCHfit object
    z_mat[, i] <- as.numeric(model$fit@fit$residuals / model$fit@fit$sigma)
    
    # Extract conditional volatilities
    sigma_mat[, i] <- as.numeric(model$fit@fit$sigma)
  }
  
  colnames(z_mat) <- names(marginals)
  colnames(sigma_mat) <- names(marginals)
  
  list(z = z_mat, sigma = sigma_mat)
}



# Train NN for all t‑copula edges
train_all_edges <- function(U, vine_fit, asset_names, z, sigma, epochs = 200, lr = 1e-3, patience = 20) {
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
      
      model <- train_nn_model(u_edge, nu_fix, z, sigma, epochs = epochs, lr = lr, patience = patience)
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
predict_vine_params <- function(nn_models, U_window, z_window = NULL, sigma_window = NULL) {
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
build_nn_vine <- function(nn_models, full_vine, U_window, z_window = NULL, sigma_window = NULL) {
  # Predict first‑tree pair copulas
  pc_first <- predict_vine_params(nn_models, U_window, z_window, sigma_window)
  
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
    structure = struct
  )
}


# Save NN models to disk
save_nn_models <- function(nn_models, dir_path) {
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

# Load NN models from disk
load_nn_models <- function(dir_path) {
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