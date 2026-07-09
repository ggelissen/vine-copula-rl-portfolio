# ==============================================================================
# dynamic_vine_NN.r
# Dynamic vine copula with neural network copula (family+parameters) selection
# ==============================================================================

library(rvinecopulib)
library(torch)

prepare_nn_data <- function(u, z, sigma, nu_fix, lookback = 1) {
  T <- nrow(u)

  # Build the network features: lagged u, z, sigma, and nu
  rho_stat <- cor(qt(u[,1], df = nu_fix), qt(u[,2], df = nu_fix))
  theta_lag <- rep(rho_stat, T)
  score_lag <- rep(0, T)

  # Pre-compute the score for all t
  x <- qt(u, df = nu_fix)
  for (t in 2:T) {
    score_lag[t] <- (nu_fix + 2) / nu_fix * (theta_lag[t-1] * (x[t-1,1]^2 + x[t-1,2]^2)
                     - (1 + theta_lag[t-1]^2) * x[t-1,1] * x[t-1,2]) / ((1 - theta_lag[t-1]^2)
                    * (nu_fix * (1 - theta_lag[t-1]^2) + (x[t-1,1]^2 + x[t-1,2]^2 - 2 * 
                    theta_lag[t-1] * x[t-1,1] * x[t-1,2])))
  }

  # Build matrix of features
  features <- cbind(u_lag1 = rbind(NA, u[-T, 1]), u_lag2 = rbind(NA, u[-T, 2]),
                    z_lag1 = rbind(NA, z[-T, 1]), z_lag2 = rbind(NA, z[-T, 2]),
                    sigma_lag1 = rbind(NA, sigma[-T, 1]), sigma_lag2 = rbind(NA, sigma[-T, 2]),
                    theta_lag = theta_lag, score_lag = score_lag
  )

  features <- features[-1, ] # Remove first row with NA
  y <- u[-1, ]               # same for the uniforms

  list(features = torch_tensor(features, dtype = torch_float()),
       y = torch_tensor(y, dtype = torch_float()))
}


# Define the neural network model 
nn_model <- nn_module(
  "nn_model", 
  # Inputs: 8, Hidden layers: 16, 8, Output: 1
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
train_nn_model <- function(u, nu_fix, epochs = 200, lr = 1e-3, patience = 20) {
  # Prepare the data
  z_dummy <- matrix(0, nrow(u), 2)
  sigma_dummy <- matrix(1, nrow(u), 2)
  data <- prepare_nn_data(u, z_dummy, sigma_dummy, nu_fix)

  model <- nn_model()
  optim <- optim_adam(model$parameters, lr = lr)
  
  best_loss <- Inf; wait <- 0; best_model <- NULL

  for (epoch in 1:epochs) {
    # Set model to training mode
    model$train()
    optim$zero_grad()

    rho <- model(data$features)$squeeze()

    # Compute log-likelihood for t-copula
    x <- qt(data$y, df = nu_fixed)
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
      cat(sprintf("Epoch %d: Loss = %.6f\n", epoch, loss$item()))
    }

    # Early stopping based on validation loss
    if (loss$item() < best_loss) {
      best_loss <- loss$item()
      wait <- 0
      best_model <- model$clone()
    } else {
      wait <- wait + 1
      if (wait >= patience) {
        cat("Early stopping...\n")
        break
      }
    }
  }
  best_model
}


# Predict the copula parameter (rho) using the trained neural network model
predict_rho_nn <- function(model, features) {
  model$eval()
  with no_grad({
    rho <- model(features)$squeeze()
    as.numeric(rho)
  })
}