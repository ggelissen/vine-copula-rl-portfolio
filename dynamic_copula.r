# ==================================================================
# dynamic_copula.r
# Turning the first tree of the D-vine dynamic using GAS functions
# ==================================================================

library(copula)
library(rvinecopulib)
library(parallel)
load("data/marginal_results.RData")
load("data/vine_fit.RData")

# Uses finite difference to calculate score of conditional likelihood
copula_score <- function(u, family, theta, fixed_param = NULL, rotation = 0) {
  delta <- 1e-5
  theta_upper <- min(theta + delta, 0.999)
  theta_lower <- max(theta - delta, -0.999)

  if (family %in% c("gumbel", "clayton", "joe")) {
    theta_upper <- max(theta_upper, 1.001)
    theta_lower <- max(theta_lower, 1.001)
  }

  ll_upper <- log(dCopula(u, make_copula(family, theta_upper, fixed_param, rotation)))
  ll_lower <- log(dCopula(u, make_copula(family, theta_lower, fixed_param, rotation)))
  (ll_upper - ll_lower) / (2 * delta)
}

# Helper function to create copula object based on family name and parameters
make_copula <- function(family, theta, fixed_param = NULL, rotation = 0) {
  copula <- switch(family,
    gaussian = normalCopula(param = theta, dim = 2),
    t        = {
      cop <- tCopula(param = theta, dim = 2, df = fixed_param)
      cop@df.fixed <- TRUE
      cop
    },
    gumbel   = gumbelCopula(param = theta, dim = 2),
    clayton  = claytonCopula(param = theta, dim = 2),
    frank    = frankCopula(param = theta, dim = 2),
    joe      = joeCopula(param = theta, dim = 2)
  )
  if (rotation != 0) copula <- rotCopula(copula, rotation)
  copula
}

clamp_theta <- function(theta, family) {
  if (family %in% c("gaussian", "t", "frank")) {
    pmax(pmin(theta, 0.999), -0.999)
  } else {
    pmax(theta, 1.001)
  }
}


# Returns varying values of theta for every t
gas_nll <- function(par, u, family, fixed_param, rotation) {
  omega <- par[1]; alpha <- par[2]; beta <- par[3]
  if (alpha <= 0 || beta <= 0 || alpha + beta >= 1) return(1e10)
  n <- nrow(u)
  theta_val <- numeric(n)
  theta_val[1] <- omega / (1 - beta)
  ll <- 0
  for (t in 2:n) {
    s <- copula_score(u[t-1, , drop = FALSE], family, theta_val[t-1], fixed_param, rotation)
    theta_val[t] <- omega + beta * theta_val[t-1] + alpha * s
    theta_val[t] <- clamp_theta(theta_val[t], family)
    copula_t <- make_copula(family, theta_val[t], fixed_param, rotation)
    dens <- dCopula(u[t, , drop = FALSE], copula_t)
    if (dens <= 0) dens <- 1e-10
    ll <- ll + log(dens)
  }
  -ll
}


# gas_filter <- function(u, omega, alpha, beta, family, fixed_param = NULL, rotation = 0) {
#   clamp_theta <- function(theta, family) {
#     if (family %in% c("gaussian", "t", "frank")) {
#       pmax(pmin(theta, 0.999), -0.999)
#     } else {
#       pmax(theta, 1.001)
#     }
#   }

#   n <- nrow(u)
#   theta <- numeric(n)
#   theta[1] <- omega / (1 - beta)
#   theta[1] <- clamp_theta(theta[1], family)

#   for (t in 2:n) {
#     s <- copula_score(u[t-1, , drop = FALSE], family, theta[t-1], fixed_param, rotation)
#     theta[t] <- omega + beta * theta[t-1] + alpha * s
#     theta[t] <- clamp_theta(theta[t], family)
#   }
#   theta
# }

# # Calculates negative log-likelihood for GAS(1,1)
# gas_nll <- function(par, u, family, fixed_param = NULL, rotation = 0) {
#   omega <- par[1]; alpha <- par[2]; beta <- par[3]
#   if (alpha <= 0 || beta <=0 || alpha + beta >= 1) return(1e10)
  
#   theta <- gas_filter(u, omega, alpha, beta, family, fixed_param, rotation)

#   ll <- 0
#   for (t in 2:nrow(u)) {
#     copula_t <- make_copula(family, theta[t], fixed_param, rotation)
#     ll <- ll + log(dCopula(u[t, , drop = FALSE], copula_t))
#   }
#   -ll
# }


# Fit GAS to uniform marginals (follows Patton 2006)
fit_gas <- function(u, family = c("gaussian","t","gumbel","clayton","gumbel","frank","joe"),
                    fixed_param = NULL, rotation = 0) {
  family <- match.arg(family)
  
  # Estimate constant copula to start
  const_cop <- make_copula(family, 0.5, fixed_param, rotation)
  fit_const <- fitCopula(const_cop, u, method = "mpl")
  theta0 <- fit_const@estimate[1]
  fixed_est <- if(length(fit_const@estimate) > 1) fit_const@estimate[-1] else fixed_param
  if (!is.null(fixed_est)) fixed_est <- fixed_param

  # Apply MLE for omega, alpha and beta
  start <- c(omega = (1 - 0.95) * theta0, alpha = 0.05, beta = 0.90)
  opt <- optim(start, gas_nll, u = u, family = family, fixed_param = fixed_est, rotation = rotation, 
               method = "L-BFGS-B", lower = c(0, 0, 0.5), upper = c(0.5, 0.3, 0.999))
  
  list(omega = opt$par[1], alpha = opt$par[2], beta = opt$par[3],
       theta0 = theta0, fixed_param = fixed_est, converged = opt$convergence == 0)
}


# ========================================================================================

# Extract edges and pair-copulas on first-tree of vine
order <- vine_fit$structure$order
d <- length(order)

tree1_edges <- cbind(order[1:(d-1)], order[2:d])
tree1_pcs <- vine_fit$pair_copulas[[1]]
n_edges <- nrow(tree1_edges)

gas_models <- vector("list", n_edges)

# Create tasks for parallel processing
tasks <- vector("list", n_edges)
for (k in seq_len(n_edges)) {
  # obtain edge information
  vars <- tree1_edges[k, ]                     # indices of the two variables
  pc   <- tree1_pcs[[k]]                       # pair‑copula object
  fam  <- pc$family                            # e.g., "student", "gumbel", "clayton", …
  rot <- pc$rotation                           # rotation angle (0, 90, 180, 270)
  
  # Fixed parameters: for families with >1 parameter, second parameter fixed
  fixed <- if (fam %in% c("student", "t")) pc$parameters[2] else NULL

  tasks[[k]] <- list(u = U[, vars], family = fam_gas, fixed = fixed, rotation = rot,
                     edge = paste(vars[1], vars[2], sep = "-"))
}

# Run tasks in parallel
n_cores <- min(detectCores() - 1, n_edges)
c1 <- makeCluster(n_cores)
clusterExport(c1, c("fit_gas", "make_copula", "copula_score", "gas_nll", "clamp_theta"))
clusterEvalQ(c1, library(copula))

cat(sprintf("Running %d edges on %d cores...\n", n_edges, n_cores))

gas_models <- parLapply(cl, tasks, function(task) {
  cat(sprintf("Fitting edge %s...\n", task$edge))
  result <- tryCatch(
    fit_gas(task$u, family = task$family, fixed_param = task$fixed, rotation = task$rotation),
    error = function(e) list(error = e$message)
  )
  cat(sprintf("Edge %s done.\n", task$edge))
  result
})

stopCluster(cl)

# Save tree results
save(gas_models, tree1_edges, file = "data/gas_first_tree.RData")

for (k in seq_len(n_edges)) {
  gm <- gas_models[[k]]
  if (!is.null(gm$error)) {
    cat(sprintf("Edge %d-%d: FAILED (%s)\n", tree1_edges[k,1], tree1_edges[k,2], gm$error))
  } else {
    cat(sprintf("Edge %d-%d: omega=%.3f, alpha=%.3f, beta=%.3f, converged=%s\n",
                tree1_edges[k,1], tree1_edges[k,2], gm$omega, gm$alpha, gm$beta, gm$converged))
  }
}