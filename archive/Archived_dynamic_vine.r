# ==================================================================
# dynamic_copula.r
# Turning the first tree of the D-vine dynamic using GAS functions
# ==================================================================

library(copula)
library(rvinecopulib)
library(parallel)
load("data/marginal_results.RData")
load("data/vine_fit.RData")

# Parameter bounds for each copula family
param_bounds <- function(family) {
  switch(family,
    gaussian = c(-0.999, 0.999),
    t        = c(-0.999, 0.999),
    gumbel   = c(1.001, 50),
    clayton  = c(0.001, 50),
    frank    = c(-50, 50),
    joe      = c(1.001, 50)
  )
}

# Bound theta to valid range
clamp_theta <- function(theta, family) {
  bound <- param_bounds(family)
  if (is.na(theta) || is.nan(theta)) theta <- 0  # in case NaN appears
  pmax(pmin(theta, bound[2]), bound[1])
}

# Creates copula object based on family name and parameters
make_copula <- function(family, theta, fixed_param = NULL, rotation = 0) {
  copula <- switch(family,
    gaussian = normalCopula(param = theta, dim = 2),
    t        = tCopula(param = theta, dim = 2, dispstr = "un", 
                       df = as.numeric(fixed_param), df.fixed = TRUE),
    gumbel   = gumbelCopula(param = theta, dim = 2),
    clayton  = claytonCopula(param = theta, dim = 2),
    frank    = frankCopula(param = theta, dim = 2),
    joe      = joeCopula(param = theta, dim = 2)
  )
  if (rotation != 0) copula <- rotCopula(copula, rotation)
  copula
}

# Uses finite difference to calculate score of conditional likelihood
copula_score <- function(u, family, theta, fixed_param = NULL, rotation = 0) {
  bound <- param_bounds(family)
  delta <- 1e-5
  theta_upper <- min(theta + delta, bound[2])
  theta_lower <- max(theta - delta, bound[1])

  ll_upper <- dCopula(u, make_copula(family, theta_upper, fixed_param, rotation), log = TRUE)
  ll_lower <- dCopula(u, make_copula(family, theta_lower, fixed_param, rotation), log = TRUE)
  score <- (ll_upper - ll_lower) / (2 * delta)
  if (is.na(score) || is.nan(score) || !is.finite(score)) score <- 0
  score
}

# Returns varying values of theta for every t
gas_nll <- function(par, u, family, fixed_param, rotation) {
  if (any(is.na(par))) return(1e10)
  omega <- par[1]; alpha <- par[2]; beta <- par[3]
  if (alpha <= 0 || beta <= 0 || alpha + beta >= 1) return(1e10)
  
  n <- nrow(u)
  theta <- clamp_theta(omega / (1 - beta), family)  # starting value for theta (Markov property)
  
  ll <- 0
  for (t in 2:n) {
    s <- copula_score(u[t-1, , drop = FALSE], family, theta, fixed_param, rotation)
    theta <- omega + beta * theta + alpha * s

    if (is.na(theta) || is.nan(theta)) theta <- clamp_theta(omega / (1 - beta), family)  # reset to unconditional mean if NaN
    theta <- clamp_theta(theta, family)

    dens <- dCopula(u[t, , drop = FALSE], make_copula(family, theta, fixed_param, rotation), log = TRUE)
    if (is.na(dens) || is.nan(dens) || !is.finite(dens)) dens <- -30
    ll <- ll + dens
  }
  -ll
}


# Fit GAS to uniform marginals (follows Patton 2006)
fit_gas <- function(u, family = c("gaussian","t","gumbel","clayton","gumbel","frank","joe"),
                    fixed_param = NULL, rotation = 0) {
  family <- match.arg(family)
  
  # Estimate constant copula to start
  start_theta <- switch(family, gaussian = 0.5, t = 0.5, gumbel = 1.5, clayton = 1.5, frank = 1, joe = 1.5)
  const_cop <- make_copula(family, start_theta, fixed_param, rotation)
  fit_const <- fitCopula(const_cop, u, method = "mpl")
  theta0 <- fit_const@estimate[1]

  # Bounds for (omega, alpha, beta)
  bound <- param_bounds(family)
  lower <- c(bound[1] * 0.5, 0.001, 0.5)
  upper <- c(bound[2] * 0.5, 0.3, 0.999)

  # Apply MLE for omega, alpha and beta
  start <- c(omega = (1 - 0.95) * theta0, alpha = 0.05, beta = 0.90)
  opt <- optim(start, gas_nll, u = u, family = family, fixed_param = fixed_param, rotation = rotation, 
               method = "L-BFGS-B", lower = lower, upper = upper)
  
  list(omega = opt$par[1], alpha = opt$par[2], beta = opt$par[3],
       theta0 = theta0, fixed_param = fixed_param, converged = opt$convergence == 0)
  }


# ========================================================================================

# Extract edges and pair-copulas on first-tree of vine
order <- vine_fit$structure$order
d <- length(order)

tree1_edges <- cbind(order[1:(d-1)], order[2:d])
tree1_pcs <- vine_fit$pair_copulas[[1]]
n_edges <- nrow(tree1_edges)

gas_models <- vector("list", n_edges)

fam_map <- c(student = "t", t = "t", gaussian = "gaussian", gumbel = "gumbel",
             clayton = "clayton", frank = "frank", joe = "joe")

# Create tasks for parallel processing
tasks <- vector("list", n_edges)
for (k in seq_len(n_edges)) {
  # obtain edge information
  vars <- tree1_edges[k, ]                     # indices of the two variables
  pc   <- tree1_pcs[[k]]                       # pair‑copula object
  fam  <- pc$family                            # e.g., "student", "gumbel", "clayton", …
  rot <- pc$rotation                           # rotation angle (0, 90, 180, 270)
  
  # Fixed parameters: for families with >1 parameter, second parameter fixed
  fam_gas <- if (fam %in% names(fam_map)) fam_map[fam] else fam
  fixed <- if (fam %in% c("student", "t")) as.numeric(pc$parameters[2]) else NULL

  tasks[[k]] <- list(u = U[, vars], family = fam_gas, fixed = fixed, rotation = rot,
                     edge = paste(vars[1], vars[2], sep = "-"))
}

# Run tasks in parallel
n_cores <- min(detectCores() - 1, n_edges)
cl <- makeCluster(n_cores)
clusterExport(cl, c("fit_gas", "make_copula", "copula_score", "gas_nll", "clamp_theta", "param_bounds"))
clusterEvalQ(cl, library(copula))

cat(sprintf("Running %d edges on %d cores...\n", n_edges, n_cores))

gas_models <- parLapply(cl, tasks, function(task) {
  cat(sprintf("Fitting edge %s...\n", task$edge))
  result <- tryCatch(
    fit_gas(task$u, family = task$family, fixed_param = task$fixed, rotation = task$rotation),
    error = function(e) {
      msg <- conditionMessage(e)
      if (is.null(msg) || is.na(msg)) msg <- "unknown error"
      list(error = msg)
    }
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