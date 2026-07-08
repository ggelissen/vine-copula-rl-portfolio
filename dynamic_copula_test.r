# ============================================================
# dynamic_copula_test.r – Fast GAS(1,1) for Student t copula
# ============================================================
library(copula)
library(parallel)

# ---- Analytical score for t-copula (vectorised) ----
t_score <- function(u, rho, nu, x1, x2) {
  # x1, x2 already pre-computed as qt(u, df=nu)
  rho2 <- rho^2
  denom1 <- 1 - rho2
  quad <- x1^2 + x2^2 - 2 * rho * x1 * x2
  denom2 <- nu * denom1 + quad
  numerator <- rho * (x1^2 + x2^2) - (1 + rho2) * x1 * x2
  (nu + 2) / nu * numerator / (denom1 * denom2)
}

# ---- Negative log-likelihood (reparameterised) ----
gas_nll_t <- function(par, u, nu, beta, x1, x2) {
  if (any(is.na(par)) || is.na(beta) || is.na(nu)) return(1e10)
  theta_inf <- par[1]   # unconditional mean rho
  alpha <- par[2]
  if (alpha <= 0 || theta_inf <= 0) return(1e10)
  
  n <- nrow(u)
  theta <- theta_inf
  ll <- 0
  for (t in 2:n) {
    # score using pre-computed x1,x2
    s <- t_score(u[t-1, , drop = FALSE], theta, nu, x1[t-1], x2[t-1])
    # GAS update
    theta <- (1 - beta) * theta_inf + beta * theta + alpha * s
    theta <- pmax(pmin(theta, 0.999), -0.999)
    # log-likelihood contribution
    dens <- dCopula(u[t, , drop = FALSE],
                    tCopula(param = theta, dim = 2, dispstr = "un",
                            df = nu, df.fixed = TRUE),
                    log = TRUE)
    if (is.na(dens) || dens == -Inf) dens <- -30
    ll <- ll + dens
  }
  -ll
}

# ---- Profile likelihood over beta (robust version) ----
fit_gas_t_fast <- function(u, nu, beta_grid = c(0.80, 0.85, 0.90, 0.95)) {
  x1 <- qt(u[,1], df = nu)
  x2 <- qt(u[,2], df = nu)

  best_nll <- Inf
  best_par <- c(max(cor(x1, x2), 0.1), 0.02)
  best_beta <- 0.85

  for (beta in beta_grid) {
    rho_start <- max(cor(x1, x2), 0.1)
    alpha_start <- 0.02
    opt <- optim(
      par = c(rho_start, alpha_start),
      fn = gas_nll_t,
      u = u, nu = nu, beta = beta, x1 = x1, x2 = x2,
      method = "L-BFGS-B",
      lower = c(0.01, 0.001),
      upper = c(0.999, 0.3),
      control = list(maxit = 200)
    )
    if (opt$convergence == 0 && opt$value < best_nll) {
      best_nll <- opt$value
      best_par <- opt$par
      best_beta <- beta
    }
  }

  # Refinement with tighter tolerance
  start3 <- c(best_par[1], best_par[2], best_beta)
  if (any(is.na(start3))) start3 <- c(0.5, 0.02, 0.85)

  opt3 <- optim(
    par = start3,
    fn = function(par, u, nu, x1, x2) {
      theta_inf <- par[1]; alpha <- par[2]; beta <- par[3]
      if (is.na(theta_inf) || is.na(alpha) || is.na(beta)) return(1e10)
      if (alpha <= 0 || beta <= 0 || beta >= 1 || theta_inf <= 0) return(1e10)
      gas_nll_t(c(theta_inf, alpha), u, nu, beta, x1, x2)
    },
    u = u, nu = nu, x1 = x1, x2 = x2,
    method = "L-BFGS-B",
    lower = c(0.01, 0.001, 0.5),
    upper = c(0.999, 0.3, 0.999),
    control = list(maxit = 500, factr = 1e-8)
  )

  list(theta_inf = opt3$par[1],
       alpha    = opt3$par[2],
       beta     = opt3$par[3],
       nll      = opt3$value,
       converged = opt3$convergence == 0)
}


# ====================================================================================

load("data/marginal_results.RData")
load("data/vine_fit.RData")

U_dev <- tail(U, 500)

order <- vine_fit$structure$order
d <- length(order)
tree1_edges <- cbind(order[1:(d-1)], order[2:d])
tree1_pcs   <- vine_fit$pair_copulas[[1]]

edge_info <- data.frame(
  edge = 1:nrow(tree1_edges),
  v1   = tree1_edges[,1],
  v2   = tree1_edges[,2],
  family = sapply(tree1_pcs, function(pc) pc$family),
  nu    = sapply(tree1_pcs, function(pc) if (pc$family %in% c("student","t")) pc$parameters[2] else NA),
  stringsAsFactors = FALSE
)

t_edges <- which(edge_info$family %in% c("student", "t"))
cat(sprintf("Fitting %d t‑copula edges...\n", length(t_edges)))

tasks <- lapply(t_edges, function(k) {
  list(
    edge_name = paste(edge_info$v1[k], edge_info$v2[k], sep = "-"),
    u = U_dev[, c(edge_info$v1[k], edge_info$v2[k])],
    nu = edge_info$nu[k]
  )
})

# Initialise parallel cluster
n_cores <- min(detectCores() - 1, length(tasks))
cl <- makeCluster(n_cores, type = "PSOCK")

# Export everything the workers need
clusterExport(cl, c("fit_gas_t_fast", "gas_nll_t", "t_score", "tasks"))
clusterEvalQ(cl, library(copula))

cat(sprintf("Running %d edges on %d cores...\n", length(tasks), n_cores))

# Run with a timing print
start_time <- Sys.time()
results <- parLapply(cl, seq_along(tasks), function(i) {
  task <- tasks[[i]]
  cat(sprintf("[Worker %d] Fitting edge %s...\n", i, task$edge_name))
  result <- tryCatch(
    fit_gas_t_fast(task$u, task$nu, beta_grid = c(0.80, 0.85, 0.90, 0.95)),
    error = function(e) list(error = conditionMessage(e))
  )
  cat(sprintf("[Worker %d] Edge %s done.\n", i, task$edge_name))
  result
})

stopCluster(cl)
cat(sprintf("All done in %.1f minutes.\n", difftime(Sys.time(), start_time, units = "mins")))

# Print results
for (i in seq_along(results)) {
  r <- results[[i]]
  edge <- tasks[[i]]$edge_name
  if (!is.null(r$error)) {
    cat(sprintf("\nEdge %s: FAILED (%s)\n", edge, r$error))
  } else {
    cat(sprintf("\nEdge %s:\n", edge))
    cat(sprintf("  theta_inf = %.4f\n", r$theta_inf))
    cat(sprintf("  alpha     = %.4f\n", r$alpha))
    cat(sprintf("  beta      = %.4f\n", r$beta))
    cat(sprintf("  nll       = %.4f\n", r$nll))
    cat(sprintf("  converged = %s\n", r$converged))
  }
}

save(results, t_edges, edge_info, file = "data/gas_results_500d.RData")