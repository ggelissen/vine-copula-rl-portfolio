###
# Computational Implementation of: 
# Li & Ng (2000)
# OPTIMAL DYNAMIC PORTFOLIO SELECTION: MULTIPERIOD MEAN-VARIANCE FORMULATION
###

# Loading packages
# ....

compute_results <- function(returns, w0, strat_param, strat) {
  
  T <- length(returns)         # number of time periods
  d <- ncol(returns[[1]]) - 1  # number of risky assets

  inv_E_ee_list <- vector("list", T)
  mean_e_list <- vector("list", T)
  E_r0e_list <- vector("list", T)

  A1 <- numeric(T)
  A2 <- numeric(T)
  B <- numeric(T)

  # Loop over periods
  for (t in 1:T) {
    returns_t <- returns[[t]]    # returns at time t
    N <- nrow(returns_t)         # number of returns at time t

    r0 <- returns_t[,1]          # reference asset returns
    e <- returns_t[,-1] - r0     # excess returns

    mean_r0 <- mean(r0)                         # expectation of reference asset
    mean_e <- colMeans(e)                       # expectation of risky assets
    E_ee <- (t(e) %*% e) / N                    # raw second moment
    E_r0e <- colMeans(r0 * e)                   # cross moment
    E2_r0 <- mean(r0^2)                         # second moment of reference asset
    inv_E_ee <- solve(E_ee + diag(1e-9,d))      # use diag to ensure invertibility

    inv_E_ee_list[[t]] <- inv_E_ee
    mean_e_list[[t]] <- mean_e
    E_r0e_list[[t]] <- E_r0e

    B[t] <- as.numeric(t(mean_e) %*% inv_E_ee %*% mean_e)    # multivariate squared Sharpe Ratio
    A1[t] <- mean_r0 - t(mean_e) %*% inv_E_ee %*% E_r0e      # expected return of 'hedged' reference asset
    A2[t] <- E2_r0 - t(E_r0e) %*% inv_E_ee %*% E_r0e         # second moment of residual
  }

  # Compute tail products
  tail_prod_A1 <- numeric(T)
  tail_prod_A2 <- numeric(T)
  tail_prod_A1[T] <- 1
  tail_prod_A2[T] <- 1

  if (T > 1) {
    for (t in (T-1):1) {
      tail_prod_A1[t] <- tail_prod_A1[t+1] * A1[t+1]
      tail_prod_A2[t] <- tail_prod_A2[t+1] * A2[t+1]
    }
  }

  # Calculate B1, B2 and aggregates
  B1 <- B * tail_prod_A1 / (2 * tail_prod_A2)
  B2 <- B * (tail_prod_A1 / (2 * tail_prod_A2))^2

  mu <- prod(A1)                   # multiplicative drift
  nu <- sum(tail_prod_A1 * B1)     # total speculative capacity
  tau <- prod(A2)                  # total second-moment multiplier

  a <- (nu / 2) - nu^2              # curvature of efficient frontier
  b <- (mu * nu) / a                # scaling coefficient
  c <- tau - mu^2 - (a * b^2)       # unavoidable risk

  if (a <= 0) warning("a <= 0: mean-variance frontier may be ill-conditioned.")
  
  # Determine optimal portfolio policy
  if (strat == "E") omega <- strat_param
  if (strat == "P1") omega <- nu / (2 * sqrt(a * (strat_param - c * w0^2)))
  if (strat == "P2") omega <- nu^2 / (2 * a * (strat_param - (mu + b * nu) * w0))
  
  spec_scalar <- 0.5 * (b * w0 + nu / (2 * omega * a))
  spec_mult <- tail_prod_A1 / tail_prod_A2

  policy <- vector("list", T)

  for (t in 1:T) {
    inv_E_ee <- inv_E_ee_list[[t]]
    mean_e <- mean_e_list[[t]]
    E_r0e <- E_r0e_list[[t]]

    Kt <- inv_E_ee %*% E_r0e                                   # hedging coefficient (indepedent of risk strategy)
    vt <- spec_scalar * spec_mult[t] * (inv_E_ee %*% mean_e)   # speculative demand (scales with risk tolerance)

    policy[[t]] <- list(Kt = Kt, vt = vt)
   }

  # Return outputs
  return(list(
    global = list(mu = mu, nu = nu, tau = tau, a = a, b = b, c = c),
    policy = policy,
    per_period = list(A1 = A1, A2 = A2, B = B, B1 = B1, B2 = B2)
  ))

}

# ==============================================================================================

set.seed(123)

# Setup
T <- 3         # number of periods
d <- 2         # number of risky assets
N <- 1000      # number of scenarios per period
w0 <- 100000   # initial wealth

# Strategy
sigma <- 22.6e6   # risk upper bound (or variance constraint)
epsilon <- 125000 # expected terminal wealth (mean constraint)
gamma <- 2        # risk-aversion coefficient
strategy <- "E"  # P1(sigma), P2(epsilon) or E(gamma)

# Create returns list
returns <- list()
for (t in 1:T) {
  r0 <- rnorm(N, mean = 1.04, sd = 0.02)
  r1 <- rnorm(N, mean = 1.02, sd = 0.15)
  r2 <- rnorm(N, mean = 1.09, sd = 0.12)
  returns[[t]] <- cbind(r0, r1, r2)
}

# Print results
if (strategy == "P1") result <- compute_results(returns, w0, sigma, strategy)
if (strategy == "P2") result <- compute_results(returns, w0, epsilon, strategy)
if (strategy == "E") result <- compute_results(returns, w0, gamma, strategy)
print(result$global) 

K1 <- result$policy[[1]]$Kt
v1 <- result$policy[[1]]$vt

cat("u1 = -(", K1[1], ") * w1 + ", v1[1], " (for asset 1)\n", sep="")
  cat("u1 = -(", K1[2], ") * w1 + ", v1[2], " (for asset 2)\n", sep="")
