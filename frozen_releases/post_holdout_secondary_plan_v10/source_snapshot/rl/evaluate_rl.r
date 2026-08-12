# Historical out-of-sample evaluation for the episodic vine-RL policy.
# The benchmark strategies and RL policy must be scored on the same realised
# monthly returns; stochastic simulated evaluation is a separate robustness
# experiment and must not be pooled with this backtest.

suppressPackageStartupMessages({ library(xts); library(ggplot2); library(data.table) })
source("benchmark_models/dynamic_vine_NN.r")
source("rl/rl_environment.r")
source("helper/time_split.r")
source("helper/load_data.r")

eval_model_dir <- Sys.getenv("EVAL_MODEL_DIR")
eval_output_dir <- Sys.getenv("EVAL_OUTPUT_DIR", "data")
eval_weights_only <- tolower(Sys.getenv("EVAL_WEIGHTS_ONLY", "false")) %in%
  c("1", "true", "yes")
eval_checkpoint_models <- trimws(Sys.getenv("EVAL_CHECKPOINT_MODELS", ""))
development_dry_run <- tolower(Sys.getenv(
  "EVAL_DEVELOPMENT_DRY_RUN", "false")) %in% c("1", "true", "yes")
eval_window_id <- Sys.getenv("EVAL_WINDOW_ID", "locked_oos_v1")
dir.create(eval_output_dir, recursive = TRUE, showWarnings = FALSE)
eval_seed <- as.integer(Sys.getenv("EVAL_SEED"))
eval_gamma <- as.numeric(Sys.getenv("EVAL_GAMMA")); eval_lambda <- as.numeric(Sys.getenv("EVAL_LAMBDA")); eval_kappa <- as.numeric(Sys.getenv("EVAL_KAPPA"))
L <- as.integer(Sys.getenv("L")); ref_col <- as.integer(Sys.getenv("REF_COL")); n_sim_cvar <- as.integer(Sys.getenv("N_SIM_CVAR")); seq_len <- as.integer(Sys.getenv("ENV_SEQ_LEN"))
holding_days <- as.integer(Sys.getenv("ENV_HOLDING_DAYS", "21"))
gross_leverage <- as.numeric(Sys.getenv("ENV_GROSS_LEVERAGE", "1.5"))
net_exposure <- as.numeric(Sys.getenv("ENV_NET_EXPOSURE", "1"))
short_borrow_rate <- as.numeric(Sys.getenv("ENV_SHORT_BORROW_RATE", "0.03"))
cash_borrow_rate <- as.numeric(Sys.getenv("ENV_CASH_BORROW_RATE", "0.02"))
utility_mode <- Sys.getenv("ENV_UTILITY_MODE", "terminal_wealth_crra")
vine_observation_mode <- Sys.getenv("VINE_OBSERVATION_MODE", "full")
if (!vine_observation_mode %in% c("full", "zero")) stop("Invalid VINE_OBSERVATION_MODE.")
vine_model <- Sys.getenv("VINE_MODEL", "nn_dynamic_t_vine")
nn_vine_epochs <- as.integer(Sys.getenv("NN_VINE_EPOCHS", "200"))
nn_vine_lr <- as.numeric(Sys.getenv("NN_VINE_LR", "0.001"))
nn_vine_patience <- as.integer(Sys.getenv("NN_VINE_PATIENCE", "20"))
nn_vine_model_dir <- Sys.getenv("NN_VINE_MODEL_DIR", "data/nn_vine_models")
hidden <- as.integer(Sys.getenv("HIDDEN", "128")); num_layers <- as.integer(Sys.getenv("NUM_LAYERS", "2"))
benchmark_file <- Sys.getenv("BENCHMARK_RESULTS_FILE", "data/benchmark_results.RData")
training_marginals_file <- Sys.getenv("TRAINING_MARGINALS_FILE", "data/training_marginal_results.RData")
if (!nzchar(eval_model_dir) || any(is.na(c(eval_seed, eval_gamma, eval_lambda, eval_kappa, L, ref_col, n_sim_cvar, seq_len, holding_days, hidden, num_layers)))) stop("Missing evaluation configuration.")

set.seed(eval_seed)
if (!file.exists(training_marginals_file)) stop(sprintf("Training-only marginal file not found: %s\nRun rl/synthetic_returns.r and retrain first.", training_marginals_file))
load(training_marginals_file)
returns <- load_returns()
T_eval <- as.integer(Sys.getenv("EVALUATION_PERIODS", "24"))
if (T_eval != 24L) stop("The locked historical evaluation must contain exactly 24 monthly holding periods.")
period_split <- split_monthly_periods(
  build_monthly_periods(returns, min_history = L), T_eval
)
validate_period_split(period_split, T_eval)
if (development_dry_run) {
  if (nrow(period_split$train) <= T_eval + seq_len) {
    stop("Insufficient training-prefix periods for the evaluation dry run.")
  }
  # Exercise the complete evaluation machinery on a development-only pseudo
  # holdout. No row from the final locked evaluation window is selected.
  eval_periods <- tail(period_split$train, T_eval)
  train_periods <- head(period_split$train, -T_eval)
  locked_training_end <- max(period_split$train$holding_end_date)
  returns <- returns[paste0("/", locked_training_end)]
  eval_window_id <- paste0(eval_window_id, "_development_dry_run")
  cat("Development-only evaluation dry run: final locked periods are not scored.\n")
} else {
  eval_periods <- period_split$evaluation
  train_periods <- period_split$train
}
eval_dates <- eval_periods$decision_date
if (!identical(vine_model, "nn_dynamic_t_vine")) stop("Evaluation supports only the NN-driven dynamic t-vine; rolling vines are disabled.")
source("helper/marginals.r")
nn_fit <- load_nn_dynamic_vine_fit(nn_vine_model_dir)
if (!exists("copula_innovation_u") || !exists("copula_monthly_log") ||
    !exists("serial_copulas")) {
  stop("Training marginal artifact predates the monthly serial-PIT vine. Regenerate synthetic data and retrain.")
}
if (as.integer(nn_fit$training_observations) != nrow(copula_innovation_u) ||
    as.integer(nn_fit$dynamic_edge_count) != length(asset_names) * (length(asset_names) - 1L) / 2L) {
  stop("Persisted NN vine does not match the locked training split/all-tree architecture. Regenerate it.")
}
if (nrow(train_periods) < seq_len) stop("Not enough pre-evaluation periods for the LSTM burn-in.")
burnin_periods <- tail(train_periods, seq_len)
context_dates <- c(burnin_periods$decision_date, eval_dates)

# Extend the training-only serial-PIT innovation sequence with realised returns
# that are available before each OOS decision.  The empirical marginal maps and
# serial-copula parameters remain frozen at the training cutoff; no OOS return
# is used before its holding period has completed.
all_monthly_periods <- build_monthly_periods(returns, min_history = 0L)
all_monthly_log <- do.call(rbind, lapply(seq_len(nrow(all_monthly_periods)), function(i) {
  log(as.numeric(realised_gross_for_period(
    returns, all_monthly_periods$decision_date[i],
    all_monthly_periods$holding_end_date[i])))
}))
colnames(all_monthly_log) <- asset_names
training_sorted_for_pit <- lapply(seq_along(asset_names), function(j)
  sort(copula_monthly_log[, j]))
raw_monthly_u <- vapply(seq_along(asset_names), function(j) {
  sorted_log <- training_sorted_for_pit[[j]]
  probabilities <- (seq_len(length(sorted_log)) - 0.5) / length(sorted_log)
  pmin(pmax(approx(sorted_log, probabilities, xout = all_monthly_log[, j],
                   rule = 2, ties = "ordered")$y, 1e-6), 1 - 1e-6)
}, numeric(nrow(all_monthly_log)))
colnames(raw_monthly_u) <- asset_names
serial_conditional_pit <- function(previous_u, current_u, model) {
  rho <- model$rho; nu <- model$nu
  previous_t <- qt(pmin(pmax(previous_u, 1e-8), 1 - 1e-8), df = nu)
  current_t <- qt(pmin(pmax(current_u, 1e-8), 1 - 1e-8), df = nu)
  conditional_scale <- sqrt((nu + previous_t^2) * (1 - rho^2) / (nu + 1))
  pt((current_t - rho * previous_t) / conditional_scale, df = nu + 1)
}
all_innovation_u <- matrix(NA_real_, nrow(raw_monthly_u) - 1L,
                           ncol(raw_monthly_u), dimnames = list(NULL, asset_names))
for (j in seq_along(asset_names)) {
  all_innovation_u[, j] <- serial_conditional_pit(
    head(raw_monthly_u[, j], -1L), tail(raw_monthly_u[, j], -1L),
    serial_copulas[[j]])
}
all_innovation_u <- pmin(pmax(all_innovation_u, 1e-6), 1 - 1e-6)
all_innovation_dates <- all_monthly_periods$holding_end_date[-1L]
all_nn_states <- derive_nn_states(all_innovation_u)
vine_seq_context <- build_nn_vine_sequence(
  nn_fit, all_innovation_u, all_nn_states$z, all_nn_states$sigma,
  context_dates, all_innovation_dates)
vine_seq_burnin <- vine_seq_context[seq.int(1L, seq_len)]
vine_seq_eval <- vine_seq_context[seq.int(seq_len + 1L, seq_len + T_eval)]
if (length(vine_seq_eval) != T_eval) stop("Could not build every NN-vine evaluation snapshot.")

# Each row 1 is the historical realised monthly gross return; remaining rows
# are only for the ex-ante CVaR feature and penalty.  This exactly aligns the
# realised wealth series with the common research protocol. Scenario marginals
# use the same training-only monthly empirical transform as pre-training.
training_gross <- do.call(rbind, lapply(seq_len(nrow(train_periods)), function(i) {
  as.numeric(realised_gross_for_period(
    returns, train_periods$decision_date[i], train_periods$holding_end_date[i]
  ))
}))
colnames(training_gross) <- asset_names
training_log <- log(training_gross)
historical_log_sorted <- lapply(seq_along(asset_names), function(j) sort(training_log[, j]))
simulate_evaluation_scenarios <- function(vine, n_draws, previous_log_returns) {
  u <- rvinecop(n_draws, vine, cores = 1L)
  previous_u <- matrix(vapply(seq_along(asset_names), function(j) {
    sorted_log <- historical_log_sorted[[j]]
    probabilities <- (seq_len(length(sorted_log)) - 0.5) / length(sorted_log)
    approx(sorted_log, probabilities, xout = previous_log_returns[j],
           rule = 2, ties = "ordered")$y
  }, numeric(1)), nrow = 1L)
  previous_u <- previous_u[rep(1L, n_draws), , drop = FALSE]
  for (j in seq_along(asset_names)) {
    rho <- serial_copulas[[j]]$rho; nu <- serial_copulas[[j]]$nu
    previous_t <- qt(pmin(pmax(previous_u[, j], 1e-8), 1 - 1e-8), df = nu)
    conditional_t <- rho * previous_t +
      sqrt((nu + previous_t^2) * (1 - rho^2) / (nu + 1)) *
      qt(pmin(pmax(u[, j], 1e-8), 1 - 1e-8), df = nu + 1)
    u[, j] <- pt(conditional_t, df = nu)
  }
  out <- matrix(NA_real_, nrow = n_draws, ncol = length(asset_names))
  for (j in seq_along(asset_names)) {
    sorted_log <- historical_log_sorted[[j]]
    probabilities <- (seq_len(length(sorted_log)) - 0.5) / length(sorted_log)
    out[, j] <- exp(approx(probabilities, sorted_log, xout = u[, j], rule = 2)$y)
  }
  out
}
previous_returns <- training_log[nrow(training_log), ]
eval_steps <- vector("list", T_eval)
for (t in seq_len(T_eval)) {
  actual_gross <- realised_gross_for_period(
    returns, eval_periods$decision_date[t], eval_periods$holding_end_date[t]
  )
  scenarios <- simulate_evaluation_scenarios(
    vine_seq_eval[[t]], n_sim_cvar, previous_returns
  )
  eval_steps[[t]] <- rbind(as.numeric(actual_gross), scenarios)
  previous_returns <- log(pmax(actual_gross, 1e-12))
}

burnin_returns <- lapply(seq_len(nrow(burnin_periods)), function(i) {
  as.numeric(realised_gross_for_period(returns, burnin_periods$decision_date[i],
                                       burnin_periods$holding_end_date[i]))
})
evaluation_episode <- list(
  burnin_returns = burnin_returns,
  burnin_vine_states = lapply(vine_seq_burnin, extract_vine_state),
  returns = eval_steps, vine_states = lapply(vine_seq_eval, extract_vine_state),
  vine_start = 1L, source = "historical_oos")

environment_arguments <- list(
  marginals = marginals, asset_names = asset_names, vine = NULL,
  vine_sequence = vine_seq_eval, ref_col = ref_col, gamma = eval_gamma,
  lambda = eval_lambda, kappa = eval_kappa, T = T_eval, w0 = 100000,
  n_sim_cvar = n_sim_cvar, seq_len = seq_len, sim_cores = 1L,
  holding_days = holding_days, gross_leverage = gross_leverage,
  net_exposure = net_exposure,
  max_long_weight = as.numeric(Sys.getenv("ENV_MAX_LONG_WEIGHT", "0.60")),
  max_short_weight = as.numeric(Sys.getenv("ENV_MAX_SHORT_WEIGHT", "0.20")),
  short_borrow_rate = short_borrow_rate,
  cash_borrow_rate = cash_borrow_rate, utility_mode = utility_mode
)
environment_initialiser <- RLEnvironment$public_methods$initialize
supports_vine_observation_mode <-
  "vine_observation_mode" %in% names(formals(environment_initialiser))
if (supports_vine_observation_mode) {
  environment_arguments$vine_observation_mode <- vine_observation_mode
} else if (!identical(vine_observation_mode, "full")) {
  stop("This RLEnvironment predates explicit vine-signal masking and cannot evaluate a no-vine policy.")
} else {
  cat("Using legacy full-vine RLEnvironment constructor; ablation-only mode argument omitted.\n")
}
env_eval <- do.call(RLEnvironment$new, environment_arguments)
env_eval$set_precomputed_returns(list(evaluation_episode))
policy_python <- Sys.getenv("POLICY_PYTHON", Sys.getenv("RETICULATE_PYTHON", ""))
if (!nzchar(policy_python)) policy_python <- Sys.which("python3")
if (!nzchar(policy_python) || !file.exists(policy_python)) {
  stop("A valid POLICY_PYTHON or RETICULATE_PYTHON is required for isolated policy inference.")
}
policy_server <- normalizePath("rl/policy_inference_server.py", mustWork = TRUE)
repo_root <- normalizePath(".", mustWork = TRUE)
obs_dim <- as.integer(env_eval$get_obs_dim())
action_dim <- as.integer(env_eval$get_action_dim())

wait_for_policy_file <- function(path, error_file, timeout_seconds = 180) {
  deadline <- Sys.time() + timeout_seconds
  repeat {
    if (file.exists(error_file)) {
      stop(paste(readLines(error_file, warn = FALSE), collapse = "\n"))
    }
    if (file.exists(path)) return(invisible(path))
    if (Sys.time() >= deadline) {
      stop(sprintf("Timed out waiting for isolated policy server file: %s", path))
    }
    Sys.sleep(0.02)
  }
}

run_isolated_policy <- function(checkpoint, model) {
  checkpoint <- normalizePath(checkpoint, mustWork = TRUE)
  ipc_dir <- tempfile(sprintf("policy_%s_", model))
  dir.create(ipc_dir, recursive = TRUE)
  stop_file <- file.path(ipc_dir, "STOP")
  error_file <- file.path(ipc_dir, "ERROR.txt")
  on.exit({
    if (dir.exists(ipc_dir)) {
      file.create(stop_file)
      Sys.sleep(0.05)
      unlink(ipc_dir, recursive = TRUE, force = TRUE)
    }
  }, add = TRUE)
  status <- system2(
    policy_python,
    c(policy_server, "--checkpoint", checkpoint, "--ipc-dir", ipc_dir,
      "--repo-root", repo_root, "--obs-dim", obs_dim,
      "--action-dim", action_dim, "--seq-len", seq_len),
    stdout = file.path(ipc_dir, "server.stdout.txt"),
    stderr = file.path(ipc_dir, "server.stderr.txt"), wait = FALSE
  )
  if (!identical(as.integer(status), 0L)) {
    stop(sprintf("Could not launch isolated policy server for %s.", model))
  }
  ready_file <- file.path(ipc_dir, "READY.json")
  wait_for_policy_file(ready_file, error_file)
  cat("Isolated policy server:",
      paste(readLines(ready_file, warn = FALSE), collapse = ""), "\n")

  env_eval$reset()
  state <- as.matrix(env_eval$get_history())
  if (!identical(dim(state), c(seq_len, obs_dim))) {
    stop(sprintf("Unexpected evaluation state shape: %s", paste(dim(state), collapse = "x")))
  }
  rows <- vector("list", T_eval)
  completed <- 0L
  for (step in seq_len(T_eval)) {
    request_file <- file.path(ipc_dir, sprintf("request_%04d.csv", step))
    temporary_request <- paste0(request_file, ".tmp")
    write.table(state, temporary_request, sep = ",", row.names = FALSE,
                col.names = FALSE, quote = FALSE)
    if (!file.rename(temporary_request, request_file)) {
      stop("Could not atomically publish a policy inference request.")
    }
    response_file <- file.path(ipc_dir, sprintf("response_%04d.csv", step))
    wait_for_policy_file(response_file, error_file)
    action <- scan(response_file, sep = ",", quiet = TRUE)
    unlink(response_file)
    if (length(action) != action_dim || any(!is.finite(action))) {
      stop(sprintf("Policy server returned an invalid action at step %d.", step))
    }
    result <- env_eval$step(action)
    info <- result$info
    realised_weights <- as.numeric(info$weights)
    row <- data.frame(
      step = step, wealth = as.numeric(info$wealth),
      gross_return = as.numeric(info$portf_ret) - 1,
      net_return = as.numeric(info$net_portf_ret) - 1,
      cvar = as.numeric(info$cvar), turnover = as.numeric(info$turnover),
      transaction_cost = as.numeric(info$transaction_cost),
      financing_cost = as.numeric(info$financing_cost),
      utility = as.numeric(info$utility), reward = as.numeric(result$reward),
      stringsAsFactors = FALSE
    )
    for (j in seq_along(realised_weights)) row[[paste0("w", j)]] <- realised_weights[j]
    row$model <- model
    rows[[step]] <- row
    completed <- step
    observation <- as.numeric(result$observation)
    state <- if (seq_len > 1L) {
      rbind(state[-1L, , drop = FALSE], observation)
    } else {
      matrix(observation, nrow = 1L)
    }
    if (isTRUE(result$done)) break
  }
  file.create(stop_file)
  wait_for_policy_file(file.path(ipc_dir, "DONE"), error_file)
  if (completed != T_eval) stop("Historical policy evaluation ended before 24 periods.")
  rbindlist(rows[seq_len(completed)], fill = TRUE)
}

if (nzchar(eval_checkpoint_models)) {
  models_to_run <- unique(trimws(strsplit(
    eval_checkpoint_models, ",", fixed = TRUE)[[1L]]))
  models_to_run <- models_to_run[nzchar(models_to_run)]
  unsupported_models <- setdiff(models_to_run, c("pretrained", "full"))
  if (!length(models_to_run) || length(unsupported_models)) {
    stop(sprintf(
      "EVAL_CHECKPOINT_MODELS must contain only pretrained/full; got: %s",
      paste(models_to_run, collapse = ",")))
  }
} else {
  # Backward-compatible default used by the frozen v4 evaluation. Secondary
  # post-holdout experiments set EVAL_CHECKPOINT_MODELS explicitly so the
  # pre-fine-tuning checkpoint can emit weights without rescoring the full one.
  models_to_run <- if (eval_weights_only) "full" else c("pretrained", "full")
}
all_logs <- as.data.frame(rbindlist(lapply(models_to_run, function(model) {
  run_isolated_policy(file.path(eval_model_dir,
                                paste0("td3_lstm_vine_", model, ".pt")), model)
}), fill = TRUE))
run_name <- basename(eval_model_dir)
all_logs$decision_date <- eval_periods$decision_date[all_logs$step]
all_logs$holding_end_date <- eval_periods$holding_end_date[all_logs$step]
all_logs$window_id <- eval_window_id
for (j in seq_along(asset_names)) names(all_logs)[names(all_logs) == paste0("w", j)] <- paste0("w_", asset_names[j])
for (model in models_to_run) {
  weight_columns <- c("window_id", "decision_date", "holding_end_date",
                      paste0("w_", asset_names))
  weight_file <- file.path(
    eval_output_dir, paste0("weights_rl_", model, "_", run_name, ".csv"))
  write.csv(all_logs[all_logs$model == model, weight_columns, drop = FALSE],
            weight_file, row.names = FALSE)
}
if (eval_weights_only) {
  cat("Historical weight generation complete for", run_name, "\n")
} else {
source("eval/ablation.r")
rl_rows <- lapply(c("pretrained", "full"), function(model) {
  returns_model <- all_logs$net_return[all_logs$model == model]
  data.frame(Strategy = paste("RL", model),
             t(annualised_path_metrics(returns_model)), check.names = FALSE)
})
rl_df <- rbindlist(rl_rows, fill = TRUE)
comparison_table <- rl_df
save(comparison_table, file = file.path(
  eval_output_dir, paste0("evaluation_comparison_", run_name, ".RData")))
write.csv(comparison_table, file.path(
  eval_output_dir, paste0("evaluation_comparison_", run_name, ".csv")), row.names = FALSE)
write.csv(all_logs, file.path(
  eval_output_dir, paste0("evaluation_logs_", run_name, ".csv")), row.names = FALSE)

plot_df <- rbindlist(lapply(c("pretrained", "full"), function(model) data.table(step = 0:T_eval, wealth = c(100000, all_logs$wealth[all_logs$model == model]), model = model)))
ggsave(paste0("figures/wealth_curves_rl_evaluation_", run_name, ".pdf"), ggplot(plot_df, aes(step, wealth, colour = model)) + geom_line(linewidth = 1) + theme_bw() + labs(title = "Historical out-of-sample RL wealth", x = "Month", y = "Wealth"), width = 9, height = 5)
weight_cols <- grep("^w_", names(all_logs), value = TRUE)
weights_long <- melt(as.data.table(all_logs), id.vars = c("model", "step"), measure.vars = weight_cols, variable.name = "asset", value.name = "weight")
ggsave(paste0("figures/weights_evolution_", run_name, ".pdf"), ggplot(weights_long, aes(step, weight, colour = asset)) + geom_line() + facet_wrap(~model) + theme_bw(), width = 9, height = 5)
cat("Historical evaluation complete. Results saved for", run_name, "\n")
}
