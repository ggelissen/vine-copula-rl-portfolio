#!/usr/bin/env Rscript
# Training-prefix/checkpoint-only smoke test for isolated Python policy inference.
# No realised holdout panel is loaded.

conda_prefix <- Sys.getenv("CONDA_PREFIX")
if (nzchar(conda_prefix)) {
  python_path <- file.path(conda_prefix, "bin", "python")
  if (file.exists(python_path)) {
    Sys.setenv(RETICULATE_PYTHON = python_path)
  }
}

suppressPackageStartupMessages({
  library(jsonlite)
  library(torch)
  library(yaml)
})

config_text <- readLines("config/config.yaml", encoding = "UTF-8", warn = FALSE)
config <- yaml::yaml.load(paste(config_text, collapse = "\n"))
required_scalar <- function(value, name) {
  if (length(value) != 1L || is.null(value) || is.na(value)) {
    stop(sprintf("Configuration field '%s' must be one non-missing scalar.", name))
  }
  as.character(value)
}
model_dir <- Sys.getenv(
  "POLICY_TEST_MODEL_DIR",
  "frozen_releases/training_schema5_v1/seeds/seed_20260741")
python <- Sys.getenv("POLICY_PYTHON", Sys.getenv("RETICULATE_PYTHON", ""))
if (!nzchar(python)) python <- Sys.which("python3")
if (!file.exists(python)) stop("Set POLICY_PYTHON to the isolated CPU Python executable.")
checkpoint <- normalizePath(file.path(model_dir, "td3_lstm_vine_full.pt"),
                            mustWork = TRUE)
report <- jsonlite::fromJSON(file.path(model_dir, "sanity_no_holdout",
                                       "sanity_report.json"))
seq_len <- as.integer(config$environment$seq_len)
obs_dim <- as.integer(report$obs_dim)
action_dim <- as.integer(report$action_dim)

Sys.setenv(
  HIDDEN = required_scalar(config$agent$hidden, "agent.hidden"),
  NUM_LAYERS = required_scalar(config$agent$num_layers, "agent.num_layers"),
  ENV_GROSS_LEVERAGE = required_scalar(config$environment$gross_leverage,
                                       "environment.gross_leverage"),
  ENV_NET_EXPOSURE = required_scalar(config$environment$net_exposure,
                                     "environment.net_exposure"),
  ENV_MAX_LONG_WEIGHT = required_scalar(config$environment$max_long_weight,
                                        "environment.max_long_weight"),
  ENV_MAX_SHORT_WEIGHT = required_scalar(config$environment$max_short_weight,
                                         "environment.max_short_weight"),
  ENV_SHORT_BORROW_RATE = required_scalar(config$environment$short_borrow_rate,
                                          "environment.short_borrow_rate"),
  ENV_CASH_BORROW_RATE = required_scalar(config$environment$cash_borrow_rate,
                                         "environment.cash_borrow_rate"),
  ENV_UTILITY_MODE = required_scalar(config$environment$utility_mode,
                                     "environment.utility_mode"),
  VINE_OBSERVATION_MODE = "full",
  DIRECTION_LOGIT_BOUND = required_scalar(config$agent$direction_logit_bound,
                                          "agent.direction_logit_bound"),
  PROJECTION_TEMPERATURE = required_scalar(config$agent$projection_temperature,
                                           "agent.projection_temperature"),
  INITIAL_LEVERAGE_GATE = required_scalar(config$agent$initial_leverage_gate,
                                          "agent.initial_leverage_gate"),
  ENTROPY_COEF = required_scalar(config$agent$entropy_coef,
                                 "agent.entropy_coef"),
  LEVERAGE_SOFT_TARGET = required_scalar(config$agent$leverage_soft_target,
                                         "agent.leverage_soft_target"),
  LEVERAGE_PENALTY_COEF = required_scalar(config$agent$leverage_penalty_coef,
                                          "agent.leverage_penalty_coef"),
  USE_AMP = tolower(required_scalar(config$agent$use_amp, "agent.use_amp"))
)

# Force Lantern to be initialized in the parent R process. The Python server
# must remain healthy because it has a separate address space.
stopifnot(abs(as.numeric(torch_tensor(c(1, 2, 3))$sum()$item()) - 6) < 1e-12)

ipc <- tempfile("policy_isolation_test_")
dir.create(ipc)
on.exit({
  if (dir.exists(ipc)) {
    file.create(file.path(ipc, "STOP"))
    Sys.sleep(0.05)
    unlink(ipc, recursive = TRUE, force = TRUE)
  }
}, add = TRUE)
server <- normalizePath("rl/policy_inference_server.py", mustWork = TRUE)
status <- system2(
  python,
  c(server, "--checkpoint", checkpoint, "--ipc-dir", ipc,
    "--repo-root", normalizePath("."), "--obs-dim", obs_dim,
    "--action-dim", action_dim, "--seq-len", seq_len),
  stdout = file.path(ipc, "stdout.txt"),
  stderr = file.path(ipc, "stderr.txt"), wait = FALSE
)
stopifnot(identical(as.integer(status), 0L))

wait_for <- function(path, timeout = 120) {
  deadline <- Sys.time() + timeout
  repeat {
    error_file <- file.path(ipc, "ERROR.txt")
    if (file.exists(error_file)) {
      stop(paste(readLines(error_file, warn = FALSE), collapse = "\n"))
    }
    if (file.exists(path)) return(invisible(path))
    if (Sys.time() >= deadline) stop("Policy isolation smoke test timed out.")
    Sys.sleep(0.02)
  }
}

wait_for(file.path(ipc, "READY.json"))
request <- file.path(ipc, "request_0001.csv")
temporary <- paste0(request, ".tmp")
write.table(matrix(0, nrow = seq_len, ncol = obs_dim), temporary, sep = ",",
            row.names = FALSE, col.names = FALSE, quote = FALSE)
stopifnot(file.rename(temporary, request))
response <- file.path(ipc, "response_0001.csv")
wait_for(response)
weights <- scan(response, sep = ",", quiet = TRUE)
stopifnot(
  length(weights) == action_dim,
  all(is.finite(weights)),
  abs(sum(weights) - as.numeric(config$environment$net_exposure)) <= 1e-6,
  sum(abs(weights)) <= as.numeric(config$environment$gross_leverage) + 1e-6,
  max(weights) <= as.numeric(config$environment$max_long_weight) + 1e-6,
  min(weights) >= -as.numeric(config$environment$max_short_weight) - 1e-6
)
file.create(file.path(ipc, "STOP"))
wait_for(file.path(ipc, "DONE"))
cat("R-Lantern/Python-PyTorch process-isolation test passed.\n")
