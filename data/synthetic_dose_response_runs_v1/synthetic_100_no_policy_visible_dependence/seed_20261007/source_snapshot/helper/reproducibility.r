# Reproducibility/provenance helpers.

file_hashes <- function(paths) {
  paths <- unique(paths[file.exists(paths)])
  data.frame(path = normalizePath(paths, winslash = "/", mustWork = TRUE),
             md5 = unname(tools::md5sum(paths)), row.names = NULL)
}

safe_git_state <- function(project_root = ".") {
  git_dir <- file.path(project_root, ".git")
  head_file <- file.path(git_dir, "HEAD")
  if (!file.exists(head_file)) return(list(head = NA_character_, note = "not a Git worktree"))
  head_value <- readLines(head_file, warn = FALSE, n = 1L)
  if (startsWith(head_value, "ref: ")) {
    reference <- sub("^ref: ", "", head_value)
    ref_file <- file.path(git_dir, reference)
    resolved <- if (file.exists(ref_file)) readLines(ref_file, warn = FALSE, n = 1L) else NA_character_
  } else resolved <- head_value
  list(head = resolved, reference = head_value,
       note = "Non-destructive manifest read; run git fsck separately.")
}

write_run_manifest <- function(output_dir, seed, config_file = "config/config.yaml",
                               data_files = character()) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  vine_observation_mode <- tolower(trimws(Sys.getenv(
    "VINE_OBSERVATION_MODE", unset = "full")))
  if (!vine_observation_mode %in% c("full", "zero")) {
    stop("Cannot write run manifest: invalid VINE_OBSERVATION_MODE=",
         vine_observation_mode)
  }
  vine_feature_mode <- tolower(trimws(Sys.getenv(
    "VINE_FEATURE_MODE", unset = vine_observation_mode)))
  cvar_observation_mode <- tolower(trimws(Sys.getenv(
    "CVAR_OBSERVATION_MODE", unset = vine_observation_mode)))
  cvar_reward_mode <- tolower(trimws(Sys.getenv(
    "CVAR_REWARD_MODE", unset = "full")))
  if (any(!c(vine_feature_mode, cvar_observation_mode, cvar_reward_mode) %in%
          c("full", "zero"))) {
    stop("Cannot write run manifest: invalid independent vine/CVaR modes.")
  }
  no_vine_signal_mask <- if (identical(vine_feature_mode, "zero") &&
                             identical(cvar_observation_mode, "zero")) {
    "explicit_vine_and_scenario_cvar_v1"
  } else {
    "not_applicable"
  }
  code_roots <- c("rl", "helper", "benchmark_models", "eval", "config", "tests")
  code_files <- unlist(lapply(code_roots[dir.exists(code_roots)], function(root) {
    list.files(root, pattern = "\\.(r|R|py|yaml|csv)$", recursive = TRUE,
               full.names = TRUE)
  }), use.names = FALSE)
  code_files <- c(code_files,
    list.files(".", pattern = "\\.(r|R)$", recursive = FALSE, full.names = TRUE))
  # Preserve the exact source bytes used by the run.  Hashes alone revealed a
  # real-world portability problem: a remote Linux run could not be reconciled
  # with several files in the Windows checkout after the fact.
  snapshot_root <- file.path(output_dir, "source_snapshot")
  snapshot_files <- file.path(snapshot_root, sub("^\\./", "", code_files))
  invisible(lapply(unique(dirname(snapshot_files)), dir.create,
                   recursive = TRUE, showWarnings = FALSE))
  copied <- mapply(file.copy, from = code_files, to = snapshot_files,
                   MoreArgs = list(overwrite = TRUE), USE.NAMES = FALSE)
  if (!all(copied)) stop("Could not preserve the complete per-run source snapshot.")
  manifest <- list(
    schema_version = 5L,
    created_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
    seed = as.integer(seed),
    config_file = normalizePath(config_file, winslash = "/", mustWork = FALSE),
    config_hash = if (file.exists(config_file)) unname(tools::md5sum(config_file)) else NA_character_,
    data_hashes = file_hashes(data_files),
    code_hashes = file_hashes(code_files),
    source_snapshot = data.frame(
      source = normalizePath(code_files, winslash = "/", mustWork = TRUE),
      snapshot = normalizePath(snapshot_files, winslash = "/", mustWork = TRUE),
      row.names = NULL),
    r_version = R.version.string,
    session_info = capture.output(sessionInfo()),
    git = safe_git_state("."),
    interventions = list(
      vine_observation_mode = vine_observation_mode,
      no_vine_signal_mask = no_vine_signal_mask,
      vine_feature_mode = vine_feature_mode,
      cvar_observation_mode = cvar_observation_mode,
      cvar_reward_mode = cvar_reward_mode,
      pretrain_data_mode = Sys.getenv("PRETRAIN_DATA_MODE", "vine_synthetic"),
      pretrain_behavior_gate_mode = Sys.getenv(
        "PRETRAIN_BEHAVIOR_GATE_MODE", "strict"),
      rl_algorithm = Sys.getenv("RL_ALGORITHM", "td3"),
      policy_encoder = Sys.getenv("POLICY_ENCODER", "lstm"),
      run_finetune = Sys.getenv("RUN_FINETUNE", "true"),
      checkpoint_prefix = Sys.getenv("CHECKPOINT_PREFIX", "td3_lstm_vine")),
    relevant_environment = Sys.getenv(c(
      "TRAIN_SEED", "TRAIN_DEVICE", "VINE_MODEL", "ENV_T", "ENV_SEQ_LEN",
      "VINE_TRUNCATION_LEVEL",
      "RETURNS_DATA_FILE", "RETURNS_DATA_KIND", "RETURNS_DATA_MANIFEST",
      "VINE_OBSERVATION_MODE", "VINE_FEATURE_MODE",
      "CVAR_OBSERVATION_MODE", "CVAR_REWARD_MODE",
      "PRETRAIN_DATA_MODE", "PRETRAIN_BEHAVIOR_GATE_MODE",
      "RL_ALGORITHM", "POLICY_ENCODER",
      "RUN_FINETUNE", "CHECKPOINT_PREFIX", "SYNTHETIC_RETURNS_FILE",
      "ENV_GAMMA", "ENV_LAMBDA", "ENV_KAPPA", "ENV_UTILITY_MODE",
      "ENV_GROSS_LEVERAGE", "ENV_NET_EXPOSURE", "ENV_SHORT_BORROW_RATE",
      "ENV_MAX_LONG_WEIGHT", "ENV_MAX_SHORT_WEIGHT",
      "ENV_CASH_BORROW_RATE", "POLICY_DELAY", "TARGET_POLICY_NOISE",
      "TARGET_NOISE_CLIP", "PRETRAIN_RANDOM_EXPLORATION_STEPS",
      "PRETRAIN_BEHAVIOR_GATE_WINDOW", "PRETRAIN_MAX_MEAN_LEVERAGE_GATE",
      "PRETRAIN_MAX_MEAN_GROSS_CAP_FRACTION", "PRETRAIN_WARN_POSITION_CAP_FRACTION",
      "PRETRAIN_MIN_MEAN_NORMALIZED_ENTROPY", "PRETRAIN_MIN_Q05_NORMALIZED_ENTROPY",
      "PRETRAIN_MIN_MEAN_EFFECTIVE_POSITIONS", "PRETRAIN_MAX_POSITION_LIMIT_VIOLATION",
      "PRETRAIN_MAX_GATE_GROSS_MAE", "PRETRAIN_MAX_MEAN_TURNOVER",
      "DIRECTION_LOGIT_BOUND", "PROJECTION_TEMPERATURE",
      "INITIAL_LEVERAGE_GATE", "ENTROPY_COEF",
      "LEVERAGE_SOFT_TARGET", "LEVERAGE_PENALTY_COEF",
      "LR_ACTOR", "LR_CRITIC", "PRETRAIN_NOISE_SCALE",
      "PRETRAIN_NOISE_DECAY", "PRETRAIN_UPDATES_PER_STEP",
      "FINETUNE_NOISE_SCALE", "FINETUNE_NOISE_DECAY",
      "FINETUNE_UPDATES_PER_STEP", "DISCOUNT", "TAU",
      "USE_AMP",
      "FINETUNE_RANDOM_EXPLORATION_STEPS", "FINETUNE_LR_ACTOR",
      "FINETUNE_LR_CRITIC", "FINETUNE_MAX_SELECTION_PASSES",
      "FINETUNE_VALIDATION_PATIENCE", "FINETUNE_VALIDATION_MIN_DELTA",
      "DIAGNOSTIC_INTERVAL"
    ))
  )
  saveRDS(manifest, file.path(output_dir, "run_manifest.rds"))
  write.csv(manifest$code_hashes, file.path(output_dir, "code_hashes.csv"),
            row.names = FALSE)
  write.csv(manifest$data_hashes, file.path(output_dir, "data_hashes.csv"),
            row.names = FALSE)
  invisible(manifest)
}
