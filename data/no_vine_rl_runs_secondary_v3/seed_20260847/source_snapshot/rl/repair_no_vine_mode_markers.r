#!/usr/bin/env Rscript
# Recover missing mode-marker files only from independently verifiable evidence.
# This script never modifies checkpoints, metrics, manifests, or prior statuses.

suppressPackageStartupMessages({
  library(jsonlite)
  library(yaml)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: repair_no_vine_mode_markers.r SWEEP_ROOT SEED_SPECIFICATION")
}
sweep_root <- normalizePath(args[[1L]], winslash = "/", mustWork = TRUE)
seed_file <- normalizePath(args[[2L]], winslash = "/", mustWork = TRUE)
training_python <- Sys.getenv("TRAIN_PYTHON", unset = "")
if (!nzchar(training_python) || !file.exists(training_python)) {
  stop("Set TRAIN_PYTHON to the exact GPU training interpreter.")
}
seed_spec <- yaml::yaml.load_file(seed_file)
raw_seeds <- unname(unlist(seed_spec$seeds, use.names = FALSE))
seeds <- unique(as.integer(raw_seeds))
raw_mode <- unname(unlist(seed_spec$vine_observation_mode, use.names = FALSE))
expected_mode <- if (length(raw_mode) == 1L) {
  tolower(trimws(as.character(raw_mode[[1L]])))
} else {
  ""
}
expected_mask <- "explicit_vine_and_scenario_cvar_v1"
frozen_seeds <- 20260841:20260850
if (!identical(expected_mode, "zero") || anyNA(seeds) ||
    length(seeds) != length(frozen_seeds) ||
    !setequal(seeds, frozen_seeds)) {
  stop("The seed specification is not the frozen ten-seed zero-vine design; ",
       "parsed mode=", dQuote(expected_mode), ", parsed seeds=",
       paste(seeds, collapse = ","), ".")
}

helper <- normalizePath(
  "publication_pipeline_draft/verify_no_vine_training_evidence.py",
  winslash = "/", mustWork = TRUE)
checkpoint_json <- tempfile(fileext = ".json")
checkpoint_stderr <- tempfile(fileext = ".stderr")
on.exit(unlink(c(checkpoint_json, checkpoint_stderr)), add = TRUE)
status <- system2(
  training_python,
  c(shQuote(helper), "--sweep-root", shQuote(sweep_root),
    "--seeds", shQuote(paste(seeds, collapse = ","))),
  stdout = checkpoint_json, stderr = checkpoint_stderr)
if (!identical(status, 0L)) {
  stop("Embedded checkpoint evidence failed:\n",
       paste(readLines(checkpoint_stderr, warn = FALSE), collapse = "\n"))
}
checkpoint_evidence <- jsonlite::fromJSON(checkpoint_json, simplifyVector = FALSE)
allowed_checkpoint_status <- c(
  "valid_embedded_no_vine_checkpoint_evidence",
  "valid_checkpoint_files_with_legacy_missing_mode_metadata")
if (!checkpoint_evidence$status %in% allowed_checkpoint_status) {
  stop("Checkpoint verifier did not return the required valid status.")
}

manifest_rows <- vector("list", length(seeds))
worker_log_files <- list.files(
  file.path(sweep_root, "worker_logs"),
  pattern = "^worker_[0-9]+\\.log$", full.names = TRUE)
if (!length(worker_log_files)) {
  stop("Legacy recovery requires the immutable worker logs, but none were found in ",
       file.path(sweep_root, "worker_logs"), ".")
}
worker_logs <- lapply(worker_log_files, function(path) {
  readLines(path, warn = FALSE, encoding = "UTF-8")
})
names(worker_logs) <- worker_log_files

locate_seed_log_evidence <- function(seed) {
  training_token <- paste("Training seed", seed)
  warning_token <- sprintf(
    "Seed %d wrote vine_observation_mode=''; expected 'zero'. Marking training invalid.",
    seed)
  containing_training <- names(Filter(
    function(lines) any(grepl(training_token, lines, fixed = TRUE)), worker_logs))
  containing_warning <- names(Filter(
    function(lines) any(grepl(warning_token, lines, fixed = TRUE)), worker_logs))
  if (length(containing_training) != 1L || length(containing_warning) != 1L ||
      !identical(containing_training, containing_warning)) {
    stop("Worker-log evidence is missing, duplicated, or inconsistent for seed ",
         seed, ".")
  }
  path <- containing_training[[1L]]
  lines <- worker_logs[[path]]
  if (!any(grepl("Sweep vine observation mode: zero", lines, fixed = TRUE))) {
    stop("Worker log does not record zero mode for seed ", seed, ": ", path)
  }
  starts <- which(grepl(training_token, lines, fixed = TRUE))
  if (length(starts) != 1L) {
    stop("Worker log does not contain exactly one training launch for seed ", seed)
  }
  next_starts <- which(seq_along(lines) > starts[[1L]] &
                         grepl("Training seed ", lines, fixed = TRUE))
  finish <- if (length(next_starts)) next_starts[[1L]] - 1L else length(lines)
  block <- lines[starts[[1L]]:finish]
  if (!any(grepl("TRAINING COMPLETE", block, fixed = TRUE))) {
    stop("Worker log does not prove completed training for seed ", seed)
  }
  list(
    path = normalizePath(path, winslash = "/", mustWork = TRUE),
    md5 = unname(tools::md5sum(path)),
    zero_mode_banner_verified = TRUE,
    unique_training_launch_verified = TRUE,
    training_complete_verified = TRUE,
    missing_marker_warning_verified = TRUE)
}

for (index in seq_along(seeds)) {
  seed <- seeds[[index]]
  directory <- file.path(sweep_root, paste0("seed_", seed))
  manifest_file <- file.path(directory, "run_manifest.rds")
  if (!file.exists(manifest_file)) stop("Missing run manifest: ", manifest_file)
  manifest <- readRDS(manifest_file)
  recorded_seed <- as.integer(manifest$seed)
  environment <- manifest$relevant_environment
  has_recorded_mode <- !is.null(environment) &&
    "VINE_OBSERVATION_MODE" %in% names(environment) &&
    length(environment["VINE_OBSERVATION_MODE"]) == 1L &&
    nzchar(unname(as.character(environment["VINE_OBSERVATION_MODE"])))
  recorded_mode <- if (has_recorded_mode) {
    tolower(trimws(unname(as.character(
      environment["VINE_OBSERVATION_MODE"]))))
  } else {
    NA_character_
  }
  if (!identical(recorded_seed, seed)) {
    stop("Run-manifest seed mismatch for seed ", seed,
         ": recorded seed=", recorded_seed)
  }
  if (has_recorded_mode && !identical(recorded_mode, expected_mode)) {
    stop("Run-manifest mode conflicts with the frozen intervention for seed ",
         seed, ": recorded mode=", recorded_mode)
  }
  if (is.null(manifest$source_snapshot) || !is.data.frame(manifest$source_snapshot)) {
    stop("Run manifest lacks a source snapshot for seed ", seed)
  }
  critical <- c(
    "helper/reproducibility.r", "rl/run_seed_sweep.r", "run_with_config.r",
    "rl/train_rl.r", "rl/rl_environment.r")
  normalized_sources <- gsub("\\\\", "/", manifest$source_snapshot$source)
  snapshot_text <- list()
  for (relative in critical) {
    matches <- which(endsWith(normalized_sources, relative))
    if (length(matches) != 1L) {
      stop("Source snapshot does not uniquely contain ", relative,
           " for seed ", seed)
    }
    snapshot <- manifest$source_snapshot$snapshot[[matches]]
    if (!file.exists(snapshot)) stop("Missing snapshotted source: ", snapshot)
    hash_rows <- which(endsWith(
      gsub("\\\\", "/", manifest$code_hashes$path), relative))
    if (length(hash_rows) != 1L ||
        !identical(unname(tools::md5sum(snapshot)),
                   as.character(manifest$code_hashes$md5[[hash_rows]]))) {
      stop("Snapshotted source hash mismatch for ", relative,
           " at seed ", seed)
    }
    snapshot_text[[relative]] <- paste(
      readLines(snapshot, warn = FALSE), collapse = "\n")
  }
  manifest_writer_has_mode <- grepl(
    '"VINE_OBSERVATION_MODE"',
    snapshot_text[["helper/reproducibility.r"]], fixed = TRUE)
  if (!has_recorded_mode && manifest_writer_has_mode) {
    stop("Seed ", seed, " has a missing manifest mode even though its hash-",
         "matched manifest writer records that field; recovery refuses this ",
         "internal inconsistency.")
  }
  if (has_recorded_mode && !manifest_writer_has_mode) {
    stop("Seed ", seed, " records a manifest mode that its hash-matched ",
         "manifest writer does not support.")
  }
  runner_text <- snapshot_text[["rl/run_seed_sweep.r"]]
  launcher_text <- snapshot_text[["run_with_config.r"]]
  training_text <- snapshot_text[["rl/train_rl.r"]]
  environment_text <- snapshot_text[["rl/rl_environment.r"]]

  # Older launchers did not mention VINE_OBSERVATION_MODE at all. That is a
  # valid passive-inheritance path: system2() places the variable in the child
  # R process and source("rl/train_rl.r") keeps the process environment. If a
  # launcher does mention the variable, it must use the non-overwriting
  # set_default_env path. A direct Sys.setenv assignment is never attested.
  launcher_mentions_mode <- grepl(
    "VINE_OBSERVATION_MODE", launcher_text, fixed = TRUE)
  launcher_direct_override <- grepl(
    "Sys\\.setenv\\s*\\(\\s*VINE_OBSERVATION_MODE\\s*=",
    launcher_text, perl = TRUE)
  launcher_safe_default <-
    grepl("set_default_env\\s*<-\\s*function\\s*\\(",
          launcher_text, perl = TRUE) &&
    grepl("Sys\\.getenv\\s*\\(\\s*name", launcher_text, perl = TRUE) &&
    grepl("set_default_env\\s*\\(\\s*[\"']VINE_OBSERVATION_MODE[\"']",
          launcher_text, perl = TRUE)
  launcher_pass_through_proven <-
    (!launcher_mentions_mode) ||
    (launcher_safe_default && !launcher_direct_override)

  proof_checks <- c(
    runner_sets_child_mode = grepl(
      "VINE_OBSERVATION_MODE=", runner_text, fixed = TRUE),
    runner_supplies_system2_environment = grepl(
      "env\\s*=\\s*environment", runner_text, perl = TRUE),
    launcher_preserves_or_passively_inherits_mode = launcher_pass_through_proven,
    trainer_reads_mode_from_process_environment = grepl(
      "Sys\\.getenv\\s*\\(\\s*[\"']VINE_OBSERVATION_MODE[\"']",
      training_text, perl = TRUE),
    trainer_passes_mode_to_environment = grepl(
      "vine_observation_mode\\s*=\\s*vine_observation_mode",
      training_text, perl = TRUE),
    environment_activates_zero_mode = grepl(
      "no_vine_observation\\s*<-\\s*identical\\s*\\(\\s*private\\$vine_observation_mode\\s*,\\s*[\"']zero[\"']\\s*\\)",
      environment_text, perl = TRUE),
    environment_zeros_cvar_observation = grepl(
      "cvar_observation\\s*<-\\s*if\\s*\\(\\s*no_vine_observation\\s*\\)\\s*0",
      environment_text, perl = TRUE))
  if (any(!proof_checks)) {
    stop(
      "Snapshotted source does not prove the frozen zero-mode masking path for seed ",
      seed, "; failed checks: ",
      paste(names(proof_checks)[!proof_checks], collapse = ", "),
      ". launcher_mentions_mode=", launcher_mentions_mode,
      ", launcher_direct_override=", launcher_direct_override,
      ", launcher_safe_default=", launcher_safe_default, ".")
  }
  worker_evidence <- locate_seed_log_evidence(seed)
  marker <- file.path(directory, "vine_observation_mode.txt")
  if (file.exists(marker)) {
    existing <- trimws(readLines(marker, warn = FALSE, n = 1L))
    if (!identical(existing, expected_mode)) {
      stop("Existing mode marker conflicts with evidence for seed ", seed)
    }
  }
  manifest_rows[[index]] <- list(
    seed = seed,
    run_manifest = manifest_file,
    run_manifest_md5 = unname(tools::md5sum(manifest_file)),
    recorded_environment_mode = recorded_mode,
    manifest_mode_evidence = if (has_recorded_mode) {
      "recorded_zero"
    } else {
      "legacy_field_absent_hash_verified_launcher_and_worker_log"
    },
    legacy_manifest_writer_omits_mode = !manifest_writer_has_mode,
    source_snapshot_hashes_verified = TRUE,
    source_path_proof_checks = as.list(proof_checks),
    launcher_mode_handling = if (!launcher_mentions_mode) {
      "passive_process_environment_inheritance"
    } else {
      "non_overwriting_set_default_env"
    },
    worker_log_evidence = worker_evidence)
}

created_utc <- format(Sys.time(), tz = "UTC", usetz = TRUE)
checkpoint_rows <- checkpoint_evidence$checkpoints
for (seed in seeds) {
  directory <- file.path(sweep_root, paste0("seed_", seed))
  seed_checkpoints <- Filter(
    function(row) identical(as.integer(row$seed), seed), checkpoint_rows)
  if (length(seed_checkpoints) != 2L) {
    stop("Expected two verified checkpoints for seed ", seed)
  }
  repair_record <- list(
    schema_version = 1L,
    repair_type = "post_hoc_missing_plaintext_mode_marker_reconstruction",
    scientific_model_or_checkpoint_changed = FALSE,
    created_utc = created_utc,
    seed = seed,
    reconstructed_value = expected_mode,
    reason = paste(
      "Training completed under the frozen zero-mode seed specification.",
      "For a legacy manifest writer that omitted the mode field, the mode is",
      "instead proven by the hash-matched sweep runner, launcher, trainer and",
      "environment source chain plus the immutable worker log's zero-mode",
      "banner, unique seed launch, completion banner and missing-marker warning.",
      "The redundant plaintext marker was absent; checkpoint bytes are",
      "unchanged and their hashes and metadata status are bound below."),
    manifest_and_worker_evidence = manifest_rows[[which(seeds == seed)]],
    checkpoint_evidence = seed_checkpoints)
  record_path <- file.path(directory, "vine_observation_mode_repair.json")
  if (file.exists(record_path)) stop("Repair record already exists: ", record_path)
  record_temp <- paste0(record_path, ".tmp")
  jsonlite::write_json(repair_record, record_temp, auto_unbox = TRUE,
                       pretty = TRUE, null = "null")
  if (!file.rename(record_temp, record_path)) stop("Could not publish ", record_path)
  marker_path <- file.path(directory, "vine_observation_mode.txt")
  if (!file.exists(marker_path)) {
    marker_temp <- paste0(marker_path, ".tmp")
    writeLines(expected_mode, marker_temp, useBytes = TRUE)
    if (!file.rename(marker_temp, marker_path)) stop("Could not publish ", marker_path)
  }
}

root_audit <- list(
  schema_version = 1L,
  status = "validated_post_hoc_mode_marker_recovery_complete",
  evidence_class = "operational_recovery_not_new_empirical_result",
  scientific_model_or_checkpoint_changed = FALSE,
  created_utc = created_utc,
  sweep_root = sweep_root,
  expected_mode = expected_mode,
  expected_signal_mask = expected_mask,
  recovery_rule = paste(
    "A manifest-recorded zero mode is accepted directly. A legacy missing",
    "field is accepted only when the snapshotted legacy manifest writer omits",
    "that field and the hash-bound runner-to-launcher-to-trainer-to-environment",
    "source chain and per-seed worker log independently prove zero mode and",
    "completed training."),
  manifests = manifest_rows,
  checkpoint_evidence = checkpoint_evidence)
audit_path <- file.path(sweep_root, "mode_marker_recovery_audit.json")
if (file.exists(audit_path)) stop("Recovery audit already exists: ", audit_path)
audit_temp <- paste0(audit_path, ".tmp")
jsonlite::write_json(root_audit, audit_temp, auto_unbox = TRUE,
                     pretty = TRUE, null = "null")
if (!file.rename(audit_temp, audit_path)) stop("Could not publish recovery audit.")
cat("Validated and reconstructed ten redundant mode markers without modifying checkpoints.\n")
