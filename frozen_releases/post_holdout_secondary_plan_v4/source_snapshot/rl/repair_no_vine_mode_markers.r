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
for (index in seq_along(seeds)) {
  seed <- seeds[[index]]
  directory <- file.path(sweep_root, paste0("seed_", seed))
  manifest_file <- file.path(directory, "run_manifest.rds")
  if (!file.exists(manifest_file)) stop("Missing run manifest: ", manifest_file)
  manifest <- readRDS(manifest_file)
  recorded_seed <- as.integer(manifest$seed)
  environment <- manifest$relevant_environment
  recorded_mode <- unname(as.character(environment[["VINE_OBSERVATION_MODE"]]))
  if (!identical(recorded_seed, seed) || !identical(recorded_mode, expected_mode)) {
    stop("Run-manifest evidence mismatch for seed ", seed,
         ": seed=", recorded_seed, ", mode=", recorded_mode)
  }
  if (is.null(manifest$source_snapshot) || !is.data.frame(manifest$source_snapshot)) {
    stop("Run manifest lacks a source snapshot for seed ", seed)
  }
  critical <- c("rl/train_rl.r", "rl/rl_environment.r")
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
  required_training_tokens <- c(
    "VINE_OBSERVATION_MODE", "vine_observation_mode = vine_observation_mode")
  required_environment_tokens <- c(
    "no_vine_observation <- identical(private$vine_observation_mode, \"zero\")",
    "cvar_observation <- if (no_vine_observation) 0")
  if (any(!vapply(required_training_tokens, grepl, logical(1),
                  x = snapshot_text[["rl/train_rl.r"]], fixed = TRUE)) ||
      any(!vapply(required_environment_tokens, grepl, logical(1),
                  x = snapshot_text[["rl/rl_environment.r"]], fixed = TRUE))) {
    stop("Snapshotted source does not prove the frozen zero-mode masking path for seed ",
         seed)
  }
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
    source_snapshot_hashes_verified = TRUE)
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
      "Training completed under a run manifest recording zero mode and a",
      "hash-verified source snapshot implementing the frozen masking path.",
      "The redundant plaintext marker was absent; checkpoint hashes and their",
      "metadata status are bound below for post-hoc reconstruction."),
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
  manifests = manifest_rows,
  checkpoint_evidence = checkpoint_evidence)
audit_path <- file.path(sweep_root, "mode_marker_recovery_audit.json")
if (file.exists(audit_path)) stop("Recovery audit already exists: ", audit_path)
audit_temp <- paste0(audit_path, ".tmp")
jsonlite::write_json(root_audit, audit_temp, auto_unbox = TRUE,
                     pretty = TRUE, null = "null")
if (!file.rename(audit_temp, audit_path)) stop("Could not publish recovery audit.")
cat("Validated and reconstructed ten redundant mode markers without modifying checkpoints.\n")
