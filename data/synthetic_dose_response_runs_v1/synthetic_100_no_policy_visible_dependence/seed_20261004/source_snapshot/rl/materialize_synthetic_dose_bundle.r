#!/usr/bin/env Rscript
# Materialize a deterministic, immutable dose subset of the validated
# synthetic NN-vine bundle. The final evaluation sample is never read.

args <- commandArgs(trailingOnly = TRUE)
source_file <- if (length(args) >= 1L) args[[1L]] else
  "data/synthetic_returns.RData"
output_file <- if (length(args) >= 2L) args[[2L]] else
  "data/synthetic_dose_response_v1/vine_synthetic_100.RData"
target_count <- if (length(args) >= 3L) as.integer(args[[3L]]) else 100L

`%||%` <- function(left, right) if (is.null(left)) right else left

sha256_file <- function(path) {
  executable <- Sys.which("sha256sum")
  if (!nzchar(executable)) stop("sha256sum is required.")
  output <- system2(executable, shQuote(path), stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status") %||% 0L
  if (!length(output) || status != 0L) stop("Could not hash ", path)
  strsplit(output[[1L]], "[[:space:]]+")[[1L]][[1L]]
}

if (!file.exists(source_file)) stop("Parent synthetic bundle not found: ", source_file)
if (!is.finite(target_count) || target_count != 100L) {
  stop("The frozen v1 dose is exactly 100 unique synthetic episodes.")
}
if (file.exists(output_file)) stop("Dose bundle is immutable and already exists: ", output_file)
dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)

bundle <- new.env(parent = emptyenv())
load(source_file, envir = bundle)
required <- c("pretrain_returns", "finetune_returns", "pretrain_vine",
              "metadata", "train_end")
if (any(!vapply(required, exists, logical(1), envir = bundle,
                inherits = FALSE))) {
  stop("Parent bundle is missing required objects.")
}
parent_pretrain <- bundle$pretrain_returns
finetune_returns <- bundle$finetune_returns
pretrain_vine <- bundle$pretrain_vine
metadata <- bundle$metadata
train_end <- bundle$train_end

if (length(parent_pretrain) != 1000L || length(finetune_returns) != 61L) {
  stop(sprintf(
    "Expected the frozen 1000-synthetic/61-historical parent; found %d/%d.",
    length(parent_pretrain), length(finetune_returns)))
}
if (!isTRUE(metadata$diagnostics_passed) ||
    !identical(metadata$pretrain_realised_source, "synthetic_vine") ||
    !identical(metadata$finetune_realised_source, "historical")) {
  stop("Parent is not the validated synthetic-pretrain/historical-finetune bundle.")
}

# Version-independent midpoint systematic sample. The episode order was fixed
# before outcome inspection; this spreads the subset across the immutable
# generator stream and performs no return- or diagnostic-based selection.
parent_count <- length(parent_pretrain)
selection_indices <- floor(
  ((seq_len(target_count) - 0.5) * parent_count) / target_count) + 1L
if (length(unique(selection_indices)) != target_count ||
    min(selection_indices) < 1L || max(selection_indices) > parent_count) {
  stop("Deterministic dose selection did not produce 100 unique valid indices.")
}
pretrain_returns <- parent_pretrain[selection_indices]

parent_sha256 <- sha256_file(source_file)
metadata$synthetic_dose_bundle <- TRUE
metadata$synthetic_dose_protocol <- "systematic_midpoint_100_of_1000_v1"
metadata$parent_bundle_sha256 <- parent_sha256
metadata$parent_pretrain_episodes <- parent_count
metadata$pretrain_episodes <- target_count
metadata$synthetic_unique_episode_count <- target_count
metadata$synthetic_episode_presentations <- target_count
metadata$selection_indices <- as.integer(selection_indices)
metadata$selection_uses_returns_or_diagnostics <- FALSE
metadata$evaluation_data_accessed <- FALSE
metadata$diagnostics_scope <- "validated_parent_nn_vine_generator"
metadata$materializer_version <- 1L

temporary <- tempfile(pattern = ".synthetic_dose_", tmpdir = dirname(output_file),
                      fileext = ".RData")
save(pretrain_returns, finetune_returns, pretrain_vine, metadata, train_end,
     file = temporary, version = 3)
if (!file.rename(temporary, output_file)) stop("Could not publish ", output_file)

manifest <- data.frame(
  protocol = metadata$synthetic_dose_protocol,
  file = normalizePath(output_file, winslash = "/", mustWork = TRUE),
  sha256 = sha256_file(output_file),
  parent_file = normalizePath(source_file, winslash = "/", mustWork = TRUE),
  parent_sha256 = parent_sha256,
  parent_pretrain_episodes = parent_count,
  selected_pretrain_episodes = target_count,
  finetune_episodes = length(finetune_returns),
  episode_length = as.integer(metadata$episode_length),
  selection_rule = "floor(((i-0.5)*N)/k)+1",
  selection_indices = paste(selection_indices, collapse = ";"),
  evaluation_data_accessed = FALSE,
  stringsAsFactors = FALSE)
manifest_file <- file.path(dirname(output_file), "synthetic_dose_bundle_manifest.csv")
write.csv(manifest, manifest_file, row.names = FALSE)
writeLines(sprintf("%s  %s", manifest$sha256, basename(output_file)),
           file.path(dirname(output_file), "CONTENTS.sha256"), useBytes = TRUE)
cat(sprintf(
  "Materialized %d-of-%d synthetic episodes; retained %d historical fine-tuning episodes.\n",
  target_count, parent_count, length(finetune_returns)))
cat("Bundle:", normalizePath(output_file, winslash = "/", mustWork = TRUE), "\n")
