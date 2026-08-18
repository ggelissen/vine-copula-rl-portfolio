#!/usr/bin/env Rscript
# Build the final synthetic-dose identification bundle without reading returns
# from the evaluation window.  The exact immutable 100-path v1 subset is
# repeated in ten ordered passes, so path diversity stays at 100 while episode
# presentations and the TD3 update horizon return to 1,000.

args <- commandArgs(trailingOnly = TRUE)
source_file <- if (length(args) >= 1L) args[[1L]] else
  "data/synthetic_dose_response_v1/vine_synthetic_100.RData"
output_file <- if (length(args) >= 2L) args[[2L]] else
  paste0("data/synthetic_presentation_response_v2/",
         "vine_synthetic_100_unique_1000_presentations.RData")
repetitions <- if (length(args) >= 3L) as.integer(args[[3L]]) else 10L

`%||%` <- function(left, right) if (is.null(left)) right else left

sha256_file <- function(path) {
  executable <- Sys.which("sha256sum")
  if (!nzchar(executable)) stop("sha256sum is required.")
  output <- system2(executable, shQuote(path), stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status") %||% 0L
  if (!length(output) || status != 0L) stop("Could not hash ", path)
  strsplit(output[[1L]], "[[:space:]]+")[[1L]][[1L]]
}

if (!file.exists(source_file)) stop("Frozen 100-path source not found: ", source_file)
if (!is.finite(repetitions) || repetitions != 10L) {
  stop("The frozen v2 presentation design requires exactly ten ordered passes.")
}
if (file.exists(output_file)) {
  stop("Presentation bundle is immutable and already exists: ", output_file)
}
dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)

bundle <- new.env(parent = emptyenv())
load(source_file, envir = bundle)
required <- c("pretrain_returns", "finetune_returns", "pretrain_vine",
              "metadata", "train_end")
if (any(!vapply(required, exists, logical(1), envir = bundle,
                inherits = FALSE))) {
  stop("Frozen 100-path bundle is missing required objects.")
}
unique_pretrain <- bundle$pretrain_returns
finetune_returns <- bundle$finetune_returns
pretrain_vine <- bundle$pretrain_vine
metadata <- bundle$metadata
train_end <- bundle$train_end

if (length(unique_pretrain) != 100L || length(finetune_returns) != 61L) {
  stop(sprintf("Expected 100 unique synthetic and 61 historical episodes; found %d/%d.",
               length(unique_pretrain), length(finetune_returns)))
}
if (!isTRUE(metadata$diagnostics_passed) ||
    !identical(metadata$pretrain_realised_source, "synthetic_vine") ||
    !identical(metadata$finetune_realised_source, "historical") ||
    !identical(metadata$synthetic_dose_protocol,
               "systematic_midpoint_100_of_1000_v1")) {
  stop("Source is not the validated immutable 100-path v1 dose bundle.")
}

# Ordered passes are intentional: every unique path appears exactly once in
# each pass.  No outcome-dependent reweighting, sorting, or path selection is
# performed.  tail(..., 100) in the behavioral gate is therefore one complete
# copy of the same frozen diagnostic set.
presentation_source_index <- rep(seq_along(unique_pretrain), times = repetitions)
presentation_pass <- rep(seq_len(repetitions), each = length(unique_pretrain))
pretrain_returns <- unique_pretrain[presentation_source_index]
if (length(pretrain_returns) != 1000L ||
    any(tabulate(presentation_source_index, nbins = 100L) != repetitions)) {
  stop("Repeated presentation construction failed its exact-balance check.")
}

source_sha256 <- sha256_file(source_file)
metadata$synthetic_dose_bundle <- TRUE
metadata$synthetic_presentation_bundle <- TRUE
metadata$synthetic_dose_protocol <-
  "ordered_10_passes_of_systematic_midpoint_100_v2"
metadata$source_100_path_bundle_sha256 <- source_sha256
metadata$synthetic_unique_episode_count <- 100L
metadata$synthetic_episode_presentations <- 1000L
metadata$synthetic_repetition_count <- repetitions
metadata$pretrain_episodes <- 1000L
metadata$presentation_source_index <- as.integer(presentation_source_index)
metadata$presentation_pass <- as.integer(presentation_pass)
metadata$selection_uses_returns_or_diagnostics <- FALSE
metadata$evaluation_data_accessed <- FALSE
metadata$diagnostics_scope <- "validated_parent_nn_vine_generator"
metadata$materializer_version <- 2L

temporary <- tempfile(pattern = ".synthetic_presentation_",
                      tmpdir = dirname(output_file), fileext = ".RData")
save(pretrain_returns, finetune_returns, pretrain_vine, metadata, train_end,
     file = temporary, version = 3)
if (!file.rename(temporary, output_file)) stop("Could not publish ", output_file)

manifest <- data.frame(
  protocol = metadata$synthetic_dose_protocol,
  file = normalizePath(output_file, winslash = "/", mustWork = TRUE),
  sha256 = sha256_file(output_file),
  source_100_path_file = normalizePath(source_file, winslash = "/", mustWork = TRUE),
  source_100_path_sha256 = source_sha256,
  parent_pretrain_episodes = as.integer(metadata$parent_pretrain_episodes),
  synthetic_unique_episode_count = 100L,
  synthetic_episode_presentations = 1000L,
  repetition_count = repetitions,
  finetune_episodes = length(finetune_returns),
  episode_length = as.integer(metadata$episode_length),
  presentation_rule = "ten_ordered_complete_passes",
  source_indices_per_pass = paste(seq_len(100L), collapse = ";"),
  selection_uses_returns_or_diagnostics = FALSE,
  evaluation_data_accessed = FALSE,
  stringsAsFactors = FALSE)
manifest_file <- file.path(dirname(output_file),
                           "synthetic_presentation_bundle_manifest.csv")
write.csv(manifest, manifest_file, row.names = FALSE)
writeLines(sprintf("%s  %s", manifest$sha256, basename(output_file)),
           file.path(dirname(output_file), "CONTENTS.sha256"), useBytes = TRUE)
cat(sprintf(paste0("Materialized %d ordered presentations from %d immutable unique ",
                   "synthetic episodes; retained %d historical episodes.\n"),
            length(pretrain_returns), length(unique_pretrain),
            length(finetune_returns)))
cat("Bundle:", normalizePath(output_file, winslash = "/", mustWork = TRUE), "\n")
