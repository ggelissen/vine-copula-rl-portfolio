#!/usr/bin/env Rscript
# Materialize the post-holdout mixed pretraining curriculum.  The immutable
# 100-path NN-vine subset and all 61 historical training-prefix trajectories
# are proportionally interleaved, then replayed to exactly 1,000 presentations.
# Historical fine-tuning episodes are copied unchanged.

args <- commandArgs(trailingOnly = TRUE)
source_file <- if (length(args) >= 1L) args[[1L]] else
  "data/synthetic_dose_response_v1/vine_synthetic_100.RData"
output_file <- if (length(args) >= 2L) args[[2L]] else paste0(
  "data/mixed_pretraining_response_v1/",
  "mixed_100synthetic_61historical_1000presentations.RData")
expected_source_sha256 <- if (length(args) >= 3L) args[[3L]] else
  "65eb5c715436f155c6cb8447d811e6cb96c2e9b55cc5b2d6ffeb560f9396b314"

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
if (!identical(tolower(sha256_file(source_file)),
               tolower(expected_source_sha256))) {
  stop("The 100-path source differs from the frozen presentation experiment.")
}
if (file.exists(output_file)) stop("Mixed bundle already exists: ", output_file)
dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)

bundle <- new.env(parent = emptyenv())
load(source_file, envir = bundle)
required <- c("pretrain_returns", "finetune_returns", "pretrain_vine",
              "metadata", "train_end")
if (any(!vapply(required, exists, logical(1), envir = bundle,
                inherits = FALSE))) {
  stop("Frozen 100-path bundle is missing required objects.")
}
synthetic_unique <- bundle$pretrain_returns
historical_unique <- bundle$finetune_returns
finetune_returns <- bundle$finetune_returns
pretrain_vine <- bundle$pretrain_vine
metadata <- bundle$metadata
train_end <- bundle$train_end
if (length(synthetic_unique) != 100L || length(historical_unique) != 61L) {
  stop("The mixed design requires exactly 100 synthetic and 61 historical paths.")
}
if (!isTRUE(metadata$diagnostics_passed) ||
    !identical(metadata$pretrain_realised_source, "synthetic_vine") ||
    !identical(metadata$finetune_realised_source, "historical")) {
  stop("Source bundle is not the validated synthetic/historical protocol.")
}

# Merge the two ordered source lists by normalized midpoints.  This is a
# deterministic proportional interleave, independent of path returns, rewards,
# diagnostics, or the evaluation window.
synthetic_position <- (seq_len(100L) - 0.5) / 100L
historical_position <- (seq_len(61L) - 0.5) / 61L
source_table <- rbind(
  data.frame(source = "synthetic", source_index = seq_len(100L),
             position = synthetic_position),
  data.frame(source = "historical", source_index = seq_len(61L),
             position = historical_position))
source_table <- source_table[order(source_table$position,
                                   source_table$source,
                                   source_table$source_index), ]
rownames(source_table) <- NULL
if (nrow(source_table) != 161L ||
    sum(source_table$source == "synthetic") != 100L ||
    sum(source_table$source == "historical") != 61L) {
  stop("Proportional interleave failed its source-count check.")
}
unique_mixed <- lapply(seq_len(nrow(source_table)), function(index) {
  row <- source_table[index, ]
  if (identical(row$source, "synthetic")) {
    synthetic_unique[[row$source_index]]
  } else {
    historical_unique[[row$source_index]]
  }
})

presentation_unique_index <- rep(seq_len(161L), length.out = 1000L)
presentation_cycle <- ((seq_len(1000L) - 1L) %/% 161L) + 1L
pretrain_returns <- unique_mixed[presentation_unique_index]
presentation_source <- source_table$source[presentation_unique_index]
presentation_source_index <- source_table$source_index[presentation_unique_index]
synthetic_presentations <- sum(presentation_source == "synthetic")
historical_presentations <- sum(presentation_source == "historical")
if (length(pretrain_returns) != 1000L ||
    synthetic_presentations + historical_presentations != 1000L ||
    length(finetune_returns) != 61L) {
  stop("Mixed presentation construction failed its exact-accounting check.")
}

metadata$mixed_pretraining_bundle <- TRUE
metadata$parent_pretrain_data_mode <- "vine_synthetic"
metadata$pretrain_realised_source <- "historical_synthetic_mixture"
metadata$finetune_realised_source <- "historical"
metadata$mixed_pretraining_protocol <-
  "proportional_midpoint_interleave_100synthetic_61historical_1000_v1"
metadata$synthetic_unique_episode_count <- 100L
metadata$historical_unique_episode_count <- 61L
metadata$mixed_unique_episode_count <- 161L
metadata$mixed_episode_presentations <- 1000L
metadata$synthetic_episode_presentations <- synthetic_presentations
metadata$historical_episode_presentations <- historical_presentations
metadata$pretrain_episodes <- 1000L
metadata$presentation_unique_index <- as.integer(presentation_unique_index)
metadata$presentation_source <- presentation_source
metadata$presentation_source_index <- as.integer(presentation_source_index)
metadata$presentation_cycle <- as.integer(presentation_cycle)
metadata$source_100_path_bundle_sha256 <- sha256_file(source_file)
metadata$selection_uses_returns_or_diagnostics <- FALSE
metadata$evaluation_data_accessed <- FALSE
metadata$materializer_version <- 1L

temporary <- tempfile(pattern = ".mixed_pretraining_",
                      tmpdir = dirname(output_file), fileext = ".RData")
save(pretrain_returns, finetune_returns, pretrain_vine, metadata, train_end,
     file = temporary, version = 3)
if (!file.rename(temporary, output_file)) stop("Could not publish ", output_file)

manifest <- data.frame(
  protocol = metadata$mixed_pretraining_protocol,
  file = normalizePath(output_file, winslash = "/", mustWork = TRUE),
  sha256 = sha256_file(output_file),
  source_file = normalizePath(source_file, winslash = "/", mustWork = TRUE),
  source_sha256 = sha256_file(source_file),
  synthetic_unique_episode_count = 100L,
  historical_unique_episode_count = 61L,
  mixed_unique_episode_count = 161L,
  mixed_episode_presentations = 1000L,
  synthetic_episode_presentations = synthetic_presentations,
  historical_episode_presentations = historical_presentations,
  finetune_episodes = length(finetune_returns),
  episode_length = as.integer(metadata$episode_length),
  presentation_rule = "proportional_midpoint_interleave_then_ordered_replay",
  selection_uses_returns_or_diagnostics = FALSE,
  evaluation_data_accessed = FALSE,
  stringsAsFactors = FALSE)
manifest_file <- file.path(dirname(output_file),
                           "mixed_pretraining_bundle_manifest.csv")
write.csv(manifest, manifest_file, row.names = FALSE)
writeLines(sprintf("%s  %s", manifest$sha256, basename(output_file)),
           file.path(dirname(output_file), "CONTENTS.sha256"), useBytes = TRUE)
cat(sprintf(paste0("Materialized 1,000 mixed presentations from 100 synthetic and ",
                   "61 historical-prefix trajectories (%d synthetic, %d historical ",
                   "presentations); retained 61 historical fine-tuning episodes.\n"),
            synthetic_presentations, historical_presentations))
cat("Bundle:", normalizePath(output_file, winslash = "/", mustWork = TRUE), "\n")
