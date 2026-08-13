#!/usr/bin/env Rscript
# Build immutable, matched-update non-vine-generator pretraining controls from
# the already frozen training-prefix bundle. No evaluation return is read.

suppressPackageStartupMessages(library(yaml))

args <- commandArgs(trailingOnly = TRUE)
source_file <- if (length(args) >= 1L) args[[1L]] else "data/synthetic_returns.RData"
output_root <- if (length(args) >= 2L) args[[2L]] else
  "data/ablation_training_bundles"
config_file <- if (length(args) >= 3L) args[[3L]] else "config/config.yaml"

if (!file.exists(source_file)) stop("Source training bundle not found: ", source_file)
if (!file.exists(config_file)) stop("Config not found: ", config_file)
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)
output_files <- c(
  historical_prefix_repeated = "historical_prefix_repeated.RData",
  moving_block_bootstrap = "moving_block_bootstrap.RData")
# file.path() does not reliably retain names across supported R versions.
# Preserve the intervention identifiers explicitly because they are part of
# the frozen causal contract and are used for fail-closed named lookup below.
outputs <- setNames(
  file.path(output_root, unname(output_files)), names(output_files))
if (!identical(names(outputs), names(output_files)) || any(!nzchar(outputs))) {
  stop("Could not construct the named ablation-bundle output contract.")
}
if (any(file.exists(outputs))) {
  stop("Ablation bundles are immutable and already exist: ",
       paste(outputs[file.exists(outputs)], collapse = ", "))
}

sha256_files <- function(paths) {
  executable <- Sys.which("sha256sum")
  if (!nzchar(executable)) stop("sha256sum is required to attest ablation bundles.")
  vapply(paths, function(path) {
    output <- system2(executable, shQuote(path), stdout = TRUE, stderr = TRUE)
    status <- attr(output, "status") %||% 0L
    if (!length(output) || status != 0L) {
      stop("Could not hash ", path)
    }
    strsplit(output[[1L]], "[[:space:]]+")[[1L]][[1L]]
  }, character(1))
}
`%||%` <- function(left, right) if (is.null(left)) right else left

config <- yaml::yaml.load_file(config_file)
seed <- as.integer(config$general$seed) + 401L
block_length <- 6L
bundle <- new.env(parent = emptyenv())
load(source_file, envir = bundle)
required <- c("pretrain_returns", "finetune_returns", "pretrain_vine",
              "metadata", "train_end")
if (any(!vapply(required, exists, logical(1), envir = bundle,
                inherits = FALSE))) {
  stop("Source bundle is missing required objects.")
}
parent_pretrain <- bundle$pretrain_returns
finetune_returns <- bundle$finetune_returns
pretrain_vine <- bundle$pretrain_vine
parent_metadata <- bundle$metadata
train_end <- bundle$train_end
if (!identical(parent_metadata$pretrain_realised_source, "synthetic_vine") ||
    !isTRUE(parent_metadata$diagnostics_passed)) {
  stop("Parent must be a diagnostics-passing NN-vine synthetic bundle.")
}
target_count <- length(parent_pretrain)
episode_length <- as.integer(parent_metadata$episode_length)
sequence_length <- as.integer(parent_metadata$sequence_length)
if (target_count < 1L || length(finetune_returns) < 2L ||
    episode_length < 2L || sequence_length < 1L) {
  stop("Parent episode geometry is invalid.")
}

write_bundle <- function(path, pretrain_returns, source_label, scenario_label,
                         method_fields = list()) {
  metadata <- parent_metadata
  metadata$ablation_bundle <- TRUE
  metadata$parent_bundle_sha256 <- unname(sha256_files(source_file))
  metadata$parent_pretrain_data_mode <- "vine_synthetic"
  metadata$pretrain_realised_source <- source_label
  metadata$pretrain_scenario_source <- scenario_label
  metadata$pretrain_episodes <- length(pretrain_returns)
  metadata$diagnostics_passed <- NA
  metadata$diagnostics_applicable <- FALSE
  metadata$generated_at <- Sys.time()
  for (name in names(method_fields)) metadata[[name]] <- method_fields[[name]]
  temporary <- tempfile(pattern = ".ablation_bundle_", tmpdir = dirname(path),
                        fileext = ".RData")
  save(pretrain_returns, finetune_returns, pretrain_vine, metadata, train_end,
       file = temporary, version = 3)
  if (!file.rename(temporary, path)) stop("Could not atomically publish ", path)
  invisible(path)
}

# Matched-update historical control: every pretraining update sees a real
# training-prefix trajectory; deterministic reshuffling between cycles avoids
# privileging one chronological start while making data reuse fully auditable.
set.seed(seed)
historical_pretrain <- vector("list", target_count)
orders <- integer()
while (length(orders) < target_count) {
  orders <- c(orders, sample.int(length(finetune_returns)))
}
orders <- orders[seq_len(target_count)]
for (i in seq_len(target_count)) {
  historical_pretrain[[i]] <- finetune_returns[[orders[[i]]]]
  historical_pretrain[[i]]$source <- "historical_prefix_repeated_matched_updates"
  historical_pretrain[[i]]$parent_episode <- orders[[i]]
}
pretrain_returns <- historical_pretrain
write_bundle(
  outputs[["historical_prefix_repeated"]], pretrain_returns,
  "historical_prefix_repeated_matched_updates",
  "parent_dynamic_nn_vine_scenarios",
  list(repetition_seed = seed, repetition_rule = "balanced_random_cycles"))

# Reconstruct the unique historical monthly path represented by the overlapping
# fine-tuning episodes. The first episode supplies the initial horizon and each
# subsequent episode contributes exactly its new terminal month.
path_returns <- finetune_returns[[1L]]$returns
path_states <- finetune_returns[[1L]]$vine_states
if (length(finetune_returns) > 1L) {
  for (i in 2:length(finetune_returns)) {
    path_returns <- c(path_returns,
                      list(finetune_returns[[i]]$returns[[episode_length]]))
    path_states <- c(path_states,
                     list(finetune_returns[[i]]$vine_states[[episode_length]]))
  }
}
realised_pool <- do.call(rbind, lapply(path_returns, function(value) value[1L, ]))
if (any(!is.finite(realised_pool)) || any(realised_pool <= 0)) {
  stop("Historical realised pool is invalid.")
}
scenario_count <- nrow(path_returns[[1L]]) - 1L
if (scenario_count < 1L) stop("Parent episodes contain no CVaR scenarios.")

sample_block_path <- function(total, pool_size, block, circular = TRUE) {
  output <- integer()
  while (length(output) < total) {
    start <- sample.int(pool_size, 1L)
    indices <- start + seq.int(0L, block - 1L)
    if (circular) indices <- ((indices - 1L) %% pool_size) + 1L
    else indices <- indices[indices <= pool_size]
    output <- c(output, indices)
  }
  output[seq_len(total)]
}

set.seed(seed + 1L)
bootstrap_pretrain <- vector("list", target_count)
total_length <- sequence_length + episode_length
for (episode in seq_len(target_count)) {
  selected <- sample_block_path(total_length, nrow(realised_pool), block_length)
  action_indices <- selected[(sequence_length + 1L):total_length]
  scenario_steps <- lapply(action_indices, function(index) {
    scenario_indices <- sample.int(nrow(realised_pool), scenario_count,
                                   replace = TRUE)
    rbind(realised_pool[index, ], realised_pool[scenario_indices, , drop = FALSE])
  })
  bootstrap_pretrain[[episode]] <- list(
    burnin_returns = lapply(selected[seq_len(sequence_length)],
                            function(index) realised_pool[index, ]),
    burnin_vine_states = path_states[selected[seq_len(sequence_length)]],
    returns = scenario_steps,
    vine_states = path_states[action_indices],
    vine_start = NA_integer_,
    source = "historical_moving_block_bootstrap",
    bootstrap_indices = selected)
}
pretrain_returns <- bootstrap_pretrain
write_bundle(
  outputs[["moving_block_bootstrap"]], pretrain_returns,
  "historical_moving_block_bootstrap",
  "independent_empirical_monthly_bootstrap",
  list(bootstrap_seed = seed + 1L, block_length_months = block_length,
       circular_blocks = TRUE, matched_episode_count = target_count))

manifest <- data.frame(
  mode = names(outputs), file = unname(outputs),
  sha256 = unname(sha256_files(outputs)),
  pretrain_episodes = target_count, finetune_episodes = length(finetune_returns),
  evaluation_data_accessed = FALSE, stringsAsFactors = FALSE)
write.csv(manifest, file.path(output_root, "ablation_bundle_manifest.csv"),
          row.names = FALSE)
writeLines(paste(manifest$sha256, basename(manifest$file)),
           file.path(output_root, "CONTENTS.sha256"), useBytes = TRUE)
cat(sprintf("Generated two immutable training-prefix ablation bundles in %s\n",
            normalizePath(output_root, winslash = "/", mustWork = TRUE)))
