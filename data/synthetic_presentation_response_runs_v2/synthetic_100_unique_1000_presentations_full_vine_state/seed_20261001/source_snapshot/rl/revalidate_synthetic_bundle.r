#!/usr/bin/env Rscript
# Revalidate an already generated synthetic bundle under the sampling-aware
# guardrailed diagnostic gate. No return, vine state, or episode is regenerated.

suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
})
source("helper/synthetic_fidelity.r")

sha256_file_local <- function(path) {
  if (requireNamespace("digest", quietly = TRUE))
    return(digest::digest(file = path, algo = "sha256", serialize = FALSE))
  executable <- Sys.which("sha256sum")
  if (!nzchar(executable)) stop("digest or sha256sum is required.")
  output <- system2(executable, shQuote(normalizePath(
    path, winslash = "/", mustWork = TRUE)), stdout = TRUE, stderr = TRUE)
  if (!length(output) || !is.null(attr(output, "status")))
    stop("sha256sum failed for ", path)
  strsplit(output[[1L]], "[[:space:]]+")[[1L]][[1L]]
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4L) stop(paste(
  "Usage: Rscript --vanilla rl/revalidate_synthetic_bundle.r",
  "INPUT_BUNDLE OUTPUT_BUNDLE OUTPUT_MANIFEST OUTPUT_DIAGNOSTICS_DIR"))
input_file <- args[[1L]]
output_file <- args[[2L]]
output_manifest <- args[[3L]]
diagnostic_dir <- args[[4L]]
input_manifest <- if (length(args) >= 5L) args[[5L]] else
  paste0(input_file, ".manifest.json")

if (!file.exists(input_file)) stop("Input bundle does not exist: ", input_file)
if (file.exists(output_file) || file.exists(output_manifest))
  stop("Revalidated output already exists; choose a new empty target.")
if (dir.exists(diagnostic_dir) && length(list.files(
    diagnostic_dir, all.files = TRUE, no.. = TRUE)))
  stop("Revalidated diagnostic directory must be empty: ", diagnostic_dir)
dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(output_manifest), recursive = TRUE, showWarnings = FALSE)
dir.create(diagnostic_dir, recursive = TRUE, showWarnings = FALSE)

bundle <- new.env(parent = emptyenv())
loaded <- load(input_file, envir = bundle)
required <- c("pretrain_returns", "finetune_returns", "metadata", "fidelity",
              "correlation_comparison", "tail_dependence",
              "temporal_dependence")
missing <- setdiff(required, loaded)
if (length(missing)) stop("Bundle is missing: ", paste(missing, collapse = ", "))

realised_matrix <- function(ep) do.call(rbind, lapply(ep$returns, function(draw)
  as.numeric(draw[1L, ])))
first <- bundle$finetune_returns[[1L]]
historical_gross <- rbind(do.call(rbind, first$burnin_returns),
                          realised_matrix(first))
if (length(bundle$finetune_returns) > 1L) {
  for (index in 2:length(bundle$finetune_returns)) {
    candidate <- realised_matrix(bundle$finetune_returns[[index]])
    overlap <- nrow(candidate) - 1L
    if (!isTRUE(all.equal(candidate[seq_len(overlap), , drop = FALSE],
                          tail(historical_gross, overlap), tolerance = 1e-12,
                          check.attributes = FALSE)))
      stop("Historical fine-tuning episodes do not form a consistent overlap chain.")
    historical_gross <- rbind(historical_gross,
                              candidate[nrow(candidate), , drop = FALSE])
  }
}
asset_names <- bundle$metadata$asset_names
if (is.null(asset_names) || ncol(historical_gross) != length(asset_names))
  stop("Bundle asset metadata is inconsistent with historical episodes.")
colnames(historical_gross) <- asset_names
historical_log <- log(historical_gross)

synthetic_episode_log <- lapply(bundle$pretrain_returns, function(ep) {
  matrix <- log(realised_matrix(ep))
  colnames(matrix) <- asset_names
  matrix
})
seed <- as.integer(bundle$metadata$seed)
if (length(seed) != 1L || !is.finite(seed)) stop("Bundle seed is invalid.")

fidelity <- apply_sampling_aware_marginal_gate(
  as.data.table(bundle$fidelity), historical_log, asset_names, seed)
correlation_comparison <- apply_sampling_aware_correlation_gate(
  as.data.table(bundle$correlation_comparison), synthetic_episode_log,
  asset_names, seed)
diagnostics_passed <- all(fidelity$statistically_compatible) &&
  all(correlation_comparison$statistically_compatible) &&
  all(bundle$tail_dependence$pass_lower_tail) &&
  all(bundle$temporal_dependence$pass_temporal)
if (!diagnostics_passed) {
  stop(sprintf(paste0("Revalidation still fails: marginals %d/%d; correlation %d/%d; ",
                      "tail %d/%d; temporal %d/%d."),
               sum(fidelity$statistically_compatible), nrow(fidelity),
               sum(correlation_comparison$statistically_compatible),
               nrow(correlation_comparison),
               sum(bundle$tail_dependence$pass_lower_tail),
               nrow(bundle$tail_dependence),
               sum(bundle$temporal_dependence$pass_temporal),
               nrow(bundle$temporal_dependence)))
}

parent_hash <- sha256_file_local(input_file)
bundle$fidelity <- fidelity
bundle$correlation_comparison <- correlation_comparison
bundle$metadata$diagnostics_passed <- TRUE
bundle$metadata$diagnostic_gate_protocol <- "sampling_aware_guardrailed_v2"
bundle$metadata$diagnostic_gate_revision <-
  "post_generation_statistical_revision_without_resimulation"
bundle$metadata$parent_bundle_sha256 <- parent_hash
bundle$metadata$confirmatory_claim_permitted <- FALSE

temporary <- paste0(output_file, ".tmp-", Sys.getpid())
on.exit(unlink(temporary, force = TRUE), add = TRUE)
save(list = ls(bundle, all.names = TRUE), envir = bundle, file = temporary,
     compress = "gzip", version = 3)
if (!file.rename(temporary, output_file)) stop("Could not publish revalidated bundle.")

write.csv(fidelity, file.path(diagnostic_dir, "fidelity_metrics.csv"),
          row.names = FALSE)
write.csv(correlation_comparison,
          file.path(diagnostic_dir, "correlation_comparison.csv"),
          row.names = FALSE)
for (name in c("tail_dependence", "temporal_dependence", "summary_stats",
               "tail_metrics", "portfolio_metrics", "episode_metrics")) {
  if (exists(name, envir = bundle, inherits = FALSE)) {
    filename <- switch(name,
      tail_dependence = "tail_dependence_comparison.csv",
      temporal_dependence = "temporal_dependence.csv",
      summary_stats = "summary_statistics.csv",
      tail_metrics = "tail_risk.csv",
      portfolio_metrics = "portfolio_metrics.csv",
      episode_metrics = "synthetic_episode_metrics.csv")
    write.csv(get(name, envir = bundle, inherits = FALSE),
              file.path(diagnostic_dir, filename), row.names = FALSE)
  }
}

manifest <- if (file.exists(input_manifest))
  jsonlite::read_json(input_manifest, simplifyVector = TRUE) else list()
if (is.null(manifest$schema_version)) manifest$schema_version <- 1L
manifest$release_status <- "revalidated_existing_training_bundle"
manifest$bundle_file <- normalizePath(output_file, winslash = "/", mustWork = TRUE)
manifest$bundle_sha256 <- sha256_file_local(output_file)
manifest$parent_bundle_sha256 <- parent_hash
manifest$diagnostics_passed <- TRUE
manifest$diagnostic_gate_protocol <- "sampling_aware_guardrailed_v2"
manifest$diagnostic_gate_revision <-
  "post_generation_statistical_revision_without_resimulation"
manifest$synthetic_returns_regenerated <- FALSE
manifest$confirmatory_claim_permitted <- FALSE
jsonlite::write_json(manifest, output_manifest, auto_unbox = TRUE,
                     pretty = TRUE, null = "null")

cat(jsonlite::toJSON(list(
  status = "synthetic_bundle_revalidated_without_resimulation",
  output_file = normalizePath(output_file, winslash = "/", mustWork = TRUE),
  bundle_sha256 = manifest$bundle_sha256,
  marginal_strict_pass = sprintf("%d/%d", sum(fidelity$pass_marginals),
                                 nrow(fidelity)),
  marginal_compatible = sprintf("%d/%d",
    sum(fidelity$statistically_compatible), nrow(fidelity)),
  correlation_strict_pass = sprintf("%d/%d",
    sum(correlation_comparison$pass_correlation), nrow(correlation_comparison)),
  correlation_compatible = sprintf("%d/%d",
    sum(correlation_comparison$statistically_compatible),
    nrow(correlation_comparison)),
  synthetic_returns_regenerated = FALSE,
  confirmatory_claim_permitted = FALSE), auto_unbox = TRUE, pretty = TRUE), "\n")
