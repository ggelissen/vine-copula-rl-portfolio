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
  code_roots <- c("rl", "helper", "benchmark_models", "eval", "config", "tests")
  code_files <- unlist(lapply(code_roots[dir.exists(code_roots)], function(root) {
    list.files(root, pattern = "\\.(r|R|py|yaml|csv)$", recursive = TRUE,
               full.names = TRUE)
  }), use.names = FALSE)
  code_files <- c(code_files,
    list.files(".", pattern = "\\.(r|R)$", recursive = FALSE, full.names = TRUE))
  manifest <- list(
    schema_version = 1L,
    created_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
    seed = as.integer(seed),
    config_file = normalizePath(config_file, winslash = "/", mustWork = FALSE),
    config_hash = if (file.exists(config_file)) unname(tools::md5sum(config_file)) else NA_character_,
    data_hashes = file_hashes(data_files),
    code_hashes = file_hashes(code_files),
    r_version = R.version.string,
    session_info = capture.output(sessionInfo()),
    git = safe_git_state("."),
    relevant_environment = Sys.getenv(c(
      "TRAIN_SEED", "TRAIN_DEVICE", "VINE_MODEL", "ENV_T", "ENV_SEQ_LEN",
      "ENV_GROSS_LEVERAGE", "ENV_NET_EXPOSURE", "ENV_SHORT_BORROW_RATE",
      "ENV_CASH_BORROW_RATE", "POLICY_DELAY", "TARGET_POLICY_NOISE",
      "TARGET_NOISE_CLIP", "RANDOM_EXPLORATION_STEPS"
    ))
  )
  saveRDS(manifest, file.path(output_dir, "run_manifest.rds"))
  write.csv(manifest$code_hashes, file.path(output_dir, "code_hashes.csv"),
            row.names = FALSE)
  write.csv(manifest$data_hashes, file.path(output_dir, "data_hashes.csv"),
            row.names = FALSE)
  invisible(manifest)
}
