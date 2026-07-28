# Benchmark rvinecop parallel simulation at the same 10,000-path scale used
# by RL CVaR. Run from the repository root with Rscript --vanilla.
library(rvinecopulib)

source("helper/load_data.r")
source("benchmark_models/expected_utility_single.r")
load("data/marginal_results.RData")
load("data/vine_fit.RData")

set.seed(20260729)
returns <- load_returns()
vine_static <- vinecop(
  U,
  var_types = rep("c", ncol(U)),
  structure = dvine_structure(1:ncol(U)),
  family_set = c("gaussian", "t", "clayton", "gumbel", "frank", "joe"),
  selcrit = "aic"
)
simulator <- build_simulator(marginals, asset_names, ref_col = 7)

cores_to_test <- as.integer(strsplit(Sys.getenv("VINE_BENCH_CORES", "1,2,4,8,16,32"), ",")[[1]])
n_sim <- as.integer(Sys.getenv("VINE_BENCH_N_SIM", "10000"))
repetitions <- as.integer(Sys.getenv("VINE_BENCH_REPS", "3"))

cat(sprintf("Benchmarking %d paths, %d repetitions per core count\n", n_sim, repetitions))
results <- data.frame(cores = integer(), median_seconds = numeric(), stringsAsFactors = FALSE)
dir.create("data/rl_runs", recursive = TRUE, showWarnings = FALSE)

for (cores in cores_to_test) {
  # Warm-up prevents one-off package/thread-pool setup from biasing timings.
  invisible(simulator$simulate_returns(vine_static, n_sim = n_sim, cores = cores))
  elapsed <- replicate(repetitions, system.time(
    simulator$simulate_returns(vine_static, n_sim = n_sim, cores = cores)
  )[["elapsed"]])
  results <- rbind(results, data.frame(cores = cores, median_seconds = median(elapsed)))
  cat(sprintf("cores=%2d  median=%.3f s\n", cores, median(elapsed)))
}

results$speedup_vs_one_core <- results$median_seconds[results$cores == 1L] / results$median_seconds
print(results, row.names = FALSE)
write.csv(results, "data/rl_runs/vine_sim_benchmark.csv", row.names = FALSE)
