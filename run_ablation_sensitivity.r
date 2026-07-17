# ============================================================================
# run_ablation_sensitivity.r
# Run all analyses and generate paper figures
# ============================================================================

# ---- Load required files ----
source("helper/plotting.r")
source("ablation.r")
source("sensitivity.r")

# Helper: Print separator
print_sep <- function() {
  cat(paste0("\n", paste(rep("=", 60), collapse = ""), "\n"))
}

# ---- 1. Load evaluation results ----
load("data/evaluation_results.RData")

# ---- 2. Generate ablation results ----
print_sep()
cat("RUNNING ABLATION STUDY\n")
print_sep()

# Run ablation (will use cache if available)
ablation_df <- run_ablation(
  use_cache = TRUE,
  full_model_metrics = rl_final_metrics,
  force_recompute = FALSE  # Set TRUE to re-run
)

# Save ablation plot
ablation_sharpe <- plot_ablation(ablation_df, metric = "sharpe_ratio", 
                                  save_path = "figures/ablation_sharpe.pdf")
ablation_cvar <- plot_ablation(ablation_df, metric = "cvar", 
                                save_path = "figures/ablation_cvar.pdf")

# ---- 3. Generate sensitivity results ----
print_sep()
cat("RUNNING SENSITIVITY ANALYSIS\n")
print_sep()

# Run sensitivity (will use cache if available)
sensitivity_results <- run_sensitivity(
  use_cache = TRUE,
  force_recompute = FALSE  # Set TRUE to re-run
)

# Save sensitivity heatmap
plot_sensitivity_heatmap(sensitivity_results,
                          save_path = "figures/sensitivity_heatmap.pdf")

# ---- 4. Generate summary for paper ----
print_sep()
cat("SUMMARY\n")
print_sep()

# Ablation summary
cat("\n--- Ablation Summary ---\n")
ablation_summary_df <- ablation_summary(ablation_df)
print(ablation_summary_df)

# Sensitivity summary
cat("\n--- Sensitivity Summary ---\n")
cat(sensitivity_summary(sensitivity_results))

# ---- 5. All figures complete ----
cat("\n✓ All figures generated:\n")
cat("  - figures/wealth_curves_full_comparison.pdf\n")
cat("  - figures/ablation_sharpe.pdf\n")
cat("  - figures/ablation_cvar.pdf\n")
cat("  - figures/sensitivity_heatmap.pdf\n")
cat("\n✓ Analysis complete!\n")