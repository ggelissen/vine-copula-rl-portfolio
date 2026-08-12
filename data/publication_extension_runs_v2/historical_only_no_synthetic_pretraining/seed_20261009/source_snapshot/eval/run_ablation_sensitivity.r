#!/usr/bin/env Rscript
# Generate figures only from completed experiment manifests.
source("eval/ablation.r")
source("eval/sensitivity.r")

ablation <- run_ablation("config/ablation_manifest.csv")
plot_ablation(ablation, "sharpe_ratio", "figures/ablation_sharpe.pdf")
plot_ablation(ablation, "cvar05_loss", "figures/ablation_cvar.pdf")

sensitivity <- run_sensitivity("config/sensitivity_manifest.csv")
plot_sensitivity_heatmap(sensitivity, "figures/sensitivity_heatmap.pdf")

print(ablation_summary(ablation))
cat(sensitivity_summary(sensitivity), "\n")

