# ============================================================================
# plotting.r
# Unified file for all plots and a common theme.
# ============================================================================

library(ggplot2)
library(showtext)
library(RColorBrewer)

# Import Computer Modern font
font_add("CMU",
         regular = "C:/Users/gabri/AppData/Local/Microsoft/Windows/Fonts/cmunrm.otf",
         bold    = "C:/Users/gabri/AppData/Local/Microsoft/Windows/Fonts/cmunbx.otf",
         italic  = "C:/Users/gabri/AppData/Local/Microsoft/Windows/Fonts/cmunti.otf",
         bolditalic = "C:/Users/gabri/AppData/Local/Microsoft/Windows/Fonts/cmunbi.otf")
showtext_auto()


# Unified theme for plots
theme_paper <- function(base_size = 9) {
  theme_minimal(base_size = base_size, base_family = "CMU") +
    theme(
      # Panel
      panel.grid.major = element_line(colour = "grey90", linewidth = 0.25),
      panel.grid.minor = element_blank(),
      panel.background = element_rect(fill = "white", colour = NA),
      panel.border   = element_rect(fill = NA, colour = "grey80", linewidth = 0.4),
      
      # Axes
      axis.line     = element_blank(),
      axis.ticks    = element_line(colour = "grey50", linewidth = 0.3),
      axis.ticks.length = unit(1.5, "mm"),
      axis.title    = element_text(size = base_size, colour = "black"),
      axis.text     = element_text(size = base_size - 1, colour = "grey30"),
      
      # Legend
      legend.position      = "bottom",
      legend.background    = element_rect(fill = "white", colour = NA),
      legend.key           = element_rect(fill = "white", colour = NA),
      legend.key.size      = unit(3, "mm"),
      legend.text          = element_text(size = base_size - 2.5, colour = "grey30"),
      legend.title         = element_blank(),
      legend.margin        = margin(t = 2, b = 2),
      
      # Facets
      strip.background = element_rect(fill = "grey95", colour = "grey80", linewidth = 0.3),
      strip.text       = element_text(size = base_size, colour = "black", face = "bold"),
      
      # Margins
      plot.margin   = margin(4, 4, 2, 2),
      plot.subtitle = element_text(size = base_size, colour = "grey40", hjust = 0.5)
    )
}

# ---- Unified colour palette for up to 8 strategies ----
palette_paper <- c(
  "Empirical MV"      = "#222222",
  "DCC--GARCH"        = "#2166AC",
  "Static Vine MV"    = "#B2182B",
  "Rolling Vine MV"   = "#1B7837",
  "NN Vine MV"        = "#762A83",
  "Single-Period EU"  = "#E08214",
  "Multi-Period EU"   = "#9970AB",
  "NN Vine EU"        = "#008837"
)

linetype_paper <- c(
  "Empirical MV"      = "dotted",
  "DCC--GARCH"        = "dashed",
  "Static Vine MV"    = "dotdash",
  "Rolling Vine MV"   = "solid",
  "NN Vine MV"        = "longdash",
  "Single-Period EU"  = "twodash",
  "Multi-Period EU"   = "dashed",
  "NN Vine EU"        = "solid"
)

linewidth_paper <- c(
  "Empirical MV"      = 0.35,
  "DCC--GARCH"        = 0.40,
  "Static Vine MV"    = 0.40,
  "Rolling Vine MV"   = 0.55,
  "NN Vine MV"        = 0.40,
  "Single-Period EU"  = 0.35,
  "Multi-Period EU"   = 0.35,
  "NN Vine EU"        = 0.55
)

# Plot wealth curves for multiple strategies
plot_wealth_curves <- function(results_list, rebal_dates, 
                                strategies = NULL, 
                                custom_labels = NULL,
                                save_path = NULL,
                                width_cm = 8.7,    
                                height_cm = 5.5) {
  
  if (is.null(strategies)) {
    strategies <- names(results_list)
  }
  strategies <- strategies[strategies != "metrics_table"]

  if (is.null(custom_labels)) {
    custom_labels <- c(
      "empirical"      = "Empirical MV",
      "dcc"            = "DCC--GARCH",
      "static"         = "Static Vine MV",
      "rolling"        = "Rolling Vine MV",
      "nn_mv"          = "NN Vine MV",
      "eu_single"      = "Single-Period EU",
      "eu_multi"       = "Multi-Period EU",
      "nn_eu"          = "NN Vine EU"
    )
  }
  
  # Build data frame
  df_list <- lapply(strategies, function(s) {
    wealth <- as.numeric(results_list[[s]]$wealth)
    n <- length(wealth)
    dates <- c(rebal_dates[1], rebal_dates)[1:n]
    label <- if (s %in% names(custom_labels)) custom_labels[s] else s
    data.frame(Date = dates, Wealth = wealth, Strategy = label, row.names = NULL)
  })
  df <- do.call(rbind, df_list)
  
  # Ensure correct ordering
  df$Strategy <- factor(df$Strategy, levels = unname(custom_labels[strategies]))
  
  p <- ggplot(df, aes(x = Date, y = Wealth*1e-3, 
                       colour = Strategy, linetype = Strategy, linewidth = Strategy)) +
    geom_line() +
    scale_colour_manual(values = palette_paper) +
    scale_linetype_manual(values = linetype_paper) +
    scale_linewidth_manual(values = linewidth_paper) +
    scale_y_continuous(labels = scales::comma) +
    labs(x = NULL, y = "Wealth (EUR x 1,000)") +
    theme_paper()
  
  # Save
  if (!is.null(save_path)) {
    ggsave(save_path, plot = p, width = width_cm, height = height_cm, 
           units = "cm", dpi = 600, device = "pdf")
  }
  
  p
}


# Plot training convergence for pre-training and fine-tuning
plot_training_convergence <- function(pretrain_rewards, finetune_rewards,
                                       save_path = NULL,
                                       width_cm = 8.7, height_cm = 6.5) {
  
  # Create data frames
  df_pretrain <- data.frame(
    Episode = seq_along(pretrain_rewards),
    Reward = pretrain_rewards,
    Stage = "Pre-training"
  )
  
  df_finetune <- data.frame(
    Episode = seq_along(finetune_rewards),
    Reward = finetune_rewards,
    Stage = "Fine-tuning"
  )
  
  df <- rbind(df_pretrain, df_finetune)
  
  # Add smoothed line
  p <- ggplot(df, aes(x = Episode, y = Reward, colour = Stage)) +
    geom_line(alpha = 0.3, linewidth = 0.3) +
    geom_smooth(se = FALSE, method = "loess", span = 0.1, linewidth = 0.8) +
    scale_colour_manual(values = c("Pre-training" = "#2166AC", 
                                   "Fine-tuning" = "#B2182B")) +
    labs(x = "Episode", y = "Average Reward") +
    theme_paper() +
    theme(legend.position = "bottom")
  
  if (!is.null(save_path)) {
    ggsave(save_path, plot = p, width = width_cm, height = height_cm,
           units = "cm", dpi = 600, device = "pdf")
  }
  
  return(p)
}


# Plot weight evolution over time for a given strategy
plot_weight_evolution <- function(weights_list, rebal_dates, asset_names,
                                   save_path = NULL,
                                   width_cm = 8.7, height_cm = 5.5) {
  
  # Convert weights to data frame
  n_assets <- length(asset_names)
  n_dates <- length(rebal_dates)
  
  # Ensure weights_list is matrix or list
  if (is.list(weights_list)) {
    weights_matrix <- do.call(rbind, weights_list)
  } else {
    weights_matrix <- weights_list
  }
  
  # Create long format
  df <- data.frame(
    Date = rep(rebal_dates[1:nrow(weights_matrix)], n_assets),
    Weight = as.vector(weights_matrix),
    Asset = rep(asset_names[1:ncol(weights_matrix)], each = nrow(weights_matrix))
  )
  
  # Palette for assets (using ColorBrewer Set1 or similar)
  asset_palette <- RColorBrewer::brewer.pal(max(n_assets, 3), "Set1")
  names(asset_palette) <- asset_names
  
  p <- ggplot(df, aes(x = Date, y = Weight, colour = Asset, fill = Asset)) +
    geom_area(alpha = 0.7, position = "stack") +
    scale_colour_manual(values = asset_palette) +
    scale_fill_manual(values = asset_palette) +
    scale_y_continuous(labels = scales::percent) +
    labs(x = NULL, y = "Portfolio Weight") +
    theme_paper() +
    theme(legend.position = "bottom",
          legend.text = element_text(size = 6))
  
  if (!is.null(save_path)) {
    ggsave(save_path, plot = p, width = width_cm, height = height_cm,
           units = "cm", dpi = 600, device = "pdf")
  }
  
  return(p)
}



# Plot ablation study results
plot_ablation <- function(ablation_results, metric = "sharpe_ratio",
                           save_path = NULL,
                           width_cm = 8.7, height_cm = 5.5) {
  
  # Ensure the metric column exists
  if (!(metric %in% colnames(ablation_results))) {
    stop(sprintf("Metric '%s' not found in ablation_results", metric))
  }
  
  # Remove rows with NA for the selected metric
  df <- ablation_results[!is.na(ablation_results[[metric]]), ]
  
  if (nrow(df) == 0) {
    stop("No data available for the selected metric after removing NAs.")
  }
  
  # Define variant order
  variant_order <- c(
    "Full Model",
    "- Vine state augmentation",
    "- Synthetic pre-training",
    "- CVaR reward penalty",
    "- LSTM temporal encoding"
  )
  
  # Reorder factor levels
  df$Variant <- factor(df$Variant, levels = rev(variant_order))
  
  # Metric labels
  metric_labels <- c(
    "sharpe_ratio" = "Sharpe Ratio",
    "cvar" = "CVaR (95%)",
    "max_drawdown" = "Maximum Drawdown",
    "annual_return" = "Annual Return (%)",
    "annual_vol" = "Annual Volatility (%)"
  )
  
  # Define colour palette for variants
  variant_colours <- c(
    "Full Model" = "#1B7837",
    "- Vine state augmentation" = "#E08214",
    "- Synthetic pre-training" = "#E08214",
    "- CVaR reward penalty" = "#E08214",
    "- LSTM temporal encoding" = "#B2182B"
  )
  
  p <- ggplot(df, aes(x = Variant, y = .data[[metric]], fill = Variant)) +
    geom_col(width = 0.6) +
    geom_text(aes(label = sprintf("%.3f", .data[[metric]])), 
              vjust = -0.3, size = 2.8, colour = "black") +
    scale_fill_manual(values = variant_colours) +
    labs(x = NULL, y = metric_labels[metric]) +
    theme_paper() +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 25, hjust = 1, size = 7))
  
  # Remove y-axis grid for cleaner look
  p <- p + theme(panel.grid.major.y = element_line(colour = "grey90", linewidth = 0.25))
  
  if (!is.null(save_path)) {
    ggsave(save_path, plot = p, width = width_cm, height = height_cm,
           units = "cm", dpi = 600, device = "pdf")
  }
  
  return(p)
}


# Plot sensitivity analysis in a heatmap
plot_sensitivity_heatmap <- function(sensitivity_df,
                                      save_path = NULL,
                                      width_cm = 8.7, height_cm = 6.5) {
  
  # Expected structure:
  # lambda | kappa | sharpe_ratio
  # 0.00   | 0.00  | 0.12
  # 0.00   | 0.01  | 0.15
  # ...
  
  p <- ggplot(sensitivity_df, aes(x = factor(kappa), y = factor(lambda), 
                                   fill = sharpe_ratio)) +
    geom_tile() +
    geom_text(aes(label = sprintf("%.3f", sharpe_ratio)), size = 2.5) +
    scale_fill_gradient2(
      low = "#B2182B", 
      mid = "#F7F7F7", 
      high = "#2166AC",
      midpoint = 0,
      name = "Sharpe\nRatio"
    ) +
    labs(x = expression(kappa ~ "(Transaction Cost Penalty)"),
         y = expression(lambda ~ "(Risk Aversion)")) +
    theme_paper() +
    theme(legend.position = "right",
          legend.key.size = unit(4, "mm"),
          legend.text = element_text(size = 6),
          legend.title = element_text(size = 7))
  
  if (!is.null(save_path)) {
    ggsave(save_path, plot = p, width = width_cm, height = height_cm,
           units = "cm", dpi = 600, device = "pdf")
  }
  
  return(p)
}


# Plot rolling correlation heatmap for a given strategy
plot_rolling_correlation <- function(correlation_array, dates, asset_names,
                                      save_path = NULL,
                                      width_cm = 8.7, height_cm = 10) {
  
  # Convert to long format
  n_assets <- length(asset_names)
  n_dates <- dim(correlation_array)[1]
  
  # Flatten the correlation matrix (upper triangle only)
  pairs <- combn(asset_names, 2, simplify = FALSE)
  df_list <- lapply(pairs, function(pair) {
    i <- which(asset_names == pair[1])
    j <- which(asset_names == pair[2])
    data.frame(
      Date = dates,
      Pair = paste(pair[1], pair[2], sep = " - "),
      Correlation = correlation_array[, i, j]
    )
  })
  df <- do.call(rbind, df_list)
  
  p <- ggplot(df, aes(x = Date, y = Pair, fill = Correlation)) +
    geom_tile() +
    scale_fill_gradient2(
      low = "#B2182B", 
      mid = "#F7F7F7", 
      high = "#2166AC",
      midpoint = 0,
      limits = c(-1, 1),
      name = "Correlation"
    ) +
    labs(x = NULL, y = NULL) +
    theme_paper() +
    theme(legend.position = "right",
          axis.text.y = element_text(size = 5),
          axis.text.x = element_text(angle = 45, hjust = 1, size = 6))
  
  if (!is.null(save_path)) {
    ggsave(save_path, plot = p, width = width_cm, height = height_cm,
           units = "cm", dpi = 600, device = "pdf")
  }
  
  return(p)
}