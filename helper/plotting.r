# ============================================================================
# plotting_theme.r
# Unified ggplot2 theme for all paper figures
# ============================================================================

library(ggplot2)
library(showtext)

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

# ---- Main plotting function ----
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