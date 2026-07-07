# ============================================================
# vine_copula.r
# Fit static vine copula to uniform residuals, simulate returns,
# and compute portfolio moments
# ============================================================

library(VineCopula)
library(xts)

load("data/marginal_results.RData")
source("load_data.r")

# Fit a static D-vine to residuals.
# Select structures using AIC and choosing from (Gauss,t,Clayton,Gumbel,Frank,Joe).
vine_fit <- RVineStructureSelect(U, familyset = c(1,2,3,4,5,6),
                                 selectioncrit = "AIC",
                                 type = 0
                                )

summary(vine_fit)
plot(vine_fit, tree = "ALL", edge.labels = "family")
save(vine_fit, file = "data/vine_fit.RData")
