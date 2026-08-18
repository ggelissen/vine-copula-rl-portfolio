# Helper: timer function
start_timer <- function(label) {
  list(label = label, start = proc.time())
}

stop_timer <- function(timer) {
  elapsed <- (proc.time() - timer$start)["elapsed"]
  cat(sprintf("  [TIMER] %s: %.2f seconds\n", timer$label, elapsed))
  elapsed
}