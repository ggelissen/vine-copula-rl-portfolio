from __future__ import annotations

from .common import FigureContext


def generate(context: FigureContext) -> None:
    body = r"""\begin{tikzpicture}[
  >=Latex, font=\small,
  stage/.style={draw=black, rounded corners=3mm, minimum height=22mm,
    inner sep=3mm, align=left},
  content/.style={draw=black, sharp corners, inner sep=2.2mm,
    align=left, font=\scriptsize},
  flow/.style={->, line width=0.55pt, draw=black},
]
\node[stage, minimum width=0.43\linewidth] (bundle) at (-3.45,1.55) {};
\node[anchor=north west, font=\small] at ([xshift=2mm,yshift=-1.5mm]bundle.north west)
  {Step 1: Training-Only Synthetic Bundle};
\node[content, text width=0.385\linewidth, anchor=south] at
  ([yshift=2.2mm]bundle.south) {
  Dynamic neural D-vine fitted before the holdout\\
  1,000 episodes: 30 burn-in + 24 decision months\\
  One realised draw + 512 CVaR scenarios per step\\
  Every synthetic episode used exactly once};

\node[stage, minimum width=0.43\linewidth] (pretrain) at (3.45,1.55) {};
\node[anchor=north west, font=\small] at ([xshift=2mm,yshift=-1.5mm]pretrain.north west)
  {Step 2: Synthetic TD3 Pre-Training};
\node[content, text width=0.385\linewidth, anchor=south] at
  ([yshift=2.2mm]pretrain.south) {
  Same state, action map, costs, and CRRA--CVaR reward\\
  Batch 128; one update per environment step\\
  Fixed held-in 100-episode behaviour gate\\
  Fail closed on constraints or degenerate behaviour};

\node[stage, minimum width=0.43\linewidth] (diagnostic) at (-3.45,-1.45) {};
\node[anchor=north west, font=\small] at ([xshift=2mm,yshift=-1.5mm]diagnostic.north west)
  {Step 3: Non-Selective Historical Diagnostic};
\node[content, text width=0.385\linewidth, anchor=south] at
  ([yshift=2.2mm]diagnostic.south) {
  37 fit trajectories; 23 overlapping paths purged\\
  One chronological validation trajectory\\
  Exactly one preregistered pass; no duration search\\
  Candidate and replay memory are then discarded};

\node[stage, minimum width=0.43\linewidth] (refit) at (3.45,-1.45) {};
\node[anchor=north west, font=\small] at ([xshift=2mm,yshift=-1.5mm]refit.north west)
  {Step 4: Historical Refit and Freezing};
\node[content, text width=0.385\linewidth, anchor=south] at
  ([yshift=2.2mm]refit.south) {
  Reload identical pretrained checkpoint; clear replay\\
  One pass over all 61 permissible trajectories\\
  Freeze 20 seeded checkpoints\\
  Arithmetic target-weight ensemble for evaluation};

\node[stage, minimum width=0.59\linewidth, minimum height=19mm]
  (holdout) at (0,-4.15) {};
\node[anchor=north west, font=\small] at ([xshift=2mm,yshift=-1.5mm]holdout.north west)
  {Locked Output: Final 24 Holding Periods};
\node[content, text width=0.545\linewidth, anchor=south] at
  ([yshift=2.2mm]holdout.south) {
  Excluded from vine fitting, episode generation, diagnostics, fine-tuning,
  and checkpoint selection\\
  Accessed once through the frozen common-path evaluation contract};

\draw[flow] (bundle.east) -- (pretrain.west);
\draw[flow] (bundle.south) -- ++(0,-0.38) -| (diagnostic.north);
\draw[flow] (pretrain.south) -- ++(0,-0.38) -| (refit.north);
\draw[flow] (diagnostic.east) -- (refit.west);
\draw[flow] (diagnostic.south) -- ++(0,-0.35) -| (holdout.north);
\draw[flow] (refit.south) -- ++(0,-0.35) -| (holdout.north);
\end{tikzpicture}
"""
    context.write(
        "figure_m02_training_strategy.tex", body,
        title="Leakage-controlled two-stage training protocol",
        evidence_class="implemented_training_protocol_schematic",
        inputs=[])
