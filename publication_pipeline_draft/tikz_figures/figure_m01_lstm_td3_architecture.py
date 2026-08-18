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
\node[stage, minimum width=0.43\linewidth] (actor) at (-3.45,1.65) {};
\node[anchor=north west, font=\small] at ([xshift=2mm,yshift=-1.5mm]actor.north west)
  {Step 1: Deterministic Actor};
\node[content, text width=0.385\linewidth, anchor=south] at
  ([yshift=2.2mm]actor.south) {
  Causal input: $S_t\in\mathbb{R}^{30\times88}$\\
  LayerNorm--LSTM: 1 layer, 64 units\\
  Dense 64 + ReLU $\rightarrow$ 7 scores + leverage gate\\
  Projection: $\sum_iw_i=1$, $\|w\|_1\leq1.5$,\;
  $-0.2\leq w_i\leq0.6$};

\node[stage, minimum width=0.43\linewidth] (critics) at (3.45,1.65) {};
\node[anchor=north west, font=\small] at ([xshift=2mm,yshift=-1.5mm]critics.north west)
  {Step 2: Independent Twin Critics};
\node[content, text width=0.385\linewidth, anchor=south] at
  ([yshift=2.2mm]critics.south) {
  Separate LayerNorm--LSTM encoders for $Q_1,Q_2$\\
  Concatenate each last hidden state with $a_t$\\
  Dense 64 + ReLU $\rightarrow$ scalar action value\\
  Both critics update on every gradient step};

\node[stage, minimum width=0.43\linewidth] (target) at (-3.45,-1.35) {};
\node[anchor=north west, font=\small] at ([xshift=2mm,yshift=-1.5mm]target.north west)
  {Step 3: TD3 Target Construction};
\node[content, text width=0.385\linewidth, anchor=south] at
  ([yshift=2.2mm]target.south) {
  Smooth target actor output: $\sigma=0.10$, clip $=0.25$\\
  Project the perturbed raw action onto constraints\\
  $y_t=r_t+(1-d_t)\min\{Q'_1,Q'_2\}$, $\gamma=1$\\
  Uniform replay; critic gradient norm clipped at 1};

\node[stage, minimum width=0.43\linewidth] (delayed) at (3.45,-1.35) {};
\node[anchor=north west, font=\small] at ([xshift=2mm,yshift=-1.5mm]delayed.north west)
  {Step 4: Delayed Policy and Target Updates};
\node[content, text width=0.385\linewidth, anchor=south] at
  ([yshift=2.2mm]delayed.south) {
  Actor updates through $Q_1$ every second critic update\\
  Entropy coefficient $0.005$; leverage penalty $0.25$\\
  Polyak update of all targets with $\tau=0.005$\\
  Deterministic actor; exploration is pre-projection};

\node[stage, minimum width=0.59\linewidth, minimum height=18mm]
  (output) at (0,-4.05) {};
\node[anchor=north west, font=\small] at ([xshift=2mm,yshift=-1.5mm]output.north west)
  {Output: Constrained Recurrent Portfolio Policy};
\node[content, text width=0.545\linewidth, anchor=south] at
  ([yshift=2.2mm]output.south) {
  Seven long--short target weights with hard net, gross, and position limits\\
  Actor, critic 1, and critic 2 retain separate temporal representations};

\draw[flow] (actor.east) -- (critics.west);
\draw[flow] (actor.south) -- ++(0,-0.38) -| (target.north);
\draw[flow] (critics.south) -- ++(0,-0.38) -| (delayed.north);
\draw[flow] (target.east) -- (delayed.west);
\draw[flow] (target.south) -- ++(0,-0.35) -| (output.north);
\draw[flow] (delayed.south) -- ++(0,-0.35) -| (output.north);
\end{tikzpicture}
"""
    context.write(
        "figure_m01_lstm_td3_architecture.tex", body,
        title="Implemented recurrent TD3 architecture",
        evidence_class="implemented_model_schematic",
        inputs=[])
