from __future__ import annotations

from .common import PublicationContext, STRATEGY_LABELS, finite, unique_row


STRATEGIES = (
    "equal_weight", "dcc_garch", "static_vine",
    "dynamic_nn_vine", "vine_td3_ensemble",
)

COLORS = {
    "equal_weight": "pubGray",
    "dcc_garch": "pubTeal",
    "static_vine": "pubGold",
    "dynamic_nn_vine": "pubPurple",
    "vine_td3_ensemble": "pubNavy",
}

LABEL_STYLES = {
    # Anchor and signed offsets are chosen from the actual frozen coordinates.
    # This keeps annotations in the panel while separating nearby observations.
    "equal_weight": ("north west", "2pt", "-2pt"),
    "dcc_garch": ("west", "3pt", "-5pt"),
    "static_vine": ("north east", "-2pt", "-3pt"),
    "dynamic_nn_vine": ("south east", "-2pt", "3pt"),
    "vine_td3_ensemble": ("west", "3pt", "4pt"),
}

SHORT_LABELS = {
    "equal_weight": "Equal weight",
    "dcc_garch": "DCC--GARCH",
    "static_vine": "Static vine",
    "dynamic_nn_vine": "Dynamic vine",
    "vine_td3_ensemble": "NN-vine TD3",
}


def generate(context: PublicationContext) -> None:
    monthly_path = context.input("primary_economic_metrics.csv")
    daily_path = context.input("daily_tail_risk_metrics.csv")
    monthly = context.rows(monthly_path.name)
    daily = context.rows(daily_path.name)
    left, right, labels = [], [], []
    for index, strategy in enumerate(STRATEGIES, start=1):
        criteria = dict(scope="complete_periods", source_id="frozen_primary_oos",
                        strategy_id=strategy, window_id="locked_oos_v1")
        m = unique_row(monthly, **criteria)
        d = unique_row(daily, **criteria)
        mdd = 100 * finite(m["max_drawdown"])
        ddd = 100 * finite(d["daily_path_max_drawdown"])
        vol = 100 * finite(d["annualized_daily_volatility"])
        cvar = 100 * finite(d["daily_cvar_95_loss"])
        color = COLORS[strategy]
        labels.append(SHORT_LABELS[strategy])
        left.append(
            f"\\addplot[{color},thick] coordinates {{({mdd:.7g},{index}) ({ddd:.7g},{index})}};\n"
            f"\\addplot[only marks,mark=o,mark size=2.2pt,{color},fill=white] "
            f"coordinates {{({mdd:.7g},{index})}};\n"
            f"\\addplot[publication point,{color},fill={color}] coordinates "
            f"{{({ddd:.7g},{index})}};")
        anchor, xshift, yshift = LABEL_STYLES[strategy]
        right.append(
            f"\\addplot[publication point,{color},fill={color}] coordinates "
            f"{{({vol:.7g},{cvar:.7g})}} node[font=\\scriptsize,anchor={anchor},"
            f"text=pubInk,xshift={xshift},yshift={yshift}] "
            f"{{{SHORT_LABELS[strategy]}}};")
    body = r"""\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=2 by 1,horizontal sep=2.00cm},
  publication axis,
  width=0.31\linewidth,
  height=0.37\linewidth,
]
\nextgroupplot[
  title={(a) Drawdown horizon},
  xlabel={Maximum drawdown (\%)}, ylabel={},
  ymin=0.55, ymax=5.45, y dir=reverse, enlarge y limits=false,
  ytick={1,2,3,4,5},
  yticklabels={""" + ",".join(labels) + r"""},
]
""" + "\n".join(left) + r"""
\nextgroupplot[
  width=0.38\linewidth,
  title={(b) Daily tail risk},
  xlabel={Annualized daily volatility (\%)},
  ylabel={Daily CVaR$_{95}$ loss (\%)},
  xmin=12.8, xmax=25.3, ymin=1.7, ymax=3.45,
]
""" + "\n".join(right) + r"""
\end{groupplot}
\end{tikzpicture}
"""
    context.write_figure(
        "figure_r02_intramonth_risk.tex", body,
        title="Monthly versus daily path risk",
        evidence_class="frozen_primary_evaluation_descriptive_daily_audit",
        inputs=[monthly_path, daily_path])
