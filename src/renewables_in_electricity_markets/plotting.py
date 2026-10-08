"""Writing results: tables as CSV + LaTeX, figures as PNG, and the plots shared by steps.

Every step saves into results/<step>/, e.g. save_table(df, "step1", "producers")
writes results/step1/producers.csv and results/step1/producers.tex. The .tex file
is a booktabs tabular, ready for \\input{} in the report (needs \\usepackage{booktabs}).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from renewables_in_electricity_markets.data import MarketData
from renewables_in_electricity_markets.model import (
    ClearingResult,
    _resolve_hour,
    clear_graphical,
    demand_curve,
    supply_curve,
)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})


# ----------------------------------------------------------------------------
# Saving
# ----------------------------------------------------------------------------
def _folder(step: str) -> Path:
    folder = RESULTS_DIR / step
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _tex_escape(text: str) -> str:
    for char, escaped in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                          ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")]:
        text = text.replace(char, escaped)
    return text


def to_latex(df: pd.DataFrame, decimals: int = 2, index: bool = True) -> str:
    """A booktabs tabular for `df`; floats rounded to `decimals`, numbers right-aligned.

    Written by hand because pandas' own to_latex needs jinja2.
    """
    if index:
        df = df.rename_axis(df.index.name or "").reset_index()
    cells = [[_tex_escape(str(c)) for c in df.columns]]
    for row in df.itertuples(index=False):
        cells.append([
            # + 0.0 turns a rounded -0.0 into 0.0
            f"{round(v, decimals) + 0.0:,.{decimals}f}" if isinstance(v, float | np.floating) and np.isfinite(v)
            else "" if v is None or (isinstance(v, float | np.floating) and np.isnan(v))
            else _tex_escape(str(v))
            for v in row
        ])
    align = "".join("r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in df.columns)
    lines = [f"\\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(cells[0]) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in cells[1:]]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def save_table(df: pd.DataFrame, step: str, name: str, decimals: int = 2, index: bool = True) -> None:
    """Write results/<step>/<name>.csv (full precision) and <name>.tex (rounded)."""
    folder = _folder(step)
    df.to_csv(folder / f"{name}.csv", index=index)
    (folder / f"{name}.tex").write_text(to_latex(df, decimals, index), encoding="utf-8")


def save_figure(fig: plt.Figure, step: str, name: str) -> None:
    """Write results/<step>/<name>.png and close the figure."""
    fig.tight_layout()
    fig.savefig(_folder(step) / f"{name}.png")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Plots shared by steps
# ----------------------------------------------------------------------------
def plot_supply_curve(data: MarketData, t: int | None = None, ax=None):
    ax = ax or plt.gca()
    curve = supply_curve(data, t)
    wind = curve[curve["kind"] == "wind"]
    if not wind.empty:
        # zero-cost bars have no height, so mark the wind block as a shaded span instead
        ax.axvspan(0, wind["cum_p"].max(), color="tab:green", alpha=0.15, label="Wind (zero cost)")
        ax.legend()
    ax.bar(curve["cum_p"] - curve["p_max"], curve["cost"], width=curve["p_max"],
           align="edge", edgecolor="white", color="tab:blue")
    ax.set(xlim=(0, None), ylim=(0, None), xlabel="Cumulative capacity (MW)",
           ylabel="Marginal cost (EUR/MWh)", title=f"Merit order, hour {_resolve_hour(data, t)}")
    return ax


def plot_demand_curve(data: MarketData, t: int | None = None, ax=None):
    ax = ax or plt.gca()
    curve = demand_curve(data, t)
    ax.bar(curve["cum_p"] - curve["p_max"], curve["bid"], width=curve["p_max"],
           align="edge", edgecolor="white", color="tab:orange")
    ax.set(xlabel="Cumulative demand (MW)", ylabel="Bid price (EUR/MWh)",
           title=f"Demand bids, hour {_resolve_hour(data, t)}")
    return ax


def plot_clearing(data: MarketData, t: int | None = None, ax=None, price: float | None = None):
    """Supply and demand step curves with the clearing point marked.

    `price` overlays a price from the optimisation, to compare it against the
    graphical one.
    """
    ax = ax or plt.gca()
    t = _resolve_hour(data, t)
    clearing = clear_graphical(data, t)
    supply, demand = supply_curve(data, t), demand_curve(data, t)

    ax.stairs(supply["cost"], [0, *supply["cum_p"]], label="Supply (marginal cost)",
              color="tab:blue", linewidth=2)
    ax.stairs(demand["bid"], [0, *demand["cum_p"]], label="Demand (bid price)",
              color="tab:orange", linewidth=2)
    ax.plot(clearing.quantity, clearing.price, "o", color="red", markersize=9, zorder=5)
    ax.hlines(clearing.price, 0, clearing.quantity, colors="red", linestyles="--")
    ax.annotate(f"Market clearing price\n{clearing.price:.2f} EUR/MWh",
                (clearing.quantity, clearing.price), textcoords="offset points",
                xytext=(0, 15), ha="left", color="red")
    if price is not None:
        ax.axhline(price, color="black", linestyle=":", linewidth=1,
                   label=f"Optimisation price ({price:.2f})")
    ax.set(xlim=(0, None), ylim=(0, None), xlabel="Quantity (MW)", ylabel="Price (EUR/MWh)",
           title=f"Market clearing, hour {t}")
    ax.legend()
    return ax


def plot_prices(result: ClearingResult, ax=None, label: str | None = None, **kwargs):
    """Hourly clearing prices; call twice on one axis to compare runs."""
    ax = ax or plt.gca()
    ax.step(result.prices.index, result.prices, where="mid", label=label, **kwargs)
    ax.set(xlabel="Hour", ylabel="Price (EUR/MWh)", ylim=(0, None), title="Hourly market-clearing prices")
    if label:
        ax.legend()
    return ax
