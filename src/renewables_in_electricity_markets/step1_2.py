"""Steps 1 and 2: copper-plate market, one hour and then 24 hours with storage.

Step 1 is step 2 with one hour and no storage, so both use the same model:

    step1 = load_market_data(hours=[STEP1_HOUR], storage=None)
    step2 = load_market_data(hours=range(1, 25), storage=Storage())

Step 2 compares three cases: no storage, the base 500 MW / 1000 MWh unit, and
a 500 MW / 2000 MWh unit (the same power, twice the energy), which is large
enough to set prices itself.

Run:  python -m renewables_in_electricity_markets.step1_2
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from renewables_in_electricity_markets.data import STEP1_HOUR, Storage, load_market_data
from renewables_in_electricity_markets.model import (
    ClearingResult,
    clear_graphical,
    clear_market,
)
from renewables_in_electricity_markets.plotting import (
    plot_clearing,
    plot_demand_curve,
    plot_prices,
    plot_supply_curve,
    save_figure,
    save_table,
)

TOL = 1e-6
LONG_STORAGE = Storage(
    capacity=2000.0, e_init=1000.0, e_final_min=1000.0
)  # 500 MW / 2000 MWh
CASES = {
    "no storage": None,
    "500 MW / 1000 MWh": Storage(),
    "500 MW / 2000 MWh": LONG_STORAGE,
}
COLORS = {
    "no storage": "tab:gray",
    "500 MW / 1000 MWh": "tab:blue",
    "500 MW / 2000 MWh": "tab:orange",
}
# Step 2 runs without the unit tie-break. Across hours with the same price the
# storage schedule is not unique, and the 1e-6 nudges make the solver pick a
# different (equally optimal) one; the results are for this one.
STEP2_TIE_BREAK = False


def run() -> dict:
    """Clear step 1 and the three cases of step 2.

    Keys: step1_data, step1, graphical (step 1); step2 (dict case -> ClearingResult).
    """
    r = {}
    r["step1_data"] = load_market_data(hours=[STEP1_HOUR], storage=None)
    r["step1"] = clear_market(r["step1_data"])
    r["graphical"] = clear_graphical(r["step1_data"])

    day = {"hours": range(1, 25), "tie_break": STEP2_TIE_BREAK}
    r["step2"] = {
        case: clear_market(load_market_data(storage=s, **day))
        for case, s in CASES.items()
    }
    return r


def check(r: dict) -> None:
    """Identities that must hold; a failure means a modelling or extraction bug."""
    # graphical and optimisation prices agree (up to the 1e-6-per-unit tie-break)
    assert (
        abs(r["graphical"].price - r["step1"].price) < 1e-4
    ), "graphical vs optimisation price"
    assert (
        abs(r["graphical"].quantity - r["step1"].consumption["p"].sum()) < 1e-4
    ), "cleared quantity"

    for name, res in [("step1", r["step1"]), *r["step2"].items()]:
        total = (
            res.producer_profit.sum() + res.demand_utility.sum() + res.storage_profit
        )
        assert abs(total - res.welfare) < 1e-6 * abs(
            res.welfare
        ), f"{name}: profits + utility != welfare"
        if res.storage is not None:
            st = res.storage
            both = (st["charge"] > TOL) & (st["discharge"] > TOL)
            assert not both.any(), f"{name}: charges and discharges at once"


def _label(df: pd.DataFrame) -> pd.Series:
    return np.where(df["kind"] == "wind", "W", "G") + df["producer"].astype(str)


def _totals(res: ClearingResult) -> dict:
    by_kind = res.production.groupby("kind")["profit"].sum()
    return {
        "welfare": res.welfare,
        "conventional profit": by_kind["conventional"],
        "wind profit": by_kind["wind"],
        "storage profit": res.storage_profit,
        "consumer utility": res.consumption["utility"].sum(),
        "energy served": res.consumption["p"].sum(),
    }


def _profits(res: ClearingResult) -> pd.Series:
    profits = res.producer_profit.reset_index()
    profits.index = _label(profits)
    profits = profits["profit"]
    profits["Storage"] = res.storage_profit
    return profits


def report(r: dict) -> None:
    """Write results/step1/ and results/step2/."""
    # --- step 1 -------------------------------------------------------------
    s1 = r["step1"]
    m = s1.model
    prod = s1.production.assign(producer=_label(s1.production))
    rent = {f"G{g}": m.dual[m.gen_limit[g, STEP1_HOUR]] for g in m.G}
    prod["scarcity_rent"] = prod["producer"].map(rent)
    save_table(
        prod.set_index("producer")[["node", "cost", "p", "profit", "scarcity_rent"]],
        "step1",
        "producers",
    )
    save_table(
        s1.consumption.set_index("load_id")[["node", "p", "bid", "utility"]],
        "step1",
        "demands",
    )
    by_kind = s1.production.groupby("kind")["profit"].sum()
    save_table(
        pd.Series(
            {
                "price": s1.price,
                "graphical price": r["graphical"].price,
                "cleared quantity": s1.consumption["p"].sum(),
                "demand offered": s1.consumption["p_max"].sum(),
                "welfare": s1.welfare,
                "conventional profit": by_kind["conventional"],
                "wind profit": by_kind["wind"],
                "consumer utility": s1.consumption["utility"].sum(),
            },
            name="value",
        ).to_frame(),
        "step1",
        "summary",
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_clearing(r["step1_data"], ax=ax)
    save_figure(fig, "step1", "clearing")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_supply_curve(r["step1_data"], ax=axes[0])
    plot_demand_curve(r["step1_data"], ax=axes[1])
    save_figure(fig, "step1", "supply_and_demand")

    # --- step 2 -------------------------------------------------------------
    s2 = r["step2"]
    save_table(
        pd.DataFrame({case: res.prices for case, res in s2.items()}), "step2", "prices"
    )
    for case, res in s2.items():
        if res.storage is not None:
            name = "storage_" + case.split("/")[1].strip().replace(
                " ", ""
            )  # storage_1000MWh
            save_table(res.storage.set_index("t"), "step2", name)
    save_table(
        pd.DataFrame({case: _profits(res) for case, res in s2.items()}).rename_axis(
            "producer"
        ),
        "step2",
        "profits",
    )
    save_table(
        pd.DataFrame({case: _totals(res) for case, res in s2.items()}),
        "step2",
        "summary",
    )

    # prices of the three cases, and the state of charge of the two storages
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6.5), sharex=True, height_ratios=[3, 2]
    )
    for case, res in s2.items():
        style = (
            {"linewidth": 5, "alpha": 0.4}
            if res.storage is None
            else {"linewidth": 1.8}
        )
        plot_prices(res, ax=ax1, label=case, color=COLORS[case], **style)
    ax1.set(xlabel="", title="Market-clearing prices, 24 hours")
    for case, res in s2.items():
        if res.storage is not None:
            st = res.storage
            ax2.plot(st["t"], st["e"], marker="o", ms=3, color=COLORS[case], label=case)
    ax2.set(
        xlabel="Hour",
        ylabel="Stored energy (MWh)",
        title="State of charge",
        xticks=range(1, 25),
        ylim=(0, None),
    )
    ax2.legend(fontsize=8)
    save_figure(fig, "step2", "prices_and_storage")


def main() -> None:
    r = run()
    check(r)
    report(r)
    s1 = r["step1"]
    print(f"step 1: price {s1.price:.2f} EUR/MWh, welfare {s1.welfare:,.2f} EUR")
    for case, res in r["step2"].items():
        print(
            f"step 2, {case}: welfare {res.welfare:,.2f} EUR, storage profit {res.storage_profit:,.2f} EUR"
        )


if __name__ == "__main__":
    main()
