"""Steps 1 and 2: copper-plate market, one hour and then 24 hours with storage.

Step 1 is step 2 with one hour and no storage, so both use the same model:

    step1 = load_market_data(hours=[STEP1_HOUR], storage=None)
    step2 = load_market_data(hours=range(1, 25), storage=Storage())

Step 2's main question is how prices form once storage is present. The
storage sets the price in an hour iff it is strictly between both of its power
bounds there (idle counts as a bound); then lambda_t = mu_t * eta_ch when
charging or mu_t / eta_dis when discharging, with mu_t the dual of the
state-of-charge constraint. `storage_price_audit` tests that hour by hour.

Run:  python -m renewables_in_electricity_markets.step1_2
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo

from renewables_in_electricity_markets.data import STEP1_HOUR, MarketData, Storage, load_market_data
from renewables_in_electricity_markets.model import ClearingResult, clear_graphical, clear_market
from renewables_in_electricity_markets.plotting import (
    plot_clearing,
    plot_demand_curve,
    plot_prices,
    plot_supply_curve,
    save_figure,
    save_table,
)

TOL = 1e-6
LARGE_STORAGE = Storage(capacity=2000.0, e_init=1000.0, e_final_min=1000.0)
# (power MW, energy MWh) for the size sweep; the storage starts and ends half full
STORAGE_SIZES = [(250, 500), (250, 700), (250, 1200), (250, 2000), (600, 2400)]
TOY_SIZES = [5.0, 10.0, 20.0, 25.0, 40.0]
# Step 2 runs without the unit tie-break. Across hours with the same price the
# storage schedule is not unique, and the 1e-6 nudges make the solver pick a
# different (equally optimal) one; the report is written around this one.
STEP2_TIE_BREAK = False


def storage_price_audit(data: MarketData, result: ClearingResult) -> pd.DataFrame:
    """Per hour: is the storage marginal, and does its own condition give the price?

    Columns: price, mu (dual of the SOC constraint), charge, discharge, e,
    ch_interior / dis_interior (strictly between 0 and the power limit),
    soc_binds (e at 0 or E), mu*eta_ch, mu/eta_dis, and explains_price
    (True/False in hours where the storage is interior, None otherwise).
    """
    s, m = data.storage, result.model
    mu = pd.Series({t: m.dual[m.soc[t]] for t in m.T})
    st = result.storage.set_index("t")

    audit = pd.DataFrame({
        "price": result.prices,
        "mu": mu,
        "charge": st["charge"],
        "discharge": st["discharge"],
        "e": st["e"],
    })
    audit["ch_interior"] = (st["charge"] > TOL) & (st["charge"] < s.p_charge_max - TOL)
    audit["dis_interior"] = (st["discharge"] > TOL) & (st["discharge"] < s.p_discharge_max - TOL)
    audit["soc_binds"] = np.isclose(st["e"], 0) | np.isclose(st["e"], s.capacity)
    audit["mu*eta_ch"] = mu * s.eta_charge
    audit["mu/eta_dis"] = mu / s.eta_discharge
    audit["explains_price"] = np.where(
        audit["dis_interior"], np.isclose(audit["price"], audit["mu/eta_dis"], atol=1e-4),
        np.where(audit["ch_interior"], np.isclose(audit["price"], audit["mu*eta_ch"], atol=1e-4), None))
    return audit.rename_axis("t")


def toy_storage(p_max: float, eta: float = 0.9) -> dict:
    """Two-hour system with a hand-computable answer (report, "A two-hour illustration").

    Cheap unit 120 MW at 10 EUR/MWh, expensive 100 MW at 30; demand 100 MW at
    night, 150 MW by day; a storage of p_max MW and p_max MWh. Built with the
    same conventions as build_model, so the duals carry the same signs.

    Returns charge (night), the two prices, storage profit, cheap unit profit,
    mu (night) and the solved model.
    """
    gen = {"cheap": (120.0, 10.0), "expensive": (100.0, 30.0)}
    demand = {1: 100.0, 2: 150.0}

    m = pyo.ConcreteModel()
    m.T = pyo.Set(initialize=[1, 2], ordered=True)
    m.G = pyo.Set(initialize=list(gen))
    m.p = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals)
    m.ch = pyo.Var(m.T, bounds=(0, p_max))
    m.dis = pyo.Var(m.T, bounds=(0, p_max))
    m.e = pyo.Var(m.T, bounds=(0, p_max))

    m.obj = pyo.Objective(expr=-sum(gen[g][1] * m.p[g, t] for g in m.G for t in m.T), sense=pyo.maximize)
    m.limit = pyo.Constraint(m.G, m.T, rule=lambda m, g, t: m.p[g, t] <= gen[g][0])
    m.soc = pyo.Constraint(m.T, rule=lambda m, t: m.e[t] == (
        0.0 if t == m.T.first() else m.e[m.T.prev(t)]) + eta * m.ch[t] - m.dis[t] / eta)
    m.balance = pyo.Constraint(m.T, rule=lambda m, t: demand[t] + m.ch[t] - (
        sum(m.p[g, t] for g in m.G) + m.dis[t]) == 0)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    pyo.SolverFactory("gurobi").solve(m)

    price = {t: m.dual[m.balance[t]] for t in m.T}
    return {
        "charge": m.ch[1].value,
        "price_night": price[1],
        "price_day": price[2],
        "storage_profit": sum(price[t] * (m.dis[t].value - m.ch[t].value) for t in m.T),
        "cheap_profit": sum((price[t] - gen["cheap"][1]) * m.p["cheap", t].value for t in m.T),
        "mu_night": m.dual[m.soc[1]],
        "model": m,
    }


def run() -> dict:
    """Clear steps 1 and 2 and the storage studies.

    Keys: step1_data, step1, graphical (step 1); step2_data, step2, no_storage,
    large_data, large, audit, audit_large, sizes, toy (step 2).
    """
    r = {}
    r["step1_data"] = load_market_data(hours=[STEP1_HOUR], storage=None)
    r["step1"] = clear_market(r["step1_data"])
    r["graphical"] = clear_graphical(r["step1_data"])

    day = {"hours": range(1, 25), "tie_break": STEP2_TIE_BREAK}
    r["step2_data"] = load_market_data(storage=Storage(), **day)
    r["step2"] = clear_market(r["step2_data"])
    r["no_storage"] = clear_market(load_market_data(storage=None, **day))
    r["large_data"] = load_market_data(storage=LARGE_STORAGE, **day)
    r["large"] = clear_market(r["large_data"])
    r["audit"] = storage_price_audit(r["step2_data"], r["step2"])
    r["audit_large"] = storage_price_audit(r["large_data"], r["large"])

    rows = []
    for power, energy in STORAGE_SIZES:
        s = Storage(p_charge_max=power, p_discharge_max=power, capacity=energy,
                    e_init=energy / 2, e_final_min=energy / 2)
        res = clear_market(load_market_data(storage=s, **day))
        rows.append({"power": power, "energy": energy,
                     "prices": ", ".join(f"{p:.2f}" for p in sorted(res.prices.round(2).unique())),
                     "storage_profit": res.storage_profit, "welfare": res.welfare})
    r["sizes"] = pd.DataFrame(rows)

    r["toy"] = pd.DataFrame(
        [{"size": size} | {k: v for k, v in toy_storage(size).items() if k != "model"} for size in TOY_SIZES]
    ).set_index("size")
    return r


def check(r: dict) -> None:
    """Identities that must hold; a failure means a modelling or extraction bug."""
    # graphical and optimisation prices agree (up to the 1e-6-per-unit tie-break)
    assert abs(r["graphical"].price - r["step1"].price) < 1e-4, "graphical vs optimisation price"
    assert abs(r["graphical"].quantity - r["step1"].consumption["p"].sum()) < 1e-4, "cleared quantity"

    for name in ("step1", "step2", "large"):
        res = r[name]
        total = res.producer_profit.sum() + res.demand_utility.sum() + res.storage_profit
        assert abs(total - res.welfare) < 1e-6 * abs(res.welfare), f"{name}: profits + utility {total} != welfare"

    for name in ("audit", "audit_large"):
        audit = r[name]
        interior = audit["explains_price"].notna()
        assert interior.any(), f"{name}: no interior hour, the check proves nothing"
        assert audit.loc[interior, "explains_price"].all(), f"{name}: storage condition fails"

    for name in ("step2", "large"):
        st = r[name].storage
        assert not ((st["charge"] > TOL) & (st["discharge"] > TOL)).any(), f"{name}: charges and discharges at once"

    toy = r["toy"]
    assert abs(toy.loc[25.0, "price_night"] - 30 * 0.9 * 0.9) < TOL, "toy: interior storage sets 24.3"
    assert abs(toy.loc[25.0, "storage_profit"]) < TOL, "toy: a marginal storage earns nothing"


def _label(df: pd.DataFrame) -> pd.Series:
    return np.where(df["kind"] == "wind", "W", "G") + df["producer"].astype(str)


def report(r: dict) -> None:
    """Write results/step1/ and results/step2/."""
    # --- step 1 -------------------------------------------------------------
    s1 = r["step1"]
    m = s1.model
    t = STEP1_HOUR
    prod = s1.production.assign(producer=_label(s1.production))
    rent = {f"G{g}": m.dual[m.gen_limit[g, t]] for g in m.G}
    prod["scarcity_rent"] = prod["producer"].map(rent)
    save_table(prod.set_index("producer")[["node", "cost", "p", "profit", "scarcity_rent"]], "step1", "producers")
    save_table(s1.consumption.set_index("load_id")[["node", "p", "bid", "utility"]], "step1", "demands")
    by_kind = s1.production.groupby("kind")["profit"].sum()
    save_table(pd.Series({
        "price": s1.price,
        "graphical price": r["graphical"].price,
        "cleared quantity": s1.consumption["p"].sum(),
        "demand offered": s1.consumption["p_max"].sum(),
        "welfare": s1.welfare,
        "conventional profit": by_kind["conventional"],
        "wind profit": by_kind["wind"],
        "consumer utility": s1.consumption["utility"].sum(),
    }, name="value").to_frame(), "step1", "summary")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    plot_supply_curve(r["step1_data"], ax=axes[0])
    plot_demand_curve(r["step1_data"], ax=axes[1])
    plot_clearing(r["step1_data"], ax=axes[2], price=s1.price)
    save_figure(fig, "step1", "clearing")

    # --- step 2 -------------------------------------------------------------
    s2, s0 = r["step2"], r["no_storage"]
    hourly = s2.hourly.join(s0.prices.rename("price_no_storage"))
    save_table(hourly, "step2", "hourly")

    profits = s2.producer_profit.reset_index()
    profits.index = _label(profits)
    profits = profits["profit"]
    profits["Storage"] = s2.storage_profit
    save_table(profits.to_frame(), "step2", "profits")

    def totals(res):
        by_kind = res.production.groupby("kind")["profit"].sum()
        return {"welfare": res.welfare, "conventional profit": by_kind["conventional"],
                "wind profit": by_kind["wind"], "storage profit": res.storage_profit,
                "consumer utility": res.consumption["utility"].sum(),
                "energy served": res.consumption["p"].sum()}

    save_table(pd.DataFrame({"with storage": totals(s2), "without storage": totals(s0)}), "step2", "summary")
    save_table(r["audit"], "step2", "storage_audit", decimals=4)
    save_table(r["audit_large"], "step2", "storage_audit_large", decimals=4)
    save_table(r["sizes"], "step2", "storage_sizes", index=False)
    save_table(r["toy"], "step2", "toy_storage")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_prices(s2, ax=ax1, label="with storage (500 MWh)")
    plot_prices(s0, ax=ax1, label="without storage", linestyle="--")
    plot_prices(r["large"], ax=ax1, label="large storage (2000 MWh)", linestyle=":")
    for res, label in ((s2, "500 MWh"), (r["large"], "2000 MWh")):
        ax2.plot(res.storage["t"], res.storage["e"], marker="o", ms=3, label=label)
    ax2.set(xlabel="Hour", ylabel="Stored energy (MWh)", title="State of charge")
    ax2.legend()
    save_figure(fig, "step2", "prices_and_storage")


def main() -> None:
    r = run()
    check(r)
    report(r)
    s1, s2 = r["step1"], r["step2"]
    print(f"step 1: price {s1.price:.2f} EUR/MWh, welfare {s1.welfare:,.2f} EUR")
    print(f"step 2: welfare {s2.welfare:,.2f} EUR, storage profit {s2.storage_profit:,.2f} EUR")


if __name__ == "__main__":
    main()
