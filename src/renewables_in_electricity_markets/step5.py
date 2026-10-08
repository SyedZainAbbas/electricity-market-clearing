"""Step 5: balancing market. Step 1 (one hour, no storage) plus real-time deviations.

The day-ahead market is cleared with the step-1 model. Then G9 trips and the
wind farms miss their forecast, and the flexible units cover the net imbalance
in a balancing market. Profits are settled under the one-price and the
two-price imbalance schemes.

Hour 14, not the step-1 hour 22: there G3 is the single marginal unit (158.5 of
350 MW), so it has headroom both ways. At hour 22 G6/G7/G11 tie at the margin.

Run:  python -m renewables_in_electricity_markets.step5
"""

import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo

from renewables_in_electricity_markets.data import load_market_data
from renewables_in_electricity_markets.model import clear_market, solve
from renewables_in_electricity_markets.plotting import save_figure, save_table

HOUR = 14
OUTAGE = 9  # 400 MW, fully dispatched, not a balancing provider
SHORT_FARMS = [1, 2, 7]  # produce 15 % below forecast, the rest 10 % above
SHORT, LONG = 0.15, 0.10
UP_MARKUP, DOWN_MARKDOWN = 0.10, 0.15  # share of the unit's own cost
C_SHED = 500.0  # EUR/MWh, load curtailment
TOL = 1e-6


def balancing_market(offers: pd.DataFrame, delta_p: float, total_load: float) -> tuple[pd.DataFrame, float, float]:
    """Clear the balancing market at least cost.

    offers: index unit; columns up_max, down_max (MW), c_up, c_down (EUR/MWh).
    delta_p: system imbalance in MW, positive = system short (needs up-regulation).

    The balance is written as supply of regulation == need under `minimize`, so
    its dual is +lambda_B. Returns (offers with columns up, down added, the
    balancing price lambda_B, load shed in MW).
    """
    m = pyo.ConcreteModel()
    m.I = pyo.Set(initialize=offers.index.tolist())
    m.p_up = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.p_down = pyo.Var(m.I, domain=pyo.NonNegativeReals)
    m.p_shed = pyo.Var(domain=pyo.NonNegativeReals)

    # pay for up, get paid back for down, shedding last
    m.obj = pyo.Objective(
        expr=sum(offers.at[i, "c_up"] * m.p_up[i] - offers.at[i, "c_down"] * m.p_down[i] for i in m.I)
        + C_SHED * m.p_shed,
        sense=pyo.minimize,
    )
    m.up_limit = pyo.Constraint(m.I, rule=lambda m, i: m.p_up[i] <= offers.at[i, "up_max"])
    m.down_limit = pyo.Constraint(m.I, rule=lambda m, i: m.p_down[i] <= offers.at[i, "down_max"])
    m.shed_limit = pyo.Constraint(expr=m.p_shed <= total_load)
    m.balance = pyo.Constraint(expr=sum(m.p_up[i] - m.p_down[i] for i in m.I) + m.p_shed == delta_p)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    solve(m)

    activated = offers.assign(up=[m.p_up[i].value for i in m.I], down=[m.p_down[i].value for i in m.I])
    return activated, m.dual[m.balance], m.p_shed.value


def settle(producers: pd.DataFrame, lam_da: float, lam_b: float, delta_p: float) -> pd.DataFrame:
    """Profit of every producer under the one-price and two-price schemes.

    producers: index label (G1, W1, ...); columns kind, cost, p_da, dev (actual -
    DA before balancing, MW), up, down (activated regulation, MW).

    Profit = DA revenue + balancing revenue + imbalance settlement - cost of what
    was actually produced. One-price settles every deviation at lambda_B.
    Two-price settles deviations that worsen the system imbalance at lambda_B
    and those that help it at lambda_DA.

    Adds columns produced, da_profit, imb_price_1, imb_price_2, profit_1, profit_2.
    """
    p = producers.copy()
    p["produced"] = p["p_da"] + p["dev"] + p["up"] - p["down"]
    p["da_profit"] = (lam_da - p["cost"]) * p["p_da"]
    base = lam_da * p["p_da"] + lam_b * (p["up"] - p["down"]) - p["cost"] * p["produced"]

    p["imb_price_1"] = lam_b
    worsens = p["dev"] * delta_p < 0
    p["imb_price_2"] = lam_da
    p.loc[worsens, "imb_price_2"] = lam_b

    p["profit_1"] = base + p["imb_price_1"] * p["dev"]
    p["profit_2"] = base + p["imb_price_2"] * p["dev"]
    return p


def run() -> dict:
    """Day-ahead, real-time deviations, balancing market and settlement.

    Keys: data, da (ClearingResult), lam_da, wind (forecast, p_da, actual, dev),
    delta_p, balancing (offers + activated up/down), lam_b, shed, producers
    (settle output), tso_net_1, tso_net_2 (TSO cash under each scheme).
    """
    data = load_market_data(hours=[HOUR], storage=None)
    da = clear_market(data)
    lam_da = da.price

    conv = data.conventional.set_index("unit")
    conv["p_da"] = da.production.query("kind == 'conventional'").set_index("producer")["p"]
    wind = data.wind.set_index("farm")[["node", "p_max"]].rename(columns={"p_max": "forecast"})
    wind["p_da"] = da.production.query("kind == 'wind'").set_index("producer")["p"]

    # what actually happens in real time
    wind["actual"] = wind["forecast"] * [1 - SHORT if w in SHORT_FARMS else 1 + LONG for w in wind.index]
    wind["dev"] = wind["actual"] - wind["p_da"]
    conv["dev"] = 0.0
    conv.loc[OUTAGE, "dev"] = -conv.at[OUTAGE, "p_da"]
    delta_p = -(conv["dev"].sum() + wind["dev"].sum())  # positive = system short

    # A unit can regulate up by its DA headroom and down by its DA output; that is
    # the "if possible, depending on their day-ahead schedule" in the brief.
    offers = conv[conv["flexible"] & (conv.index != OUTAGE)].copy()
    offers["up_max"] = offers["p_max"] - offers["p_da"]
    offers["down_max"] = offers["p_da"]
    offers["c_up"] = lam_da + UP_MARKUP * offers["cost"]
    offers["c_down"] = lam_da - DOWN_MARKDOWN * offers["cost"]
    total_load = data.demands["p_max"].sum()
    balancing, lam_b, shed = balancing_market(offers[["p_da", "up_max", "down_max", "c_up", "c_down"]],
                                              delta_p, total_load)

    producers = pd.concat([
        pd.DataFrame({"kind": "conventional", "cost": conv["cost"], "p_da": conv["p_da"], "dev": conv["dev"],
                      "up": balancing["up"].reindex(conv.index, fill_value=0.0),
                      "down": balancing["down"].reindex(conv.index, fill_value=0.0)}).rename(index=lambda g: f"G{g}"),
        pd.DataFrame({"kind": "wind", "cost": 0.0, "p_da": wind["p_da"], "dev": wind["dev"],
                      "up": 0.0, "down": 0.0}).rename(index=lambda w: f"W{w}"),
    ]).rename_axis("producer")
    producers = settle(producers, lam_da, lam_b, delta_p)

    # TSO cash: collects -price * dev from deviators, pays lambda_B * (up - down) to providers
    tso_paid = lam_b * (balancing["up"].sum() - balancing["down"].sum())
    return {
        "data": data, "da": da, "lam_da": lam_da, "wind": wind, "delta_p": delta_p,
        "balancing": balancing, "lam_b": lam_b, "shed": shed, "producers": producers,
        "tso_net_1": -(producers["imb_price_1"] * producers["dev"]).sum() - tso_paid,
        "tso_net_2": -(producers["imb_price_2"] * producers["dev"]).sum() - tso_paid,
    }


def check(r: dict) -> None:
    bal, lam_b, shed, delta_p = r["balancing"], r["lam_b"], r["shed"], r["delta_p"]
    assert abs(bal["up"].sum() - bal["down"].sum() + shed - delta_p) < TOL, "regulation must cover delta_p"

    # lambda_B must be the offer of the partly activated unit (catches a flipped dual)
    partial_up = bal[(bal["up"] > TOL) & (bal["up"] < bal["up_max"] - TOL)]
    partial_down = bal[(bal["down"] > TOL) & (bal["down"] < bal["down_max"] - TOL)]
    if shed > TOL:
        assert abs(lam_b - C_SHED) < TOL
    elif len(partial_up):
        assert abs(lam_b - partial_up["c_up"].iloc[0]) < TOL, f"lambda_B {lam_b} != marginal up offer"
    elif len(partial_down):
        assert abs(lam_b - partial_down["c_down"].iloc[0]) < TOL, f"lambda_B {lam_b} != marginal down offer"

    if shed < TOL:
        p = r["producers"]
        assert abs(r["tso_net_1"]) < 1e-4, f"one-price should be revenue-neutral, TSO net {r['tso_net_1']}"
        helping = p.loc[(p["dev"] * delta_p > 0), "dev"]
        expected = (lam_b - r["lam_da"]) * helping.abs().sum()
        assert abs(r["tso_net_2"] - expected) < 1e-4, f"two-price surplus {r['tso_net_2']} != {expected}"


def report(r: dict) -> None:
    """Write results/step5/."""
    p = r["producers"]
    save_table(pd.Series({
        "lambda_DA": r["lam_da"], "DA welfare": r["da"].welfare, "delta_P": r["delta_p"],
        "lambda_B": r["lam_b"], "load shed": r["shed"],
        "TSO net, one-price": r["tso_net_1"], "TSO net, two-price": r["tso_net_2"],
    }, name="value").to_frame(), "step5", "summary")
    save_table(r["wind"], "step5", "wind_deviations")
    save_table(r["balancing"], "step5", "balancing")
    save_table(p[["kind", "p_da", "dev", "up", "down", "da_profit", "imb_price_2", "profit_1", "profit_2"]],
               "step5", "profits")
    save_table(p.groupby("kind")[["da_profit", "profit_1", "profit_2"]].sum(), "step5", "profit_totals")

    changed = p[(p["dev"].abs() > TOL) | (p["up"] > TOL) | (p["down"] > TOL)]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(changed))
    for k, (col, label) in enumerate([("da_profit", "day-ahead only"), ("profit_1", "one-price"),
                                      ("profit_2", "two-price")]):
        ax.bar([i + (k - 1) * 0.27 for i in x], changed[col], width=0.27, label=label)
    ax.set_xticks(list(x), changed.index)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(ylabel="Profit (EUR)", title=f"Profits of producers that deviate or regulate, hour {HOUR}")
    ax.legend()
    save_figure(fig, "step5", "profits")


def main() -> None:
    r = run()
    check(r)
    report(r)
    print(f"step 5: lambda_DA {r['lam_da']:.2f}, delta_P {r['delta_p']:.2f} MW, lambda_B {r['lam_b']:.2f} EUR/MWh")


if __name__ == "__main__":
    main()
