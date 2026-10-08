"""Step 6: reserve market. Step 1 (one hour, no storage) plus up/down reserves.

Three clearings of the same hour as step 5:

- no reserves: plain step 1, the reference;
- sequential (European): a reserve-only market first, then the day-ahead
  market with the awarded reserves fixed;
- joint (US): reserves and energy co-optimised in one problem.

The requirements are 15 % (up) and 10 % (down) of total demand; the providers
are the flexible units of step 5. Reserve offers are tie-broken like energy
offers (G1/G2 tie at 15 up; G1, G2, G6, G7, G12 at 14 down).

Run:  python -m renewables_in_electricity_markets.step6
"""

import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo

from renewables_in_electricity_markets.data import MarketData, load_market_data
from renewables_in_electricity_markets.model import ClearingResult, build_model, clear_market, extract, solve
from renewables_in_electricity_markets.model import copper_plate_prices
from renewables_in_electricity_markets.plotting import save_figure, save_table
from renewables_in_electricity_markets.step5 import HOUR

UP_SHARE, DOWN_SHARE = 0.15, 0.10  # of total demand, from the brief
TOL = 1e-6


def _units(data: MarketData) -> pd.DataFrame:
    """Conventional units with tie-broken reserve offers c_up_offer, c_down_offer.

    The nudge is the same one the energy offer carries (cost - offer), so
    switching the tie-break off in load_market_data switches it off here too.
    """
    conv = data.conventional.set_index("unit")
    nudge = conv["cost"] - conv["offer"]
    return conv.assign(c_up_offer=conv["c_up"] - nudge, c_down_offer=conv["c_down"] - nudge)


def requirements(data: MarketData) -> tuple[float, float]:
    """Up and down reserve requirement (MW) for the single hour in `data`."""
    total_demand = data.demands["p_max"].sum()
    return UP_SHARE * total_demand, DOWN_SHARE * total_demand


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def reserve_market(data: MarketData, req_up: float, req_down: float) -> pyo.ConcreteModel:
    """Reserve-only market (first stage of the sequential design), solved.

    Minimises the reserve bill subject to each unit's R+/R- limits and the two
    requirements. It does not know the energy dispatch, so it can only make sure
    that a unit's up and down reserve fit inside its range: r_up + r_down <= P_max.
    The requirements are written requirement - supply == 0 under `maximize`, so
    their duals are the positive reserve prices.
    """
    conv = _units(data)
    m = pyo.ConcreteModel()
    m.G = pyo.Set(initialize=conv.index.tolist())
    m.r_up = pyo.Var(m.G, domain=pyo.NonNegativeReals)
    m.r_down = pyo.Var(m.G, domain=pyo.NonNegativeReals)
    m.obj = pyo.Objective(expr=-sum(conv.at[g, "c_up_offer"] * m.r_up[g] + conv.at[g, "c_down_offer"] * m.r_down[g]
                                    for g in m.G), sense=pyo.maximize)
    m.r_up_limit = pyo.Constraint(m.G, rule=lambda m, g: m.r_up[g] <= conv.at[g, "r_up_max"])
    m.r_down_limit = pyo.Constraint(m.G, rule=lambda m, g: m.r_down[g] <= conv.at[g, "r_down_max"])
    m.req_up = pyo.Constraint(expr=req_up - sum(m.r_up[g] for g in m.G) == 0)
    m.req_down = pyo.Constraint(expr=req_down - sum(m.r_down[g] for g in m.G) == 0)
    m.r_fit = pyo.Constraint(m.G, rule=lambda m, g: m.r_up[g] + m.r_down[g] <= conv.at[g, "p_max"])
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return solve(m)


def _add_reserve_room(m: pyo.ConcreteModel, data: MarketData) -> None:
    """Reserve variables r_up[g, t], r_down[g, t] and the room the energy schedule
    must leave for them: p + r_up <= P_max (headroom), p - r_down >= 0 (footroom)."""
    p_max = data.conventional.set_index("unit")["p_max"]
    m.r_up = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals)
    m.r_down = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals)
    m.headroom = pyo.Constraint(m.G, m.T, rule=lambda m, g, t: m.p_g[g, t] + m.r_up[g, t] <= p_max[g])
    m.footroom = pyo.Constraint(m.G, m.T, rule=lambda m, g, t: m.p_g[g, t] - m.r_down[g, t] >= 0)


def fix_reserves(m: pyo.ConcreteModel, data: MarketData, r_up: pd.Series, r_down: pd.Series) -> pyo.ConcreteModel:
    """Day-ahead market after the reserve market: awarded reserves fixed (sequential)."""
    _add_reserve_room(m, data)
    for g in m.G:
        for t in m.T:
            m.r_up[g, t].fix(r_up[g])
            m.r_down[g, t].fix(r_down[g])
    return m


def add_reserves(m: pyo.ConcreteModel, data: MarketData, req_up: float, req_down: float) -> pyo.ConcreteModel:
    """Joint energy and reserve clearing (US style) on top of `build_model`.

    Adds the reserve variables with their R+/R- limits, headroom/footroom, the
    requirements req_up[t], req_down[t] (duals = reserve prices, same sign
    convention as the energy balance) and the reserve bill in the objective.
    """
    conv = _units(data)
    _add_reserve_room(m, data)
    m.r_up_limit = pyo.Constraint(m.G, m.T, rule=lambda m, g, t: m.r_up[g, t] <= conv.at[g, "r_up_max"])
    m.r_down_limit = pyo.Constraint(m.G, m.T, rule=lambda m, g, t: m.r_down[g, t] <= conv.at[g, "r_down_max"])
    m.req_up = pyo.Constraint(m.T, rule=lambda m, t: req_up - sum(m.r_up[g, t] for g in m.G) == 0)
    m.req_down = pyo.Constraint(m.T, rule=lambda m, t: req_down - sum(m.r_down[g, t] for g in m.G) == 0)
    reserve_bill = sum(conv.at[g, "c_up_offer"] * m.r_up[g, t] + conv.at[g, "c_down_offer"] * m.r_down[g, t]
                       for g in m.G for t in m.T)
    m.obj.expr = m.obj.expr - reserve_bill
    return m


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
def outcome(result: ClearingResult, data: MarketData, name: str,
            r_up: pd.Series, r_down: pd.Series, lam_up: float, lam_down: float) -> tuple[pd.Series, pd.DataFrame]:
    """Summary and per-unit table of one clearing, at true costs.

    Returns (summary Series named `name`: lambda_DA, lambda_up, lambda_down,
    served, wind, welfare_DA, reserve_cost, welfare_net, conv_profit,
    wind_profit; units DataFrame: cost, c_up, c_down, p, r_up, r_down,
    energy_profit, reserve_profit, profit).
    """
    conv = data.conventional.set_index("unit")
    prod = result.production
    units = conv[["cost", "c_up", "c_down"]].copy()
    units["p"] = prod.query("kind == 'conventional'").set_index("producer")["p"]
    units["r_up"], units["r_down"] = r_up, r_down
    units["energy_profit"] = prod.query("kind == 'conventional'").set_index("producer")["profit"]
    units["reserve_profit"] = ((lam_up - units["c_up"]) * units["r_up"]
                               + (lam_down - units["c_down"]) * units["r_down"])
    units["profit"] = units["energy_profit"] + units["reserve_profit"]

    reserve_cost = (units["c_up"] * units["r_up"] + units["c_down"] * units["r_down"]).sum()
    summary = pd.Series({
        "lambda_DA": result.price, "lambda_up": lam_up, "lambda_down": lam_down,
        "served": result.consumption["p"].sum(), "wind": prod.query("kind == 'wind'")["p"].sum(),
        "welfare_DA": result.welfare, "reserve_cost": reserve_cost, "welfare_net": result.welfare - reserve_cost,
        "conv_profit": units["profit"].sum(), "wind_profit": prod.query("kind == 'wind'")["profit"].sum(),
    }, name=name)
    return summary, units


def run() -> dict:
    """Clear no-reserves, sequential and joint.

    Keys: data, req_up, req_down, reserve_model (first stage of sequential),
    results ({case: ClearingResult}), summary (one column per case),
    units ({case: per-unit table}).
    """
    data = load_market_data(hours=[HOUR], storage=None)
    req_up, req_down = requirements(data)
    zero = pd.Series(0.0, index=data.conventional["unit"])

    # no reserves: step 1 at this hour
    none = clear_market(data)

    # sequential: reserve market, then day-ahead with the awarded reserves fixed
    rm = reserve_market(data, req_up, req_down)
    awarded_up = pd.Series({g: rm.r_up[g].value for g in rm.G})
    awarded_down = pd.Series({g: rm.r_down[g].value for g in rm.G})
    lam_up_seq, lam_down_seq = rm.dual[rm.req_up], rm.dual[rm.req_down]
    m = solve(fix_reserves(build_model(data), data, awarded_up, awarded_down))
    seq = extract(m, data, copper_plate_prices(m))

    # joint: co-optimised
    m = solve(add_reserves(build_model(data), data, req_up, req_down))
    joint = extract(m, data, copper_plate_prices(m))
    joint_up = pd.Series({g: m.r_up[g, HOUR].value for g in m.G})
    joint_down = pd.Series({g: m.r_down[g, HOUR].value for g in m.G})
    lam_up_joint, lam_down_joint = m.dual[m.req_up[HOUR]], m.dual[m.req_down[HOUR]]

    cases = {
        "no reserves": outcome(none, data, "no reserves", zero, zero, 0.0, 0.0),
        "sequential": outcome(seq, data, "sequential", awarded_up, awarded_down, lam_up_seq, lam_down_seq),
        "joint": outcome(joint, data, "joint", joint_up, joint_down, lam_up_joint, lam_down_joint),
    }
    return {
        "data": data, "req_up": req_up, "req_down": req_down, "reserve_model": rm,
        "results": {"no reserves": none, "sequential": seq, "joint": joint},
        "summary": pd.concat([s for s, _ in cases.values()], axis=1),
        "units": {name: u for name, (_, u) in cases.items()},
    }


def check(r: dict) -> None:
    conv = r["data"].conventional.set_index("unit")
    for name in ("sequential", "joint"):
        units = r["units"][name]
        assert abs(units["r_up"].sum() - r["req_up"]) < TOL and abs(units["r_down"].sum() - r["req_down"]) < TOL
        assert (units["p"] + units["r_up"] <= conv["p_max"] + TOL).all(), f"{name}: no headroom for r_up"
        assert (units["p"] >= units["r_down"] - TOL).all(), f"{name}: no footroom for r_down"
        assert (units.loc[~conv["flexible"], ["r_up", "r_down"]] < TOL).all().all(), f"{name}: inflexible reserve"

    # In the reserve-only market a partly awarded unit sets the price: its offer plus
    # the dual of its r_up + r_down <= P_max limit (non-zero when that limit binds, as
    # for G5, whose down price then carries the up margin it gives up). Catches a
    # flipped dual; the tie-break moves prices by at most 12e-6.
    rm, units = r["reserve_model"], r["units"]["sequential"]
    for col in ("up", "down"):
        lam = r["summary"].at[f"lambda_{col}", "sequential"]
        awarded = units[f"r_{col}"]
        partial = units[(awarded > TOL) & (awarded < conv[f"r_{col}_max"] - TOL)]
        assert any(abs(lam - c - abs(rm.dual[rm.r_fit[g]])) < 1e-4 for g, c in partial[f"c_{col}"].items()), \
            f"{col} reserve price {lam} not explained"

    # joint co-optimisation can only do as well or better than sequential
    s = r["summary"]
    assert s.at["welfare_net", "joint"] >= s.at["welfare_net", "sequential"] - 1e-4


def report(r: dict) -> None:
    """Write results/step6/."""
    save_table(r["summary"], "step6", "summary")
    for name, units in r["units"].items():
        save_table(units, "step6", "units_" + name.replace(" ", "_"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    s = r["summary"]
    s.loc[["lambda_DA", "lambda_up", "lambda_down"]].T.plot.bar(ax=ax1, rot=0)
    ax1.set(ylabel="Price (EUR/MWh or EUR/MW)", title=f"Energy and reserve prices, hour {HOUR}")
    flexible = r["data"].conventional.set_index("unit")["flexible"]
    units = pd.DataFrame({name: u.loc[flexible, "p"] for name, u in r["units"].items()})
    units.index = [f"G{g}" for g in units.index]
    units.plot.bar(ax=ax2, rot=0)
    ax2.set(ylabel="Energy dispatch (MW)", title="Dispatch of the flexible units")
    save_figure(fig, "step6", "prices_and_dispatch")


def main() -> None:
    r = run()
    check(r)
    report(r)
    s = r["summary"]
    print(f"step 6: lambda_DA none/seq/joint "
          f"{s.at['lambda_DA', 'no reserves']:.2f}/{s.at['lambda_DA', 'sequential']:.2f}/{s.at['lambda_DA', 'joint']:.2f}")


if __name__ == "__main__":
    main()
