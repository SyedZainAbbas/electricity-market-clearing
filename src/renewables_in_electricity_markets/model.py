"""The market-clearing model shared by every step, and the graphical cross-check.

`build_model` is the copper-plate model of steps 1 and 2. The later steps start
from it and add what they need in their own file:

    step 3: drop the single balance, add angles, flows and one balance per node
    step 6: add reserve variables, headroom/footroom and the reserve requirements
    step 5: clears the day-ahead market with it unchanged

Producers are split into two sets, conventional units `m.G` and wind farms
`m.W`, because their profits are reported separately and step 5 only
perturbs the wind forecast.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from renewables_in_electricity_markets.data import MarketData

SOLVER = "gurobi"
# tight enough for the solver to see the 1e-6 tie-break nudges (data.TIE_BREAK)
SOLVER_OPTIONS = {"OptimalityTol": 1e-9}


# ----------------------------------------------------------------------------
# Optimisation-based market clearing
# ----------------------------------------------------------------------------
def build_model(data: MarketData) -> pyo.ConcreteModel:
    """Copper-plate welfare maximisation over the hours in `data` (not solved).

    Components a later step can rely on:
        sets       T (hours), G (conventional), W (wind), D (demands)
        variables  p_g[g, t], p_w[w, t], p_d[d, t]; with storage p_ch, p_dis, e [t]
        objective  obj = sum(bid * p_d) - sum(offer * p_g); wind and storage bid at zero
        limits     gen_limit, wind_limit, dem_limit: constraints, not variable
                   bounds, so their duals (scarcity rents) come back
        balance    balance[t]: consumption - production == 0. Under `maximize`
                   that sign makes the dual the positive market price.
        storage    soc[t] (dual = value of stored energy), e_max[t], e_min[t]
                   (duals = value of more / less reservoir in hour t) and e_final
        m.dual     declared here; Pyomo only fills it if it exists before solve()
    """
    conv = data.conventional.set_index("unit")

    m = pyo.ConcreteModel()
    m.T = pyo.Set(initialize=data.hours, ordered=True)
    m.G = pyo.Set(initialize=conv.index.tolist())
    m.W = pyo.Set(initialize=data.wind["farm"].unique().tolist())
    m.D = pyo.Set(initialize=data.demands["load_id"].unique().tolist())

    m.P_G = pyo.Param(m.G, initialize=conv["p_max"].to_dict())
    m.C_G = pyo.Param(
        m.G, initialize=conv["offer"].to_dict()
    )  # tie-broken, see data.TIE_BREAK
    m.P_W = pyo.Param(
        m.W, m.T, initialize=data.wind.set_index(["farm", "t"])["p_max"].to_dict()
    )
    m.P_D = pyo.Param(
        m.D, m.T, initialize=data.demands.set_index(["load_id", "t"])["p_max"].to_dict()
    )
    m.C_D = pyo.Param(
        m.D, m.T, initialize=data.demands.set_index(["load_id", "t"])["bid"].to_dict()
    )

    m.p_g = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals)
    m.p_w = pyo.Var(m.W, m.T, domain=pyo.NonNegativeReals)
    m.p_d = pyo.Var(m.D, m.T, domain=pyo.NonNegativeReals)

    # total social welfare: wind offers at zero cost and storage bids/offers at zero (as in the assignment),
    # so neither appears in the objective
    m.obj = pyo.Objective(
        expr=sum(m.C_D[d, t] * m.p_d[d, t] for d in m.D for t in m.T)
        - sum(m.C_G[g] * m.p_g[g, t] for g in m.G for t in m.T),
        sense=pyo.maximize,
    )

    m.gen_limit = pyo.Constraint(m.G, m.T, rule=lambda m, g, t: m.p_g[g, t] <= m.P_G[g])
    m.wind_limit = pyo.Constraint(
        m.W, m.T, rule=lambda m, w, t: m.p_w[w, t] <= m.P_W[w, t]
    )
    m.dem_limit = pyo.Constraint(
        m.D, m.T, rule=lambda m, d, t: m.p_d[d, t] <= m.P_D[d, t]
    )

    s = data.storage
    if s is not None:
        m.p_ch = pyo.Var(m.T, bounds=(0, s.p_charge_max))
        m.p_dis = pyo.Var(m.T, bounds=(0, s.p_discharge_max))
        # energy limits as constraints, not variable bounds, so their duals come back:
        # they are what makes the value of stored energy (dual of soc) change between hours
        m.e = pyo.Var(m.T)
        m.e_max = pyo.Constraint(m.T, rule=lambda m, t: m.e[t] <= s.capacity)
        m.e_min = pyo.Constraint(m.T, rule=lambda m, t: m.e[t] >= 0)

        def soc_rule(m, t):
            previous = s.e_init if t == m.T.first() else m.e[m.T.prev(t)]
            return (
                m.e[t]
                == previous + s.eta_charge * m.p_ch[t] - m.p_dis[t] / s.eta_discharge
            )

        m.soc = pyo.Constraint(m.T, rule=soc_rule)
        m.e_final = pyo.Constraint(expr=m.e[m.T.last()] >= s.e_final_min)

    def balance_rule(m, t):
        consumption = sum(m.p_d[d, t] for d in m.D)
        production = sum(m.p_g[g, t] for g in m.G) + sum(m.p_w[w, t] for w in m.W)
        if s is not None:
            consumption += m.p_ch[t]
            production += m.p_dis[t]
        return consumption - production == 0

    m.balance = pyo.Constraint(m.T, rule=balance_rule)

    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return m


def solve(m: pyo.ConcreteModel) -> pyo.ConcreteModel:
    """Solve in place with Gurobi; raise unless the result is optimal."""
    results = pyo.SolverFactory(SOLVER).solve(m, options=SOLVER_OPTIONS)
    condition = results.solver.termination_condition
    if condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"market clearing did not solve to optimality: {condition}")
    return m


@dataclass
class ClearingResult:
    """Outcomes of one run. Prices in EUR/MWh, powers in MW, money in EUR.

    prices       Series of balance duals: index t (copper plate) or (node, t) (step 3)
    welfare      social welfare at the true costs (the objective uses tie-broken offers)
    production   producer, kind ("conventional"/"wind"), node, t, p, cost, price, profit
    consumption  load_id, node, t, p_max, bid, p, price, utility
    storage      t, charge, discharge, e, price, profit (None without storage)
    """

    prices: pd.Series
    welfare: float
    production: pd.DataFrame
    consumption: pd.DataFrame
    storage: pd.DataFrame | None
    model: pyo.ConcreteModel

    @property
    def price(self) -> float:
        """The single clearing price, for a one-hour copper-plate run."""
        if len(self.prices) != 1:
            raise ValueError(f"{len(self.prices)} prices; use .prices")
        return self.prices.iloc[0]

    @property
    def producer_profit(self) -> pd.Series:
        """Profit per producer over the whole horizon, index (kind, producer)."""
        return self.production.groupby(["kind", "producer"])["profit"].sum()

    @property
    def demand_utility(self) -> pd.Series:
        """Utility per demand over the whole horizon."""
        return self.consumption.groupby("load_id")["utility"].sum()

    @property
    def storage_profit(self) -> float:
        return 0.0 if self.storage is None else self.storage["profit"].sum()

    @property
    def hourly(self) -> pd.DataFrame:
        """One row per hour: price, what was served and where it came from (copper plate)."""
        by_kind = self.production.pivot_table(
            index="t", columns="kind", values="p", aggfunc="sum"
        )
        summary = pd.DataFrame(
            {
                "price": self.prices,
                "served": self.consumption.groupby("t")["p"].sum(),
                "curtailed": self.consumption.groupby("t").apply(
                    lambda g: (g["p_max"] - g["p"]).sum(), include_groups=False
                ),
            }
        ).join(by_kind)
        if self.storage is not None:
            summary = summary.join(
                self.storage.set_index("t")[["charge", "discharge", "e"]]
            )
        return summary


def copper_plate_prices(m: pyo.ConcreteModel) -> pd.Series:
    """The dual of each hour's balance, i.e. the uniform price per hour."""
    return pd.Series({t: m.dual[m.balance[t]] for t in m.T}, name="price").rename_axis(
        "t"
    )


def extract(
    m: pyo.ConcreteModel, data: MarketData, prices: pd.Series
) -> ClearingResult:
    """Read a solved model into tables, settling everyone at the price of their node.

    `prices` is indexed by t (one price per hour) or by (node, t) (nodal prices).
    """
    conv = data.conventional.set_index("unit")
    wind_node = data.wind.drop_duplicates("farm").set_index("farm")["node"]

    def price_of(df):
        if prices.index.nlevels == 2:
            return [prices[(n, t)] for n, t in zip(df["node"], df["t"])]
        return df["t"].map(prices)

    production = pd.DataFrame(
        [
            {
                "producer": g,
                "kind": "conventional",
                "node": conv.at[g, "node"],
                "t": t,
                "p": m.p_g[g, t].value,
                "cost": conv.at[g, "cost"],
            }
            for g in m.G
            for t in m.T
        ]
        + [
            {
                "producer": w,
                "kind": "wind",
                "node": wind_node.at[w],
                "t": t,
                "p": m.p_w[w, t].value,
                "cost": 0.0,
            }
            for w in m.W
            for t in m.T
        ]
    )
    production["price"] = price_of(production)
    production["profit"] = (production["price"] - production["cost"]) * production["p"]

    consumption = data.demands.copy()
    consumption["p"] = [
        m.p_d[d, t].value for d, t in zip(consumption["load_id"], consumption["t"])
    ]
    consumption["price"] = price_of(consumption)
    consumption["utility"] = (consumption["bid"] - consumption["price"]) * consumption[
        "p"
    ]

    storage = None
    if data.storage is not None:
        storage = pd.DataFrame(
            {
                "t": list(m.T),
                "charge": [m.p_ch[t].value for t in m.T],
                "discharge": [m.p_dis[t].value for t in m.T],
                "e": [m.e[t].value for t in m.T],
            }
        )
        storage["price"] = storage["t"].map(prices)
        storage["profit"] = storage["price"] * (
            storage["discharge"] - storage["charge"]
        )

    welfare = (consumption["bid"] * consumption["p"]).sum() - (
        production["cost"] * production["p"]
    ).sum()
    return ClearingResult(prices, welfare, production, consumption, storage, m)


def clear_market(data: MarketData) -> ClearingResult:
    """Build, solve and read out the copper-plate market for `data`."""
    m = solve(build_model(data))
    return extract(m, data, copper_plate_prices(m))


# ----------------------------------------------------------------------------
# Graphical market clearing (a cross-check, no solver involved)
# ----------------------------------------------------------------------------
def _resolve_hour(data, t):
    if t is not None:
        return t
    hours = data.hours
    if len(hours) != 1:
        raise ValueError(f"data covers hours {hours[0]}-{hours[-1]}; pass t=<hour>")
    return hours[0]


def supply_curve(data: MarketData, t: int | None = None) -> pd.DataFrame:
    """Merit order in hour `t`: units sorted by cost, with label and cumulative capacity cum_p."""
    t = _resolve_hour(data, t)
    conventional = data.conventional.rename(columns={"unit": "id"}).assign(
        kind="conventional"
    )
    wind = (
        data.wind[data.wind["t"] == t]
        .rename(columns={"farm": "id"})
        .assign(kind="wind", cost=0.0)
    )
    curve = pd.concat([conventional, wind], ignore_index=True)[
        ["id", "kind", "node", "p_max", "cost"]
    ]
    curve = curve.sort_values("cost", kind="stable").reset_index(drop=True)
    curve["label"] = np.where(curve["kind"] == "wind", "W", "G") + curve["id"].astype(
        str
    )
    curve["cum_p"] = curve["p_max"].cumsum()
    return curve


def demand_curve(data: MarketData, t: int | None = None) -> pd.DataFrame:
    """Demand bids in hour `t`, sorted by decreasing price, with cumulative quantity cum_p."""
    t = _resolve_hour(data, t)
    curve = data.demands[data.demands["t"] == t].sort_values(
        "bid", ascending=False, kind="stable"
    )
    curve = curve.reset_index(drop=True)
    curve["cum_p"] = curve["p_max"].cumsum()
    return curve


@dataclass
class GraphicalClearing:
    quantity: float
    price_low: float
    price_high: float

    @property
    def price(self) -> float:
        return (self.price_low + self.price_high) / 2


def clear_graphical(data: MarketData, t: int | None = None) -> GraphicalClearing:
    """Intersect the supply and demand step curves in hour `t`.

    Returns the traded quantity and the price band [price_low, price_high]
    between the marginal accepted and the first rejected bid/offer; `.price` is
    its midpoint. When a unit is partly loaded the band has zero width.

    Ignores storage, which shifts energy between hours and so cannot be drawn on
    a single-hour diagram. With storage the price only matches the optimisation
    in hours where the unit neither charges nor discharges.
    """
    supply, demand = supply_curve(data, t), demand_curve(data, t)
    s_cum, s_price = supply["cum_p"].to_numpy(), supply["cost"].to_numpy()
    d_cum, d_price = demand["cum_p"].to_numpy(), demand["bid"].to_numpy()

    def price_at(cum, price, q, side, beyond):
        return np.append(price, beyond)[np.searchsorted(cum, q, side=side)]

    # both curves are constant between consecutive edges, so test one point per segment
    edges = np.unique([0, *s_cum, *d_cum])
    edges = edges[edges <= min(s_cum[-1], d_cum[-1])]
    mid = (edges[:-1] + edges[1:]) / 2
    traded = price_at(d_cum, d_price, mid, "left", -np.inf) >= price_at(
        s_cum, s_price, mid, "left", np.inf
    )
    if not traded.any():
        return GraphicalClearing(0.0, d_price[0], s_price[0])

    q = edges[1:][traded].max()
    low = max(
        price_at(s_cum, s_price, q, "left", -np.inf),
        price_at(d_cum, d_price, q, "right", -np.inf),
    )
    high = min(
        price_at(d_cum, d_price, q, "left", np.inf),
        price_at(s_cum, s_price, q, "right", np.inf),
    )
    return GraphicalClearing(q, low, high)
