"""Step 3: step 1 (hour 22, no storage) on a DC network, nodal and zonal.

Sections:
    1. Network model      copper plate + angles, flows, one balance per node
    2. Nodal vs copper    dispatch, prices, binding lines, welfare split, and every
                          nodal price rebuilt from the binding lines with PTDFs
    3. Line sensitivity   capacity of the most valuable binding line (14-16) from
                          20 % to 120 % of its limit
    4. Zonal vs nodal     north/south with the ATC swept down; zonal prices, welfare
                          split, and the feasibility of the zonal dispatch on the
                          real network (overloads and the re-dispatch to fix them)

Network assumptions: uniform susceptance B = 500 pu (x = 0.002 pu, from the
assignment), S_base = 100 MVA (assumed; it only rescales the angles), reference angle
at node 13 (the RTS-24 slack bus), wind-case line limits (data.load_lines).

Run:  python -m renewables_in_electricity_markets.step3
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo

from renewables_in_electricity_markets.data import (
    STEP1_HOUR,
    MarketData,
    load_lines,
    load_market_data,
)
from renewables_in_electricity_markets.model import (
    ClearingResult,
    build_model,
    clear_market,
    extract,
    solve,
)
from renewables_in_electricity_markets.plotting import save_figure, save_table

HOUR = STEP1_HOUR
S_BASE = 100.0  # MVA, assumed: the assignment does not give one, line limits are in MW
B = 500.0  # pu, uniform susceptance from the assignment (x = 0.002 pu)
REF = 13  # reference (slack) node
TOL = 1e-6

# line sensitivity: capacity of the swept line as a share of its own limit
SWEEP_SCALES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

# zonal: 138 kV south (nodes 1-10, 3 wind farms) vs 230 kV north (11-24, 4 farms).
# ATC starts at the summed capacity of the border lines (as the assignment
# suggests) and is cut down.
ZONES = {n: "S" if n <= 10 else "N" for n in range(1, 25)}
ATC_SCALES = [1.0, 0.25, 0.1, 0.05]  # cases shown in the bar charts
# finer grid for the zonal table (the cases above are on it)
ATC_GRID = [round(1 - 0.05 * k, 2) for k in range(20)]  # 1.00, 0.95, ..., 0.05

SPLIT = ["consumer utility", "conventional profit", "wind profit", "congestion rent"]
WELFARE_PARTS = SPLIT + ["welfare"]  # the four parts and their sum
COPPER_COLOR, NODAL_COLOR = "tab:orange", "tab:blue"


def line_name(lines: pd.DataFrame, l: int) -> str:
    return f"{lines.at[l, 'from_node']}-{lines.at[l, 'to_node']}"


def _node_members(data: MarketData) -> dict:
    """Which conventional units, farms and loads sit at each node."""

    def group(df, col):
        return df.drop_duplicates(col).groupby("node")[col].apply(list).to_dict()

    return {
        "G": group(data.conventional, "unit"),
        "W": group(data.wind, "farm"),
        "D": group(data.demands, "load_id"),
    }


# ----------------------------------------------------------------------------
# 1. Network model
# ----------------------------------------------------------------------------
def add_network(
    m: pyo.ConcreteModel,
    data: MarketData,
    lines: pd.DataFrame,
    capacity: pd.Series | None = None,
) -> pyo.ConcreteModel:
    """Replace the copper-plate balance of `build_model` with a DC network.

    Adds nodes N, lines L, limits F_L (from `capacity`, default lines.capacity),
    free angles theta[n, t] and flows flow[l, t] = S_base B (theta_from - theta_to),
    |flow| <= F_L, theta[REF] = 0, and balance[n, t]:
    demand + outflow - inflow - production == 0. Same sign convention as the
    copper plate, so the dual of balance[n, t] is the positive nodal price.
    """
    capacity = lines["capacity"] if capacity is None else capacity
    members = _node_members(data)
    from_node, to_node = lines["from_node"].to_dict(), lines["to_node"].to_dict()

    m.del_component(m.balance)
    m.N = pyo.Set(initialize=sorted(set(from_node.values()) | set(to_node.values())))
    m.L = pyo.Set(initialize=lines.index.tolist())
    m.F_L = pyo.Param(m.L, initialize=capacity.to_dict(), mutable=True)
    m.theta = pyo.Var(m.N, m.T)  # rad, free
    m.flow = pyo.Var(m.L, m.T)  # MW, positive from -> to

    m.flow_def = pyo.Constraint(
        m.L,
        m.T,
        rule=lambda m, l, t: m.flow[l, t]
        == S_BASE * B * (m.theta[from_node[l], t] - m.theta[to_node[l], t]),
    )
    m.flow_max = pyo.Constraint(m.L, m.T, rule=lambda m, l, t: m.flow[l, t] <= m.F_L[l])
    m.flow_min = pyo.Constraint(m.L, m.T, rule=lambda m, l, t: -m.flow[l, t] <= m.F_L[l])
    # angles are only defined up to a constant, so pin the reference node
    m.ref = pyo.Constraint(m.T, rule=lambda m, t: m.theta[REF, t] == 0)

    def node_balance(m, n, t):
        consumption = sum(m.p_d[d, t] for d in members["D"].get(n, []))
        production = sum(m.p_g[g, t] for g in members["G"].get(n, [])) + sum(
            m.p_w[w, t] for w in members["W"].get(n, [])
        )
        outflow = sum(m.flow[l, t] for l in m.L if from_node[l] == n)
        inflow = sum(m.flow[l, t] for l in m.L if to_node[l] == n)
        return consumption + outflow - inflow - production == 0

    m.balance = pyo.Constraint(m.N, m.T, rule=node_balance)
    return m


def clear_nodal(
    data: MarketData, lines: pd.DataFrame, capacity: pd.Series | None = None
) -> ClearingResult:
    """Nodal market; `prices` is indexed by (node, t)."""
    m = solve(add_network(build_model(data), data, lines, capacity))
    prices = pd.Series({(n, t): m.dual[m.balance[n, t]] for n in m.N for t in m.T}, name="price")
    return extract(m, data, prices.rename_axis(["node", "t"]))


def line_table(m: pyo.ConcreteModel, lines: pd.DataFrame, t: int = HOUR) -> pd.DataFrame:
    """Per line: from_node, to_node, limit, flow, the signed dual mu and binding.

    mu = dual(flow_max) - dual(flow_min): positive when the line is full in its
    from-to direction, negative when full the other way, zero when not binding.
    |mu| is what one more MW of capacity on that line would add to welfare.
    """
    table = lines[["from_node", "to_node"]].copy()
    table["limit"] = [pyo.value(m.F_L[l]) for l in m.L]
    table["flow"] = [m.flow[l, t].value for l in m.L]
    table["mu"] = [m.dual[m.flow_max[l, t]] - m.dual[m.flow_min[l, t]] for l in m.L]
    table["binding"] = table["flow"].abs() > table["limit"] - TOL
    return table


# ----------------------------------------------------------------------------
# 2. Nodal vs copper plate
# ----------------------------------------------------------------------------
def dispatch(result: ClearingResult, t: int = HOUR) -> pd.Series:
    """Output per producer (index G1.., W1..), in MW."""
    prod = result.production[result.production["t"] == t]
    labels = np.where(prod["kind"] == "conventional", "G", "W") + prod["producer"].astype(str)
    return pd.Series(prod["p"].to_numpy(), index=labels, name="p")


def welfare_split(result: ClearingResult) -> dict:
    """Welfare = consumer utility + producer profits + congestion rent.

    The rent is what consumers pay minus what producers receive; it is zero when
    everyone sees the same price.
    """
    prod, cons = result.production, result.consumption
    paid_by_load = (cons["price"] * cons["p"]).sum()
    paid_to_producers = (prod["price"] * prod["p"]).sum()
    return {
        "consumer utility": cons["utility"].sum(),
        "conventional profit": prod.loc[prod["kind"] == "conventional", "profit"].sum(),
        "wind profit": prod.loc[prod["kind"] == "wind", "profit"].sum(),
        "congestion rent": paid_by_load - paid_to_producers,
        "welfare": result.welfare,
    }


def ptdf(lines: pd.DataFrame) -> pd.DataFrame:
    """PTDF[l, n]: MW on line l when 1 MW is injected at node n and taken out at REF.

    DC power flow in matrix form. With b = S_base * B (MW per rad, the same for
    every line):
      - A (lines x nodes) is +1 at the from node and -1 at the to node, so
        flow = b * A @ theta;
      - the nodal injections are what leaves through the lines,
        injection = A.T @ flow = Bbus @ theta with Bbus = b * A.T @ A;
      - remove the REF row and column (theta_REF = 0) and invert: theta = X @ injection.
    So flow = (b * A @ X) @ injection and PTDF = b * A @ X. The REF column is
    zero: injecting and withdrawing at the same node moves nothing.
    """
    nodes = range(1, 25)
    b = S_BASE * B
    A = pd.DataFrame(0.0, index=lines.index, columns=nodes)
    for l, r in lines.iterrows():
        A.at[l, r.from_node], A.at[l, r.to_node] = 1.0, -1.0
    Bbus = b * A.T @ A  # diagonal: b x number of lines at the node; off-diagonal: -b per line
    keep = [n for n in nodes if n != REF]
    X = pd.DataFrame(0.0, index=nodes, columns=nodes)
    X.loc[keep, keep] = np.linalg.inv(Bbus.loc[keep, keep].to_numpy())
    return b * A @ X


def lmp_from_ptdf(lmp: pd.Series, flows: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    """Rebuild every nodal price from the reference price and the binding lines.

    lambda_n = lambda_REF - sum_l mu_l * PTDF[l, n]

    One more MW consumed at node n (supplied from REF) changes the flow on line
    l by -PTDF[l, n]. On a binding line that flow is worth mu_l per MW (its
    dual), so each binding line adds -mu_l * PTDF[l, n] to the price at n.
    Non-binding lines have mu = 0 and add nothing.

    Returns one row per node: the solver's lmp, the reference price, the
    contribution of each binding line (column = line name), and their sum
    "lmp from ptdf", which must equal the lmp.
    """
    binding = flows[flows["binding"]]
    table = pd.DataFrame({"lmp": lmp, f"lambda_{REF}": lmp[REF]})
    for l, row in binding.iterrows():
        table[f"{row.from_node}-{row.to_node}"] = -row["mu"] * P.loc[l]
    table["lmp from ptdf"] = table.drop(columns="lmp").sum(axis=1)
    return table.rename_axis("node")


# ----------------------------------------------------------------------------
# 3. Line sensitivity
# ----------------------------------------------------------------------------
def capacity_sweep(
    data: MarketData, lines: pd.DataFrame, line: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clear the nodal market with `line`'s capacity scaled by each SWEEP_SCALES.

    Returns (summary with one row per scale: capacity, nodal price range, prices
    at the line's two end nodes, the line's |dual|, welfare split and binding
    lines; nodal prices, one column per scale).
    """
    a, b = lines.at[line, "from_node"], lines.at[line, "to_node"]
    rows, lmps = [], {}
    for scale in SWEEP_SCALES:
        capacity = lines["capacity"].copy()
        capacity[line] *= scale
        res = clear_nodal(data, lines, capacity)
        lmp = res.prices.xs(HOUR, level="t")
        flows = line_table(res.model, lines)
        lmps[scale] = lmp
        rows.append({
            "scale": scale,
            "capacity": capacity[line],
            "lmp_min": lmp.min(),
            "lmp_max": lmp.max(),
            f"lmp_{a}": lmp[a],
            f"lmp_{b}": lmp[b],
            "dual": abs(flows.at[line, "mu"]),
            **welfare_split(res),
            "binding": ", ".join(line_name(lines, l) for l in flows.index[flows["binding"]]),
        })
    return pd.DataFrame(rows).set_index("scale"), pd.DataFrame(lmps).rename_axis("node")


# ----------------------------------------------------------------------------
# 4. Zonal vs nodal
# ----------------------------------------------------------------------------
def borders(zone_of: dict, lines: pd.DataFrame) -> dict:
    """ATC per zone pair (a, b) = summed capacity of the lines between them."""
    atc = {}
    for r in lines.itertuples():
        za, zb = zone_of[r.from_node], zone_of[r.to_node]
        if za != zb:
            key = tuple(sorted((za, zb)))
            atc[key] = atc.get(key, 0.0) + r.capacity
    return atc


def clear_zonal(data: MarketData, zone_of: dict, atc: dict) -> ClearingResult:
    """Copper plate inside each zone, exchange x[a, b] between zones limited by the ATC.

    `prices` is indexed by (node, t): every node carries its zone's price, so
    the result settles like a nodal one. The zonal model is also available as
    result.model, with zone prices as duals of balance[zone, t].
    """
    m = build_model(data)
    m.del_component(m.balance)
    m.Z = pyo.Set(initialize=sorted(set(zone_of.values())))
    m.X = pyo.Set(initialize=list(atc), dimen=2)  # one exchange per border, positive a -> b
    m.x = pyo.Var(m.X, m.T)
    m.atc_max = pyo.Constraint(m.X, m.T, rule=lambda m, a, b, t: m.x[a, b, t] <= atc[a, b])
    m.atc_min = pyo.Constraint(m.X, m.T, rule=lambda m, a, b, t: -m.x[a, b, t] <= atc[a, b])

    members = _node_members(data)

    def in_zone(kind, k):
        return [i for n, ids in members[kind].items() if zone_of[n] == k for i in ids]

    # same sign convention as before: consumption + exports - production == 0
    def zone_balance(m, k, t):
        consumption = sum(m.p_d[d, t] for d in in_zone("D", k))
        production = sum(m.p_g[g, t] for g in in_zone("G", k)) + sum(
            m.p_w[w, t] for w in in_zone("W", k)
        )
        exports = sum(m.x[a, b, t] for a, b in m.X if a == k) - sum(
            m.x[a, b, t] for a, b in m.X if b == k
        )
        return consumption + exports - production == 0

    m.balance = pyo.Constraint(m.Z, m.T, rule=zone_balance)
    solve(m)
    prices = pd.Series(
        {(n, t): m.dual[m.balance[zone_of[n], t]] for n in zone_of for t in m.T}, name="price"
    )
    return extract(m, data, prices.rename_axis(["node", "t"]))


def zone_average(result: ClearingResult) -> dict:
    """Load-weighted average price per zone: what the zone's consumers pay on average."""
    cons = result.consumption.assign(zone=result.consumption["node"].map(ZONES))
    paid = (cons["price"] * cons["p"]).groupby(cons["zone"]).sum()
    return (paid / cons.groupby("zone")["p"].sum()).to_dict()


def network_flows(data: MarketData, lines: pd.DataFrame, result: ClearingResult) -> pd.Series:
    """Line flows (MW) if `result`'s dispatch is put on the real network.

    A DC power flow with the same equations as the nodal market: every
    injection and withdrawal is fixed to the dispatch, the line limits are
    dropped, and the angles follow.
    """
    m = add_network(build_model(data), data, lines)
    for limit in (m.gen_limit, m.wind_limit, m.dem_limit, m.flow_max, m.flow_min):
        limit.deactivate()
    p = result.production.set_index(["kind", "producer", "t"])["p"]
    for g, t in m.p_g:
        m.p_g[g, t].fix(p["conventional", g, t])
    for w, t in m.p_w:
        m.p_w[w, t].fix(p["wind", w, t])
    served = result.consumption.set_index(["load_id", "t"])["p"]
    for d, t in m.p_d:
        m.p_d[d, t].fix(served[d, t])
    solve(m)
    return pd.Series({l: m.flow[l, HOUR].value for l in m.L}, name="flow")


def overloads(flows: pd.Series, lines: pd.DataFrame) -> pd.Series:
    """MW above the limit per line (0 where the line is within its limit)."""
    return (flows.abs() - lines["capacity"]).clip(lower=0)


def redispatch(zonal: ClearingResult, nodal: ClearingResult) -> pd.Series:
    """Change in output per producer (MW) to go from the zonal to the nodal dispatch.

    The nodal dispatch is the cheapest dispatch that respects every line for the
    same (fully served) demand, so it is the least-cost re-dispatch that makes a
    zonal outcome feasible. Positive: turn up; negative: turn down.
    """
    return (dispatch(nodal) - dispatch(zonal)).rename("redispatch")


def zonal_study(
    data: MarketData, lines: pd.DataFrame, nodal: ClearingResult
) -> tuple[pd.DataFrame, dict]:
    """Nodal reference plus the zonal market at every ATC scale, one row each.

    Returns (table, {atc_scale: re-dispatch Series}). Table columns: case,
    atc_scale, atc, exchange (MW, positive N -> S), price_N, price_S (zonal
    prices; for the nodal row the load-weighted nodal average per zone), the
    welfare split, the overloads of the dispatch on the real network
    (n_overloaded, total_overload, overloaded), and the re-dispatch to the
    nodal dispatch (redispatch_mw: MW turned up = MW turned down;
    redispatch_cost: the extra generation cost, negative if the zonal dispatch
    was the more expensive one).
    """
    def row(result, **info):
        over = overloads(network_flows(data, lines, result), lines)
        return info | welfare_split(result) | {
            "n_overloaded": int((over > TOL).sum()),
            "total_overload": over.sum(),
            "overloaded": ", ".join(
                f"{line_name(lines, l)} (+{v:.0f})" for l, v in over[over > TOL].items()
            ),
        }

    avg = zone_average(nodal)
    rows = [row(nodal, case="nodal", atc_scale=np.nan, atc=np.nan, exchange=np.nan,
                price_N=avg["N"], price_S=avg["S"], redispatch_mw=0.0, redispatch_cost=0.0)]
    moves = {}
    full_atc = borders(ZONES, lines)[("N", "S")]
    for scale in ATC_GRID:
        res = clear_zonal(data, ZONES, {("N", "S"): scale * full_atc})
        m = res.model
        moves[scale] = move = redispatch(res, nodal)
        rows.append(row(
            res, case="zonal", atc_scale=scale, atc=scale * full_atc,
            exchange=m.x["N", "S", HOUR].value,
            price_N=m.dual[m.balance["N", HOUR]], price_S=m.dual[m.balance["S", HOUR]],
            redispatch_mw=move.clip(lower=0).sum(),
            # same demand served, so the welfare difference is the cost difference
            redispatch_cost=res.welfare - nodal.welfare,
        ))
    return pd.DataFrame(rows), moves


# ----------------------------------------------------------------------------
# Run, check, report
# ----------------------------------------------------------------------------
def run() -> dict:
    """Everything step 3 reports.

    Keys: data, lines, copper (step 1 at this hour), nodal (ClearingResult),
    nodal_base (nodal with the original line limits), lmp (Series by node),
    flows (line_table), dispatch (copper plate vs nodal), ptdf, lmp_ptdf,
    welfare_split ({"copper plate", "nodal"}), sweep_line, sweep, sweep_lmp,
    zonal, redispatch.
    """
    r = {"data": load_market_data(hours=[HOUR], storage=None), "lines": load_lines()}
    data, lines = r["data"], r["lines"]

    # 1-2. nodal vs copper plate
    r["copper"] = clear_market(data)
    r["nodal"] = nodal = clear_nodal(data, lines)
    r["nodal_base"] = clear_nodal(data, load_lines("base"))  # "are nodal prices identical?"
    r["lmp"] = nodal.prices.xs(HOUR, level="t")
    r["flows"] = line_table(nodal.model, lines)
    r["dispatch"] = pd.DataFrame({"copper plate": dispatch(r["copper"]), "nodal": dispatch(nodal)})
    r["ptdf"] = ptdf(lines)
    r["lmp_ptdf"] = lmp_from_ptdf(r["lmp"], r["flows"], r["ptdf"])
    r["welfare_split"] = {"copper plate": welfare_split(r["copper"]), "nodal": welfare_split(nodal)}

    # 3. line sensitivity: the binding line with the largest dual
    binding = r["flows"][r["flows"]["binding"]]
    r["sweep_line"] = binding["mu"].abs().idxmax()
    r["sweep"], r["sweep_lmp"] = capacity_sweep(data, lines, r["sweep_line"])

    # 4. zonal vs nodal
    r["zonal"], r["redispatch"] = zonal_study(data, lines, nodal)
    return r


def check(r: dict) -> None:
    flows, split = r["flows"], r["welfare_split"]["nodal"]
    binding = flows[flows["binding"]]

    parts = sum(split[k] for k in SPLIT)
    assert abs(parts - split["welfare"]) < 1e-6, "welfare split does not add up"
    # the rent is what the binding lines earn: |dual| x capacity
    rent = (binding["mu"].abs() * binding["limit"]).sum()
    assert abs(rent - split["congestion rent"]) < 1e-4, "rent != sum |mu| * limit"
    # original limits: nothing binds, so one price everywhere
    base_lmp = r["nodal_base"].prices
    assert base_lmp.max() - base_lmp.min() < 1e-6, "original limits should give one price"
    # every nodal price follows from the reference price and the binding lines' duals
    gap = (r["lmp_ptdf"]["lmp from ptdf"] - r["lmp_ptdf"]["lmp"]).abs().max()
    assert gap < 1e-6, f"PTDF prices differ from the solver's by {gap}"
    # PTDF and the power-flow replay both reproduce the market's flows
    nodes = range(1, 25)
    prod = r["nodal"].production.groupby("node")["p"].sum().reindex(nodes, fill_value=0)
    cons = r["nodal"].consumption.groupby("node")["p"].sum().reindex(nodes, fill_value=0)
    assert (r["ptdf"] @ (prod - cons) - flows["flow"]).abs().max() < 1e-6, "PTDF flows"
    replay = network_flows(r["data"], r["lines"], r["nodal"])
    assert (replay - flows["flow"]).abs().max() < 1e-6, "network_flows != nodal flows"
    # zonal with a non-binding border is the copper plate, and its re-dispatch
    # to nodal costs exactly the welfare the network takes away
    full = r["zonal"].query("case == 'zonal' and atc_scale == 1.0").iloc[0]
    assert abs(full["welfare"] - r["copper"].welfare) < 1e-6, "full ATC should equal copper plate"
    assert abs(full["redispatch_cost"] - (r["copper"].welfare - r["nodal"].welfare)) < 1e-6


def report(r: dict) -> None:
    """Write results/step3/: tables, then the figures."""
    lines = r["lines"]
    flows = r["flows"].copy()
    flows.index = [line_name(lines, l) for l in flows.index]
    lmp = pd.DataFrame({"wind-case limits": r["lmp"],
                        "original limits": r["nodal_base"].prices.xs(HOUR, level="t")})
    save_table(lmp.rename_axis("node"), "step3", "lmp")
    save_table(flows, "step3", "flows")
    save_table(r["dispatch"].rename_axis("unit"), "step3", "dispatch")
    save_table(r["lmp_ptdf"], "step3", "lmp_from_ptdf")
    save_table(pd.DataFrame(r["welfare_split"]), "step3", "welfare_split")
    save_table(r["sweep"], "step3", "sensitivity")
    save_table(r["sweep_lmp"], "step3", "sensitivity_lmp")
    save_table(r["zonal"], "step3", "zonal", index=False)
    moves = pd.DataFrame({f"ATC {s:.0%}": r["redispatch"][s] for s in ATC_SCALES})
    save_table(moves.rename_axis("unit"), "step3", "redispatch")

    _plot_nodal_prices(r)
    _plot_welfare_change(
        {"nodal": r["welfare_split"]["nodal"]},
        reference=r["welfare_split"]["copper plate"],
        colors=[NODAL_COLOR],
        title="Change in the welfare split, nodal vs copper plate",
        ylabel="Change vs copper plate (EUR)",
        name="welfare_split",
    )
    _plot_sensitivity(r)
    _plot_zonal_prices(r)
    zonal = _bar_cases(r["zonal"])
    cases = {
        f"zonal, ATC {row.atc_scale:.0%}": zonal.loc[i, WELFARE_PARTS].to_dict()
        for i, row in zonal.iloc[1:].iterrows()
    }
    _plot_welfare_change(
        cases,
        reference=zonal.iloc[0][WELFARE_PARTS].to_dict(),  # the nodal row
        colors=plt.cm.Oranges(np.linspace(0.85, 0.35, len(cases))),
        title="Change in the welfare split, zonal vs nodal",
        ylabel="Change vs nodal (EUR)",
        name="zonal_welfare_split",
    )


def _plot_nodal_prices(r: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(r["lmp"].index, r["lmp"], color=NODAL_COLOR)
    ax.axhline(r["copper"].price, color="black", linestyle=":",
               label=f"copper plate ({r['copper'].price:.2f})")
    ax.set(xlabel="Node", ylabel="Nodal price (EUR/MWh)", title=f"Nodal prices, hour {HOUR}",
           xticks=range(1, 25))
    ax.legend()
    save_figure(fig, "step3", "nodal_prices")


def _plot_welfare_change(
    cases: dict, reference: dict, colors: list, title: str, ylabel: str, name: str
) -> None:
    """Grouped bars: one group per welfare component, one bar per case.

    Each bar is the case's value minus the reference's value of the same
    component, in EUR (positive: the case gains). Absolute changes rather than
    percentages, because the copper-plate congestion rent is zero, and because
    the component changes then add up to the welfare change in the last group.
    """
    fig, ax = plt.subplots(figsize=(10 + 0.6 * len(cases), 4.2))
    x = np.arange(len(WELFARE_PARTS))
    width = 0.8 / len(cases)
    for k, (case, split) in enumerate(cases.items()):
        values = [round(split[c] - reference[c], 6) + 0.0 for c in WELFARE_PARTS]  # + 0.0: no '-0'
        bars = ax.bar(x + (k - (len(cases) - 1) / 2) * width, values, width,
                      color=colors[k], label=case)
        ax.bar_label(bars, fmt="%+.0f", fontsize=6.5, padding=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(len(SPLIT) - 0.5, color="gray", linestyle=":", linewidth=1)  # parts | total
    ax.set_xticks(x, [c.capitalize() for c in SPLIT] + ["Social welfare\n(sum)"])
    ax.set(ylabel=ylabel, title=title)
    ax.margins(y=0.12)
    ax.legend(fontsize=8)
    save_figure(fig, "step3", name)


def _plot_sensitivity(r: dict) -> None:
    lines, sweep = r["lines"], r["sweep"]
    name = line_name(lines, r["sweep_line"])
    a, b = lines.at[r["sweep_line"], "from_node"], lines.at[r["sweep_line"], "to_node"]
    pct = sweep.index * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1.fill_between(pct, sweep["lmp_min"], sweep["lmp_max"], color=NODAL_COLOR, alpha=0.15,
                     label="range of nodal prices")
    ax1.plot(pct, sweep[f"lmp_{a}"], marker="o", label=f"node {a}")
    ax1.plot(pct, sweep[f"lmp_{b}"], marker="o", label=f"node {b}")
    ax1.axhline(r["copper"].price, color="black", linestyle=":", label="copper plate")
    ax1.set(xlabel=f"Capacity of line {name} (% of its limit)", ylabel="Price (EUR/MWh)",
            title="Nodal prices")
    ax1.legend(fontsize=8)
    ax2.plot(pct, sweep["dual"], marker="o", color="tab:red")
    ax2.set(xlabel=f"Capacity of line {name} (% of its limit)", ylabel="Line dual (EUR/MWh)",
            title=f"Value of 1 MW more on {name}")
    save_figure(fig, "step3", "sensitivity")


def _case_labels(zonal: pd.DataFrame) -> list[str]:
    return ["nodal\n(zone average)"] + [
        f"zonal\nATC {s:.0%} ({a:.0f} MW)" for s, a in zip(zonal["atc_scale"][1:], zonal["atc"][1:])
    ]


def _bar_cases(zonal: pd.DataFrame) -> pd.DataFrame:
    """The nodal row and the ATC_SCALES rows, for the bar charts."""
    return zonal[(zonal["case"] == "nodal") | zonal["atc_scale"].isin(ATC_SCALES)].reset_index(drop=True)


def _plot_zonal_prices(r: dict) -> None:
    zonal = _bar_cases(r["zonal"])
    x = np.arange(len(zonal))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for k, (zone, color) in enumerate((("N", NODAL_COLOR), ("S", COPPER_COLOR))):
        bars = ax.bar(x + (k - 0.5) * 0.38, zonal[f"price_{zone}"], 0.38, color=color,
                      label="north" if zone == "N" else "south")
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks(x, _case_labels(zonal), fontsize=8)
    ax.set(ylabel="Price (EUR/MWh)", title="Zonal prices and load-weighted nodal prices per zone")
    ax.legend()
    save_figure(fig, "step3", "zonal_prices")


def main() -> None:
    r = run()
    check(r)
    report(r)
    print(f"step 3: nodal welfare {r['nodal'].welfare:,.2f} EUR, "
          f"prices {r['lmp'].min():.2f}-{r['lmp'].max():.2f} EUR/MWh")


if __name__ == "__main__":
    main()
