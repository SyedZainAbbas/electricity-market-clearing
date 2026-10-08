"""Step 3: step 1 (one hour, no storage) on a DC network, nodal and zonal.

Sections
    1. Nodal market    copper plate + angles, flows, one balance per node
    2. Price formation which units are marginal, how every other nodal price is a
                       re-dispatch of them, and the congestion rent
    3. Sensitivity     cut two lines at a time to 60 % of their base-case flow
    4. Zonal market    one price per zone, exchanges limited by ATC, then the zonal
                       dispatch put on the real network to measure overloads

Network assumptions: uniform susceptance B = 500 pu (x = 0.002 pu, from the
brief), S_base = 100 MVA (assumed; it only rescales the angles), reference angle
at node 13 (the RTS-24 slack bus), wind-case line limits (data.load_lines).

Run:  python -m renewables_in_electricity_markets.step3
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo

from renewables_in_electricity_markets.data import STEP1_HOUR, MarketData, load_lines, load_market_data
from renewables_in_electricity_markets.model import ClearingResult, build_model, clear_market, extract, solve
from renewables_in_electricity_markets.plotting import save_figure, save_table

HOUR = STEP1_HOUR
S_BASE = 100.0  # MVA, assumed: the brief does not give one, line limits are in MW
B = 500.0       # pu, uniform susceptance from the brief (x = 0.002 pu)
REF = 13        # reference (slack) node
TOL = 1e-6

# sensitivity: the two lines of each case are cut to CUT x their base-case flow, so both bind
CUT = 0.6
CASES = {
    "A: 13-23, 12-23": [(13, 23), (12, 23)],  # both exits of node 23
    "B: 14-16, 15-21": [(14, 16), (15, 21)],  # two of the wind-case bottlenecks, tighter
    "C: 3-24, 15-24": [(3, 24), (15, 24)],    # north-to-south path through node 24
    "D: 16-17, 17-22": [(16, 17), (17, 22)],  # out of the cheap 17/18/21/22 pocket
    "E: 9-12, 10-12": [(9, 12), (10, 12)],    # two of the transformers into the south
}

# zonal: ATC starts at the summed capacity of the border lines (as the brief
# suggests) and is swept down
ATC_SCALES = [1.0, 0.5, 0.25, 0.1, 0.05]
SOUTH = set(range(1, 11))
LAYOUTS = {
    # 138 kV south vs 230 kV north: 3 farms (600 MW) vs 4 farms (900 MW)
    "equal": {n: "S" if n in SOUTH else "N" for n in range(1, 25)},
    # wind-heavy zone W (farms at 3, 5, 7, 16, 21, 22: 1300 MW) vs R (farm at 23: 200 MW)
    "unequal": {n: "R" if n in {11, 12, 13, 14, 19, 20, 23} else "W" for n in range(1, 25)},
}


def line_name(lines: pd.DataFrame, l: int) -> str:
    return f"{lines.at[l, 'from_node']}-{lines.at[l, 'to_node']}"


def _node_members(data: MarketData) -> dict:
    """Which conventional units, farms and loads sit at each node."""
    def group(df, col):
        return df.drop_duplicates(col).groupby("node")[col].apply(list).to_dict()
    return {"G": group(data.conventional, "unit"), "W": group(data.wind, "farm"), "D": group(data.demands, "load_id")}


# ----------------------------------------------------------------------------
# 1. Nodal market
# ----------------------------------------------------------------------------
def add_network(m: pyo.ConcreteModel, data: MarketData, lines: pd.DataFrame,
                capacity: pd.Series | None = None) -> pyo.ConcreteModel:
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
    m.flow = pyo.Var(m.L, m.T)   # MW, positive from -> to

    m.flow_def = pyo.Constraint(m.L, m.T, rule=lambda m, l, t: m.flow[l, t]
                                == S_BASE * B * (m.theta[from_node[l], t] - m.theta[to_node[l], t]))
    m.flow_max = pyo.Constraint(m.L, m.T, rule=lambda m, l, t: m.flow[l, t] <= m.F_L[l])
    m.flow_min = pyo.Constraint(m.L, m.T, rule=lambda m, l, t: -m.flow[l, t] <= m.F_L[l])
    # angles are only defined up to a constant, so pin the reference node
    m.ref = pyo.Constraint(m.T, rule=lambda m, t: m.theta[REF, t] == 0)

    def node_balance(m, n, t):
        consumption = sum(m.p_d[d, t] for d in members["D"].get(n, []))
        production = (sum(m.p_g[g, t] for g in members["G"].get(n, []))
                      + sum(m.p_w[w, t] for w in members["W"].get(n, [])))
        outflow = sum(m.flow[l, t] for l in m.L if from_node[l] == n)
        inflow = sum(m.flow[l, t] for l in m.L if to_node[l] == n)
        return consumption + outflow - inflow - production == 0

    m.balance = pyo.Constraint(m.N, m.T, rule=node_balance)
    return m


def clear_nodal(data: MarketData, lines: pd.DataFrame, capacity: pd.Series | None = None) -> ClearingResult:
    """Nodal market; `prices` is indexed by (node, t)."""
    m = solve(add_network(build_model(data), data, lines, capacity))
    prices = pd.Series({(n, t): m.dual[m.balance[n, t]] for n in m.N for t in m.T}, name="price")
    return extract(m, data, prices.rename_axis(["node", "t"]))


def line_table(m: pyo.ConcreteModel, lines: pd.DataFrame, t: int = HOUR) -> pd.DataFrame:
    """Per line: from_node, to_node, limit, flow and the signed dual mu.

    mu = dual(flow_max) - dual(flow_min); it is non-zero only for binding lines.
    With it, lambda_n = lambda_REF - sum_l mu_l PTDF[l, n].
    """
    table = lines[["from_node", "to_node"]].copy()
    table["limit"] = [pyo.value(m.F_L[l]) for l in m.L]
    table["flow"] = [m.flow[l, t].value for l in m.L]
    table["mu"] = [m.dual[m.flow_max[l, t]] - m.dual[m.flow_min[l, t]] for l in m.L]
    return table


def check_with_pandapower(result: ClearingResult, lines: pd.DataFrame, t: int = HOUR) -> float:
    """Largest gap (MW) between our flows and pandapower's DC power flow.

    pandapower gets only the nodal injections and finds the angles itself. If it
    reproduces our flows, the flow equations in the Pyomo model are right.
    """
    import pandapower as pp  # optional, only used for this cross-check

    z_base = 100.0**2 / S_BASE  # ohm, for vn_kv = 100 kV
    net = pp.create_empty_network(sn_mva=S_BASE)
    pp.create_buses(net, 24, vn_kv=100.0, index=list(range(1, 25)))
    for r in lines.itertuples():
        pp.create_line_from_parameters(net, from_bus=r.from_node, to_bus=r.to_node, length_km=1.0,
                                       r_ohm_per_km=0.0, x_ohm_per_km=z_base / B, c_nf_per_km=0.0,
                                       max_i_ka=100.0)
    for n, p in net_demand(result, t).items():
        pp.create_load(net, bus=n, p_mw=p)
    pp.create_ext_grid(net, bus=REF)
    pp.rundcpp(net)
    ours = line_table(result.model, lines, t)["flow"].to_numpy()
    return float(np.abs(net.res_line["p_from_mw"].to_numpy() - ours).max())


# ----------------------------------------------------------------------------
# 2. Price formation and congestion rent
# ----------------------------------------------------------------------------
def ptdf(lines: pd.DataFrame) -> pd.DataFrame:
    """PTDF[l, n]: MW on line l per MW injected at node n and withdrawn at REF."""
    b = S_BASE * B
    n_nodes = 24
    Bbus = np.zeros((n_nodes, n_nodes))
    A = np.zeros((len(lines), n_nodes))  # line-node incidence
    for i, r in enumerate(lines.itertuples()):
        f, t = r.from_node - 1, r.to_node - 1
        A[i, f], A[i, t] = 1, -1
        Bbus[f, f] += b
        Bbus[t, t] += b
        Bbus[f, t] -= b
        Bbus[t, f] -= b
    keep = [n for n in range(n_nodes) if n != REF - 1]
    X = np.zeros((n_nodes, n_nodes))
    X[np.ix_(keep, keep)] = np.linalg.inv(Bbus[np.ix_(keep, keep)])
    return pd.DataFrame(b * A @ X, index=lines.index, columns=range(1, n_nodes + 1))


def net_demand(result: ClearingResult, t: int = HOUR) -> pd.Series:
    """Consumption minus production per node (MW), all 24 nodes."""
    cons = result.consumption.query("t == @t").groupby("node")["p"].sum()
    prod = result.production.query("t == @t").groupby("node")["p"].sum()
    return cons.sub(prod, fill_value=0).reindex(range(1, 25), fill_value=0.0)


def marginal_units(data: MarketData, result: ClearingResult, t: int = HOUR) -> pd.DataFrame:
    """Every resource strictly between its limits, i.e. free to set the price at its node.

    Index: label (G2, W7, D4). Columns: node, offer (tie-broken offer, 0 for wind,
    bid for demand), p, p_max, lmp.
    """
    conv = data.conventional.set_index("unit")
    rows = []
    for r in result.production.query("t == @t").itertuples():
        if r.kind == "conventional":
            p_max, offer, label = conv.at[r.producer, "p_max"], conv.at[r.producer, "offer"], f"G{r.producer}"
        else:
            wind = data.wind
            p_max = wind.loc[(wind["farm"] == r.producer) & (wind["t"] == t), "p_max"].iloc[0]
            offer, label = 0.0, f"W{r.producer}"
        if TOL < r.p < p_max - TOL:
            rows.append((label, r.node, offer, r.p, p_max))
    for r in result.consumption.query("t == @t").itertuples():
        if TOL < r.p < r.p_max - TOL:
            rows.append((f"D{r.load_id}", r.node, r.bid, r.p, r.p_max))
    marginal = pd.DataFrame(rows, columns=["label", "node", "offer", "p", "p_max"]).set_index("label")
    marginal["lmp"] = marginal["node"].map(result.prices.xs(t, level="t"))
    return marginal


def price_recipe(marginal: pd.DataFrame, flows: pd.DataFrame, P: pd.DataFrame, lmp: pd.Series) -> pd.DataFrame:
    """How each nodal price is made of the marginal units' offers.

    Serving 1 extra MW at node n without changing the flow on any binding line
    needs a re-dispatch dp of the marginal units that solves
        sum_k dp_k = 1                              (energy balance)
        sum_k PTDF[l, node_k] dp_k = PTDF[l, n]     (each binding line l)
    and the price is its cost, sum_k offer_k dp_k. Some dp_k are negative: a unit
    is turned down to relieve a line.

    Returns one row per node: dp per marginal unit, "price from recipe", "lmp",
    and "price from duals" = lambda_REF - sum_l mu_l PTDF[l, n].
    """
    binding = flows.index[flows["mu"].abs() > TOL]
    M = np.vstack([np.ones(len(marginal))] + [P.loc[l, marginal["node"]].to_numpy() for l in binding])
    rows = {}
    for n in P.columns:
        rhs = np.concatenate([[1.0], [P.loc[l, n] for l in binding]])
        dp = np.linalg.solve(M, rhs)
        rows[n] = dict(zip(marginal.index, dp)) | {"price from recipe": dp @ marginal["offer"].to_numpy()}
    recipe = pd.DataFrame(rows).T.rename_axis("node")
    recipe["lmp"] = lmp
    recipe["price from duals"] = [lmp[REF] - sum(flows.at[l, "mu"] * P.at[l, n] for l in binding) for n in P.columns]
    return recipe


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
        "paid by load": paid_by_load,
        "paid to producers": paid_to_producers,
    }


# ----------------------------------------------------------------------------
# 3. Line-capacity sensitivity
# ----------------------------------------------------------------------------
def sensitivity(data: MarketData, lines: pd.DataFrame, base: ClearingResult) -> dict:
    """Clear each case of CASES on a fresh model; returns {case: dict(welfare, lmp, flows, cut)}."""
    base_flows = line_table(base.model, lines)
    line_id = {(r.from_node, r.to_node): l for l, r in lines.iterrows()}
    out = {"base": {"welfare": base.welfare, "lmp": base.prices.xs(HOUR, level="t"), "flows": base_flows, "cut": []}}
    for name, pairs in CASES.items():
        cut = [line_id[p] for p in pairs]
        capacity = lines["capacity"].copy()
        capacity[cut] = CUT * base_flows.loc[cut, "flow"].abs()
        res = clear_nodal(data, lines, capacity)
        out[name] = {"welfare": res.welfare, "lmp": res.prices.xs(HOUR, level="t"),
                     "flows": line_table(res.model, lines), "cut": cut}
    return out


# ----------------------------------------------------------------------------
# 4. Zonal market
# ----------------------------------------------------------------------------
def is_connected(nodes, lines: pd.DataFrame) -> bool:
    """Zones must be contiguous, otherwise "lines between zones" means little."""
    nodes, seen, todo = set(nodes), set(), [min(nodes)]
    while todo:
        n = todo.pop()
        seen.add(n)
        for r in lines.itertuples():
            for a, b in [(r.from_node, r.to_node), (r.to_node, r.from_node)]:
                if a == n and b in nodes and b not in seen:
                    todo.append(b)
    return seen == nodes


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
        production = sum(m.p_g[g, t] for g in in_zone("G", k)) + sum(m.p_w[w, t] for w in in_zone("W", k))
        exports = sum(m.x[a, b, t] for a, b in m.X if a == k) - sum(m.x[a, b, t] for a, b in m.X if b == k)
        return consumption + exports - production == 0

    m.balance = pyo.Constraint(m.Z, m.T, rule=zone_balance)
    solve(m)
    prices = pd.Series({(n, t): m.dual[m.balance[zone_of[n], t]] for n in zone_of for t in m.T}, name="price")
    return extract(m, data, prices.rename_axis(["node", "t"]))


def overloads(result: ClearingResult, lines: pd.DataFrame, P: pd.DataFrame) -> pd.Series:
    """MW above the limit per line when `result`'s dispatch flows on the real network."""
    injection = -net_demand(result)
    flows = P.to_numpy() @ injection.to_numpy()
    return (np.abs(flows) - lines["capacity"]).clip(lower=0)


def zonal_study(data: MarketData, lines: pd.DataFrame, nodal: ClearingResult, P: pd.DataFrame) -> pd.DataFrame:
    """Nodal reference plus every layout x ATC scale, one row each.

    Columns: layout, atc_scale, atc, exchange, prices, the welfare split
    (welfare, utility, conv_profit, wind_profit, rent) and the overloads of the
    dispatch on the real network (n_overloaded, max_overload, total_overload,
    overloaded).
    """
    def row(result, **info):
        split = welfare_split(result)
        over = overloads(result, lines, P)
        return info | {
            "welfare": split["welfare"], "utility": split["consumer utility"],
            "conv_profit": split["conventional profit"], "wind_profit": split["wind profit"],
            "rent": split["congestion rent"],
            "n_overloaded": int((over > TOL).sum()), "max_overload": over.max(), "total_overload": over.sum(),
            "overloaded": ", ".join(f"{line_name(lines, l)} (+{v:.0f})" for l, v in over[over > TOL].items()),
        }

    lmp = nodal.prices.xs(HOUR, level="t")
    rows = [row(nodal, layout="nodal", atc_scale=np.nan, prices=f"{lmp.min():.2f}-{lmp.max():.2f}")]
    for name, zone_of in LAYOUTS.items():
        for k in set(zone_of.values()):
            assert is_connected([n for n in zone_of if zone_of[n] == k], lines), f"{name}: zone {k} not contiguous"
        full_atc = borders(zone_of, lines)
        for scale in ATC_SCALES:
            atc = {k: v * scale for k, v in full_atc.items()}
            res = clear_zonal(data, zone_of, atc)
            m = res.model
            price = {k: m.dual[m.balance[k, HOUR]] for k in m.Z}
            rows.append(row(
                res, layout=name, atc_scale=scale,
                atc=", ".join(f"{a}-{b} {v:.0f}" for (a, b), v in atc.items()),
                exchange=", ".join(f"{a}->{b} {m.x[a, b, HOUR].value:.0f}" for a, b in m.X),
                prices=", ".join(f"{k} {price[k]:.2f}" for k in m.Z),
            ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Run, check, report
# ----------------------------------------------------------------------------
def run() -> dict:
    """Everything step 3 reports.

    Keys: data, lines, ptdf, copper (step 1 at this hour, for comparison),
    nodal (ClearingResult), lmp (Series by node), flows (line_table), marginal,
    recipe, welfare_split ({"nodal", "copper plate"}: dict), sensitivity, zonal.
    """
    r = {"data": load_market_data(hours=[HOUR], storage=None), "lines": load_lines()}
    data, lines = r["data"], r["lines"]
    r["ptdf"] = P = ptdf(lines)
    r["copper"] = clear_market(data)
    r["nodal"] = nodal = clear_nodal(data, lines)
    r["lmp"] = nodal.prices.xs(HOUR, level="t")
    r["flows"] = line_table(nodal.model, lines)
    r["marginal"] = marginal_units(data, nodal)
    r["recipe"] = price_recipe(r["marginal"], r["flows"], P, r["lmp"])
    r["welfare_split"] = {"nodal": welfare_split(nodal), "copper plate": welfare_split(r["copper"])}
    r["sensitivity"] = sensitivity(data, lines, nodal)
    r["zonal"] = zonal_study(data, lines, nodal, P)
    return r


def check(r: dict) -> None:
    flows, recipe, split = r["flows"], r["recipe"], r["welfare_split"]["nodal"]
    binding = flows[flows["mu"].abs() > TOL]

    # k binding lines leave k + 1 marginal resources, one per degree of freedom
    assert len(r["marginal"]) == len(binding) + 1, "expected one more marginal unit than binding lines"
    # both explanations reproduce every nodal price (catches a flipped line dual)
    assert (recipe["price from recipe"] - recipe["lmp"]).abs().max() < 1e-6, "re-dispatch recipe != lmp"
    assert (recipe["price from duals"] - recipe["lmp"]).abs().max() < 1e-6, "line duals != lmp"
    # welfare splits exactly, and the rent is what the binding lines earn
    parts = split["consumer utility"] + split["conventional profit"] + split["wind profit"] + split["congestion rent"]
    assert abs(parts - split["welfare"]) < 1e-6, "welfare split does not add up"
    assert abs((binding["mu"].abs() * binding["limit"]).sum() - split["congestion rent"]) < 1e-4, "rent != mu * limit"
    # zonal with a non-binding border is the copper plate
    full = r["zonal"].query("layout == 'equal' and atc_scale == 1.0").iloc[0]
    assert abs(full["welfare"] - r["copper"].welfare) < 1e-6, "zonal with full ATC should equal copper plate"
    try:
        gap = check_with_pandapower(r["nodal"], r["lines"])
    except ImportError:
        return
    assert gap < 1e-6, f"pandapower flows differ by {gap} MW"


def report(r: dict) -> None:
    """Write results/step3/."""
    lines = r["lines"]
    flows = r["flows"].copy()
    flows.index = [line_name(lines, l) for l in flows.index]
    save_table(r["lmp"].rename("lmp").to_frame(), "step3", "lmp")
    save_table(flows, "step3", "flows")
    save_table(flows[flows["mu"].abs() > TOL], "step3", "binding_lines")
    save_table(r["marginal"], "step3", "marginal_units")
    save_table(r["recipe"], "step3", "price_recipe", decimals=3)
    save_table(pd.DataFrame(r["welfare_split"]), "step3", "welfare_split")

    # sensitivity
    sens = r["sensitivity"]
    base_welfare = sens["base"]["welfare"]
    summary = pd.DataFrame({
        name: {
            "welfare": s["welfare"], "loss": base_welfare - s["welfare"],
            "lmp_min": s["lmp"].min(), "lmp_max": s["lmp"].max(),
            "binding": ", ".join(line_name(lines, l) for l in s["flows"].index
                                 if abs(s["flows"].at[l, "flow"]) > s["flows"].at[l, "limit"] - TOL),
        } for name, s in sens.items()
    }).T.rename_axis("case")
    save_table(summary, "step3", "sensitivity")
    cut_rows = [{"case": name, "line": line_name(lines, l), "new limit": s["flows"].at[l, "limit"],
                 "base flow": sens["base"]["flows"].at[l, "flow"], "flow": s["flows"].at[l, "flow"]}
                for name, s in sens.items() for l in s["cut"]]
    save_table(pd.DataFrame(cut_rows), "step3", "sensitivity_cut_lines", decimals=1, index=False)
    all_flows = pd.DataFrame({name: s["flows"]["flow"] for name, s in sens.items()})
    all_flows.index = [line_name(lines, l) for l in all_flows.index]
    save_table(all_flows, "step3", "sensitivity_flows")
    all_lmp = pd.DataFrame({name: s["lmp"] for name, s in sens.items()})
    save_table(all_lmp, "step3", "sensitivity_lmp", decimals=3)

    # zonal
    zonal = r["zonal"]
    save_table(zonal, "step3", "zonal", index=False)

    # figures
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(r["lmp"].index, r["lmp"], color="tab:blue")
    ax.axhline(r["copper"].price, color="black", linestyle=":", label=f"copper plate ({r['copper'].price:.2f})")
    ax.set(xlabel="Node", ylabel="Nodal price (EUR/MWh)", title=f"Nodal prices, hour {HOUR}", xticks=range(1, 25))
    ax.legend()
    save_figure(fig, "step3", "nodal_prices")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    names = list(CASES)
    ax1.bar(names, summary.loc[names, "loss"].astype(float), color="tab:blue")
    ax1.set(ylabel="Welfare loss vs base (EUR)", title="Cost of tightening two lines")
    ax1.tick_params(axis="x", rotation=30)
    for name, s in sens.items():
        base = name == "base"
        ax2.plot(s["lmp"].index, s["lmp"].values, marker="o", ms=3,
                 color="black" if base else None, lw=2 if base else 1, label=name)
    ax2.set(xlabel="Node", ylabel="Nodal price (EUR/MWh)", title="Nodal prices")
    ax2.legend(fontsize=7)
    save_figure(fig, "step3", "sensitivity")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, zone_of in LAYOUTS.items():
        sub = zonal[zonal["layout"] == name]
        for zone in sorted(set(zone_of.values())):
            price = [float(s.split(f"{zone} ")[1].split(",")[0]) for s in sub["prices"]]
            ax1.plot(sub["atc_scale"] * 100, price, marker="o", label=f"{name}, zone {zone}")
        ax2.plot(sub["atc_scale"] * 100, sub["total_overload"], marker="o", label=name)
    ax1.set(xlabel="ATC (% of border line capacity)", ylabel="Zonal price (EUR/MWh)", title="Zonal prices")
    ax2.set(xlabel="ATC (% of border line capacity)", ylabel="Sum of line overloads (MW)",
            title="Overloads on the real network")
    for ax in (ax1, ax2):
        ax.legend(fontsize=8)
        ax.invert_xaxis()
    save_figure(fig, "step3", "zonal")


def main() -> None:
    r = run()
    check(r)
    report(r)
    print(f"step 3: nodal welfare {r['nodal'].welfare:,.2f} EUR, "
          f"prices {r['lmp'].min():.2f}-{r['lmp'].max():.2f} EUR/MWh")


if __name__ == "__main__":
    main()
