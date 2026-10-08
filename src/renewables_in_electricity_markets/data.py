"""Input data for every step: read the CSVs, clean them, return tidy tables.

Step 1 is step 2 with one hour and no storage, so one loader serves both:

    step1 = load_market_data(hours=[STEP1_HOUR], storage=None)
    step2 = load_market_data(hours=range(1, 25), storage=Storage())

Steps 3, 5 and 6 build on step 1 and use the same loader; step 3 adds the lines
from `load_lines`.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# hour used by step 1 and step 3: the peak-wind hour of 2025-03-21, capacity
# factor 0.561, so wind is a real share of supply
STEP1_HOUR = 22

# G1/G2 (13.32) and G6/G7/G11 (10.52) offer identical prices, so when one of them
# is marginal the split between them is arbitrary. Every offer is lowered by
# TIE_BREAK per unit number (higher number slightly cheaper), which makes the
# dispatch unique and reproduces the step-1 dispatch (G6 partly loaded). Welfare
# and profits are always computed with the true cost.
TIE_BREAK = 1e-6


@dataclass(frozen=True)
class Storage:
    """Energy storage unit (assignment step 2, eqs. 1-4).

    Sizes are on the order of a conventional unit. ``eta_charge < eta_discharge``
    follows the assignment; any pair with a round-trip efficiency below 1 rules
    out simultaneous charging and discharging.
    """

    p_charge_max: float = 250.0     # MW, P^ch
    p_discharge_max: float = 250.0  # MW, P^dis
    capacity: float = 500.0         # MWh, E
    eta_charge: float = 0.87        # eta^ch
    eta_discharge: float = 0.90     # eta^dis
    e_init: float = 250.0           # MWh at the start of the horizon (assumption)
    e_final_min: float = 250.0      # MWh required at the end, stops a free end-of-day drain


@dataclass
class MarketData:
    """Inputs of one market-clearing run.

    conventional : one row per unit (static)
        unit, node, p_max, cost, offer (cost with the tie-break), flexible,
        r_up_max, r_down_max (reserve capacity R+/R-, MW), c_up, c_down (reserve
        offer prices, EUR/MW)
    wind         : one row per (farm, t): farm, node, t, capacity_factor, p_max
    demands      : one row per (load_id, t): load_id, node, t, p_max, bid
    """

    conventional: pd.DataFrame
    wind: pd.DataFrame
    demands: pd.DataFrame
    storage: Storage | None = None
    wind_day: object = None  # the date the wind profile was taken from, for labelling

    @property
    def hours(self) -> list[int]:
        return sorted(self.demands["t"].unique().tolist())


def demand_bids(load_ids: Iterable[int], system_load: pd.Series, c_ref: float, seed: int = 42,
                markup: tuple[float, float] = (0.2, 1.2), peak_uplift: float = 0.3) -> pd.DataFrame:
    """Bid price of every load in every hour: bid[d, t] = c_ref * (1 + m_d) * phi_t.

    The brief asks for bids that are high relative to the conventional costs, so
    that most demand is served, and higher in peak hours.

    - c_ref: the most expensive conventional offer, so every bid is above every offer.
    - m_d ~ U(markup), drawn once per load (reproducible via `seed`): the load's
      own willingness to pay, 1.2x to 2.2x c_ref with the default markup.
    - phi_t = 1 + peak_uplift * (L_t - L_min) / (L_max - L_min) over `system_load`.
      Pass the FULL-day profile, so that an hour gets the same bids whether it
      is cleared alone (step 1) or as part of the day (step 2).

    Returns a DataFrame with columns load_id, t, bid (EUR/MWh, rounded to cents).
    """
    load_ids = list(load_ids)
    rng = np.random.default_rng(seed)
    willingness = pd.Series(1 + rng.uniform(*markup, size=len(load_ids)), index=load_ids)
    phi = 1 + peak_uplift * (system_load - system_load.min()) / (system_load.max() - system_load.min())
    return pd.DataFrame(
        [(d, t, round(c_ref * willingness[d] * phi[t], 2)) for d in load_ids for t in system_load.index],
        columns=["load_id", "t", "bid"],
    )


def wind_capacity_factors(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Hourly national wind capacity factor for 2025 (timestamp, capacity_factor)."""
    data_dir = Path(data_dir)

    def to_float(s):
        return s.str.replace(",", "", regex=False).astype(float)

    installed = pd.read_csv(data_dir / "Installed_generation_capacity_2025.csv", sep=";")
    installed_wind = to_float(installed.loc[0, [c for c in installed.columns if c.startswith("Wind")]]).sum()

    actual = pd.read_csv(data_dir / "Actual_generation_hourly.csv", sep=";")
    generated = sum(to_float(actual[c]) for c in actual.columns if c.startswith("Wind"))

    return pd.DataFrame({
        "timestamp": pd.to_datetime(actual["Start date"], format="%b %d, %Y %I:%M %p"),
        "capacity_factor": generated / installed_wind,
    })


def select_wind_day(factors: pd.DataFrame, day: str = "high") -> tuple[object, pd.Series]:
    """Pick one 24 h day of capacity factors.

    `day` is "high" or "low" (most / least volatile day of the year) or a date
    such as "2025-03-21". Days that do not have 24 hourly values, i.e. the
    daylight-saving days, are skipped.

    Returns the date and a Series of capacity factors indexed by t = 1..24.
    """
    by_day = factors.assign(date=factors["timestamp"].dt.date).groupby("date")["capacity_factor"]
    full_days = by_day.std()[by_day.count() == 24]

    if day in ("high", "low"):
        date = full_days.idxmax() if day == "high" else full_days.idxmin()
    else:
        date = pd.to_datetime(day).date()

    hourly = factors[factors["timestamp"].dt.date == date].copy()
    hourly["t"] = hourly["timestamp"].dt.hour + 1  # hours 0-23 -> t 1-24, matching load_profile.csv
    if len(hourly) != 24:
        raise ValueError(f"{date} has {len(hourly)} hourly values, expected 24")
    return date, hourly.set_index("t")["capacity_factor"]


def load_units(data_dir: Path = DATA_DIR, tie_break: bool = True) -> pd.DataFrame:
    """All 12 RTS-24 units with node, capacity, energy cost and reserve data.

    Columns: unit, node, p_max, cost, offer, flexible, r_up_max, r_down_max,
    c_up, c_down. `offer` is the cost lowered by TIE_BREAK per unit number, or
    equal to the cost with `tie_break=False`. Flexible units are those with a
    non-zero reserve capacity R+ or R-; they are the balancing (step 5) and
    reserve (step 6) providers.
    """
    data_dir = Path(data_dir)
    technical = pd.read_csv(data_dir / "gen_technical.csv")
    costs = pd.read_csv(data_dir / "gen_cost.csv")
    units = technical.merge(costs, on="unit").rename(columns={
        "p_max_mw": "p_max",
        "c_i_eur_per_mwh": "cost",
        "r_up_mw": "r_up_max",
        "r_down_mw": "r_down_max",
        "cu_i_eur_per_mwh": "c_up",
        "cd_i_eur_per_mwh": "c_down",
    })
    units["offer"] = units["cost"] - (TIE_BREAK if tie_break else 0.0) * units["unit"]
    units["flexible"] = (units["r_up_max"] > 0) | (units["r_down_max"] > 0)
    return units[["unit", "node", "p_max", "cost", "offer", "flexible",
                  "r_up_max", "r_down_max", "c_up", "c_down"]]


def load_lines(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Transmission lines indexed by line id 1..34.

    Columns: from_node, to_node, capacity (MW). The capacity is the wind-case
    limit from the RTS-24 document, where 13-23, 14-16 and 15-21 are reduced to
    create bottlenecks. All lines get the same susceptance (see step3.B).
    """
    lines = pd.read_csv(Path(data_dir) / "transmission_lines.csv")
    lines.index = pd.RangeIndex(1, len(lines) + 1, name="line")
    lines = lines.rename(columns={"capacity_mva_wind_case": "capacity"})
    return lines[["from_node", "to_node", "capacity"]]


def load_market_data(hours: Iterable[int] = range(1, 25), wind_day: str = "high",
                     storage: Storage | None = None, seed: int = 42, tie_break: bool = True,
                     data_dir: Path = DATA_DIR) -> MarketData:
    """Build the inputs for one run. `hours=[h]` with `storage=None` gives step 1.

    Any unit with zero marginal cost is moved out of the conventional set and
    treated as an extra wind farm (unit 10 becomes farm 7), so it follows the
    hourly capacity factor like the other farms. Wind farms offer their forecast,
    nameplate x capacity factor, at zero cost. Demand bids come from
    `demand_bids`, anchored to the most expensive conventional unit.
    """
    hours = list(hours)
    data_dir = Path(data_dir)

    units = load_units(data_dir, tie_break)
    # the assignment prices renewables at zero, so any zero-cost unit belongs in the wind set
    conventional = units[units["cost"] > 0].reset_index(drop=True)
    renewable_units = units[units["cost"] == 0]

    farms = (
        pd.read_csv(data_dir / "wind_farms.csv")
        .rename(columns={"wind_farm_id": "farm", "capacity_mw": "capacity"})[["farm", "node", "capacity"]]
    )
    farms = pd.concat([farms, pd.DataFrame({
        "farm": range(farms["farm"].max() + 1, farms["farm"].max() + 1 + len(renewable_units)),
        "node": renewable_units["node"].to_numpy(),
        "capacity": renewable_units["p_max"].to_numpy(),
    })], ignore_index=True)

    date, factors = select_wind_day(wind_capacity_factors(data_dir), wind_day)
    wind = farms.merge(factors.loc[hours].rename_axis("t").reset_index(), how="cross")
    wind["p_max"] = wind["capacity"] * wind["capacity_factor"]

    full_day = pd.read_csv(data_dir / "load_profile.csv").set_index("hour")["system_demand_mw"].rename_axis("t")
    shares = pd.read_csv(data_dir / "load_distribution.csv")
    bids = demand_bids(shares["load_id"], full_day, c_ref=conventional["cost"].max(), seed=seed)

    demands = shares.merge(full_day.loc[hours].rename("system_load").reset_index(), how="cross")
    demands["p_max"] = demands["system_load"] * demands["pct_of_system_load"] / 100
    demands = demands.merge(bids, on=["load_id", "t"])

    return MarketData(
        conventional=conventional,
        wind=wind[["farm", "node", "t", "capacity_factor", "p_max"]],
        demands=demands[["load_id", "node", "t", "p_max", "bid"]],
        storage=storage,
        wind_day=date,
    )
