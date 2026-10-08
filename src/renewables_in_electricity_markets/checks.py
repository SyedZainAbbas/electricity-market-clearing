"""Invariant checks for the whole package.

Run with:  python -m renewables_in_electricity_markets.checks

Plain asserts, no test framework needed. The functions are named `check_*` so
pytest would also collect them if it is ever added to the project.

The first two use a two-hour toy system whose answer can be worked out by hand
(step1_2.toy_storage), which makes them the ones that catch a flipped dual sign
or a misplaced efficiency. The step checks run each step and call its own
`check()`, which holds the identities specific to that step.
"""

import numpy as np

from renewables_in_electricity_markets import step1_2, step3, step5, step6
from renewables_in_electricity_markets.data import Storage, load_market_data
from renewables_in_electricity_markets.model import clear_market
from renewables_in_electricity_markets.step1_2 import toy_storage

TOL = 1e-6


# ----------------------------------------------------------------------------
# A two-hour system with a hand-computable answer
# ----------------------------------------------------------------------------
# Cheap     120 MW at 10 EUR/MWh      demand: night 100 MW, day 150 MW
# Expensive 100 MW at 30 EUR/MWh      so the cheap unit has 20 MW spare at night
def check_toy_small_storage_is_a_price_taker():
    """At 10 MW the storage is stuck at its power limit, so generators set both prices."""
    toy = toy_storage(10.0)
    assert abs(toy["charge"] - 10.0) < TOL, "a small storage should charge all it can"
    assert abs(toy["price_night"] - 10.0) < TOL, "night price should still be the cheap unit's cost"
    assert abs(toy["price_day"] - 30.0) < TOL, "day price should be the expensive unit's cost"
    assert toy["storage_profit"] > 0, "a price-taking storage must make money on the spread"


def check_toy_large_storage_sets_the_price():
    """At 40 MW it charges only the 20 MW of spare cheap capacity, so it goes interior."""
    toy = toy_storage(40.0)
    assert abs(toy["charge"] - 20.0) < TOL, "it should stop at the cheap unit's spare capacity"

    # interior -> its indifference condition pins the night price
    expected = 30.0 * 0.9 * 0.9  # day price x round-trip efficiency = 24.3
    assert abs(toy["price_night"] - expected) < TOL, f"night price should be {expected}, got {toy['price_night']}"
    # and the storage condition must reproduce it from the energy value
    assert abs(toy["price_night"] - toy["mu_night"] * 0.9) < TOL, "lambda_ch should equal mu * eta_ch"
    assert abs(toy["storage_profit"]) < TOL, f"a marginal storage earns zero rent, got {toy['storage_profit']}"


# ----------------------------------------------------------------------------
# Identities that must hold for any input
# ----------------------------------------------------------------------------
def check_dispatch_respects_its_limits():
    """Nothing may exceed its capacity, and every hour must balance."""
    data = load_market_data(hours=range(1, 25), storage=Storage())
    r = clear_market(data)

    gen_cap = data.conventional.set_index("unit")["p_max"]
    conv = r.production[r.production["kind"] == "conventional"]
    assert (conv["p"] <= conv["producer"].map(gen_cap) + 1e-6).all(), "a unit exceeded its capacity"

    wind_cap = data.wind.set_index(["farm", "t"])["p_max"]
    wind = r.production[r.production["kind"] == "wind"]
    offered = [wind_cap.at[(w, t)] for w, t in zip(wind["producer"], wind["t"])]
    assert (wind["p"] <= np.array(offered) + 1e-6).all(), "a farm exceeded its forecast"

    st = r.storage.set_index("t")
    for t in data.hours:
        served = r.consumption.loc[r.consumption["t"] == t, "p"].sum()
        produced = r.production.loc[r.production["t"] == t, "p"].sum()
        assert abs(served + st.at[t, "charge"] - produced - st.at[t, "discharge"]) < 1e-6, (
            f"hour {t} does not balance"
        )


# ----------------------------------------------------------------------------
# Each step's own checks
# ----------------------------------------------------------------------------
def check_step1_2():
    """Graphical = optimisation price, welfare split, storage conditions, no cycling."""
    step1_2.check(step1_2.run())


def check_step3():
    """Marginal units, price recipe, line duals, rent, pandapower flows."""
    step3.check(step3.run())


def check_step5():
    """Balancing price, regulation covers the imbalance, TSO cash per scheme."""
    step5.check(step5.run())


def check_step6():
    """Reserve requirements, headroom/footroom, reserve prices, joint >= sequential."""
    step6.check(step6.run())


def main():
    checks = [v for k, v in globals().items() if k.startswith("check_")]
    for check in checks:
        check()
        print(f"  ok  {check.__name__}")
    print(f"\n{len(checks)} checks passed")


if __name__ == "__main__":
    main()
