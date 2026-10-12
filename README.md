# Renewables in Electricity Markets: Market Clearing on the IEEE 24-Bus System

Python models of electricity market clearing on the IEEE RTS 24-bus test system
with wind power. The repository works through the market-clearing problems of a
system-perspective course assignment:

- **Day-ahead market (copper plate):** a single hour, then 24 hours with an
  energy storage unit.
- **Network constraints:** nodal pricing with a DC power flow, and a two-zone
  market with limited transfer capacity.
- **Balancing market:** real-time imbalance from a generator outage and wind
  forecast errors, settled under one-price and two-price schemes.
- **Reserve market:** up- and down-reserve procured before the day-ahead market,
  cleared sequentially (European design) and jointly with energy (US design).

Each market is a linear program that maximises social welfare, written in Pyomo
and solved with Gurobi; prices are the dual variables of the balance
constraints.

## Repository layout

```
src/renewables_in_electricity_markets/
    data.py       load and prepare the input data
    model.py      copper-plate market model shared by all markets
    plotting.py   saving tables and figures
    step1_2.py    day-ahead market: single hour, and 24 hours with storage
    step3.py      nodal and zonal markets
    step5.py      balancing market and imbalance settlement
    step6.py      reserve market, sequential and joint
data/             input data (CSV)
results/          generated tables (CSV) and figures (PNG), one folder per step
```

File names follow the steps of the assignment; step 4 is a derivation without
code.

## Running

Requires Python 3.13, [uv](https://docs.astral.sh/uv/) and a Gurobi licence
(free for academic use).

```bash
uv sync
uv run renewables-in-electricity-markets                   # every step -> results/
uv run python -m renewables_in_electricity_markets.step3   # a single step
```

## Data

- Network, generators and loads: the updated IEEE RTS 24-bus system of
  C. Ordoudis, P. Pinson, J. M. Morales and M. Zugno (DTU, 2016), based on the
  IEEE Reliability Test System (Grigg et al., 1999).
- Wind profile: hourly German wind generation and installed capacity, 2025,
  Bundesnetzagentur | [SMARD.de](https://www.smard.de), licensed under CC BY 4.0.
- Demand bids: generated with a fixed random seed (see `data.py`).

## Credits

The assignment comes from the course *Renewables in Electricity Markets* by
Prof. Jalal Kazempour (DTU): <https://www.jalalkazempour.com/teaching/>.
This repository is an independent self-study; it is not affiliated with or
endorsed by DTU and is not an official solution.
