"""Market clearing for an IEEE 24-bus system (DTU Renewables in Electricity Markets, Assignment 1).

    data.py      inputs           model.py     shared market-clearing model
    plotting.py  tables/figures   step*.py     one file per step, each with run/check/report

`main()` runs every step, checks it and writes results/<step>/.
"""


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")  # files only, no windows

    from renewables_in_electricity_markets import step1_2, step3, step5, step6

    for step in (step1_2, step3, step5, step6):
        step.main()
