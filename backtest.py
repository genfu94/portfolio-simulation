import numpy as np
import pandas as pd

from simulation import InvestmentSimulator


def backtest_portfolio(
    simulator: InvestmentSimulator,
    asset_data: list[pd.Series],
    weights: list[float],
    inflation_data: pd.Series = None,
    sampling_frequency: str = "M",
) -> np.array:
    series = [a.resample(sampling_frequency).last().pct_change() for a in asset_data]

    if inflation_data is not None:
        series += [inflation_data.resample("M").last()]

    assets_return = simulator.simulate(series)
    inflation_return = np.ones((assets_return.shape[0], 1))

    if inflation_data is not None:
        inflation_return = assets_return[:, -1]
        assets_return = assets_return[:, :-1]

    assets_return *= weights
    assets_return = assets_return.sum(axis=1)
    return assets_return / inflation_return
