import numpy as np

from backtest import backtest_portfolio
from data_providers.inflation_us import USInflationProvider
from data_providers.yahoo_ticker import YahooDataProvider
from simulation.monte_carlo import MonteCarloSimulator, NormalSamplingStrategy

if __name__ == "__main__":
    ticker_provider = YahooDataProvider()
    inflation_provider_us = USInflationProvider("./data/inflation_usa.tsv")

    sp500_data = ticker_provider.get_data("^SP500TR")
    ftse100_data = ticker_provider.get_data("^FTSE")

    us_inflation_data = inflation_provider_us.get_data()

    sampler = NormalSamplingStrategy(sample_size=30 * 12)
    simulator = MonteCarloSimulator(sampler, n_simulations=10000)

    results = backtest_portfolio(
        simulator, [sp500_data, ftse100_data], [0.5, 0.5], us_inflation_data
    )
    print(np.percentile(results, [10, 25, 50, 75, 90]))
