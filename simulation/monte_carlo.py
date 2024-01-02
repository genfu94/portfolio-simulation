from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from . import InvestmentSimulator


class SamplingStrategy(ABC):
    @abstractmethod
    def sample(self, series: list[pd.Series]) -> list[pd.Series]:
        pass


class NormalSamplingStrategy(SamplingStrategy):
    def __init__(self, sample_size: int = 12):
        self.sample_size = sample_size

    def sample(self, series: list[pd.Series]) -> list[pd.Series]:
        returns_df = pd.concat(series, join="inner", axis=1)

        mean = returns_df.mean()
        cov_matrix = returns_df.cov()

        rng = np.random.default_rng()
        return rng.multivariate_normal(
            mean, cov_matrix, size=self.sample_size, method="eigh"
        )


class MonteCarloSimulator(InvestmentSimulator):
    def __init__(self, sampler: SamplingStrategy, n_simulations: int):
        self.sampler = sampler
        self.n_simulations = n_simulations

    def simulate(
        self,
        assets: list[pd.Series],
    ) -> np.array:
        simulation_returns = []

        for _ in range(self.n_simulations):
            sampled_growth = self.sampler.sample(assets)
            cum_growth = (1 + sampled_growth).cumprod(axis=1)
            simulation_returns.append(cum_growth[-1].tolist())

        return np.array(simulation_returns)
