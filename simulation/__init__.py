from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class InvestmentSimulator(ABC):
    @abstractmethod
    def simulate(self, assets: list[pd.Series]) -> np.array:
        """
        Simulates future returns for a portfolio.

        Parameters:
        - assets (list[pd.Series]): List of assets represented as pandas Series.
          Each series should contain the percent change between each period.
          All assets should be sampled in the same way.

        Returns:
        - np.array: List of ending values for the assets.
        """
        pass
