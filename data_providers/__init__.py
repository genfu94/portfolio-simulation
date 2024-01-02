from abc import ABC, abstractmethod

import pandas as pd


class InvestmentDataProvider(ABC):
    def __init__(self, provider_type: str):
        """
        Initialize the data provider instance.

        Parameters:
        - provider_type (str): Type of the data provider.
          Possible values:
            - "ticker": For specific asset classes (AAPL, GOOGL, ...).
            - "asset_class": For generic or custom asset class data (Gold, SP500, ...).
            - "inflation": For inflation data provider.
        """
        self.provider_type = provider_type

    @abstractmethod
    def _fetch_data(self, asset_name: str = None) -> pd.Series:
        pass

    def get_data(self, asset_name: str = None) -> pd.Series:
        if asset_name is None and self.provider_type == "ticker":
            raise ValueError(
                "This SecurityDataProvider requires a ticker name to fetch data"
            )

        return self.__validate_data(self._fetch_data(asset_name))

    def __validate_data(self, data: pd.Series) -> pd.Series:
        if type(data) != pd.Series:
            raise TypeError("Data returned by asset is not a dataframe")

        if not pd.api.types.is_datetime64_any_dtype(data.index):
            raise ValueError(
                "Returned dataframe should have a 'date' column in 'YYYY-MM-DD format"
            )

        return data
