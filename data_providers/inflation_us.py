from datetime import date

import numpy as np
import pandas as pd

from . import InvestmentDataProvider


def generate_dates(years):
    all_dates = []

    for year in years:
        for month in range(1, 13):
            first_day_of_month = date(year, month, 1)
            all_dates.append(first_day_of_month)

    return all_dates


class USInflationProvider(InvestmentDataProvider):
    def __init__(self, data_path: str = None):
        super(USInflationProvider, self).__init__("inflation")
        self.data_path = data_path

    def _fetch_data(self, asset_name: str = None) -> pd.Series:
        data = pd.read_csv(self.data_path, sep="\t")
        inflation_values = np.delete(data.values, 0, axis=1).flatten() / 12
        index = generate_dates(range(2022, 1913, -1))
        data = pd.DataFrame({"date": index, "inflation": inflation_values})
        data["date"] = pd.to_datetime(data["date"])
        data = data.set_index("date")
        data = data.dropna()

        return data["inflation"].sort_index()
