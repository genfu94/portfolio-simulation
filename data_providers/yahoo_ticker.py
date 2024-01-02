import pandas as pd
import yfinance as yf

from . import InvestmentDataProvider


class YahooDataProvider(InvestmentDataProvider):
    def __init__(self):
        super(YahooDataProvider, self).__init__("ticker")

    def _fetch_data(self, asset_name: str = None) -> pd.Series:
        data = yf.download(asset_name, progress=False).reset_index()
        data = data.rename(columns={"Date": "date", "Adj Close": "adj_close"})
        data = data.set_index("date")

        return data["adj_close"]
