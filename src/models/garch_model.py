import os
import sys
import pandas as pd
from arch import arch_model

# Ensure the script can find 'src' when running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.api.yahoo_finance_api import fetch_stock_data  # Now Python will find this module

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(BASE_DIR, "data", "stock_data.csv")

def garch_forecast(ticker: str):
    """
    Perform GARCH forecasting on historical stock data.

    :param ticker: Stock ticker symbol (e.g., 'AAPL')
    :return: Forecasted volatility
    """
    try:
        if not os.path.exists(DATA_PATH):
            print(f"Stock data missing. Fetching data for {ticker}...")
            stock_data = fetch_stock_data(ticker)
            os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
            stock_data.to_csv(DATA_PATH, index=False)

        data = pd.read_csv(DATA_PATH)
        if "close" not in data.columns:
            raise ValueError("Missing 'close' column in stock data.")

        # GARCH model fitting and forecasting
        returns = data["close"].pct_change().dropna()
        model = arch_model(returns, vol="Garch", p=1, q=1)
        model_fit = model.fit(disp="off")
        forecast = model_fit.forecast(horizon=10)
        return forecast.variance.values[-1, :].mean()
    except Exception as e:
        print(f"Error in GARCH forecasting: {e}")
        return None