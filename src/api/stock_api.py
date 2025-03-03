import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker):
    """
    Fetches stock data from Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        DataFrame or None: Stock data if available, else None.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None
