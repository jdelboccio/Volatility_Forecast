import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str):
    """
    Fetch historical stock price data from Yahoo Finance.
    
    :param ticker: Stock ticker symbol (e.g., 'AAPL')
    :return: DataFrame with historical stock prices
    """
    try:
        stock = yf.Ticker(ticker)
        stock_data = stock.history(period="1y")  # Fetch 1 year of data
        stock_data.reset_index(inplace=True)
        stock_data.rename(columns={"Date": "date", "Close": "close"}, inplace=True)
        return stock_data[["date", "close"]]
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure
