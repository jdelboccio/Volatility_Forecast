import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1mo")  # Fetch 1 month of data
        if data.empty or 'Close' not in data.columns:
            raise ValueError("Invalid response from Yahoo Finance API")
        
        data.reset_index(inplace=True)
        data = data[['Date', 'Close']]
        data.columns = ['date', 'close']  # Ensure lowercase column names for consistency
        return data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure
