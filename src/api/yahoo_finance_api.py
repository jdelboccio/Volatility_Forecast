import requests
import pandas as pd

API_KEY = "XYUCA96Q6KEN1T86"

def fetch_stock_data(symbol):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}&outputsize=compact"
    response = requests.get(url)
    data = response.json()

    if "Time Series (Daily)" not in data:
        raise ValueError("Error fetching stock data from Alpha Vantage.")

    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df = df.rename(columns={"4. close": "Close"})
    df["Close"] = df["Close"].astype(float)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()
