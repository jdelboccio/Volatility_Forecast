import requests
import pandas as pd

ALPHA_VANTAGE_API_KEY = "WTMEPC2650Q2JW8O"  # Replace with your actual API key

def fetch_stock_data(ticker):
    """
    Fetch historical stock data from Alpha Vantage API.
    """
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)

        if response.status_code != 200:
            raise ValueError(f"HTTP error {response.status_code} when fetching stock data.")

        data = response.json()

        if "Time Series (Daily)" not in data:
            raise ValueError(f"Invalid API response: {data}. Check API key or rate limit.")

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        df = df.rename(columns={"4. close": "Close"})  # Rename Close column
        df.index = pd.to_datetime(df.index)  # Convert index to datetime
        df = df.sort_index()  # Sort by date

        return df

    except Exception as e:
        print(f"❌ Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()  # Return empty DataFrame if error occurs
