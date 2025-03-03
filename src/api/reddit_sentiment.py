import requests
import pandas as pd

PUSHSHIFT_URL = "https://api.pushshift.io/reddit/search/comment/"

def fetch_reddit_sentiment(ticker):
    """
    Fetches recent Reddit comments mentioning the stock ticker.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        DataFrame: Processed sentiment data.
    """
    try:
        params = {"q": ticker, "limit": 100, "sort": "desc"}
        response = requests.get(PUSHSHIFT_URL, params=params)
        data = response.json()["data"]

        df = pd.DataFrame(data)
        df = df[["created_utc", "body"]]
        df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s")

        return df
    except Exception as e:
        print(f"Error fetching Reddit sentiment: {e}")
        return pd.DataFrame()
    