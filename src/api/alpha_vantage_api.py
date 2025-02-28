import requests

ALPHA_VANTAGE_API_KEY = "EOQ4JR8FDI9F3I8B"

def get_stock_data(symbol):
    """Fetch stock price data from Alpha Vantage."""
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return data.get("Time Series (Daily)", {})
    else:
        print("Error fetching stock data:", response.json())
        return None
