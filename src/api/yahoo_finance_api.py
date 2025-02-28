import yfinance as yf

def get_stock_data(ticker, start="2023-01-01", end="2024-01-01"):
    stock = yf.download(ticker, start=start, end=end)
    return stock

if __name__ == "__main__":
    print(get_stock_data("AAPL").head())
