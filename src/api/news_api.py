from yahoo_fin import news

def fetch_yahoo_news(ticker):
    """
    Fetch latest Yahoo Finance news for a stock ticker.
    """
    try:
        articles = news.get_yf_rss(ticker)  # Fetch Yahoo Finance RSS news
        return [{"title": article["title"], "link": article["link"]} for article in articles]
    except Exception as e:
        print(f"❌ Error fetching Yahoo Finance news: {e}")
        return None

# Example usage (for testing)
if __name__ == "__main__":
    ticker = "AAPL"
    news_data = fetch_yahoo_news(ticker)
    print(news_data)
