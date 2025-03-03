from yahoo_fin import news
from textblob import TextBlob

# Yahoo Finance News API
def fetch_news_data(ticker):
    """
    Fetch latest Yahoo Finance news for a stock ticker.
    """
    try:
        articles = news.get_yf_rss(ticker)  # Fetch Yahoo Finance RSS news
        if not articles:
            return []

        return [{"source": "Yahoo Finance", "title": article.get("title", "No Title"), "link": article.get("link", "#")} for article in articles]

    except Exception as e:
        print(f"❌ Error fetching Yahoo Finance news: {e}")
        return []

# **New Function: Fetch News Sentiment**
def fetch_news_sentiment(ticker):
    """
    Fetch Yahoo Finance news and analyze sentiment for each article.
    """
    news_articles = fetch_news_data(ticker)  # Fetch news from Yahoo Finance

    if not news_articles:
        return [{"source": "None", "title": "No news found", "sentiment": "Neutral", "polarity": 0.0}]

    analyzed_news = []
    for article in news_articles:
        text = article["title"]
        sentiment = TextBlob(text).sentiment  # Analyze sentiment
        sentiment_label = "Positive" if sentiment.polarity > 0 else "Negative" if sentiment.polarity < 0 else "Neutral"

        analyzed_news.append({
            "source": article["source"],
            "title": article["title"],
            "link": article["link"],
            "sentiment": sentiment_label,
            "polarity": round(sentiment.polarity, 2)
        })

    return analyzed_news

# Example Usage (For Testing)
if __name__ == "__main__":
    ticker = "AAPL"
    sentiment_data = fetch_news_sentiment(ticker)
    print(sentiment_data)
