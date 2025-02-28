import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

NEWS_API_KEY = "1fb155db44f6415dbf0c5d3561f7fcef"

def fetch_news_sentiment(ticker):
    """
    Fetch recent news articles and calculate sentiment scores.
    
    :param ticker: Stock ticker symbol (e.g., "AAPL")
    :return: DataFrame with headlines and sentiment scores
    """
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        news_data = response.json()

        articles = news_data.get("articles", [])
        analyzer = SentimentIntensityAnalyzer()
        
        processed_articles = []
        for article in articles:
            headline = article["title"]
            sentiment_score = analyzer.polarity_scores(headline)["compound"]
            
            processed_articles.append({
                "date": article["publishedAt"],
                "source": article["source"]["name"],
                "headline": headline,
                "sentiment_score": sentiment_score
            })

        return pd.DataFrame(processed_articles)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching news for {ticker}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    print(fetch_news_sentiment("AAPL"))