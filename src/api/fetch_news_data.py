import os
import sys
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Ensure Python can find 'src' when running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from news_api import fetch_news_sentiment  # Importing the function from news_api.py

analyzer = SentimentIntensityAnalyzer()

def get_news_sentiment(ticker):
    """Fetch latest news and compute sentiment scores."""
    news = fetch_news_sentiment(ticker)
    print(news)  # Debug print to see the structure of news
    
    results = []
    for article in news:
        print(article)  # Debug print to see each article
        if isinstance(article, dict):
            headline = article["headline"]
            date = article["date"]
            source = article["source"]
        else:
            headline = article
            date = "Unknown"
            source = "Unknown"
        
        sentiment_score = analyzer.polarity_scores(headline)['compound']
        volatility_change = round(-sentiment_score * 3, 2)  # Example: Strong negative news ↑ vol, positive ↓ vol
        results.append({
            "Date": date,
            "Source": source,
            "Headline": headline,
            "Sentiment Score": sentiment_score,
            "Expected Volatility Change": f"{volatility_change}%"
        })

    df_news = pd.DataFrame(results)
    df_news.to_csv("data/news_sentiment_analysis.csv", index=False)
    return df_news

# Run sentiment analysis
ticker = "AAPL"  # Example ticker symbol
get_news_sentiment(ticker)