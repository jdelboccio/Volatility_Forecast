import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.api.news_api import fetch_latest_news

analyzer = SentimentIntensityAnalyzer()

def get_news_sentiment():
    """Fetch latest news and compute sentiment scores."""
    news = fetch_latest_news()
    
    results = []
    for article in news:
        sentiment_score = analyzer.polarity_scores(article["headline"])['compound']
        volatility_change = round(-sentiment_score * 3, 2)  # Example: Strong negative news ↑ vol, positive ↓ vol
        results.append({
            "Date": article["date"],
            "Source": article["source"],
            "Headline": article["headline"],
            "Sentiment Score": sentiment_score,
            "Expected Volatility Change": f"{volatility_change}%"
        })

    df_news = pd.DataFrame(results)
    df_news.to_csv("data/news_sentiment_analysis.csv", index=False)
    return df_news

# Run sentiment analysis
get_news_sentiment()
