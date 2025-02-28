import os
import sys
import pandas as pd

# Ensure Python can find 'src' when running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from reddit_api import fetch_reddit_sentiment

def compute_reddit_sentiment(ticker):
    """Fetch Reddit comments & compute sentiment index."""
    try:
        comments = fetch_reddit_sentiment(ticker)
    except Exception as e:
        print(f"Error fetching Reddit sentiment for {ticker}: {e}")
        return None, None
    
    if comments.empty:
        print(f"No comments fetched for {ticker}.")
        return None, None
    
    scores = [1 if "bullish" in c.lower() else -1 if "bearish" in c.lower() else 0 for c in comments['comment']]
    
    if len(scores) == 0:
        print(f"No valid comments for sentiment analysis for {ticker}.")
        return None, None
    
    sentiment_score = sum(scores) / len(scores)
    volatility_impact = round(-sentiment_score * 2, 2)  # Example: Strong bearish = ↑ vol

    return sentiment_score, volatility_impact

# Compute and store
ticker = "AAPL"  # Example ticker symbol
sentiment, impact = compute_reddit_sentiment(ticker)
if sentiment is not None and impact is not None:
    print(f"Reddit Sentiment Score: {sentiment}, Volatility Impact: {impact}%")
else:
    print("Sentiment analysis could not be performed.")