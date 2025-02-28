import pandas as pd
from src.api.reddit_api import fetch_reddit_comments

def compute_reddit_sentiment():
    """Fetch Reddit comments & compute sentiment index."""
    comments = fetch_reddit_comments()
    
    scores = [1 if "bullish" in c.lower() else -1 if "bearish" in c.lower() else 0 for c in comments]
    sentiment_score = sum(scores) / len(scores)

    volatility_impact = round(-sentiment_score * 2, 2)  # Example: Strong bearish = ↑ vol

    return sentiment_score, volatility_impact

# Compute and store
sentiment, impact = compute_reddit_sentiment()
print(f"Reddit Sentiment Score: {sentiment}, Volatility Impact: {impact}%")
