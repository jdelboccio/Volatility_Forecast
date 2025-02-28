import praw
import pandas as pd

# Reddit API credentials (WARNING: Hardcoding credentials is not recommended)
REDDIT_CLIENT_ID = "YrhIHjfVx_gA_Gxxd3tuYg"
REDDIT_CLIENT_SECRET = "T0pf2vkPLfJtpb7NYsmIrfoPmr9FNQ"
REDDIT_USER_AGENT = "VolatilitySentimentAnalyzer"

# Configure Reddit API
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

def fetch_reddit_sentiment(ticker: str):
    """
    Fetch Reddit sentiment for a given stock ticker.

    :param ticker: Stock ticker symbol (e.g., 'AAPL')
    :return: DataFrame with sentiment analysis
    """
    try:
        subreddit = reddit.subreddit("stocks")
        posts = subreddit.search(ticker, limit=10)

        sentiments = []
        for post in posts:
            sentiments.append({"Title": post.title, "Score": post.score})

        return pd.DataFrame(sentiments)

    except Exception as e:
        print(f"Error fetching Reddit sentiment for {ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure

# Test the function
if __name__ == "__main__":
    print(fetch_reddit_sentiment("AAPL"))
