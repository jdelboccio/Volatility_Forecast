import praw

REDDIT_CLIENT_ID = "YrhlHjfVx_gA_Gxxd3tuYg"
REDDIT_SECRET = "T0pf2vkPLfJtpb7NYsmIrfoPmr9FNQ"
USER_AGENT = "volatility_sentiment_analyzer"

def get_reddit_sentiment(subreddit, query):
    """Fetch top Reddit posts mentioning a stock."""
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_SECRET,
        user_agent=USER_AGENT
    )
    
    posts = []
    for post in reddit.subreddit(subreddit).search(query, limit=5):
        posts.append((post.title, post.score, post.url))
    
    return posts
