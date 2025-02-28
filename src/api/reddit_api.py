import praw

REDDIT_CLIENT_ID = "Your-Reddit-Client-ID"
REDDIT_SECRET = "Your-Reddit-Secret"
REDDIT_USER_AGENT = "VolatilitySentimentAnalyzer"

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

def get_reddit_sentiment(subreddit, keyword):
    subreddit = reddit.subreddit(subreddit)
    posts = subreddit.search(keyword, limit=10)
    
    for post in posts:
        print(post.title, post.score)

if __name__ == "__main__":
    get_reddit_sentiment("stocks", "Tesla")
