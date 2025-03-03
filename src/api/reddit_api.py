import requests
import os
import pandas as pd

# Reddit API Credentials (Set these in your environment variables)
REDDIT_CLIENT_ID = "YrhIHjfVx_gA_Gxxd3tuYg"
REDDIT_CLIENT_SECRET = "T0pf2vkPLfJtpb7NYsmIrfoPmr9FNQ"
REDDIT_USER_AGENT = "VolatilitySentimentAnalyzer"

def get_reddit_token():
    """
    Fetch OAuth2 access token for Reddit API.
    """
    url = "https://www.reddit.com/api/v1/access_token"
    auth = (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
    headers = {"User-Agent": REDDIT_USER_AGENT}
    data = {"grant_type": "client_credentials"}

    response = requests.post(url, auth=auth, data=data, headers=headers)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        print(f"⚠️ Error fetching Reddit token: {response.json()}")
        return None

def fetch_reddit_sentiment(ticker):
    """
    Fetch Reddit mentions & sentiment for a given stock ticker.
    """
    token = get_reddit_token()
    if not token:
        return pd.DataFrame({"Error": ["Failed to authenticate with Reddit API"]})

    url = f"https://oauth.reddit.com/search?q={ticker}&sort=new&limit=10"
    headers = {
        "Authorization": f"bearer {token}",
        "User-Agent": REDDIT_USER_AGENT
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        posts = response.json()["data"]["children"]
        data = [{"title": post["data"]["title"], "score": post["data"]["score"], "url": post["data"]["url"]} for post in posts]
        return pd.DataFrame(data)
    else:
        print(f"⚠️ Error fetching Reddit sentiment: {response.json()}")
        return pd.DataFrame({"Error": ["Failed to fetch Reddit data"]})
