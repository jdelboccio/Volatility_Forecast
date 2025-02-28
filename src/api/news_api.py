import requests

NEWS_API_KEY = "1fb155db44f6415dbf0c5d3561f7fcef"

def get_news_sentiment(query):
    """Fetch financial news sentiment from NewsAPI."""
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        articles = response.json().get("articles", [])
        return [(article["title"], article["description"]) for article in articles]
    else:
        print("Error fetching news data:", response.json())
        return None
