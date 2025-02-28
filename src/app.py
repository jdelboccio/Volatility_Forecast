import streamlit as st
import pandas as pd
import yfinance as yf
import praw
import requests
import datetime
import plotly.express as px
import os

# API KEYS (Ensure these are stored securely, ideally in environment variables)
FRED_API_KEY = "624bac6373fd1a4120556dd9a0beba3e"
SEC_EDGAR_API_KEY = "99be306db2183c302f5a228ae3a03f516e515c0b15957f0002455cf7673f9471"
ALPHA_VANTAGE_API_KEY = "EOQ4JR8FDI9F3I8B"
NEWS_API_KEY = "1fb155db44f6415dbf0c5d3561f7fcef"
REDDIT_CLIENT_ID = "YrhlHjfVx_gA_Gxxd3tuYg"
REDDIT_CLIENT_SECRET = "T0pf2vkPLfJtpb7NYsmIrfoPmr9FNQ"
REDDIT_USER_AGENT = "volatility_sentiment"

# Function to fetch stock data
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")
    return hist

# Function to fetch economic indicators from FRED API
def fetch_fred_data(series_id):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame(data['observations'])

    print("DEBUG: Raw Data from FRED API:\n", df.head())

    # Convert to float safely, setting errors='coerce' to replace invalid values with NaN
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    # Drop rows where 'value' is NaN
    df.dropna(subset=['value'], inplace=True)

    print("DEBUG: Converted Data:\n", df.head())

    return df


# Function to fetch financial statements from SEC EDGAR API
def fetch_sec_filings(ticker):
    url = f"https://data.sec.gov/submissions/CIK{ticker}.json"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return None

# Function to fetch news data
def fetch_news_data(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    return response.json()

# Function to fetch sentiment analysis from Reddit
def fetch_reddit_sentiment(ticker):
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    
    subreddit = reddit.subreddit("stocks")
    mentions = []
    
    for post in subreddit.search(ticker, limit=10):
        mentions.append({"title": post.title, "score": post.score, "created": datetime.datetime.fromtimestamp(post.created)})

    return pd.DataFrame(mentions)

# Streamlit UI
st.title("Volatility Forecasting Dashboard")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker:", "AAPL")

if ticker:
    st.subheader(f"Stock Data for {ticker}")
    
    # Fetch stock data
    stock_data = fetch_stock_data(ticker)
    st.write(stock_data.tail())

    # Plot stock closing prices
    fig = px.line(stock_data, x=stock_data.index, y="Close", title=f"{ticker} Stock Closing Prices")
    st.plotly_chart(fig)

    # Fetch economic indicators
    st.subheader("Economic Indicators")
    fred_data = fetch_fred_data("GDP")
    st.write(fred_data.tail())

    # Fetch SEC filings
    st.subheader("Recent SEC Filings")
    sec_filings = fetch_sec_filings(ticker)
    if sec_filings:
        st.write(sec_filings)
    else:
        st.write("No SEC filings found.")

    # Fetch news sentiment
    st.subheader("News Sentiment")
    news_data = fetch_news_data(ticker)
    if news_data.get("articles"):
        for article in news_data["articles"][:5]:
            st.write(f"**{article['title']}**")
            st.write(f"_{article['source']['name']}_ - {article['publishedAt']}")
            st.write(f"[Read More]({article['url']})")
    else:
        st.write("No news articles found.")

    # Fetch Reddit sentiment
    st.subheader("Reddit Sentiment")
    reddit_data = fetch_reddit_sentiment(ticker)
    if not reddit_data.empty:
        st.write(reddit_data)
    else:
        st.write("No mentions found.")

st.sidebar.header("About")
st.sidebar.info("This dashboard integrates stock, economic, financial, and sentiment data to provide a holistic volatility forecast.")
