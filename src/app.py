import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load Data Functions
from src.api.stock_api import fetch_stock_data
from src.api.fred_api import fetch_fred_data
from src.api.sec_api import fetch_sec_filings
from src.api.news_api import fetch_news_sentiment
from src.api.reddit_api import fetch_reddit_sentiment

# Load Model Functions
from src.models.final_volatility_forecast import compute_final_forecast

# Streamlit Page Configuration
st.set_page_config(page_title="Volatility Forecasting Dashboard", layout="wide")

# Dashboard Title
st.title("Volatility Forecasting Dashboard")

# Stock Ticker Input
ticker = st.text_input("Enter Stock Ticker:", "AAPL")

# Stock Price Data
st.subheader(f"Stock Data for {ticker}")
stock_data = fetch_stock_data(ticker)
st.dataframe(stock_data.head())

# Stock Price Chart
fig, ax = plt.subplots()
ax.plot(stock_data["date"], stock_data["close"], color="cyan")
ax.set_title(f"{ticker} Stock Closing Prices")
ax.set_xlabel("Date")
ax.set_ylabel("Close")
st.pyplot(fig)

# Economic Indicators
st.subheader("Economic Indicators")
fred_data = fetch_fred_data(["GDP", "Interest_Rates", "Inflation", "Unemployment"])
st.dataframe(fred_data)

# Recent SEC Filings
st.subheader("Recent SEC Filings")
sec_filings = fetch_sec_filings(ticker)
if sec_filings.empty:
    st.write("No SEC filings found.")
else:
    st.dataframe(sec_filings)

# News Sentiment Analysis
st.subheader("News Sentiment")
news_sentiment = fetch_news_sentiment(ticker)
st.dataframe(news_sentiment)

# Reddit Sentiment Analysis
st.subheader("Reddit Sentiment")
reddit_sentiment = fetch_reddit_sentiment(ticker)
st.dataframe(reddit_sentiment)

# Final Volatility Forecast
st.subheader("Final Volatility Forecast")
df_forecast = compute_final_forecast(ticker)
st.dataframe(df_forecast)

# Dashboard Footer
st.markdown("Built for Machine Learning-Based Equity Volatility Forecasting")
