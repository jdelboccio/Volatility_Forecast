import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load Data Functions (Fixed imports)
from src.api.yahoo_finance_api import fetch_stock_data  # FIXED: Use Yahoo API for stock data
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

try:
    stock_data = fetch_stock_data(ticker)
    if stock_data is not None and not stock_data.empty:
        st.dataframe(stock_data.head())

        # Stock Price Chart
        fig, ax = plt.subplots()
        ax.plot(stock_data.index, stock_data["Close"], color="cyan")  # Fixed column name
        ax.set_title(f"{ticker} Stock Closing Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        st.pyplot(fig)
    else:
        st.warning(f"No stock data found for {ticker}.")
except Exception as e:
    st.error(f"Error fetching stock data: {e}")

# Economic Indicators
st.subheader("Economic Indicators")
try:
    fred_data = fetch_fred_data(["GDP", "Interest_Rates", "Inflation", "Unemployment"])
    if fred_data is not None and not fred_data.empty:
        st.dataframe(fred_data)
    else:
        st.warning("No economic indicator data available.")
except Exception as e:
    st.error(f"Error fetching economic indicators: {e}")

# Recent SEC Filings
st.subheader("Recent SEC Filings")
try:
    sec_filings = fetch_sec_filings(ticker)
    if sec_filings is not None and not sec_filings.empty:
        st.dataframe(sec_filings)
    else:
        st.write("No SEC filings found.")
except Exception as e:
    st.error(f"Error fetching SEC filings: {e}")

# News Sentiment Analysis
st.subheader("News Sentiment")
try:
    news_sentiment = fetch_news_sentiment(ticker)
    if news_sentiment is not None and not news_sentiment.empty:
        st.dataframe(news_sentiment)
    else:
        st.warning("No news sentiment data available.")
except Exception as e:
    st.error(f"Error fetching news sentiment: {e}")

# Reddit Sentiment Analysis
st.subheader("Reddit Sentiment")
try:
    reddit_sentiment = fetch_reddit_sentiment(ticker)
    if reddit_sentiment is not None and not reddit_sentiment.empty:
        st.dataframe(reddit_sentiment)
    else:
        st.warning("No Reddit sentiment data available.")
except Exception as e:
    st.error(f"Error fetching Reddit sentiment: {e}")

# Final Volatility Forecast
st.subheader("Final Volatility Forecast")
try:
    df_forecast = compute_final_forecast(ticker)
    if df_forecast is not None and not df_forecast.empty:
        st.dataframe(df_forecast)
    else:
        st.warning("No forecast data available.")
except Exception as e:
    st.error(f"Error computing final forecast: {e}")

# Dashboard Footer
st.markdown("Built for Machine Learning-Based Equity Volatility Forecasting")
