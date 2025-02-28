import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load Data Functions
from api.yahoo_finance_api import fetch_stock_data
from api.fred_api import fetch_fred_data
from api.sec_api import fetch_sec_filings
from api.news_api import fetch_news_sentiment
from api.reddit_api import fetch_reddit_sentiment

# Load Model Functions
from models.final_volatility_forecast import compute_final_forecast

# Streamlit Page Configuration
st.set_page_config(page_title="Volatility Forecasting Dashboard", layout="wide")

# Dashboard Title
st.title("Volatility Forecasting Dashboard")

# Stock Ticker Input
ticker = st.text_input("Enter Stock Ticker:", "AAPL")

# Fetch Stock Price Data
st.subheader(f"Stock Data for {ticker}")
try:
    stock_data = fetch_stock_data(ticker)
    if stock_data.empty:
        st.warning(f"No stock data found for {ticker}. Please check the ticker symbol.")
    else:
        st.line_chart(stock_data['Close'])
except Exception as e:
    st.error(f"Error fetching stock data: {e}")

# Fetch FRED Data
st.subheader("FRED Economic Data")
try:
    gdp_data = fetch_fred_data("GDP")
    interest_rate_data = fetch_fred_data("DGS10")
    inflation_data = fetch_fred_data("CPIAUCSL")
    unemployment_data = fetch_fred_data("UNRATE")
    
    st.write("GDP Data")
    st.line_chart(gdp_data['value'])
    
    st.write("10-Year Treasury Yield")
    st.line_chart(interest_rate_data['value'])
    
    st.write("Inflation (CPI)")
    st.line_chart(inflation_data['value'])
    
    st.write("Unemployment Rate")
    st.line_chart(unemployment_data['value'])
except Exception as e:
    st.error(f"Error fetching FRED data: {e}")

# Fetch SEC Filings
st.subheader("SEC Filings")
try:
    sec_filings = fetch_sec_filings(ticker)
    if sec_filings.empty:
        st.warning(f"No SEC filings found for {ticker}.")
    else:
        st.write(sec_filings)
except Exception as e:
    st.error(f"Error fetching SEC filings: {e}")

# Fetch News Sentiment
st.subheader("News Sentiment")
try:
    news_sentiment = fetch_news_sentiment(ticker)
    if news_sentiment.empty:
        st.warning(f"No news sentiment data found for {ticker}.")
    else:
        st.write(news_sentiment)
except Exception as e:
    st.error(f"Error fetching news sentiment: {e}")

# Fetch Reddit Sentiment
st.subheader("Reddit Sentiment")
try:
    reddit_sentiment = fetch_reddit_sentiment(ticker)
    if reddit_sentiment.empty:
        st.warning(f"No Reddit sentiment data found for {ticker}.")
    else:
        st.write(reddit_sentiment)
except Exception as e:
    st.error(f"Error fetching Reddit sentiment: {e}")

# Compute Final Volatility Forecast
st.subheader("Volatility Forecast")
try:
    forecast = compute_final_forecast(ticker)
    st.write(f"10-Day Volatility Forecast for {ticker}: {forecast:.2f}%")
except Exception as e:
    st.error(f"Error computing volatility forecast: {e}")