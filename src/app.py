import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import matplotlib.pyplot as plt

# Load Data Functions
from api.yahoo_finance_api import fetch_stock_data
from api.fred_api import fetch_fred_data
from api.sec_api import fetch_sec_filings
from api.news_api import fetch_news_sentiment
from api.reddit_api import fetch_reddit_sentiment

# Load Model Functions
from models.final_volatility_forecast import compute_final_forecast
from models.lstm_model import predict_volatility

# Streamlit Page Configuration
st.set_page_config(page_title="Volatility Forecasting Dashboard", layout="wide")

# Dashboard Title
st.title("📊 Volatility Forecasting Dashboard")

# Stock Ticker Input
ticker = st.text_input("🔍 Enter Stock Ticker:", "AAPL")

# Fetch Stock Data
st.subheader(f"📈 Stock Data for {ticker}")
try:
    stock_data = fetch_stock_data(ticker)
    if stock_data is None or stock_data.empty:
        st.warning(f"⚠️ No stock data found for {ticker}. Please check the ticker symbol.")
    else:
        st.line_chart(stock_data['Close'])
except Exception as e:
    st.error(f"❌ Error fetching stock data: {e}")

# Fetch Economic Data
st.subheader("📊 Economic Indicators")
try:
    gdp_data = fetch_fred_data("GDP")
    interest_rate_data = fetch_fred_data("DGS10")
    inflation_data = fetch_fred_data("CPIAUCSL")
    unemployment_data = fetch_fred_data("UNRATE")

    col1, col2 = st.columns(2)
    with col1:
        st.write("📉 GDP")
        st.line_chart(gdp_data['value'])
    with col2:
        st.write("📈 10-Year Treasury Yield")
        st.line_chart(interest_rate_data['value'])

    col1, col2 = st.columns(2)
    with col1:
        st.write("📊 Inflation (CPI)")
        st.line_chart(inflation_data['value'])
    with col2:
        st.write("📉 Unemployment Rate")
        st.line_chart(unemployment_data['value'])
except Exception as e:
    st.error(f"❌ Error fetching economic data: {e}")

# Fetch SEC Filings
st.subheader("📝 SEC Filings")
try:
    sec_filings = fetch_sec_filings(ticker)
    if sec_filings.empty:
        st.warning(f"⚠️ No SEC filings found for {ticker}.")
    else:
        st.write(sec_filings)
except Exception as e:
    st.error(f"❌ Error fetching SEC filings: {e}")

# Fetch News Sentiment
st.subheader("📰 News Sentiment")
try:
    news_sentiment = fetch_news_sentiment(ticker)
    if news_sentiment is None or len(news_sentiment) == 0:
        st.warning(f"⚠️ No news sentiment data found for {ticker}.")
    else:
        st.write(news_sentiment)
except Exception as e:
    st.error(f"❌ Error fetching news sentiment: {e}")

# Fetch Reddit Sentiment
st.subheader("📢 Reddit Sentiment")
try:
    reddit_sentiment = fetch_reddit_sentiment(ticker)
    if reddit_sentiment is None or len(reddit_sentiment) == 0:
        st.warning(f"⚠️ No Reddit sentiment data found for {ticker}.")
    else:
        st.write(reddit_sentiment)
except Exception as e:
    st.error(f"❌ Error fetching Reddit sentiment: {e}")

# Compute Final Volatility Forecast
st.subheader("📉 Volatility Forecast")
try:
    forecast = compute_final_forecast(ticker)
    if forecast is not None:
        st.write(f"📊 10-Day Volatility Forecast for {ticker}: **{forecast:.2f}%**")
    else:
        st.error("❌ Volatility forecast computation failed.")
except Exception as e:
    st.error(f"❌ Error computing volatility forecast: {e}")

# 3D Factor Weighting Visualization
st.subheader("🟢 3D Factor Weighting Visualization")
try:
    # Sample weighting values for visualization
    factors = ['Fundamentals', 'Valuation', 'Sentiment']
    weights = np.random.rand(3)  # Replace this with actual factor weights from model
    fig = px.scatter_3d(
        x=[weights[0]], y=[weights[1]], z=[weights[2]],
        text=[f"{factors[i]}: {weights[i]:.2f}" for i in range(3)],
        labels={'x': 'Fundamentals', 'y': 'Valuation', 'z': 'Sentiment'},
        title="Factor Weighting in Model",
        size=[10],
        opacity=0.8
    )
    st.plotly_chart(fig)
except Exception as e:
    st.error(f"❌ Error generating 3D visualization: {e}")

# AI Explainability: SHAP Factor Importance
st.subheader("🧠 AI Explainability: SHAP Factor Importance")
try:
    # Placeholder SHAP values (Replace with actual computation)
    explainer = shap.Explainer(lambda x: x)  # Dummy function for now
    sample_data = np.random.rand(10, 3)  # Replace with actual model input
    shap_values = explainer(sample_data)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, sample_data, feature_names=['GDP', 'Interest Rates', 'Sentiment'])
    st.pyplot(fig)
except Exception as e:
    st.error(f"❌ Error computing SHAP values: {e}")

st.success("🚀 Dashboard Loaded Successfully")
