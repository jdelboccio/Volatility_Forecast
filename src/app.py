import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
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
from models.lstm_model import lstm_model  # Needed for SHAP values

# Streamlit Page Configuration
st.set_page_config(page_title="Volatility Forecasting Dashboard", layout="wide")

# Dashboard Title
st.title("Volatility Forecasting Dashboard")

# Stock Ticker Input
ticker = st.text_input("Enter Stock Ticker:", "AAPL")

# Time Slider for Historical Data Visualization
st.sidebar.subheader("Historical Timeframe")
time_range = st.sidebar.slider("Select Time Period (Days)", min_value=30, max_value=365, value=90, step=30)

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

# Fetch Real-Time Factor Weightings
st.subheader("Real-Time Factor Weightings")
try:
    gdp = fetch_fred_data("GDP")['value'].iloc[-1]
    interest_rate = fetch_fred_data("DGS10")['value'].iloc[-1]
    inflation = fetch_fred_data("CPIAUCSL")['value'].iloc[-1]
    sentiment = fetch_news_sentiment(ticker)['score'].iloc[-1]

    # Normalize values (scaled between 0-1)
    factors = {
        "Fundamentals": gdp / 100000,
        "Valuations": interest_rate / 10,
        "Sentiment": sentiment / 10
    }

    st.json(factors)

except Exception as e:
    st.error(f"Error fetching factor weightings: {e}")

# 3D Factor Weighting Visualization (Live Data)
st.subheader("3D Factor Weighting Visualization")
try:
    data = pd.DataFrame({
        "Fundamentals": [factors["Fundamentals"]],
        "Valuations": [factors["Valuations"]],
        "Sentiment": [factors["Sentiment"]],
        "Stock": [ticker],
        "Confidence": [np.random.uniform(0.7, 0.95)]  # Placeholder confidence
    })

    fig = px.scatter_3d(
        data,
        x="Fundamentals",
        y="Valuations",
        z="Sentiment",
        color="Confidence",
        text="Stock",
        opacity=0.8,
        title="3D Visualization of Model Inputs (Fundamentals, Valuations, Sentiment)",
        labels={"Fundamentals": "Fundamentals Score", "Valuations": "Valuation Score", "Sentiment": "Sentiment Score"},
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="Fundamentals",
            yaxis_title="Valuations",
            zaxis_title="Sentiment",
        )
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error generating 3D visualization: {e}")

# AI Explainability (SHAP Analysis for Factor Importance)
st.subheader("AI Explainability: SHAP Factor Importance")
try:
    # Prepare SHAP values for LSTM model
    explainer = shap.Explainer(lstm_model)
    sample_input = np.array([[factors["Fundamentals"], factors["Valuations"], factors["Sentiment"]]])
    shap_values = explainer(sample_input)

    # Plot SHAP values
    fig, ax = plt.subplots()
    shap.bar_plot(shap_values, show=False)
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error computing SHAP values: {e}")

# Compute Final Volatility Forecast
st.subheader("Volatility Forecast")
try:
    forecast = compute_final_forecast(ticker)
    st.write(f"10-Day Volatility Forecast for {ticker}: {forecast:.2f}%")
except Exception as e:
    st.error(f"Error computing volatility forecast: {e}")
