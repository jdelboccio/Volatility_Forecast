import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from pathlib import Path
from models.generate_forecasts import generate_forecasts

# Set up Streamlit page
st.set_page_config(page_title="Volatility Forecasting Dashboard", layout="wide")
st.title("Volatility Forecasting Dashboard")

# File paths for data
DATA_PATH = Path("volatility_data.csv")

# Load dataset with debugging
if DATA_PATH.is_file():
    st.info(f"✅ Found dataset: `{DATA_PATH}`")
    try:
        data = pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"❌ Error reading `{DATA_PATH}`: {e}")
        st.stop()
else:
    st.warning("⚠️ `volatility_data.csv` not found. Fetching new data...")

    # Generate synthetic dataset if missing
    tickers = ["AAPL", "GOOGL", "AMZN", "MSFT", "TSLA"]
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")

    data = pd.DataFrame({
        "date": np.random.choice(dates, 100),
        "ticker": np.random.choice(tickers, 100),
        "log_return": np.random.normal(0, 0.02, 100),
        "GDP": np.random.uniform(2.0, 4.0, 100),
        "Interest_Rates": np.random.uniform(1.0, 5.0, 100),
        "PE": np.random.uniform(10, 40, 100),
        "Sentiment_Score": np.random.uniform(-1, 1, 100),
        "volatility": np.random.uniform(0.1, 0.5, 100)
    })

    data.to_csv(DATA_PATH, index=False)
    st.success("✅ Synthetic dataset created: `volatility_data.csv`")

# Ensure required columns exist
required_columns = ["ticker", "log_return", "GDP", "Interest_Rates", "PE", "Sentiment_Score", "volatility"]
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    st.warning(f"⚠️ Missing columns detected: {missing_columns}. Generating synthetic values.")
    for col in missing_columns:
        if col == "ticker":
            data[col] = np.random.choice(["AAPL", "GOOGL", "AMZN", "MSFT", "TSLA"], len(data))
        else:
            data[col] = np.random.randn(len(data))

# Get unique tickers for selection
tickers = sorted(data['ticker'].dropna().unique())
selected_ticker = st.selectbox("Select a stock ticker:", tickers)

st.markdown("Once a ticker is selected, click **Generate Forecast** to run the models.")

# Initialize session state
if 'last_ticker' not in st.session_state:
    st.session_state['last_ticker'] = selected_ticker
if 'forecast' not in st.session_state:
    st.session_state['forecast'] = False
    st.session_state['results'] = None

# Reset forecast state if ticker changes
if selected_ticker != st.session_state['last_ticker']:
    st.session_state['forecast'] = False
    st.session_state['results'] = None
    st.session_state['last_ticker'] = selected_ticker

# Forecast button
if st.button("Generate Forecast"):
    results = generate_forecasts(selected_ticker, data)
    st.session_state['forecast'] = True
    st.session_state['results'] = results

# Display results if available
if st.session_state['forecast'] and st.session_state['results'] is not None:
    res = st.session_state['results']
    vol_lstm = res.get('lstm')
    vol_garch = res.get('garch')
    vol_rf = res.get('random_forest')
    vol_ens = None

    available_preds = [v for v in [vol_lstm, vol_garch, vol_rf] if v is not None]
    if available_preds:
        vol_ens = sum(available_preds) / len(available_preds)

    # Display forecast results
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LSTM Forecast", f"{vol_lstm:.2%}" if vol_lstm else "N/A")
    col2.metric("GARCH Forecast", f"{vol_garch:.2%}" if vol_garch else "N/A")
    col3.metric("RF Forecast", f"{vol_rf:.2%}" if vol_rf else "N/A")
    col4.metric("Ensemble Forecast", f"{vol_ens:.2%}" if vol_ens else "N/A")

    # Compute weights for models
    w_lstm = w_garch = w_rf = 0.0
    count = len(available_preds)
    if count == 3:
        w_lstm = w_garch = w_rf = 1/3
    elif count == 2:
        if vol_lstm is None:
            w_lstm = 0.0; w_garch = w_rf = 0.5
        elif vol_garch is None:
            w_garch = 0.0; w_lstm = w_rf = 0.5
        elif vol_rf is None:
            w_rf = 0.0; w_lstm = w_garch = 0.5
    elif count == 1:
        if vol_lstm is not None: w_lstm = 1.0
        if vol_garch is not None: w_garch = 1.0
        if vol_rf is not None: w_rf = 1.0

    # Create 3D visualization
    fig = go.Figure(data=[go.Scatter3d(
        x=[w_lstm * 100], y=[w_garch * 100], z=[w_rf * 100],
        mode='markers',
        marker=dict(size=8, color='red')
    )])
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="LSTM weight (%)", range=[0, 100]),
            yaxis=dict(title="GARCH weight (%)", range=[0, 100]),
            zaxis=dict(title="RF weight (%)", range=[0, 100])
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    # Display formula and chart
    colA, colB = st.columns([1, 2])
    with colA:
        st.latex(r"\hat{\sigma}_{forecast} = w_{LSTM}\hat{\sigma}_{LSTM} + w_{GARCH}\hat{\sigma}_{GARCH} + w_{RF}\hat{\sigma}_{RF}")
        if count == 3:
            st.markdown("*Using equal weights: $w_{LSTM}=w_{GARCH}=w_{RF}=\\frac{1}{3}$. *")
        elif count == 2:
            if vol_lstm is None:
                st.markdown("*LSTM not used: $w_{LSTM}=0$, $w_{GARCH}=w_{RF}=0.5$. *")
            elif vol_garch is None:
                st.markdown("*GARCH not used: $w_{GARCH}=0$, $w_{LSTM}=w_{RF}=0.5$. *")
            elif vol_rf is None:
                st.markdown("*RF not used: $w_{RF}=0$, $w_{LSTM}=w_{GARCH}=0.5$. *")
        elif count == 1:
            if vol_lstm is not None:
                st.markdown("*Only LSTM used: $w_{LSTM}=1$. *")
            elif vol_garch is not None:
                st.markdown("*Only GARCH used: $w_{GARCH}=1$. *")
            elif vol_rf is not None:
                st.markdown("*Only RF used: $w_{RF}=1$. *")
    with colB:
        st.plotly_chart(fig, use_container_width=True)
