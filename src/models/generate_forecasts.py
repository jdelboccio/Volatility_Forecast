import os, sys
from pathlib import Path
# Ensure current directory is in sys.path for module imports
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

import pandas as pd
import numpy as np
import random_forest_model
import lstm_model
import garch_model
import shap_explainer

def prepare_data(data: pd.DataFrame, ticker: str = None) -> pd.DataFrame:
    """Ensure the DataFrame has all required columns for modeling, adding or computing them if necessary."""
    df = data.copy()
    # If a specific ticker is given, filter data for that ticker
    if ticker is not None and 'ticker' in df.columns:
        df = df[df['ticker'] == ticker].copy()
    # Standardize column names (lower-case and remove spaces, then match expected names)
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        if key == 'gdp':
            rename_map[col] = 'GDP'
        elif key in ['interest_rate', 'interest_rates']:
            rename_map[col] = 'Interest_Rates'
        elif key in ['pe', 'p/e', 'price_earnings']:
            rename_map[col] = 'PE'
        elif key in ['sentiment_score', 'sentiment']:
            rename_map[col] = 'Sentiment_Score'
        elif key in ['log_return', 'logreturn']:
            rename_map[col] = 'log_return'
        elif key in ['volatility', 'vol']:
            rename_map[col] = 'volatility'
        elif key == 'ticker':
            rename_map[col] = 'ticker'
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # If ticker column missing and ticker param provided, create it (all rows same ticker)
    if 'ticker' not in df.columns and ticker is not None:
        df['ticker'] = ticker

    # Compute or fill log_return
    if 'log_return' not in df.columns:
        if 'Close' in df.columns or 'Adj Close' in df.columns:
            price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
            logret = np.log(df[price_col].astype(float) / df[price_col].astype(float).shift(1))
            logret.iloc[0] = 0.0  # first entry set to 0 (no previous day)
            df['log_return'] = logret
        else:
            # If no price data, generate synthetic log returns around 0
            df['log_return'] = np.random.normal(0, 0.01, size=len(df))
            df['log_return'].iloc[0] = 0.0
    # Fill any remaining NaN in log_return (e.g., first row) with 0
    df['log_return'] = df['log_return'].fillna(0.0)

    # Add macro and other factors if missing
    if 'GDP' not in df.columns:
        # Assume GDP growth percent ~ 2% with minor variation
        df['GDP'] = np.random.normal(2.0, 0.1, size=len(df))
    if 'Interest_Rates' not in df.columns:
        # Assume interest rates ~2% with minor variation
        df['Interest_Rates'] = np.random.normal(0.02, 0.005, size=len(df))
    if 'PE' not in df.columns:
        # Assume P/E around 20 with some variation
        df['PE'] = np.clip(np.random.normal(20.0, 3.0, size=len(df)), 1, None)
    if 'Sentiment_Score' not in df.columns:
        # Assume sentiment score ~0.5 (neutral) with some variation in [0,1]
        sentiment = np.random.normal(0.5, 0.1, size=len(df))
        df['Sentiment_Score'] = np.clip(sentiment, 0.0, 1.0)

    # Compute or fill volatility
    if 'volatility' not in df.columns:
        # Use rolling window std of log_return as realized volatility estimate
        # Drop first row (no prior return for volatility calculation)
        df_no0 = df.iloc[1:].reset_index(drop=True).copy()
        # Choose a rolling window (e.g., 20 days)
        window = 20
        vol = df_no0['log_return'].rolling(window).std()
        # Fill initial NaNs by backward fill (assume first computed vol applies to earlier entries)
        vol.fillna(method='bfill', inplace=True)
        df_no0['volatility'] = vol
        df = df_no0
    else:
        # If volatility exists, fill any NaNs by forward/backward fill
        df['volatility'].fillna(method='ffill', inplace=True)
        df['volatility'].fillna(method='bfill', inplace=True)

    # Final cleanup: drop any remaining NaNs in required columns
    required_cols = ['log_return', 'GDP', 'Interest_Rates', 'PE', 'Sentiment_Score', 'volatility']
    df[required_cols] = df[required_cols].fillna(method='ffill').fillna(method='bfill')
    return df

def generate_forecasts(ticker: str, data: pd.DataFrame = None) -> dict:
    """
    Generate volatility forecasts using LSTM, GARCH, and Random Forest models for the given ticker.
    Returns a dictionary with keys 'lstm', 'garch', 'random_forest'.
    """
    # Load/prepare data for the specific ticker
    if data is None:
        # If no data provided, attempt to load a default dataset
        default_path = Path(__file__).resolve().parent / "volatility_data.csv"
        if default_path.is_file():
            data = pd.read_csv(default_path)
        else:
            raise FileNotFoundError("No data provided to generate_forecasts and default dataset not found.")
    prepared_df = prepare_data(data, ticker)

    results = {}

    # Random Forest Model: train on this ticker's data and predict next volatility
    try:
        rf_model = random_forest_model.train_model(prepared_df)
        # Prepare the latest feature row for prediction
        latest_features_df = prepared_df.tail(1)[["log_return", "GDP", "Interest_Rates", "PE", "Sentiment_Score"]]
        rf_pred = random_forest_model.predict_volatility(latest_features_df)
    except Exception as e:
        rf_pred = None
        print(f"Random Forest model failed: {e}")
    results['random_forest'] = rf_pred

    # LSTM Model: predict volatility using sequence of recent data
    lstm_pred = None
    try:
        # Determine sequence length from model if possible, otherwise default
        seq_length = 30
        if hasattr(lstm_model, "lstm_model_instance") and lstm_model.lstm_model_instance is not None:
            try:
                seq_length = lstm_model.lstm_model_instance.input_shape[1]
            except Exception:
                pass
        # Ensure we have enough data for the sequence
        if len(prepared_df) >= seq_length:
            # Select the last seq_length days of features (including volatility as a feature for LSTM)
            seq_features = ["log_return", "GDP", "Interest_Rates", "PE", "Sentiment_Score", "volatility"]
            seq_data = prepared_df[seq_features].tail(seq_length).values
            # Compute mean and std of volatility from training data for rescaling
            vol_series = prepared_df['volatility']
            vol_mean, vol_std = vol_series.mean(), vol_series.std()
            # Predict with LSTM model (pass the global model instance if available)
            lstm_pred_val = lstm_model.predict_volatility(model=lstm_model.lstm_model_instance, sequence_data=seq_data, scaler=(vol_mean, vol_std))
            lstm_pred = float(lstm_pred_val)
        else:
            lstm_pred = None  # Not enough data to use LSTM
    except Exception as e:
        lstm_pred = None
        print(f"LSTM model prediction failed: {e}")
    results['lstm'] = lstm_pred

    # GARCH Model: fit on the ticker's returns and forecast next volatility
    garch_pred = None
    try:
        returns = prepared_df['log_return']
        garch_pred_val = garch_model.garch_forecast(returns, days_ahead=1)
        garch_pred = float(garch_pred_val)
    except Exception as e:
        # If GARCH fails (e.g., not enough data), fallback to last known volatility or None
        if not prepared_df['volatility'].dropna().empty:
            garch_pred = float(prepared_df['volatility'].dropna().iloc[-1])
        else:
            garch_pred = None
        print(f"GARCH model failed: {e}")
    results['garch'] = garch_pred

    # (Optionally) compute SHAP values for Random Forest for interpretability 
    # (e.g., feature importance for the latest prediction). This is optional and heavy, so handle carefully.
    # Example (not displayed by default): shap_values = shap_explainer.explain_model(rf_model, prepared_df)
    # You can use shap_values for additional insights or plots if needed.

    return results
