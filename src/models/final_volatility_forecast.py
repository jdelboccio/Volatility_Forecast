import os
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Ensure Python can find the 'models' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.lstm_model import predict_volatility
from models.garch_model import garch_forecast
from models.random_forest_model import rf_model

def compute_final_forecast(ticker):
    """
    Compute 10-day volatility forecast for a given stock ticker.
    """
    try:
        # Load stock data
        data = pd.read_csv("data/stock_data.csv")

        # Normalize column names (case-insensitive)
        data.columns = data.columns.str.lower()

        # Ensure required columns exist
        required_columns = ['close', 'log_return', 'gdp', 'interest_rates', 'p/e', 'sentiment_score', 'volatility']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}. Available columns: {list(data.columns)}")

        # Split dataset
        train_size = int(len(data) * 0.8)
        train, test = data[:train_size], data[train_size:]

        # Prepare test data
        X_test = test[['log_return', 'gdp', 'interest_rates', 'p/e', 'sentiment_score']]
        y_test = test['volatility']

        # Predict with LSTM model
        y_pred_lstm = predict_volatility(ticker)

        # Predict with Random Forest model
        y_pred_rf = rf_model.predict(X_test)

        # Get GARCH benchmark
        garch_vol = garch_forecast(test)

        # Combine predictions (Weighted average)
        final_forecast = (y_pred_lstm[-1] + y_pred_rf[-1] + garch_vol) / 3

        return round(final_forecast * 100, 2)  # Return as percentage
    except Exception as e:
        print(f"Error computing final forecast: {e}")
        return None
