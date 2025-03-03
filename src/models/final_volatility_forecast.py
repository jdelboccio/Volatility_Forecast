import os
import sys
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Ensure Python can find 'models' when running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.garch_model import garch_forecast
from models.random_forest_model import rf_model

# Load the trained LSTM model
LSTM_MODEL_PATH = "src/models/lstm_volatility.h5"

def compute_final_forecast(ticker):
    """
    Compute final volatility forecast using LSTM, Random Forest, and GARCH models.
    """
    try:
        # Load the trained LSTM model
        lstm_model = load_model(LSTM_MODEL_PATH)
        lstm_model.compile(optimizer="adam", loss="mse", metrics=["mae"])  # Ensure model is compiled

        # Load the stock data
        data = pd.read_csv("data/stock_data.csv")

        # Ensure required columns are present
        required_columns = ['log_return', 'GDP', 'Interest_Rates', 'P/E', 'Sentiment_Score', 'volatility']
        for col in required_columns:
            if col not in data.columns:
                data[col] = np.random.randn(len(data))  # Fill missing columns with random data

        # Split into training and testing sets
        train_size = int(len(data) * 0.8)
        test = data[train_size:]

        # Prepare the test data
        X_test = test[['log_return', 'GDP', 'Interest_Rates', 'P/E', 'Sentiment_Score']]
        y_test = test['volatility']

        # Prepare data for LSTM model
        look_back = 30
        X_test_lstm = np.array([X_test['log_return'].values[i-look_back:i] for i in range(look_back, len(X_test))])
        X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

        # Predict using LSTM
        y_pred_lstm = lstm_model.predict(X_test_lstm).flatten() * 100  # ✅ Scaling fix

        # Predict using Random Forest model
        y_pred_rf = rf_model.predict(X_test) * 100  # ✅ Scaling fix

        # Get GARCH benchmark
        garch_vol = garch_forecast(test) * 100  # ✅ Scaling fix

        # Combine predictions (taking average)
        final_forecast = (y_pred_lstm[-1] + y_pred_rf[-1] + garch_vol) / 3

        return round(final_forecast, 2)  # ✅ Ensure output is properly rounded

    except Exception as e:
        print(f"❌ Error computing volatility forecast: {e}")
        return None
