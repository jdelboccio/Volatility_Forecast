import os
import sys
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf

# Ensure Python can find 'src' when running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from data.preprocess import prepare_features
from garch_model import garch_forecast

# Load models
lstm_model_path = os.path.abspath("src/models/lstm_volatility.h5")
rf_model_path = os.path.abspath("src/models/random_forest_volatility.pkl")

if not os.path.exists(lstm_model_path):
    raise FileNotFoundError(f"Model file not found: {lstm_model_path}")
if not os.path.exists(rf_model_path):
    raise FileNotFoundError(f"Model file not found: {rf_model_path}")

lstm_model = tf.keras.models.load_model(lstm_model_path)
rf_model = joblib.load(rf_model_path)

# Load or generate stock data
DATA_PATH = os.path.abspath("data/stock_data.csv")
data = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else pd.DataFrame({
    'Close': np.random.randn(100),  # Generate random close prices if data is missing
    'log_return': np.random.randn(100),
    'GDP': np.random.randn(100),
    'Interest_Rates': np.random.randn(100),
    'P/E': np.random.randn(100),
    'Sentiment_Score': np.random.randn(100)
})

# Ensure 'Close' column is present and contains valid positive values
if 'Close' not in data.columns or (data['Close'] <= 0).any():
    data['Close'] = np.abs(np.random.randn(len(data))) + 1  # Generate positive close prices

# Ensure 'log_return' column is correctly generated
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1)).fillna(0)

# Check for NaN values in 'log_return' column
if data['log_return'].isna().any():
    raise ValueError("NaN values found in 'log_return' column after generation.")

# Ensure required columns are present
required_columns = ['log_return', 'GDP', 'Interest_Rates', 'P/E', 'Sentiment_Score']
for col in required_columns:
    if col not in data.columns:
        data[col] = np.random.randn(len(data))

processed_data = prepare_features(data)

# Ensure required columns are present in processed_data
missing_columns = [col for col in required_columns if col not in processed_data.columns]

if missing_columns:
    raise KeyError(f"Missing columns in processed data: {missing_columns}")

# Make Predictions
X_lstm = np.expand_dims(processed_data['log_return'].values[-30:], axis=0)

if X_lstm.shape[1] < 30:
    raise ValueError(f"Insufficient data for LSTM prediction. Needed: 30, Found: {X_lstm.shape[1]}")

lstm_pred = lstm_model.predict(X_lstm)[0][0]


X_rf = processed_data[['log_return', 'GDP', 'Interest_Rates', 'P/E', 'Sentiment_Score']].iloc[-1]
rf_pred = rf_model.predict([X_rf.values])[0]

# GARCH Prediction
garch_pred = garch_forecast(processed_data)

# Save results
df_results = pd.DataFrame({
    "Model": ["LSTM", "Random Forest", "GARCH"],
    "10-Day Volatility Forecast": [lstm_pred, rf_pred, garch_pred]
})

print(df_results)