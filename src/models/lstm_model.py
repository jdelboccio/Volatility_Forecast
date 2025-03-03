import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load stock data
data = pd.read_csv("data/stock_data.csv")

# Normalize column names (convert all to lowercase)
data.columns = data.columns.str.lower()

# Check for the "close" column (handles different variations)
if "close" not in data.columns:
    raise ValueError(f"Missing 'Close' column in stock data. Available columns: {list(data.columns)}")

# Data Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data["close"].values.reshape(-1, 1))

# Define function to create dataset
def create_dataset(dataset, look_back=30):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Prepare dataset
look_back = 30
X, Y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build LSTM model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
    LSTM(50),
    Dense(1)
])

# Compile model
lstm_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train model
lstm_model.fit(X, Y, epochs=20, batch_size=32, verbose=1)

# Save model
lstm_model.save("src/models/lstm_volatility.h5")

# Function to predict volatility
def predict_volatility(ticker):
    """
    Predict 10-day volatility using trained LSTM model.
    """
    try:
        model = load_model("src/models/lstm_volatility.h5")

        # Prepare last 30 days of data for prediction
        last_30_days = scaled_data[-look_back:]
        X_input = np.reshape(last_30_days, (1, look_back, 1))

        # Make prediction
        y_pred = model.predict(X_input)
        
        # Rescale prediction
        y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        return y_pred_rescaled
    except Exception as e:
        print(f"Error predicting volatility: {e}")
        return None
