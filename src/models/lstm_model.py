import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

# Load and preprocess stock data
data_path = "data/stock_data.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Data file not found at: {data_path}")

data = pd.read_csv(data_path)

# Normalize column names to lowercase
data.columns = data.columns.str.lower()

# Ensure "close" column exists
if "close" not in data.columns:
    raise ValueError(f"Missing 'Close' column in stock data. Available columns: {list(data.columns)}")

# Handle missing values by filling with the last valid value
data["close"].fillna(method="ffill", inplace=True)

# Scale the "close" column
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data["close"].values.reshape(-1, 1))

# Create dataset function
def create_dataset(dataset, look_back=30):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Prepare dataset
look_back = 30
X, Y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Check if model is already trained
model_path = "src/models/lstm_volatility.keras"
if not os.path.exists(model_path):
    print("🔄 Training new LSTM model...")

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
    lstm_model.save(model_path)
else:
    print("✅ LSTM model found. Skipping training.")

# Function to predict volatility
def predict_volatility():
    """
    Predicts the next 10-day volatility using the trained LSTM model.
    """
    try:
        # Load the trained model
        model = load_model(model_path)

        # Prepare the last 30 days of data for prediction
        last_30_days = scaled_data[-look_back:]
        X_input = np.reshape(last_30_days, (1, look_back, 1))

        # Make prediction
        y_pred = model.predict(X_input)

        # Rescale prediction back to original scale
        y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        return round(y_pred_rescaled[0], 2)
    
    except Exception as e:
        print(f"❌ Error predicting volatility: {e}")
        return None
