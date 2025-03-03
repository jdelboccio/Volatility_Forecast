import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

# Constants
LOOK_BACK = 30
LSTM_MODEL_PATH = "src/models/lstm_volatility.h5"
DATA_FILE = "data/stock_data.csv"

def create_dataset(dataset, look_back=1):
    """
    Convert time series data into supervised learning format.
    """
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def train_lstm_model():
    """
    Train an LSTM model for volatility prediction.
    """
    try:
        # Load and preprocess data
        data = pd.read_csv(DATA_FILE)
        if "close" not in data.columns:
            raise ValueError("❌ Missing 'close' column in stock_data.csv!")

        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))

        # Prepare training data
        X, Y = create_dataset(scaled_data, LOOK_BACK)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(LOOK_BACK, 1)))
        model.add(LSTM(50))
        model.add(Dense(1))

        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=["mae"])

        # Train the model
        model.fit(X, Y, epochs=20, batch_size=32, verbose=1)

        # Save the trained model
        model.save(LSTM_MODEL_PATH)
        print("✅ LSTM Model saved successfully!")

    except Exception as e:
        print(f"❌ Error training LSTM model: {e}")

def predict_volatility(ticker):
    """
    Load trained LSTM model and make predictions.
    """
    try:
        # Load trained model
        model = load_model(LSTM_MODEL_PATH)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        # Load and preprocess data
        data = pd.read_csv(DATA_FILE)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))

        # Prepare input data for prediction
        X_input = np.array([scaled_data[-LOOK_BACK:]])  # Last 30 days
        X_input = np.reshape(X_input, (1, LOOK_BACK, 1))

        # Make prediction
        prediction = model.predict(X_input).flatten() * 100  # ✅ Scaling fix
        return prediction

    except Exception as e:
        print(f"❌ Error making LSTM prediction: {e}")
        return None
