import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

def create_sequences(data, sequence_length=10):
    """
    Create sequences for LSTM model.

    :param data: DataFrame with time-series data.
    :param sequence_length: Number of time steps to look back.
    :return: Sequences (X) and Targets (y).
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

def train_lstm_model(data: pd.DataFrame, target_column: str, model_path: str = "lstm_model.h5"):
    """
    Train an LSTM model for volatility forecasting.

    :param data: DataFrame containing time-series data.
    :param target_column: The column to predict (volatility).
    :param model_path: Path to save the trained model.
    """
    data = data[[target_column]].dropna()

    # Scale Data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create Sequences
    sequence_length = 10
    X, y = create_sequences(scaled_data, sequence_length)

    # Split Data
    train_size = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    # Define LSTM Model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        LSTM(50, return_sequences=False),
        Dense(25, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    
    # Train Model
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

    # Save Model & Scaler
    model.save(model_path)
    joblib.dump(scaler, "scaler.pkl")
    
    print(f"LSTM Model saved to {model_path}")

    return model

if __name__ == "__main__":
    df = pd.read_csv("../data/volatility_data.csv")
    train_lstm_model(df, target_column="volatility_10d")
