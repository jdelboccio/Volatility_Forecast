import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess data
data = pd.read_csv("data/stock_data.csv")
processed_data = prepare_features(data)

# Feature selection
feature_cols = ['log_return']  # Add more macro/valuation/sentiment features if needed
target_col = 'future_vol'

# Scale data
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(processed_data[feature_cols])

# Prepare rolling sequences
def create_sequences(data, target, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(target[i+seq_length])  # Predict future 10-day vol
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_features, processed_data[target_col])

# Train/Test Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Save model
model.save("models/lstm_volatility.h5")
