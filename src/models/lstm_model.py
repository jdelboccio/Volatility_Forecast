import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

MODEL_PATH = "src/models/lstm_volatility.h5"

def create_dataset(dataset, look_back=30):
    """Transforms time-series data into LSTM training format."""
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Load and preprocess stock data
data_path = "data/stock_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Data file not found: {data_path}")

data = pd.read_csv(data_path)

if 'close' not in data.columns:
    raise ValueError("❌ Missing 'close' column in stock_data.csv")

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))

# Prepare training data
look_back = 30
X, Y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build LSTM model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
    LSTM(50),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model (only if not already trained)
if not os.path.exists(MODEL_PATH):
    print("⚡ Training LSTM model...")
    lstm_model.fit(X, Y, epochs=20, batch_size=32, verbose=1)
    lstm_model.save(MODEL_PATH)
    print("✅ Model trained and saved!")

# Load model if already trained
else:
    print("🔄 Loading pre-trained model...")
    lstm_model = load_model(MODEL_PATH)