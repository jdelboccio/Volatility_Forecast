import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
from random_forest_model import rf_model  # Assuming rf_model is the trained Random Forest model
from garch_model import garch_forecast

# Load the trained LSTM model
lstm_model = load_model("src/models/lstm_volatility.h5")

# Load the test data
data = pd.read_csv("data/stock_data.csv")

# Ensure required columns are present
required_columns = ['log_return', 'GDP', 'Interest_Rates', 'P/E', 'Sentiment_Score', 'volatility']
for col in required_columns:
    if col not in data.columns:
        data[col] = np.random.randn(len(data))

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Prepare the test data
X_test = test[['log_return', 'GDP', 'Interest_Rates', 'P/E', 'Sentiment_Score']]
y_test = test['volatility']

# Predict using trained models
look_back = 30
X_test_lstm = np.array([X_test['log_return'].values[i-look_back:i] for i in range(look_back, len(X_test))])
X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], 1))
y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()

y_pred_rf = rf_model.predict(X_test)

# Get GARCH benchmark
garch_vol = garch_forecast(test)

# Compare MAE
mae_lstm = mean_absolute_error(y_test[look_back:], y_pred_lstm)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# Display results
print(f"Model Performance:")
print(f"- LSTM MAE: {mae_lstm:.4f}")
print(f"- Random Forest MAE: {mae_rf:.4f}")
print(f"- GARCH(1,1) Forecast: {garch_vol:.4f}")