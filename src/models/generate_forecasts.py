import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from src.data.preprocess import prepare_features
from src.models.garch_model import garch_forecast

# Load models
lstm_model = tf.keras.models.load_model("src/models/lstm_volatility.h5")
rf_model = joblib.load("src/models/random_forest_volatility.pkl")

# Load data
data = pd.read_csv("data/stock_data.csv")
processed_data = prepare_features(data)

# Make Predictions
X_lstm = np.expand_dims(processed_data['log_return'].values[-30:], axis=0)
lstm_pred = lstm_model.predict(X_lstm)[0][0]

X_rf = processed_data[['log_return', 'GDP', 'Interest_Rates', 'P/E', 'Sentiment_Score']].iloc[-1]
rf_pred = rf_model.predict([X_rf.values])[0]

# GARCH Prediction
garch_pred = garch_forecast(data)

# Save results
df_results = pd.DataFrame({
    "Model": ["LSTM", "Random Forest", "GARCH"],
    "10-Day Volatility Forecast": [lstm_pred, rf_pred, garch_pred]
})

df_results.to_csv("data/volatility_forecasts.csv", index=False)
print(df_results)
