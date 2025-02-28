import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from src.models.garch_model import garch_forecast
from src.data.preprocess import prepare_features

# Load models
lstm_model = tf.keras.models.load_model("models/lstm_volatility.h5")
rf_model = joblib.load("models/random_forest_volatility.pkl")

# Load data
data = pd.read_csv("data/stock_data.csv")
processed_data = prepare_features(data)

# Make Predictions
X_lstm = np.expand_dims(processed_data['log_return'].values[-30:], axis=0)
lstm_pred = lstm_model.predict(X_lstm)[0][0]

X_rf = processed_data[['log_return', 'GDP', 'Interest_Rates', 'P/E', 'Sentiment_Score']].iloc[-1]
rf_pred = rf_model.predict([X_rf.values])[0]

# Load Economic Factor Betas (Precomputed Sensitivity from Regression)
factor_betas = {
    "GDP": -0.32,  # Negative beta (higher GDP → lower volatility)
    "Interest_Rates": +0.45,  # Higher rates → higher vol
    "Unemployment": +0.28,  # Higher unemployment → more vol
}

# Get Latest Macro Data
latest_macro = processed_data.iloc[-1][['GDP', 'Interest_Rates', 'Unemployment']]
macro_adjustment = sum(latest_macro[key] * beta for key, beta in factor_betas.items())

# Load Valuation Data (P/E, CAPM Beta)
valuation_adjustment = 0
if processed_data.iloc[-1]["P/E"] > 25:  # Overvalued
    valuation_adjustment += 2.5
elif processed_data.iloc[-1]["P/E"] < 10:  # Undervalued
    valuation_adjustment -= 1.5

# Load Sentiment Scores
news_sentiment = processed_data.iloc[-1]["News_Sentiment"]
reddit_sentiment = processed_data.iloc[-1]["Reddit_Sentiment"]

# Compute Sentiment Adjustment
sentiment_adjustment = 0
if news_sentiment > 0.5:
    sentiment_adjustment -= 2.0  # Bullish → Reduce volatility
elif news_sentiment < -0.5:
    sentiment_adjustment += 2.0  # Bearish → Increase volatility

if reddit_sentiment > 0.5:
    sentiment_adjustment -= 1.0  # Strong retail sentiment → Reduce vol
elif reddit_sentiment < -0.5:
    sentiment_adjustment += 1.5  # Negative social sentiment → Increase vol

# Compute Final Volatility Forecast
base_forecast = (lstm_pred + rf_pred) / 2
final_vol_forecast = base_forecast + macro_adjustment + valuation_adjustment + sentiment_adjustment

# Save results
df_results = pd.DataFrame({
    "Model": ["LSTM + RF"],
    "Base Forecast": [base_forecast],
    "Macro Adj.": [macro_adjustment],
    "Valuation Adj.": [valuation_adjustment],
    "Sentiment Adj.": [sentiment_adjustment],
    "Final 10-Day Volatility Forecast": [final_vol_forecast]
})
df_results.to_csv("data/final_volatility_forecasts.csv", index=False)

print(df_results)
