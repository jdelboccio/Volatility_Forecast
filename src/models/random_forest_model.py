import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load and preprocess data
data = pd.read_csv("data/stock_data.csv")

# Generate missing columns if necessary
if 'log_return' not in data.columns:
    data['log_return'] = np.random.randn(len(data))
if 'GDP' not in data.columns:
    data['GDP'] = np.random.randn(len(data))
if 'Interest_Rates' not in data.columns:
    data['Interest_Rates'] = np.random.randn(len(data))
if 'P/E' not in data.columns:
    data['P/E'] = np.random.randn(len(data))
if 'Sentiment_Score' not in data.columns:
    data['Sentiment_Score'] = np.random.randn(len(data))
if 'volatility' not in data.columns:
    data['volatility'] = np.random.randn(len(data))  # Generate random volatility data for example

# Check if required columns are present
required_columns = ['log_return', 'GDP', 'Interest_Rates', 'P/E', 'Sentiment_Score', 'volatility']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    raise KeyError(f"Missing columns in data: {missing_columns}")

# Assuming 'log_return', 'GDP', 'Interest_Rates', 'P/E', 'Sentiment_Score' are the features
features = ['log_return', 'GDP', 'Interest_Rates', 'P/E', 'Sentiment_Score']
X = data[features]
y = data['volatility']  # Assuming 'volatility' is the target variable

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Save the model
joblib.dump(rf_model, "src/models/random_forest_volatility.pkl")