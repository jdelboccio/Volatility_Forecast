import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load and preprocess data
data = pd.read_csv("data/stock_data.csv")
processed_data = prepare_features(data)

# Feature selection (Include Fundamentals, Valuation, Sentiment)
feature_cols = ['log_return', 'GDP', 'Interest_Rates', 'P/E', 'Sentiment_Score']
target_col = 'future_vol'

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    processed_data[feature_cols], processed_data[target_col], test_size=0.2, random_state=42
)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Model
preds = rf_model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, preds))

# Save Model
import joblib
joblib.dump(rf_model, "models/random_forest_volatility.pkl")
