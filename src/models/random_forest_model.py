import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

rf_model = None  # will hold the trained RandomForest model

def train_model(data: pd.DataFrame):
    """Train a RandomForestRegressor on the given DataFrame (expects required feature columns and 'volatility' as target)."""
    global rf_model
    # Define feature names expected (must match data preparation)
    features = ["log_return", "GDP", "Interest_Rates", "PE", "Sentiment_Score"]
    # Verify all required features exist
    for col in features + ["volatility"]:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' is missing for Random Forest training.")
    # Drop any rows with NaN in features or target
    df = data[features + ["volatility"]].dropna()
    if df.empty:
        raise ValueError("No training data available for Random Forest after dropping NaNs.")
    X = df[features]
    y = df["volatility"]
    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    return rf_model

def predict_volatility(input_data):
    """
    Predict volatility using the trained Random Forest model for the given input features.
    input_data: Can be a pandas Series, pandas DataFrame (one row), or array/list of feature values.
    """
    global rf_model
    if rf_model is None:
        raise ValueError("Random Forest model is not trained. Call train_model first.")
    # Prepare input with correct shape and feature names
    if isinstance(input_data, pd.Series):
        # Series: convert to DataFrame (1 row)
        X = input_data.to_frame().T
    elif isinstance(input_data, pd.DataFrame):
        X = input_data.copy()
        if X.shape[0] != 1:
            # If multiple rows provided, use the last row for prediction
            X = X.tail(1)
    else:
        arr = np.array(input_data, dtype=float).reshape(1, -1)
        # If model has feature name info, use it; otherwise proceed with array
        if hasattr(rf_model, "feature_names_in_"):
            X = pd.DataFrame(arr, columns=rf_model.feature_names_in_)
        else:
            X = arr
    # Make prediction
    y_pred = rf_model.predict(X)
    # Return a scalar prediction (float)
    return float(y_pred[0])
