import pandas as pd
import numpy as np

def calculate_realized_volatility(returns, window=10):
    """Compute realized volatility as rolling standard deviation of log returns."""
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized volatility

def prepare_features(data):
    required_columns = ['log_return', 'GDP', 'Interest_Rates', 'P/E', 'Sentiment_Score']

    for col in required_columns:
        if col not in data.columns:
            print(f"Warning: {col} missing. Filling with default values.")
            data[col] = 0  # Fill missing columns with default values

    return data[required_columns]
