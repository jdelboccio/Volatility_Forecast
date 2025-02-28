import pandas as pd
import numpy as np

def calculate_realized_volatility(returns, window=10):
    """Compute realized volatility as rolling standard deviation of log returns."""
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized volatility

def prepare_features(data):
    """
    Preprocess the data and ensure required features are present.
    
    :param data: DataFrame containing the raw data
    :return: DataFrame with processed features
    """
    # Ensure required columns are present
    required_columns = ['log_return', 'GDP', 'Interest_Rates', 'P/E', 'Sentiment_Score']
    for col in required_columns:
        if col not in data.columns:
            raise KeyError(f"Missing required column: {col}")
    
    # Example preprocessing steps (add your actual preprocessing logic here)
    processed_data = data.copy()
    
    # Add any additional preprocessing steps here
    # For example, scaling, encoding, etc.
    
    return processed_data
