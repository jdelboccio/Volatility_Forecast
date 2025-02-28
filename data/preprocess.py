import pandas as pd
import numpy as np

def calculate_realized_volatility(returns, window=10):
    """Compute realized volatility as rolling standard deviation of log returns."""
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized volatility

def prepare_features(data, target_window=10):
    """
    Create lagged features for LSTM & Random Forest.
    - Features: Past 30 days (rolling window)
    - Target: Future 10-day realized volatility
    """
    data = data.copy()
    
    # Calculate log returns
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Compute realized volatility
    data['realized_vol'] = calculate_realized_volatility(data['log_return'], window=target_window)

    # Shift the target volatility forward (future prediction)
    data['future_vol'] = data['realized_vol'].shift(-target_window)

    # Drop NaNs
    data = data.dropna()

    return data
