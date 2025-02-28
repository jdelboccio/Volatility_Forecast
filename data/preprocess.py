import pandas as pd

def compute_realized_volatility(df, window=10):
    """
    Compute future realized volatility using rolling 10-day standard deviation of returns.
    This shifts the target variable to ensure we predict future volatility.
    """
    df['returns'] = df['close'].pct_change()  # Calculate log returns
    df['realized_vol_t+10'] = df['returns'].rolling(window=window).std().shift(-window)  # Shift target forward

    return df.dropna()  # Drop NaN values caused by shifting
