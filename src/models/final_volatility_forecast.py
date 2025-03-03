def compute_final_forecast(ticker):
    """
    Computes the final volatility forecast for a given ticker.
    """
    # Load the LSTM model and make prediction
    try:
        volatility = predict_volatility(ticker)
        return volatility
    except Exception as e:
        print(f"Error computing volatility forecast: {e}")
        return None
