import numpy as np
from models.lstm_model import predict_volatility
from models.random_forest_model import compute_rf_volatility
from models.garch_model import compute_garch_volatility

def compute_final_forecast(ticker):
    """
    Compute final volatility forecast by averaging different models.
    """
    try:
        lstm_forecast = predict_volatility(ticker)
        rf_forecast = compute_rf_volatility(ticker)
        garch_forecast = compute_garch_volatility(ticker)

        # Ensure forecasts are valid before averaging
        forecasts = [lstm_forecast, rf_forecast, garch_forecast]
        valid_forecasts = [f for f in forecasts if f is not None]

        if not valid_forecasts:
            return None  # No valid forecasts

        final_forecast = np.mean(valid_forecasts)
        return final_forecast
    except Exception as e:
        print(f"Error computing final volatility forecast: {e}")
        return None
