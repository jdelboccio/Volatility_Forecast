import numpy as np
import pandas as pd
from arch import arch_model

def garch_forecast(returns_series, days_ahead=1):
    """
    Fit a GARCH(1,1) model on the provided return series and forecast volatility for the given horizon.
    returns_series: pandas Series or array of returns.
    days_ahead: how many days ahead to forecast (default 1).
    Returns the forecasted volatility (standard deviation) for the next period (or list for multiple days).
    """
    # Convert input to a clean numpy array
    if returns_series is None:
        raise ValueError("No returns data provided for GARCH forecast.")
    if isinstance(returns_series, pd.Series):
        returns = returns_series.dropna().values.astype(float)
    else:
        returns = np.array(returns_series, dtype=float)
        returns = returns[np.isfinite(returns)]
    n = len(returns)
    if n < 2:
        raise ValueError("Not enough data points for GARCH model.")
    # Scale returns if very small to help convergence
    scale_factor = 1.0
    max_ret = np.max(np.abs(returns))
    if max_ret < 0.001:
        scale_factor = 10000.0  # extremely small returns, scale up a lot
    elif max_ret < 0.01:
        scale_factor = 100.0
    returns_scaled = returns * scale_factor
    # Fit GARCH(1,1) model (with zero mean)
    am = arch_model(returns_scaled, p=1, q=1, mean='Zero')  # zero mean since we model volatility
    res = am.fit(disp='off')
    # Forecast variance for the given horizon
    forecast = res.forecast(horizon=days_ahead)
    # Get variance forecast for the latest period
    var_forecast = forecast.variance.iloc[-1]  # last row (variance for forecasts)
    if days_ahead == 1:
        # If one-step ahead, extract single value
        if hasattr(var_forecast, 'values'):
            var_pred_scaled = var_forecast.values[0]
        else:
            var_pred_scaled = float(var_forecast)
        vol_pred_scaled = np.sqrt(var_pred_scaled)
        # Scale back to original returns scale
        vol_pred = vol_pred_scaled / scale_factor
        return float(vol_pred)
    else:
        # If multiple days ahead, return a list of vols
        if hasattr(var_forecast, 'values'):
            var_list = var_forecast.values[:days_ahead]
        else:
            var_list = np.array(var_forecast)[:days_ahead]
        vol_list_scaled = np.sqrt(var_list)
        vol_list = vol_list_scaled / scale_factor
        return vol_list.tolist()
