from arch import arch_model

def fit_garch_model(df):
    """
    Fits a GARCH(1,1) model and forecasts future volatility.
    """
    returns = df['returns'].dropna()  # Use daily returns
    garch_model = arch_model(returns, vol='Garch', p=1, q=1)
    garch_fitted = garch_model.fit(disp='off')

    # Predict future volatility
    garch_forecast = garch_fitted.forecast(start=len(returns), horizon=10)
    garch_vol_forecast = np.sqrt(garch_forecast.variance.values[-1])  # Convert variance to volatility

    print(f"GARCH(1,1) 10-day volatility forecast: {garch_vol_forecast:.4f}")
    return garch_vol_forecast
