from arch import arch_model
import pandas as pd

# Load Data
data = pd.read_csv("data/stock_data.csv")
data['log_return'] = np.log(data['Close'] / data['Close'].shift(1)).dropna()

# Fit GARCH(1,1)
garch = arch_model(data['log_return'], vol='Garch', p=1, q=1)
garch_fit = garch.fit()

# Predict next 10-day vol
garch_forecast = garch_fit.forecast(horizon=10)
print(garch_forecast.variance[-1:])
