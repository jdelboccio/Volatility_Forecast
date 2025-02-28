import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Ensure Python can find 'src' when running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fred_api import fetch_fred_data

def get_macro_factors():
    """Fetch macroeconomic indicators from FRED API."""
    gdp = fetch_fred_data("GDP").iloc[-1]['value']
    interest_rates = fetch_fred_data("DGS10").iloc[-1]['value']  # 10Y Treasury Yield
    inflation = fetch_fred_data("CPIAUCSL").iloc[-1]['value']  # Inflation (CPI)
    unemployment = fetch_fred_data("UNRATE").iloc[-1]['value']  # Unemployment Rate
    
    return pd.DataFrame([{
        "Indicator": "GDP Growth", "Latest Value": f"{gdp:.1f}%", "Impact on Volatility": None,
        "Indicator": "10Y Treasury Yield", "Latest Value": f"{interest_rates:.2f}%", "Impact on Volatility": None,
        "Indicator": "Inflation Rate", "Latest Value": f"{inflation:.1f}%", "Impact on Volatility": None,
        "Indicator": "Unemployment Rate", "Latest Value": f"{unemployment:.1f}%", "Impact on Volatility": None
    }])

def compute_factor_sensitivities():
    """Estimate how macro factors affect volatility using linear regression."""
    # Load volatility & macro data
    data = pd.read_csv("data/historical_volatility.csv")  # Historical realized vol
    macro = pd.read_csv("data/macro_data.csv")  # Macro indicators over time
    
    # Align time periods
    merged = data.merge(macro, on="date")
    
    # Define independent (X) and dependent (Y) variables
    X = merged[['GDP', 'Interest_Rates', 'Inflation', 'Unemployment']]
    Y = merged['volatility']

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, Y)
    
    # Return coefficients
    return pd.DataFrame({
        "Factor": ['GDP', 'Interest_Rates', 'Inflation', 'Unemployment'],
        "Sensitivity": model.coef_
    })