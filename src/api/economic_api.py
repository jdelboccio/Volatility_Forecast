import requests
import pandas as pd

WORLD_BANK_URL = "https://api.worldbank.org/v2/country/US/indicator/"

def fetch_economic_data(indicator):
    """
    Fetches economic data from World Bank API.

    Args:
        indicator (str): Economic indicator code (e.g., "NY.GDP.MKTP.CD" for GDP).

    Returns:
        DataFrame or None: Economic data if available, else None.
    """
    url = f"{WORLD_BANK_URL}{indicator}?format=json"
    try:
        response = requests.get(url)
        data = response.json()

        if "error" in data[0]:
            return None

        df = pd.DataFrame(data[1])
        df = df[["date", "value"]].dropna()
        df = df.sort_values("date", ascending=True)
        return df
    except Exception as e:
        print(f"Error fetching economic data: {e}")
        return None
