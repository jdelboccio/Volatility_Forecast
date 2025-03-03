import requests
import pandas as pd

FRED_API_KEY = "your_api_key_here"  # Replace with your actual API key
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

def fetch_fred_data(series_id):
    """
    Fetch economic data from the FRED API.
    """
    try:
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json"
        }
        response = requests.get(FRED_BASE_URL, params=params)
        data = response.json()
        
        if "observations" not in data:
            raise ValueError(f"Unexpected response format from FRED API: {data}")

        df = pd.DataFrame(data["observations"])
        
        if "value" not in df.columns:
            raise ValueError(f"Missing 'value' column in FRED data. Available columns: {list(df.columns)}")
        
        df["value"] = pd.to_numeric(df["value"], errors="coerce")  # Convert to numeric
        df.dropna(inplace=True)

        return df

    except Exception as e:
        print(f"Error fetching FRED data: {e}")
        return pd.DataFrame()  # Return empty DataFrame if error occurs
