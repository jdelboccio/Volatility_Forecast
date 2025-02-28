import requests
import pandas as pd

FRED_API_KEY = "624bac6373fd1a4120556dd9a0beba3e"  # Ensure correct key

def fetch_fred_data(series_id):
    try:
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
        response = requests.get(url).json()

        if 'observations' not in response:
            raise ValueError(f"Invalid FRED response: {response}")

        df = pd.DataFrame(response['observations'])
        if 'value' not in df.columns:
            raise ValueError("Missing 'value' field in FRED data")

        df = df[['date', 'value']].rename(columns={'value': series_id})  # Rename column to series ID
        return df
    except Exception as e:
        print(f"Error fetching FRED data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure
