import requests

FRED_API_KEY = "624bac6373fd1a4120556dd9a0beba3e"

def get_fred_data(series_id):
    """Fetch macroeconomic data from the FRED API."""
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return data['observations']  # List of observations
    else:
        print("Error fetching data from FRED:", response.json())
        return None
