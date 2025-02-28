import requests

FRED_API_KEY = "624bac6373fd1a4120556dd9a0beba3e"

def get_fred_data(series_id):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
    response = requests.get(url)
    data = response.json()
    return data['observations']

if __name__ == "__main__":
    print(get_fred_data("GDP")[:5])  # Fetches US GDP data
