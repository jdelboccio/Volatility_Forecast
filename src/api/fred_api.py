import pandas as pd
import requests

def fetch_fred_data(indicators):
    """
    Fetches economic indicators from the Federal Reserve Economic Data (FRED) API.

    :param indicators: List of FRED indicators to fetch.
    :return: DataFrame containing the fetched data.
    """
    try:
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        api_key = "624bac6373fd1a4120556dd9a0beba3e"  

        data = []
        for indicator in indicators:
            response = requests.get(
                f"{base_url}?series_id={indicator}&api_key={api_key}&file_type=json"
            )
            json_data = response.json()

            for obs in json_data["observations"]:
                data.append({"indicator": indicator, "date": obs["date"], "value": obs["value"]})

        return pd.DataFrame(data)

    except Exception as e:
        print(f"Error fetching FRED data: {e}")
        return pd.DataFrame()