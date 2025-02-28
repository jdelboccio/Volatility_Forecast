import requests
import pandas as pd

SEC_API_KEY = "99be306db2183c302f5a228ae3a03f516e515c0b15957f0002455cf7673f9471"

def fetch_sec_filings(ticker):
    url = f"https://api.sec.gov/edgar/search/company/{ticker}?apikey={SEC_API_KEY}"
    print(f"Trying to fetch SEC data from: {url}")  # Debugging line
    
    try:
        response = requests.get(url, timeout=10)  # Adding timeout for stability
        response.raise_for_status()
        data = response.json()

        filings = []
        if "filings" in data:
            for filing in data["filings"]:
                filings.append({
                    "date": filing.get("filingDate", "N/A"),
                    "type": filing.get("form", "N/A"),
                    "url": f"https://www.sec.gov{filing.get('href', '')}"
                })

        return pd.DataFrame(filings)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching SEC filings for {ticker}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    print(fetch_sec_filings("AAPL"))
