import requests

SEC_API_KEY = "99be306db2183c302f5a228ae3a03f516e515c0b15957f0002455cf7673f9471"

def get_sec_filings(ticker):
    url = f"https://api.sec.gov/edgar/search/company/{ticker}?apikey={SEC_API_KEY}"
    response = requests.get(url)
    return response.json()

if __name__ == "__main__":
    print(get_sec_filings("AAPL"))  # Fetches Apple filings
