import requests

SEC_API_KEY = "99be306db2183c302f5a228ae3a03f516e515c0b15957f0002455cf7673f9471"

def fetch_sec_filings(ticker):
    try:
        url = f"https://api.sec.gov/edgar/search/company/{ticker}?apikey={SEC_API_KEY}"
        response = requests.get(url).json()

        if 'filings' not in response:
            raise ValueError(f"Invalid SEC response: {response}")

        filings = response.get('filings', [])
        if not filings:
            return "No SEC filings found."

        return filings
    except Exception as e:
        print(f"Error fetching SEC filings for {ticker}: {e}")
        return "Error fetching SEC data."
