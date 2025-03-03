import requests
import pandas as pd


USER_AGENT = "jdelboccio (juandelboccio@gmail.com)"

def get_cik_from_ticker(ticker):
    """
    Fetch CIK (Central Index Key) from stock ticker.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": USER_AGENT}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        for item in data.values():
            if item["ticker"].upper() == ticker.upper():
                return str(item["cik_str"]).zfill(10)  # Ensure 10-digit CIK
    print(f"⚠️ CIK not found for ticker {ticker}.")
    return None

def fetch_sec_filings(ticker):
    """
    Fetch recent SEC filings for a given stock ticker.
    """
    cik = get_cik_from_ticker(ticker)
    if not cik:
        return pd.DataFrame({"Error": [f"CIK not found for {ticker}"]})

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {"User-Agent": USER_AGENT}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if "filings" in data:
            df = pd.DataFrame(data["filings"]["recent"])
            return df[["accessionNumber", "filingDate", "form", "primaryDocument"]]
    
    print(f"⚠️ Error fetching SEC filings: {response.json()}")
    return pd.DataFrame({"Error": ["Failed to fetch SEC filings"]})
