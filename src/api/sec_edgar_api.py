import requests

SEC_API_KEY = "99be306db2183c302f5a228ae3a03f516e515c0b15957f0002455cf7673f9471"

def get_sec_filings(cik):
    """Fetch company financial reports from SEC EDGAR API."""
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/Assets.json"
    headers = {"User-Agent": "YourAppName/1.0 (your@email.com)"}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error fetching data from SEC EDGAR:", response.json())
        return None
