"""Download 10-K/10-Q filings from SEC EDGAR."""

import requests
import time
import json
import re
from pathlib import Path
import click
import yaml

def load_config():
    with open(Path(__file__).parent.parent.parent / "configs" / "pipeline.yaml") as f:
        return yaml.safe_load(f)

EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"

def get_headers():
    cfg = load_config()["edgar"]
    return {"User-Agent": cfg["user_agent"], "Accept-Encoding": "gzip, deflate"}

def get_cik(ticker: str) -> str:
    url = "https://www.sec.gov/cgi-bin/browse-edgar"
    params = {"action": "getcompany", "company": ticker, "type": "10-K",
              "dateb": "", "owner": "include", "count": "1", "search_text": "",
              "output": "atom"}
    headers = get_headers()
    # Use the company tickers JSON for reliable CIK lookup
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    tickers = resp.json()
    for entry in tickers.values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker {ticker} not found in SEC EDGAR")

def get_recent_filings(cik: str, filing_type: str = "10-K", count: int = 1):
    headers = get_headers()
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    filings = data["filings"]["recent"]
    results = []
    for i, form in enumerate(filings["form"]):
        if form == filing_type and len(results) < count:
            accession = filings["accessionNumber"][i].replace("-", "")
            primary_doc = filings["primaryDocument"][i]
            filing_date = filings["filingDate"][i]
            results.append({
                "accession": accession,
                "primary_doc": primary_doc,
                "filing_date": filing_date,
                "url": f"{EDGAR_ARCHIVES}/{cik.lstrip('0')}/{accession}/{primary_doc}"
            })
    return results

def download_filing(ticker: str, filing_type: str = "10-K", output_dir: str = "data/raw"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Looking up CIK for {ticker}...")
    cik = get_cik(ticker)
    print(f"  CIK: {cik}")

    print(f"Fetching recent {filing_type} filings...")
    time.sleep(0.2)  # SEC rate limit
    filings = get_recent_filings(cik, filing_type)

    if not filings:
        print(f"  No {filing_type} filings found for {ticker}")
        return None

    filing = filings[0]
    print(f"  Found: {filing['filing_date']} — {filing['url']}")

    headers = get_headers()
    time.sleep(0.2)
    resp = requests.get(filing["url"], headers=headers)
    resp.raise_for_status()

    ext = Path(filing["primary_doc"]).suffix or ".htm"
    filename = f"{ticker}_{filing_type.replace('-', '')}_{filing['filing_date']}{ext}"
    filepath = output_path / filename

    filepath.write_bytes(resp.content)
    print(f"  Saved: {filepath}")

    # Save metadata
    meta_path = output_path / f"{filename}.meta.json"
    meta_path.write_text(json.dumps({
        "ticker": ticker, "filing_type": filing_type,
        "filing_date": filing["filing_date"], "cik": cik,
        "source_url": filing["url"], "local_file": str(filepath)
    }, indent=2))

    return str(filepath)

@click.command()
@click.option("--ticker", required=True, help="Stock ticker (e.g., AAPL)")
@click.option("--filing-type", default="10-K", help="Filing type (10-K or 10-Q)")
def main(ticker, filing_type):
    download_filing(ticker, filing_type)

if __name__ == "__main__":
    main()
