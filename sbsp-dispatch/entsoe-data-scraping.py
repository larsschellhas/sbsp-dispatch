#!/usr/bin/env python3
"""
Download historical day-ahead energy prices for all domains (bidding zones)
using entsoe-py’s Pandas client.
"""

import os
from pathlib import Path
from typing import Dict

import pandas as pd
import requests
from dotenv import load_dotenv
from entsoe import EntsoeRawClient, EntsoePandasClient
from entsoe.mappings import Area
from ratelimit import limits, sleep_and_retry

# ENTSO-E allows 400 requests per minute
CALLS = 300
RATE_LIMIT_PERIOD = 60  # seconds

# Load environment variables from .env in script’s directory
dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

class RateLimitedEntsoeRawClient(EntsoeRawClient):
    @sleep_and_retry
    @limits(calls=CALLS, period=RATE_LIMIT_PERIOD)
    def _base_request(self, params: Dict, start: pd.Timestamp, end: pd.Timestamp) -> requests.Response:
        return super()._base_request(params, start, end)

class RateLimitedEntsoePandasClient(EntsoePandasClient, RateLimitedEntsoeRawClient):
    pass


def fetch_all_day_ahead_prices(start: pd.Timestamp, end: pd.Timestamp, api_key: str, areas: list = Area) -> pd.DataFrame:
    client = EntsoePandasClient(api_key=api_key, retry_count=5, retry_delay=5)
    results = []
    total_areas = len(areas)
    
    for i, area in enumerate(areas, 1):
        code = area.code
        name = area.name
        print(f"\rFetching {name} ({i}/{total_areas})...", end="", flush=True)
        try:
            ts: pd.Series = client.query_day_ahead_prices(
                country_code=code,
                start=start,
                end=end,
                resolution='60min'
            )
            df = ts.rename("price").to_frame().reset_index().rename(columns={"index": "datetime"})
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            df["area_name"] = name
            df.to_csv(f"data/day_ahead_prices_{name}.csv")
            results.append(df)
            print(f"\rFetching {name} ({i}/{total_areas})... [OK] ({len(df)} entries)")
        except Exception as e:
            print(f"\rFetching {name} ({i}/{total_areas})... [ERROR] -> {type(e).__name__}: {e}")
    
    if not results:
        return pd.DataFrame()
    
    combined_df = pd.concat(results, ignore_index=True)
    pivoted_df = combined_df.pivot(index='datetime', columns='area_name', values='price')
    pivoted_df = pivoted_df.fillna(-10000)
    
    return pivoted_df

if __name__ == "__main__":
    # Read environment variable for API token
    API_TOKEN = os.getenv("ENTSOE_API_TOKEN")
    if not API_TOKEN:
        raise RuntimeError("Please set the ENTSOE_API_TOKEN environment variable.")
    
    # Define your period (example: January 2025 UTC)
    START = pd.Timestamp("20150101T0000", tz="UTC")
    END   = pd.Timestamp("20241231T2359", tz="UTC")
    
    print(f"Fetching day-ahead prices from {START} to {END} for all domains...")
    df_prices = fetch_all_day_ahead_prices(START, END, API_TOKEN)
    
    # Save to CSV for further analysis
    out_file = f"data/day_ahead_prices_all_domains_{START.strftime('%Y%m%d')}-{END.strftime('%Y%m%d')}.csv"
    df_prices.to_csv(out_file)
    print(f"Saved {len(df_prices)} records to {out_file}")
