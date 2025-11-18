"""
Polygon Client - Updated with column normalization
UPDATED: Timestamp kept as column (not index), ticker added as column
"""
import time
import pandas as pd
from typing import Dict, Any
from base_api_client import BaseAPIClient
from polygon import RESTClient


# Column mapping for Polygon: short codes → standard names
POLYGON_COLUMN_MAP = {
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "t": "timestamp",
    "vw": "vwap",
    "n": "transactions"
}


class PolygonClient(BaseAPIClient):
    """API Client for Polygon with column normalization"""
    
    def __init__(self, api_key: str):
        """Initialize Polygon client"""
        self.client = RESTClient(api_key)
        self.endpoint_descriptions = {
            0: "Aggregated bars for a given ticker over a specified interval",
            1: "Grouped daily aggregates for the specified date",
            2: "Daily open/close aggregate for a given ticker and date",
            3: "Previous close aggregate for the given ticker"
        }
        # Standard columns after normalization
        self._standard_columns = ['open', 'high', 'low', 'close', 'volume']

    def fetch_data(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch data using the official client's methods"""
        endpoint_mapping = {
            0: lambda f: self.client.get_aggs(
                ticker=f['ticker'],
                multiplier=f['multiplier'],
                timespan=f['timespan'],
                from_=f['from'],
                to=f['to']
            ),
            1: lambda f: self.client.get_grouped_daily_aggs(
                date=f['from']
            ),
            2: lambda f: self.client.get_daily_open_close_agg(
                ticker=f['ticker'],
                date=f['from']
            ),
            3: lambda f: self.client.get_previous_close_agg(
                ticker=f['ticker']  
            )
        }
        endpoint_type = features.get('endpoint_type', 0)
        if endpoint_type not in endpoint_mapping:
            raise ValueError(f"Invalid endpoint_type: {endpoint_type}")

        attempts = 0
        max_attempts = 3
        delay = 2
        while attempts < max_attempts:
            try:
                response = endpoint_mapping[endpoint_type](features)
                return {'data': response, 'features': features}
            except Exception as e:
                attempts += 1
                if attempts == max_attempts:
                    print(f"Request failed after {max_attempts} attempts: {e}")
                    raise
                print(f"Attempt {attempts} failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)

    def parse_response(self, response_package: Dict[str, Any]) -> pd.DataFrame:
        """
        Parse raw response into DataFrame with normalized column names
        UPDATED: Timestamp kept as column (not index), ticker added as column
        """
        raw = response_package['data']
        params = response_package['features']
        
        # Normalize records
        if not isinstance(raw, (dict, list)):
            raw = raw.__dict__
        records = raw.get('results', [raw]) if isinstance(raw, dict) else raw

        df = pd.DataFrame(records)
        
        if df.empty:
            return df
        
        # *** NORMALIZE COLUMN NAMES ***
        df.rename(columns=POLYGON_COLUMN_MAP, inplace=True)
        
        # Handle timestamp - convert to UTC datetime and KEEP AS COLUMN (not index)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        
        # Add ticker column
        ticker = params.get('ticker') or params.get('symbol')
        if ticker:
            df.insert(0, 'ticker', ticker)
        
        return df