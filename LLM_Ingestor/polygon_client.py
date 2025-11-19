"""
Polygon Client - Fixed FULL_MARKET_SNAPSHOT parsing
"""
import time
from typing import Dict, Any

import pandas as pd
import requests
from polygon import RESTClient

from base_api_client import BaseAPIClient


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
    """API Client for Polygon"""
    
    def __init__(self, api_key: str):
        self.client = RESTClient(api_key)
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.fed_base_url = "https://api.massive.com/fed/v1"
        self._standard_columns = ['open', 'high', 'low', 'close', 'volume']

    def fetch_data(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch data using the Polygon client"""
        endpoint_type = features.get('endpoint_type', 0)
        
        if isinstance(endpoint_type, str) and endpoint_type == 'economic_indicator':
            indicator = features.get('indicator')
            return self._fetch_economic_indicator(indicator, features)
        
        endpoint_mapping = {
            0: lambda f: self.client.get_aggs(
                ticker=f['ticker'],
                multiplier=f.get('multiplier', 1),
                timespan=f.get('timespan', 'day'),
                from_=f['from'],
                to=f['to']
            ),
            1: lambda f: self.client.get_grouped_daily_aggs(date=f['from']),
            2: lambda f: self.client.get_daily_open_close_agg(ticker=f['ticker'], date=f['from']),
            3: lambda f: self.client.get_previous_close_agg(ticker=f['ticker']),
            4: lambda f: self.client.get_snapshot_all("stocks"),
        }
        
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
        """Parse Polygon response into DataFrame"""
        raw = response_package['data']
        params = response_package['features']
        endpoint_type = params.get('endpoint_type', 0)
        
        if isinstance(endpoint_type, str) and endpoint_type == 'economic_indicator':
            return self._parse_economic_indicator(raw, params)
        
        # Handle FULL_MARKET_SNAPSHOT
        if endpoint_type == 4:
            return self._parse_snapshot_response(raw, params)
        
        # Convert response object to dict if needed
        if hasattr(raw, '__dict__'):
            raw = raw.__dict__
        
        # Get results
        if isinstance(raw, dict):
            records = raw.get('results', [raw])
        elif isinstance(raw, list):
            records = raw
        else:
            records = [raw]

        df = pd.DataFrame(records)
        
        if df.empty:
            return df
        
        # Normalize column names
        df.rename(columns=POLYGON_COLUMN_MAP, inplace=True)
        
        # Handle timestamp - convert to date only
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df["timestamp"] = df["timestamp"].dt.date
        
        # Add ticker column
        ticker = params.get('ticker') or params.get('symbol')
        if ticker and 'ticker' not in df.columns:
            df.insert(0, 'ticker', ticker)
        
        return df
    
    def _parse_snapshot_response(self, raw: Any, params: Dict[str, Any]) -> pd.DataFrame:
        """Parse FULL_MARKET_SNAPSHOT response"""
        # The snapshot returns a list of ticker objects
        if isinstance(raw, list):
            tickers_data = raw
        elif hasattr(raw, '__dict__'):
            raw_dict = raw.__dict__
            tickers_data = raw_dict.get('tickers', raw_dict.get('results', []))
        elif isinstance(raw, dict):
            tickers_data = raw.get('tickers', raw.get('results', []))
        else:
            tickers_data = []
        
        if not tickers_data:
            return pd.DataFrame()
        
        records = []
        for ticker_info in tickers_data:
            # Convert to dict if needed
            if hasattr(ticker_info, '__dict__'):
                ticker_info = ticker_info.__dict__
            
            record = {'ticker': ticker_info.get('ticker', '')}
            
            # Extract day data
            day_data = ticker_info.get('day', {})
            if hasattr(day_data, '__dict__'):
                day_data = day_data.__dict__
            
            record['open'] = day_data.get('o', day_data.get('open'))
            record['high'] = day_data.get('h', day_data.get('high'))
            record['low'] = day_data.get('l', day_data.get('low'))
            record['close'] = day_data.get('c', day_data.get('close'))
            record['volume'] = day_data.get('v', day_data.get('volume'))
            record['vwap'] = day_data.get('vw', day_data.get('vwap'))
            
            # Extract prev day
            prev_data = ticker_info.get('prevDay', {})
            if hasattr(prev_data, '__dict__'):
                prev_data = prev_data.__dict__
            record['prev_close'] = prev_data.get('c', prev_data.get('close'))
            
            # Timestamp
            updated = ticker_info.get('updated')
            if updated:
                try:
                    record['timestamp'] = pd.to_datetime(updated, unit='ns', utc=True).date()
                except:
                    record['timestamp'] = None
            
            records.append(record)
        
        return pd.DataFrame(records)

    def _fetch_economic_indicator(self, indicator: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch Polygon economic indicator data (no ticker required)"""
        if not indicator:
            raise ValueError("Missing indicator name for Polygon economic data request")
        
        slug = self._resolve_indicator_slug(indicator)
        query_params = {
            key: value for key, value in features.items()
            if key not in {'endpoint_type', 'indicator'}
        }
        query_params['apiKey'] = self.api_key
        
        url = f"{self.fed_base_url}/{slug}"
        response = requests.get(url, params=query_params)
        response.raise_for_status()
        data = response.json()
        return {'data': data, 'features': features}
    
    def _parse_economic_indicator(self, raw: Any, params: Dict[str, Any]) -> pd.DataFrame:
        """Parse economic indicator dataset into a normalized DataFrame"""
        records = []
        if isinstance(raw, dict):
            if isinstance(raw.get('results'), list):
                records = raw['results']
            elif isinstance(raw.get('data'), list):
                records = raw['data']
        elif isinstance(raw, list):
            records = raw
        
        df = pd.DataFrame(records)
        if df.empty:
            return df
        
        if 'date' in df.columns and 'timestamp' not in df.columns:
            df.rename(columns={'date': 'timestamp'}, inplace=True)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.date
        
        value_column = next((col for col in ['value', 'v', 'measure'] if col in df.columns), None)
        if value_column and value_column != 'value':
            df.rename(columns={value_column: 'value'}, inplace=True)
            value_column = 'value'
        if value_column == 'value':
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        measurement_cols = [col for col in df.columns if col != 'timestamp' and col != 'value']
        if 'value' not in df.columns and measurement_cols:
            primary_col = measurement_cols[0]
            df['value'] = pd.to_numeric(df[primary_col], errors='coerce')
        
        ordered = [col for col in ['timestamp', 'value'] if col in df.columns]
        remaining = [col for col in df.columns if col not in ordered]
        if ordered:
            df = df[ordered + remaining]
            df = df.sort_values(ordered[0])
        df.reset_index(drop=True, inplace=True)
        return df
    
    def _resolve_indicator_slug(self, indicator_name: str) -> str:
        """Map friendly indicator names to Polygon slugs"""
        slug_map = {
            'INFLATION': 'inflation',
            'TREASURY_YIELD': 'treasury-yields',
            'CPI': 'inflation/cpi',
            'FEDERAL_FUNDS_RATE': 'federal-funds-rate',
            'RETAIL_SALES': 'retail-sales',
            'DURABLES': 'durable-goods',
            'UNEMPLOYMENT': 'unemployment-rate',
            'NONFARM_PAYROLL': 'nonfarm-payrolls',
        }
        lookup = indicator_name.upper()
        slug = slug_map.get(lookup, indicator_name.lower())
        return slug.replace('_', '-')
