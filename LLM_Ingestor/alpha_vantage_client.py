"""
Alpha Vantage Client - Fixed for economic indicators (no ticker required)
"""
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import requests

from base_api_client import BaseAPIClient

class AlphaVantageClient(BaseAPIClient):
    """API Client for Alpha Vantage - supports both stock and economic data"""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required for AlphaVantageClient")
        self.api_key = api_key
        self.base_url = 'https://www.alphavantage.co/query'
        self._standard_columns = ['open', 'high', 'low', 'close', 'volume']
        self.default_window_days = 30
        self.frequency_window_map = {
            'intraday': 100,
            '1min': 100,
            '5min': 100,
            '15min': 100,
            '30min': 100,
            '60min': 100,
            'daily': 30,
            'day': 30,
            'weekly': 26,
            'week': 26,
            'monthly': 12,
            'month': 12,
            'quarterly': 20,
            'quarter': 20,
            'annual': 10,
            'year': 10
        }
        
        # Economic indicator endpoints that DON'T require ticker
        self._economic_endpoints = {
            'REAL_GDP', 'REAL_GDP_PER_CAPITA', 'TREASURY_YIELD', 'FEDERAL_FUNDS_RATE',
            'CPI', 'INFLATION', 'RETAIL_SALES', 'DURABLES', 'UNEMPLOYMENT', 'NONFARM_PAYROLL'
        }

    def fetch_data(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch data from Alpha Vantage"""
        function_name = features.get('function', '')
        
        params = {
            "apikey": self.api_key,
            "datatype": "json"
        }
        
        # Check if this is an economic indicator (no ticker needed)
        is_economic = function_name in self._economic_endpoints
        
        if is_economic:
            params['function'] = function_name
            # Add optional parameters for economic endpoints
            if 'interval' in features:
                params['interval'] = features['interval']
            if 'maturity' in features:
                params['maturity'] = features['maturity']
        else:
            # Stock endpoint - requires ticker
            ticker = features.get('ticker') or features.get('symbol')
            if not ticker:
                raise ValueError("Missing required parameter: 'ticker' or 'symbol'")
            
            params['symbol'] = ticker
            
            if function_name:
                params['function'] = function_name
                
                if function_name == 'TIME_SERIES_INTRADAY':
                    interval = features.get('interval') or features.get('timespan', '5min')
                    params['interval'] = interval
                    params['outputsize'] = features.get('outputsize', 'compact')
                    if 'month' in features:
                        params['month'] = features['month']
                        params['outputsize'] = features.get('outputsize', 'full')
                elif function_name in ['TIME_SERIES_DAILY', 'TIME_SERIES_DAILY_ADJUSTED']:
                    params['outputsize'] = features.get('outputsize', 'full')
            else:
                # Fallback to timespan-based logic
                timespan = features.get('timespan') or features.get('interval', 'day')
                timespan = timespan.lower()

                if timespan in ['day', 'daily']:
                    params['function'] = 'TIME_SERIES_DAILY'
                    params['outputsize'] = features.get('outputsize', 'full')
                elif timespan in ['week', 'weekly']:
                    params['function'] = 'TIME_SERIES_WEEKLY'
                elif timespan in ['month', 'monthly']:
                    params['function'] = 'TIME_SERIES_MONTHLY'
                elif timespan in ['1min', '5min', '15min', '30min', '60min']:
                    params['function'] = 'TIME_SERIES_INTRADAY'
                    params['interval'] = timespan
                    params['outputsize'] = features.get('outputsize', 'compact')
                else:
                    raise ValueError(f"Unsupported timespan: {timespan}")

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data:
                raise ValueError("Alpha Vantage API returned an empty response.")
            if "Error Message" in data:
                raise ValueError(f"Alpha Vantage API Error: {data['Error Message']}")
            if "Note" in data and "API call frequency" in data["Note"]:
                print(f"Warning: Rate limit hit. Message: {data['Note']}")

            return {"data": data, "features": features}

        except requests.exceptions.RequestException as e:
            print(f"HTTP Request failed: {e}")
            raise

    def parse_response(self, response_package: Dict[str, Any]) -> pd.DataFrame:
        """Parse Alpha Vantage response into DataFrame"""
        data = response_package["data"]
        features = response_package["features"]
        function_name = features.get('function', '')
        
        # Check if economic indicator
        is_economic = function_name in self._economic_endpoints
        
        if is_economic:
            return self._parse_economic_response(data, features)
        else:
            return self._parse_stock_response(data, features)
    
    def _parse_economic_response(self, data: Dict, features: Dict) -> pd.DataFrame:
        """Parse economic indicator response"""
        # Economic data comes in 'data' key
        if 'data' not in data:
            raise ValueError(f"Could not find 'data' key in economic response: {list(data.keys())}")
        
        records = data['data']
        df = pd.DataFrame(records)
        
        if df.empty:
            return df
        
        # Rename 'date' to 'timestamp' for consistency
        if 'date' in df.columns:
            df.rename(columns={'date': 'timestamp'}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date
        
        # Convert value to numeric
        if 'value' in df.columns:
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
            df.reset_index(drop=True, inplace=True)
        
        return self._apply_requested_window(df, features)
    
    def _parse_stock_response(self, data: Dict, features: Dict) -> pd.DataFrame:
        """Parse stock time series response"""
        # Find time series key
        time_series_key = None
        for key in data.keys():
            if "Time Series" in key:
                time_series_key = key
                break

        if not time_series_key:
            if data.get("Meta Data"):
                print(f"Warning: No time series data found. Returning empty DataFrame.")
                return pd.DataFrame()
            else:
                raise ValueError(f"Could not find time series data key: {list(data.keys())}")

        time_series_data = data[time_series_key]

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series_data, orient='index')

        # Clean column names
        def clean_column_name(col_name):
            if '. ' in col_name:
                cleaned = col_name.split('. ')[1]
                return cleaned.replace(' ', '_').lower()
            return col_name.lower()
        
        df.rename(columns=clean_column_name, inplace=True)

        # Convert index to timestamp column (date only)
        df.index = pd.to_datetime(df.index)
        df = df.reset_index()
        df.rename(columns={'index': 'timestamp'}, inplace=True)
        df['timestamp'] = df['timestamp'].dt.date
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        df.reset_index(drop=True, inplace=True)

        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'adjusted_close', 'dividend_amount', 'split_coefficient']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Add ticker column
        ticker = features.get('ticker') or features.get('symbol')
        if ticker:
            df.insert(0, 'ticker', ticker)

        return self._apply_requested_window(df, features)
    
    def _apply_requested_window(self, df: pd.DataFrame, features: Dict[str, Any]) -> pd.DataFrame:
        """Trim Alpha Vantage datasets to the requested timeframe"""
        if df.empty or 'timestamp' not in df.columns:
            return df
        
        df = df.sort_values('timestamp')
        timestamps = df['timestamp']
        latest_timestamp = timestamps.max()
        if isinstance(latest_timestamp, pd.Timestamp):
            latest_timestamp = latest_timestamp.date()
        if pd.isna(latest_timestamp):
            return df
        
        start_value = features.get('from') or features.get('start_date')
        end_value = features.get('to') or features.get('end_date')
        explicit_window = any([start_value, end_value, features.get('date')])
        
        if not start_value and 'date' in features:
            start_value = features['date']
        if not end_value and 'date' in features:
            end_value = features['date']
        
        start_date = self._parse_date(start_value)
        end_date = self._parse_date(end_value)
        
        filtered = df.copy()
        applied_window = False
        
        if start_date or end_date:
            if end_date is None:
                end_date = latest_timestamp
            if start_date is None:
                start_date = end_date - timedelta(days=self.default_window_days)
            filtered = filtered[(filtered['timestamp'] >= start_date) & (filtered['timestamp'] <= end_date)]
            applied_window = True
        
        if not applied_window and explicit_window and end_date:
            filtered = filtered[filtered['timestamp'] <= end_date]
            applied_window = True
        
        if not applied_window:
            limit_value = self._safe_int(features.get('limit'))
            if limit_value:
                filtered = filtered.tail(limit_value)
                applied_window = True
        
        if not applied_window:
            frequency = self._extract_frequency(features)
            if frequency in self.frequency_window_map:
                rows_to_keep = self.frequency_window_map[frequency]
                filtered = filtered.tail(rows_to_keep)
                applied_window = True
        
        if not applied_window:
            start_date = latest_timestamp - timedelta(days=self.default_window_days)
            filtered = filtered[filtered['timestamp'] >= start_date]
        
        if filtered.empty:
            filtered = df.tail(self.default_window_days).copy()
        
        filtered = filtered.sort_values('timestamp')
        filtered.reset_index(drop=True, inplace=True)
        return filtered
    
    def _parse_date(self, value: Any):
        if not value:
            return None
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime().date()
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if hasattr(value, 'date'):
            try:
                return value.date()
            except Exception:
                return None
        try:
            return datetime.strptime(str(value), "%Y-%m-%d").date()
        except ValueError:
            return None
    
    def _extract_frequency(self, features: Dict[str, Any]) -> Optional[str]:
        frequency = features.get('interval') or features.get('timespan') or features.get('frequency')
        if isinstance(frequency, str):
            return frequency.lower()
        return None
    
    def _safe_int(self, value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except (ValueError, TypeError):
            return None
