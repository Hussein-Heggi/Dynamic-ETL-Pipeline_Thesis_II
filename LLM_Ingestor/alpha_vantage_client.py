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

        # ------------------------------------------------------------------
        # Fundamental data endpoints
        #
        # The Alpha Vantage "Fundamental Data" section exposes a number of
        # endpoints that return company financial statements, corporate actions,
        # ETF profiles and various calendars.  They fall into two broad
        # categories: JSON responses and CSV responses.  We keep sets of
        # function names here to simplify routing in `fetch_data` and
        # `parse_response`.

        # Fundamental endpoints that return JSON.  Most of these require a
        # `symbol` (ticker) parameter, except where noted in the official
        # documentation.
        self._fundamental_json_endpoints = {
            'OVERVIEW',            # Company overview and ratios for a ticker
            'ETF_PROFILE',         # ETF profile and holdings for an ETF symbol
            'DIVIDENDS',           # Dividend history and future declarations
            'SPLITS',              # Historical stock split events
            'INCOME_STATEMENT',    # Annual and quarterly income statements
            'BALANCE_SHEET',       # Annual and quarterly balance sheets
            'CASH_FLOW',           # Annual and quarterly cash flow statements
            'SHARES_OUTSTANDING',  # Quarterly diluted and basic shares outstanding
            'EARNINGS',            # EPS history with analyst estimates and surprises
            'EARNINGS_ESTIMATES'   # EPS and revenue estimates (annual & quarterly)
        }

        # Fundamental endpoints that return CSV.  These endpoints may or may
        # not accept a `symbol`; they return tabular data and are parsed as
        # CSV rather than JSON.  A default horizon or state may apply when
        # omitted (see documentation for details).
        self._fundamental_csv_endpoints = {
            'LISTING_STATUS',      # Listing & delisting status for US stocks/ETFs
            'EARNINGS_CALENDAR',   # Upcoming earnings events (optionally for a symbol)
            'IPO_CALENDAR'         # Upcoming IPOs (global)
        }

    def fetch_data(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch data from Alpha Vantage"""
        function_name = (features.get('function') or '').strip()

        # Prepare base parameters.  By default return JSON unless a CSV endpoint
        # is requested explicitly below.  Do not set the `function` here yet
        # since some endpoints require no function (timespan fallback logic).
        params: Dict[str, Any] = {
            "apikey": self.api_key,
        }

        # ------------------------------------------------------------------
        # Economic indicators
        # ------------------------------------------------------------------
        is_economic = function_name in self._economic_endpoints
        if is_economic:
            # Economic endpoints never require a ticker.  Pass through
            # optional interval and maturity parameters.
            params['function'] = function_name
            params['datatype'] = 'json'
            if 'interval' in features:
                params['interval'] = features['interval']
            if 'maturity' in features:
                params['maturity'] = features['maturity']
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

        # ------------------------------------------------------------------
        # Fundamental data endpoints (CSV)
        # ------------------------------------------------------------------
        if function_name in self._fundamental_csv_endpoints:
            params['function'] = function_name
            # For CSV endpoints we explicitly request CSV format unless the user
            # overrides it.  Alpha Vantage serves CSV by default for these
            # endpoints, but including the parameter makes the intent clear.
            params['datatype'] = features.get('datatype', 'csv')
            # Optional parameters
            ticker = features.get('ticker') or features.get('symbol')
            if ticker:
                params['symbol'] = ticker
            if 'date' in features:
                params['date'] = features['date']
            if 'state' in features:
                params['state'] = features['state']
            if 'horizon' in features:
                params['horizon'] = features['horizon']
            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                csv_text = response.text
                if not csv_text:
                    raise ValueError("Alpha Vantage API returned an empty response.")
                return {"data": csv_text, "features": features}
            except requests.exceptions.RequestException as e:
                print(f"HTTP Request failed: {e}")
                raise

        # ------------------------------------------------------------------
        # Fundamental data endpoints (JSON)
        # ------------------------------------------------------------------
        if function_name in self._fundamental_json_endpoints:
            params['function'] = function_name
            params['datatype'] = features.get('datatype', 'json')
            # Many fundamental endpoints require a symbol; some such as
            # dividends/splits do not operate without one.  We include it if
            # provided but do not enforce here, allowing endpoints like
            # LISTING_STATUS and EARNINGS_CALENDAR to omit it.
            ticker = features.get('ticker') or features.get('symbol')
            if ticker:
                params['symbol'] = ticker
            # Pass through optional parameters
            for opt in ('horizon', 'date', 'state'):
                if opt in features:
                    params[opt] = features[opt]
            try:
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
                if not data:
                    raise ValueError("Alpha Vantage API returned an empty response.")
                if "Error Message" in data:
                    raise ValueError(f"Alpha Vantage API Error: {data['Error Message']}")
                if "Note" in data and "API call frequency" in data.get("Note", ""):
                    print(f"Warning: Rate limit hit. Message: {data['Note']}")
                return {"data": data, "features": features}
            except requests.exceptions.RequestException as e:
                print(f"HTTP Request failed: {e}")
                raise

        # ------------------------------------------------------------------
        # Stock time series endpoints (default)
        # ------------------------------------------------------------------
        # Determine ticker/symbol; these endpoints always require a ticker
        ticker = features.get('ticker') or features.get('symbol')
        if not ticker:
            raise ValueError("Missing required parameter: 'ticker' or 'symbol'")
        params['symbol'] = ticker
        # If an explicit function is provided (e.g., TIME_SERIES_DAILY) we use
        # that; otherwise we derive it from the timespan/interval.
        if function_name:
            params['function'] = function_name
            params['datatype'] = features.get('datatype', 'json')
            if function_name == 'TIME_SERIES_INTRADAY':
                interval = features.get('interval') or features.get('timespan', '5min')
                params['interval'] = interval
                params['outputsize'] = features.get('outputsize', 'compact')
                # If the user specifies a month for intraday data, request
                # full output for that month.
                if 'month' in features:
                    params['month'] = features['month']
                    params['outputsize'] = features.get('outputsize', 'full')
            elif function_name in ['TIME_SERIES_DAILY', 'TIME_SERIES_DAILY_ADJUSTED']:
                params['outputsize'] = features.get('outputsize', 'full')
        else:
            # Fallback to timespan-based logic
            timespan = features.get('timespan') or features.get('interval', 'day')
            timespan = str(timespan).lower()
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
        # Execute request for time series endpoints
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            if not data:
                raise ValueError("Alpha Vantage API returned an empty response.")
            if "Error Message" in data:
                raise ValueError(f"Alpha Vantage API Error: {data['Error Message']}")
            if "Note" in data and "API call frequency" in data.get("Note", ""):
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
        function_name = (function_name or '').strip()
        # Economic indicators
        if function_name in self._economic_endpoints:
            return self._parse_economic_response(data, features)

        # Fundamental CSV endpoints
        if function_name in self._fundamental_csv_endpoints:
            return self._parse_csv_response(data, features)

        # Fundamental JSON endpoints
        if function_name in self._fundamental_json_endpoints:
            # Statements (income, balance sheet, cash flow)
            if function_name in {'INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW'}:
                return self._parse_fundamental_statement(data, features)
            # Earnings history
            if function_name == 'EARNINGS':
                return self._parse_earnings_response(data, features)
            # Earnings estimates
            if function_name == 'EARNINGS_ESTIMATES':
                return self._parse_earnings_estimates_response(data, features)
            # All other fundamental JSON responses (overview, ETF_PROFILE,
            # dividends, splits, shares_outstanding)
            return self._parse_simple_json_response(data, features)

        # Default to stock time series parsing
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

    # ----------------------------------------------------------------------
    # Fundamental parsing helpers
    #
    def _parse_csv_response(self, csv_text: str, features: Dict) -> pd.DataFrame:
        """Parse CSV responses from fundamental endpoints (listing status, calendars)"""
        from io import StringIO
        try:
            df = pd.read_csv(StringIO(csv_text))
        except Exception as e:
            raise ValueError(f"Failed to parse CSV response: {e}")
        # If there is a ticker provided, insert it as a column for consistency
        ticker = features.get('ticker') or features.get('symbol')
        if ticker and 'symbol' not in df.columns and 'ticker' not in df.columns:
            df.insert(0, 'ticker', ticker)
        return df

    def _parse_fundamental_statement(self, data: Dict[str, Any], features: Dict[str, Any]) -> pd.DataFrame:
        """Parse annual and quarterly fundamental statements (income, balance sheet, cash flow)"""
        symbol = data.get('symbol') or features.get('ticker') or features.get('symbol')
        frames: List[pd.DataFrame] = []
        # Each statement has annualReports and quarterlyReports lists
        for key, period_label in [('annualReports', 'annual'), ('quarterlyReports', 'quarterly')]:
            reports = data.get(key)
            if isinstance(reports, list) and reports:
                df = pd.DataFrame(reports)
                # Add a period indicator
                df['period'] = period_label
                # Rename fiscalDateEnding to timestamp
                if 'fiscalDateEnding' in df.columns:
                    df = df.rename(columns={'fiscalDateEnding': 'timestamp'})
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.date
                # Convert numeric-like columns to numeric where possible
                for col in df.columns:
                    if col not in ['timestamp', 'period', 'reportedCurrency']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True, sort=False)
        # Include ticker column if available
        if symbol:
            combined.insert(0, 'ticker', symbol)
        # Sort by timestamp if present
        if 'timestamp' in combined.columns:
            combined = combined.sort_values(['timestamp', 'period'])
            combined.reset_index(drop=True, inplace=True)
        return combined

    def _parse_earnings_response(self, data: Dict[str, Any], features: Dict[str, Any]) -> pd.DataFrame:
        """Parse earnings history (EPS) with analyst estimates and surprise metrics"""
        symbol = data.get('symbol') or features.get('ticker') or features.get('symbol')
        frames: List[pd.DataFrame] = []
        for key, period_label in [('annualEarnings', 'annual'), ('quarterlyEarnings', 'quarterly')]:
            items = data.get(key)
            if isinstance(items, list) and items:
                df = pd.DataFrame(items)
                df['period'] = period_label
                # Rename fiscalDateEnding to timestamp
                if 'fiscalDateEnding' in df.columns:
                    df = df.rename(columns={'fiscalDateEnding': 'timestamp'})
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.date
                # Rename reportedDate to reported_timestamp as date
                if 'reportedDate' in df.columns:
                    df['reportedDate'] = pd.to_datetime(df['reportedDate'], errors='coerce').dt.date
                # Convert numeric columns (reportedEPS, estimatedEPS, surprise, surprisePercentage)
                numeric_cols = ['reportedEPS', 'estimatedEPS', 'surprise', 'surprisePercentage']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True, sort=False)
        if symbol:
            combined.insert(0, 'ticker', symbol)
        # Sort by timestamp if available
        if 'timestamp' in combined.columns:
            combined = combined.sort_values(['timestamp', 'period'])
            combined.reset_index(drop=True, inplace=True)
        return combined

    def _parse_earnings_estimates_response(self, data: Dict[str, Any], features: Dict[str, Any]) -> pd.DataFrame:
        """Parse earnings estimates response which may contain multiple lists"""
        symbol = data.get('symbol') or features.get('ticker') or features.get('symbol')
        frames: List[pd.DataFrame] = []
        for key, value in data.items():
            if isinstance(value, list) and value:
                # Determine period label based on key
                lower_key = key.lower()
                if 'annual' in lower_key:
                    period_label = 'annual'
                elif 'quarter' in lower_key:
                    period_label = 'quarterly'
                else:
                    period_label = key
                df = pd.DataFrame(value)
                df['period'] = period_label
                # Rename fiscalDateEnding or fiscalDate to timestamp if exists
                for date_col in ['fiscalDateEnding', 'fiscalDate']:
                    if date_col in df.columns:
                        df = df.rename(columns={date_col: 'timestamp'})
                        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.date
                        break
                # Convert potential numeric columns
                for col in df.columns:
                    if col not in ['timestamp', 'period']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True, sort=False)
        if symbol:
            combined.insert(0, 'ticker', symbol)
        if 'timestamp' in combined.columns:
            combined = combined.sort_values(['timestamp', 'period'])
            combined.reset_index(drop=True, inplace=True)
        return combined

    def _parse_simple_json_response(self, data: Dict[str, Any], features: Dict[str, Any]) -> pd.DataFrame:
        """Parse simple JSON responses (overview, ETF profile, dividends, splits, shares outstanding)"""
        # Flatten the dictionary into a single-row DataFrame.  If any values
        # contain nested lists or dictionaries, keep them as-is.
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object for simple fundamental endpoint")
        df = pd.DataFrame([data])
        # Insert ticker column if provided but not already present
        ticker = data.get('symbol') or features.get('ticker') or features.get('symbol')
        if ticker and ('ticker' not in df.columns):
            df.insert(0, 'ticker', ticker)
        # Attempt to convert numeric-like columns
        for col in df.columns:
            # Skip object columns that are clearly non-numeric (like lists/dicts)
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except Exception:
                    # If conversion fails leave as is
                    pass
        return df
    
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
