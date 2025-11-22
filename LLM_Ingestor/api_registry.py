"""
API Registry - Single source of truth for all API metadata, endpoints, and parameter schemas.
COMPREHENSIVE VERSION - Includes all stock and economic indicator endpoints
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Literal
from enum import Enum


# ============================================================================
# Parameter Schema Models
# ============================================================================

class ParameterType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    DATE = "date"
    ENUM = "enum"
    BOOLEAN = "boolean"


class ParameterSchema(BaseModel):
    """Schema definition for an API parameter"""
    name: str
    type: ParameterType
    required: bool
    description: str
    aliases: List[str] = Field(default_factory=list)
    valid_values: Optional[List[Any]] = None  # For enums
    default_value: Optional[Any] = None
    pattern: Optional[str] = None  # Regex pattern for strings
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    format: Optional[str] = None  # e.g., "YYYY-MM-DD" for dates


class ParameterMapping(BaseModel):
    """Maps QueryIntent entities to endpoint parameters"""
    source: str  # e.g., "extracted_entities.tickers[0]"
    transform: Optional[Dict[str, Any]] = None  # Transformation rules


class EndpointSpec(BaseModel):
    """Complete specification for an API endpoint"""
    name: str
    description: str
    parameters: List[ParameterSchema]
    parameter_mappings: Dict[str, ParameterMapping] = Field(default_factory=dict)
    alternative_endpoints: List[str] = Field(default_factory=list)
    data_category: Literal["stock", "economic_indicator", "forex", "crypto"] = "stock"


class APISpec(BaseModel):
    """Complete specification for an API"""
    name: str
    base_url: str
    endpoints: List[EndpointSpec]


# ============================================================================
# Polygon API Builder
# ============================================================================

def build_polygon_api() -> APISpec:
    """Build comprehensive Polygon API specification"""
    return APISpec(
        name="polygon",
        base_url="https://api.polygon.io",
        endpoints=[
            # ================================================================
            # STOCK ENDPOINTS
            # ================================================================
            
            # Stock Aggregates (Primary endpoint for stock data)
            EndpointSpec(
                name="get_aggs",
                description="Historical stock price data with aggregated bars showing open high low close volume (OHLCV) across flexible timeframes from minute to year for comprehensive technical analysis, backtesting, and financial modeling of stock prices and trading activity",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="ticker",
                        type=ParameterType.STRING,
                        required=True,
                        description="Stock ticker symbol",
                        aliases=["symbol", "stock"],
                        pattern="^[A-Z]{1,5}$"
                    ),
                    ParameterSchema(
                        name="multiplier",
                        type=ParameterType.INTEGER,
                        required=False,
                        description="Size of the timespan multiplier",
                        default_value=1,
                        min_value=1,
                        max_value=1000
                    ),
                    ParameterSchema(
                        name="timespan",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Size of the time window",
                        aliases=["interval", "frequency"],
                        valid_values=["minute", "hour", "day", "week", "month", "quarter", "year"],
                        default_value="day"
                    ),
                    ParameterSchema(
                        name="from",
                        type=ParameterType.DATE,
                        required=False,
                        description="Start of the aggregate time window",
                        aliases=["start_date", "from_date"],
                        format="YYYY-MM-DD"
                    ),
                    ParameterSchema(
                        name="to",
                        type=ParameterType.DATE,
                        required=False,
                        description="End of the aggregate time window",
                        aliases=["end_date", "to_date"],
                        format="YYYY-MM-DD"
                    ),
                ],
                parameter_mappings={
                    "ticker": ParameterMapping(source="extracted_entities.tickers[0]"),
                    "multiplier": ParameterMapping(
                        source="extracted_entities.frequency",
                        transform={
                            "daily": 1,
                            "weekly": 1,
                            "monthly": 1,
                            "intraday": 5
                        }
                    ),
                    "timespan": ParameterMapping(
                        source="extracted_entities.frequency",
                        transform={
                            "daily": "day",
                            "weekly": "week",
                            "monthly": "month",
                            "intraday": "minute"
                        }
                    ),
                    "from": ParameterMapping(source="extracted_entities.time_range.from_date"),
                    "to": ParameterMapping(source="extracted_entities.time_range.to_date"),
                },
                alternative_endpoints=["TIME_SERIES_DAILY", "TIME_SERIES_WEEKLY", "TIME_SERIES_MONTHLY"]
            ),
            
            # Previous Close
            EndpointSpec(
                name="get_previous_close_agg",
                description="Previous trading day closing prices and market summary data for quick recent lookback and baseline comparison",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="ticker",
                        type=ParameterType.STRING,
                        required=True,
                        description="Stock ticker symbol",
                        aliases=["symbol", "stock"]
                    ),
                ],
                parameter_mappings={
                    "ticker": ParameterMapping(source="extracted_entities.tickers[0]"),
                },
                alternative_endpoints=[]
            ),
            
            # Daily Open/Close
            EndpointSpec(
                name="get_daily_open_close_agg",
                description="Specific daily opening and closing prices for individual stocks on a particular trading date with full OHLC data",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="ticker",
                        type=ParameterType.STRING,
                        required=True,
                        description="Stock ticker symbol",
                        aliases=["symbol", "stock"]
                    ),
                    ParameterSchema(
                        name="date",
                        type=ParameterType.DATE,
                        required=True,
                        description="Date for the open/close data",
                        format="YYYY-MM-DD"
                    ),
                ],
                parameter_mappings={
                    "ticker": ParameterMapping(source="extracted_entities.tickers[0]"),
                    "date": ParameterMapping(source="extracted_entities.time_range.from_date"),
                },
                alternative_endpoints=[]
            ),
            
            # Grouped Daily Aggregates
            EndpointSpec(
                name="get_grouped_daily_aggs",
                description="Grouped daily aggregates for entire stock market on a specific date for market-wide analysis and screening",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="date",
                        type=ParameterType.DATE,
                        required=True,
                        description="Date for the grouped daily aggregates",
                        aliases=["from"],
                        format="YYYY-MM-DD"
                    ),
                ],
                parameter_mappings={
                    "date": ParameterMapping(source="extracted_entities.time_range.from_date"),
                },
                alternative_endpoints=[]
            ),

            # ----------------------------------------------------------------
            # ADDITIONAL STOCK ENDPOINTS (Polygon Fundamentals & Snapshots)

            # Short Interest
            EndpointSpec(
                name="SHORT_INTEREST",
                description=(
                    "Bi-monthly short interest data showing the total shares sold short yet to be covered, "
                    "average daily volume and days-to-cover ratio for measuring sentiment and potential short squeezes"
                ),
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="ticker",
                        type=ParameterType.STRING,
                        required=False,
                        description="Filter results to a specific stock ticker symbol",
                        aliases=["symbol", "stock"]
                    ),
                    ParameterSchema(
                        name="days_to_cover",
                        type=ParameterType.FLOAT,
                        required=False,
                        description="Minimum days-to-cover ratio to filter results (short interest divided by average daily volume)",
                        min_value=0
                    ),
                    ParameterSchema(
                        name="settlement_date",
                        type=ParameterType.DATE,
                        required=False,
                        description="Date of the FINRA reporting period (YYYY-MM-DD)",
                        format="YYYY-MM-DD"
                    ),
                    ParameterSchema(
                        name="avg_daily_volume",
                        type=ParameterType.FLOAT,
                        required=False,
                        description="Minimum average daily volume to filter results",
                        min_value=0
                    ),
                    ParameterSchema(
                        name="limit",
                        type=ParameterType.INTEGER,
                        required=False,
                        description="Maximum number of results to return (default 10, max 50000)",
                        default_value=10,
                        min_value=1,
                        max_value=50000
                    ),
                    ParameterSchema(
                        name="sort",
                        type=ParameterType.STRING,
                        required=False,
                        description="Column to sort by with optional .asc/.desc suffix"
                    ),
                ],
                parameter_mappings={
                    "ticker": ParameterMapping(source="extracted_entities.tickers[0]"),
                    "settlement_date": ParameterMapping(source="extracted_entities.time_range.from_date"),
                    "limit": ParameterMapping(source="extracted_entities.limit"),
                    "sort": ParameterMapping(source="extracted_entities.sort_column"),
                },
                alternative_endpoints=[]
            ),

            # Short Volume
            EndpointSpec(
                name="SHORT_VOLUME",
                description=(
                    "Daily short sale volume data across trading venues including exempt volumes and short volume ratios for tracking short-term sentiment"
                ),
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="ticker",
                        type=ParameterType.STRING,
                        required=False,
                        description="Filter results to a specific stock ticker symbol",
                        aliases=["symbol", "stock"]
                    ),
                    ParameterSchema(
                        name="date",
                        type=ParameterType.DATE,
                        required=False,
                        description="Calendar date of the short volume observation (YYYY-MM-DD)",
                        format="YYYY-MM-DD",
                        aliases=["on"]
                    ),
                    ParameterSchema(
                        name="short_volume_ratio",
                        type=ParameterType.FLOAT,
                        required=False,
                        description="Minimum short volume ratio (%) to filter results",
                        min_value=0
                    ),
                    ParameterSchema(
                        name="total_volume",
                        type=ParameterType.INTEGER,
                        required=False,
                        description="Minimum total traded volume to filter results",
                        min_value=0
                    ),
                    ParameterSchema(
                        name="limit",
                        type=ParameterType.INTEGER,
                        required=False,
                        description="Maximum number of results to return (default 10, max 50000)",
                        default_value=10,
                        min_value=1,
                        max_value=50000
                    ),
                    ParameterSchema(
                        name="sort",
                        type=ParameterType.STRING,
                        required=False,
                        description="Column to sort by with optional .asc/.desc suffix"
                    ),
                ],
                parameter_mappings={
                    "ticker": ParameterMapping(source="extracted_entities.tickers[0]"),
                    "date": ParameterMapping(source="extracted_entities.time_range.from_date"),
                    "limit": ParameterMapping(source="extracted_entities.limit"),
                    "sort": ParameterMapping(source="extracted_entities.sort_column"),
                },
                alternative_endpoints=[]
            ),

            # Unified Snapshot
            EndpointSpec(
                name="UNIFIED_SNAPSHOT",
                description=(
                    "Unified snapshot data consolidating last trade, last quote, session statistics and price information across supported asset classes; "
                    "defaults to stocks but can query other types via the type parameter"
                ),
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="ticker",
                        type=ParameterType.STRING,
                        required=False,
                        description="Filter snapshot for a single ticker; omit to retrieve multiple assets",
                        aliases=["symbol", "stock"]
                    ),
                    ParameterSchema(
                        name="type",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Asset class type to snapshot",
                        valid_values=["stocks", "options", "indices", "crypto", "fx", "commodities"],
                        default_value="stocks"
                    ),
                    ParameterSchema(
                        name="order",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Sort order",
                        valid_values=["asc", "desc"],
                        default_value="desc"
                    ),
                    ParameterSchema(
                        name="limit",
                        type=ParameterType.INTEGER,
                        required=False,
                        description="Maximum number of results to return (default 10, max 250)",
                        default_value=10,
                        min_value=1,
                        max_value=250
                    ),
                    ParameterSchema(
                        name="sort",
                        type=ParameterType.STRING,
                        required=False,
                        description="Field to sort results by"
                    ),
                ],
                parameter_mappings={
                    "ticker": ParameterMapping(source="extracted_entities.tickers[0]"),
                    "limit": ParameterMapping(source="extracted_entities.limit"),
                    "sort": ParameterMapping(source="extracted_entities.sort_column"),
                },
                alternative_endpoints=[]
            ),

            # Single Ticker Snapshot
            EndpointSpec(
                name="SNAPSHOT_TICKER",
                description="Real-time snapshot for a single US equity ticker including last trade, last quote, minute bar, day bar and previous day bar",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="ticker",
                        type=ParameterType.STRING,
                        required=True,
                        description="Stock ticker symbol",
                        aliases=["symbol", "stock"],
                        pattern="^[A-Z]{1,5}$"
                    ),
                ],
                parameter_mappings={
                    "ticker": ParameterMapping(source="extracted_entities.tickers[0]"),
                },
                alternative_endpoints=[]
            ),

            # Full Market Snapshot
            EndpointSpec(
                name="FULL_MARKET_SNAPSHOT",
                description="Comprehensive snapshot of the entire US equity market with optional filtering by ticker list and inclusion of OTC securities",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="tickers",
                        type=ParameterType.STRING,
                        required=False,
                        description="Comma-separated list of tickers to query; omit to return all stocks",
                        aliases=["symbols", "ticker_list"]
                    ),
                    ParameterSchema(
                        name="include_otc",
                        type=ParameterType.BOOLEAN,
                        required=False,
                        description="Whether to include over-the-counter securities",
                        default_value=False
                    ),
                ],
                parameter_mappings={
                    "tickers": ParameterMapping(source="extracted_entities.tickers"),
                    "include_otc": ParameterMapping(source="extracted_entities.include_otc"),
                },
                alternative_endpoints=[]
            ),

            # Top Market Movers
            EndpointSpec(
                name="TOP_MARKET_MOVERS",
                description="Top 20 gainers or losers in the US equity market for a given session, useful for momentum screening",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="direction",
                        type=ParameterType.ENUM,
                        required=True,
                        description="Direction of movers to return",
                        valid_values=["gainers", "losers"],
                        default_value="gainers"
                    ),
                    ParameterSchema(
                        name="include_otc",
                        type=ParameterType.BOOLEAN,
                        required=False,
                        description="Whether to include over-the-counter securities",
                        default_value=False
                    ),
                ],
                parameter_mappings={
                    "direction": ParameterMapping(source="extracted_entities.direction"),
                    "include_otc": ParameterMapping(source="extracted_entities.include_otc"),
                },
                alternative_endpoints=[]
            ),
            
            # ================================================================
            # ECONOMIC INDICATOR ENDPOINTS (Polygon)
            # ================================================================
            
            # Treasury Yields
            EndpointSpec(
                name="TREASURY_YIELD",
                description="U.S. Treasury yield curve data across standard maturities (1-month through 30-year) for interest-rate analysis and fixed-income valuation. Returns daily yields going back to 1962.",
                data_category="economic_indicator",
                parameters=[
                    ParameterSchema(
                        name="date",
                        type=ParameterType.STRING,
                        required=False,
                        description="Calendar date of the yield observation (YYYY-MM-DD)",
                        format="YYYY-MM-DD",
                        aliases=["on"],
                    ),
                    ParameterSchema(
                        name="limit",
                        type=ParameterType.INTEGER,
                        required=False,
                        description="Maximum number of results to return (default 100, max 50000)",
                        default_value=100,
                        min_value=1,
                        max_value=50000
                    ),
                    ParameterSchema(
                        name="sort",
                        type=ParameterType.STRING,
                        required=False,
                        description="Comma-separated list of sort columns with optional .asc or .desc suffix to specify sort direction (defaults to date ascending)"
                    ),
                ],
                parameter_mappings={
                    "date": ParameterMapping(source="extracted_entities.time_range.from_date"),
                    "limit": ParameterMapping(source="extracted_entities.limit"),
                    "sort": ParameterMapping(source="extracted_entities.sort_column"),
                },
                alternative_endpoints=[]
            ),
            
            # Inflation
            EndpointSpec(
                name="INFLATION",
                description="Consumer inflation indicators including headline and core CPI and PCE measures. Provides historical price change data for monetary policy and purchasing-power analysis.",
                data_category="economic_indicator",
                parameters=[
                    ParameterSchema(
                        name="date",
                        type=ParameterType.STRING,
                        required=False,
                        description="Calendar date of the observation (YYYY-MM-DD)",
                        format="YYYY-MM-DD",
                        aliases=["on"],
                    ),
                    ParameterSchema(
                        name="limit",
                        type=ParameterType.INTEGER,
                        required=False,
                        description="Maximum number of results to return (default 100, max 50000)",
                        default_value=100,
                        min_value=1,
                        max_value=50000
                    ),
                    ParameterSchema(
                        name="sort",
                        type=ParameterType.STRING,
                        required=False,
                        description="Comma-separated list of sort columns with optional .asc or .desc suffix (defaults to date ascending)"
                    ),
                ],
                parameter_mappings={
                    "date": ParameterMapping(source="extracted_entities.time_range.from_date"),
                    "limit": ParameterMapping(source="extracted_entities.limit"),
                    "sort": ParameterMapping(source="extracted_entities.sort_column"),
                },
                alternative_endpoints=[]
            ),
        ]
    )


# ============================================================================
# Alpha Vantage API Builder  
# ============================================================================

def build_alpha_vantage_api() -> APISpec:
    """Build comprehensive Alpha Vantage API specification"""
    return APISpec(
        name="alpha_vantage",
        base_url="https://www.alphavantage.co/query",
        endpoints=[
            # ================================================================
            # STOCK ENDPOINTS
            # ================================================================
            
            # Intraday
            EndpointSpec(
                name="TIME_SERIES_INTRADAY",
                description="Intraday time series stock data with intervals from 1min to 60min for short-term trading and real-time analysis",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="ticker",
                        type=ParameterType.STRING,
                        required=True,
                        description="Stock ticker symbol",
                        aliases=["symbol", "stock"],
                    ),
                    ParameterSchema(
                        name="timespan",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Time interval for data points",
                        aliases=["interval"],
                        valid_values=["1min", "5min", "15min", "30min", "60min"],
                        default_value="5min"
                    ),
                    ParameterSchema(
                        name="outputsize",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Compact returns latest 100 data points, full returns full-length intraday data",
                        valid_values=["compact", "full"],
                        default_value="compact"
                    ),
                ],
                parameter_mappings={
                    "ticker": ParameterMapping(source="extracted_entities.tickers[0]"),
                },
                alternative_endpoints=["get_aggs"]
            ),
            
            # Daily
            EndpointSpec(
                name="TIME_SERIES_DAILY",
                description="Daily historical stock prices with open, high, low, close and volume data for fundamental and technical analysis",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="ticker",
                        type=ParameterType.STRING,
                        required=True,
                        description="Stock ticker symbol",
                        aliases=["symbol", "stock"],
                    ),
                    ParameterSchema(
                        name="outputsize",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Compact returns latest 100 data points, full returns 20+ years",
                        valid_values=["compact", "full"],
                        default_value="compact"
                    ),
                ],
                parameter_mappings={
                    "ticker": ParameterMapping(source="extracted_entities.tickers[0]"),
                },
                alternative_endpoints=["get_aggs"]
            ),
            
            # Daily Adjusted
            EndpointSpec(
                name="TIME_SERIES_DAILY_ADJUSTED",
                description="Daily adjusted stock prices accounting for splits and dividends for accurate long-term historical analysis",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="ticker",
                        type=ParameterType.STRING,
                        required=True,
                        description="Stock ticker symbol",
                        aliases=["symbol", "stock"],
                    ),
                    ParameterSchema(
                        name="outputsize",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Compact returns latest 100 data points, full returns 20+ years",
                        valid_values=["compact", "full"],
                        default_value="compact"
                    ),
                ],
                parameter_mappings={
                    "ticker": ParameterMapping(source="extracted_entities.tickers[0]"),
                },
                alternative_endpoints=["get_aggs"]
            ),
            
            # Weekly
            EndpointSpec(
                name="TIME_SERIES_WEEKLY",
                description="Weekly aggregated stock price data for longer-term trend analysis and investment decisions",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="ticker",
                        type=ParameterType.STRING,
                        required=True,
                        description="Stock ticker symbol",
                        aliases=["symbol", "stock"],
                    ),
                    ParameterSchema(
                        name="outputsize",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Compact or full data",
                        valid_values=["compact", "full"],
                        default_value="full"
                    ),
                ],
                parameter_mappings={
                    "ticker": ParameterMapping(source="extracted_entities.tickers[0]"),
                },
                alternative_endpoints=["get_aggs"]
            ),
            
            # Weekly Adjusted
            EndpointSpec(
                name="TIME_SERIES_WEEKLY_ADJUSTED",
                description="Weekly adjusted stock prices accounting for splits and dividends for long-term portfolio analysis",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="ticker",
                        type=ParameterType.STRING,
                        required=True,
                        description="Stock ticker symbol",
                        aliases=["symbol", "stock"],
                    ),
                ],
                parameter_mappings={
                    "ticker": ParameterMapping(source="extracted_entities.tickers[0]"),
                },
                alternative_endpoints=["get_aggs"]
            ),
            
            # Monthly
            EndpointSpec(
                name="TIME_SERIES_MONTHLY",
                description="Monthly historical stock prices for long-term investment analysis and portfolio management",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="ticker",
                        type=ParameterType.STRING,
                        required=True,
                        description="Stock ticker symbol",
                        aliases=["symbol", "stock"],
                    ),
                    ParameterSchema(
                        name="outputsize",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Compact or full data",
                        valid_values=["compact", "full"],
                        default_value="full"
                    ),
                ],
                parameter_mappings={
                    "ticker": ParameterMapping(source="extracted_entities.tickers[0]"),
                    "outputsize": ParameterMapping(source="extracted_entities.time_range"),
                },
                alternative_endpoints=["get_aggs"]
            ),
            
            # Monthly Adjusted
            EndpointSpec(
                name="TIME_SERIES_MONTHLY_ADJUSTED",
                description="Monthly adjusted stock prices accounting for splits and dividends for comprehensive historical backesting",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="ticker",
                        type=ParameterType.STRING,
                        required=True,
                        description="Stock ticker symbol",
                        aliases=["symbol", "stock"],
                    ),
                ],
                parameter_mappings={
                    "ticker": ParameterMapping(source="extracted_entities.tickers[0]"),
                },
                alternative_endpoints=["get_aggs"]
            ),
            
            # ================================================================
            # ECONOMIC INDICATOR ENDPOINTS
            # ================================================================
            
            # Real GDP
            EndpointSpec(
                name="REAL_GDP",
                description="Real Gross Domestic Product data showing economic growth adjusted for inflation for macroeconomic analysis",
                data_category="economic_indicator",
                parameters=[
                    ParameterSchema(
                        name="interval",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Data interval frequency",
                        valid_values=["quarterly", "annual"],
                        default_value="quarterly"
                    ),
                    ParameterSchema(
                        name="datatype",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Response data format",
                        valid_values=["json", "csv"],
                        default_value="json"
                    ),
                ],
                parameter_mappings={
                    "interval": ParameterMapping(source="extracted_entities.frequency"),
                },
                alternative_endpoints=[]
            ),
            
            # Real GDP Per Capita
            EndpointSpec(
                name="REAL_GDP_PER_CAPITA",
                description="Real GDP per capita measuring economic output per person adjusted for inflation for standard of living analysis",
                data_category="economic_indicator",
                parameters=[
                    ParameterSchema(
                        name="datatype",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Response data format",
                        valid_values=["json", "csv"],
                        default_value="json"
                    ),
                ],
                parameter_mappings={},
                alternative_endpoints=[]
            ),
            
            # Treasury Yield
            EndpointSpec(
                name="TREASURY_YIELD",
                description="US Treasury bond yields across different maturities for interest rate analysis and fixed income valuation",
                data_category="economic_indicator",
                parameters=[
                    ParameterSchema(
                        name="interval",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Data interval frequency",
                        valid_values=["daily", "weekly", "monthly"],
                        default_value="daily"
                    ),
                    ParameterSchema(
                        name="maturity",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Bond maturity period",
                        valid_values=["3month", "2year", "5year", "7year", "10year", "30year"],
                        default_value="10year"
                    ),
                    ParameterSchema(
                        name="datatype",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Response data format",
                        valid_values=["json", "csv"],
                        default_value="json"
                    ),
                ],
                parameter_mappings={
                    "interval": ParameterMapping(source="extracted_entities.frequency"),
                },
                alternative_endpoints=[]
            ),
            
            # Federal Funds Rate
            EndpointSpec(
                name="FEDERAL_FUNDS_RATE",
                description="Federal Reserve federal funds rate for monetary policy analysis and interest rate forecasting",
                data_category="economic_indicator",
                parameters=[
                    ParameterSchema(
                        name="interval",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Data interval frequency",
                        valid_values=["daily", "weekly", "monthly"],
                        default_value="monthly"
                    ),
                    ParameterSchema(
                        name="datatype",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Response data format",
                        valid_values=["json", "csv"],
                        default_value="json"
                    ),
                ],
                parameter_mappings={
                    "interval": ParameterMapping(source="extracted_entities.frequency"),
                },
                alternative_endpoints=[]
            ),
            
            # CPI (Consumer Price Index)
            EndpointSpec(
                name="CPI",
                description="Consumer Price Index measuring inflation and cost of living changes for purchasing power analysis",
                data_category="economic_indicator",
                parameters=[
                    ParameterSchema(
                        name="interval",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Data interval frequency",
                        valid_values=["monthly", "semiannual"],
                        default_value="monthly"
                    ),
                    ParameterSchema(
                        name="datatype",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Response data format",
                        valid_values=["json", "csv"],
                        default_value="json"
                    ),
                ],
                parameter_mappings={
                    "interval": ParameterMapping(source="extracted_entities.frequency"),
                },
                alternative_endpoints=[]
            ),
            
            # Inflation
            EndpointSpec(
                name="INFLATION",
                description="Inflation rate data showing price level changes over time for economic stability assessment",
                data_category="economic_indicator",
                parameters=[
                    ParameterSchema(
                        name="datatype",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Response data format",
                        valid_values=["json", "csv"],
                        default_value="json"
                    ),
                ],
                parameter_mappings={},
                alternative_endpoints=[]
            ),
            
            # Retail Sales
            EndpointSpec(
                name="RETAIL_SALES",
                description="Monthly retail sales data measuring consumer spending patterns for economic health evaluation",
                data_category="economic_indicator",
                parameters=[
                    ParameterSchema(
                        name="datatype",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Response data format",
                        valid_values=["json", "csv"],
                        default_value="json"
                    ),
                ],
                parameter_mappings={},
                alternative_endpoints=[]
            ),
            
            # Durable Goods Orders
            EndpointSpec(
                name="DURABLES",
                description="Durable goods orders indicating business investment and manufacturing activity for economic forecasting",
                data_category="economic_indicator",
                parameters=[
                    ParameterSchema(
                        name="datatype",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Response data format",
                        valid_values=["json", "csv"],
                        default_value="json"
                    ),
                ],
                parameter_mappings={},
                alternative_endpoints=[]
            ),
            
            # Unemployment Rate
            EndpointSpec(
                name="UNEMPLOYMENT",
                description="Unemployment rate data showing labor market health and economic conditions for policy analysis",
                data_category="economic_indicator",
                parameters=[
                    ParameterSchema(
                        name="datatype",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Response data format",
                        valid_values=["json", "csv"],
                        default_value="json"
                    ),
                ],
                parameter_mappings={},
                alternative_endpoints=[]
            ),
            
            # Nonfarm Payroll
            EndpointSpec(
                name="NONFARM_PAYROLL",
                description="Monthly nonfarm payroll employment data measuring job creation and economic growth momentum",
                data_category="economic_indicator",
                parameters=[
                    ParameterSchema(
                        name="datatype",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Response data format",
                        valid_values=["json", "csv"],
                        default_value="json"
                    ),
                ],
                parameter_mappings={},
                alternative_endpoints=[]
            ),

            # =============================================================
            # FUNDAMENTAL DATA ENDPOINTS
            # =============================================================
            # Company Overview
            EndpointSpec(
                name="OVERVIEW",
                description="Comprehensive fundamental snapshot (sector, market cap, valuation ratios, profitability metrics, beta, share counts) for the specified equity",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="symbol",
                        type=ParameterType.STRING,
                        required=True,
                        description="Stock or equity ticker symbol",
                        aliases=["ticker", "stock"]
                    ),
                ],
                parameter_mappings={
                    "symbol": ParameterMapping(source="extracted_entities.tickers[0]")
                },
                alternative_endpoints=[]
            ),

            # ETF Profile & Holdings
            EndpointSpec(
                name="ETF_PROFILE",
                description="Detailed ETF fundamentals including strategy description, top holdings, asset/sector allocation weights, expense ratio, and issuance details",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="symbol",
                        type=ParameterType.STRING,
                        required=True,
                        description="ETF ticker symbol",
                        aliases=["ticker", "stock"]
                    ),
                ],
                parameter_mappings={
                    "symbol": ParameterMapping(source="extracted_entities.tickers[0]")
                },
                alternative_endpoints=[]
            ),

            # Corporate Action - Dividends
            EndpointSpec(
                name="DIVIDENDS",
                description="Complete dividend history including declaration/ex-date, payment date, currency, and amount for the specified equity",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="symbol",
                        type=ParameterType.STRING,
                        required=True,
                        description="Stock ticker symbol",
                        aliases=["ticker", "stock"]
                    ),
                    ParameterSchema(
                        name="datatype",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Response data format",
                        valid_values=["json", "csv"],
                        default_value="json"
                    ),
                ],
                parameter_mappings={
                    "symbol": ParameterMapping(source="extracted_entities.tickers[0]")
                },
                alternative_endpoints=[]
            ),

            # Corporate Action - Splits
            EndpointSpec(
                name="SPLITS",
                description="Historical stock split events with split ratios and effective dates for the specified equity",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="symbol",
                        type=ParameterType.STRING,
                        required=True,
                        description="Stock ticker symbol",
                        aliases=["ticker", "stock"]
                    ),
                    ParameterSchema(
                        name="datatype",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Response data format",
                        valid_values=["json", "csv"],
                        default_value="json"
                    ),
                ],
                parameter_mappings={
                    "symbol": ParameterMapping(source="extracted_entities.tickers[0]")
                },
                alternative_endpoints=[]
            ),

            # Income Statement
            EndpointSpec(
                name="INCOME_STATEMENT",
                description="Annual and quarterly income statements with GAAP/IFRS normalized revenue, gross profit, operating income, net income, EPS, and margin metrics",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="symbol",
                        type=ParameterType.STRING,
                        required=True,
                        description="Stock ticker symbol",
                        aliases=["ticker", "stock"]
                    ),
                    ParameterSchema(
                        name="period",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Reporting period frequency",
                        valid_values=["quarterly", "annual"],
                        default_value="quarterly"
                    ),
                ],
                parameter_mappings={
                    "symbol": ParameterMapping(source="extracted_entities.tickers[0]")
                },
                alternative_endpoints=[]
            ),

            # Balance Sheet
            EndpointSpec(
                name="BALANCE_SHEET",
                description="Annual and quarterly balance sheets highlighting assets, liabilities, shareholder equity, working capital, and leverage metrics",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="symbol",
                        type=ParameterType.STRING,
                        required=True,
                        description="Stock ticker symbol",
                        aliases=["ticker", "stock"]
                    ),
                    ParameterSchema(
                        name="period",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Reporting period frequency",
                        valid_values=["quarterly", "annual"],
                        default_value="quarterly"
                    ),
                ],
                parameter_mappings={
                    "symbol": ParameterMapping(source="extracted_entities.tickers[0]")
                },
                alternative_endpoints=[]
            ),

            # Cash Flow Statement
            EndpointSpec(
                name="CASH_FLOW",
                description="Annual and quarterly cash flow statements detailing operating, investing, financing flows plus free-cash-flow metrics",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="symbol",
                        type=ParameterType.STRING,
                        required=True,
                        description="Stock ticker symbol",
                        aliases=["ticker", "stock"]
                    ),
                    ParameterSchema(
                        name="period",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Reporting period frequency",
                        valid_values=["quarterly", "annual"],
                        default_value="quarterly"
                    ),
                ],
                parameter_mappings={
                    "symbol": ParameterMapping(source="extracted_entities.tickers[0]")
                },
                alternative_endpoints=[]
            ),

            # Shares Outstanding
            EndpointSpec(
                name="SHARES_OUTSTANDING",
                description="Quarterly basic and diluted shares outstanding plus source type to support per-share fundamental calculations",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="symbol",
                        type=ParameterType.STRING,
                        required=True,
                        description="Stock ticker symbol",
                        aliases=["ticker", "stock"]
                    ),
                    ParameterSchema(
                        name="datatype",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Response data format",
                        valid_values=["json", "csv"],
                        default_value="json"
                    ),
                ],
                parameter_mappings={
                    "symbol": ParameterMapping(source="extracted_entities.tickers[0]")
                },
                alternative_endpoints=[]
            ),

            # Earnings History
            EndpointSpec(
                name="EARNINGS",
                description="Annual and quarterly earnings (EPS) history including actual vs. estimate, surprise percentages, and announcement dates",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="symbol",
                        type=ParameterType.STRING,
                        required=True,
                        description="Stock ticker symbol",
                        aliases=["ticker", "stock"]
                    ),
                    ParameterSchema(
                        name="period",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Reporting period frequency",
                        valid_values=["quarterly", "annual"],
                        default_value="quarterly"
                    ),
                ],
                parameter_mappings={
                    "symbol": ParameterMapping(source="extracted_entities.tickers[0]")
                },
                alternative_endpoints=[]
            ),

            # Earnings Estimates
            EndpointSpec(
                name="EARNINGS_ESTIMATES",
                description="Forward EPS and revenue consensus estimates with analyst counts, revision trends, and confidence metrics",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="symbol",
                        type=ParameterType.STRING,
                        required=True,
                        description="Stock ticker symbol",
                        aliases=["ticker", "stock"]
                    ),
                ],
                parameter_mappings={
                    "symbol": ParameterMapping(source="extracted_entities.tickers[0]")
                },
                alternative_endpoints=[]
            ),

            # Listing & Delisting Status
            EndpointSpec(
                name="LISTING_STATUS",
                description="Regulatory listing roster showing active vs. delisted US equities/ETFs with IPO dates and status change timestamps",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="date",
                        type=ParameterType.DATE,
                        required=False,
                        description="Specific date (YYYY-MM-DD) to query historical listing status since 2010-01-01",
                        aliases=["as_of_date", "on_date"],
                        format="YYYY-MM-DD"
                    ),
                    ParameterSchema(
                        name="state",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Listing state to filter by",
                        valid_values=["active", "delisted"],
                        default_value="active"
                    ),
                ],
                parameter_mappings={
                    "date": ParameterMapping(source="extracted_entities.time_range.from_date")
                },
                alternative_endpoints=[]
            ),

            # Earnings Calendar
            EndpointSpec(
                name="EARNINGS_CALENDAR",
                description="Forward-looking earnings calendar containing announcement dates, estimated EPS, prior-year comparisons, and time-of-day info",
                data_category="stock",
                parameters=[
                    ParameterSchema(
                        name="symbol",
                        type=ParameterType.STRING,
                        required=False,
                        description="Optional stock ticker symbol to filter earnings calendar",
                        aliases=["ticker", "stock"]
                    ),
                    ParameterSchema(
                        name="horizon",
                        type=ParameterType.ENUM,
                        required=False,
                        description="Time horizon to look ahead for earnings events",
                        valid_values=["3month", "6month", "12month"],
                        default_value="3month"
                    ),
                ],
                parameter_mappings={
                    "symbol": ParameterMapping(source="extracted_entities.tickers[0]"),
                    "horizon": ParameterMapping(source="extracted_entities.time_range.duration")
                },
                alternative_endpoints=[]
            ),

            # IPO Calendar
            EndpointSpec(
                name="IPO_CALENDAR",
                description="Pipeline of expected IPOs with filing date, expected pricing window, lead underwriters, and share counts for the next three months",
                data_category="stock",
                parameters=[
                    # no additional parameters required
                ],
                parameter_mappings={},
                alternative_endpoints=[]
            ),
        ]
    )


# ============================================================================
# Registry Class
# ============================================================================

class APIRegistry:
    """Central registry for all API specifications"""
    
    def __init__(self):
        self.apis: Dict[str, APISpec] = {
            "polygon": build_polygon_api(),
            "alpha_vantage": build_alpha_vantage_api(),
        }
        
        # Build endpoint index for fast lookup
        self.endpoint_index: Dict[str, List[str]] = {}  # endpoint_name -> [api_names]
        self._build_endpoint_index()
    
    def _build_endpoint_index(self):
        """Build reverse index: endpoint_name -> list of APIs that support it"""
        for api_name, api_spec in self.apis.items():
            for endpoint in api_spec.endpoints:
                if endpoint.name not in self.endpoint_index:
                    self.endpoint_index[endpoint.name] = []
                self.endpoint_index[endpoint.name].append(api_name)
    
    def get_endpoint_spec(self, api_name: str, endpoint_name: str) -> Optional[EndpointSpec]:
        """Get endpoint specification"""
        api_spec = self.apis.get(api_name)
        if not api_spec:
            return None
        
        for endpoint in api_spec.endpoints:
            if endpoint.name == endpoint_name:
                return endpoint
        return None
    
    def get_all_endpoint_descriptions(self) -> Dict[str, Dict[str, Any]]:
        """Get all endpoint descriptions for semantic matching"""
        descriptions = {}
        for api_name, api_spec in self.apis.items():
            for endpoint in api_spec.endpoints:
                descriptions[endpoint.name] = {
                    "description": endpoint.description,
                    "apis": self.endpoint_index.get(endpoint.name, []),
                    "data_category": endpoint.data_category
                }
        return descriptions
    
    def get_parameter_schema(self, api_name: str, endpoint_name: str, param_name: str) -> Optional[ParameterSchema]:
        """Get parameter schema for validation"""
        endpoint_spec = self.get_endpoint_spec(api_name, endpoint_name)
        if not endpoint_spec:
            return None
        
        for param in endpoint_spec.parameters:
            if param.name == param_name or param_name in param.aliases:
                return param
        return None

    def list_all_endpoints(self) -> List[tuple]:
        """
        List all endpoints as (api_name, endpoint_name, description) tuples.
        Maintained for backward compatibility with EndpointValidator.
        """
        endpoints = []
        for api_name, api_spec in self.apis.items():
            for endpoint in api_spec.endpoints:
                endpoints.append((api_name, endpoint.name, endpoint.description))
        return endpoints

    def get_compact_manifest(self) -> Dict[str, Any]:
        """
        Produce a compact manifest summarizing supported APIs and endpoints for LLM routing.
        The manifest contains only names, descriptions, data categories and a simple
        parameter list indicating required status and allowed values. This reduces
        token usage while still giving the model enough context to select an
        appropriate endpoint and construct a valid request.
        """
        manifest: Dict[str, Any] = {}
        for api_name, api_spec in self.apis.items():
            endpoints_summary: List[Dict[str, Any]] = []
            for endpoint in api_spec.endpoints:
                param_summaries: List[Dict[str, Any]] = []
                for param in endpoint.parameters:
                    param_info: Dict[str, Any] = {
                        "name": param.name,
                        "required": param.required,
                    }
                    # Include allowed values for enums to guide the model
                    if param.valid_values:
                        param_info["valid_values"] = param.valid_values
                    # Provide parameter type to help differentiate date vs numeric vs string
                    param_info["type"] = param.type.value
                    param_summaries.append(param_info)
                endpoints_summary.append({
                    "name": endpoint.name,
                    "description": endpoint.description,
                    "data_category": endpoint.data_category,
                    "parameters": param_summaries,
                })
            manifest[api_name] = {
                "base_url": api_spec.base_url,
                "endpoints": endpoints_summary,
            }
        return manifest


# Global registry instance
registry = APIRegistry()