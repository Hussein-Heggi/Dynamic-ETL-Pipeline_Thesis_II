"""
Query Analyzer - LLM-powered endpoint selection and feature extraction
UPDATED: Removed native features, only enrichment and semantic_keywords
"""
import json
import time
from datetime import datetime, timedelta
from openai import OpenAI
from contracts import LLMResponse, FeatureSpec, LLMAPIRequest
from api_registry import registry


class QueryAnalyzer:
    """
    Analyzes user queries using LLM to:
    1. Determine if query is finance-related
    2. Extract enrichment features and semantic keywords
    3. Select best endpoint for EACH API that supports the query
    4. Propose parameters for each selected endpoint
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4", temperature: float = None):
        """
        Initialize Query Analyzer
        
        Args:
            api_key: OpenAI API key
            model: Model to use
            temperature: Temperature (None=default)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Default date range (1 month back)
        one_month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        self.default_from_date = one_month_ago
        self.default_to_date = self.current_date
        
        # Get compact manifest from registry
        self.registry_manifest = registry.get_compact_manifest()
    
    def analyze(self, prompt: str) -> LLMResponse:
        """
        Analyze user prompt and return LLM response with features + endpoint selections
        
        Args:
            prompt: Natural language user query
            
        Returns:
            LLMResponse with features and api_requests
        """
        system_prompt = self._build_system_prompt()
        
        try:
            # Build API call
            api_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"}
            }
            
            if self.temperature is not None:
                api_params["temperature"] = self.temperature
            
            response = self.client.chat.completions.create(**api_params)
            response_text = response.choices[0].message.content
            parsed_data = json.loads(response_text)
            
            # Convert to LLMResponse
            llm_response = self._parse_llm_response(parsed_data)
            
            return llm_response
            
        except Exception as e:
            print(f"Error in Query Analyzer: {e}")
            return self._create_fallback_response(prompt)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with registry manifest"""
        manifest_json = json.dumps(self.registry_manifest, indent=2)
        
        return f"""You are a financial data API router. Your job is to:
1. Determine if the query is finance-related (proceed flag)
2. Extract enrichment features and semantic keywords
3. Select the best endpoint from EACH API that supports the query
4. Propose parameters for each endpoint

AVAILABLE APIs:
```json
{manifest_json}
```

========================================
OUTPUT SCHEMA
========================================

You must always return a SINGLE JSON object with this shape:

{{
  "proceed": true/false,
  "features": {{
    "enrichment": ["SMA_20", "RSI_14"]
  }},
  "semantic_keywords": ["daily", "stock", "price", "historical"],
  "tickers": ["AAPL"],
  "api_requests": [
    {{
      "api_name": "polygon" | "alpha_vantage",
      "endpoint_name": "endpoint name from manifest",
      "parameters": {{ ... }},
      "reasoning": "short explanation of why this endpoint and parameters were chosen"
    }}
  ]
}}

Field definitions:

- proceed: true if finance-related, false otherwise (everything else must be empty/[])
- features.enrichment: List of derived/technical features (e.g., ["SMA_20", "RSI_14", "bollinger_bands_20_2"]). MUST NOT contain raw fields like open, high, low, close, volume. If none requested → []
- semantic_keywords: Short list of tags describing query intent (3-7 keywords)
- tickers: List of ticker symbols. May be inferred from company names. [] if none applicable
- api_requests: List of concrete API calls with api_name, endpoint_name, parameters, reasoning

========================================
GLOBAL CONSTRAINTS
========================================

1. MANIFEST IS CANONICAL
   - Do NOT invent new endpoint names, parameter names, or structures
   - Always use names exactly as they appear in manifest
   - If no endpoint perfectly matches, choose closest valid one and explain in reasoning

2. NON-FINANCE QUERIES
   - Set proceed=false for queries unrelated to finance (weather, recipes, etc.)
   - All other fields must be empty/[]

3. TICKER EXTRACTION RULES
   - Two ways to identify tickers:
     a) Explicit symbol (e.g., "AAPL", "MSFT", "BRK.B")
     b) Company name inference (e.g., "Microsoft" → "MSFT", "Apple" → "AAPL", "Tesla" → "TSLA")
   - Do NOT treat macro abbreviations as tickers: GDP, CPI, PCE, FED, FOMC, PMI, UNRATE, INFLATION, RECESSION

4. MULTI-TICKER HANDLING
   - Extract ALL mentioned tickers (no limit)
   - Generate separate api_requests for each ticker
   - If comparison requested, include "comparison" in semantic_keywords

5. SEMANTIC KEYWORDS VOCABULARY
   - Time granularity: ["intraday", "daily", "weekly", "monthly", "quarterly", "yearly"]
   - Data type: ["stock", "fundamentals", "economic", "macro", "etf"]
   - Purpose: ["history", "realtime", "snapshot", "comparison"]
   - Metrics: ["price", "volume", "returns", "volatility"]
   - Statements: ["income_statement", "cash_flow", "balance_sheet", "dividends"]
   - Indicators: ["cpi", "gdp", "unemployment", "pce", etc.]

6. ENRICHMENT FEATURES
   - ONLY derived/technical metrics: SMA_20, EMA_50, RSI_14, bollinger_bands_20_2, log_returns, realized_volatility_30d
   - Do NOT include: open, high, low, close, volume
   - If none requested → []

7. DEFAULT TIMEFRAME
   - Use 1-month (30-day) default if no timeframe specified:
     - from: {self.default_from_date}
     - to: {self.default_to_date}
   - Honor user-specified dates/periods

========================================
CATEGORY-SPECIFIC ROUTING RULES
========================================

4.1 STOCK PRICE / TIME-SERIES

4.1.a Vague stock queries – NO ticker
Examples: "I want stock data", "show me stocks", "stock market overview"

Behavior:
- tickers: []
- Polygon: Use FULL_MARKET_SNAPSHOT if available
- Alpha Vantage: Use TIME_SERIES_DAILY_ADJUSTED for default basket:
  ["AAPL", "AMZN", "GOOGL", "MSFT", "META", "TSLA", "NFLX", "UBER"]
  One request per ticker with default 1-month window

4.1.b Vague stock queries – WITH ticker (MUST USE BOTH APIS)
Examples: "Show me AAPL", "IBM data", "TSLA stock price"

Behavior:
- tickers: include all mentioned (symbols or inferred from names)
- Polygon: get_aggs with multiplier=1, timespan="day", default dates
- Alpha Vantage: TIME_SERIES_DAILY_ADJUSTED
- ALWAYS use BOTH APIs

4.1.c Intraday stock queries
Examples: "TSLA 5-minute data today", "AAPL intraday price"

Behavior:
- Polygon: get_aggs with timespan="minute", appropriate multiplier
- Alpha Vantage: TIME_SERIES_INTRADAY with matching interval
- Use BOTH APIs

4.2 ECONOMIC / MACRO DATA

4.2.a Vague economic queries
Examples: "economic indicators", "macro data", "economic situation"

Behavior:
- tickers: []
- Polygon: INFLATION endpoint
- Alpha Vantage: CPI (optionally add UNEMPLOYMENT, FEDERAL_FUNDS_RATE if available)
- Keep to 1-3 endpoints per API

4.2.b Specific economic indicator queries
Examples: "US GDP data for 5 years", "monthly CPI for 10 years"

Behavior:
- Focus on named indicator (GDP, CPI, unemployment, etc.)
- tickers: []
- Use parameters matching requested timeframe

4.3 FUNDAMENTALS / FINANCIAL STATEMENTS

- Alpha Vantage: Has fundamental endpoints (when available in manifest)
- Polygon: Use ONLY for price context via get_aggs (NOT for financial statements)

4.3.a Vague fundamentals WITH ticker
Examples: "IBM fundamentals", "fundamentals for AAPL"

Behavior:
- Alpha Vantage: Return fundamental endpoints if available
- Polygon: Add get_aggs (1-month daily) for price context

4.3.b Vague fundamentals WITHOUT ticker
Examples: "Show me some company fundamentals"

Behavior:
- tickers: []
- Use default basket: ["AAPL", "IBM", "AMZN"]
- For EACH: fundamental endpoints + Polygon get_aggs

4.3.c Specific fundamental statement queries
Examples: "IBM income statement", "cash flow for AAPL"

Behavior:
- Alpha Vantage: Use specific statement endpoint only
- Polygon: Optionally add get_aggs for price context

========================================
API_REQUESTS GENERATION RULES
========================================

- For each ticker × each relevant API → create api_requests entries
- No duplicates (same api_name, endpoint_name, identical parameters)
- Ordering:
  1) Polygon stock/time-series
  2) Alpha Vantage stock/time-series
  3) Polygon economic
  4) Alpha Vantage economic
  5) Alpha Vantage fundamentals
- Each entry must have concise "reasoning" explaining choices
- User instructions override defaults

========================================
EXAMPLES
========================================

Example 1 – Non-finance:
User: "What's the weather today?"
{{
  "proceed": false,
  "features": {{ "enrichment": [] }},
  "semantic_keywords": [],
  "tickers": [],
  "api_requests": []
}}

Example 2 – Vague stock (no ticker):
User: "I want stock data"
{{
  "proceed": true,
  "features": {{ "enrichment": [] }},
  "semantic_keywords": ["stock", "market", "snapshot", "daily"],
  "tickers": [],
  "api_requests": [
    {{
      "api_name": "polygon",
      "endpoint_name": "get_grouped_daily_aggs",
      "parameters": {{ "date": "{self.default_to_date}" }},
      "reasoning": "No specific ticker, use Polygon grouped daily for market snapshot"
    }},
    {{
      "api_name": "alpha_vantage",
      "endpoint_name": "TIME_SERIES_DAILY_ADJUSTED",
      "parameters": {{ "symbol": "AAPL", "outputsize": "full" }},
      "reasoning": "Default basket ticker AAPL daily prices"
    }},
    {{
      "api_name": "alpha_vantage",
      "endpoint_name": "TIME_SERIES_DAILY_ADJUSTED",
      "parameters": {{ "symbol": "AMZN", "outputsize": "full" }},
      "reasoning": "Default basket ticker AMZN daily prices"
    }}
  ]
}}

Example 3 – Vague stock with ticker (MUST USE BOTH APIS):
User: "Show me AAPL"
{{
  "proceed": true,
  "features": {{ "enrichment": [] }},
  "semantic_keywords": ["stock", "daily", "price", "history"],
  "tickers": ["AAPL"],
  "api_requests": [
    {{
      "api_name": "polygon",
      "endpoint_name": "get_aggs",
      "parameters": {{
        "ticker": "AAPL",
        "multiplier": 1,
        "timespan": "day",
        "from": "{self.default_from_date}",
        "to": "{self.default_to_date}"
      }},
      "reasoning": "Polygon daily aggregates for AAPL over default 1-month window"
    }},
    {{
      "api_name": "alpha_vantage",
      "endpoint_name": "TIME_SERIES_DAILY_ADJUSTED",
      "parameters": {{ "symbol": "AAPL", "outputsize": "full" }},
      "reasoning": "Alpha Vantage daily prices for AAPL"
    }}
  ]
}}

Example 4 – Company name inference:
User: "Get Microsoft stock"
{{
  "proceed": true,
  "features": {{ "enrichment": [] }},
  "semantic_keywords": ["stock", "daily", "price", "history"],
  "tickers": ["MSFT"],
  "api_requests": [
    {{
      "api_name": "polygon",
      "endpoint_name": "get_aggs",
      "parameters": {{
        "ticker": "MSFT",
        "multiplier": 1,
        "timespan": "day",
        "from": "{self.default_from_date}",
        "to": "{self.default_to_date}"
      }},
      "reasoning": "Polygon daily aggregates for MSFT (inferred from Microsoft)"
    }},
    {{
      "api_name": "alpha_vantage",
      "endpoint_name": "TIME_SERIES_DAILY_ADJUSTED",
      "parameters": {{ "symbol": "MSFT", "outputsize": "full" }},
      "reasoning": "Alpha Vantage daily for MSFT"
    }}
  ]
}}

Example 5 – Intraday:
User: "TSLA 5-minute data today"
{{
  "proceed": true,
  "features": {{ "enrichment": [] }},
  "semantic_keywords": ["intraday", "stock", "price", "today"],
  "tickers": ["TSLA"],
  "api_requests": [
    {{
      "api_name": "polygon",
      "endpoint_name": "get_aggs",
      "parameters": {{
        "ticker": "TSLA",
        "multiplier": 5,
        "timespan": "minute",
        "from": "{self.current_date}",
        "to": "{self.current_date}"
      }},
      "reasoning": "Polygon 5-minute intraday aggregates for TSLA today"
    }},
    {{
      "api_name": "alpha_vantage",
      "endpoint_name": "TIME_SERIES_INTRADAY",
      "parameters": {{
        "symbol": "TSLA",
        "interval": "5min",
        "outputsize": "compact"
      }},
      "reasoning": "Alpha Vantage 5-minute intraday prices for TSLA"
    }}
  ]
}}

Example 6 – With technical indicators:
User: "Tesla daily with 20-day and 50-day SMA"
{{
  "proceed": true,
  "features": {{ "enrichment": ["SMA_20", "SMA_50"] }},
  "semantic_keywords": ["stock", "daily", "price", "history"],
  "tickers": ["TSLA"],
  "api_requests": [
    {{
      "api_name": "polygon",
      "endpoint_name": "get_aggs",
      "parameters": {{
        "ticker": "TSLA",
        "multiplier": 1,
        "timespan": "day",
        "from": "{self.default_from_date}",
        "to": "{self.default_to_date}"
      }},
      "reasoning": "Get price data for SMA calculation"
    }},
    {{
      "api_name": "alpha_vantage",
      "endpoint_name": "TIME_SERIES_DAILY_ADJUSTED",
      "parameters": {{ "symbol": "TSLA", "outputsize": "full" }},
      "reasoning": "Get price data for SMA calculation"
    }}
  ]
}}

Example 7 – Vague economic:
User: "Show me economic indicators"
{{
  "proceed": true,
  "features": {{ "enrichment": [] }},
  "semantic_keywords": ["economic", "macro", "cpi", "inflation", "monthly"],
  "tickers": [],
  "api_requests": [
    {{
      "api_name": "polygon",
      "endpoint_name": "INFLATION",
      "parameters": {{ "limit": 100 }},
      "reasoning": "Polygon inflation endpoint for general macro data"
    }},
    {{
      "api_name": "alpha_vantage",
      "endpoint_name": "CPI",
      "parameters": {{}},
      "reasoning": "Alpha Vantage CPI monthly data as core macro indicator"
    }}
  ]
}}

Example 8 – Specific economic (GDP):
User: "US GDP data for 5 years"
{{
  "proceed": true,
  "features": {{ "enrichment": [] }},
  "semantic_keywords": ["economic", "gdp", "quarterly", "history"],
  "tickers": [],
  "api_requests": [
    {{
      "api_name": "alpha_vantage",
      "endpoint_name": "REAL_GDP",
      "parameters": {{ "interval": "quarterly" }},
      "reasoning": "Alpha Vantage REAL_GDP provides US GDP data"
    }}
  ]
}}

Example 9 – Multiple tickers:
User: "Compare AAPL and MSFT daily data"
{{
  "proceed": true,
  "features": {{ "enrichment": [] }},
  "semantic_keywords": ["stock", "daily", "price", "comparison"],
  "tickers": ["AAPL", "MSFT"],
  "api_requests": [
    {{
      "api_name": "polygon",
      "endpoint_name": "get_aggs",
      "parameters": {{ "ticker": "AAPL", "multiplier": 1, "timespan": "day", "from": "{self.default_from_date}", "to": "{self.default_to_date}" }},
      "reasoning": "Polygon daily for AAPL"
    }},
    {{
      "api_name": "polygon",
      "endpoint_name": "get_aggs",
      "parameters": {{ "ticker": "MSFT", "multiplier": 1, "timespan": "day", "from": "{self.default_from_date}", "to": "{self.default_to_date}" }},
      "reasoning": "Polygon daily for MSFT"
    }},
    {{
      "api_name": "alpha_vantage",
      "endpoint_name": "TIME_SERIES_DAILY_ADJUSTED",
      "parameters": {{ "symbol": "AAPL", "outputsize": "full" }},
      "reasoning": "Alpha Vantage daily for AAPL"
    }},
    {{
      "api_name": "alpha_vantage",
      "endpoint_name": "TIME_SERIES_DAILY_ADJUSTED",
      "parameters": {{ "symbol": "MSFT", "outputsize": "full" }},
      "reasoning": "Alpha Vantage daily for MSFT"
    }}
  ]
}}

IMPORTANT:
- Always output valid JSON
- Only use endpoints from the manifest
- Generate requests for ALL APIs that support the query type
- For multi-ticker queries, create separate requests per ticker
- Include parameters in enrichment feature names (SMA_20, not SMA)
- Infer tickers from company names when applicable
- Always use BOTH APIs for stock time-series when both support it
"""
    
    def _parse_llm_response(self, data: dict) -> LLMResponse:
        """Parse LLM JSON response into LLMResponse object"""
        features = FeatureSpec(
            enrichment=data.get("features", {}).get("enrichment", [])
        )
        
        api_requests = [
            LLMAPIRequest(
                api_name=req["api_name"],
                endpoint_name=req["endpoint_name"],
                parameters=req["parameters"],
                reasoning=req.get("reasoning")
            )
            for req in data.get("api_requests", [])
        ]
        
        return LLMResponse(
            proceed=data.get("proceed", True),
            features=features,
            semantic_keywords=data.get("semantic_keywords", []),
            api_requests=api_requests,
            tickers=data.get("tickers", [])
        )
    
    def _create_fallback_response(self, prompt: str) -> LLMResponse:
        """Create minimal response when LLM fails"""
        return LLMResponse(
            proceed=True,
            features=FeatureSpec(
                enrichment=[]
            ),
            semantic_keywords=[],
            api_requests=[],
            tickers=[]
        )