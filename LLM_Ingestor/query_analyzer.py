"""
Query Analyzer - Fixed to select defaults from BOTH APIs where possible
"""
import json
import time
from datetime import datetime, timedelta
from openai import OpenAI
from contracts import LLMResponse, FeatureSpec, LLMAPIRequest
from api_registry import registry


class QueryAnalyzer:
    """Analyzes queries and selects endpoints from BOTH APIs"""
    
    def __init__(self, api_key: str, model: str = "gpt-4", temperature: float = None):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Default date range (1 month back)
        one_month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        self.default_from_date = one_month_ago
        self.default_to_date = self.current_date
        
        self.registry_manifest = registry.get_compact_manifest()
    
    def analyze(self, prompt: str) -> LLMResponse:
        """Analyze prompt and return endpoint selections"""
        system_prompt = self._build_system_prompt()
        
        try:
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
            
            return self._parse_llm_response(parsed_data)
            
        except Exception as e:
            print(f"Error in Query Analyzer: {e}")
            return self._create_fallback_response(prompt)
    
    def _build_system_prompt(self) -> str:
        manifest_json = json.dumps(self.registry_manifest, indent=2)
        
        return f"""You are a financial data API router. Your job is to:
1. Determine if the query is finance-related (proceed flag)
2. Extract features and semantic keywords
3. Select the best endpoint from EACH API that supports the query
4. Propose parameters for each endpoint

AVAILABLE APIs:
```json
{manifest_json}
```

OUTPUT SCHEMA:
{{
  "proceed": true/false,
  "features": {{
    "native": ["open", "high", "low", "close", "volume"],
    "enrichment": ["SMA_20", "RSI_14"]
  }},
  "semantic_keywords": ["daily", "stock", "price", "historical"],
  "tickers": ["AAPL"],
  "api_requests": [
    {{
      "api_name": "polygon" | "alpha_vantage",
      "endpoint_name": "endpoint name",
      "parameters": {{}},
      "reasoning": "why"
    }}
  ]
}}

CRITICAL DEFAULT RULES:

1. **NON-FINANCE QUERIES**: Set proceed=false for queries completely unrelated to finance.

2. **VAGUE STOCK QUERIES (no ticker)**:
   Example: "I want stock data", "show me stocks"
   → Polygon: FULL_MARKET_SNAPSHOT
   → Alpha Vantage: Skip (requires ticker)
   
3. **VAGUE STOCK QUERIES (with ticker)**:
   Example: "Show me AAPL", "IBM data"
   → Polygon: get_aggs (1-month daily)
   → Alpha Vantage: TIME_SERIES_DAILY_ADJUSTED
   **MUST SELECT FROM BOTH APIs**

4. **VAGUE ECONOMIC QUERIES**:
   Example: "economic indicators", "show me economic data"
   → Polygon: INFLATION
   → Alpha Vantage: CPI
   **MUST SELECT FROM BOTH APIs**

5. **SPECIFIC QUERIES**: Honor user preferences, but still try to select from both APIs when both support it.

6. **DEFAULT TIMEFRAME**: 1 month (30 days), daily granularity
   - from: {self.default_from_date}
   - to: {self.default_to_date}

EXAMPLES:

Example 1 - Non-finance:
User: "What's the weather?"
{{
  "proceed": false,
  "features": {{"native": [], "enrichment": []}},
  "semantic_keywords": [],
  "tickers": [],
  "api_requests": []
}}

Example 2 - Vague stock (no ticker):
User: "I want stock data"
{{
  "proceed": true,
  "features": {{"native": ["open", "high", "low", "close", "volume"], "enrichment": []}},
  "semantic_keywords": ["stock", "data", "market"],
  "tickers": [],
  "api_requests": [
    {{
      "api_name": "polygon",
      "endpoint_name": "FULL_MARKET_SNAPSHOT",
      "parameters": {{}},
      "reasoning": "No ticker - use market snapshot"
    }}
  ]
}}

Example 3 - Vague stock (with ticker) - MUST SELECT BOTH:
User: "Show me AAPL"
{{
  "proceed": true,
  "features": {{"native": ["open", "high", "low", "close", "volume"], "enrichment": []}},
  "semantic_keywords": ["stock", "data", "daily", "price"],
  "tickers": ["AAPL"],
  "api_requests": [
    {{
      "api_name": "polygon",
      "endpoint_name": "get_aggs",
      "parameters": {{"ticker": "AAPL", "multiplier": 1, "timespan": "day", "from": "{self.default_from_date}", "to": "{self.default_to_date}"}},
      "reasoning": "Polygon daily aggregates"
    }},
    {{
      "api_name": "alpha_vantage",
      "endpoint_name": "TIME_SERIES_DAILY_ADJUSTED",
      "parameters": {{"ticker": "AAPL", "outputsize": "full"}},
      "reasoning": "Alpha Vantage daily adjusted"
    }}
  ]
}}

Example 4 - Vague economic - MUST SELECT BOTH:
User: "Show me economic indicators"
{{
  "proceed": true,
  "features": {{"native": ["value"], "enrichment": []}},
  "semantic_keywords": ["economic", "indicator", "data"],
  "tickers": [],
  "api_requests": [
    {{
      "api_name": "polygon",
      "endpoint_name": "INFLATION",
      "parameters": {{"limit": 100}},
      "reasoning": "Polygon inflation data"
    }},
    {{
      "api_name": "alpha_vantage",
      "endpoint_name": "CPI",
      "parameters": {{"interval": "monthly"}},
      "reasoning": "Alpha Vantage CPI data"
    }}
  ]
}}

Example 5 - Specific GDP query:
User: "US GDP data for 5 years"
{{
  "proceed": true,
  "features": {{"native": ["value"], "enrichment": []}},
  "semantic_keywords": ["GDP", "economic", "growth", "quarterly"],
  "tickers": [],
  "api_requests": [
    {{
      "api_name": "alpha_vantage",
      "endpoint_name": "REAL_GDP",
      "parameters": {{"interval": "quarterly"}},
      "reasoning": "GDP only available on Alpha Vantage"
    }}
  ]
}}

Example 6 - Intraday:
User: "TSLA 5-minute data today"
{{
  "proceed": true,
  "features": {{"native": ["open", "high", "low", "close", "volume"], "enrichment": []}},
  "semantic_keywords": ["intraday", "5-minute", "stock", "today"],
  "tickers": ["TSLA"],
  "api_requests": [
    {{
      "api_name": "polygon",
      "endpoint_name": "get_aggs",
      "parameters": {{"ticker": "TSLA", "multiplier": 5, "timespan": "minute", "from": "{self.current_date}", "to": "{self.current_date}"}},
      "reasoning": "Polygon 5-minute aggregates"
    }},
    {{
      "api_name": "alpha_vantage",
      "endpoint_name": "TIME_SERIES_INTRADAY",
      "parameters": {{"ticker": "TSLA", "interval": "5min", "outputsize": "compact"}},
      "reasoning": "Alpha Vantage 5min intraday"
    }}
  ]
}}

IMPORTANT:
- Always select from BOTH APIs when both support the query type
- Include semantic_keywords that capture query intent
- Economic indicators don't have tickers
- Default to 1-month daily when no timeframe specified
"""
    
    def _parse_llm_response(self, data: dict) -> LLMResponse:
        features = FeatureSpec(
            native=data.get("features", {}).get("native", ["open", "high", "low", "close", "volume"]),
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
        return LLMResponse(
            proceed=True,
            features=FeatureSpec(
                native=["open", "high", "low", "close", "volume"],
                enrichment=[]
            ),
            semantic_keywords=[],
            api_requests=[],
            tickers=[]
        )