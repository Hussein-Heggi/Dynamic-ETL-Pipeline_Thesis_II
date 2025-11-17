"""
Query Analyzer - LLM-powered endpoint selection and feature extraction
MAJOR UPDATE: LLM now selects specific endpoints per API instead of just extracting entities
"""
import json
import time
from datetime import datetime, timedelta
from typing import Optional
from openai import OpenAI
from contracts import LLMResponse, FeatureSpec, LLMAPIRequest
from api_registry import registry


class QueryAnalyzer:
    """
    Analyzes user queries using LLM to:
    1. Extract native and enrichment features
    2. Select best endpoint for EACH API that supports the query
    3. Propose parameters for each selected endpoint
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
        start_time = time.time()
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
            
            if self.temperature is not None and "nano" not in self.model:
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
1. Extract features the user wants (native API features + enrichment features)
2. Select the best endpoint from EACH API that can fulfill the request
3. Propose parameters for each endpoint

AVAILABLE APIs AND ENDPOINTS:
```json
{manifest_json}
```

OUTPUT SCHEMA:
{{
  "features": {{
    "native": ["list of API-provided features like open, close, volume, value, rate"],
    "enrichment": ["list of calculated features with parameters like SMA_20, RSI_14, MACD_12_26_9"]
  }},
  "tickers": ["list of ticker symbols if applicable"],
  "api_requests": [
    {{
      "api_name": "polygon" | "alpha_vantage",
      "endpoint_name": "exact endpoint name from registry",
      "parameters": {{"param1": "value1", "param2": "value2"}},
      "reasoning": "why you chose this endpoint"
    }}
  ]
}}

CRITICAL RULES:

1. **ENDPOINT SELECTION**:
   - For EACH API, select its best endpoint OR skip if unsupported
   - Only use endpoint names that EXACTLY match the registry
   - If both APIs support the query, include requests for BOTH
   - Example: "AAPL daily data" â†’ Both polygon.get_aggs AND alpha_vantage.TIME_SERIES_DAILY

2. **FEATURES**:
   - Native features: open, high, low, close, volume (stocks) OR value, rate (economic)
   - Enrichment features: MUST include parameters (SMA_20, not SMA)
   - Default stock features: ["open", "high", "low", "close", "volume"]

3. **PARAMETERS**:
   - Use parameter names from registry's parameter definitions
   - Respect parameter types and valid_values
   - Fill required parameters
   - Use defaults for optional parameters

4. **MULTI-TICKER SUPPORT**:
   - Generate separate requests for each ticker
   - Example: "AAPL and MSFT" â†’ 2 requests for polygon, 2 for alpha_vantage

5. **TIME RANGES**:
   - Today's date: {self.current_date}
   - Convert relative dates ("last 30 days") to absolute dates
   - Default stock queries: last 30 days
   - Default economic queries: last 5 years

6. **ECONOMIC INDICATORS**:
   - Polygon: Only INFLATION supported
   - Alpha Vantage: REAL_GDP, CPI, INFLATION, UNEMPLOYMENT, TREASURY_YIELD, etc.
   - Economic queries DON'T have tickers
   - Economic native features: ["value"] or ["rate"]

EXAMPLES:

Example 1 - Stock data (both APIs):
User: "Get AAPL daily prices for last month"
Output:
{{
  "features": {{
    "native": ["open", "high", "low", "close", "volume"],
    "enrichment": []
  }},
  "tickers": ["AAPL"],
  "api_requests": [
    {{
      "api_name": "polygon",
      "endpoint_name": "get_aggs",
      "parameters": {{
        "ticker": "AAPL",
        "multiplier": 1,
        "timespan": "day",
        "from": "2025-10-17",
        "to": "2025-11-16"
      }},
      "reasoning": "Polygon supports daily aggregates for stocks"
    }},
    {{
      "api_name": "alpha_vantage",
      "endpoint_name": "TIME_SERIES_DAILY",
      "parameters": {{
        "symbol": "AAPL",
        "outputsize": "full"
      }},
      "reasoning": "Alpha Vantage supports daily time series for stocks"
    }}
  ]
}}

Example 2 - Multiple tickers:
User: "Compare AAPL and MSFT daily data"
Output:
{{
  "features": {{
    "native": ["open", "high", "low", "close", "volume"],
    "enrichment": []
  }},
  "tickers": ["AAPL", "MSFT"],
  "api_requests": [
    {{
      "api_name": "polygon",
      "endpoint_name": "get_aggs",
      "parameters": {{"ticker": "AAPL", "multiplier": 1, "timespan": "day", "from": "2025-10-17", "to": "2025-11-16"}},
      "reasoning": "Polygon for AAPL"
    }},
    {{
      "api_name": "polygon",
      "endpoint_name": "get_aggs",
      "parameters": {{"ticker": "MSFT", "multiplier": 1, "timespan": "day", "from": "2025-10-17", "to": "2025-11-16"}},
      "reasoning": "Polygon for MSFT"
    }},
    {{
      "api_name": "alpha_vantage",
      "endpoint_name": "TIME_SERIES_DAILY",
      "parameters": {{"symbol": "AAPL", "outputsize": "full"}},
      "reasoning": "Alpha Vantage for AAPL"
    }},
    {{
      "api_name": "alpha_vantage",
      "endpoint_name": "TIME_SERIES_DAILY",
      "parameters": {{"symbol": "MSFT", "outputsize": "full"}},
      "reasoning": "Alpha Vantage for MSFT"
    }}
  ]
}}

Example 3 - With technical indicators:
User: "Tesla daily with 20-day and 50-day SMA"
Output:
{{
  "features": {{
    "native": ["open", "high", "low", "close", "volume"],
    "enrichment": ["SMA_20", "SMA_50"]
  }},
  "tickers": ["TSLA"],
  "api_requests": [
    {{
      "api_name": "polygon",
      "endpoint_name": "get_aggs",
      "parameters": {{"ticker": "TSLA", "multiplier": 1, "timespan": "day", "from": "2025-10-17", "to": "2025-11-16"}},
      "reasoning": "Get price data for SMA calculation"
    }},
    {{
      "api_name": "alpha_vantage",
      "endpoint_name": "TIME_SERIES_DAILY",
      "parameters": {{"symbol": "TSLA", "outputsize": "full"}},
      "reasoning": "Get price data for SMA calculation"
    }}
  ]
}}

Example 4 - Economic indicator:
User: "Get US GDP data for last 5 years"
Output:
{{
  "features": {{
    "native": ["value"],
    "enrichment": []
  }},
  "tickers": [],
  "api_requests": [
    {{
      "api_name": "alpha_vantage",
      "endpoint_name": "REAL_GDP",
      "parameters": {{"interval": "quarterly"}},
      "reasoning": "Alpha Vantage supports REAL_GDP economic indicator"
    }}
  ]
}}

Example 5 - Intraday:
User: "NVDA 5-minute data today"
Output:
{{
  "features": {{
    "native": ["open", "high", "low", "close", "volume"],
    "enrichment": []
  }},
  "tickers": ["NVDA"],
  "api_requests": [
    {{
      "api_name": "polygon",
      "endpoint_name": "get_aggs",
      "parameters": {{"ticker": "NVDA", "multiplier": 5, "timespan": "minute", "from": "{self.current_date}", "to": "{self.current_date}"}},
      "reasoning": "Polygon supports intraday minute-level aggregates"
    }},
    {{
      "api_name": "alpha_vantage",
      "endpoint_name": "TIME_SERIES_INTRADAY",
      "parameters": {{"symbol": "NVDA", "interval": "5min", "outputsize": "compact"}},
      "reasoning": "Alpha Vantage supports 5min intraday data"
    }}
  ]
}}

IMPORTANT:
- Always output valid JSON
- Only use endpoints from the registry
- Generate requests for ALL APIs that support the query
- For multi-ticker queries, create separate requests per ticker
- Include parameters in enrichment feature names (SMA_20, not SMA)
"""
    
    def _parse_llm_response(self, data: dict) -> LLMResponse:
        """Parse LLM JSON response into LLMResponse object"""
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
            features=features,
            api_requests=api_requests,
            tickers=data.get("tickers", [])
        )
    
    def _create_fallback_response(self, prompt: str) -> LLMResponse:
        """Create minimal response when LLM fails"""
        return LLMResponse(
            features=FeatureSpec(
                native=["open", "high", "low", "close", "volume"],
                enrichment=[]
            ),
            api_requests=[],
            tickers=[]
        )