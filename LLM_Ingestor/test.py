"""
llm_routing_test_harness.py

Drop this file into your repo (same directory as query_analyzer.py).

Usage:
    python llm_routing_test_harness.py --api-key YOUR_OPENAI_KEY --model gpt-5-nano

It will:
  - Instantiate QueryAnalyzer
  - Run routing_test_cases + spec_test_cases
  - Write results to llm_routing_test_results.csv
"""

import argparse
import csv
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Import your components
from query_analyzer import QueryAnalyzer


# ============================================================
# Helpers to interpret LLMResponse in a generic way
# ============================================================

@dataclass
class APIRequestView:
    api_name: str
    endpoint_name: str
    parameters: Dict[str, Any]
    reasoning: str


@dataclass
class LLMResponseView:
    proceed: bool
    native: List[str]
    enrichment: List[str]
    semantic_keywords: List[str]
    tickers: List[str]
    api_requests: List[APIRequestView]
    raw: Any = field(default=None)


def _to_view(resp: Any) -> LLMResponseView:
    """
    Convert your LLMResponse Pydantic model into a simple dataclass
    so the test harness is decoupled from your internal schema.
    """
    proceed = getattr(resp, "proceed", True)
    features = getattr(resp, "features", None)
    native = getattr(features, "native", []) if features else []
    enrichment = getattr(features, "enrichment", []) if features else []
    semantic_keywords = getattr(resp, "semantic_keywords", []) or []
    tickers = getattr(resp, "tickers", []) or []

    api_requests_raw = getattr(resp, "api_requests", []) or []
    api_requests = []
    for r in api_requests_raw:
        api_requests.append(
            APIRequestView(
                api_name=getattr(r, "api_name", ""),
                endpoint_name=getattr(r, "endpoint_name", ""),
                parameters=getattr(r, "parameters", {}) or {},
                reasoning=getattr(r, "reasoning", "") or "",
            )
        )

    return LLMResponseView(
        proceed=proceed,
        native=native,
        enrichment=enrichment,
        semantic_keywords=semantic_keywords,
        tickers=tickers,
        api_requests=api_requests,
        raw=resp,
    )


# ============================================================
# Test case definitions (routing + spec/rules)
# ============================================================

# 1) ROUTING TESTS (endpoint selection & multi-endpoint behavior)
routing_test_cases: List[Dict[str, Any]] = [
    # ----- Core stock time-series (both APIs) -----
    {
        "tc_id": "ROUTE_S1_DAILY_30D",
        "category": "stock_core",
        "prompt": "I need a full daily OHLCV dataset for Apple for the last 30 calendar days for backtesting.",
        "expected_endpoints": [
            "polygon.get_aggs",
            "alpha_vantage.TIME_SERIES_DAILY_ADJUSTED",
        ],
        "notes": "Daily OHLCV last 30 days from both providers",
        "checks": {
            "proceed": True,
            "tickers_include": ["AAPL"],
            "semantic_keywords_include": ["stock", "daily", "price"],
            "must_have_endpoints": [
                "polygon.get_aggs",
                "alpha_vantage.TIME_SERIES_DAILY_ADJUSTED",
            ],
            "api_requests_min": 2,
        },
    },
    {
        "tc_id": "ROUTE_S2_INTRADAY_5MIN",
        "category": "stock_intraday",
        "prompt": "Give me 5-minute intraday price bars for NVDA for the most recent trading day, as a dataset.",
        "expected_endpoints": [
            "polygon.get_aggs",
            "alpha_vantage.TIME_SERIES_INTRADAY",
        ],
        "notes": "Intraday 5min, both APIs",
        "checks": {
            "proceed": True,
            "tickers_include": ["NVDA"],
            "semantic_keywords_include": ["intraday", "stock", "price"],
            "must_have_endpoints": [
                "polygon.get_aggs",
                "alpha_vantage.TIME_SERIES_INTRADAY",
            ],
            "api_requests_min": 2,
        },
    },
    {
        "tc_id": "ROUTE_S3_WEEKLY_5Y",
        "category": "stock_core",
        "prompt": "Return a weekly price history dataset for Microsoft covering roughly the past five years.",
        "expected_endpoints": [
            "polygon.get_aggs",
            "alpha_vantage.TIME_SERIES_WEEKLY",
        ],
        "notes": "Weekly SP, 5 year period (params logic tested separately).",
        "checks": {
            "proceed": True,
            "tickers_include": ["MSFT"],
            "semantic_keywords_include": ["stock", "weekly", "price"],
            "must_have_endpoints": [
                "polygon.get_aggs",
                "alpha_vantage.TIME_SERIES_WEEKLY",
            ],
        },
    },
    {
        "tc_id": "ROUTE_M1_INFLATION_VS_STOCK",
        "category": "mixed_macro_stock",
        "prompt": "Compare US inflation with AAPL's monthly adjusted returns over the last 10 years. I want datasets for both.",
        "expected_endpoints": [
            "polygon.INFLATION",
            "polygon.get_aggs",
            "alpha_vantage.INFLATION",
            "alpha_vantage.TIME_SERIES_MONTHLY",
        ],
        "notes": "Stock + macro from both APIs where possible.",
        "checks": {
            "proceed": True,
            "tickers_include": ["AAPL"],
            "semantic_keywords_include": ["stock", "monthly", "inflation", "economic"],
            "must_have_endpoints": [
                "polygon.INFLATION",
                "polygon.get_aggs",
                "alpha_vantage.INFLATION",
                "alpha_vantage.TIME_SERIES_MONTHLY",
            ],
        },
    },
    {
        "tc_id": "ROUTE_A2_EVERYTHING_FOR_AAPL",
        "category": "adversarial_big_pack",
        "prompt": "For AAPL, give me everything useful: price history, key fundamentals, earnings history and estimates, and any upcoming events.",
        "expected_endpoints": [],
        "notes": "Stress test: model should pick a rich set of endpoints, not just one.",
        "checks": {
            "proceed": True,
            "tickers_include": ["AAPL"],
            "semantic_keywords_include": ["stock", "fundamentals"],
            "api_requests_min": 5,
        },
    },
]


# 2) SPEC TESTS (schema + rules + defaults + fallbacks)
spec_test_cases: List[Dict[str, Any]] = [
    # ============ 1. Schema + proceed flag + non-finance ============
    {
        "tc_id": "SPEC_SCHEMA_NON_FIN_1",
        "category": "schema_non_finance",
        "prompt": "What's the weather like in New York tomorrow and should I bring an umbrella?",
        "expected_endpoints": [],
        "notes": "Non-finance → proceed=false, everything else empty.",
        "checks": {
            "proceed": False,
            "tickers_exact": [],
            "semantic_keywords_exact": [],
            "enrichment_exact": [],
            "api_requests_exact": 0,
            "require_top_level_fields": ["proceed", "features", "semantic_keywords", "tickers", "api_requests"],
        },
    },
    {
        "tc_id": "SPEC_SCHEMA_FIN_1",
        "category": "schema_finance",
        "prompt": "Show me AAPL daily prices for the last month as a dataset.",
        "expected_endpoints": [
            "polygon.get_aggs",
            "alpha_vantage.TIME_SERIES_DAILY_ADJUSTED",
        ],
        "notes": "Basic finance query, correct schema.",
        "checks": {
            "proceed": True,
            "tickers_include": ["AAPL"],
            "require_top_level_fields": ["proceed", "features", "semantic_keywords", "tickers", "api_requests"],
            "require_api_request_fields": ["api_name", "endpoint_name", "parameters", "reasoning"],
            "api_requests_min": 2,
            "must_have_endpoints": [
                "polygon.get_aggs",
                "alpha_vantage.TIME_SERIES_DAILY_ADJUSTED",
            ],
        },
    },
    {
        "tc_id": "SPEC_SCHEMA_REASONING_1",
        "category": "schema_reasoning",
        "prompt": "TSLA 5-minute intraday data today as a table.",
        "expected_endpoints": [
            "polygon.get_aggs",
            "alpha_vantage.TIME_SERIES_INTRADAY",
        ],
        "notes": "Every api_request must have reasoning.",
        "checks": {
            "proceed": True,
            "tickers_include": ["TSLA"],
            "semantic_keywords_include": ["intraday", "stock", "today"],
            "api_requests_min": 2,
            "must_have_endpoints": [
                "polygon.get_aggs",
                "alpha_vantage.TIME_SERIES_INTRADAY",
            ],
            "require_api_request_fields": ["api_name", "endpoint_name", "parameters", "reasoning"],
        },
    },

    # ============ 2. Enrichment features & semantic_keywords ============
    {
        "tc_id": "SPEC_ENRICH_NONE_1",
        "category": "enrichment",
        "prompt": "Give me Apple's daily price and volume history for the last 30 days.",
        "expected_endpoints": [
            "polygon.get_aggs",
            "alpha_vantage.TIME_SERIES_DAILY_ADJUSTED",
        ],
        "notes": "No enrichment requested.",
        "checks": {
            "proceed": True,
            "tickers_include": ["AAPL"],
            "enrichment_exact": [],
            "semantic_keywords_include": ["stock", "daily", "price"],
            "must_have_endpoints": [
                "polygon.get_aggs",
                "alpha_vantage.TIME_SERIES_DAILY_ADJUSTED",
            ],
        },
    },
    {
        "tc_id": "SPEC_ENRICH_SMA_RSI_1",
        "category": "enrichment",
        "prompt": "For MSFT, I want a daily price dataset with 20-day SMA and 14-day RSI added.",
        "expected_endpoints": [
            "polygon.get_aggs",
            "alpha_vantage.TIME_SERIES_DAILY_ADJUSTED",
        ],
        "notes": "SMA + RSI enrichment only; no OHLCV in features.",
        "checks": {
            "proceed": True,
            "tickers_include": ["MSFT"],
            "enrichment_include": ["SMA_20", "RSI_14"],
            "enrichment_forbidden": ["open", "high", "low", "close", "volume"],
            "semantic_keywords_include": ["stock", "daily", "price"],
            "must_have_endpoints": [
                "polygon.get_aggs",
                "alpha_vantage.TIME_SERIES_DAILY_ADJUSTED",
            ],
        },
    },
    {
        "tc_id": "SPEC_ENRICH_RET_VOL_1",
        "category": "enrichment",
        "prompt": "Build a dataset of SPY daily log returns and 30-day realized volatility over the last year.",
        "expected_endpoints": [
            "polygon.get_aggs",
            "alpha_vantage.TIME_SERIES_DAILY_ADJUSTED",
        ],
        "notes": "log_returns + realized_volatility_30d",
        "checks": {
            "proceed": True,
            "tickers_include": ["SPY"],
            "enrichment_include": ["log_returns", "realized_volatility_30d"],
            "enrichment_forbidden": ["open", "high", "low", "close", "volume"],
            "semantic_keywords_include": ["stock", "daily", "returns", "volatility"],
            "must_have_endpoints": [
                "polygon.get_aggs",
                "alpha_vantage.TIME_SERIES_DAILY_ADJUSTED",
            ],
        },
    },

    # ============ 3. Tickers & macro non-tickers ============
    {
        "tc_id": "SPEC_TICKERS_SYMBOL_1",
        "category": "tickers",
        "prompt": "Show me TSLA stock data for the last month.",
        "expected_endpoints": [
            "polygon.get_aggs",
            "alpha_vantage.TIME_SERIES_DAILY_ADJUSTED",
        ],
        "notes": "Explicit ticker TSLA.",
        "checks": {
            "proceed": True,
            "tickers_exact": ["TSLA"],
            "semantic_keywords_include": ["stock", "daily", "price"],
        },
    },
    {
        "tc_id": "SPEC_TICKERS_NAME_TO_SYMBOL_1",
        "category": "tickers",
        "prompt": "Give me Microsoft stock prices for the past 3 months as a dataset.",
        "expected_endpoints": [],
        "notes": "Name → MSFT",
        "checks": {
            "proceed": True,
            "tickers_include": ["MSFT"],
            "semantic_keywords_include": ["stock", "daily", "price"],
        },
    },
    {
        "tc_id": "SPEC_TICKERS_MACRO_NOT_TICKERS_1",
        "category": "tickers_macro",
        "prompt": "Build datasets for CPI, unemployment, and nonfarm payrolls for the last 10 years.",
        "expected_endpoints": [
            "alpha_vantage.CPI",
            "alpha_vantage.UNEMPLOYMENT",
        ],
        "notes": "CPI/UNEMPLOYMENT/NFP are NOT tickers.",
        "checks": {
            "proceed": True,
            "tickers_exact": [],
            "semantic_keywords_include": ["economic", "cpi", "unemployment"],
            "must_have_endpoints": [
                "alpha_vantage.CPI",
                "alpha_vantage.UNEMPLOYMENT",
            ],
        },
    },

    # ============ 4. Default timeframe & override ============
    {
        "tc_id": "SPEC_TIME_DEFAULT_30D_1",
        "category": "timeframe",
        "prompt": "Show me AAPL daily prices as a dataset (no need for a long history).",
        "expected_endpoints": [
            "polygon.get_aggs",
            "alpha_vantage.TIME_SERIES_DAILY_ADJUSTED",
        ],
        "notes": "Should use default 30-day window.",
        "checks": {
            "proceed": True,
            "tickers_include": ["AAPL"],
            "semantic_keywords_include": ["stock", "daily", "price"],
            "must_have_endpoints": [
                "polygon.get_aggs",
                "alpha_vantage.TIME_SERIES_DAILY_ADJUSTED",
            ],
        },
    },
    {
        "tc_id": "SPEC_TIME_LONG_RANGE_1",
        "category": "timeframe",
        "prompt": "Give me daily prices for TSLA for the last 5 years.",
        "expected_endpoints": [],
        "notes": "Should override default 30 days and use long history.",
        "checks": {
            "proceed": True,
            "tickers_include": ["TSLA"],
            "semantic_keywords_include": ["stock", "daily", "price"],
        },
    },
    {
        "tc_id": "SPEC_TIME_TODAY_INTRADAY_1",
        "category": "timeframe_intraday",
        "prompt": "TSLA 5-minute data today as a dataset.",
        "expected_endpoints": [
            "polygon.get_aggs",
            "alpha_vantage.TIME_SERIES_INTRADAY",
        ],
        "notes": "Intraday today mapping.",
        "checks": {
            "proceed": True,
            "tickers_include": ["TSLA"],
            "semantic_keywords_include": ["intraday", "stock", "today"],
            "must_have_endpoints": [
                "polygon.get_aggs",
                "alpha_vantage.TIME_SERIES_INTRADAY",
            ],
        },
    },

    # ============ 5. Vague fundamentals, econ, etc. ============
    {
        "tc_id": "SPEC_STOCK_VAGUE_NO_TICKER_1",
        "category": "stock_vague",
        "prompt": "I want stock data as a dataset, nothing specific.",
        "expected_endpoints": [],
        "notes": "Stock market overview; default basket + full market snapshot.",
        "checks": {
            "proceed": True,
            "tickers_exact": [],
            "semantic_keywords_include": ["stock", "market", "snapshot", "daily"],
            "api_requests_min": 1,
        },
    },
    {
        "tc_id": "SPEC_ECON_VAGUE_1",
        "category": "economic_vague",
        "prompt": "Show me economic indicators so I can get a sense of the macro situation.",
        "expected_endpoints": [],
        "notes": "Macro: polygon.INFLATION + AV.CPI.",
        "checks": {
            "proceed": True,
            "tickers_exact": [],
            "semantic_keywords_include": ["economic", "macro", "cpi", "inflation"],
            "must_have_endpoints": [
                "polygon.INFLATION",
                "alpha_vantage.CPI",
            ],
        },
    },

    # ============ 6. Manifest canonical + fallback ============
    {
        "tc_id": "SPEC_MANIFEST_NO_INVENT_1",
        "category": "manifest",
        "prompt": "Give me a dataset of the option chain for TSLA.",
        "expected_endpoints": [],
        "notes": "No options endpoint in manifest, must not invent one; use closest fallback.",
        "checks": {
            "proceed": True,
            "tickers_include": ["TSLA"],
            "semantic_keywords_include": ["stock"],
            "api_requests_min": 1,
        },
    },
    {
        "tc_id": "SPEC_MANIFEST_OVERRIDE_DEFAULT_1",
        "category": "manifest_time_override",
        "prompt": "For TSLA, please don't use a 1-month window – instead, give me a full history of daily prices.",
        "expected_endpoints": [],
        "notes": "User overrides default timeframe.",
        "checks": {
            "proceed": True,
            "tickers_include": ["TSLA"],
            "semantic_keywords_include": ["stock", "daily", "price"],
        },
    },
]


ALL_TEST_CASES: List[Dict[str, Any]] = routing_test_cases + spec_test_cases


# ============================================================
# Validation engine
# ============================================================

def _endpoint_key(req: APIRequestView) -> str:
    return f"{req.api_name}.{req.endpoint_name}"


def validate_response(tc: Dict[str, Any], view: LLMResponseView) -> Dict[str, Any]:
    """
    Apply tc["checks"] to the LLMResponseView.
    Returns:
        {
          "passed": bool,
          "errors": [str, ...]
        }
    """
    checks = tc.get("checks", {}) or {}
    errors: List[str] = []

    # 1) Proceed
    if "proceed" in checks:
        if view.proceed != checks["proceed"]:
            errors.append(f"proceed mismatch: expected {checks['proceed']} got {view.proceed}")

    # 2) Top-level fields (we can only check presence on raw pydantic dict)
    raw_dict = None
    try:
        if hasattr(view.raw, "model_dump"):
            raw_dict = view.raw.model_dump()
        elif hasattr(view.raw, "dict"):
            raw_dict = view.raw.dict()
        else:
            raw_dict = json.loads(view.raw)
    except Exception:
        raw_dict = None

    if raw_dict is not None and "require_top_level_fields" in checks:
        for fld in checks["require_top_level_fields"]:
            if fld not in raw_dict:
                errors.append(f"top-level field missing: {fld}")

    # 3) Tickers
    if "tickers_exact" in checks:
        expected = checks["tickers_exact"]
        if view.tickers != expected:
            errors.append(f"tickers_exact mismatch: expected {expected} got {view.tickers}")

    if "tickers_include" in checks:
        for t in checks["tickers_include"]:
            if t not in view.tickers:
                errors.append(f"ticker {t} not found in {view.tickers}")

    if "tickers_exclude" in checks:
        for t in checks["tickers_exclude"]:
            if t in view.tickers:
                errors.append(f"ticker {t} should NOT be present in {view.tickers}")

    # 4) semantic_keywords
    if "semantic_keywords_exact" in checks:
        exp = checks["semantic_keywords_exact"]
        if view.semantic_keywords != exp:
            errors.append(
                f"semantic_keywords_exact mismatch: expected {exp} got {view.semantic_keywords}"
            )

    if "semantic_keywords_include" in checks:
        for kw in checks["semantic_keywords_include"]:
            if kw not in view.semantic_keywords:
                errors.append(
                    f"semantic keyword '{kw}' not in {view.semantic_keywords}"
                )

    if "semantic_keywords_exclude" in checks:
        for kw in checks["semantic_keywords_exclude"]:
            if kw in view.semantic_keywords:
                errors.append(
                    f"semantic keyword '{kw}' should NOT be in {view.semantic_keywords}"
                )

    # 5) enrichment
    if "enrichment_exact" in checks:
        exp = checks["enrichment_exact"]
        if view.enrichment != exp:
            errors.append(f"enrichment_exact mismatch: expected {exp} got {view.enrichment}")

    if "enrichment_include" in checks:
        for e in checks["enrichment_include"]:
            if e not in view.enrichment:
                errors.append(f"enrichment feature '{e}' not found in {view.enrichment}")

    if "enrichment_forbidden" in checks:
        for e in checks["enrichment_forbidden"]:
            if e in view.enrichment:
                errors.append(f"forbidden enrichment feature '{e}' found in {view.enrichment}")

    # 6) api_requests length
    if "api_requests_exact" in checks:
        exp_n = checks["api_requests_exact"]
        if len(view.api_requests) != exp_n:
            errors.append(f"api_requests_exact mismatch: expected {exp_n} got {len(view.api_requests)}")

    if "api_requests_min" in checks:
        exp_min = checks["api_requests_min"]
        if len(view.api_requests) < exp_min:
            errors.append(
                f"api_requests_min violation: expected at least {exp_min}, got {len(view.api_requests)}"
            )

    # 7) endpoints presence/forbidden
    actual_endpoints = [_endpoint_key(r) for r in view.api_requests]

    if "must_have_endpoints" in checks:
        for ep in checks["must_have_endpoints"]:
            if ep not in actual_endpoints:
                errors.append(f"missing required endpoint: {ep} (actual: {actual_endpoints})")

    if "forbidden_endpoints" in checks:
        for ep in checks["forbidden_endpoints"]:
            if ep in actual_endpoints:
                errors.append(f"forbidden endpoint present: {ep}")

    # 8) require_api_request_fields
    if "require_api_request_fields" in checks:
        required_fields = checks["require_api_request_fields"]
        for idx, r in enumerate(view.api_requests):
            r_dict = {
                "api_name": r.api_name,
                "endpoint_name": r.endpoint_name,
                "parameters": r.parameters,
                "reasoning": r.reasoning,
            }
            for fld in required_fields:
                if fld not in r_dict:
                    errors.append(f"api_request[{idx}] missing field '{fld}'")
                else:
                    # simple non-empty check for reasoning
                    if fld == "reasoning" and not r_dict[fld]:
                        errors.append(f"api_request[{idx}] has empty reasoning")

    return {
        "passed": len(errors) == 0,
        "errors": errors,
    }


# ============================================================
# Runner: executes all test cases and writes CSV
# ============================================================

CSV_FIELDS = [
    "tc_id",
    "category",
    "prompt",
    "notes",
    "expected_endpoints",
    "actual_endpoints",
    "proceed",
    "tickers",
    "semantic_keywords",
    "enrichment",
    "passed",
    "error_count",
    "errors",
    "raw_json",
]


def run_all_tests(api_key: str, model: str, csv_path: str = "llm_routing_test_results.csv") -> None:
    qa = QueryAnalyzer(api_key=api_key, model=model)

    rows: List[Dict[str, Any]] = []
    passed_count = 0

    for tc in ALL_TEST_CASES:
        tc_id = tc["tc_id"]
        prompt = tc["prompt"]

        try:
            resp = qa.analyze(prompt)
        except Exception as e:
            # Hard failure for this test case
            error_msg = f"Exception calling QueryAnalyzer: {e}"
            print(f"[{tc_id}] ERROR: {error_msg}")
            rows.append(
                {
                    "tc_id": tc_id,
                    "category": tc.get("category", ""),
                    "prompt": prompt,
                    "notes": tc.get("notes", ""),
                    "expected_endpoints": ";".join(tc.get("expected_endpoints", [])),
                    "actual_endpoints": "",
                    "proceed": "",
                    "tickers": "",
                    "semantic_keywords": "",
                    "enrichment": "",
                    "passed": False,
                    "error_count": 1,
                    "errors": error_msg,
                    "raw_json": "",
                }
            )
            continue

        view = _to_view(resp)
        result = validate_response(tc, view)
        passed = result["passed"]
        errors = result["errors"]

        if passed:
            passed_count += 1

        # flatten for CSV
        actual_endpoints = ";".join(_endpoint_key(r) for r in view.api_requests)
        try:
            if hasattr(view.raw, "model_dump_json"):
                raw_json = view.raw.model_dump_json()
            elif hasattr(view.raw, "json"):
                raw_json = view.raw.json()
            else:
                raw_json = json.dumps(view.raw, default=str)
        except Exception:
            raw_json = ""

        row = {
            "tc_id": tc_id,
            "category": tc.get("category", ""),
            "prompt": prompt,
            "notes": tc.get("notes", ""),
            "expected_endpoints": ";".join(tc.get("expected_endpoints", [])),
            "actual_endpoints": actual_endpoints,
            "proceed": view.proceed,
            "tickers": ";".join(view.tickers),
            "semantic_keywords": ";".join(view.semantic_keywords),
            "enrichment": ";".join(view.enrichment),
            "passed": passed,
            "error_count": len(errors),
            "errors": " | ".join(errors),
            "raw_json": raw_json,
        }
        rows.append(row)

        status = "PASS" if passed else "FAIL"
        print(f"[{tc_id}] {status}")
        if errors:
            for e in errors:
                print(f"   - {e}")

    # write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    total = len(ALL_TEST_CASES)
    print(f"\nDone. {passed_count}/{total} tests passed.")
    print(f"Results written to: {csv_path}")


# ============================================================
# CLI entrypoint
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LLM routing + spec test harness")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-5-nano", help="Model name for QueryAnalyzer")
    parser.add_argument("--csv-path", default="llm_routing_test_results.csv", help="Output CSV path")
    args = parser.parse_args()

    run_all_tests(api_key='', model=args.model, csv_path=args.csv_path)


if __name__ == "__main__":
    main()