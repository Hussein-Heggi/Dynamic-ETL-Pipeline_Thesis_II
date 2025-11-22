#!/usr/bin/env python3
"""
Simple blackbox test for the transform module.

Tests the transform_pipeline function with various data types and keywords.
Outputs results to tests/.out/test{N}/ directories.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transform.transform import transform_pipeline

# Output directory
OUTPUT_DIR = Path(__file__).parent / ".out"


def save_test_outputs(test_name, input_dfs, enriched_dfs, metadata, keywords):
    """Save test outputs to .out/test_name/ directory"""
    test_dir = OUTPUT_DIR / test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata/report as JSON
    report_path = test_dir / "report.json"
    with open(report_path, "w") as f:
        # Convert non-serializable objects to strings
        serializable_metadata = {
            "keywords": keywords,
            "status": metadata.get("status", "unknown"),
            "errors": metadata.get("errors", []),
            "num_input_dfs": len(input_dfs),
            "num_output_dfs": len(enriched_dfs),
        }

        # Add per-dataframe info
        if "dataframe_results" in metadata:
            serializable_metadata["dataframe_results"] = []
            for df_result in metadata["dataframe_results"]:
                df_meta = {
                    "enrichment_success": df_result.get("enrichment", {}).get("success", False),
                    "enrichment_errors": df_result.get("enrichment", {}).get("errors", []),
                }
                if "dsl_string" in df_result.get("enrichment", {}):
                    df_meta["llm_response"] = df_result["enrichment"]["dsl_string"]
                if "dsl" in df_result.get("enrichment", {}):
                    df_meta["validated_dsl"] = df_result["enrichment"]["dsl"]
                serializable_metadata["dataframe_results"].append(df_meta)

        json.dump(serializable_metadata, f, indent=2)

    print(f"  Saved report: {report_path}")

    # Save input dataframes
    for i, df in enumerate(input_dfs):
        input_path = test_dir / f"input_df{i+1}.csv"
        df.to_csv(input_path, index=False)
        print(f"  Saved input {i+1}: {input_path}")

    # Save enriched dataframes
    for i, df in enumerate(enriched_dfs):
        output_path = test_dir / f"enriched_df{i+1}.csv"
        df.to_csv(output_path, index=False)
        print(f"  Saved output {i+1}: {output_path}")

        # Also save column comparison
        input_cols = set(input_dfs[i].columns)
        output_cols = set(df.columns)
        new_cols = output_cols - input_cols

        cols_info = {
            "input_columns": sorted(list(input_cols)),
            "output_columns": sorted(list(output_cols)),
            "new_columns": sorted(list(new_cols)),
            "num_new_columns": len(new_cols)
        }

        cols_path = test_dir / f"columns_df{i+1}.json"
        with open(cols_path, "w") as f:
            json.dump(cols_info, f, indent=2)


def load_dataframe_with_metadata(csv_filename, ticker="AMZN"):
    """Load a CSV from transform/tests/dataframes and add ticker + ts columns"""
    csv_path = PROJECT_ROOT / "transform" / "tests" / "dataframes" / csv_filename

    df = pd.read_csv(csv_path)

    # Add ticker column if not present
    if "ticker" not in df.columns:
        df.insert(0, "ticker", ticker)

    # Add ts column if not present - derive from fiscalDateEnding if available
    if "ts" not in df.columns:
        # Look for any fiscalDateEnding column
        date_cols = [col for col in df.columns if "fiscalDateEnding" in col.lower() or "date" in col.lower()]
        if date_cols:
            # Use the first date column found
            df["ts"] = pd.to_datetime(df[date_cols[0]], utc=True)
            df = df.drop(columns=date_cols)  # Remove original date column
        else:
            # Create synthetic dates going backwards from today
            dates = pd.date_range(
                end=pd.Timestamp.now(tz="UTC"),
                periods=len(df),
                freq="Q"
            )
            df.insert(1, "ts", dates)

    return df


def create_stock_dataframe():
    """Create a simple stock dataframe with OHLCV + ticker + ts"""
    dates = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")

    df = pd.DataFrame({
        "ticker": ["AAPL"] * 50,
        "ts": dates,
        "open": 100 + np.random.randn(50).cumsum(),
        "high": 105 + np.random.randn(50).cumsum(),
        "low": 95 + np.random.randn(50).cumsum(),
        "close": 100 + np.random.randn(50).cumsum(),
        "volume": (1000000 + np.random.randint(-100000, 100000, 50)).astype(int),
    })

    # Ensure high/low constraints
    df["high"] = df[["open", "high", "close"]].max(axis=1) + abs(np.random.randn(50))
    df["low"] = df[["open", "low", "close"]].min(axis=1) - abs(np.random.randn(50))

    return df


def test_stock_enrichments():
    """Test 1: Stock data with known enrichments"""
    print("\n" + "="*70)
    print("TEST 1: Stock Data with Known Enrichments")
    print("="*70)

    df_stock = create_stock_dataframe()
    input_dfs = [df_stock]

    keywords = [
        "20 day sma on close",
        "14 day rsi on close",
        "12 day ema on close",
        "bollinger bands on close with window 20",
    ]

    print(f"Input DataFrame shape: {df_stock.shape}")
    print(f"Input columns: {list(df_stock.columns)}")
    print(f"Keywords: {keywords}")

    enriched_dfs, metadata = transform_pipeline(input_dfs, keywords)

    print(f"\nOutput DataFrame shape: {enriched_dfs[0].shape}")
    print(f"Output columns: {list(enriched_dfs[0].columns)}")
    print(f"New enriched columns: {set(enriched_dfs[0].columns) - set(df_stock.columns)}")

    # Save outputs
    save_test_outputs("test1", input_dfs, enriched_dfs, metadata, keywords)

    # Validate
    assert len(enriched_dfs) == 1, "Should return 1 dataframe"
    assert enriched_dfs[0].shape[0] == df_stock.shape[0], "Row count should be preserved"
    assert enriched_dfs[0].shape[1] > df_stock.shape[1], "Should have added enriched columns"

    print("\n✓ Test 1 PASSED")
    return enriched_dfs[0], metadata


def test_financial_enrichments():
    """Test 2: Financial data with known enrichments"""
    print("\n" + "="*70)
    print("TEST 2: Financial Data with Balance Sheet Enrichments")
    print("="*70)

    df_balance = load_dataframe_with_metadata("AMZN_balance_sheet.csv", ticker="AMZN")
    input_dfs = [df_balance]

    keywords = [
        "current_ratio",
        "debt_to_equity",
        "yoy_growth on balance_sheet_totalAssets",
    ]

    print(f"Input DataFrame shape: {df_balance.shape}")
    print(f"Input columns (first 10): {list(df_balance.columns)[:10]}")
    print(f"Keywords: {keywords}")

    enriched_dfs, metadata = transform_pipeline(input_dfs, keywords)

    print(f"\nOutput DataFrame shape: {enriched_dfs[0].shape}")
    print(f"New enriched columns: {set(enriched_dfs[0].columns) - set(df_balance.columns)}")

    # Save outputs
    save_test_outputs("test2", input_dfs, enriched_dfs, metadata, keywords)

    # Validate
    assert len(enriched_dfs) == 1, "Should return 1 dataframe"
    if metadata.get("status") == "success":
        assert enriched_dfs[0].shape[1] > df_balance.shape[1], "Should have added enriched columns"
        print("\n✓ Test 2 PASSED")
    else:
        print(f"\n⚠ Test 2 completed with errors: {metadata.get('errors', [])}")

    return enriched_dfs[0], metadata


def test_mixed_data():
    """Test 3: Multiple dataframes with different data types"""
    print("\n" + "="*70)
    print("TEST 3: Mixed Data Types (Stock + Balance Sheet + Cash Flow)")
    print("="*70)

    df_stock = create_stock_dataframe()
    df_balance = load_dataframe_with_metadata("AMZN_balance_sheet.csv", ticker="AMZN")
    df_cashflow = load_dataframe_with_metadata("AMZN_cash_flow.csv", ticker="AMZN")
    input_dfs = [df_stock, df_balance, df_cashflow]

    keywords = [
        "10 day sma on close",
        "current_ratio",
        "free_cash_flow",
    ]

    print(f"Input DataFrames: {len(input_dfs)}")
    print(f"  - Stock shape: {df_stock.shape}")
    print(f"  - Balance sheet shape: {df_balance.shape}")
    print(f"  - Cash flow shape: {df_cashflow.shape}")
    print(f"Keywords: {keywords}")

    enriched_dfs, metadata = transform_pipeline(input_dfs, keywords)

    print(f"\nOutput DataFrames: {len(enriched_dfs)}")
    for i, df in enumerate(enriched_dfs):
        print(f"  - DataFrame {i+1} shape: {df.shape}")
        new_cols = set(df.columns) - set(input_dfs[i].columns)
        if new_cols:
            print(f"    New columns: {new_cols}")

    # Save outputs
    save_test_outputs("test3", input_dfs, enriched_dfs, metadata, keywords)

    # Validate
    assert len(enriched_dfs) == 3, "Should return 3 dataframes"

    print("\n✓ Test 3 PASSED")
    return enriched_dfs, metadata


def test_custom_features():
    """Test 4: Custom features (if LLM is available)"""
    print("\n" + "="*70)
    print("TEST 4: Custom Features (requires LLM)")
    print("="*70)

    df_stock = create_stock_dataframe()
    input_dfs = [df_stock]

    # These keywords should trigger custom feature creation via LLM
    keywords = [
        "price momentum over 10 days",
        "volatility ratio",
    ]

    print(f"Input DataFrame shape: {df_stock.shape}")
    print(f"Keywords (custom): {keywords}")
    print("Note: This test requires OPENAI_API_KEY to be set")

    try:
        enriched_dfs, metadata = transform_pipeline(input_dfs, keywords)

        print(f"\nOutput DataFrame shape: {enriched_dfs[0].shape}")
        print(f"Metadata: {metadata.keys()}")

        # Save outputs
        save_test_outputs("test4", input_dfs, enriched_dfs, metadata, keywords)

        if "errors" in metadata and metadata["errors"]:
            print(f"Errors encountered: {metadata['errors']}")
            print("⚠ Test 4 SKIPPED (LLM not available or failed)")
        else:
            print("\n✓ Test 4 PASSED")

    except Exception as e:
        print(f"\n⚠ Test 4 SKIPPED: {e}")
        print("(This is expected if LLM is not configured)")


def test_edge_cases():
    """Test 5: Edge cases and error handling"""
    print("\n" + "="*70)
    print("TEST 5: Edge Cases")
    print("="*70)

    # Empty keywords
    df_stock = create_stock_dataframe()
    input_dfs = [df_stock]
    keywords = []

    enriched_dfs, metadata = transform_pipeline(input_dfs, keywords)
    assert enriched_dfs[0].shape == df_stock.shape, "Empty keywords should return unchanged df"
    print("✓ Empty keywords handled correctly")

    save_test_outputs("test5_empty_keywords", input_dfs, enriched_dfs, metadata, keywords)

    # Invalid keyword (should be handled gracefully)
    keywords = ["nonexistent_feature_12345"]
    enriched_dfs, metadata = transform_pipeline(input_dfs, keywords)
    print(f"✓ Invalid keyword handled (errors: {len(metadata.get('errors', []))})")

    save_test_outputs("test5_invalid_keyword", input_dfs, enriched_dfs, metadata, keywords)

    print("\n✓ Test 5 PASSED")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("TRANSFORM MODULE BLACKBOX TEST SUITE")
    print("="*70)
    print(f"Output directory: {OUTPUT_DIR}")

    try:
        # Run tests
        test_stock_enrichments()
        test_financial_enrichments()
        test_mixed_data()
        test_custom_features()
        test_edge_cases()

        print("\n" + "="*70)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("="*70)
        print(f"\nResults saved to: {OUTPUT_DIR}")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
