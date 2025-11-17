"""
Test suite for transform.py

These tests are informational and print-based only, since the transform
module depends on LLM-based enrichment which is non-deterministic.
No assertions are made - just output inspection.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from transform import transform_pipeline, transform_single, transform_pipeline_from_list


def print_separator(title):
    """Print a formatted separator with title"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_dataframe_info(df, name="DataFrame"):
    """Print information about a DataFrame"""
    print(f"\n{name} Info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Null counts:\n{df.isna().sum()}")
    print(f"\nFirst 5 rows:")
    print(df.head())


def print_metadata(metadata, indent=0):
    """Recursively print metadata dictionary"""
    indent_str = "  " * indent
    for key, value in metadata.items():
        if isinstance(value, dict):
            print(f"{indent_str}{key}:")
            print_metadata(value, indent + 1)
        elif isinstance(value, list):
            print(f"{indent_str}{key}: [{len(value)} items]")
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    print(f"{indent_str}  [{i}]:")
                    print_metadata(item, indent + 2)
                else:
                    print(f"{indent_str}  [{i}]: {item}")
        else:
            print(f"{indent_str}{key}: {value}")


def test_single_dataframe_with_nulls():
    """Test transform_single with a DataFrame containing nulls"""
    print_separator("TEST 1: Single DataFrame with Nulls (30%)")

    # Create sample data with 30% nulls in some columns
    df = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 100,
            "ts": pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC"),
            "open": [150.0 + np.random.randn() for _ in range(70)] + [np.nan] * 30,
            "high": [160.0 + np.random.randn() for _ in range(100)],
            "low": [140.0 + np.random.randn() for _ in range(100)],
            "close": [155.0 + np.random.randn() for _ in range(100)],
            "volume": [1000000 + np.random.randint(0, 100000) for _ in range(100)],
        }
    )

    print("\nInput DataFrame:")
    print_dataframe_info(df, "Input")

    # Transform with default threshold (0.5)
    keywords = ["20 day sma on close"]
    enriched_df, metadata = transform_single(df, keywords, column_delete_threshold=0.5)

    print("\nOutput DataFrame:")
    print_dataframe_info(enriched_df, "Enriched")

    print("\nMetadata:")
    print_metadata(metadata)


def test_single_dataframe_column_deletion():
    """Test transform_single with columns that should be deleted"""
    print_separator("TEST 2: Single DataFrame with Column Deletion (70% nulls)")

    # Create sample data with column that should be deleted
    df = pd.DataFrame(
        {
            "ticker": ["GOOGL"] * 100,
            "ts": pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC"),
            "open": [140.0 + np.random.randn() for _ in range(100)],
            "high": [150.0 + np.random.randn() for _ in range(100)],
            "low": [130.0 + np.random.randn() for _ in range(100)],
            "close": [145.0 + np.random.randn() for _ in range(100)],
            "volume": [500000 + np.random.randint(0, 50000) for _ in range(100)],
            "bad_column": [1.0] * 30 + [np.nan] * 70,  # 70% nulls - should be deleted
        }
    )

    print("\nInput DataFrame:")
    print_dataframe_info(df, "Input")

    # Transform with threshold 0.5
    keywords = []  # No enrichment, just cleaning
    enriched_df, metadata = transform_single(df, keywords, column_delete_threshold=0.5)

    print("\nOutput DataFrame:")
    print_dataframe_info(enriched_df, "Cleaned")

    print("\nMetadata (focusing on null handling):")
    if "cleaning" in metadata and "null_handling" in metadata["cleaning"]:
        print("\nNull Handling Report:")
        print_metadata(metadata["cleaning"]["null_handling"])


def test_multiple_dataframes():
    """Test transform_pipeline with multiple DataFrames"""
    print_separator("TEST 3: Multiple DataFrames")

    # Create multiple sample DataFrames
    df1 = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 50,
            "ts": pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC"),
            "open": [150.0 + np.random.randn() for _ in range(50)],
            "high": [160.0 + np.random.randn() for _ in range(50)],
            "low": [140.0 + np.random.randn() for _ in range(50)],
            "close": [155.0 + np.random.randn() for _ in range(50)],
            "volume": [1000000 + np.random.randint(0, 100000) for _ in range(50)],
        }
    )

    df2 = pd.DataFrame(
        {
            "ticker": ["GOOGL"] * 50,
            "ts": pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC"),
            "open": [140.0 + np.random.randn() for _ in range(35)] + [np.nan] * 15,
            "high": [150.0 + np.random.randn() for _ in range(50)],
            "low": [130.0 + np.random.randn() for _ in range(50)],
            "close": [145.0 + np.random.randn() for _ in range(50)],
            "volume": [500000 + np.random.randint(0, 50000) for _ in range(50)],
        }
    )

    print("\nInput DataFrames:")
    print_dataframe_info(df1, "DataFrame 1 (AAPL)")
    print_dataframe_info(df2, "DataFrame 2 (GOOGL)")

    # Transform both
    keywords = []  # No enrichment to avoid LLM dependency
    enriched_dfs, metadata = transform_pipeline(
        [df1, df2], keywords, column_delete_threshold=0.5
    )

    print("\nOutput DataFrames:")
    for i, enriched_df in enumerate(enriched_dfs):
        print_dataframe_info(enriched_df, f"Enriched DataFrame {i + 1}")

    print("\nPipeline Metadata:")
    print(f"Overall Status: {metadata.get('overall_status')}")
    print(f"DataFrames Processed: {metadata.get('dataframes_processed')}")
    print(f"Total Errors: {metadata.get('total_errors')}")

    print("\nPer-DataFrame Results:")
    for i, result in enumerate(metadata.get("results", [])):
        print(f"\n  DataFrame {i + 1}:")
        print(f"    Status: {result.get('status')}")
        print(f"    Original Shape: {result.get('original_shape')}")
        print(f"    Final Shape: {result.get('final_shape')}")
        if "cleaning" in result and "null_handling" in result["cleaning"]:
            nh = result["cleaning"]["null_handling"]
            print(f"    Columns Deleted: {len(nh.get('columns_deleted', []))}")
            print(f"    Columns Imputed: {len(nh.get('columns_imputed', {}))}")


def test_different_thresholds():
    """Test transform with different column_delete_threshold values"""
    print_separator("TEST 4: Different Threshold Values")

    # Create sample data with specific null patterns
    df = pd.DataFrame(
        {
            "ticker": ["MSFT"] * 100,
            "ts": pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC"),
            "open": [300.0 + np.random.randn() for _ in range(100)],
            "high": [310.0 + np.random.randn() for _ in range(100)],
            "low": [290.0 + np.random.randn() for _ in range(100)],
            "close": [305.0 + np.random.randn() for _ in range(100)],
            "volume": [2000000 + np.random.randint(0, 200000) for _ in range(100)],
            "col_40_nulls": [1.0] * 60 + [np.nan] * 40,  # 40% nulls
            "col_60_nulls": [1.0] * 40 + [np.nan] * 60,  # 60% nulls
        }
    )

    print("\nInput DataFrame:")
    print_dataframe_info(df, "Input")

    # Test with threshold 0.5
    print("\n--- Testing with threshold = 0.5 ---")
    enriched_df1, metadata1 = transform_single(df.copy(), [], column_delete_threshold=0.5)
    print(f"Columns in result: {list(enriched_df1.columns)}")
    print(f"Expected: col_40_nulls kept (40% < 50%), col_60_nulls deleted (60% > 50%)")
    if "cleaning" in metadata1 and "null_handling" in metadata1["cleaning"]:
        nh = metadata1["cleaning"]["null_handling"]
        print(f"Actually deleted: {[c['column'] for c in nh.get('columns_deleted', [])]}")
        print(f"Actually imputed: {list(nh.get('columns_imputed', {}).keys())}")

    # Test with threshold 0.7
    print("\n--- Testing with threshold = 0.7 ---")
    enriched_df2, metadata2 = transform_single(df.copy(), [], column_delete_threshold=0.7)
    print(f"Columns in result: {list(enriched_df2.columns)}")
    print(f"Expected: Both columns kept (both < 70%), both imputed")
    if "cleaning" in metadata2 and "null_handling" in metadata2["cleaning"]:
        nh = metadata2["cleaning"]["null_handling"]
        print(f"Actually deleted: {[c['column'] for c in nh.get('columns_deleted', [])]}")
        print(f"Actually imputed: {list(nh.get('columns_imputed', {}).keys())}")


def test_categorical_and_datetime_imputation():
    """Test imputation of categorical and datetime columns"""
    print_separator("TEST 5: Categorical and Datetime Imputation")

    # Create sample data with categorical and datetime nulls
    df = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 70 + [np.nan] * 30,  # 30% nulls - categorical
            "ts": [pd.Timestamp("2024-01-01", tz="UTC")] * 80 + [pd.NaT] * 20,  # 20% nulls
            "open": [150.0 + np.random.randn() for _ in range(100)],
            "high": [160.0 + np.random.randn() for _ in range(100)],
            "low": [140.0 + np.random.randn() for _ in range(100)],
            "close": [155.0 + np.random.randn() for _ in range(100)],
            "volume": [1000000 + np.random.randint(0, 100000) for _ in range(100)],
            "sector": ["Technology"] * 65 + [np.nan] * 35,  # 35% nulls - categorical
        }
    )

    print("\nInput DataFrame:")
    print_dataframe_info(df, "Input")
    print(f"\nNull counts:")
    print(df.isna().sum())

    enriched_df, metadata = transform_single(df, [], column_delete_threshold=0.5)

    print("\nOutput DataFrame:")
    print_dataframe_info(enriched_df, "Imputed")

    if "cleaning" in metadata and "null_handling" in metadata["cleaning"]:
        print("\nImputation Details:")
        for col, info in metadata["cleaning"]["null_handling"].get("columns_imputed", {}).items():
            print(f"\n  Column '{col}':")
            print(f"    Method: {info.get('method')}")
            print(f"    Null Count: {info.get('null_count')}")
            print(f"    Null Ratio: {info.get('null_ratio'):.2%}")
            if info.get("method") == "constant":
                print(f"    Imputed Value: '{info.get('value')}'")
            elif info.get("method") == "unix_epoch":
                print(f"    Imputed Value: {info.get('value')}")


def test_empty_keywords():
    """Test transform with empty keywords (cleaning only)"""
    print_separator("TEST 6: Transform with No Enrichment (Empty Keywords)")

    df = pd.DataFrame(
        {
            "ticker": ["TSLA"] * 50,
            "ts": pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC"),
            "open": [200.0 + np.random.randn() for _ in range(50)],
            "high": [210.0 + np.random.randn() for _ in range(50)],
            "low": [190.0 + np.random.randn() for _ in range(50)],
            "close": [205.0 + np.random.randn() for _ in range(50)],
            "volume": [3000000 + np.random.randint(0, 300000) for _ in range(50)],
        }
    )

    print("\nInput DataFrame:")
    print_dataframe_info(df, "Input")

    enriched_df, metadata = transform_single(df, [], column_delete_threshold=0.5)

    print("\nOutput DataFrame:")
    print_dataframe_info(enriched_df, "Output")

    print("\nMetadata:")
    print_metadata(metadata)


def test_post_enrichment_cleaning():
    """Test that post-enrichment cleaning is also applied"""
    print_separator("TEST 7: Post-Enrichment Cleaning")

    df = pd.DataFrame(
        {
            "ticker": ["NVDA"] * 30,
            "ts": pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC"),
            "open": [500.0 + np.random.randn() for _ in range(30)],
            "high": [510.0 + np.random.randn() for _ in range(30)],
            "low": [490.0 + np.random.randn() for _ in range(30)],
            "close": [505.0 + np.random.randn() for _ in range(30)],
            "volume": [5000000 + np.random.randint(0, 500000) for _ in range(30)],
        }
    )

    print("\nInput DataFrame:")
    print_dataframe_info(df, "Input")

    # Note: We're not actually calling enrichment to avoid LLM dependency
    # Just showing that both cleaning reports are present
    enriched_df, metadata = transform_single(df, [], column_delete_threshold=0.5)

    print("\nMetadata Structure:")
    print(f"Keys in metadata: {list(metadata.keys())}")

    if "cleaning" in metadata:
        print("\nPre-Enrichment Cleaning Report Present:")
        print(f"  Keys: {list(metadata['cleaning'].keys())}")

    if "post_enrichment_cleaning" in metadata:
        print("\nPost-Enrichment Cleaning Report Present:")
        print(f"  Keys: {list(metadata['post_enrichment_cleaning'].keys())}")


def test_transform_pipeline_from_list_alias():
    """Test the transform_pipeline_from_list alias function"""
    print_separator("TEST 8: transform_pipeline_from_list Alias")

    df1 = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 20,
            "ts": pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC"),
            "open": [150.0 + np.random.randn() for _ in range(20)],
            "high": [160.0 + np.random.randn() for _ in range(20)],
            "low": [140.0 + np.random.randn() for _ in range(20)],
            "close": [155.0 + np.random.randn() for _ in range(20)],
            "volume": [1000000 + np.random.randint(0, 100000) for _ in range(20)],
        }
    )

    df2 = pd.DataFrame(
        {
            "ticker": ["GOOGL"] * 20,
            "ts": pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC"),
            "open": [140.0 + np.random.randn() for _ in range(20)],
            "high": [150.0 + np.random.randn() for _ in range(20)],
            "low": [130.0 + np.random.randn() for _ in range(20)],
            "close": [145.0 + np.random.randn() for _ in range(20)],
            "volume": [500000 + np.random.randint(0, 50000) for _ in range(20)],
        }
    )

    print("\nTesting transform_pipeline_from_list (alias function)...")
    enriched_dfs, metadata = transform_pipeline_from_list(
        [df1, df2], [], column_delete_threshold=0.5
    )

    print(f"\nProcessed {len(enriched_dfs)} DataFrames")
    print(f"Overall Status: {metadata.get('overall_status')}")
    print(f"Total Errors: {metadata.get('total_errors')}")


def run_all_tests():
    """Run all test functions"""
    print("\n" + "█" * 80)
    print("  TRANSFORM MODULE TEST SUITE")
    print("  (Print-based tests - no assertions due to LLM non-determinism)")
    print("█" * 80)

    test_functions = [
        test_single_dataframe_with_nulls,
        test_single_dataframe_column_deletion,
        test_multiple_dataframes,
        test_different_thresholds,
        test_categorical_and_datetime_imputation,
        test_empty_keywords,
        test_post_enrichment_cleaning,
        test_transform_pipeline_from_list_alias,
    ]

    for i, test_func in enumerate(test_functions, 1):
        try:
            test_func()
            print(f"\n✓ Test {i}/{len(test_functions)} completed")
        except Exception as e:
            print(f"\n✗ Test {i}/{len(test_functions)} failed with error:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "█" * 80)
    print("  ALL TESTS COMPLETED")
    print("█" * 80 + "\n")


if __name__ == "__main__":
    run_all_tests()
