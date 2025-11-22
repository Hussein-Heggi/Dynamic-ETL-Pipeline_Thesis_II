"""
Test suite for data_cleaning.py

Tests the new null handling logic with column_delete_threshold,
imputation strategies, and data validation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transform.data_cleaning import clean_stock_bars, pipeline_clean


class TestNullHandling:
    """Test suite for null handling with column_delete_threshold"""

    def test_column_deletion_above_threshold(self):
        """Test that columns with null ratio > threshold are deleted"""
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 10,
                "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
                "open": [100.0] * 10,
                "high": [110.0] * 10,
                "low": [90.0] * 10,
                "close": [105.0] * 10,
                "volume": [1000000] * 10,
                "bad_column": [np.nan] * 8 + [1.0, 2.0],  # 80% nulls
            }
        )

        cleaned, report = clean_stock_bars(df, column_delete_threshold=0.5)

        # Column should be deleted since 80% > 50%
        assert "bad_column" not in cleaned.columns
        assert len(report["clean"]["null_handling"]["columns_deleted"]) == 1
        assert report["clean"]["null_handling"]["columns_deleted"][0]["column"] == "bad_column"
        assert report["clean"]["null_handling"]["columns_deleted"][0]["null_ratio"] == 0.8

    def test_numerical_imputation_with_normal_distribution(self):
        """Test that numerical columns are imputed using normal distribution"""
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 10,
                "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
                "open": [100.0] * 7 + [np.nan] * 3,  # 30% nulls
                "high": [110.0] * 10,
                "low": [90.0] * 10,
                "close": [105.0] * 10,
                "volume": [1000000] * 10,
            }
        )

        cleaned, report = clean_stock_bars(df, column_delete_threshold=0.5)

        # Column should be imputed since 30% < 50%
        assert "open" in cleaned.columns
        assert cleaned["open"].isna().sum() == 0  # No nulls should remain
        assert "open" in report["clean"]["null_handling"]["columns_imputed"]

        imputation_info = report["clean"]["null_handling"]["columns_imputed"]["open"]
        assert imputation_info["method"] == "normal_distribution"
        assert imputation_info["null_count"] == 3
        assert imputation_info["null_ratio"] == 0.3
        assert "mean" in imputation_info
        assert "std" in imputation_info
        assert imputation_info["mean"] == 100.0  # Mean of the 7 non-null values

    def test_datetime_imputation_with_unix_epoch(self):
        """Test that datetime columns are imputed with Unix epoch"""
        # Create unique rows to avoid duplicate removal affecting null count
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 10,
                "ts": [pd.Timestamp(f"2024-01-{i+1:02d}", tz="UTC") for i in range(7)] + [pd.NaT] * 3,  # 30% nulls
                "open": [100.0 + i for i in range(10)],  # Make rows unique
                "high": [110.0 + i for i in range(10)],
                "low": [90.0 + i for i in range(10)],
                "close": [105.0 + i for i in range(10)],
                "volume": [1000000 + i for i in range(10)],
            }
        )

        cleaned, report = clean_stock_bars(df, column_delete_threshold=0.5)

        # Column should be imputed with Unix epoch
        assert "ts" in cleaned.columns
        assert cleaned["ts"].isna().sum() == 0  # No nulls should remain
        assert "ts" in report["clean"]["null_handling"]["columns_imputed"]

        imputation_info = report["clean"]["null_handling"]["columns_imputed"]["ts"]
        assert imputation_info["method"] == "unix_epoch"
        assert imputation_info["null_count"] == 3

        # Verify the imputed values are Unix epoch
        # After sorting by ts, Unix epoch dates (1970-01-01) will be first
        unix_epoch = pd.Timestamp("1970-01-01", tz="UTC")
        # Count how many rows have Unix epoch timestamp
        unix_epoch_count = (cleaned["ts"] == unix_epoch).sum()
        assert unix_epoch_count == 3

    def test_categorical_imputation_with_constant(self):
        """Test that categorical/string columns are imputed with 'Unknown'"""
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 7 + [np.nan] * 3,  # 30% nulls
                "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
                "open": [100.0] * 10,
                "high": [110.0] * 10,
                "low": [90.0] * 10,
                "close": [105.0] * 10,
                "volume": [1000000] * 10,
                "category": ["tech"] * 6 + [np.nan] * 4,  # 40% nulls
            }
        )

        cleaned, report = clean_stock_bars(df, column_delete_threshold=0.5)

        # Columns should be imputed with 'Unknown'
        assert "ticker" in cleaned.columns
        assert "category" in cleaned.columns
        assert (cleaned["ticker"].iloc[7:] == "Unknown").all()
        assert (cleaned["category"].iloc[6:] == "Unknown").all()

        assert "ticker" in report["clean"]["null_handling"]["columns_imputed"]
        assert "category" in report["clean"]["null_handling"]["columns_imputed"]

        ticker_info = report["clean"]["null_handling"]["columns_imputed"]["ticker"]
        assert ticker_info["method"] == "constant"
        assert ticker_info["value"] == "Unknown"

    def test_threshold_boundary(self):
        """Test behavior at exactly the threshold value"""
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 10,
                "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
                "open": [100.0] * 10,
                "high": [110.0] * 10,
                "low": [90.0] * 10,
                "close": [105.0] * 10,
                "volume": [1000000] * 10,
                "exactly_threshold": [1.0] * 5 + [np.nan] * 5,  # Exactly 50% nulls
            }
        )

        cleaned, report = clean_stock_bars(df, column_delete_threshold=0.5)

        # At exactly 50%, should be imputed (ratio <= threshold)
        assert "exactly_threshold" in cleaned.columns
        assert "exactly_threshold" in report["clean"]["null_handling"]["columns_imputed"]

    def test_multiple_columns_mixed_behavior(self):
        """Test mixed scenario with both deletion and imputation"""
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 10,
                "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
                "open": [100.0] * 10,
                "high": [110.0] * 10,
                "low": [90.0] * 10,
                "close": [105.0] * 10,
                "volume": [1000000] * 10,
                "delete_me": [np.nan] * 9 + [1.0],  # 90% nulls - should delete
                "impute_me": [1.0] * 9 + [np.nan],  # 10% nulls - should impute
            }
        )

        cleaned, report = clean_stock_bars(df, column_delete_threshold=0.5)

        # Check deletions
        assert "delete_me" not in cleaned.columns
        assert len(report["clean"]["null_handling"]["columns_deleted"]) == 1

        # Check imputations
        assert "impute_me" in cleaned.columns
        assert cleaned["impute_me"].isna().sum() == 0
        assert "impute_me" in report["clean"]["null_handling"]["columns_imputed"]

    def test_no_nulls(self):
        """Test that clean data passes through unchanged"""
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 10,
                "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
                "open": [100.0] * 10,
                "high": [110.0] * 10,
                "low": [90.0] * 10,
                "close": [105.0] * 10,
                "volume": [1000000] * 10,
            }
        )

        cleaned, report = clean_stock_bars(df, column_delete_threshold=0.5)

        assert len(cleaned) == len(df)
        assert list(cleaned.columns) == list(df.columns)
        assert len(report["clean"]["null_handling"]["columns_deleted"]) == 0
        assert len(report["clean"]["null_handling"]["columns_imputed"]) == 0


class TestHardFilters:
    """Test suite for hard filters (non-null validation)"""

    def test_future_timestamp_filtering(self):
        """Test that future timestamps are filtered out"""
        future_date = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 10,
                "ts": [pd.Timestamp(f"2024-01-{i+1:02d}", tz="UTC") for i in range(9)] + [future_date],
                "open": [100.0 + i for i in range(10)],  # Make rows unique
                "high": [110.0 + i for i in range(10)],
                "low": [90.0 + i for i in range(10)],
                "close": [105.0 + i for i in range(10)],
                "volume": [1000000 + i for i in range(10)],
            }
        )

        cleaned, report = clean_stock_bars(df, column_delete_threshold=0.5)

        # Future timestamp row should be filtered
        assert len(cleaned) == 9
        assert report["clean"]["hard_rows_dropped"] == 1

    def test_negative_price_filtering(self):
        """Test that negative prices are filtered out"""
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 10,
                "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
                "open": [100.0] * 9 + [-10.0],  # Negative price
                "high": [110.0] * 10,
                "low": [90.0] * 10,
                "close": [105.0] * 10,
                "volume": [1000000] * 10,
            }
        )

        cleaned, report = clean_stock_bars(df, column_delete_threshold=0.5)

        assert len(cleaned) == 9
        assert report["clean"]["hard_rows_dropped"] == 1

    def test_invalid_high_low_relationship(self):
        """Test that invalid high/low relationships are filtered"""
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 10,
                "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
                "open": [100.0] * 10,
                "high": [110.0] * 9 + [80.0],  # High < open, invalid
                "low": [90.0] * 10,
                "close": [105.0] * 10,
                "volume": [1000000] * 10,
            }
        )

        cleaned, report = clean_stock_bars(df, column_delete_threshold=0.5)

        assert len(cleaned) == 9
        assert report["clean"]["hard_rows_dropped"] == 1


class TestPipelineClean:
    """Test suite for pipeline_clean function"""

    def test_pipeline_clean_with_dataframe(self):
        """Test pipeline_clean with DataFrame input"""
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 10,
                "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
                "open": [100.0] * 8 + [np.nan] * 2,
                "high": [110.0] * 10,
                "low": [90.0] * 10,
                "close": [105.0] * 10,
                "volume": [1000000] * 10,
            }
        )

        cleaned, report = pipeline_clean(df, column_delete_threshold=0.3)

        assert isinstance(cleaned, pd.DataFrame)
        assert "clean" in report
        assert "null_handling" in report["clean"]
        assert cleaned["open"].isna().sum() == 0  # Nulls should be imputed

    def test_pipeline_clean_threshold_parameter(self):
        """Test that pipeline_clean passes threshold parameter correctly"""
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 10,
                "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
                "open": [100.0] * 10,
                "high": [110.0] * 10,
                "low": [90.0] * 10,
                "close": [105.0] * 10,
                "volume": [1000000] * 10,
                "bad_col": [np.nan] * 6 + [1.0] * 4,  # 60% nulls
            }
        )

        # With threshold 0.5, column should be deleted
        cleaned1, report1 = pipeline_clean(df, column_delete_threshold=0.5)
        assert "bad_col" not in cleaned1.columns

        # With threshold 0.7, column should be imputed
        cleaned2, report2 = pipeline_clean(df, column_delete_threshold=0.7)
        assert "bad_col" in cleaned2.columns
        assert cleaned2["bad_col"].isna().sum() == 0


class TestEdgeCases:
    """Test suite for edge cases and error conditions"""

    def test_all_required_columns_deleted(self):
        """Test behavior when all required columns exceed threshold"""
        # Create unique rows to maintain null ratio after deduplication
        # Each row needs to be unique, so we use an ID column that varies
        df = pd.DataFrame(
            {
                "ticker": [np.nan] * 9 + ["AAPL"],
                "ts": [pd.NaT] * 9 + [pd.Timestamp("2024-01-01", tz="UTC")],
                "open": [np.nan] * 9 + [100.0],
                "high": [np.nan] * 9 + [110.0],
                "low": [np.nan] * 9 + [90.0],
                "close": [np.nan] * 9 + [105.0],
                "volume": [np.nan] * 9 + [1000000],
                "id": list(range(10)),  # Make rows unique
            }
        )

        cleaned, report = clean_stock_bars(df, column_delete_threshold=0.5)

        # All required columns should be deleted (90% nulls > 50% threshold)
        # Check that required columns are not present
        required_cols_present = [col for col in ["ticker", "ts", "open", "high", "low", "close", "volume"] if col in cleaned.columns]

        # Since all required columns had 90% nulls, they should be deleted
        # The function should return empty DataFrame when all required columns are deleted
        assert cleaned.empty or len(required_cols_present) == 0
        assert "warning" in report["clean"] or report["clean"]["null_handling"]["total_columns_deleted"] == 7

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        df = pd.DataFrame()

        cleaned, report = clean_stock_bars(df, column_delete_threshold=0.5)

        assert cleaned.empty

    def test_zero_std_imputation(self):
        """Test numerical imputation when std is zero"""
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 10,
                "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
                "open": [100.0] * 10,
                "high": [110.0] * 10,
                "low": [90.0] * 10,
                "close": [105.0] * 10,
                "volume": [1000000] * 10,
                "constant_col": [42.0] * 7 + [np.nan] * 3,  # All non-null values identical
            }
        )

        cleaned, report = clean_stock_bars(df, column_delete_threshold=0.5)

        # Should impute with mean when std is 0
        assert "constant_col" in cleaned.columns
        assert (cleaned["constant_col"] == 42.0).all()
        assert report["clean"]["null_handling"]["columns_imputed"]["constant_col"]["std"] == 0.0


class TestReporting:
    """Test suite for report structure and contents"""

    def test_report_structure(self):
        """Test that report has expected structure"""
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 10,
                "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
                "open": [100.0] * 8 + [np.nan] * 2,
                "high": [110.0] * 10,
                "low": [90.0] * 10,
                "close": [105.0] * 10,
                "volume": [1000000] * 10,
                "delete_me": [np.nan] * 9 + [1.0],
            }
        )

        cleaned, report = clean_stock_bars(df, column_delete_threshold=0.5)

        # Check report structure
        assert "clean" in report
        assert "null_handling" in report["clean"]
        assert "threshold" in report["clean"]["null_handling"]
        assert "columns_deleted" in report["clean"]["null_handling"]
        assert "columns_imputed" in report["clean"]["null_handling"]
        assert "exact_duplicates_dropped" in report["clean"]
        assert "hard_rows_dropped" in report["clean"]

    def test_null_handling_report_details(self):
        """Test that null handling report contains correct details"""
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"] * 10,
                "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
                "open": [100.0] * 7 + [np.nan] * 3,
                "high": [110.0] * 10,
                "low": [90.0] * 10,
                "close": [105.0] * 10,
                "volume": [1000000] * 10,
            }
        )

        cleaned, report = clean_stock_bars(df, column_delete_threshold=0.5)

        imputation_info = report["clean"]["null_handling"]["columns_imputed"]["open"]
        assert imputation_info["null_count"] == 3
        assert imputation_info["null_ratio"] == 0.3
        assert imputation_info["method"] == "normal_distribution"
        assert isinstance(imputation_info["mean"], float)
        assert isinstance(imputation_info["std"], float)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
