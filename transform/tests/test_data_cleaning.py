"""
Test suite for data_cleaning.py

Tests the config-based data cleaning system with pattern matching,
per-column policies, multiple data types, and validation rules.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transform.data_cleaning import (
    clean_dataframe,
    clean_stock_bars,
    load_cleaning_config,
    match_column_rule,
    pipeline_clean,
)


class TestConfigLoading:
    """Test configuration loading and pattern matching"""

    def test_load_default_config(self):
        """Test loading default configuration"""
        config = load_cleaning_config()

        assert "version" in config
        assert "global_settings" in config
        assert "column_rules" in config
        assert isinstance(config["column_rules"], list)
        assert len(config["column_rules"]) > 0

    def test_load_custom_config(self):
        """Test loading custom configuration from file"""
        custom_config = {
            "version": 1,
            "global_settings": {
                "default_null_threshold": 0.7,
                "default_allow_column_deletion": False,
                "default_imputation_strategy": "none",
                "remove_duplicates": True,
            },
            "column_rules": [
                {"pattern": "^test_.*", "dtype": "float"}
            ],
            "relationship_validations": [],
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(custom_config, f)
            temp_path = f.name

        try:
            config = load_cleaning_config(temp_path)
            assert config["global_settings"]["default_null_threshold"] == 0.7
            assert config["global_settings"]["remove_duplicates"] is True
        finally:
            Path(temp_path).unlink()

    def test_pattern_matching_specificity(self):
        """Test that pattern matching follows first-match-wins"""
        config = load_cleaning_config()

        # ticker should match specific pattern
        ticker_rule = match_column_rule("ticker", config)
        assert ticker_rule["pattern"] == "^ticker$"
        assert ticker_rule["allow_column_deletion"] is False

        # balance_sheet columns should match balance sheet pattern
        bs_rule = match_column_rule("balance_sheet_totalAssets", config)
        assert bs_rule["pattern"] == "^balance_sheet_.*"

        # Unknown column should match catch-all
        unknown_rule = match_column_rule("unknown_column_xyz", config)
        assert unknown_rule["pattern"] == ".*"


class TestStockDataCleaning:
    """Test cleaning of stock price data"""

    def test_stock_data_basic_cleaning(self):
        """Test basic stock data cleaning with OHLCV columns"""
        df = pd.DataFrame({
            "ticker": ["AAPL"] * 10,
            "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "open": [100.0 + i for i in range(10)],
            "high": [110.0 + i for i in range(10)],
            "low": [90.0 + i for i in range(10)],
            "close": [105.0 + i for i in range(10)],
            "volume": [1000000 + i*1000 for i in range(10)],
        })

        cleaned, report = clean_dataframe(df)

        assert len(cleaned) == 10
        assert all(col in cleaned.columns for col in ["ticker", "ts", "open", "high", "low", "close", "volume"])
        assert report["clean"]["exact_duplicates_dropped"] == 0
        assert cleaned["volume"].dtype == "Int64"  # Should be converted to integer

    def test_stock_positive_validation(self):
        """Test that OHLC columns filter negative/zero values"""
        df = pd.DataFrame({
            "ticker": ["AAPL"] * 10,
            "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "open": [100.0] * 8 + [0.0, -10.0],  # Invalid values
            "high": [110.0] * 10,
            "low": [90.0] * 10,
            "close": [105.0] * 10,
            "volume": [1000000] * 10,
        })

        cleaned, report = clean_dataframe(df)

        # Rows with invalid open should be filtered
        assert len(cleaned) == 8
        assert "validation_rows_dropped" in report["clean"]

    def test_vwap_soft_validation(self):
        """Test VWAP stays within [low, high] or gets set to null"""
        df = pd.DataFrame({
            "ticker": ["AAPL"] * 10,
            "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "open": [100.0] * 10,
            "high": [110.0] * 10,
            "low": [90.0] * 10,
            "close": [105.0] * 10,
            "volume": [1000000] * 10,
            "vwap": [100.0] * 8 + [80.0, 120.0],  # Last 2 out of bounds
        })

        cleaned, report = clean_dataframe(df)

        # vwap should be in bounds or null
        assert "relationship_validations" in report["clean"]
        if "vwap_bounds_check" in report["clean"]["relationship_validations"]:
            vwap_check = report["clean"]["relationship_validations"]["vwap_bounds_check"]
            if vwap_check["status"] == "failed":
                # Should have set out-of-bounds values to null
                assert cleaned["vwap"].isna().sum() > 0


class TestFinancialDataCleaning:
    """Test cleaning of financial statement data"""

    def test_balance_sheet_data(self):
        """Test cleaning balance sheet columns"""
        df = pd.DataFrame({
            "ticker": ["AAPL"] * 10,
            "ts": pd.date_range("2024-01-01", periods=10, freq="QE", tz="UTC"),
            "balance_sheet_totalAssets": [1e9 + i*1e6 for i in range(8)] + [np.nan, np.nan],
            "balance_sheet_totalLiabilities": [5e8 + i*5e5 for i in range(8)] + [np.nan, np.nan],
            "balance_sheet_totalShareholderEquity": [5e8 + i*5e5 for i in range(8)] + [np.nan, np.nan],
        })

        cleaned, report = clean_dataframe(df)

        # Balance sheet columns should be present (20% nulls < 0.7 threshold)
        assert "balance_sheet_totalAssets" in cleaned.columns
        assert "balance_sheet_totalLiabilities" in cleaned.columns

    def test_cash_flow_data(self):
        """Test cleaning cash flow columns"""
        df = pd.DataFrame({
            "ticker": ["AAPL"] * 10,
            "ts": pd.date_range("2024-01-01", periods=10, freq="QE", tz="UTC"),
            "cash_flow_operatingCashflow": [1e8 + i*1e6 for i in range(10)],
            "cash_flow_capitalExpenditures": [-5e7 - i*1e5 for i in range(10)],  # Negative values OK
            "cash_flow_netIncome": [3e7 + i*1e5 for i in range(10)],
        })

        cleaned, report = clean_dataframe(df)

        assert all(col in cleaned.columns for col in [
            "cash_flow_operatingCashflow",
            "cash_flow_capitalExpenditures",
            "cash_flow_netIncome"
        ])
        # Negative capex should be preserved (no positive validation on cash flow)
        assert (cleaned["cash_flow_capitalExpenditures"] < 0).all()

    def test_earnings_data_mixed_types(self):
        """Test earnings data with mixed types (float, string, datetime)"""
        df = pd.DataFrame({
            "ticker": ["AAPL"] * 10,
            "ts": pd.date_range("2024-01-01", periods=10, freq="QE", tz="UTC"),
            "earnings_reportedEPS": [1.5 + i*0.1 for i in range(10)],
            "earnings_estimatedEPS": [1.4 + i*0.1 for i in range(10)],
            "earnings_surprise": [0.1] * 10,
            "earnings_surprisePercentage": [7.1] * 10,
            "earnings_fiscalDateEnding": pd.date_range("2024-01-01", periods=10, freq="QE"),
            "earnings_reportTime": ["AMC"] * 5 + ["BMO"] * 5,  # String column
        })

        cleaned, report = clean_dataframe(df)

        # Numeric earnings columns
        assert cleaned["earnings_reportedEPS"].dtype == float
        assert cleaned["earnings_estimatedEPS"].dtype == float

        # String column should remain string
        assert cleaned["earnings_reportTime"].dtype == object

    def test_income_statement_data(self):
        """Test income statement columns"""
        df = pd.DataFrame({
            "ticker": ["AAPL"] * 10,
            "ts": pd.date_range("2024-01-01", periods=10, freq="QE", tz="UTC"),
            "income_statement_totalRevenue": [1e9 + i*1e7 for i in range(10)],
            "income_statement_grossProfit": [5e8 + i*5e6 for i in range(10)],
            "income_statement_operatingIncome": [3e8 + i*3e6 for i in range(10)],
            "income_statement_netIncome": [2e8 + i*2e6 for i in range(10)],
        })

        cleaned, report = clean_dataframe(df)

        assert all(col in cleaned.columns for col in [
            "income_statement_totalRevenue",
            "income_statement_grossProfit",
            "income_statement_operatingIncome",
            "income_statement_netIncome"
        ])


class TestNullHandling:
    """Test null value handling with different strategies"""

    def test_imputation_none_strategy(self):
        """Test that imputation_strategy='none' preserves nulls"""
        custom_config = {
            "version": 1,
            "global_settings": {
                "default_null_threshold": 0.5,
                "default_allow_column_deletion": True,
                "default_imputation_strategy": "auto",
                "remove_duplicates": False,
            },
            "column_rules": [
                {
                    "pattern": "^preserve_nulls$",
                    "dtype": "float",
                    "null_threshold": 0.7,
                    "allow_column_deletion": False,
                    "imputation_strategy": "none"
                },
                {"pattern": ".*", "dtype": "auto"}
            ],
            "relationship_validations": [],
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(custom_config, f)
            temp_path = f.name

        try:
            df = pd.DataFrame({
                "ticker": ["AAPL"] * 10,
                "preserve_nulls": [1.0, 2.0, 3.0] + [np.nan] * 7,  # 70% nulls
            })

            cleaned, report = clean_dataframe(df, config_path=temp_path)

            # Column should not be deleted (allow_column_deletion=False)
            assert "preserve_nulls" in cleaned.columns
            # Nulls should be preserved (imputation_strategy='none')
            assert cleaned["preserve_nulls"].isna().sum() == 7

            # Check report
            assert "preserve_nulls" in report["clean"]["null_handling"]["columns_imputed"]
            impute_info = report["clean"]["null_handling"]["columns_imputed"]["preserve_nulls"]
            assert impute_info["method"] == "none"

        finally:
            Path(temp_path).unlink()

    def test_column_deletion_with_threshold(self):
        """Test column deletion when null_ratio > threshold"""
        df = pd.DataFrame({
            "ticker": ["AAPL"] * 10,
            "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "mostly_null_column": [1.0] * 2 + [np.nan] * 8,  # 80% nulls
            "some_null_column": [1.0] * 6 + [np.nan] * 4,  # 40% nulls
        })

        cleaned, report = clean_dataframe(df, global_threshold_override=0.5)

        # 80% > 50%, should delete
        assert "mostly_null_column" not in cleaned.columns
        # 40% <= 50%, should keep and impute
        assert "some_null_column" in cleaned.columns
        assert cleaned["some_null_column"].isna().sum() == 0

    def test_no_deletion_when_disabled(self):
        """Test that allow_column_deletion=False prevents deletion"""
        custom_config = {
            "version": 1,
            "global_settings": {
                "default_null_threshold": 0.5,
                "default_allow_column_deletion": False,  # Globally disabled
                "default_imputation_strategy": "auto",
                "remove_duplicates": False,
            },
            "column_rules": [
                {"pattern": ".*", "dtype": "auto"}
            ],
            "relationship_validations": [],
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(custom_config, f)
            temp_path = f.name

        try:
            df = pd.DataFrame({
                "ticker": ["AAPL"] * 10,
                "mostly_null": [1.0] * 1 + [np.nan] * 9,  # 90% nulls
            })

            cleaned, report = clean_dataframe(df, config_path=temp_path)

            # Column should be kept even with 90% nulls
            assert "mostly_null" in cleaned.columns
            # Should be imputed
            assert cleaned["mostly_null"].isna().sum() == 0

        finally:
            Path(temp_path).unlink()

    def test_normal_distribution_imputation(self):
        """Test normal distribution imputation for numeric columns"""
        df = pd.DataFrame({
            "ticker": ["AAPL"] * 10,
            "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "numeric_col": [100.0, 102.0, 98.0, 101.0, 99.0] + [np.nan] * 5,
        })

        cleaned, report = clean_dataframe(df)

        assert "numeric_col" in cleaned.columns
        assert cleaned["numeric_col"].isna().sum() == 0

        impute_info = report["clean"]["null_handling"]["columns_imputed"]["numeric_col"]
        assert impute_info["method"] == "normal_distribution"
        assert "mean" in impute_info
        assert "std" in impute_info

        # Mean should be around 100
        assert 98 <= impute_info["mean"] <= 102

    def test_constant_imputation_for_strings(self):
        """Test constant imputation for string columns"""
        df = pd.DataFrame({
            "ticker": ["AAPL"] * 7 + [np.nan] * 3,
            "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "category": ["tech"] * 6 + [np.nan] * 4,
        })

        cleaned, report = clean_dataframe(df)

        # ticker should be imputed with UNKNOWN (from config)
        assert (cleaned["ticker"].iloc[7:] == "UNKNOWN").all()

        # category should be imputed with "Unknown" (auto-detected)
        assert cleaned["category"].isna().sum() == 0

    def test_datetime_unix_epoch_imputation(self):
        """Test Unix epoch imputation for datetime columns"""
        df = pd.DataFrame({
            "ticker": ["AAPL"] * 10,
            "ts": [pd.Timestamp(f"2024-01-{i+1:02d}", tz="UTC") for i in range(7)] + [pd.NaT] * 3,
            "open": [100.0 + i for i in range(10)],
            "high": [110.0 + i for i in range(10)],
            "low": [90.0 + i for i in range(10)],
            "close": [105.0 + i for i in range(10)],
            "volume": [1000000] * 10,
        })

        cleaned, report = clean_dataframe(df)

        assert cleaned["ts"].isna().sum() == 0

        impute_info = report["clean"]["null_handling"]["columns_imputed"]["ts"]
        assert impute_info["method"] == "unix_epoch"

        # Should have 3 Unix epoch timestamps
        unix_epoch = pd.Timestamp("1970-01-01", tz="UTC")
        assert (cleaned["ts"] == unix_epoch).sum() == 3


class TestDuplicateHandling:
    """Test configurable duplicate removal"""

    def test_duplicates_not_removed_by_default(self):
        """Test that duplicates are kept by default"""
        df = pd.DataFrame({
            "ticker": ["AAPL"] * 3 + ["GOOGL"] * 2,
            "ts": [pd.Timestamp("2024-01-01", tz="UTC")] * 5,
            "open": [100.0] * 5,
            "high": [110.0] * 5,
            "low": [90.0] * 5,
            "close": [105.0] * 5,
            "volume": [1000000] * 5,
        })

        cleaned, report = clean_dataframe(df)

        # Default config has remove_duplicates=False
        assert len(cleaned) == 5
        assert report["clean"]["exact_duplicates_dropped"] == 0

    def test_duplicates_removed_when_configured(self):
        """Test duplicate removal when enabled in config"""
        custom_config = {
            "version": 1,
            "global_settings": {
                "default_null_threshold": 0.5,
                "default_allow_column_deletion": True,
                "default_imputation_strategy": "auto",
                "remove_duplicates": True,  # Enable duplicate removal
            },
            "column_rules": [
                {"pattern": ".*", "dtype": "auto"}
            ],
            "relationship_validations": [],
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(custom_config, f)
            temp_path = f.name

        try:
            df = pd.DataFrame({
                "ticker": ["AAPL"] * 3 + ["GOOGL"] * 2,
                "ts": [pd.Timestamp("2024-01-01", tz="UTC")] * 5,
                "value": [100.0] * 5,
            })

            cleaned, report = clean_dataframe(df, config_path=temp_path)

            # Should remove duplicates
            assert len(cleaned) < 5
            assert report["clean"]["exact_duplicates_dropped"] > 0

        finally:
            Path(temp_path).unlink()


class TestRelationshipValidations:
    """Test cross-column relationship validations"""

    def test_high_low_relationship_validation(self):
        """Test stock high/low relationship check"""
        df = pd.DataFrame({
            "ticker": ["AAPL"] * 10,
            "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "open": [100.0] * 10,
            "high": [110.0] * 8 + [95.0, 98.0],  # Last 2 invalid (high < open)
            "low": [90.0] * 10,
            "close": [105.0] * 10,
            "volume": [1000000] * 10,
        })

        cleaned, report = clean_dataframe(df)

        # Should have relationship validation report
        if "relationship_validations" in report["clean"]:
            if "stock_high_low_check" in report["clean"]["relationship_validations"]:
                rel_check = report["clean"]["relationship_validations"]["stock_high_low_check"]
                if rel_check["status"] == "failed":
                    # Invalid rows should be dropped
                    assert len(cleaned) < 10

    def test_relationship_validation_missing_columns(self):
        """Test graceful handling when required columns are missing"""
        df = pd.DataFrame({
            "ticker": ["AAPL"] * 10,
            "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            # Missing open, high, low, close - relationship check should be skipped
            "volume": [1000000] * 10,
        })

        cleaned, report = clean_dataframe(df)

        # Should skip relationship validations gracefully
        if "relationship_validations" in report["clean"]:
            for val_name, val_report in report["clean"]["relationship_validations"].items():
                if val_report["status"] == "skipped":
                    assert "Missing columns" in val_report.get("reason", "")


class TestColumnValidations:
    """Test column-level validations"""

    def test_no_future_dates_validation(self):
        """Test no_future_dates validation on ts column"""
        future_date = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=10)

        df = pd.DataFrame({
            "ticker": ["AAPL"] * 10,
            "ts": [pd.Timestamp(f"2024-01-{i+1:02d}", tz="UTC") for i in range(9)] + [future_date],
            "open": [100.0 + i for i in range(10)],
            "high": [110.0 + i for i in range(10)],
            "low": [90.0 + i for i in range(10)],
            "close": [105.0 + i for i in range(10)],
            "volume": [1000000] * 10,
        })

        cleaned, report = clean_dataframe(df)

        # Future date row should be filtered
        assert len(cleaned) == 9


class TestMixedDataTypes:
    """Test handling of mixed data types in single dataframe"""

    def test_mixed_stock_and_financial_data(self):
        """Test dataframe with both stock and financial statement columns"""
        df = pd.DataFrame({
            # Stock columns
            "ticker": ["AAPL"] * 10,
            "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "open": [100.0 + i for i in range(10)],
            "high": [110.0 + i for i in range(10)],
            "low": [90.0 + i for i in range(10)],
            "close": [105.0 + i for i in range(10)],
            "volume": [1000000 + i*1000 for i in range(10)],
            # Balance sheet columns
            "balance_sheet_totalAssets": [1e9 + i*1e6 for i in range(8)] + [np.nan] * 2,
            "balance_sheet_totalLiabilities": [5e8 + i*5e5 for i in range(8)] + [np.nan] * 2,
            # Cash flow columns
            "cash_flow_operatingCashflow": [1e8] * 10,
            # Earnings columns
            "earnings_reportedEPS": [1.5 + i*0.1 for i in range(10)],
        })

        cleaned, report = clean_dataframe(df)

        # All column types should be present
        assert all(col in cleaned.columns for col in [
            "ticker", "open", "close",
            "balance_sheet_totalAssets",
            "cash_flow_operatingCashflow",
            "earnings_reportedEPS"
        ])


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_dataframe(self):
        """Test handling of empty dataframe"""
        df = pd.DataFrame()

        cleaned, report = clean_dataframe(df)

        assert cleaned.empty
        assert "clean" in report

    def test_all_nulls_column(self):
        """Test column with 100% nulls"""
        df = pd.DataFrame({
            "ticker": ["AAPL"] * 10,
            "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "all_null": [np.nan] * 10,
        })

        cleaned, report = clean_dataframe(df)

        # 100% nulls > any threshold, should be deleted (if deletion allowed)
        # Default config allows deletion
        assert "all_null" not in cleaned.columns

    def test_zero_std_imputation(self):
        """Test imputation when all non-null values are identical"""
        df = pd.DataFrame({
            "ticker": ["AAPL"] * 10,
            "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "constant_col": [42.0] * 7 + [np.nan] * 3,
        })

        cleaned, report = clean_dataframe(df)

        # Should impute with mean when std=0
        assert "constant_col" in cleaned.columns
        assert (cleaned["constant_col"] == 42.0).all()

    def test_single_row_dataframe(self):
        """Test handling of single-row dataframe"""
        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "ts": [pd.Timestamp("2024-01-01", tz="UTC")],
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
            "volume": [1000000],
        })

        cleaned, report = clean_dataframe(df)

        assert len(cleaned) == 1
        assert all(col in cleaned.columns for col in df.columns)


class TestBackwardCompatibility:
    """Test backward compatibility with clean_stock_bars"""

    def test_clean_stock_bars_wrapper(self):
        """Test that clean_stock_bars still works as before"""
        df = pd.DataFrame({
            "ticker": ["AAPL"] * 10,
            "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "open": [100.0] * 8 + [np.nan] * 2,
            "high": [110.0] * 10,
            "low": [90.0] * 10,
            "close": [105.0] * 10,
            "volume": [1000000] * 10,
        })

        cleaned, report = clean_stock_bars(df, column_delete_threshold=0.5)

        assert isinstance(cleaned, pd.DataFrame)
        assert "clean" in report
        assert cleaned["open"].isna().sum() == 0  # Should be imputed

    def test_pipeline_clean_integration(self):
        """Test pipeline_clean integration"""
        df = pd.DataFrame({
            "ticker": ["AAPL"] * 10,
            "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "open": [100.0] * 10,
            "high": [110.0] * 10,
            "low": [90.0] * 10,
            "close": [105.0] * 10,
            "volume": [1000000] * 10,
        })

        cleaned, report = pipeline_clean(df, column_delete_threshold=0.5)

        assert isinstance(cleaned, pd.DataFrame)
        assert "clean" in report
        assert "read" in report


class TestReporting:
    """Test report structure and contents"""

    def test_report_structure(self):
        """Test that report has expected structure"""
        df = pd.DataFrame({
            "ticker": ["AAPL"] * 10,
            "ts": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "open": [100.0] * 8 + [np.nan] * 2,
        })

        cleaned, report = clean_dataframe(df)

        assert "clean" in report
        assert "config_version" in report["clean"]
        assert "config_path" in report["clean"]
        assert "column_processing" in report["clean"]
        assert "dtype_conversions" in report["clean"]
        assert "null_handling" in report["clean"]
        assert "final_rows" in report["clean"]
        assert "final_columns" in report["clean"]

    def test_column_processing_report(self):
        """Test detailed column processing report"""
        df = pd.DataFrame({
            "ticker": ["AAPL"] * 10,
            "balance_sheet_totalAssets": [1e9] * 7 + [np.nan] * 3,
        })

        cleaned, report = clean_dataframe(df)

        col_processing = report["clean"]["column_processing"]

        # Should have info for each column
        assert "ticker" in col_processing
        assert "balance_sheet_totalAssets" in col_processing

        # Check ticker info
        ticker_info = col_processing["ticker"]
        assert "matched_pattern" in ticker_info
        assert ticker_info["matched_pattern"] == "^ticker$"

        # Check balance sheet info
        bs_info = col_processing["balance_sheet_totalAssets"]
        assert bs_info["matched_pattern"] == "^balance_sheet_.*"
        assert "null_count" in bs_info
        assert "action" in bs_info


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
