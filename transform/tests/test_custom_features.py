"""
Test suite for custom feature execution with RestrictedPython.

Tests both functionality and security of custom features in the enrichment module.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transform.dsl_validator import validate_dsl
from transform.enrichment import apply_features


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    df = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 50 + ["GOOGL"] * 50,
            "ts": list(dates[:50]) + list(dates[:50]),
            "open": np.random.uniform(100, 200, 100),
            "high": np.random.uniform(100, 200, 100),
            "low": np.random.uniform(100, 200, 100),
            "close": np.random.uniform(100, 200, 100),
            "volume": np.random.uniform(1000000, 5000000, 100),
        }
    )

    # Ensure high is the highest and low is the lowest
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

    return df


@pytest.fixture
def registry():
    """Load the feature registry"""
    registry_path = Path(__file__).parent.parent / "registry.yaml"
    with open(registry_path, "r") as f:
        return yaml.safe_load(f)


class TestCustomFeatureBasics:
    """Test basic custom feature functionality"""

    def test_simple_single_line_custom_feature(self, sample_df, registry):
        """Test simple single-line custom feature (price-to-open ratio)"""
        dsl_string = json.dumps(
            {
                "features": [
                    {
                        "name": "custom_price_ratio",
                        "params": {
                            "code": "series = g['close'] / g['open']",
                            "as": "close_open_ratio",
                        },
                    }
                ]
            }
        )

        dsl, errors = validate_dsl(dsl_string, registry)

        assert not errors, f"Validation failed: {errors}"

        df_result = apply_features(sample_df.copy(), dsl, registry)

        assert "close_open_ratio" in df_result.columns
        assert df_result["close_open_ratio"].notna().any()
        # Verify the calculation is correct for first row
        expected_ratio = sample_df.iloc[0]["close"] / sample_df.iloc[0]["open"]
        assert abs(df_result.iloc[0]["close_open_ratio"] - expected_ratio) < 1e-10

    def test_multiline_custom_feature(self, sample_df, registry):
        """Test multi-line custom feature (normalized momentum)"""
        dsl_string = json.dumps(
            {
                "features": [
                    {
                        "name": "custom_norm_momentum",
                        "params": {
                            "code": "momentum = g['close'] - g['close'].shift(5)\nrolling_std = g['close'].rolling(10).std()\nseries = momentum / rolling_std",
                            "as": "normalized_momentum_5_10",
                        },
                    }
                ]
            }
        )

        dsl, errors = validate_dsl(dsl_string, registry)

        assert not errors, f"Validation failed: {errors}"

        df_result = apply_features(sample_df.copy(), dsl, registry)

        assert "normalized_momentum_5_10" in df_result.columns

    def test_mix_of_standard_and_custom_features(self, sample_df, registry):
        """Test mixing standard and custom features"""
        dsl_string = json.dumps(
            {
                "features": [
                    {"name": "sma", "params": {"on": "close", "window": 10}},
                    {
                        "name": "custom_midpoint",
                        "params": {
                            "code": "series = (g['high'] + g['low']) / 2",
                            "as": "hl_midpoint",
                        },
                    },
                    {"name": "rsi", "params": {"on": "close", "window": 14}},
                ]
            }
        )

        dsl, errors = validate_dsl(dsl_string, registry)

        assert not errors, f"Validation failed: {errors}"

        df_result = apply_features(sample_df.copy(), dsl, registry)

        assert "sma_close_10" in df_result.columns
        assert "hl_midpoint" in df_result.columns
        assert "rsi_close_14" in df_result.columns


class TestCustomFeatureValidation:
    """Test custom feature validation"""

    def test_invalid_syntax_caught_by_validator(self, registry):
        """Test that syntax errors are caught during validation"""
        dsl_string = json.dumps(
            {
                "features": [
                    {
                        "name": "custom_bad_syntax",
                        "params": {
                            "code": "series = g['close'] / ",  # Syntax error
                            "as": "bad_feature",
                        },
                    }
                ]
            }
        )

        dsl, errors = validate_dsl(dsl_string, registry)

        assert errors, "Expected validation to fail but it passed"
        assert any("Invalid Python syntax" in err for err in errors)

    def test_missing_series_variable_fails_at_runtime(self, sample_df, registry):
        """Test that missing 'series' assignment fails at runtime"""
        dsl_string = json.dumps(
            {
                "features": [
                    {
                        "name": "custom_no_series",
                        "params": {
                            "code": "result = g['close'] / g['open']",  # Wrong variable name
                            "as": "wrong_var",
                        },
                    }
                ]
            }
        )

        dsl, errors = validate_dsl(dsl_string, registry)

        assert not errors, f"Validation failed unexpectedly: {errors}"

        # Should fail at runtime due to missing 'series' variable
        with pytest.raises(ValueError):
            apply_features(sample_df.copy(), dsl, registry)


class TestSecurityRestrictions:
    """Test that RestrictedPython blocks malicious operations"""

    def test_file_access_blocked(self, sample_df, registry):
        """Test that file access is blocked"""
        dsl_string = json.dumps(
            {
                "features": [
                    {
                        "name": "custom_file_access",
                        "params": {
                            "code": "with open('/etc/passwd', 'r') as f:\n    content = f.read()\nseries = g['close']",
                            "as": "malicious",
                        },
                    }
                ]
            }
        )

        dsl, errors = validate_dsl(dsl_string, registry)
        assert not errors, f"Validation failed: {errors}"

        # Should fail at runtime due to security restrictions
        with pytest.raises((RuntimeError, NameError)):
            apply_features(sample_df.copy(), dsl, registry)

    def test_import_os_blocked(self, sample_df, registry):
        """Test that importing os is blocked"""
        dsl_string = json.dumps(
            {
                "features": [
                    {
                        "name": "custom_import_os",
                        "params": {
                            "code": "import os\nseries = g['close']",
                            "as": "malicious",
                        },
                    }
                ]
            }
        )

        dsl, errors = validate_dsl(dsl_string, registry)
        assert not errors, f"Validation failed: {errors}"

        with pytest.raises((RuntimeError, ImportError)):
            apply_features(sample_df.copy(), dsl, registry)

    def test_eval_blocked(self, sample_df, registry):
        """Test that eval() is blocked"""
        dsl_string = json.dumps(
            {
                "features": [
                    {
                        "name": "custom_eval",
                        "params": {
                            "code": "series = eval('g[\\'close\\']')",
                            "as": "malicious",
                        },
                    }
                ]
            }
        )

        dsl, errors = validate_dsl(dsl_string, registry)
        assert not errors, f"Validation failed: {errors}"

        with pytest.raises((RuntimeError, NameError)):
            apply_features(sample_df.copy(), dsl, registry)

    def test_exec_blocked(self, sample_df, registry):
        """Test that exec() is blocked"""
        dsl_string = json.dumps(
            {
                "features": [
                    {
                        "name": "custom_exec",
                        "params": {
                            "code": "exec('series = g[\\'close\\']')",
                            "as": "malicious",
                        },
                    }
                ]
            }
        )

        dsl, errors = validate_dsl(dsl_string, registry)
        assert not errors, f"Validation failed: {errors}"

        with pytest.raises((RuntimeError, NameError)):
            apply_features(sample_df.copy(), dsl, registry)

    def test_directory_listing_blocked(self, sample_df, registry):
        """Test that directory listing is blocked"""
        dsl_string = json.dumps(
            {
                "features": [
                    {
                        "name": "custom_listdir",
                        "params": {
                            "code": "import os\nfiles = os.listdir('.')\nseries = g['close']",
                            "as": "malicious",
                        },
                    }
                ]
            }
        )

        dsl, errors = validate_dsl(dsl_string, registry)
        assert not errors, f"Validation failed: {errors}"

        with pytest.raises((RuntimeError, ImportError, NameError)):
            apply_features(sample_df.copy(), dsl, registry)

    def test_builtins_access_blocked(self, sample_df, registry):
        """Test that access to __builtins__ is blocked"""
        dsl_string = json.dumps(
            {
                "features": [
                    {
                        "name": "custom_builtins",
                        "params": {
                            "code": "b = __builtins__['open']\nseries = g['close']",
                            "as": "malicious",
                        },
                    }
                ]
            }
        )

        dsl, errors = validate_dsl(dsl_string, registry)
        assert not errors, f"Validation failed: {errors}"

        with pytest.raises((RuntimeError, TypeError, KeyError, NameError)):
            apply_features(sample_df.copy(), dsl, registry)


class TestCustomFeatureEdgeCases:
    """Test edge cases and special scenarios"""

    def test_custom_feature_with_numpy_operations(self, sample_df, registry):
        """Test custom feature using numpy operations"""
        dsl_string = json.dumps(
            {
                "features": [
                    {
                        "name": "custom_log_returns",
                        "params": {
                            "code": "series = np.log(g['close'] / g['close'].shift(1))",
                            "as": "log_returns",
                        },
                    }
                ]
            }
        )

        dsl, errors = validate_dsl(dsl_string, registry)
        assert not errors, f"Validation failed: {errors}"

        df_result = apply_features(sample_df.copy(), dsl, registry)

        assert "log_returns" in df_result.columns
        # Some values should be NaN due to shift (at least one per ticker group)
        assert df_result["log_returns"].isna().sum() > 0

    def test_custom_feature_with_conditional_logic(self, sample_df, registry):
        """Test custom feature with conditional logic"""
        dsl_string = json.dumps(
            {
                "features": [
                    {
                        "name": "custom_trend",
                        "params": {
                            "code": "series = g['close'].apply(lambda x: 1 if x > g['close'].mean() else 0)",
                            "as": "above_avg",
                        },
                    }
                ]
            }
        )

        dsl, errors = validate_dsl(dsl_string, registry)
        assert not errors, f"Validation failed: {errors}"

        df_result = apply_features(sample_df.copy(), dsl, registry)

        assert "above_avg" in df_result.columns
        # Values should be 0 or 1
        assert set(df_result["above_avg"].unique()).issubset({0, 1, np.nan})

    def test_multiple_custom_features_in_sequence(self, sample_df, registry):
        """Test applying multiple custom features in sequence"""
        dsl_string = json.dumps(
            {
                "features": [
                    {
                        "name": "custom_range",
                        "params": {
                            "code": "series = g['high'] - g['low']",
                            "as": "price_range",
                        },
                    },
                    {
                        "name": "custom_range_pct",
                        "params": {
                            "code": "series = (g['high'] - g['low']) / g['close'] * 100",
                            "as": "range_pct",
                        },
                    },
                ]
            }
        )

        dsl, errors = validate_dsl(dsl_string, registry)
        assert not errors, f"Validation failed: {errors}"

        df_result = apply_features(sample_df.copy(), dsl, registry)

        assert "price_range" in df_result.columns
        assert "range_pct" in df_result.columns


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
