"""
Test suite for dsl_validator.py

Tests DSL validation logic including JSON parsing, feature validation,
parameter validation, type checking, and custom feature handling.
"""

import json
import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transform.dsl_validator import validate_dsl


# Sample registry for testing
SAMPLE_REGISTRY = {
    "features": {
        "sma": {
            "params": {
                "window": {"type": "int", "required": True},
                "on": {"type": "string", "required": True, "allowed": ["open", "close", "high", "low"]},
                "as": {"type": "string", "required": False, "default": "sma"},
            }
        },
        "ema": {
            "params": {
                "window": {"type": "int", "required": True},
                "on": {"type": "string", "required": True},
                "as": {"type": "string", "required": True},
            }
        },
        "rsi": {
            "params": {
                "window": {"type": "int", "required": False, "default": 14},
                "as": {"type": "string", "required": False, "default": "rsi"},
            }
        },
    }
}


class TestValidDSL:
    """Test suite for valid DSL inputs"""

    def test_valid_simple_dsl(self):
        """Test validation of a simple valid DSL"""
        dsl_string = json.dumps({
            "features": [
                {
                    "name": "sma",
                    "params": {
                        "window": 20,
                        "on": "close",
                        "as": "sma_20"
                    }
                }
            ]
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        assert result is not None
        assert len(errors) == 0
        assert result["features"][0]["name"] == "sma"
        assert result["features"][0]["params"]["window"] == 20

    def test_multiple_features(self):
        """Test validation with multiple features"""
        dsl_string = json.dumps({
            "features": [
                {
                    "name": "sma",
                    "params": {"window": 20, "on": "close", "as": "sma_20"}
                },
                {
                    "name": "ema",
                    "params": {"window": 12, "on": "close", "as": "ema_12"}
                },
                {
                    "name": "rsi",
                    "params": {"as": "rsi_14"}
                }
            ]
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        assert result is not None
        assert len(errors) == 0
        assert len(result["features"]) == 3

    def test_default_values_applied(self):
        """Test that default values are applied for missing parameters"""
        dsl_string = json.dumps({
            "features": [
                {
                    "name": "rsi",
                    "params": {}  # Missing window and as, should use defaults
                }
            ]
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        assert result is not None
        assert len(errors) == 0
        assert result["features"][0]["params"]["window"] == 14  # Default
        assert result["features"][0]["params"]["as"] == "rsi"  # Default

    def test_partial_defaults(self):
        """Test that defaults are applied only for missing params"""
        dsl_string = json.dumps({
            "features": [
                {
                    "name": "rsi",
                    "params": {"window": 21}  # Provide window, use default for as
                }
            ]
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        assert result is not None
        assert len(errors) == 0
        assert result["features"][0]["params"]["window"] == 21  # User-provided
        assert result["features"][0]["params"]["as"] == "rsi"  # Default


class TestInvalidJSON:
    """Test suite for invalid JSON inputs"""

    def test_malformed_json(self):
        """Test handling of malformed JSON"""
        dsl_string = "{ this is not valid json }"

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        assert result is None
        assert len(errors) == 1
        assert "not valid JSON" in errors[0]

    def test_missing_features_key(self):
        """Test handling of missing 'features' key"""
        dsl_string = json.dumps({
            "wrong_key": []
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        assert result is None
        assert len(errors) == 1
        assert "features" in errors[0]

    def test_features_not_list(self):
        """Test handling when 'features' is not a list"""
        dsl_string = json.dumps({
            "features": "not a list"
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        assert result is None
        assert len(errors) == 1


class TestFeatureValidation:
    """Test suite for feature validation"""

    def test_unsupported_feature(self):
        """Test error when feature is not in registry"""
        dsl_string = json.dumps({
            "features": [
                {
                    "name": "nonexistent_feature",
                    "params": {}
                }
            ]
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        assert result is None
        assert len(errors) == 1
        assert "Not a supported feature" in errors[0]

    def test_missing_required_parameter(self):
        """Test error when required parameter is missing"""
        dsl_string = json.dumps({
            "features": [
                {
                    "name": "ema",
                    "params": {
                        "window": 12
                        # Missing required 'on' and 'as' parameters
                    }
                }
            ]
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        assert result is None
        assert len(errors) >= 1
        assert any("Required parameter" in err for err in errors)

    def test_unsupported_parameter(self):
        """Test error when parameter is not in registry"""
        dsl_string = json.dumps({
            "features": [
                {
                    "name": "sma",
                    "params": {
                        "window": 20,
                        "on": "close",
                        "as": "sma_20",
                        "invalid_param": 123  # Not in registry
                    }
                }
            ]
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        assert result is None
        assert len(errors) == 1
        assert "not supported" in errors[0]


class TestTypeValidation:
    """Test suite for type validation"""

    def test_string_type_validation(self):
        """Test string type validation"""
        dsl_string = json.dumps({
            "features": [
                {
                    "name": "sma",
                    "params": {
                        "window": 20,
                        "on": 123,  # Should be string
                        "as": "sma_20"
                    }
                }
            ]
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        assert result is None
        assert len(errors) == 1
        assert "must be a string" in errors[0]

    def test_int_type_validation(self):
        """Test integer type validation"""
        dsl_string = json.dumps({
            "features": [
                {
                    "name": "sma",
                    "params": {
                        "window": "twenty",  # Should be int
                        "on": "close",
                        "as": "sma_20"
                    }
                }
            ]
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        assert result is None
        assert len(errors) == 1
        assert "must be an integer" in errors[0]


class TestAllowedValues:
    """Test suite for allowed values validation"""

    def test_valid_allowed_value(self):
        """Test validation with valid allowed value"""
        dsl_string = json.dumps({
            "features": [
                {
                    "name": "sma",
                    "params": {
                        "window": 20,
                        "on": "close",  # Valid: in allowed list
                        "as": "sma_20"
                    }
                }
            ]
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        assert result is not None
        assert len(errors) == 0

    def test_invalid_allowed_value(self):
        """Test error when value is not in allowed list"""
        dsl_string = json.dumps({
            "features": [
                {
                    "name": "sma",
                    "params": {
                        "window": 20,
                        "on": "volume",  # Invalid: not in allowed list
                        "as": "sma_20"
                    }
                }
            ]
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        assert result is None
        assert len(errors) == 1
        assert "must be one of" in errors[0]


class TestCustomFeatures:
    """Test suite for custom feature validation"""

    def test_valid_custom_feature(self):
        """Test validation of valid custom feature"""
        dsl_string = json.dumps({
            "features": [
                {
                    "name": "custom_my_feature",
                    "params": {
                        "code": "df['result'] = df['close'] * 2",
                        "as": "doubled_close"
                    }
                }
            ]
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        assert result is not None
        assert len(errors) == 0
        assert result["features"][0]["name"] == "custom_my_feature"

    def test_custom_feature_missing_code(self):
        """Test error when custom feature is missing 'code' parameter"""
        dsl_string = json.dumps({
            "features": [
                {
                    "name": "custom_my_feature",
                    "params": {
                        "as": "result"
                        # Missing 'code' parameter
                    }
                }
            ]
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        assert result is None
        assert len(errors) >= 1
        assert any("Missing required 'code'" in err for err in errors)

    def test_custom_feature_missing_as(self):
        """Test error when custom feature is missing 'as' parameter"""
        dsl_string = json.dumps({
            "features": [
                {
                    "name": "custom_my_feature",
                    "params": {
                        "code": "df['result'] = df['close'] * 2"
                        # Missing 'as' parameter
                    }
                }
            ]
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        assert result is None
        assert len(errors) >= 1
        assert any("Missing required 'as'" in err for err in errors)

    def test_custom_feature_invalid_syntax(self):
        """Test error when custom feature has invalid Python syntax"""
        dsl_string = json.dumps({
            "features": [
                {
                    "name": "custom_my_feature",
                    "params": {
                        "code": "this is not valid python syntax +++",
                        "as": "result"
                    }
                }
            ]
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        assert result is None
        assert len(errors) >= 1
        assert any("Invalid Python syntax" in err for err in errors)

    def test_custom_feature_valid_complex_code(self):
        """Test validation with complex but valid Python code"""
        dsl_string = json.dumps({
            "features": [
                {
                    "name": "custom_complex",
                    "params": {
                        "code": """
import numpy as np
df['result'] = df['close'].rolling(20).mean()
df['result'] = df['result'].fillna(0)
""",
                        "as": "custom_sma"
                    }
                }
            ]
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        assert result is not None
        assert len(errors) == 0


class TestMultipleErrors:
    """Test suite for handling multiple errors"""

    def test_multiple_validation_errors(self):
        """Test that multiple errors are collected"""
        dsl_string = json.dumps({
            "features": [
                {
                    "name": "sma",
                    "params": {
                        "window": "not_an_int",  # Error 1: wrong type
                        "on": "volume",  # Error 2: not in allowed values
                    }
                    # Error 3: missing required 'as' parameter (no default)
                }
            ]
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        assert result is None
        # Should have at least 2 errors (type and allowed values)
        assert len(errors) >= 2

    def test_errors_from_multiple_features(self):
        """Test collecting errors from multiple features"""
        dsl_string = json.dumps({
            "features": [
                {
                    "name": "nonexistent1",
                    "params": {}
                },
                {
                    "name": "nonexistent2",
                    "params": {}
                }
            ]
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        assert result is None
        assert len(errors) == 2


class TestEdgeCases:
    """Test suite for edge cases"""

    def test_empty_features_list(self):
        """Test validation with empty features list"""
        dsl_string = json.dumps({
            "features": []
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        # Empty features list is valid
        assert result is not None
        assert len(errors) == 0
        assert result["features"] == []

    def test_feature_without_params(self):
        """Test feature with missing params key"""
        dsl_string = json.dumps({
            "features": [
                {
                    "name": "rsi"
                    # Missing 'params' key
                }
            ]
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        # Should apply all defaults
        assert result is not None
        assert len(errors) == 0
        assert result["features"][0]["params"]["window"] == 14

    def test_empty_registry(self):
        """Test validation with empty registry"""
        dsl_string = json.dumps({
            "features": [
                {
                    "name": "any_feature",
                    "params": {}
                }
            ]
        })

        empty_registry = {"features": {}}
        result, errors = validate_dsl(dsl_string, empty_registry)

        assert result is None
        assert len(errors) == 1
        assert "Not a supported feature" in errors[0]


class TestErrorMessages:
    """Test suite for error message quality"""

    def test_error_includes_feature_index(self):
        """Test that errors include feature index for clarity"""
        dsl_string = json.dumps({
            "features": [
                {"name": "sma", "params": {"window": 20, "on": "close", "as": "sma"}},
                {"name": "nonexistent", "params": {}},  # This is feature index 1
            ]
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        assert result is None
        assert len(errors) == 1
        assert "Feature 1" in errors[0]

    def test_error_includes_feature_name(self):
        """Test that errors include feature name when available"""
        dsl_string = json.dumps({
            "features": [
                {
                    "name": "sma",
                    "params": {
                        "window": "bad_type",
                        "on": "close",
                        "as": "sma"
                    }
                }
            ]
        })

        result, errors = validate_dsl(dsl_string, SAMPLE_REGISTRY)

        assert result is None
        assert len(errors) == 1
        assert "'sma'" in errors[0]


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
