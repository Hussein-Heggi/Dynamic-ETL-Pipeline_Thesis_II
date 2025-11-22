from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional: Configure logging to see reports in your console/notebook output
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)

# Legacy constant for backward compatibility
REQUIRED_COLS = ["ticker", "ts", "open", "high", "low", "close", "volume"]


# --- Configuration-Based Cleaning Helper Functions ---


def load_cleaning_config(config_path: str | Path = Path(__file__).parent / "cleaning_config.json") -> Dict[str, Any]:
    """
    Load cleaning configuration from JSON file.

    Args:
        config_path: Path to config JSON file.

    Returns:
        Dictionary containing configuration
    """
    # Handle None case - use default path
    if config_path is None:
        config_path = Path(__file__).parent / "cleaning_config.json"

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        logger.debug(f"Loaded cleaning config from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}, using minimal default config")
        # Return minimal default configuration
        return {
            "version": 1,
            "global_settings": {
                "default_null_threshold": 0.5,
                "default_allow_column_deletion": True,
                "default_imputation_strategy": "auto",
                "remove_duplicates": False,
            },
            "column_rules": [
                {
                    "pattern": ".*",
                    "dtype": "auto",
                    "null_threshold": 0.5,
                    "imputation_strategy": "auto",
                    "allow_column_deletion": True,
                    "validations": [],
                }
            ],
            "relationship_validations": [],
        }
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse config file: {e}")
        raise


def match_column_rule(column_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find the first matching rule for a column based on regex patterns.

    Args:
        column_name: Name of the column to match
        config: Cleaning configuration dictionary

    Returns:
        Dictionary containing the matched rule with defaults applied
    """
    global_settings = config.get("global_settings", {})
    column_rules = config.get("column_rules", [])

    # Find first matching pattern
    for rule in column_rules:
        pattern = rule.get("pattern", "")
        if re.match(pattern, column_name):
            # Merge with global defaults
            matched_rule = {
                "pattern": pattern,
                "dtype": rule.get("dtype", "auto"),
                "null_threshold": rule.get(
                    "null_threshold", global_settings.get("default_null_threshold", 0.5)
                ),
                "allow_column_deletion": rule.get(
                    "allow_column_deletion",
                    global_settings.get("default_allow_column_deletion", True),
                ),
                "imputation_strategy": rule.get(
                    "imputation_strategy",
                    global_settings.get("default_imputation_strategy", "auto"),
                ),
                "imputation_value": rule.get("imputation_value", None),
                "validations": rule.get("validations", []),
            }
            logger.debug(f"Column '{column_name}' matched pattern '{pattern}'")
            return matched_rule

    # Fallback to defaults if no pattern matched (shouldn't happen with catch-all)
    logger.warning(f"No pattern matched for column '{column_name}', using defaults")
    return {
        "pattern": "default",
        "dtype": "auto",
        "null_threshold": global_settings.get("default_null_threshold", 0.5),
        "allow_column_deletion": global_settings.get("default_allow_column_deletion", True),
        "imputation_strategy": global_settings.get("default_imputation_strategy", "auto"),
        "imputation_value": None,
        "validations": [],
    }


def apply_dtype_conversion(
    df: pd.DataFrame, column: str, dtype: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Convert column to specified dtype.

    Args:
        df: DataFrame to modify
        column: Column name
        dtype: Target dtype (string, float, int, datetime, auto)

    Returns:
        Tuple of (modified df, conversion report)
    """
    report = {"column": column, "target_dtype": dtype, "status": "success"}

    if dtype == "auto":
        # Skip conversion, keep existing dtype
        report["actual_dtype"] = str(df[column].dtype)
        return df, report

    try:
        if dtype == "datetime":
            df[column] = pd.to_datetime(df[column], errors="coerce", utc=True)
        elif dtype == "float":
            df[column] = pd.to_numeric(df[column], errors="coerce").astype(float)
        elif dtype == "int":
            # Convert to float first to handle NaN, then to nullable Int64
            df[column] = pd.to_numeric(df[column], errors="coerce")
        elif dtype == "string":
            # Use pandas StringDtype to preserve null values (not convert NaN to "nan")
            df[column] = df[column].astype("string")
        else:
            logger.warning(f"Unknown dtype '{dtype}' for column '{column}', skipping conversion")
            report["status"] = "skipped"
            report["reason"] = f"unknown dtype: {dtype}"

        report["actual_dtype"] = str(df[column].dtype)
    except Exception as e:
        logger.error(f"Failed to convert column '{column}' to {dtype}: {e}")
        report["status"] = "failed"
        report["error"] = str(e)

    return df, report


def apply_column_validations(
    df: pd.DataFrame, column: str, validations: list[str], report: dict[str, Any]
) -> tuple[pd.DataFrame, int]:
    """
    Apply validation rules to a column and filter rows.

    Args:
        df: DataFrame to validate
        column: Column name
        validations: List of validation rule names
        report: Report dictionary to update

    Returns:
        Tuple of (filtered df, number of rows dropped)
    """
    if not validations or column not in df.columns:
        return df, 0

    keep_mask = pd.Series([True] * len(df), index=df.index)
    rows_before = len(df)

    for validation in validations:
        if validation == "positive":
            keep_mask &= df[column] > 0
        elif validation == "non_negative":
            keep_mask &= df[column] >= 0
        elif validation == "no_future_dates":
            if pd.api.types.is_datetime64_any_dtype(df[column]):
                now_utc = pd.Timestamp.now(tz="UTC")
                keep_mask &= df[column] <= now_utc
        else:
            logger.warning(f"Unknown validation '{validation}' for column '{column}'")

    df_filtered = df.loc[keep_mask].copy()
    rows_dropped = rows_before - len(df_filtered)

    if rows_dropped > 0:
        if "validation_rows_dropped" not in report:
            report["validation_rows_dropped"] = {}
        report["validation_rows_dropped"][column] = {
            "validations": validations,
            "rows_dropped": int(rows_dropped),
        }

    return df_filtered, rows_dropped


def impute_column_values(
    df: pd.DataFrame, column: str, rule: dict[str, Any], null_count: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Impute missing values in a column based on strategy.

    Args:
        df: DataFrame to modify
        column: Column name
        rule: Column rule dictionary with imputation strategy
        null_count: Number of null values

    Returns:
        Tuple of (modified df, imputation report)
    """
    imputation_info = {
        "column": column,
        "null_count": int(null_count),
        "method": None,
    }

    strategy = rule.get("imputation_strategy", "auto")

    # Handle "none" strategy - skip imputation entirely
    if strategy == "none":
        imputation_info["method"] = "none"
        imputation_info["reason"] = "Imputation disabled by configuration"
        return df, imputation_info

    # Auto-detect strategy based on dtype
    if strategy == "auto":
        if pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_datetime64_any_dtype(df[column]):
            strategy = "normal_distribution"
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            strategy = "unix_epoch"
        else:
            strategy = "constant"

    try:
        if strategy == "normal_distribution":
            non_null_values = df[column].dropna()
            if len(non_null_values) > 0:
                mean_val = non_null_values.mean()
                std_val = non_null_values.std()
                if pd.isna(std_val) or std_val == 0:
                    imputed_values = np.full(null_count, mean_val)
                else:
                    imputed_values = np.random.normal(mean_val, std_val, null_count)
                df.loc[df[column].isna(), column] = imputed_values
                imputation_info["method"] = "normal_distribution"
                imputation_info["mean"] = float(mean_val)
                imputation_info["std"] = float(std_val) if not pd.isna(std_val) else 0.0

        elif strategy == "unix_epoch":
            unix_epoch = pd.Timestamp(
                "1970-01-01", tz=df[column].dt.tz if hasattr(df[column].dt, "tz") else None
            )
            df.loc[df[column].isna(), column] = unix_epoch
            imputation_info["method"] = "unix_epoch"
            imputation_info["value"] = str(unix_epoch)

        elif strategy == "constant":
            constant_val = rule.get("imputation_value")
            if constant_val is None:
                constant_val = "Unknown"
            df.loc[df[column].isna(), column] = constant_val
            imputation_info["method"] = "constant"
            imputation_info["value"] = constant_val

        else:
            logger.warning(f"Unknown imputation strategy '{strategy}' for column '{column}'")
            imputation_info["method"] = "skipped"
            imputation_info["reason"] = f"unknown strategy: {strategy}"

    except Exception as e:
        logger.error(f"Imputation failed for column '{column}': {e}")
        imputation_info["method"] = "failed"
        imputation_info["error"] = str(e)

    return df, imputation_info


def clean_dataframe(
    df: pd.DataFrame,
    config_path: str | None = None,
    global_threshold_override: float | None = None,
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    """
    Config-based flexible data cleaner that works with any data type.

    Features:
     - Pattern-based column rules from JSON configuration
     - Per-column dtype conversion, null handling, and validation
     - Configurable duplicate removal
     - Relationship validations (optional, with graceful failures)
     - Returns (clean_df, detailed_report)

    Args:
        df: Input DataFrame to clean
        config_path: Path to cleaning config JSON. If None, uses default config.
        global_threshold_override: Override global null threshold from config

    Returns:
        Tuple of (cleaned DataFrame, cleaning report dictionary)
    """
    # Load configuration
    config = load_cleaning_config(config_path)
    global_settings = config.get("global_settings", {})

    # Override threshold if provided
    if global_threshold_override is not None:
        global_settings["default_null_threshold"] = global_threshold_override

    # Initialize report
    rep: Dict[str, Any] = {
        "clean": {
            "config_version": config.get("version", 1),
            "config_path": str(config_path) if config_path else "default",
        }
    }

    d = df.copy()
    total_rows = len(d)

    # 1. Duplicate Removal (optional)
    if global_settings.get("remove_duplicates", False):
        before = len(d)
        d = d.drop_duplicates(keep="first")
        duplicates_dropped = before - len(d)
        rep["clean"]["exact_duplicates_dropped"] = int(duplicates_dropped)
        logger.debug(f"Dropped {duplicates_dropped} exact duplicate rows")
    else:
        rep["clean"]["exact_duplicates_dropped"] = 0

    # 2. Per-Column Processing
    rep["clean"]["column_processing"] = {}
    rep["clean"]["dtype_conversions"] = []
    rep["clean"]["null_handling"] = {"columns_deleted": [], "columns_imputed": {}}

    columns_to_delete = []

    for col in d.columns:
        # Match column rule
        rule = match_column_rule(col, config)

        col_report = {
            "column": col,
            "matched_pattern": rule["pattern"],
            "target_dtype": rule["dtype"],
        }

        # 2a. Dtype Conversion
        d, dtype_report = apply_dtype_conversion(d, col, rule["dtype"])
        rep["clean"]["dtype_conversions"].append(dtype_report)

        # 2b. Null Handling
        null_count = d[col].isna().sum()
        null_ratio = null_count / total_rows if total_rows > 0 else 0

        col_report["null_count"] = int(null_count)
        col_report["null_ratio"] = float(null_ratio)

        threshold = rule["null_threshold"]
        allow_deletion = rule["allow_column_deletion"]

        if null_ratio > threshold and allow_deletion:
            # Mark for deletion
            columns_to_delete.append(col)
            rep["clean"]["null_handling"]["columns_deleted"].append(
                {
                    "column": col,
                    "null_ratio": float(null_ratio),
                    "null_count": int(null_count),
                    "threshold": float(threshold),
                }
            )
            col_report["action"] = "deleted"

        elif null_count > 0:
            # Impute missing values
            d, imputation_info = impute_column_values(d, col, rule, null_count)
            imputation_info["null_ratio"] = float(null_ratio)
            imputation_info["threshold"] = float(threshold)
            rep["clean"]["null_handling"]["columns_imputed"][col] = imputation_info
            col_report["action"] = "imputed"
        else:
            col_report["action"] = "none_needed"

        # 2c. Column Validations (filters rows)
        validations = rule.get("validations", [])
        if validations:
            d, rows_dropped = apply_column_validations(d, col, validations, rep["clean"])
            if rows_dropped > 0:
                col_report["validation_rows_dropped"] = rows_dropped

        rep["clean"]["column_processing"][col] = col_report

    # Delete columns marked for deletion
    if columns_to_delete:
        d = d.drop(columns=columns_to_delete)
        rep["clean"]["null_handling"]["total_columns_deleted"] = len(columns_to_delete)
        logger.debug(f"Deleted {len(columns_to_delete)} columns due to null ratio")

    # 3. Final dtype adjustments for integer columns
    for col in d.columns:
        rule = match_column_rule(col, config)
        if rule["dtype"] == "int" and col in d.columns:
            # Convert to nullable Int64
            d[col] = d[col].round().astype("Int64")

    # 4. Relationship Validations (optional cross-column checks)
    d, rel_validation_report = validate_relationships(d, config)
    if rel_validation_report:
        rep["clean"]["relationship_validations"] = rel_validation_report

    # 5. Sort and tidy
    sort_cols = [c for c in ["ticker", "ts"] if c in d.columns]
    if sort_cols:
        d = d.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    else:
        d = d.reset_index(drop=True)

    rep["clean"]["final_rows"] = len(d)
    rep["clean"]["final_columns"] = list(d.columns)

    return d, rep


def validate_relationships(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply relationship validations defined in config.

    Args:
        df: DataFrame to validate
        config: Configuration dictionary

    Returns:
        Tuple of (modified df, validation report)
    """
    report = {}
    rel_validations = config.get("relationship_validations", [])

    for validation in rel_validations:
        name = validation.get("name", "unnamed")
        required_cols = validation.get("required_columns", [])
        check_type = validation.get("check_type", "")
        action_on_failure = validation.get("action_on_failure", "drop_rows")

        val_report = {
            "name": name,
            "description": validation.get("description", ""),
            "required_columns": required_cols,
            "status": "skipped",
        }

        # Check if all required columns exist
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            val_report["status"] = "skipped"
            val_report["reason"] = f"Missing columns: {missing_cols}"
            val_report["level"] = "warning"
            logger.warning(f"Relationship validation '{name}' skipped: missing {missing_cols}")
            report[name] = val_report
            continue

        # Apply the check
        try:
            if check_type == "high_low_relationship":
                # Stock-specific: high >= max(open,close), low <= min(open,close)
                oc_max = df[["open", "close"]].max(axis=1)
                oc_min = df[["open", "close"]].min(axis=1)
                valid_mask = (df["high"] >= oc_max) & (df["low"] <= oc_min)
                failed_rows = (~valid_mask).sum()

                if failed_rows > 0:
                    val_report["status"] = "failed"
                    val_report["failed_rows"] = int(failed_rows)
                    val_report["level"] = "error"
                    logger.error(f"Relationship validation '{name}' failed for {failed_rows} rows")

                    if action_on_failure == "drop_rows":
                        df = df.loc[valid_mask].copy()
                        val_report["action_taken"] = "dropped_rows"
                else:
                    val_report["status"] = "passed"

            elif check_type == "vwap_in_range":
                # VWAP should be within [low, high]
                vwap_col = "vwap"
                if vwap_col in df.columns:
                    bad_vwap = (~df[vwap_col].isna()) & (
                        (df[vwap_col] < df["low"]) | (df[vwap_col] > df["high"])
                    )
                    failed_count = bad_vwap.sum()

                    if failed_count > 0:
                        val_report["status"] = "failed"
                        val_report["failed_rows"] = int(failed_count)
                        val_report["level"] = "error"
                        logger.error(f"Relationship validation '{name}' failed for {failed_count} rows")

                        if action_on_failure == "set_null":
                            df.loc[bad_vwap, vwap_col] = np.nan
                            val_report["action_taken"] = "set_to_null"
                    else:
                        val_report["status"] = "passed"

            else:
                val_report["status"] = "skipped"
                val_report["reason"] = f"Unknown check type: {check_type}"
                logger.warning(f"Unknown relationship check type '{check_type}' for validation '{name}'")

        except Exception as e:
            val_report["status"] = "error"
            val_report["error"] = str(e)
            val_report["level"] = "error"
            logger.error(f"Relationship validation '{name}' encountered error: {e}")

        report[name] = val_report

    return df, report


def clean_stock_bars(
    df: pd.DataFrame, column_delete_threshold: float = 0.5
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Backward-compatible wrapper around clean_dataframe for stock data.

    This function maintains the original API for legacy code while using
    the new config-based cleaning system.

    Args:
        df: Input DataFrame to clean
        column_delete_threshold: Ratio threshold for column deletion (default 0.5)

    Returns:
        Tuple of (cleaned DataFrame, cleaning report)
    """
    logger.debug("clean_stock_bars called, delegating to clean_dataframe with config")
    return clean_dataframe(
        df, config_path=None, global_threshold_override=column_delete_threshold
    )


def pipeline_clean(
    data: Union[str, pd.DataFrame],
    column_delete_threshold: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Accepts CSV path or DataFrame.
    Returns (cleaned_df, report). Never raises.

    Args:
        data: CSV file path or pandas DataFrame to clean
        column_delete_threshold: Ratio threshold for column deletion (default 0.5).
                                 Columns with null_ratio > threshold are deleted.
                                 Columns with null_ratio <= threshold get imputed.
    """
    report: Dict[str, Any] = {}

    # Read data
    if isinstance(data, str):
        try:
            df = pd.read_csv(
                data, skipinitialspace=True, engine="python", on_bad_lines="skip"
            )
            report["read"] = {
                "path": data,
                "rows": len(df),
                "columns": list(df.columns),
                "on_bad_lines": "skip",
            }
        except Exception as e:
            report["read"] = {"path": data, "error": str(e)}
            return pd.DataFrame(), report
    else:
        df = data.copy()
        report["read"] = {"path": None, "rows": len(df), "columns": list(df.columns)}

    # Clean data using config-based system
    logger.debug("data_cleaning: calling clean_dataframe via clean_stock_bars wrapper")
    cleaned, clean_rep = clean_stock_bars(df, column_delete_threshold)
    logger.debug(
        f"data_cleaning: finished cleaning. resulting report {clean_rep}. dataframe cleaned {cleaned}."
    )
    report.update(clean_rep)

    # Log the final report for easy viewing
    # logger.info(f"Pipeline Report: {report}")

    return cleaned, report


# clean_df, report = pipeline_clean("data_cleaning_dirty.csv")
# print(clean_df)
# print(report)
