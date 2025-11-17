from __future__ import annotations

import logging
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from pandera import Check, Column
from pandera import pandas as pa

# Optional: Configure logging to see reports in your console/notebook output
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)

REQUIRED_COLS = ["ticker", "ts", "open", "high", "low", "close", "volume"]


def clean_stock_bars(
    df: pd.DataFrame, column_delete_threshold: float = 0.5
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    No-exception cleaner:
     - keeps tz-aware UTC in 'ts'
     - drops exact duplicate rows only
     - handles null values based on column_delete_threshold:
       * if column null_ratio > threshold: delete entire column
       * if column null_ratio <= threshold: impute missing values
     - hard-drops invalid rows (future ts, negative/zero prices, bad high/low, negative volume)
     - soft-fixes vwap (set NaN if out of [low, high]) and transactions (nullable Int64, <0 -> NA)
     - returns (clean_df, report)

    Args:
        df: Input DataFrame to clean
        column_delete_threshold: Ratio threshold for column deletion (default 0.5).
                                 Columns with null_ratio > threshold are deleted.
                                 Columns with null_ratio <= threshold get imputed.
    """
    rep: Dict[str, Any] = {"clean": {}}
    d = df.copy()

    # Ensure required columns exist
    missing = [c for c in REQUIRED_COLS if c not in d.columns]
    if missing:
        rep["clean"]["missing_required_columns"] = missing

    # Parse/normalize types - only for columns that exist
    if "ts" in d.columns:
        d["ts"] = pd.to_datetime(d["ts"], errors="coerce", utc=True)

    for c in ["open", "high", "low", "close", "volume"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    if "vwap" in d.columns:
        d["vwap"] = pd.to_numeric(d["vwap"], errors="coerce")
    if "transactions" in d.columns:
        d["transactions"] = pd.to_numeric(d["transactions"], errors="coerce").astype(
            "Int64"
        )

    # Drop exact duplicates only
    before = len(d)
    d = d.drop_duplicates(keep="first")
    rep["clean"]["exact_duplicates_dropped"] = int(before - len(d))

    # Handle null values based on column_delete_threshold
    rep["clean"]["null_handling"] = {
        "threshold": column_delete_threshold,
        "columns_deleted": [],
        "columns_imputed": {},
    }

    total_rows = len(d)
    columns_to_delete = []

    for col in d.columns:
        null_count = d[col].isna().sum()
        null_ratio = null_count / total_rows if total_rows > 0 else 0

        if null_ratio > column_delete_threshold:
            # Delete column if null ratio exceeds threshold
            columns_to_delete.append(col)
            rep["clean"]["null_handling"]["columns_deleted"].append(
                {
                    "column": col,
                    "null_ratio": float(null_ratio),
                    "null_count": int(null_count),
                }
            )
        elif null_count > 0:
            # Impute missing values if null ratio is within threshold
            imputation_info = {
                "null_count": int(null_count),
                "null_ratio": float(null_ratio),
                "method": None,
            }

            # Determine column type and impute accordingly
            if pd.api.types.is_numeric_dtype(
                d[col]
            ) and not pd.api.types.is_datetime64_any_dtype(d[col]):
                # Numerical column: use normal distribution
                non_null_values = d[col].dropna()
                if len(non_null_values) > 0:
                    mean_val = non_null_values.mean()
                    std_val = non_null_values.std()
                    if pd.isna(std_val) or std_val == 0:
                        # If std is 0 or NaN, just use the mean
                        imputed_values = np.full(null_count, mean_val)
                    else:
                        # Sample from normal distribution
                        imputed_values = np.random.normal(mean_val, std_val, null_count)
                    d.loc[d[col].isna(), col] = imputed_values
                    imputation_info["method"] = "normal_distribution"
                    imputation_info["mean"] = float(mean_val)
                    imputation_info["std"] = (
                        float(std_val) if not pd.isna(std_val) else 0.0
                    )

            elif pd.api.types.is_datetime64_any_dtype(d[col]):
                # Datetime column: use Unix epoch
                unix_epoch = pd.Timestamp(
                    "1970-01-01", tz=d[col].dt.tz if hasattr(d[col].dt, "tz") else None
                )
                d.loc[d[col].isna(), col] = unix_epoch
                imputation_info["method"] = "unix_epoch"
                imputation_info["value"] = str(unix_epoch)

            else:
                # Categorical/string column: use constant "Unknown"
                d.loc[d[col].isna(), col] = "Unknown"
                imputation_info["method"] = "constant"
                imputation_info["value"] = "Unknown"

            rep["clean"]["null_handling"]["columns_imputed"][col] = imputation_info

    # Delete columns that exceed threshold
    if columns_to_delete:
        d = d.drop(columns=columns_to_delete)
        rep["clean"]["null_handling"]["total_columns_deleted"] = len(columns_to_delete)

    # Hard filters (note: nulls are already handled above via imputation/column deletion)
    # Only check columns that still exist after potential column deletions
    existing_required = [c for c in REQUIRED_COLS if c in d.columns]
    deleted_required = [c for c in REQUIRED_COLS if c in columns_to_delete]

    if deleted_required:
        # Warn if any required columns were deleted
        if "warnings" not in rep["clean"]:
            rep["clean"]["warnings"] = {}
        rep["clean"]["warnings"]["deleted_required_columns"] = (
            f"Some columns labeled as required were deleted due to high null ratio: {deleted_required}"
        )

    # Build keep_mask based on which columns still exist
    keep_mask = pd.Series([True] * len(d), index=d.index)

    if "ts" in d.columns:
        now_utc = pd.Timestamp.now(tz="UTC")
        keep_mask &= d["ts"] <= now_utc

    if "open" in d.columns:
        keep_mask &= d["open"] > 0

    if "high" in d.columns:
        keep_mask &= d["high"] > 0

    if "low" in d.columns:
        keep_mask &= d["low"] > 0

    if "close" in d.columns:
        keep_mask &= d["close"] > 0

    if "volume" in d.columns:
        keep_mask &= d["volume"] >= 0

    # Check high/low relationships if all relevant columns exist
    if all(c in d.columns for c in ["open", "close", "high", "low"]):
        oc_max = d[["open", "close"]].max(axis=1)
        oc_min = d[["open", "close"]].min(axis=1)
        keep_mask &= (d["high"] >= oc_max) & (d["low"] <= oc_min)

    rep["clean"]["hard_rows_dropped"] = int((~keep_mask).sum())
    d = d.loc[keep_mask].copy()

    # Soft fixes
    if "vwap" in d.columns and "low" in d.columns and "high" in d.columns:
        bad_vwap = (~d["vwap"].isna()) & (
            (d["vwap"] < d["low"]) | (d["vwap"] > d["high"])
        )
        rep["clean"]["vwap_set_null"] = int(bad_vwap.sum())
        d.loc[bad_vwap, "vwap"] = np.nan

    if "transactions" in d.columns:
        bad_tx = d["transactions"].notna() & (d["transactions"] < 0)
        rep["clean"]["transactions_set_null"] = int(bad_tx.sum())
        d.loc[bad_tx, "transactions"] = pd.NA
        d["transactions"] = d["transactions"].astype("Int64")

    # Optional tidy & dtype align - only sort by columns that exist
    sort_cols = [c for c in ["ticker", "ts"] if c in d.columns]
    if sort_cols:
        d = d.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    else:
        d = d.reset_index(drop=True)

    return d, rep


def get_default_schema_config() -> Dict[str, Any]:
    """Returns the default validation schema configuration for stock bars."""

    # Vectorized checks for better performance
    is_tz_aware = Check(
        lambda s: pd.api.types.is_datetime64_any_dtype(s) and s.dt.tz is not None,
        element_wise=False,
        error="ts must be a timezone-aware Series.",
    )
    is_utc = Check(
        lambda s: str(s.dt.tz) in ("UTC", "utc"),
        element_wise=False,
        error="ts must be UTC.",
    )
    is_not_future = Check(
        lambda s: (s <= pd.Timestamp.now(tz="UTC")).all(),
        element_wise=False,
        error="Found future timestamps.",
    )

    schema_config = {
        "columns": {
            "ticker": Column(pa.String, checks=Check.str_length(min_value=1)),
            # THIS IS THE FIX: Directly specify the pandas dtype string for a timezone-aware column.
            "ts": Column(
                dtype="datetime64[ns, UTC]", checks=[is_tz_aware, is_utc, is_not_future]
            ),
            "open": Column(
                pa.Float, checks=[Check.gt(0), Check(lambda s: np.isfinite(s))]
            ),
            "high": Column(
                pa.Float, checks=[Check.gt(0), Check(lambda s: np.isfinite(s))]
            ),
            "low": Column(
                pa.Float, checks=[Check.gt(0), Check(lambda s: np.isfinite(s))]
            ),
            "close": Column(
                pa.Float, checks=[Check.gt(0), Check(lambda s: np.isfinite(s))]
            ),
            "volume": Column(pa.Int, checks=Check.ge(0)),
            "vwap": Column(pa.Float, nullable=True, required=False),
            "transactions": Column(
                pa.Int, nullable=True, checks=Check.ge(0), required=False
            ),
        },
        "dataframe_checks": [
            Check(
                lambda df: (df["high"] >= df[["open", "close"]].max(axis=1)).all(),
                error="high < max(open, close).",
            ),
            Check(
                lambda df: (df["low"] <= df[["open", "close"]].min(axis=1)).all(),
                error="low > min(open, close).",
            ),
            Check(
                lambda df: ("vwap" not in df.columns)
                or (
                    (df["vwap"].isna())
                    | ((df["vwap"] >= df["low"]) & (df["vwap"] <= df["high"]))
                ).all(),
                error="vwap must be within [low, high] when present.",
            ),
        ],
    }
    return schema_config


def pandera_validate_safely(
    df: pd.DataFrame, schema_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Validate using a flexible schema. Never raises."""
    info: Dict[str, Any] = {"pandera": {"enabled": False, "status": "skipped"}}

    if schema_config is None:
        schema_config = get_default_schema_config()

    columns_to_validate = {
        col_name: col_schema
        for col_name, col_schema in schema_config["columns"].items()
        if col_name in df.columns
    }

    schema = pa.DataFrameSchema(
        columns=columns_to_validate,
        checks=schema_config.get("dataframe_checks", []),
        coerce=False,
        strict=False,
        unique=None,
    )

    _v = df.copy()

    try:
        schema.validate(_v, lazy=True)
        info["pandera"] = {"enabled": True, "status": "passed"}
    except Exception as e:
        info["pandera"] = {
            "enabled": True,
            "status": f"failed: {type(e).__name__}",
            "details": str(e),
        }
    return info


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

    # Clean data
    logger.debug("data_cleaning: calling clean_stock_bars")
    cleaned, clean_rep = clean_stock_bars(df, column_delete_threshold)
    logger.debug(
        f"data_cleaning: finished clean_stock_bars. resulting report {clean_rep}. dataframe cleaned {cleaned}."
    )
    report.update(clean_rep)

    # Validate with Pandera
    logger.debug("data_cleaning: calling pandera_validate_safely")
    pandera_rep = pandera_validate_safely(cleaned)
    logger.debug(
        f"data_cleaning: finished pandera_validate_safely. resulting report {pandera_rep}. dataframe cleaned {cleaned}."
    )
    report.update(pandera_rep)

    # Log the final report for easy viewing
    # logger.info(f"Pipeline Report: {report}")

    return cleaned, report


# clean_df, report = pipeline_clean("data_cleaning_dirty.csv")
# print(clean_df)
# print(report)
