import logging
import math
import random
import traceback

import numpy as np
import pandas as pd
import yaml
from RestrictedPython import limited_builtins, safe_globals
from RestrictedPython.Guards import safe_builtins

from .dsl_validator import validate_dsl
from .llm_translator import get_llm_recipe

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)


# --- 1. Helper Functions ---
def _get_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _execute_custom_code(code: str, group_df: pd.DataFrame) -> pd.Series:
    """
    Executes custom user code in a restricted environment using RestrictedPython.

    Args:
        code: Python code string that should assign result to 'series'
        group_df: DataFrame for one ticker group

    Returns:
        pd.Series: The computed series

    Raises:
        ValueError: If code compilation fails or 'series' is not assigned
        RuntimeError: If code execution fails
    """
    bytecode = compile(code, "<inline>", "exec")

    # Create restricted globals using safe_globals (includes all necessary guards)
    restricted_globals = safe_globals.copy()

    # Merge safe_builtins and limited_builtins for more complete builtin support
    merged_builtins = {**safe_builtins, **limited_builtins}
    restricted_globals["__builtins__"] = merged_builtins

    # Add allowed modules and input data
    restricted_globals.update(
        {
            "np": np,
            "pd": pd,
            "math": math,
            "random": random,
            # Input data
            "g": group_df,
        }
    )

    # Execute the code
    try:
        exec(bytecode, restricted_globals)
    except Exception as e:
        # Get full traceback for debugging
        tb_str = traceback.format_exc()
        raise RuntimeError(
            f"Custom code execution failed: {str(e)}\n\nFull traceback:\n{tb_str}"
        ) from e

    # Retrieve the 'series' variable
    if "series" not in restricted_globals:
        raise ValueError(
            "Custom code must assign result to variable 'series'. "
            "Example: series = g['close'] / g['open']"
        )

    result = restricted_globals["series"]

    # Validate that result is a pandas Series
    if not isinstance(result, pd.Series):
        raise ValueError(
            f"Custom code must assign a pandas Series to 'series', "
            f"but got {type(result).__name__}"
        )

    return result


# --- 2. Feature Implementations (Templates) ---
# 📈 Trend Indicators
def feat_sma(g: pd.DataFrame, on: str, window: int) -> pd.Series:
    return g[on].rolling(window, min_periods=window).mean()


def feat_ema(g: pd.DataFrame, on: str, window: int) -> pd.Series:
    return g[on].ewm(span=window, adjust=False, min_periods=window).mean()


def feat_macd(
    g: pd.DataFrame, on: str, fast_period: int, slow_period: int, signal_period: int
) -> pd.DataFrame:
    ema_fast = g[on].ewm(span=fast_period, adjust=False).mean()
    ema_slow = g[on].ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    # Return a DataFrame instead of a dict
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": macd_line - signal_line}
    )


# 🏃 Momentum Indicators
def feat_rsi(g: pd.DataFrame, on: str, window: int) -> pd.Series:
    delta = g[on].diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def feat_stoch(
    g: pd.DataFrame, high: str, low: str, close: str, k_window: int, d_window: int
) -> pd.DataFrame:
    low_k = g[low].rolling(k_window).min()
    high_k = g[high].rolling(k_window).max()
    k_line = 100 * ((g[close] - low_k) / (high_k - low_k).replace(0, np.nan))
    d_line = k_line.rolling(d_window).mean()
    # Return a DataFrame instead of a dict
    return pd.DataFrame({"stoch_k": k_line, "stoch_d": d_line})


# 🌊 Volatility Indicators
def feat_rolling_vol(g: pd.DataFrame, on: str, window: int) -> pd.Series:
    return g[on].rolling(window, min_periods=window).std()


def feat_atr(
    g: pd.DataFrame, high: str, low: str, close: str, window: int
) -> pd.Series:
    true_range = _get_true_range(g[high], g[low], g[close])
    return true_range.ewm(span=window, adjust=False).mean()


def feat_bbands(g: pd.DataFrame, on: str, window: int, std_dev: int) -> pd.DataFrame:
    middle_band = g[on].rolling(window).mean()
    std = g[on].rolling(window).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    # Return a DataFrame instead of a dict
    return pd.DataFrame(
        {
            "bband_upper": upper_band,
            "bband_middle": middle_band,
            "bband_lower": lower_band,
        }
    )


#  Volume Indicators
def feat_obv(g: pd.DataFrame, close: str, volume: str) -> pd.Series:
    signed_vol = g[volume] * np.sign(g[close].diff()).fillna(0)
    return signed_vol.cumsum()


#  Basic Transformations & Statistics
def feat_ret(g: pd.DataFrame, on: str, periods: int, method: str) -> pd.Series:
    if method == "log":
        return np.log(g[on] / g[on].shift(periods))
    return g[on].pct_change(periods)


def feat_lag(g: pd.DataFrame, on: str, periods: int) -> pd.Series:
    return g[on].shift(periods)


def feat_diff(g: pd.DataFrame, on: str, periods: int) -> pd.Series:
    return g[on].diff(periods)


def feat_rolling_max(g: pd.DataFrame, on: str, window: int) -> pd.Series:
    return g[on].rolling(window).max()


def feat_rolling_min(g: pd.DataFrame, on: str, window: int) -> pd.Series:
    return g[on].rolling(window).min()


def feat_zscore(g: pd.DataFrame, on: str, window: int) -> pd.Series:
    rolling_mean = g[on].rolling(window).mean()
    rolling_std = g[on].rolling(window).std()
    return (g[on] - rolling_mean) / rolling_std.replace(0, np.nan)


#  Calendar Features
def feat_session_flags(g: pd.DataFrame) -> pd.DataFrame:
    if "ts" not in g.columns:
        raise ValueError(
            "feat_session_flags requires 'ts' column to be present in the DataFrame"
        )

    ts = g["ts"]
    # Return a DataFrame instead of a dict
    return pd.DataFrame(
        {
            "dow": ts.dt.dayofweek,
            "month": ts.dt.month,
            "week": ts.dt.isocalendar().week,
            "hour": ts.dt.hour,
            "is_month_start": ts.dt.is_month_start.astype("int8"),
            "is_month_end": ts.dt.is_month_end.astype("int8"),
        }
    )


# --- 3. Generic Helper Functions (Work on Any Column) ---
def feat_yoy_growth(g: pd.DataFrame, on: str, periods: int = 4) -> pd.Series:
    """Year-over-year growth (default periods=4 for quarterly data)"""
    return g[on].pct_change(periods)


def feat_qoq_growth(g: pd.DataFrame, on: str) -> pd.Series:
    """Quarter-over-quarter growth"""
    return g[on].pct_change(1)


def feat_rolling_avg(g: pd.DataFrame, on: str, window: int) -> pd.Series:
    """Rolling average over N periods"""
    return g[on].rolling(window, min_periods=1).mean()


def feat_pct_change(g: pd.DataFrame, on: str, periods: int) -> pd.Series:
    """Percentage change over N periods"""
    return g[on].pct_change(periods)


# --- 4. Balance Sheet Enrichments ---
def feat_current_ratio(g: pd.DataFrame) -> pd.Series:
    """Current Ratio = Current Assets / Current Liabilities"""
    return g["balance_sheet_totalCurrentAssets"] / g[
        "balance_sheet_totalCurrentLiabilities"
    ].replace(0, np.nan)


def feat_quick_ratio(g: pd.DataFrame) -> pd.Series:
    """Quick Ratio = (Current Assets - Inventory) / Current Liabilities"""
    return (
        g["balance_sheet_totalCurrentAssets"] - g["balance_sheet_inventory"]
    ) / g["balance_sheet_totalCurrentLiabilities"].replace(0, np.nan)


def feat_debt_to_equity(g: pd.DataFrame) -> pd.Series:
    """Debt-to-Equity Ratio = Total Debt / Total Shareholder Equity"""
    total_debt = g["balance_sheet_longTermDebt"].fillna(0) + g[
        "balance_sheet_shortTermDebt"
    ].fillna(0)
    return total_debt / g["balance_sheet_totalShareholderEquity"].replace(0, np.nan)


def feat_debt_to_assets(g: pd.DataFrame) -> pd.Series:
    """Debt-to-Assets Ratio = Total Debt / Total Assets"""
    total_debt = g["balance_sheet_longTermDebt"].fillna(0) + g[
        "balance_sheet_shortTermDebt"
    ].fillna(0)
    return total_debt / g["balance_sheet_totalAssets"].replace(0, np.nan)


def feat_working_capital(g: pd.DataFrame) -> pd.Series:
    """Working Capital = Current Assets - Current Liabilities"""
    return (
        g["balance_sheet_totalCurrentAssets"]
        - g["balance_sheet_totalCurrentLiabilities"]
    )


def feat_equity_ratio(g: pd.DataFrame) -> pd.Series:
    """Equity Ratio = Total Shareholder Equity / Total Assets"""
    return g["balance_sheet_totalShareholderEquity"] / g[
        "balance_sheet_totalAssets"
    ].replace(0, np.nan)


# --- 5. Cash Flow Enrichments ---
def feat_free_cash_flow(g: pd.DataFrame) -> pd.Series:
    """Free Cash Flow = Operating Cash Flow - Capital Expenditures"""
    return g["cash_flow_operatingCashflow"] - g["cash_flow_capitalExpenditures"].fillna(
        0
    )


def feat_operating_cash_margin(g: pd.DataFrame) -> pd.Series:
    """Operating Cash Margin = Operating Cash Flow / Net Income"""
    return g["cash_flow_operatingCashflow"] / g["cash_flow_netIncome"].replace(
        0, np.nan
    )


def feat_capex_intensity(g: pd.DataFrame) -> pd.Series:
    """CapEx Intensity = Capital Expenditures / Operating Cash Flow"""
    return g["cash_flow_capitalExpenditures"] / g[
        "cash_flow_operatingCashflow"
    ].replace(0, np.nan)


def feat_dividend_payout_ratio(g: pd.DataFrame) -> pd.Series:
    """Dividend Payout Ratio = Dividends / Operating Cash Flow"""
    return g["cash_flow_dividendPayout"] / g["cash_flow_operatingCashflow"].replace(
        0, np.nan
    )


def feat_cash_conversion_ratio(g: pd.DataFrame) -> pd.Series:
    """Cash Conversion Ratio = Operating Cash Flow / Net Income"""
    return g["cash_flow_operatingCashflow"] / g["cash_flow_netIncome"].replace(
        0, np.nan
    )


# --- 6. Earnings Enrichments ---
def feat_earnings_beat(g: pd.DataFrame) -> pd.Series:
    """Earnings Beat = 1 if reported EPS > estimated EPS, else 0"""
    return (g["earnings_reportedEPS"] > g["earnings_estimatedEPS"]).astype(int)


def feat_avg_surprise(g: pd.DataFrame, window: int) -> pd.Series:
    """Rolling Average of Surprise Percentage"""
    return g["earnings_surprisePercentage"].rolling(window, min_periods=1).mean()


def feat_earnings_momentum(g: pd.DataFrame, window: int) -> pd.Series:
    """Earnings Momentum = Rolling mean of surprise values"""
    return g["earnings_surprise"].rolling(window, min_periods=1).mean()


def feat_forecast_accuracy(g: pd.DataFrame) -> pd.Series:
    """Forecast Accuracy = Absolute difference between estimated and reported EPS"""
    return abs(g["earnings_estimatedEPS"] - g["earnings_reportedEPS"])


# --- 7. Income Statement Enrichments ---
def feat_gross_margin(g: pd.DataFrame) -> pd.Series:
    """Gross Margin = Gross Profit / Total Revenue"""
    return g["income_statement_grossProfit"] / g[
        "income_statement_totalRevenue"
    ].replace(0, np.nan)


def feat_operating_margin(g: pd.DataFrame) -> pd.Series:
    """Operating Margin = Operating Income / Total Revenue"""
    return g["income_statement_operatingIncome"] / g[
        "income_statement_totalRevenue"
    ].replace(0, np.nan)


def feat_net_margin(g: pd.DataFrame) -> pd.Series:
    """Net Margin = Net Income / Total Revenue"""
    return g["income_statement_netIncome"] / g["income_statement_totalRevenue"].replace(
        0, np.nan
    )


def feat_ebitda_margin(g: pd.DataFrame) -> pd.Series:
    """EBITDA Margin = EBITDA / Total Revenue"""
    return g["income_statement_ebitda"] / g["income_statement_totalRevenue"].replace(
        0, np.nan
    )


def feat_rd_intensity(g: pd.DataFrame) -> pd.Series:
    """R&D Intensity = Research & Development / Total Revenue"""
    return g["income_statement_researchAndDevelopment"] / g[
        "income_statement_totalRevenue"
    ].replace(0, np.nan)


def feat_interest_coverage(g: pd.DataFrame) -> pd.Series:
    """Interest Coverage = EBIT / Interest Expense"""
    return g["income_statement_ebit"] / g["income_statement_interestExpense"].replace(
        0, np.nan
    )


# dispatcher
FEATURE_IMPLEMENTATIONS = {
    # Stock data features
    "sma": feat_sma,
    "ema": feat_ema,
    "macd": feat_macd,
    "rsi": feat_rsi,
    "stoch": feat_stoch,
    "rolling_vol": feat_rolling_vol,
    "atr": feat_atr,
    "bbands": feat_bbands,
    "obv": feat_obv,
    "ret": feat_ret,
    "lag": feat_lag,
    "diff": feat_diff,
    "rolling_max": feat_rolling_max,
    "rolling_min": feat_rolling_min,
    "zscore": feat_zscore,
    "session_flags": feat_session_flags,
    # Generic helpers
    "yoy_growth": feat_yoy_growth,
    "qoq_growth": feat_qoq_growth,
    "rolling_avg": feat_rolling_avg,
    "pct_change": feat_pct_change,
    # Balance sheet enrichments
    "current_ratio": feat_current_ratio,
    "quick_ratio": feat_quick_ratio,
    "debt_to_equity": feat_debt_to_equity,
    "debt_to_assets": feat_debt_to_assets,
    "working_capital": feat_working_capital,
    "equity_ratio": feat_equity_ratio,
    # Cash flow enrichments
    "free_cash_flow": feat_free_cash_flow,
    "operating_cash_margin": feat_operating_cash_margin,
    "capex_intensity": feat_capex_intensity,
    "dividend_payout_ratio": feat_dividend_payout_ratio,
    "cash_conversion_ratio": feat_cash_conversion_ratio,
    # Earnings enrichments
    "earnings_beat": feat_earnings_beat,
    "avg_surprise": feat_avg_surprise,
    "earnings_momentum": feat_earnings_momentum,
    "forecast_accuracy": feat_forecast_accuracy,
    # Income statement enrichments
    "gross_margin": feat_gross_margin,
    "operating_margin": feat_operating_margin,
    "net_margin": feat_net_margin,
    "ebitda_margin": feat_ebitda_margin,
    "rd_intensity": feat_rd_intensity,
    "interest_coverage": feat_interest_coverage,
}


#  The Main Executor
def apply_features(df: pd.DataFrame, dsl: dict, registry: dict) -> pd.DataFrame:
    """
    Applies features to a DataFrame based on a validated DSL recipe.
    Note: DSL should be validated and enriched with defaults before calling this function.
    """
    # Check if ticker and ts columns are available for grouping/sorting
    has_ticker = "ticker" in df.columns
    has_ts = "ts" in df.columns

    # Log warning if expected columns are missing
    if not has_ticker:
        logger.warning(
            "'ticker' column not found in DataFrame. Features will be applied without grouping by ticker."
        )
    if not has_ts:
        logger.warning(
            "'ts' column not found in DataFrame. Features will not be sorted by timestamp."
        )

    # Sort only by available columns
    sort_cols = [c for c in ["ticker", "ts"] if c in df.columns]
    if sort_cols:
        df_enriched = df.sort_values(sort_cols).copy()
    else:
        df_enriched = df.copy()

    all_new_cols = []

    for idx, request in enumerate(dsl.get("features", [])):
        name: str = request["name"]
        final_params = request.get("params", {})

        logger.debug(f"Processing feature {idx+1}/{len(dsl.get('features', []))}: '{name}' with params: {final_params}")

        try:
            # Check if this is a custom feature
            if name.startswith("custom_"):
                # Execute custom code for each group
                code = final_params["code"]
                output_col_name = final_params["as"]

                logger.debug(f"Executing custom feature '{name}' with code:\n{code}")

                # TODO: check if we should groupby ticker
                if has_ticker:
                    result_list = [
                        _execute_custom_code(code, group_df)
                        for _, group_df in df_enriched.groupby("ticker")
                    ]
                    full_result = pd.concat(result_list)
                else:
                    # Apply to entire DataFrame if no ticker column
                    full_result = _execute_custom_code(code, df_enriched)

                all_new_cols.append(full_result.rename(output_col_name))
                logger.debug(f"Successfully applied custom feature '{name}' -> column '{output_col_name}'")
            else:
                # Use standard feature implementation
                impl_func = FEATURE_IMPLEMENTATIONS.get(name)

                if impl_func is None:
                    logger.error(f"Feature '{name}' not found in FEATURE_IMPLEMENTATIONS")
                    raise ValueError(f"Unknown feature: {name}")

                # Direct Calculation per Group (or entire DataFrame if no ticker)
                # TODO: check if we should groupby ticker
                if has_ticker:
                    result_list = [
                        impl_func(group_df, **final_params)
                        for _, group_df in df_enriched.groupby("ticker")
                    ]
                    full_result = pd.concat(result_list)
                else:
                    # Apply to entire DataFrame if no ticker column
                    full_result = impl_func(df_enriched, **final_params)

                # Assign Results
                if isinstance(full_result, pd.DataFrame):
                    for col in full_result.columns:
                        output_col_name = f"{name}_{col}"
                        all_new_cols.append(
                            full_result[[col]].rename(columns={col: output_col_name})
                        )
                        logger.debug(f"Applied feature '{name}' -> column '{output_col_name}'")
                else:
                    output_col_name = request.get(
                        "as",
                        f"{name}_{final_params.get('on', '')}_{final_params.get('window', '')}".rstrip(
                            "_"
                        ),
                    )
                    all_new_cols.append(full_result.rename(output_col_name))
                    logger.debug(f"Applied feature '{name}' -> column '{output_col_name}'")

        except Exception as e:
            logger.exception(f"Failed to apply feature '{name}' (feature {idx+1}/{len(dsl.get('features', []))})")
            logger.error(f"Feature params: {final_params}")
            logger.error(f"Available DataFrame columns: {list(df_enriched.columns)}")
            logger.error(f"Feature request: {request}")
            raise  # Re-raise to be caught by outer try-except

    # Combine original df with all new feature columns at once
    if all_new_cols:
        df_final = pd.concat([df_enriched] + all_new_cols, axis=1)
        return df_final.copy()

    return df_enriched.copy()


def enrich_dataframe_from_keywords(
    df: pd.DataFrame, user_keywords: list[str], registry_path: str = "registry.yaml"
) -> tuple[pd.DataFrame, dict]:
    """
    Main orchestration function that ties together LLM translation,
    DSL validation, and feature application.

    Args:
        df: DataFrame with at least 'ticker' and 'ts' columns
        user_keywords: List of feature keywords/descriptions from the user
        registry_path: Path to the registry YAML file

    Returns:
        tuple: (enriched_dataframe, metadata_dict)
            - enriched_dataframe: DataFrame with new features added
            - metadata_dict: Contains 'dsl', 'errors', and 'success' status

    Example:
        >>> df_enriched, metadata = enrich_dataframe_from_keywords(
        ...     df,
        ...     ["20 day sma on close", "14 day rsi"],
        ...     "registry.yaml"
        ... )
        >>> if metadata['success']:
        ...     print(f"Added features: {metadata['dsl']}")
    """
    # Load registry
    with open(registry_path, "r") as f:
        registry = yaml.safe_load(f)

    # Create allowed features prompt from registry
    allowed_features_prompt = _create_features_prompt(registry)

    # Get available columns from the DataFrame
    available_columns = list(df.columns)

    # Get DSL from LLM
    dsl_string = get_llm_recipe(
        user_keywords, allowed_features_prompt, available_columns
    )

    # Log the LLM response for debugging
    logger.info(f"LLM returned DSL recipe: {dsl_string}")

    # Validate and enrich DSL with defaults
    dsl, errors = validate_dsl(dsl_string, registry)

    # Log the validated DSL
    logger.debug(f"Validated DSL: {dsl}")

    metadata = {
        "dsl_string": dsl_string,
        "dsl": dsl,
        "errors": errors,
        "success": False,
    }

    if errors:
        logger.error(f"DSL Validation failed with {len(errors)} error(s):")
        for error in errors:
            logger.error(f"  - {error}")
        return df, metadata

    # Apply features
    try:
        logger.debug(f"Applying {len(dsl.get('features', []))} features to DataFrame")
        df_enriched = apply_features(df, dsl, registry)
        metadata["success"] = True
        logger.info(f"Successfully applied all features. New columns: {set(df_enriched.columns) - set(df.columns)}")
        return df_enriched, metadata
    except Exception as e:
        metadata["errors"].append(f"Feature application error: {str(e)}")
        logger.exception(f"Feature application failed with exception: {e}")
        logger.error(f"Failed while processing DSL: {dsl}")
        logger.error(f"Available DataFrame columns: {list(df.columns)}")
        return df, metadata


def _create_features_prompt(registry: dict) -> str:
    """
    Helper function to create a formatted prompt describing available features
    for the LLM from the registry.
    """
    lines = []
    for feature_name, feature_info in registry["features"].items():
        desc = feature_info.get("description", "")
        params = feature_info.get("params", {})

        param_list = []
        for p_name, p_rules in params.items():
            p_type = p_rules.get("type", "")
            required = p_rules.get("required", False)
            default = p_rules.get("default", None)
            allowed = p_rules.get("allowed", None)

            param_str = f"{p_name} ({p_type})"
            if required:
                param_str += " [required]"
            elif default is not None:
                param_str += f" [default: {default}]"
            if allowed:
                param_str += f" [allowed: {', '.join(map(str, allowed))}]"

            param_list.append(param_str)

        params_str = ", ".join(param_list) if param_list else "no parameters"
        lines.append(f"- {feature_name}: {desc} ({params_str})")

    return "\n".join(lines)
