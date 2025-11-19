"""
OutputValidator - Validates that requested features exist in DataFrames
"""
import pandas as pd
from typing import List, Dict, Any
from difflib import SequenceMatcher
from contracts import ValidationReport


class OutputValidator:
    def __init__(self, fuzzy_threshold: float = 0.8):
        self.fuzzy_threshold = fuzzy_threshold
        self.aliases = {
            "open": ["open", "openprice", "o"],
            "high": ["high", "highprice", "h"],
            "low": ["low", "lowprice", "l"],
            "close": ["close", "closeprice", "c"],
            "volume": ["volume", "vol", "v"],
            "value": ["value", "val"],
        }
    
    def validate_dataset(self, df: pd.DataFrame, native_features: List[str],
                         api_name: str, endpoint_name: str, ticker: str = None) -> ValidationReport:
        if df.empty:
            return ValidationReport(
                api_name=api_name, endpoint_name=endpoint_name, ticker=ticker,
                missing_features=native_features, actual_columns=[], validation_passed=False
            )
        
        normalized_columns = {self._normalize(col): col for col in df.columns}
        found, fuzzy_matched, missing = [], [], []
        
        for feature in native_features:
            normalized_feature = self._normalize(feature)
            
            if normalized_feature in normalized_columns:
                found.append(feature)
            elif self._check_aliases(normalized_feature, normalized_columns):
                found.append(feature)
            elif fuzzy_result := self._fuzzy_match(normalized_feature, normalized_columns):
                fuzzy_matched.append(fuzzy_result)
            else:
                missing.append(feature)
        
        return ValidationReport(
            api_name=api_name, endpoint_name=endpoint_name, ticker=ticker,
            found_features=found, fuzzy_matched_features=fuzzy_matched,
            missing_features=missing, actual_columns=list(df.columns),
            validation_passed=len(missing) == 0
        )
    
    def validate_multiple(self, results: List[Any], native_features: List[str]) -> List[ValidationReport]:
        reports = []
        for result in results:
            if result.status == "SUCCESS" and isinstance(result.data, pd.DataFrame):
                report = self.validate_dataset(
                    df=result.data, native_features=native_features,
                    api_name=result.api_name, endpoint_name=result.endpoint_name,
                    ticker=result.used_parameters.get("ticker") or result.used_parameters.get("symbol")
                )
                reports.append(report)
        return reports
    
    def _normalize(self, text: str) -> str:
        return text.lower().replace(" ", "").replace("_", "").replace("-", "")
    
    def _check_aliases(self, normalized_feature: str, normalized_columns: Dict[str, str]) -> bool:
        if normalized_feature not in self.aliases:
            return False
        return any(self._normalize(alias) in normalized_columns for alias in self.aliases[normalized_feature])
    
    def _fuzzy_match(self, normalized_feature: str, normalized_columns: Dict[str, str]) -> Dict[str, Any]:
        best_match, best_score = None, 0.0
        for norm_col, original_col in normalized_columns.items():
            score = SequenceMatcher(None, normalized_feature, norm_col).ratio()
            if score > best_score:
                best_score, best_match = score, original_col
        
        if best_score >= self.fuzzy_threshold:
            return {"feature": normalized_feature, "matched_column": best_match, "similarity_score": best_score}
        return None