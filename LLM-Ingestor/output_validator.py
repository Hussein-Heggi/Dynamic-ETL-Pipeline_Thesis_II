"""
OutputValidator - Validates that requested features exist in returned DataFrames
"""
import pandas as pd
from typing import List, Dict, Any
from difflib import SequenceMatcher
from contracts import ValidationReport


class OutputValidator:
    """
    Validates that native features requested exist in DataFrame columns.
    Uses normalization + aliases + fuzzy matching.
    """
    
    def __init__(self, fuzzy_threshold: float = 0.8):
        """
        Initialize validator
        
        Args:
            fuzzy_threshold: Minimum similarity score for fuzzy matches (0-1)
        """
        self.fuzzy_threshold = fuzzy_threshold
        
        # Column aliases for common variations
        self.aliases = {
            "open": ["open", "openprice", "o"],
            "high": ["high", "highprice", "h"],
            "low": ["low", "lowprice", "l"],
            "close": ["close", "closeprice", "c"],
            "volume": ["volume", "vol", "v"],
            "value": ["value", "val"],
            "rate": ["rate", "percentage", "pct"]
        }
    
    def validate_dataset(
        self,
        df: pd.DataFrame,
        native_features: List[str],
        api_name: str,
        endpoint_name: str,
        ticker: str = None
    ) -> ValidationReport:
        """Validate that native features exist in DataFrame columns"""
        if df.empty:
            return ValidationReport(
                api_name=api_name,
                endpoint_name=endpoint_name,
                ticker=ticker,
                missing_features=native_features,
                actual_columns=[],
                validation_passed=False
            )
        
        # Normalize column names
        normalized_columns = {self._normalize(col): col for col in df.columns}
        
        found = []
        fuzzy_matched = []
        missing = []
        
        for feature in native_features:
            normalized_feature = self._normalize(feature)
            
            # Check exact match
            if normalized_feature in normalized_columns:
                found.append(feature)
                continue
            
            # Check aliases
            if self._check_aliases(normalized_feature, normalized_columns):
                found.append(feature)
                continue
            
            # Check fuzzy match
            fuzzy_result = self._fuzzy_match(normalized_feature, normalized_columns)
            if fuzzy_result:
                fuzzy_matched.append(fuzzy_result)
                continue
            
            # Not found
            missing.append(feature)
        
        validation_passed = len(missing) == 0
        
        return ValidationReport(
            api_name=api_name,
            endpoint_name=endpoint_name,
            ticker=ticker,
            found_features=found,
            fuzzy_matched_features=fuzzy_matched,
            missing_features=missing,
            actual_columns=list(df.columns),
            validation_passed=validation_passed
        )
    
    def validate_multiple(
        self,
        results: List[Any],
        native_features: List[str]
    ) -> List[ValidationReport]:
        """Validate multiple datasets"""
        reports = []
        
        for result in results:
            if result.status == "SUCCESS" and isinstance(result.data, pd.DataFrame):
                report = self.validate_dataset(
                    df=result.data,
                    native_features=native_features,
                    api_name=result.api_name,
                    endpoint_name=result.endpoint_name,
                    ticker=result.used_parameters.get("ticker") or result.used_parameters.get("symbol")
                )
                reports.append(report)
        
        return reports
    
    def _normalize(self, text: str) -> str:
        """Normalize text: lowercase, remove spaces/underscores/hyphens"""
        return text.lower().replace(" ", "").replace("_", "").replace("-", "")
    
    def _check_aliases(self, normalized_feature: str, normalized_columns: Dict[str, str]) -> bool:
        """Check if feature matches any column via aliases"""
        if normalized_feature not in self.aliases:
            return False
        
        for alias in self.aliases[normalized_feature]:
            if self._normalize(alias) in normalized_columns:
                return True
        
        return False
    
    def _fuzzy_match(self, normalized_feature: str, normalized_columns: Dict[str, str]) -> Dict[str, Any]:
        """Perform fuzzy matching against columns"""
        best_match = None
        best_score = 0.0
        
        for norm_col, original_col in normalized_columns.items():
            score = SequenceMatcher(None, normalized_feature, norm_col).ratio()
            if score > best_score:
                best_score = score
                best_match = original_col
        
        if best_score >= self.fuzzy_threshold:
            return {
                "feature": normalized_feature,
                "matched_column": best_match,
                "similarity_score": best_score
            }
        
        return None