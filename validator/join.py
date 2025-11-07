"""
Join Engine: Row matching and horizontal concatenation
Includes 25 feature extraction, XGBoost prediction, greedy matching, and 2-stage join
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import pickle
import logging
from scipy.stats import pearsonr

from .config import ValidatorConfig


class JoinEngine:
    """Handles all Join operations"""
    
    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.logger = logging.getLogger('JoinEngine')
        
        # Load model
        self.join_model = self._load_join_model()
    
    def _load_join_model(self):
        """Load pre-trained Join XGBoost model"""
        try:
            with open(self.config.JOIN_MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            self.logger.info("Join model loaded successfully")
            return model
        except FileNotFoundError:
            self.logger.warning(f"Join model not found at {self.config.JOIN_MODEL_PATH}")
            return None
    
    def extract_features(self, row_a: pd.Series, row_b: pd.Series) -> Dict[str, float]:
        """
        Extract 25 similarity features for a single row pair
        
        Features:
        - Statistical aggregates (min, max, mean, median, std)
        - Relative differences
        - Distance metrics (L1, L2)
        - Correlation
        - Count features
        """
        # Convert to numeric arrays
        arr_a = pd.to_numeric(row_a, errors='coerce').values
        arr_b = pd.to_numeric(row_b, errors='coerce').values
        
        # Handle NaN values
        valid_mask = ~(np.isnan(arr_a) | np.isnan(arr_b))
        
        if valid_mask.sum() == 0:
            # No valid pairs, return zero features
            return {feat: 0.0 for feat in self.config.JOIN_FEATURES}
        
        a_valid = arr_a[valid_mask]
        b_valid = arr_b[valid_mask]
        
        eps = self.config.EPSILON
        
        # Compute differences
        abs_diff = np.abs(a_valid - b_valid)
        rel_diff = abs_diff / (np.abs(b_valid) + eps)
        ratio = a_valid / (b_valid + eps)
        
        # Z-scores
        a_z = (a_valid - a_valid.mean()) / (a_valid.std() + eps)
        b_z = (b_valid - b_valid.mean()) / (b_valid.std() + eps)
        z_diff = np.abs(a_z - b_z)
        
        # Percentage differences
        pct_diff = 100 * (a_valid - b_valid) / (b_valid + eps)
        
        # Count features
        n_close = np.sum(abs_diff < self.config.CLOSE_TOLERANCE)
        n_very_close = np.sum(abs_diff < self.config.VERY_CLOSE_TOLERANCE)
        n_both_zero = np.sum((np.abs(a_valid) < eps) & (np.abs(b_valid) < eps))
        
        # Sign agreement
        sign_agreement = np.mean(np.sign(a_valid) == np.sign(b_valid))
        
        # Correlation
        try:
            corr, _ = pearsonr(a_valid, b_valid)
            if np.isnan(corr):
                corr = 0.0
        except:
            corr = 0.0
        
        # Distance metrics
        L1_raw = np.sum(abs_diff)
        L1_z = np.sum(np.abs(a_z - b_z))
        L1_pct = np.sum(np.abs(pct_diff))
        
        L2_raw = np.sqrt(np.sum(abs_diff ** 2))
        L2_z = np.sqrt(np.sum((a_z - b_z) ** 2))
        L2_pct = np.sqrt(np.sum(pct_diff ** 2))
        
        # Compile features
        features = {
            'absdiff_min': np.min(abs_diff),
            'absdiff_max': np.max(abs_diff),
            'absdiff_mean': np.mean(abs_diff),
            'absdiff_median': np.median(abs_diff),
            'absdiff_std': np.std(abs_diff),
            
            'reldiff_mean': np.mean(rel_diff),
            'reldiff_median': np.median(rel_diff),
            
            'ratio_mean': np.mean(ratio),
            'ratio_median': np.median(ratio),
            'ratio_std': np.std(ratio),
            
            'zdiff_mean': np.mean(z_diff),
            'zdiff_max': np.max(z_diff),
            
            'pcdiff_mean': np.mean(pct_diff),
            'pcdiff_sq_mean': np.mean(pct_diff ** 2),
            
            'L1_raw': L1_raw,
            'L1_z': L1_z,
            'L1_pct': L1_pct,
            
            'L2_raw': L2_raw,
            'L2_z': L2_z,
            'L2_pct': L2_pct,
            
            'correlation': corr,
            
            'n_close': n_close,
            'n_very_close': n_very_close,
            'n_both_zero': n_both_zero,
            
            'sign_agreement': sign_agreement
        }
        
        return features
    
    def compute_batch_features(
        self,
        df_small: pd.DataFrame,
        df_large: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate features for all row pair combinations
        
        Args:
            df_small: Smaller dataframe (fewer rows)
            df_large: Larger dataframe (more rows)
        
        Returns:
            DataFrame with columns: small_idx, large_idx, + 25 feature columns
        """
        n_small = len(df_small)
        n_large = len(df_large)
        total_combinations = n_small * n_large
        
        self.logger.info(f"Generating features for {total_combinations} row pairs...")
        
        # Check limit
        if total_combinations > self.config.MAX_COMBINATIONS:
            self.logger.warning(f"Combinations ({total_combinations}) exceed limit ({self.config.MAX_COMBINATIONS})")
            # Could implement sampling here if needed
        
        # Initialize lists
        small_indices = []
        large_indices = []
        feature_dicts = []
        
        # Generate all combinations
        for i in range(n_small):
            row_small = df_small.iloc[i]
            
            for j in range(n_large):
                row_large = df_large.iloc[j]
                
                # Extract features
                features = self.extract_features(row_small, row_large)
                
                small_indices.append(i)
                large_indices.append(j)
                feature_dicts.append(features)
        
        # Create DataFrame
        result = pd.DataFrame(feature_dicts)
        result.insert(0, 'small_idx', small_indices)
        result.insert(1, 'large_idx', large_indices)
        
        self.logger.info(f"Feature generation complete: {len(result)} rows")
        
        return result
    
    def greedy_match(
        self,
        probabilities: np.ndarray,
        small_indices: np.ndarray,
        large_indices: np.ndarray
    ) -> List[Tuple[int, int, float]]:
        """
        Greedy matching algorithm ensuring no duplicate assignments
        
        Algorithm:
        1. Create list of all pairs with prob > threshold
        2. Sort by probability (descending)
        3. Greedily assign:
           - Take highest probability pair
           - Mark both rows as used
           - Continue with next highest
           - Skip if either row already used
        
        Returns:
            List of (small_idx, large_idx, probability) tuples
        """
        # Filter pairs above threshold
        valid_mask = probabilities > self.config.JOIN_ROW_THRESHOLD
        
        if valid_mask.sum() == 0:
            self.logger.warning("No valid matches found (all probs below threshold)")
            return []
        
        valid_probs = probabilities[valid_mask]
        valid_small = small_indices[valid_mask]
        valid_large = large_indices[valid_mask]
        
        # Sort by probability (descending)
        sort_idx = np.argsort(valid_probs)[::-1]
        
        sorted_probs = valid_probs[sort_idx]
        sorted_small = valid_small[sort_idx]
        sorted_large = valid_large[sort_idx]
        
        # Greedy assignment
        matches = []
        used_small = set()
        used_large = set()
        
        for prob, s_idx, l_idx in zip(sorted_probs, sorted_small, sorted_large):
            if s_idx not in used_small and l_idx not in used_large:
                matches.append((int(s_idx), int(l_idx), float(prob)))
                used_small.add(s_idx)
                used_large.add(l_idx)
        
        self.logger.debug(f"Greedy matching: {len(matches)} matches from {len(valid_probs)} candidates")
        
        return matches
    
    def compute_compatibility(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        denominator: int
    ) -> Tuple[float, List[Tuple[int, int, float]]]:
        """
        Compute join compatibility between two dataframes
        
        Steps:
        1. Identify smaller and larger dataframes
        2. Generate all row pair combinations
        3. Extract 25 features for each pair
        4. XGBoost prediction for each pair
        5. Greedy matching to find best assignments
        6. Calculate retention score
        
        Args:
            df1, df2: DataFrames to join
            denominator: Fixed denominator for retention calculation
        
        Returns:
            retention: float (matched_rows / denominator)
            matches: List of (small_idx, large_idx, probability)
        """
        # Identify smaller and larger
        if len(df1) <= len(df2):
            df_small, df_large = df1, df2
        else:
            df_small, df_large = df2, df1
            # Note: we'll need to swap indices back later if needed
        
        self.logger.info(f"Join compatibility check: {len(df_small)} x {len(df_large)} rows")
        
        # Select only numeric columns for join
        numeric_cols_small = df_small.select_dtypes(include=[np.number]).columns
        numeric_cols_large = df_large.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols_small) == 0 or len(numeric_cols_large) == 0:
            self.logger.warning("No numeric columns found for join")
            return 0.0, []
        
        df_small_numeric = df_small[numeric_cols_small]
        df_large_numeric = df_large[numeric_cols_large]
        
        # Generate features for all combinations
        feature_df = self.compute_batch_features(df_small_numeric, df_large_numeric)
        
        # XGBoost prediction
        if self.join_model is None:
            self.logger.error("Join model not loaded, cannot predict")
            return 0.0, []
        
        X = feature_df[self.config.JOIN_FEATURES].values
        
        try:
            probabilities = self.join_model.predict_proba(X)[:, 1]
        except Exception as e:
            self.logger.error(f"Model prediction failed: {e}")
            return 0.0, []
        
        # Greedy matching
        matches = self.greedy_match(
            probabilities,
            feature_df['small_idx'].values,
            feature_df['large_idx'].values
        )
        
        # Calculate retention
        matched_rows = len(matches)
        retention = matched_rows / denominator if denominator > 0 else 0.0
        
        self.logger.info(f"Retention: {matched_rows}/{denominator} = {retention:.3f}")
        
        # If we swapped, need to swap indices back
        if len(df1) > len(df2):
            matches = [(l_idx, s_idx, prob) for s_idx, l_idx, prob in matches]
        
        return retention, matches
    
    def execute_join(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        matches: List[Tuple[int, int, float]]
    ) -> pd.DataFrame:
        """
        Execute join based on matched row pairs
        
        Args:
            df1, df2: DataFrames to join
            matches: List of (df1_idx, df2_idx, probability)
        
        Returns:
            Joined dataframe with matched rows concatenated horizontally
        """
        if len(matches) == 0:
            self.logger.warning("No matches to execute join")
            return pd.DataFrame()
        
        # Extract indices
        indices_1 = [m[0] for m in matches]
        indices_2 = [m[1] for m in matches]
        
        # Select matched rows
        df1_matched = df1.iloc[indices_1].reset_index(drop=True)
        df2_matched = df2.iloc[indices_2].reset_index(drop=True)
        
        # Concatenate horizontally
        result = pd.concat([df1_matched, df2_matched], axis=1)
        
        self.logger.info(f"Join executed: {len(matches)} rows, {result.shape[1]} columns")
        
        return result
    
    def stage_1(
        self,
        dataframes: List[pd.DataFrame],
        denominator: int
    ) -> Tuple[List[pd.DataFrame], List[Dict]]:
        """
        Stage 1: Initial pairwise joins
        
        For each dataframe in order:
        - Try joining with all remaining dataframes
        - Pick the one with highest retention > threshold
        - Execute join and mark both as used
        
        Returns:
            List of joined/remaining dataframes
            List of operation reports
        """
        operations = []
        remaining = list(range(len(dataframes)))
        result_dfs = []
        
        while remaining:
            # Take first remaining dataframe
            i = remaining.pop(0)
            df_i = dataframes[i]
            
            best_j = None
            best_retention = 0.0
            best_matches = []
            
            # Try pairing with all other remaining dataframes
            for j in remaining:
                df_j = dataframes[j]
                
                # Compute compatibility
                retention, matches = self.compute_compatibility(df_i, df_j, denominator)
                
                # Check if better than current best
                if retention > self.config.JOIN_RETENTION_THRESHOLD and retention > best_retention:
                    best_retention = retention
                    best_j = j
                    best_matches = matches
            
            # Execute join if found good match
            if best_j is not None:
                df_j = dataframes[best_j]
                
                # Execute join
                joined_df = self.execute_join(df_i, df_j, best_matches)
                
                # Log operation
                operations.append({
                    'stage': 1,
                    'dataframes': [i, best_j],
                    'retention': best_retention,
                    'matched_rows': len(best_matches),
                    'result_shape': joined_df.shape
                })
                
                # Add to results
                result_dfs.append(joined_df)
                
                # Remove best_j from remaining
                remaining.remove(best_j)
                
                self.logger.info(f"Stage 1: Joined DF{i} + DF{best_j}, retention={best_retention:.3f}")
            else:
                # No good join found, keep df_i as is
                result_dfs.append(df_i)
                self.logger.info(f"Stage 1: DF{i} remains unjoinable")
        
        self.logger.info(f"Stage 1 complete: {len(dataframes)} → {len(result_dfs)} dataframes")
        
        return result_dfs, operations
    
    def stage_2(
        self,
        dataframes: List[pd.DataFrame],
        denominator: int
    ) -> Tuple[List[pd.DataFrame], List[Dict]]:
        """
        Stage 2: Join the joined groups
        
        Try to combine results from Stage 1
        - Attempt all pairwise joins
        - Prefer joins that combine all dataframes
        - If multiple valid joins, pick best retention
        - If no valid joins, return Stage 1 results
        
        Returns:
            Final list of dataframes
            List of operation reports
        """
        if len(dataframes) <= 1:
            self.logger.info("Stage 2: Only 1 dataframe, nothing to join")
            return dataframes, []
        
        operations = []
        best_combination = None
        best_retention = 0.0
        best_matches = []
        best_pair = None
        
        # Try all pairs
        for i in range(len(dataframes)):
            for j in range(i + 1, len(dataframes)):
                df_i = dataframes[i]
                df_j = dataframes[j]
                
                # Compute compatibility
                retention, matches = self.compute_compatibility(df_i, df_j, denominator)
                
                if retention > self.config.JOIN_RETENTION_THRESHOLD:
                    # Check if this combines all dataframes
                    if len(dataframes) == 2:
                        # Only 2 dataframes, this would combine all
                        if retention > best_retention:
                            best_retention = retention
                            best_matches = matches
                            best_pair = (i, j)
                            best_combination = 'all'
                    else:
                        # More than 2 dataframes, partial combination
                        if retention > best_retention:
                            best_retention = retention
                            best_matches = matches
                            best_pair = (i, j)
                            best_combination = 'partial'
        
        # Execute best join if found
        if best_pair is not None:
            i, j = best_pair
            df_i = dataframes[i]
            df_j = dataframes[j]
            
            # Execute join
            joined_df = self.execute_join(df_i, df_j, best_matches)
            
            # Log operation
            operations.append({
                'stage': 2,
                'dataframes': [i, j],
                'retention': best_retention,
                'matched_rows': len(best_matches),
                'result_shape': joined_df.shape,
                'combination': best_combination
            })
            
            # Build result
            result_dfs = [joined_df]
            
            # Add remaining dataframes
            for k in range(len(dataframes)):
                if k != i and k != j:
                    result_dfs.append(dataframes[k])
            
            self.logger.info(f"Stage 2: Joined DF{i} + DF{j}, retention={best_retention:.3f}, combination={best_combination}")
        else:
            # No valid joins, return as is
            result_dfs = dataframes
            self.logger.info("Stage 2: No valid joins found, returning Stage 1 results")
        
        self.logger.info(f"Stage 2 complete: {len(dataframes)} → {len(result_dfs)} dataframes")
        
        return result_dfs, operations