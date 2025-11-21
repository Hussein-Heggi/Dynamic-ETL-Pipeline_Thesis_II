"""
Join Engine: Row matching and horizontal concatenation
Optimized with chunked parallelization, batch prediction, and limited many-to-many matching
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import logging
from joblib import Parallel, delayed
from tqdm import tqdm
import time

from config import ValidatorConfig

# GPU imports (optional, will fallback to CPU if not available)
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


class JoinEngine:
    """Handles all Join operations with optimized parallel processing"""
    
    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.logger = logging.getLogger('JoinEngine')
        
        # Load pre-trained Join XGBoost model
        self.join_model = self._load_join_model()
    
    def _load_join_model(self):
        """Load pre-trained Join XGBoost model"""
        try:
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(self.config.JOIN_MODEL_PATH)
            self.logger.info("Join model loaded successfully")
            return model
        except Exception as e:
            self.logger.warning(f"Failed to load Join model: {e}")
            return None
    
    def _align_columns(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align columns between two dataframes using union + mean padding
        
        Strategy:
        1. Take union of all numeric columns
        2. For missing columns in each df, pad with mean of existing columns
        
        Returns:
            (df1_aligned, df2_aligned) - both with same columns
        """
        # Extract numeric columns only
        df1_numeric = df1.select_dtypes(include=[np.number])
        df2_numeric = df2.select_dtypes(include=[np.number])
        
        if df1_numeric.empty or df2_numeric.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Get union of column names
        all_cols = list(set(df1_numeric.columns) | set(df2_numeric.columns))
        
        # Align df1
        df1_aligned = df1_numeric.copy()
        missing_in_df1 = set(all_cols) - set(df1_numeric.columns)
        if missing_in_df1:
            # Pad with mean of existing columns
            mean_val = df1_numeric.mean().mean()
            for col in missing_in_df1:
                df1_aligned[col] = mean_val
        
        # Align df2
        df2_aligned = df2_numeric.copy()
        missing_in_df2 = set(all_cols) - set(df2_numeric.columns)
        if missing_in_df2:
            # Pad with mean of existing columns
            mean_val = df2_numeric.mean().mean()
            for col in missing_in_df2:
                df2_aligned[col] = mean_val
        
        # Ensure same column order
        df1_aligned = df1_aligned[all_cols]
        df2_aligned = df2_aligned[all_cols]
        
        return df1_aligned, df2_aligned
    
    def _extract_features_for_pair(
        self,
        row_a: np.ndarray,
        row_b: np.ndarray
    ) -> np.ndarray:
        """
        Extract 26 statistical features comparing two rows
        Returns features in EXACT order expected by XGBoost model
        
        Args:
            row_a: First row as numpy array
            row_b: Second row as numpy array
        
        Returns:
            Feature vector of shape (26,)
        """
        epsilon = self.config.EPSILON
        
        # Compute basic differences
        diff = row_a - row_b
        abs_diff = np.abs(diff)
        
        # Handle division by zero
        ratio = row_a / (row_b + epsilon)
        rel_diff = diff / (np.abs(row_b) + epsilon)
        
        # Z-score normalization
        mean_a, std_a = np.mean(row_a), np.std(row_a) + epsilon
        mean_b, std_b = np.mean(row_b), np.std(row_b) + epsilon
        z_a = (row_a - mean_a) / std_a
        z_b = (row_b - mean_b) / std_b
        z_diff = z_a - z_b
        
        # Percentage change
        pct_change = diff / (row_a + epsilon)
        
        # Compute all feature values
        absdiff_mean = np.mean(abs_diff)
        absdiff_median = np.median(abs_diff)
        absdiff_max = np.max(abs_diff)
        absdiff_min = np.min(abs_diff)
        absdiff_std = np.std(abs_diff)
        
        reldiff_mean = np.mean(rel_diff)
        reldiff_median = np.median(rel_diff)
        
        zdiff_mean = np.mean(z_diff)
        zdiff_max = np.max(np.abs(z_diff))
        
        pcdiff_mean = np.mean(pct_change)
        pcdiff_sq_mean = np.mean(pct_change ** 2)
        
        L1_raw = np.sum(abs_diff)
        L2_raw = np.sqrt(np.sum(diff ** 2))
        
        L1_z = np.sum(np.abs(z_diff))
        L2_z = np.sqrt(np.sum(z_diff ** 2))
        
        L1_pct = np.sum(np.abs(pct_change))
        L2_pct = np.sqrt(np.sum(pct_change ** 2))
        
        # Correlation
        if len(row_a) > 1:
            correlation = np.corrcoef(row_a, row_b)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        ratio_mean = np.mean(ratio)
        ratio_median = np.median(ratio)
        ratio_std = np.std(ratio)
        
        sign_agreement = np.mean(np.sign(row_a) == np.sign(row_b))
        
        # Count features
        n_features = len(row_a)
        n_both_zero = np.sum((row_a == 0) & (row_b == 0))
        
        close_tol = self.config.CLOSE_TOLERANCE
        very_close_tol = self.config.VERY_CLOSE_TOLERANCE
        n_close = np.sum(abs_diff <= close_tol)
        n_very_close = np.sum(abs_diff <= very_close_tol)
        
        # Return in EXACT order model expects (26 features)
        features = np.array([
            absdiff_mean, absdiff_median, absdiff_max, absdiff_min, absdiff_std,
            reldiff_mean, reldiff_median,
            zdiff_mean, zdiff_max,
            pcdiff_mean, pcdiff_sq_mean,
            L1_raw, L2_raw,
            L1_z, L2_z,
            L1_pct, L2_pct,
            correlation,
            ratio_mean, ratio_median, ratio_std,
            sign_agreement,
            n_features,
            n_both_zero,
            n_close, n_very_close
        ])
        
        return features

    def _extract_features_batch_gpu(
        self,
        rows_a: np.ndarray,
        rows_b: np.ndarray
    ) -> np.ndarray:
        """
        Extract 26 statistical features for BATCH of row pairs using GPU

        This is a vectorized version of _extract_features_for_pair that processes
        multiple pairs simultaneously on GPU for massive speedup via SIMD.

        Args:
            rows_a: (batch_size, n_features) - First rows in each pair
            rows_b: (batch_size, n_features) - Second rows in each pair

        Returns:
            (batch_size, 26) feature matrix - one 26-feature vector per pair
        """
        if not GPU_AVAILABLE or not self.config.USE_GPU:
            # Fallback to CPU batch processing
            return self._extract_features_batch_cpu(rows_a, rows_b)

        # Move data to GPU
        rows_a_gpu = cp.asarray(rows_a, dtype=cp.float32)
        rows_b_gpu = cp.asarray(rows_b, dtype=cp.float32)

        epsilon = self.config.EPSILON
        batch_size = rows_a_gpu.shape[0]

        # Vectorized operations on GPU (all pairs processed in parallel via SIMD)
        diff = rows_a_gpu - rows_b_gpu
        abs_diff = cp.abs(diff)

        # Handle division by zero
        ratio = rows_a_gpu / (rows_b_gpu + epsilon)
        rel_diff = diff / (cp.abs(rows_b_gpu) + epsilon)

        # Z-score normalization (per row)
        mean_a = cp.mean(rows_a_gpu, axis=1, keepdims=True)
        std_a = cp.std(rows_a_gpu, axis=1, keepdims=True) + epsilon
        mean_b = cp.mean(rows_b_gpu, axis=1, keepdims=True)
        std_b = cp.std(rows_b_gpu, axis=1, keepdims=True) + epsilon

        z_a = (rows_a_gpu - mean_a) / std_a
        z_b = (rows_b_gpu - mean_b) / std_b
        z_diff = z_a - z_b

        # Percentage change
        pct_change = diff / (rows_a_gpu + epsilon)

        # Aggregate statistics across features (axis=1) for each pair
        absdiff_mean = cp.mean(abs_diff, axis=1)
        absdiff_median = cp.median(abs_diff, axis=1)
        absdiff_max = cp.max(abs_diff, axis=1)
        absdiff_min = cp.min(abs_diff, axis=1)
        absdiff_std = cp.std(abs_diff, axis=1)

        reldiff_mean = cp.mean(rel_diff, axis=1)
        reldiff_median = cp.median(rel_diff, axis=1)

        zdiff_mean = cp.mean(z_diff, axis=1)
        zdiff_max = cp.max(cp.abs(z_diff), axis=1)

        pcdiff_mean = cp.mean(pct_change, axis=1)
        pcdiff_sq_mean = cp.mean(pct_change ** 2, axis=1)

        L1_raw = cp.sum(abs_diff, axis=1)
        L2_raw = cp.sqrt(cp.sum(diff ** 2, axis=1))

        L1_z = cp.sum(cp.abs(z_diff), axis=1)
        L2_z = cp.sqrt(cp.sum(z_diff ** 2, axis=1))

        L1_pct = cp.sum(cp.abs(pct_change), axis=1)
        L2_pct = cp.sqrt(cp.sum(pct_change ** 2, axis=1))

        # Correlation (fully vectorized for batch)
        if rows_a_gpu.shape[1] > 1:
            # Center the data (subtract row-wise means)
            centered_a = rows_a_gpu - mean_a
            centered_b = rows_b_gpu - mean_b

            # Compute correlation for all pairs at once
            numerator = cp.sum(centered_a * centered_b, axis=1)
            denominator = cp.sqrt(cp.sum(centered_a ** 2, axis=1) * cp.sum(centered_b ** 2, axis=1)) + epsilon

            correlation = numerator / denominator
            correlation = cp.nan_to_num(correlation, nan=0.0)
        else:
            correlation = cp.zeros(batch_size, dtype=cp.float32)

        ratio_mean = cp.mean(ratio, axis=1)
        ratio_median = cp.median(ratio, axis=1)
        ratio_std = cp.std(ratio, axis=1)

        sign_agreement = cp.mean((cp.sign(rows_a_gpu) == cp.sign(rows_b_gpu)).astype(cp.float32), axis=1)

        # Count features
        n_features = cp.full(batch_size, rows_a_gpu.shape[1], dtype=cp.float32)
        n_both_zero = cp.sum((rows_a_gpu == 0) & (rows_b_gpu == 0), axis=1).astype(cp.float32)

        close_tol = self.config.CLOSE_TOLERANCE
        very_close_tol = self.config.VERY_CLOSE_TOLERANCE
        n_close = cp.sum(abs_diff <= close_tol, axis=1).astype(cp.float32)
        n_very_close = cp.sum(abs_diff <= very_close_tol, axis=1).astype(cp.float32)

        # Stack all features into (batch_size, 26) matrix
        features = cp.column_stack([
            absdiff_mean, absdiff_median, absdiff_max, absdiff_min, absdiff_std,
            reldiff_mean, reldiff_median,
            zdiff_mean, zdiff_max,
            pcdiff_mean, pcdiff_sq_mean,
            L1_raw, L2_raw,
            L1_z, L2_z,
            L1_pct, L2_pct,
            correlation,
            ratio_mean, ratio_median, ratio_std,
            sign_agreement,
            n_features,
            n_both_zero,
            n_close, n_very_close
        ])

        # Move back to CPU as numpy array
        return cp.asnumpy(features)

    def _extract_features_batch_cpu(
        self,
        rows_a: np.ndarray,
        rows_b: np.ndarray
    ) -> np.ndarray:
        """
        CPU fallback for batch feature extraction
        Processes pairs one-by-one using existing method
        """
        batch_size = rows_a.shape[0]
        features = np.zeros((batch_size, 26), dtype=np.float32)

        for i in range(batch_size):
            features[i] = self._extract_features_for_pair(rows_a[i], rows_b[i])

        return features

    def _create_chunks(self, n1: int, n2: int) -> List[Tuple[int, int, int, int]]:
        """
        Create chunks of row pairs for parallel processing
        
        Args:
            n1: Number of rows in df1
            n2: Number of rows in df2
        
        Returns:
            List of (start_i, end_i, start_j, end_j) tuples
            Each tuple represents a rectangular chunk of the pair matrix
        
        Strategy:
            - Split df1 into row ranges
            - Each chunk processes a range of df1 rows against ALL df2 rows
            - Creates N_JOBS * 4 chunks for load balancing (80 chunks for 20 cores)
        """
        total_pairs = n1 * n2
        num_chunks = self.config.N_JOBS * 4  # Hardcoded multiplier
        
        # Calculate how many rows of df1 each chunk should process
        pairs_per_chunk = max(1, total_pairs // num_chunks)
        rows_per_chunk = max(1, pairs_per_chunk // n2)
        
        chunks = []
        start_i = 0
        
        while start_i < n1:
            end_i = min(start_i + rows_per_chunk, n1)
            chunks.append((start_i, end_i, 0, n2))  # Process all of df2
            start_i = end_i
        
        return chunks
    
    def _process_chunk(
        self,
        df1_values: np.ndarray,
        df2_values: np.ndarray,
        start_i: int,
        end_i: int,
        start_j: int,
        end_j: int
    ) -> List[Tuple[np.ndarray, int, int]]:
        """
        Process a chunk of row pairs and extract features
        
        Args:
            df1_values: Full df1 as NumPy array (n1, features)
            df2_values: Full df2 as NumPy array (n2, features)
            start_i, end_i: Row range in df1 to process
            start_j, end_j: Row range in df2 to process
        
        Returns:
            List of (features, i, j) tuples for all pairs in this chunk
        """
        results = []
        
        for i in range(start_i, end_i):
            row_a = df1_values[i]
            
            for j in range(start_j, end_j):
                row_b = df2_values[j]
                
                # Extract features (no indices needed)
                features = self._extract_features_for_pair(row_a, row_b)
                
                # Skip if any NaN features
                if not np.isnan(features).any():
                    results.append((features, i, j))
        
        return results
    
    def find_compatible_rows(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame
    ) -> List[Tuple[int, int, float]]:
        """
        Find compatible row pairs between two dataframes using ML model

        Optimizations:
        - GPU batch processing (if enabled) for massive SIMD speedup
        - Chunked parallelization (N_JOBS * 4 chunks instead of N*M jobs)
        - Batch XGBoost prediction (single call instead of N*M calls)
        - Limited many-to-many assignment (max K matches per row)

        Returns:
            List of (idx_a, idx_b, probability) tuples (LIMITED MANY-TO-MANY matches)
        """
        # Route to GPU or CPU implementation
        if GPU_AVAILABLE and self.config.USE_GPU:
            return self._find_compatible_rows_gpu(df1, df2)
        else:
            return self._find_compatible_rows_cpu(df1, df2)

    def _find_compatible_rows_gpu(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame
    ) -> List[Tuple[int, int, float]]:
        """
        GPU-accelerated version using batched feature extraction
        Processes ALL pairs in GPU batches for maximum SIMD utilization
        """
        print("\nFinding compatible rows between datasets (GPU MODE):")

        # Step 1: Align columns and extract numeric data
        print("  [1/5] Aligning columns...")
        df1_numeric, df2_numeric = self._align_columns(df1, df2)

        if df1_numeric.empty or df2_numeric.empty:
            print("  ✗ No numeric columns to compare")
            return []

        n1, n2 = len(df1_numeric), len(df2_numeric)
        total_pairs = n1 * n2

        print(f"  → Processing {n1:,} × {n2:,} = {total_pairs:,} row pairs")
        print(f"  → Aligned to {df1_numeric.shape[1]} numeric features")
        print(f"  → Using GPU with batch size: {self.config.GPU_BATCH_SIZE:,}")

        # Step 2: Pre-convert to NumPy
        print("  [2/5] Converting to NumPy arrays...")
        df1_values = df1_numeric.values.astype('float32')
        df2_values = df2_numeric.values.astype('float32')

        # Step 3: Calculate batch parameters (don't generate all pairs - too much memory!)
        print("  [3/5] Setting up batch processing...")
        batch_size = self.config.GPU_BATCH_SIZE
        num_batches = (total_pairs + batch_size - 1) // batch_size

        print(f"  → Processing {num_batches} batches of up to {batch_size:,} pairs each")

        # Step 4 & 5: Streaming batch processing (extract features + predict immediately)
        print(f"  [4/5] Processing batches with GPU (extract + predict)...")

        # Enable GPU predictor once
        import xgboost as xgb
        try:
            self.join_model.set_param({'predictor': 'gpu_predictor', 'device': f'cuda:{self.config.GPU_ID}'})
            using_gpu_predict = True
        except:
            using_gpu_predict = False

        feature_names = [
            'absdiff_mean', 'absdiff_median', 'absdiff_max', 'absdiff_min', 'absdiff_std',
            'reldiff_mean', 'reldiff_median',
            'zdiff_mean', 'zdiff_max',
            'pcdiff_mean', 'pcdiff_sq_mean',
            'L1_raw', 'L2_raw',
            'L1_z', 'L2_z',
            'L1_pct', 'L2_pct',
            'correlation',
            'ratio_mean', 'ratio_median', 'ratio_std',
            'sign_agreement',
            'n_features',
            'n_both_zero',
            'n_close', 'n_very_close'
        ]

        # Store only matches above threshold (much smaller than all features)
        filtered_matches = []
        total_pairs_processed = 0

        start_time = time.time()

        # Process pairs in batches: extract features + predict + filter immediately
        for batch_idx in tqdm(range(num_batches), desc="  → GPU batches", unit="batch", ncols=80,
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):

            # Calculate start and end pair indices for this batch
            pair_start = batch_idx * batch_size
            pair_end = min(pair_start + batch_size, total_pairs)

            # Generate indices for this batch using arithmetic
            pair_indices = np.arange(pair_start, pair_end, dtype=np.int64)
            batch_indices_a = (pair_indices // n2).astype(np.int32)
            batch_indices_b = (pair_indices % n2).astype(np.int32)

            # Index into data arrays
            rows_a = df1_values[batch_indices_a]
            rows_b = df2_values[batch_indices_b]

            # Extract features on GPU (vectorized)
            batch_features = self._extract_features_batch_gpu(rows_a, rows_b)

            # Filter out NaN features
            valid_mask = ~np.isnan(batch_features).any(axis=1)
            if not valid_mask.any():
                continue

            valid_features = batch_features[valid_mask]
            valid_indices_a = batch_indices_a[valid_mask]
            valid_indices_b = batch_indices_b[valid_mask]

            total_pairs_processed += len(valid_features)

            # Predict immediately on this batch
            dmatrix = xgb.DMatrix(valid_features, feature_names=feature_names)
            batch_probs = self.join_model.predict(dmatrix)

            # Filter by threshold and store only matches
            threshold = self.config.JOIN_ROW_THRESHOLD
            above_threshold = batch_probs >= threshold

            for idx in np.where(above_threshold)[0]:
                filtered_matches.append((
                    int(valid_indices_a[idx]),
                    int(valid_indices_b[idx]),
                    float(batch_probs[idx])
                ))

        total_time = time.time() - start_time
        print(f"  → Processed {total_pairs_processed:,} pairs in {total_time:.1f}s")
        print(f"  → GPU throughput: {total_pairs_processed/total_time:,.0f} pairs/second")
        print(f"  → Found {len(filtered_matches):,} pairs above threshold ({threshold})")

        if len(filtered_matches) == 0:
            print("  ✗ No pairs passed threshold")
            return []

        # Step 5: Limited many-to-many assignment
        max_matches = self.config.MAX_MATCHES_PER_ROW

        if max_matches == 0:
            print(f"  [5/5] Using unlimited many-to-many matching...")
            final_matches = filtered_matches
        else:
            print(f"  [5/5] Performing limited many-to-many assignment (max {max_matches} per row)...")

            # Sort candidates by probability
            filtered_matches.sort(key=lambda x: x[2], reverse=True)

            final_matches = []
            matches_per_df1 = {}
            matches_per_df2 = {}

            start_time = time.time()

            for i, j, prob in tqdm(filtered_matches, desc="  → Limited assignment", unit="pair", ncols=80,
                                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]'):
                count_i = matches_per_df1.get(i, 0)
                count_j = matches_per_df2.get(j, 0)

                if count_i < max_matches and count_j < max_matches:
                    final_matches.append((i, j, prob))
                    matches_per_df1[i] = count_i + 1
                    matches_per_df2[j] = count_j + 1

            assignment_time = time.time() - start_time

            print(f"  → After limited assignment: {len(final_matches):,} pairs ({assignment_time:.1f}s)")
            print(f"  → Assignment rate: {len(filtered_matches)/assignment_time:,.0f} candidates/second")

            unique_df1 = len(matches_per_df1)
            unique_df2 = len(matches_per_df2)
            print(f"  → Coverage: {unique_df1}/{n1} rows from df1, {unique_df2}/{n2} rows from df2")

        if len(final_matches) > 0:
            avg_prob = np.mean([prob for _, _, prob in final_matches])
            print(f"  → Average match probability: {avg_prob:.3f}")

        print(f"  ✓ Compatible row finding complete (GPU)")

        return final_matches

    def _find_compatible_rows_cpu(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame
    ) -> List[Tuple[int, int, float]]:
        """
        CPU version (original implementation)
        Uses chunked parallelization
        """
        print("\nFinding compatible rows between datasets (CPU MODE):")

        # Step 1: Align columns and extract numeric data
        print("  [1/5] Aligning columns...")
        df1_numeric, df2_numeric = self._align_columns(df1, df2)
        
        if df1_numeric.empty or df2_numeric.empty:
            print("  ✗ No numeric columns to compare")
            return []
        
        n1, n2 = len(df1_numeric), len(df2_numeric)
        total_pairs = n1 * n2
        
        print(f"  → Processing {n1:,} × {n2:,} = {total_pairs:,} row pairs")
        print(f"  → Aligned to {df1_numeric.shape[1]} numeric features")
        
        # Step 2: Pre-convert to NumPy (avoid pandas overhead)
        print("  [2/5] Converting to NumPy arrays...")
        df1_values = df1_numeric.values
        df2_values = df2_numeric.values
        
        # Step 3: Chunked feature extraction (PARALLEL)
        print(f"  [3/5] Extracting features (using {self.config.N_JOBS} cores)...")
        
        chunks = self._create_chunks(n1, n2)
        print(f"  → Created {len(chunks)} chunks for parallel processing")
        
        start_time = time.time()
        
        # Parallel processing with progress bar
        chunk_results = Parallel(n_jobs=self.config.N_JOBS)(
            delayed(self._process_chunk)(
                df1_values, df2_values, start_i, end_i, start_j, end_j
            )
            for start_i, end_i, start_j, end_j in tqdm(
                chunks,
                desc="  → Feature extraction",
                unit="chunk",
                ncols=80
            )
        )
        
        # Flatten results from all chunks
        all_results = []
        for chunk_result in chunk_results:
            all_results.extend(chunk_result)
        
        feature_time = time.time() - start_time
        print(f"  → Extracted features for {len(all_results):,} pairs in {feature_time:.1f}s")
        
        if len(all_results) == 0:
            print("  ✗ No valid feature pairs (all contained NaN)")
            return []
        
        # Step 4: Batch XGBoost prediction
        print("  [4/5] Running XGBoost prediction...")
        
        # Stack all features into single array
        all_features = np.array([feat for feat, i, j in all_results])  # (N, 26)
        all_indices = [(i, j) for feat, i, j in all_results]  # List of (i, j)
        
        start_time = time.time()
        
        # SINGLE batch prediction with proper feature names
        import xgboost as xgb
        
        # Feature names in EXACT order model expects
        feature_names = [
            'absdiff_mean', 'absdiff_median', 'absdiff_max', 'absdiff_min', 'absdiff_std',
            'reldiff_mean', 'reldiff_median',
            'zdiff_mean', 'zdiff_max',
            'pcdiff_mean', 'pcdiff_sq_mean',
            'L1_raw', 'L2_raw',
            'L1_z', 'L2_z',
            'L1_pct', 'L2_pct',
            'correlation',
            'ratio_mean', 'ratio_median', 'ratio_std',
            'sign_agreement',
            'n_features',
            'n_both_zero',
            'n_close', 'n_very_close'
        ]
        
        # Verify feature count
        if all_features.shape[1] != len(feature_names):
            print(f"  ✗ Error: Expected {len(feature_names)} features, got {all_features.shape[1]}")
            return []
        
        # Create DMatrix with feature names
        dmatrix = xgb.DMatrix(all_features, feature_names=feature_names)
        all_probs = self.join_model.predict(dmatrix)  # (N,)
        
        pred_time = time.time() - start_time
        print(f"  → Predicted {len(all_probs):,} pairs in {pred_time:.1f}s")
        
        # Filter by threshold
        threshold = self.config.JOIN_ROW_THRESHOLD  # 0.44
        above_threshold = all_probs >= threshold
        
        filtered_probs = all_probs[above_threshold]
        filtered_indices = [all_indices[i] for i in range(len(all_indices)) if above_threshold[i]]
        
        print(f"  → {len(filtered_probs):,} pairs above threshold ({threshold})")
        
        if len(filtered_probs) == 0:
            print("  ✗ No pairs passed threshold")
            return []
        
        # Step 5: Limited many-to-many assignment
        max_matches = self.config.MAX_MATCHES_PER_ROW
        
        if max_matches == 0:
            # Unlimited many-to-many
            print(f"  [5/5] Using unlimited many-to-many matching...")
            final_matches = [(i, j, prob) for (i, j), prob in zip(filtered_indices, filtered_probs)]
        else:
            # Limited many-to-many
            print(f"  [5/5] Performing limited many-to-many assignment (max {max_matches} per row)...")
            
            # Create list of (i, j, prob) and sort by probability descending
            candidates = [(i, j, prob) for (i, j), prob in zip(filtered_indices, filtered_probs)]
            candidates.sort(key=lambda x: x[2], reverse=True)
            
            # Limited many-to-many assignment
            final_matches = []
            matches_per_df1 = {}
            matches_per_df2 = {}
            
            start_time = time.time()
            
            for i, j, prob in tqdm(
                candidates,
                desc="  → Limited assignment",
                unit="pair",
                ncols=80
            ):
                count_i = matches_per_df1.get(i, 0)
                count_j = matches_per_df2.get(j, 0)
                
                if count_i < max_matches and count_j < max_matches:
                    final_matches.append((i, j, prob))
                    matches_per_df1[i] = count_i + 1
                    matches_per_df2[j] = count_j + 1
            
            assignment_time = time.time() - start_time
            
            print(f"  → After limited assignment: {len(final_matches):,} pairs ({assignment_time:.1f}s)")
            
            # Report coverage statistics
            unique_df1 = len(matches_per_df1)
            unique_df2 = len(matches_per_df2)
            print(f"  → Coverage: {unique_df1}/{n1} rows from df1, {unique_df2}/{n2} rows from df2")
        
        if len(final_matches) > 0:
            avg_prob = np.mean([prob for _, _, prob in final_matches])
            print(f"  → Average match probability: {avg_prob:.3f}")
        
        print(f"  ✓ Compatible row finding complete")
        
        return final_matches
    
    def check_compatibility(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        denominator: int
    ) -> Tuple[bool, float, pd.DataFrame]:
        """
        Check if two dataframes can be joined based on row compatibility
        
        Args:
            df1: First dataframe
            df2: Second dataframe
            denominator: min(len(df1), len(df2)) for retention calculation
        
        Returns:
            (compatible, retention, joined_df)
            - compatible: True if retention >= threshold
            - retention: Based on unique row coverage (not total pairs)
            - joined_df: Horizontally joined dataframe (empty if not compatible)
        """
        # Find compatible rows (now returns limited many-to-many matches)
        matches = self.find_compatible_rows(df1, df2)
        
        if not matches:
            return False, 0.0, pd.DataFrame()
        
        # Calculate retention based on output rows (can be > 1.0 for many-to-many)
        unique_df1_rows = len(set(i for i, j, prob in matches))
        unique_df2_rows = len(set(j for i, j, prob in matches))
        output_rows = len(matches)

        # Retention = number of output rows / denominator (can exceed 1.0 for many-to-many)
        retention = output_rows / denominator if denominator > 0 else 0.0

        print(f"\n  Matched pairs: {output_rows:,}")
        print(f"    → Unique df1 rows matched: {unique_df1_rows}/{len(df1)}")
        print(f"    → Unique df2 rows matched: {unique_df2_rows}/{len(df2)}")
        print(f"    → Retention: {output_rows} / {denominator} = {retention:.3f}")
        
        # Check retention threshold
        compatible = retention >= self.config.JOIN_RETENTION_THRESHOLD
        
        if compatible:
            print(f"  ✓ JOIN compatible (retention {retention:.3f} ≥ {self.config.JOIN_RETENTION_THRESHOLD})")
            
            # Execute join
            print(f"\n  Executing join...")
            joined_df = self.execute_join(df1, df2, matches)
            print(f"    → Joined: {joined_df.shape[0]} rows × {joined_df.shape[1]} columns")
            
            return True, retention, joined_df
        else:
            print(f"  ✗ JOIN not compatible (retention {retention:.3f} < {self.config.JOIN_RETENTION_THRESHOLD})")
            return False, retention, pd.DataFrame()
    
    def execute_join(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        matches: List[Tuple[int, int, float]]
    ) -> pd.DataFrame:
        """
        Execute join operation by concatenating matched rows horizontally
        Adds suffixes (_x, _y) for duplicate column names
        
        With many-to-many matching, the same row may appear multiple times
        
        Args:
            df1: First dataframe
            df2: Second dataframe
            matches: List of (idx_a, idx_b, probability) tuples
        
        Returns:
            Joined dataframe with matched rows concatenated horizontally
        """
        if not matches:
            return pd.DataFrame()
        
        # Build list of joined rows
        joined_rows = []
        
        for i, j, prob in matches:
            row1 = df1.iloc[i].to_dict()
            row2 = df2.iloc[j].to_dict()
            
            # Merge dictionaries, add suffix to duplicates
            joined_row = {}
            
            # Add df1 columns
            for col, val in row1.items():
                if col in row2:
                    joined_row[f"{col}_x"] = val  # Add suffix for duplicate
                else:
                    joined_row[col] = val  # No suffix needed
            
            # Add df2 columns
            for col, val in row2.items():
                if col in row1:
                    joined_row[f"{col}_y"] = val  # Add suffix for duplicate
                else:
                    joined_row[col] = val  # No suffix needed
            
            joined_rows.append(joined_row)
        
        # Create result dataframe
        result = pd.DataFrame(joined_rows)
        
        return result
    
    def stage_1(
    self,
    dataframes: List[pd.DataFrame],
    denominator: int
) -> Tuple[List[pd.DataFrame], List[Dict], bool]:
        """
        Stage 1: Initial pairwise joins
        
        For each dataframe, find its best compatible partner among remaining dataframes
        A dataframe can appear in multiple joins
        
        Args:
            dataframes: List of dataframes after UNION stage
            denominator: min row count for retention calculation
        
        Returns:
            (joined_dataframes, operations_log, any_joins_succeeded)
        """
        operations = []
        outputs = []
        created_pairs = set()  # Track which (i,j) pairs we've already created
        any_joins_succeeded = False  # Track if ANY join succeeded
        
        print("\n" + "=" * 70)
        print("JOIN STAGE 1: Initial Pairwise Joins")
        print("=" * 70)
        
        # Process each dataframe sequentially
        for i, df_i in enumerate(dataframes):
            print(f"\n{'─' * 70}")
            print(f"Processing Group {i} ({df_i.shape[0]} rows × {df_i.shape[1]} cols)")
            print(f"{'─' * 70}")
            
            best_j = None
            best_retention = 0.0
            best_joined = None
            
            # Find best partner among remaining dataframes (only check j > i to avoid duplicates)
            for j in range(i + 1, len(dataframes)):  # ← FIXED: Only check j > i
                df_j = dataframes[j]
                
                # Check if this pair already processed (shouldn't happen now, but keep as safety)
                if (i, j) in created_pairs or (j, i) in created_pairs:
                    print(f"\n  Skipping Group {i} × Group {j} (already processed)")
                    continue
                
                print(f"\n  Checking Group {i} × Group {j}:")
                
                # Calculate compatibility
                compatible, retention, joined = self.check_compatibility(df_i, df_j, denominator)
                
                if compatible and retention > best_retention:
                    best_j = j
                    best_retention = retention
                    best_joined = joined
            
            # If found compatible partner, create join
            if best_j is not None:
                print(f"\n  ✓ Best partner for Group {i}: Group {best_j} (retention: {best_retention:.3f})")
                outputs.append(best_joined)
                created_pairs.add((i, best_j))
                any_joins_succeeded = True  # Mark that at least one join succeeded
                
                # Log operation
                operations.append({
                    'dataframes': [i, best_j],
                    'compatible': True,
                    'retention': best_retention,
                    'matched_rows': len(best_joined),
                    'result_shape': best_joined.shape
                })
            else:
                print(f"\n  ✗ No compatible partner found for Group {i}")
        
        # Add any dataframes that never appeared in any join
        for i, df in enumerate(dataframes):
            appears_in_join = any(i in pair for pair in created_pairs)
            if not appears_in_join:
                print(f"\n  → Group {i} kept separate (no compatible partners)")
                outputs.append(df)
                
                operations.append({
                    'dataframes': [i],
                    'compatible': False,
                    'retention': 0.0,
                    'matched_rows': 0,
                    'result_shape': df.shape
                })
        
        print(f"\n{'=' * 70}")
        print(f"STAGE 1 COMPLETE: {len(dataframes)} groups → {len(outputs)} outputs")
        if any_joins_succeeded:
            print(f"  ✓ At least one join succeeded")
        else:
            print(f"  ✗ No joins succeeded (all groups incompatible)")
        print(f"{'=' * 70}")
        
        return outputs, operations, any_joins_succeeded
    
    def stage_2(
        self,
        dataframes: List[pd.DataFrame],
        denominator: int
    ) -> Tuple[List[pd.DataFrame], List[Dict]]:
        """
        Stage 2: Join the joined groups
        
        Try to join outputs from Stage 1 with each other
        
        Args:
            dataframes: List of dataframes from Stage 1
            denominator: min row count for retention calculation
        
        Returns:
            (final_dataframes, operations_log)
        """
        operations = []
        
        print("\n" + "=" * 70)
        print("JOIN STAGE 2: Join the Joined Groups")
        print("=" * 70)
        
        if len(dataframes) <= 1:
            print("\n  Only 1 dataframe remaining - skipping Stage 2")
            return dataframes, operations
        
        # Try all combinations
        outputs = []
        processed = set()
        
        for i in range(len(dataframes)):
            for j in range(i + 1, len(dataframes)):
                if i in processed or j in processed:
                    continue
                
                df_i = dataframes[i]
                df_j = dataframes[j]
                
                print(f"\n{'─' * 70}")
                print(f"Attempting to join Stage1 output {i} × {j}")
                print(f"{'─' * 70}")
                
                compatible, retention, joined = self.check_compatibility(df_i, df_j, denominator)
                
                operations.append({
                    'dataframes': [i, j],
                    'compatible': compatible,
                    'retention': retention,
                    'matched_rows': len(joined) if compatible else 0,
                    'result_shape': joined.shape if compatible else None,
                    'combination': f"Stage1[{i}] × Stage1[{j}]"
                })
                
                if compatible:
                    outputs.append(joined)
                    processed.add(i)
                    processed.add(j)
        
        # Add any dataframes that weren't joined
        for i, df in enumerate(dataframes):
            if i not in processed:
                print(f"\n  → Stage1 output {i} kept separate")
                outputs.append(df)
        
        print(f"\n{'=' * 70}")
        print(f"STAGE 2 COMPLETE: {len(dataframes)} groups → {len(outputs)} final outputs")
        print(f"{'=' * 70}")
        
        return outputs, operations