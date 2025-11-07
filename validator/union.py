"""
Union Engine: Column matching and vertical concatenation
Includes FinBERT embeddings, hybrid scoring, and sequential union algorithm
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import pickle
import logging
from difflib import SequenceMatcher
import torch
from transformers import AutoTokenizer, AutoModel

from .config import ValidatorConfig


class UnionEngine:
    """Handles all Union operations"""
    
    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.logger = logging.getLogger('UnionEngine')
        
        # Load models
        self.union_model = self._load_union_model()
        self.tokenizer, self.finbert_model = self._load_finbert()
    
    def _load_union_model(self):
        """Load pre-trained Union XGBoost model"""
        try:
            with open(self.config.UNION_MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            self.logger.info("Union model loaded successfully")
            return model
        except FileNotFoundError:
            self.logger.warning(f"Union model not found at {self.config.UNION_MODEL_PATH}")
            return None
    
    def _load_finbert(self):
        """Load FinBERT tokenizer and model"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.FINBERT_MODEL_NAME)
            model = AutoModel.from_pretrained(self.config.FINBERT_MODEL_NAME)
            model.eval()  # Set to evaluation mode
            self.logger.info("FinBERT loaded successfully")
            return tokenizer, model
        except Exception as e:
            self.logger.warning(f"Failed to load FinBERT: {e}")
            return None, None
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate FinBERT embedding for a single text"""
        if self.tokenizer is None or self.finbert_model is None:
            # Fallback: return zero vector
            return np.zeros(768)
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.finbert_model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        
        return embedding
    
    def compute_name_similarity(self, name1: str, name2: str) -> float:
        """
        Compute string similarity between column names
        Uses SequenceMatcher for Levenshtein-like similarity
        """
        # Normalize names
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()
        
        # Compute similarity ratio
        similarity = SequenceMatcher(None, n1, n2).ratio()
        
        return similarity
    
    def compute_hybrid_score(self, name1: str, name2: str) -> float:
        """
        Compute hybrid compatibility score:
        85% name similarity + 15% model probability
        """
        # Name similarity
        name_sim = self.compute_name_similarity(name1, name2)
        
        # Model probability (if model available)
        if self.union_model is not None:
            # Get embeddings
            emb1 = self.get_embedding(name1)
            emb2 = self.get_embedding(name2)
            
            # Concatenate embeddings as features
            features = np.concatenate([emb1, emb2]).reshape(1, -1)
            
            # Predict probability
            try:
                model_prob = self.union_model.predict_proba(features)[0, 1]
            except:
                model_prob = 0.5  # Fallback
        else:
            model_prob = 0.5  # Fallback if no model
        
        # Hybrid score
        hybrid_score = (self.config.UNION_NAME_WEIGHT * name_sim + 
                       self.config.UNION_MODEL_WEIGHT * model_prob)
        
        return hybrid_score
    
    def find_column_mapping(self, cols_a: List[str], cols_b: List[str]) -> Tuple[Dict, float]:
        """
        Find best column mapping between two dataframes
        
        Returns:
        - mapping: Dict[col_b -> col_a] for matching columns
        - avg_score: Average compatibility score
        """
        mapping = {}
        scores = []
        
        # For each column in B, find best match in A
        for col_b in cols_b:
            best_col_a = None
            best_score = 0.0
            
            for col_a in cols_a:
                score = self.compute_hybrid_score(col_a, col_b)
                
                if score > best_score and score >= self.config.UNION_THRESHOLD:
                    best_score = score
                    best_col_a = col_a
            
            if best_col_a is not None:
                mapping[col_b] = best_col_a
                scores.append(best_score)
        
        # Compute average score (coverage-weighted)
        if scores:
            avg_score = np.mean(scores)
        else:
            avg_score = 0.0
        
        return mapping, avg_score
    
    def compute_coverage(self, mapping: Dict, cols_a: List[str], cols_b: List[str]) -> float:
        """
        Compute coverage score using harmonic mean
        Coverage = harmonic_mean(len(mapping)/len(cols_a), len(mapping)/len(cols_b))
        """
        if len(mapping) == 0:
            return 0.0
        
        coverage_a = len(mapping) / len(cols_a) if len(cols_a) > 0 else 0
        coverage_b = len(mapping) / len(cols_b) if len(cols_b) > 0 else 0
        
        # Harmonic mean
        if coverage_a + coverage_b == 0:
            return 0.0
        
        harmonic_mean = 2 * (coverage_a * coverage_b) / (coverage_a + coverage_b)
        
        return harmonic_mean
    
    def check_compatibility(
        self, 
        df1: pd.DataFrame, 
        df2: pd.DataFrame
    ) -> Tuple[bool, float, Dict]:
        """
        Check if two dataframes can be unioned
        
        Returns:
        - compatible: bool
        - score: float (coverage score)
        - column_mapping: Dict[col_df2 -> col_df1]
        """
        cols_a = df1.columns.tolist()
        cols_b = df2.columns.tolist()
        
        # Find column mapping
        mapping, avg_score = self.find_column_mapping(cols_a, cols_b)
        
        # Compute coverage
        coverage = self.compute_coverage(mapping, cols_a, cols_b)
        
        # Check if compatible (good coverage)
        # Use a threshold of 0.5 for coverage (at least 50% columns match)
        compatible = coverage >= 0.5
        
        self.logger.debug(f"Union check: coverage={coverage:.3f}, avg_score={avg_score:.3f}, compatible={compatible}")
        
        return compatible, coverage, mapping
    
    def execute_union(
        self, 
        df1: pd.DataFrame, 
        df2: pd.DataFrame,
        column_mapping: Dict
    ) -> pd.DataFrame:
        """
        Execute union operation with column standardization
        
        Steps:
        1. Rename df2 columns according to mapping
        2. Align columns (use df1 column names as standard)
        3. Concatenate vertically
        4. Remove duplicates
        """
        # Create copy of df2
        df2_copy = df2.copy()
        
        # Rename columns in df2 according to mapping
        df2_copy.rename(columns=column_mapping, inplace=True)
        
        # Get all unique columns from both dataframes
        all_cols = list(set(df1.columns.tolist() + df2_copy.columns.tolist()))
        
        # Add missing columns with NaN
        for col in all_cols:
            if col not in df1.columns:
                df1[col] = np.nan
            if col not in df2_copy.columns:
                df2_copy[col] = np.nan
        
        # Ensure same column order
        df2_copy = df2_copy[df1.columns]
        
        # Concatenate vertically
        result = pd.concat([df1, df2_copy], axis=0, ignore_index=True)
        
        # Remove exact duplicates
        result = result.drop_duplicates()
        
        self.logger.info(f"Union executed: {df1.shape[0]} + {df2.shape[0]} → {result.shape[0]} rows")
        
        return result
    
    def process(
        self, 
        dataframes: List[pd.DataFrame]
    ) -> Tuple[List[pd.DataFrame], List[Dict]]:
        """
        Sequential union algorithm:
        
        1. Try to union A with B → AB (if compatible)
        2. Try AB with C → ABC (if compatible)
        3. If ABC not compatible with D, keep D separate
        4. Try remaining dataframes with each other recursively
        
        Returns:
        - List of unioned dataframe groups
        - List of operation reports
        """
        if len(dataframes) == 0:
            return [], []
        
        if len(dataframes) == 1:
            return dataframes, []
        
        operations = []
        remaining = dataframes.copy()
        result_groups = []
        
        while remaining:
            # Start with first dataframe
            current = remaining.pop(0)
            current_name = f"DF{len(result_groups)}"
            
            # Try to union with remaining dataframes sequentially
            i = 0
            while i < len(remaining):
                next_df = remaining[i]
                
                # Check compatibility
                compatible, score, mapping = self.check_compatibility(current, next_df)
                
                if compatible:
                    # Execute union
                    current = self.execute_union(current, next_df, mapping)
                    
                    # Log operation
                    operations.append({
                        'operation': 'union',
                        'group': current_name,
                        'score': score,
                        'result_shape': current.shape
                    })
                    
                    # Remove the unioned dataframe
                    remaining.pop(i)
                    
                    self.logger.info(f"Unioned into {current_name}: {current.shape}")
                else:
                    # Not compatible, try next
                    i += 1
            
            # Add current (possibly unioned) dataframe to results
            result_groups.append(current)
        
        self.logger.info(f"Union complete: {len(dataframes)} → {len(result_groups)} groups")
        
        return result_groups, operations