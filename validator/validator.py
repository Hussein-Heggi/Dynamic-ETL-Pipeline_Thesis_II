"""
Main Validator Class - Orchestrates Union and Join operations
"""

import pandas as pd
from typing import List, Tuple, Dict, Optional
import logging

from .config import ValidatorConfig
from .union import UnionEngine
from .join import JoinEngine


class Validator:
    """
    Main Validator Component for ETL Pipeline
    
    Performs UNION (vertical concatenation) and JOIN (horizontal concatenation)
    operations on input dataframes based on ML model predictions.
    
    Usage:
        validator = Validator()
        output_dfs, report = validator.process(input_dataframes)
    
    Pipeline:
        1. Validate input
        2. UNION stage (sequential column matching)
        3. Check for early termination (all unioned)
        4. JOIN stage (2-stage row matching)
        5. Return results + detailed report
    """
    
    def __init__(self, config: Optional[ValidatorConfig] = None):
        """
        Initialize Validator
        
        Args:
            config: ValidatorConfig object (optional, uses defaults if None)
        """
        self.config = config or ValidatorConfig()
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Initialize engines
        self.logger.info("Initializing Validator...")
        self.union_engine = UnionEngine(self.config)
        self.join_engine = JoinEngine(self.config)
        self.logger.info("Validator initialized successfully")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('Validator')
        
        # Set level
        if self.config.VERBOSE:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(getattr(logging, self.config.LOG_LEVEL))
        
        # Add handler if not already present
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _validate_input(self, dataframes: List[pd.DataFrame]) -> None:
        """
        Validate input dataframes
        
        Checks:
        - Non-empty list
        - All elements are DataFrames
        - No empty dataframes
        - Max 4 dataframes limit
        
        Raises:
            ValueError: If validation fails
        """
        if not dataframes:
            raise ValueError("Input dataframe list is empty")
        
        if not isinstance(dataframes, list):
            raise ValueError(f"Expected list of DataFrames, got {type(dataframes)}")
        
        if len(dataframes) > self.config.MAX_DATAFRAMES:
            raise ValueError(
                f"Too many dataframes: {len(dataframes)} (max: {self.config.MAX_DATAFRAMES})"
            )
        
        for i, df in enumerate(dataframes):
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Element {i} is not a DataFrame: {type(df)}")
            
            if df.empty:
                raise ValueError(f"DataFrame {i} is empty")
        
        self.logger.debug(f"Input validation passed: {len(dataframes)} dataframes")
    
    def process(
        self,
        dataframes: List[pd.DataFrame]
    ) -> Tuple[List[pd.DataFrame], Dict]:
        """
        Main processing pipeline
        
        Args:
            dataframes: List of input DataFrames (max 4)
        
        Returns:
            output_dataframes: List of processed DataFrames
            report: Dict containing detailed operation metadata
        
        Pipeline:
            1. Validate input
            2. UNION stage (sequential)
            3. Early termination check
            4. JOIN stage (2-stage)
            5. Generate report
        """
        # 1. Validate input
        self._validate_input(dataframes)
        
        self.logger.info("=" * 60)
        self.logger.info(f"VALIDATOR PIPELINE STARTED")
        self.logger.info(f"Input: {len(dataframes)} dataframes")
        for i, df in enumerate(dataframes):
            self.logger.info(f"  DF{i}: {df.shape}")
        self.logger.info("=" * 60)
        
        # Initialize report
        report = {
            'input_count': len(dataframes),
            'input_shapes': [df.shape for df in dataframes],
            'union_operations': [],
            'join_operations': [],
            'early_termination': False,
            'output_count': 0,
            'output_shapes': []
        }
        
        # 2. UNION STAGE
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("UNION STAGE")
        self.logger.info("=" * 60)
        
        unioned_dfs, union_ops = self.union_engine.process(dataframes)
        report['union_operations'] = union_ops
        
        self.logger.info(f"UNION complete: {len(dataframes)} → {len(unioned_dfs)} dataframes")
        for i, df in enumerate(unioned_dfs):
            self.logger.info(f"  Group{i}: {df.shape}")
        
        # 3. Early termination check
        if len(unioned_dfs) == 1:
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("EARLY TERMINATION: All dataframes successfully unioned")
            self.logger.info("=" * 60)
            
            report['early_termination'] = True
            report['output_count'] = 1
            report['output_shapes'] = [unioned_dfs[0].shape]
            
            return unioned_dfs, report
        
        # 4. JOIN STAGE
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("JOIN STAGE")
        self.logger.info("=" * 60)
        
        # Calculate denominator (fixed for all join operations)
        denominator = min(df.shape[0] for df in unioned_dfs)
        self.logger.info(f"Join denominator: {denominator} rows")
        
        # Stage 1: Initial pairwise joins
        self.logger.info("")
        self.logger.info("-" * 60)
        self.logger.info("JOIN STAGE 1: Initial Pairwise Joins")
        self.logger.info("-" * 60)
        
        stage1_dfs, stage1_ops = self.join_engine.stage_1(unioned_dfs, denominator)
        
        self.logger.info(f"Stage 1 complete: {len(unioned_dfs)} → {len(stage1_dfs)} dataframes")
        for i, df in enumerate(stage1_dfs):
            self.logger.info(f"  Result{i}: {df.shape}")
        
        # Stage 2: Join the joined groups
        self.logger.info("")
        self.logger.info("-" * 60)
        self.logger.info("JOIN STAGE 2: Join the Joined Groups")
        self.logger.info("-" * 60)
        
        final_dfs, stage2_ops = self.join_engine.stage_2(stage1_dfs, denominator)
        
        self.logger.info(f"Stage 2 complete: {len(stage1_dfs)} → {len(final_dfs)} dataframes")
        for i, df in enumerate(final_dfs):
            self.logger.info(f"  Final{i}: {df.shape}")
        
        # Update report
        report['join_operations'] = {
            'stage_1': stage1_ops,
            'stage_2': stage2_ops
        }
        
        # 5. Finalize report
        report['output_count'] = len(final_dfs)
        report['output_shapes'] = [df.shape for df in final_dfs]
        
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("VALIDATOR PIPELINE COMPLETE")
        self.logger.info(f"Output: {len(final_dfs)} dataframe(s)")
        for i, df in enumerate(final_dfs):
            self.logger.info(f"  Output{i}: {df.shape}")
        self.logger.info("=" * 60)
        
        return final_dfs, report
    
    def get_summary(self, report: Dict) -> str:
        """
        Generate human-readable summary from report
        
        Args:
            report: Report dictionary from process()
        
        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("VALIDATOR SUMMARY")
        lines.append("=" * 60)
        
        # Input
        lines.append(f"\nInput: {report['input_count']} dataframes")
        for i, shape in enumerate(report['input_shapes']):
            lines.append(f"  DF{i}: {shape[0]} rows × {shape[1]} columns")
        
        # Union operations
        lines.append(f"\nUnion Operations: {len(report['union_operations'])}")
        for op in report['union_operations']:
            lines.append(f"  - {op['operation']}: score={op['score']:.3f}, result={op['result_shape']}")
        
        # Early termination
        if report['early_termination']:
            lines.append("\nEarly Termination: YES (all dataframes unioned)")
        else:
            lines.append("\nEarly Termination: NO")
            
            # Join operations
            join_ops = report['join_operations']
            
            lines.append(f"\nJoin Stage 1: {len(join_ops['stage_1'])} operations")
            for op in join_ops['stage_1']:
                lines.append(f"  - DFs {op['dataframes']}: retention={op['retention']:.3f}, "
                           f"matched={op['matched_rows']}, result={op['result_shape']}")
            
            lines.append(f"\nJoin Stage 2: {len(join_ops['stage_2'])} operations")
            for op in join_ops['stage_2']:
                lines.append(f"  - DFs {op['dataframes']}: retention={op['retention']:.3f}, "
                           f"matched={op['matched_rows']}, result={op['result_shape']}, "
                           f"combination={op['combination']}")
        
        # Output
        lines.append(f"\nOutput: {report['output_count']} dataframe(s)")
        for i, shape in enumerate(report['output_shapes']):
            lines.append(f"  Output{i}: {shape[0]} rows × {shape[1]} columns")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)