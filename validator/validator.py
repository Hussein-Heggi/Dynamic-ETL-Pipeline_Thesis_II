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
        4. JOIN stage 1 (pairwise row matching)
        5. CHECK: Skip Stage 2 if no joins succeeded in Stage 1
        6. JOIN stage 2 (join the joined groups)
        7. Return results + detailed report
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

    def _create_versions(self, dataframes: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """
        Create versioned outputs from dataframes with duplicate columns (_x and _y suffixes)

        For each dataframe:
        - If it has columns with _x and _y suffixes, create 2 versions:
          - Version 1: Keep _x columns, drop _y columns, rename _x → original
          - Version 2: Keep _y columns, drop _x columns, rename _y → original
        - If no duplicate columns exist, keep dataframe as is

        Args:
            dataframes: List of dataframes (may contain duplicate columns)

        Returns:
            List of versioned dataframes (may be larger than input)
        """
        versioned_dfs = []

        print("\n" + "=" * 70)
        print("CREATING VERSIONED OUTPUTS")
        print("=" * 70)
        print("\nSplitting dataframes with duplicate columns (_x/_y suffixes)...")

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("CREATING VERSIONED OUTPUTS")
        self.logger.info("=" * 60)

        for idx, df in enumerate(dataframes):
            # Identify columns with _x and _y suffixes
            x_cols = [col for col in df.columns if col.endswith('_x')]
            y_cols = [col for col in df.columns if col.endswith('_y')]

            # Check if we have duplicate columns
            has_duplicates = len(x_cols) > 0 and len(y_cols) > 0

            if has_duplicates:
                print(f"\n  DataFrame {idx}:")
                print(f"    → Found {len(x_cols)} duplicate column pairs")
                print(f"    → Creating 2 versions...")

                self.logger.info(f"DataFrame {idx}: Found {len(x_cols)} duplicate pairs, creating versions")

                # Get base column names (without suffixes)
                base_names_x = {col[:-2]: col for col in x_cols}  # Remove '_x'
                base_names_y = {col[:-2]: col for col in y_cols}  # Remove '_y'

                # Get non-duplicate columns (columns without _x or _y)
                all_suffixed = set(x_cols) | set(y_cols)
                non_duplicate_cols = [col for col in df.columns if col not in all_suffixed]

                # VERSION 1: Use _x columns
                version1 = df.copy()
                # Drop all _y columns
                version1 = version1.drop(columns=y_cols)
                # Rename _x columns to original names
                rename_map_x = {col: col[:-2] for col in x_cols}
                version1 = version1.rename(columns=rename_map_x)

                # VERSION 2: Use _y columns
                version2 = df.copy()
                # Drop all _x columns
                version2 = version2.drop(columns=x_cols)
                # Rename _y columns to original names
                rename_map_y = {col: col[:-2] for col in y_cols}
                version2 = version2.rename(columns=rename_map_y)

                versioned_dfs.append(version1)
                versioned_dfs.append(version2)

                print(f"    → Version 1: {version1.shape[0]} rows × {version1.shape[1]} columns (using _x values)")
                print(f"    → Version 2: {version2.shape[0]} rows × {version2.shape[1]} columns (using _y values)")

                self.logger.info(f"  Version 1: {version1.shape}")
                self.logger.info(f"  Version 2: {version2.shape}")
            else:
                # No duplicate columns, keep as is
                versioned_dfs.append(df)
                print(f"\n  DataFrame {idx}:")
                print(f"    → No duplicate columns found")
                print(f"    → Keeping original: {df.shape[0]} rows × {df.shape[1]} columns")

                self.logger.info(f"DataFrame {idx}: No duplicates, keeping original ({df.shape})")

        print(f"\n  VERSIONING COMPLETE: {len(dataframes)} input → {len(versioned_dfs)} output dataframes")
        print("=" * 70)

        self.logger.info(f"Versioning complete: {len(dataframes)} → {len(versioned_dfs)} dataframes")
        self.logger.info("=" * 60)

        return versioned_dfs

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
            4. JOIN stage 1 (pairwise)
            5. CHECK: Skip Stage 2 if no Stage 1 joins succeeded
            6. JOIN stage 2 (join the joined)
            7. Generate report
        """
        # 1. Validate input
        self._validate_input(dataframes)
        
        print("\n" + "=" * 70)
        print("VALIDATOR PIPELINE STARTED")
        print("=" * 70)
        print(f"\nInput: {len(dataframes)} dataframes")
        for i, df in enumerate(dataframes):
            print(f"  DF{i}: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"       Columns: {list(df.columns)[:5]}{' ...' if len(df.columns) > 5 else ''}")
        
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
            'join_operations': {},
            'early_termination': False,
            'stage_2_skipped': False,
            'output_count': 0,
            'output_shapes': []
        }
        
        # 2. UNION STAGE
        print("\n" + "=" * 70)
        print("STAGE 1: UNION (Vertical Concatenation - Similar Columns)")
        print("=" * 70)
        print("\nAttempting to stack dataframes with similar column structures...")
        
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("UNION STAGE")
        self.logger.info("=" * 60)
        
        unioned_dfs, union_ops = self.union_engine.process(dataframes)
        report['union_operations'] = union_ops
        
        print(f"\nUNION RESULT: {len(dataframes)} dataframes → {len(unioned_dfs)} groups")
        for i, df in enumerate(unioned_dfs):
            print(f"  Group{i}: {df.shape[0]} rows × {df.shape[1]} columns")
        
        self.logger.info(f"UNION complete: {len(dataframes)} → {len(unioned_dfs)} dataframes")
        for i, df in enumerate(unioned_dfs):
            self.logger.info(f"  Group{i}: {df.shape}")
        
        # 3. Early termination check
        if len(unioned_dfs) == 1:
            print("\n" + "=" * 70)
            print("EARLY TERMINATION")
            print("=" * 70)
            print("\n✓ All dataframes successfully unioned into one!")
            print(f"  Final output: {unioned_dfs[0].shape[0]} rows × {unioned_dfs[0].shape[1]} columns")
            print("\n  Skipping JOIN stage (not needed)")
            
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("EARLY TERMINATION: All dataframes successfully unioned")
            self.logger.info("=" * 60)
            
            report['early_termination'] = True
            report['output_count'] = 1
            report['output_shapes'] = [unioned_dfs[0].shape]

            # Create versioned outputs
            versioned_outputs = self._create_versions(unioned_dfs)

            # Update report with versioned counts
            report['output_count'] = len(versioned_outputs)
            report['output_shapes'] = [df.shape for df in versioned_outputs]

            print("\n" + "=" * 70)
            print("VALIDATOR PIPELINE COMPLETE")
            print("=" * 70)
            print(f"\nFinal Summary:")
            print(f"  Input:  {len(dataframes)} dataframes")
            print(f"  Output: {len(versioned_outputs)} dataframe(s)")
            print()
            for i, df in enumerate(versioned_outputs):
                print(f"  Output {i}: {df.shape[0]} rows × {df.shape[1]} columns")
            print("\n" + "=" * 70 + "\n")

            return versioned_outputs, report
        
        # 4. JOIN STAGE 1
        print("\n" + "=" * 70)
        print("STAGE 2: JOIN (Horizontal Concatenation - Compatible Rows)")
        print("=" * 70)
        print(f"\n{len(unioned_dfs)} groups remaining after union - attempting to join...")
        
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("JOIN STAGE")
        self.logger.info("=" * 60)
        
        # Calculate denominator (fixed for all join operations)
        denominator = min(df.shape[0] for df in unioned_dfs)
        print(f"\nJoin denominator (minimum row count): {denominator} rows")
        print("(Retention = unique_row_coverage / {})".format(denominator))
        
        self.logger.info(f"Join denominator: {denominator} rows")
        
        # Stage 1: Initial pairwise joins
        print("\n" + "-" * 70)
        print("JOIN STAGE 1: Initial Pairwise Joins")
        print("-" * 70)
        print("\nFinding best join partner for each group...")
        
        self.logger.info("")
        self.logger.info("-" * 60)
        self.logger.info("JOIN STAGE 1: Initial Pairwise Joins")
        self.logger.info("-" * 60)
        
        stage1_dfs, stage1_ops, stage1_success = self.join_engine.stage_1(unioned_dfs, denominator)
        
        print(f"\nSTAGE 1 RESULT: {len(unioned_dfs)} groups → {len(stage1_dfs)} groups")
        for i, df in enumerate(stage1_dfs):
            print(f"  Result{i}: {df.shape[0]} rows × {df.shape[1]} columns")
        
        self.logger.info(f"Stage 1 complete: {len(unioned_dfs)} → {len(stage1_dfs)} dataframes")
        for i, df in enumerate(stage1_dfs):
            self.logger.info(f"  Result{i}: {df.shape}")
        
        report['join_operations']['stage_1'] = stage1_ops
        
        # 5. CHECK: Skip Stage 2 if no joins succeeded in Stage 1
        if not stage1_success:
            print("\n" + "=" * 70)
            print("SKIPPING JOIN STAGE 2")
            print("=" * 70)
            print("\n✗ No compatible joins found in Stage 1")
            print("  All groups are incompatible with each other")
            print("  Stage 2 would not find any matches either")
            print("\n  Returning Stage 1 outputs as final results")
            
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("SKIPPING STAGE 2: No joins succeeded in Stage 1")
            self.logger.info("=" * 60)
            
            report['join_operations']['stage_2'] = []
            report['stage_2_skipped'] = True
            report['output_count'] = len(stage1_dfs)
            report['output_shapes'] = [df.shape for df in stage1_dfs]

            # Create versioned outputs
            versioned_outputs = self._create_versions(stage1_dfs)

            # Update report with versioned counts
            report['output_count'] = len(versioned_outputs)
            report['output_shapes'] = [df.shape for df in versioned_outputs]

            print("\n" + "=" * 70)
            print("VALIDATOR PIPELINE COMPLETE")
            print("=" * 70)
            print(f"\nFinal Summary:")
            print(f"  Input:  {len(dataframes)} dataframes")
            print(f"  Output: {len(versioned_outputs)} dataframe(s)")
            print()
            for i, df in enumerate(versioned_outputs):
                print(f"  Output {i}: {df.shape[0]} rows × {df.shape[1]} columns")
            print("\n" + "=" * 70 + "\n")

            return versioned_outputs, report
        
        # 6. Stage 2: Join the joined groups
        print("\n" + "-" * 70)
        print("JOIN STAGE 2: Join the Joined Groups")
        print("-" * 70)
        print("\nAttempting to combine Stage 1 results...")
        
        self.logger.info("")
        self.logger.info("-" * 60)
        self.logger.info("JOIN STAGE 2: Join the Joined Groups")
        self.logger.info("-" * 60)
        
        final_dfs, stage2_ops = self.join_engine.stage_2(stage1_dfs, denominator)
        
        print(f"\nSTAGE 2 RESULT: {len(stage1_dfs)} groups → {len(final_dfs)} final groups")
        for i, df in enumerate(final_dfs):
            print(f"  Final{i}: {df.shape[0]} rows × {df.shape[1]} columns")
        
        self.logger.info(f"Stage 2 complete: {len(stage1_dfs)} → {len(final_dfs)} dataframes")
        for i, df in enumerate(final_dfs):
            self.logger.info(f"  Final{i}: {df.shape}")
        
        # Update report
        report['join_operations']['stage_2'] = stage2_ops
        report['stage_2_skipped'] = False

        # 7. Create versioned outputs
        versioned_outputs = self._create_versions(final_dfs)

        # 8. Finalize report
        report['output_count'] = len(versioned_outputs)
        report['output_shapes'] = [df.shape for df in versioned_outputs]

        print("\n" + "=" * 70)
        print("VALIDATOR PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\nFinal Summary:")
        print(f"  Input:  {len(dataframes)} dataframes")
        print(f"  Output: {len(versioned_outputs)} dataframe(s)")
        print()
        for i, df in enumerate(versioned_outputs):
            print(f"  Output {i}: {df.shape[0]} rows × {df.shape[1]} columns")
        print("\n" + "=" * 70 + "\n")

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("VALIDATOR PIPELINE COMPLETE")
        self.logger.info(f"Output: {len(versioned_outputs)} dataframe(s)")
        for i, df in enumerate(versioned_outputs):
            self.logger.info(f"  Output{i}: {df.shape}")
        self.logger.info("=" * 60)

        return versioned_outputs, report
    
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
                if op['compatible']:
                    lines.append(f"  - DFs {op['dataframes']}: retention={op['retention']:.3f}, "
                               f"matched={op['matched_rows']}, result={op['result_shape']}")
                else:
                    lines.append(f"  - DF {op['dataframes'][0]}: no compatible partner")
            
            # Stage 2
            if report.get('stage_2_skipped', False):
                lines.append(f"\nJoin Stage 2: SKIPPED (no Stage 1 joins succeeded)")
            else:
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
    
    def save_outputs(
        self,
        dataframes: List[pd.DataFrame],
        output_dir: str = "outputs",
        prefix: str = "output"
    ) -> List[str]:
        """
        Save output dataframes to CSV files
        
        Args:
            dataframes: List of output DataFrames to save
            output_dir: Directory to save files (default: "outputs")
            prefix: Prefix for filenames (default: "output")
        
        Returns:
            List of saved file paths
        
        Creates files like:
            outputs/output_0.csv
            outputs/output_1.csv
            ...
        """
        from pathlib import Path

        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        print("\n" + "=" * 70)
        print("SAVING OUTPUTS TO CSV")
        print("=" * 70)
        print(f"\nOutput directory: {output_path.absolute()}")
        
        for i, df in enumerate(dataframes):
            # Generate filename
            filename = f"{prefix}_{i}.csv"
            filepath = output_path / filename
            
            # Save to CSV
            try:
                df.to_csv(filepath, index=False)
                saved_files.append(str(filepath))
                
                print(f"\n  ✓ Saved output {i}:")
                print(f"    File: {filename}")
                print(f"    Path: {filepath.absolute()}")
                print(f"    Size: {df.shape[0]} rows × {df.shape[1]} columns")
                
                self.logger.info(f"Saved output {i} to {filepath}")
                
            except Exception as e:
                print(f"\n  ✗ Failed to save output {i}: {e}")
                self.logger.error(f"Failed to save output {i}: {e}")
        
        print(f"\n{'=' * 70}")
        print(f"SAVED {len(saved_files)}/{len(dataframes)} FILES")
        print(f"{'=' * 70}\n")
        
        return saved_files