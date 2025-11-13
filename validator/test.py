"""
Test script for optimized JOIN implementation
Validates correctness and measures performance
"""

import pandas as pd
import numpy as np
import time
import logging
import sys
from datetime import datetime
from validator import Validator

# Setup logging to both file and console
log_filename = 'test_output.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Log test start
logger.info("=" * 70)
logger.info("OPTIMIZED VALIDATOR TEST STARTED")
logger.info(f"Timestamp: {datetime.now()}")
logger.info("=" * 70)

print("=" * 70)
print("OPTIMIZED VALIDATOR TEST")
print("=" * 70)

# Load datasets
print("\nLoading datasets...")
logger.info("Loading datasets...")
try:
    df1 = pd.read_csv('../../Datasets/Join +/Test 2/DatasetX.csv')
    df2 = pd.read_csv('../../Datasets/Join +/Test 2/DatasetY.csv')

    msg1 = f"✓ Dataset 1: {df1.shape[0]:,} rows × {df1.shape[1]} columns"
    msg2 = f"✓ Dataset 2: {df2.shape[0]:,} rows × {df2.shape[1]} columns"
    msg3 = f"  Total pairs to process: {df1.shape[0] * df2.shape[0]:,}"

    print(msg1)
    print(msg2)
    print(msg3)

    logger.info(msg1)
    logger.info(msg2)
    logger.info(msg3)

except FileNotFoundError as e:
    error_msg = f"✗ Error: {e}"
    print(error_msg)
    logger.error(error_msg)
    exit(1)

# Performance test
print("\n" + "=" * 70)
print("PERFORMANCE TEST")
print("=" * 70)
logger.info("PERFORMANCE TEST")

start_time = time.time()

validator = Validator()
output, report = validator.process([df1, df2])

elapsed = time.time() - start_time

print("\n" + "=" * 70)
print("PERFORMANCE RESULTS")
print("=" * 70)
result_msg = f"Total time: {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)"
print(result_msg)
logger.info(result_msg)


# Correctness validation
print("\n" + "=" * 70)
print("CORRECTNESS VALIDATION")
print("=" * 70)

print(f"\nNumber of output dataframes: {len(output)}")

for idx, df in enumerate(output):
    print(f"\nOutput DataFrame {idx}:")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)[:10]}" + ("..." if len(df.columns) > 10 else ""))
    
    # Check for duplicate column suffixes
    suffixed_cols = [col for col in df.columns if col.endswith('_x') or col.endswith('_y')]
    if suffixed_cols:
        print(f"  Duplicate columns (with suffixes): {len(suffixed_cols)}")
        print(f"    Examples: {suffixed_cols[:5]}")
    
    # Check data types
    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
    print(f"  Numeric columns: {numeric_cols}/{len(df.columns)}")

# Report summary
print("\n" + "=" * 70)
print("DETAILED REPORT")
print("=" * 70)
print(validator.get_summary(report))

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
logger.info("TEST COMPLETE")

saved_files = validator.save_outputs(output, output_dir="outputs", prefix="result")
save_msg = f"Saved files: {len(saved_files)}"
print(f"  {save_msg}")
logger.info(save_msg)

if saved_files:
    print("\nSaved files:")
    logger.info("Saved files:")
    for filepath in saved_files:
        file_msg = f"  • {filepath}"
        print(file_msg)
        logger.info(file_msg)

# Final log entry
logger.info("=" * 70)
logger.info("TEST SESSION COMPLETE")
logger.info("=" * 70)
logger.info("")  # Blank line for separation between test runs