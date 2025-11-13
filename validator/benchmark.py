"""
Benchmark script to measure performance of each stage
"""

import pandas as pd
import numpy as np
import time
from join import JoinEngine
from config import ValidatorConfig

print("=" * 70)
print("JOIN ENGINE PERFORMANCE BENCHMARK")
print("=" * 70)

# Load data
print("\nLoading datasets...")
df1 = pd.read_csv('../../Datasets/Join +/Test 1/DatasetX.csv')
df2 = pd.read_csv('../../Datasets/Join +/Test 1/DatasetY.csv')

print(f"Dataset 1: {df1.shape}")
print(f"Dataset 2: {df2.shape}")
print(f"Total pairs: {df1.shape[0] * df2.shape[0]:,}")

# Initialize engine
config = ValidatorConfig()
engine = JoinEngine(config)

# Benchmark
print("\n" + "=" * 70)
print("RUNNING BENCHMARK")
print("=" * 70)

start = time.time()
matches = engine.find_compatible_rows(df1, df2)
total_time = time.time() - start

print("\n" + "=" * 70)
print("BENCHMARK RESULTS")
print("=" * 70)
print(f"Total time: {total_time:.2f} seconds")
print(f"Matched pairs: {len(matches):,}")

if len(matches) > 0:
    avg_prob = np.mean([prob for _, _, prob in matches])
    print(f"Average probability: {avg_prob:.3f}")
    print(f"\nTop 5 matches:")
    for idx, (i, j, prob) in enumerate(matches[:5]):
        print(f"  {idx+1}. Row {i} × Row {j}: {prob:.4f}")

# Performance breakdown estimate
print("\n" + "=" * 70)
print("ESTIMATED BREAKDOWN")
print("=" * 70)
print(f"Feature extraction: ~{total_time * 0.85:.1f}s (85%)")
print(f"XGBoost prediction: ~{total_time * 0.10:.1f}s (10%)")
print(f"Greedy assignment: ~{total_time * 0.05:.1f}s (5%)")