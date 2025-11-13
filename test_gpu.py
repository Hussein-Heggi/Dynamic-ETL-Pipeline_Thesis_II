"""
Test script for GPU-accelerated join engine
Verifies CuPy, XGBoost GPU, and compares CPU vs GPU performance
"""

import numpy as np
import pandas as pd
import time
import sys

# Test 1: Check CuPy installation
print("=" * 70)
print("TEST 1: CuPy Installation")
print("=" * 70)

try:
    import cupy as cp
    print("✓ CuPy installed successfully")
    try:
        print(f"  CuPy version: {cp.__version__}")
    except AttributeError:
        # cupy-cuda12x doesn't expose __version__, get from package metadata
        import importlib.metadata
        version = importlib.metadata.version('cupy-cuda12x')
        print(f"  CuPy version: {version}")

    # Test GPU availability
    print(f"  CUDA available: {cp.cuda.is_available()}")
    print(f"  Device count: {cp.cuda.runtime.getDeviceCount()}")

    device = cp.cuda.Device(0)
    print(f"  Device 0: {device.compute_capability}")

    # Simple GPU test
    x_gpu = cp.array([1, 2, 3])
    y_gpu = cp.array([4, 5, 6])
    result = x_gpu + y_gpu
    print(f"  Simple GPU operation test: [1,2,3] + [4,5,6] = {cp.asnumpy(result)}")

except Exception as e:
    print(f"✗ CuPy test failed: {e}")
    sys.exit(1)

# Test 2: Check XGBoost GPU support
print("\n" + "=" * 70)
print("TEST 2: XGBoost GPU Support")
print("=" * 70)

try:
    import xgboost as xgb
    print("✓ XGBoost installed successfully")
    print(f"  XGBoost version: {xgb.__version__}")

    # Test GPU predictor (XGBoost 3.1+ uses 'device' instead of 'gpu_id')
    dtrain = xgb.DMatrix(np.random.rand(100, 10), label=np.random.rand(100))
    params = {'predictor': 'gpu_predictor', 'device': 'cuda:0', 'tree_method': 'hist'}
    bst = xgb.train(params, dtrain, num_boost_round=10)
    print("✓ XGBoost GPU training successful")

except Exception as e:
    print(f"✗ XGBoost GPU test failed: {e}")
    print("  Note: CPU fallback will be used")

# Test 3: GPU Memory Check
print("\n" + "=" * 70)
print("TEST 3: GPU Memory")
print("=" * 70)

try:
    mempool = cp.get_default_memory_pool()
    print(f"  GPU memory pool used: {mempool.used_bytes() / 1024**2:.2f} MB")
    print(f"  GPU memory pool total: {mempool.total_bytes() / 1024**2:.2f} MB")

    # Test large array allocation (1M pairs × 50 features)
    test_size = 1_000_000
    test_features = 50

    print(f"\n  Testing allocation of {test_size:,} × {test_features} array...")
    start = time.time()
    test_array = cp.random.rand(test_size, test_features, dtype=cp.float32)
    alloc_time = time.time() - start

    memory_used = test_array.nbytes / 1024**2
    print(f"  ✓ Allocated {memory_used:.2f} MB in {alloc_time:.3f}s")

    del test_array
    mempool.free_all_blocks()
    print(f"  ✓ Memory freed successfully")

except Exception as e:
    print(f"✗ GPU memory test failed: {e}")

# Test 4: Benchmark CPU vs GPU for feature extraction
print("\n" + "=" * 70)
print("TEST 4: CPU vs GPU Performance Benchmark")
print("=" * 70)

try:
    # Create test data
    n_pairs = 10000
    n_features = 50

    print(f"\nGenerating {n_pairs:,} random row pairs with {n_features} features...")
    rows_a = np.random.rand(n_pairs, n_features).astype(np.float32)
    rows_b = np.random.rand(n_pairs, n_features).astype(np.float32)

    # CPU benchmark
    print("\n  Running CPU computation...")
    start = time.time()
    diff_cpu = rows_a - rows_b
    abs_diff_cpu = np.abs(diff_cpu)
    mean_cpu = np.mean(abs_diff_cpu, axis=1)
    cpu_time = time.time() - start
    print(f"  CPU time: {cpu_time*1000:.2f} ms")

    # GPU benchmark
    print("\n  Running GPU computation...")
    start = time.time()
    rows_a_gpu = cp.asarray(rows_a)
    rows_b_gpu = cp.asarray(rows_b)
    diff_gpu = rows_a_gpu - rows_b_gpu
    abs_diff_gpu = cp.abs(diff_gpu)
    mean_gpu = cp.mean(abs_diff_gpu, axis=1)
    result_gpu = cp.asnumpy(mean_gpu)
    gpu_time = time.time() - start
    print(f"  GPU time: {gpu_time*1000:.2f} ms")

    # Compare results
    speedup = cpu_time / gpu_time
    print(f"\n  Speedup: {speedup:.2f}x")

    # Verify correctness
    diff = np.abs(mean_cpu - result_gpu).max()
    print(f"  Max difference: {diff:.2e} (should be < 1e-5)")

    if diff < 1e-5:
        print("  ✓ Results match!")
    else:
        print("  ✗ Results don't match (may be due to floating point precision)")

except Exception as e:
    print(f"✗ Benchmark failed: {e}")

# Test 5: Load JoinEngine and verify config
print("\n" + "=" * 70)
print("TEST 5: JoinEngine GPU Configuration")
print("=" * 70)

try:
    sys.path.insert(0, 'validator')
    from config import ValidatorConfig
    from join import JoinEngine, GPU_AVAILABLE

    config = ValidatorConfig()
    print(f"  GPU_AVAILABLE: {GPU_AVAILABLE}")
    print(f"  USE_GPU: {config.USE_GPU}")
    print(f"  GPU_BATCH_SIZE: {config.GPU_BATCH_SIZE:,}")
    print(f"  GPU_ID: {config.GPU_ID}")

    if GPU_AVAILABLE and config.USE_GPU:
        print("\n  ✓ JoinEngine will use GPU mode")
    else:
        print("\n  ✗ JoinEngine will use CPU mode")
        if not GPU_AVAILABLE:
            print("    Reason: CuPy not available")
        if not config.USE_GPU:
            print("    Reason: USE_GPU is False in config")

except Exception as e:
    print(f"✗ JoinEngine test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("All tests complete!")
print("=" * 70)
