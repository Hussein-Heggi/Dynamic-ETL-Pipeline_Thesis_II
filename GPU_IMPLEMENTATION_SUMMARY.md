# GPU Implementation Summary

## Overview
Implemented GPU-accelerated batch processing for join operations using CuPy (GPU arrays) and XGBoost GPU prediction. This provides massive speedup for large datasets through SIMD parallelization.

## Changes Made

### 1. Configuration ([validator/config.py](validator/config.py))

Added GPU acceleration settings:

```python
# GPU ACCELERATION
USE_GPU = True  # Enable/disable GPU acceleration
GPU_BATCH_SIZE = 1000000  # 1M pairs per batch (~2.4GB VRAM)
GPU_ID = 0  # GPU device to use

# Batch size recommendations for RTX 3090 (24GB VRAM):
# -   50,000: Safe baseline (~120 MB per batch)
# -  500,000: Conservative (~1.2 GB per batch)
# - 1,000,000: Recommended (~2.4 GB per batch) ⭐
# - 2,000,000: Aggressive (~4.8 GB per batch)
# - 5,000,000: Maximum (~12 GB per batch)
```

### 2. Join Engine ([validator/join.py](validator/join.py))

#### New Methods:

1. **`_extract_features_batch_gpu(rows_a, rows_b)`** (lines 191-318)
   - Extracts 26 statistical features for BATCH of row pairs
   - Uses CuPy for vectorized GPU operations
   - Processes 50k-5M pairs simultaneously via SIMD
   - Returns numpy array of features

2. **`_extract_features_batch_cpu(rows_a, rows_b)`** (lines 320-335)
   - CPU fallback for batch processing
   - Uses existing `_extract_features_for_pair()` in loop

3. **`_find_compatible_rows_gpu(df1, df2)`** (lines 286-455)
   - Complete GPU-accelerated pipeline
   - Generates ALL pair combinations
   - Processes in large GPU batches
   - Uses XGBoost GPU prediction
   - Returns matched pairs

4. **`_find_compatible_rows_cpu(df1, df2)`** (lines 457-625)
   - Original CPU implementation (unchanged)
   - Uses chunked parallelization

#### Modified Methods:

1. **`find_compatible_rows(df1, df2)`** (lines 263-284)
   - Routes to GPU or CPU version based on config
   - Checks GPU availability automatically

## GPU Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Align columns (CPU)                                 │
│  - Extract numeric columns                                  │
│  - Union + mean padding                                     │
└───────────────────┬─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Convert to float32 NumPy (CPU)                      │
│  - df1_values: (n1, features)                               │
│  - df2_values: (n2, features)                               │
└───────────────────┬─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Generate all pairs (CPU)                            │
│  - all_pairs = [(i,j) for i in df1 for j in df2]           │
│  - 30k × 30k = 900M pairs                                   │
└───────────────────┬─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Batch GPU feature extraction (GPU) ⚡                │
│  For each batch of 1M pairs:                                │
│    1. Stack rows into matrices [1M × features]              │
│    2. Transfer to GPU (CuPy arrays)                         │
│    3. Vectorized operations (all 26 features in parallel)   │
│    4. Transfer results back to CPU                          │
│                                                              │
│  900M pairs ÷ 1M batch = 900 batches                        │
│  Each batch: ~2-3 seconds                                   │
│  Total: ~30-45 minutes for 900M pairs                       │
└───────────────────┬─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: XGBoost GPU prediction (GPU) ⚡                      │
│  - Single batch prediction on all features                  │
│  - GPU tree evaluation (10-20x faster than CPU)             │
│  - Filter by threshold                                      │
└───────────────────┬─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 6: Limited many-to-many assignment (CPU)               │
│  - Sort by probability                                      │
│  - Apply MAX_MATCHES_PER_ROW constraint                     │
│  - Return final matches                                     │
└─────────────────────────────────────────────────────────────┘
```

## Performance Estimates

### 30k × 30k Dataset (900M pairs)

| Component | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Feature Extraction | ~10 hours | **~30-45 min** | **13-20x** |
| XGBoost Prediction | ~2 min | **~10 sec** | **12x** |
| **Total** | **~10 hours** | **~35-50 min** | **~12-17x** |

### Batch Size Impact (for 900M pairs)

| Batch Size | # Batches | Transfer Overhead | Total Time | VRAM Used |
|------------|-----------|-------------------|------------|-----------|
| 50,000 | 18,000 | ~90s | ~50 min | ~120 MB |
| 500,000 | 1,800 | ~9s | ~40 min | ~1.2 GB |
| **1,000,000** | **900** | **~4.5s** | **~35 min** ⭐ | **~2.4 GB** |
| 2,000,000 | 450 | ~2.3s | ~32 min | ~4.8 GB |
| 5,000,000 | 180 | ~0.9s | ~30 min | ~12 GB |

## GPU Memory Calculation

For batch_size pairs with n_features:

```
input_memory = batch_size × n_features × 2 × 4 bytes (rows_a + rows_b)
intermediate_memory = batch_size × n_features × 10 × 4 bytes (diff, abs_diff, etc.)
output_memory = batch_size × 26 × 4 bytes (26 features)

total_memory = (batch_size × n_features × 2 × 4) +
               (batch_size × n_features × 10 × 4) +
               (batch_size × 26 × 4)

For n_features = 50:
total_memory = batch_size × (400 + 2000 + 104) bytes
             = batch_size × 2504 bytes
             = batch_size × 0.00239 MB

Examples:
- 1M pairs: ~2.4 GB
- 2M pairs: ~4.8 GB
- 5M pairs: ~12 GB
```

## Hardware Requirements

**Minimum:**
- NVIDIA GPU with CUDA support
- 4GB VRAM (for batch_size = 500k)
- CUDA 11.x or 12.x

**Recommended:**
- NVIDIA GPU with 8GB+ VRAM
- CUDA 12.x
- 16GB+ system RAM

**Your Hardware (RTX 3090):**
- ✓ 24GB VRAM (excellent)
- ✓ CUDA 12.9
- ✓ Can use batch_size up to 5M

## Installation

```bash
# Activate virtual environment
cd /home/g7/Desktop/Thesis
source .venv/bin/activate

# Install packages
pip install cupy-cuda12x  # For CUDA 12.x
pip install xgboost

# Verify installation
python test_gpu.py
```

## Usage

### Enable/Disable GPU

Edit `validator/config.py`:

```python
USE_GPU = True   # Enable GPU
USE_GPU = False  # Disable GPU (fallback to CPU)
```

### Adjust Batch Size

Edit `validator/config.py`:

```python
GPU_BATCH_SIZE = 1000000  # Default: 1M pairs

# Increase for better performance (if you have VRAM):
GPU_BATCH_SIZE = 2000000  # 2M pairs (~4.8 GB)
GPU_BATCH_SIZE = 5000000  # 5M pairs (~12 GB)

# Decrease if running out of VRAM:
GPU_BATCH_SIZE = 500000   # 500k pairs (~1.2 GB)
GPU_BATCH_SIZE = 100000   # 100k pairs (~240 MB)
```

### Run Join Operations

The GPU pipeline is automatically used when:
1. CuPy is installed
2. `USE_GPU = True` in config
3. CUDA-capable GPU is available

```python
from validator.config import ValidatorConfig
from validator.join import JoinEngine

config = ValidatorConfig()
engine = JoinEngine(config)

# This will use GPU if available, CPU otherwise
matches = engine.find_compatible_rows(df1, df2)
```

## Testing

Run the test script to verify GPU installation and benchmark performance:

```bash
cd /home/g7/Desktop/Thesis/Thesis\ II/Dynamic-ETL-Pipeline_Thesis_II
source ../../.venv/bin/activate
python test_gpu.py
```

Expected output:
```
TEST 1: CuPy Installation
✓ CuPy installed successfully
  CuPy version: 13.x.x
  CUDA available: True
  Device 0: (8, 6)  # RTX 3090 compute capability

TEST 2: XGBoost GPU Support
✓ XGBoost installed successfully
✓ XGBoost GPU training successful

TEST 3: GPU Memory
  GPU memory pool used: X MB
  ✓ Allocated 200.00 MB in 0.XXXs
  ✓ Memory freed successfully

TEST 4: CPU vs GPU Performance Benchmark
  CPU time: XX.XX ms
  GPU time: X.XX ms
  Speedup: XXx
  ✓ Results match!

TEST 5: JoinEngine GPU Configuration
  GPU_AVAILABLE: True
  USE_GPU: True
  GPU_BATCH_SIZE: 1,000,000
  ✓ JoinEngine will use GPU mode
```

## Fallback Behavior

The implementation gracefully falls back to CPU mode if:

1. **CuPy not installed**: Uses `_extract_features_batch_cpu()`
2. **No CUDA GPU available**: Uses CPU parallel processing
3. **USE_GPU = False**: Explicitly disables GPU
4. **GPU memory error**: Catches error and retries with smaller batch or falls back to CPU

## Key Optimizations

1. **Vectorized Operations**: All 26 features computed in parallel across entire batch
2. **Minimal GPU Transfers**: Only transfer data at batch boundaries
3. **Float32 Precision**: Uses 32-bit floats instead of 64-bit (2x memory savings)
4. **Batch Processing**: Processes up to 5M pairs at once (vs 1 pair at a time)
5. **GPU XGBoost**: Tree evaluation on GPU (10-20x faster)

## Debugging

### Check GPU Mode

```python
from validator.join import GPU_AVAILABLE

print(f"GPU Available: {GPU_AVAILABLE}")
```

### Monitor GPU Usage

```bash
# Watch GPU utilization in real-time
watch -n 1 nvidia-smi
```

### Common Issues

**Issue**: `ImportError: No module named 'cupy'`
**Solution**: Install CuPy: `pip install cupy-cuda12x`

**Issue**: `CUDARuntimeError: out of memory`
**Solution**: Reduce `GPU_BATCH_SIZE` in config.py

**Issue**: GPU slower than expected
**Solution**: Check if GPU is being used for display (Xorg). Results may vary.

**Issue**: XGBoost not using GPU
**Solution**: XGBoost will print "Using CPU for XGBoost prediction". This is normal - feature extraction is the main bottleneck.

## Files Modified

1. `validator/config.py` - Added GPU configuration
2. `validator/join.py` - Added GPU methods
3. `test_gpu.py` - GPU verification script (NEW)
4. `GPU_IMPLEMENTATION_SUMMARY.md` - This file (NEW)

## Next Steps

1. **Run tests**: `python test_gpu.py`
2. **Benchmark**: Compare CPU vs GPU on your actual data
3. **Tune batch size**: Experiment with different GPU_BATCH_SIZE values
4. **Monitor VRAM**: Watch `nvidia-smi` during execution
5. **Profile**: Identify any remaining bottlenecks

## References

- CuPy Documentation: https://docs.cupy.dev/
- XGBoost GPU Support: https://xgboost.readthedocs.io/en/stable/gpu/
- CUDA Programming: https://docs.nvidia.com/cuda/

---

**Generated**: 2025-11-13
**GPU**: NVIDIA RTX 3090 (24GB VRAM, CUDA 12.9)
**Python**: 3.10.12
**Status**: Ready for testing (pending CuPy installation)
