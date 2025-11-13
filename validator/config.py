"""
Configuration for Validator Component
Contains all thresholds, paths, and settings including parallel processing
"""

class ValidatorConfig:
    """Configuration class for Validator operations"""
    
    # ==================== THRESHOLDS ====================
    
    # Union thresholds
    UNION_THRESHOLD = 0.72  # Minimum score for column compatibility
    UNION_NAME_WEIGHT = 0.85  # Weight for name similarity in hybrid score
    UNION_MODEL_WEIGHT = 0.15  # Weight for model probability in hybrid score
    UNION_COMPATIBILITY_THRESHOLD = 0.7  # Minimum hybrid score for column pair
    
    # Join thresholds
    JOIN_ROW_THRESHOLD = 0.44  # Minimum XGBoost probability for row pair
    JOIN_RETENTION_THRESHOLD = 0.2  # Minimum retention ratio (matched_rows / denominator)
    MAX_MATCHES_PER_ROW = 5  # Maximum matches per row in many-to-many join (0 = unlimited)
    
    # ==================== MODEL PATHS ====================
    
    UNION_MODEL_PATH = "models/union_model.pkl"
    JOIN_MODEL_PATH = "models/join_model.json"
    FINBERT_MODEL_NAME = "ProsusAI/finbert"  # HuggingFace model name
    
    # ==================== PROCESSING LIMITS ====================
    
    MAX_DATAFRAMES = 10  # Maximum input dataframes
    
    # ==================== PARALLEL PROCESSING ====================

    USE_PARALLEL = True  # Enable parallel processing
    N_JOBS = 20  # Number of CPU cores to use (set to 20)

    # ==================== GPU ACCELERATION ====================

    USE_GPU = True  # Enable GPU acceleration for feature extraction and XGBoost
    GPU_BATCH_SIZE = 1000000  # Number of row pairs to process per GPU batch (1M pairs, ~2.4GB VRAM)
    GPU_ID = 0  # GPU device ID to use (0 for first GPU)

    # Batch size recommendations for RTX 3090 (24GB VRAM):
    # -   50,000: Safe baseline (~120 MB per batch)
    # -  500,000: Conservative (~1.2 GB per batch)
    # - 1,000,000: Recommended (~2.4 GB per batch) ⭐
    # - 2,000,000: Aggressive (~4.8 GB per batch)
    # - 5,000,000: Maximum (~12 GB per batch, for very large datasets)
    
    # ==================== FEATURE ENGINEERING ====================
    
    # Join features (26 total - including n_features)
    JOIN_FEATURES = [
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
    
    # Tolerance values for feature computation
    CLOSE_TOLERANCE = 0.1  # For n_close feature
    VERY_CLOSE_TOLERANCE = 0.01  # For n_very_close feature
    EPSILON = 1e-10  # Small value to avoid division by zero
    
    # ==================== LOGGING ====================
    
    VERBOSE = True  # Enable detailed logging
    LOG_LEVEL = "INFO"  # Logging level: DEBUG, INFO, WARNING, ERROR