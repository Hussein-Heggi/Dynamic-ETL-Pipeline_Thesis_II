"""
Configuration for Validator Component
Contains all thresholds, paths, and settings
"""

class ValidatorConfig:
    """Configuration class for Validator operations"""
    
    # ==================== THRESHOLDS ====================
    
    # Union thresholds
    UNION_THRESHOLD = 0.72  # Minimum score for column compatibility
    UNION_NAME_WEIGHT = 0.85  # Weight for name similarity in hybrid score
    UNION_MODEL_WEIGHT = 0.15  # Weight for model probability in hybrid score
    
    # Join thresholds
    JOIN_ROW_THRESHOLD = 0.44  # Minimum XGBoost probability for row pair
    JOIN_RETENTION_THRESHOLD = 0.5  # Minimum retention ratio (matched_rows / denominator)
    
    # ==================== MODEL PATHS ====================
    
    UNION_MODEL_PATH = "validator/models/union_model.pkl"
    JOIN_MODEL_PATH = "validator/models/join_model.pkl"
    FINBERT_MODEL_NAME = "ProsusAI/finbert"  # HuggingFace model name
    
    # ==================== PROCESSING LIMITS ====================
    
    MAX_DATAFRAMES = 4  # Maximum input dataframes
    MAX_COMBINATIONS = 1000000  # Safety limit for row pair combinations
    
    # ==================== FEATURE ENGINEERING ====================
    
    # Join features (25 total)
    JOIN_FEATURES = [
        'absdiff_min', 'absdiff_max', 'absdiff_mean', 'absdiff_median', 'absdiff_std',
        'reldiff_mean', 'reldiff_median',
        'ratio_mean', 'ratio_median', 'ratio_std',
        'zdiff_mean', 'zdiff_max',
        'pcdiff_mean', 'pcdiff_sq_mean',
        'L1_raw', 'L1_z', 'L1_pct',
        'L2_raw', 'L2_z', 'L2_pct',
        'correlation',
        'n_close', 'n_very_close', 'n_both_zero',
        'sign_agreement'
    ]
    
    # Tolerance values for feature computation
    CLOSE_TOLERANCE = 0.1  # For n_close feature
    VERY_CLOSE_TOLERANCE = 0.01  # For n_very_close feature
    EPSILON = 1e-10  # Small value to avoid division by zero
    
    # ==================== LOGGING ====================
    
    VERBOSE = True  # Enable detailed logging
    LOG_LEVEL = "INFO"  # Logging level: DEBUG, INFO, WARNING, ERROR