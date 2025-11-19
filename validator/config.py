"""
Configuration for Validator Component
Contains all thresholds, paths, and settings including parallel processing
"""

class ValidatorConfig:
    """Configuration class for Validator operations"""
    
    # ==================== QUALITY PROFILES ====================
    # Update the values in QUALITY_PROFILES to tweak each preset
    QUALITY_PROFILES = {
        "high_quality": {
            "UNION_COMPATIBILITY_THRESHOLD": 0.7,
            "JOIN_RETENTION_THRESHOLD": 0.65,
            "MAX_MATCHES_PER_ROW": 2,
            "COLUMN_DELETE_THRESHOLD": 0.3,
        },
        "balanced": {
            "UNION_COMPATIBILITY_THRESHOLD": 0.6,
            "JOIN_RETENTION_THRESHOLD": 0.5,
            "MAX_MATCHES_PER_ROW": 5,
            "COLUMN_DELETE_THRESHOLD": 0.5,
        },
        "high_volume": {
            "UNION_COMPATIBILITY_THRESHOLD": 0.4,
            "JOIN_RETENTION_THRESHOLD": 0.35,
            "MAX_MATCHES_PER_ROW": 0,  # no limit
            "COLUMN_DELETE_THRESHOLD": 0.7,
        },
    }
    DEFAULT_QUALITY_PROFILE = "balanced"

    def __init__(self, quality_profile: str | None = None):
        self.current_quality_profile = self.DEFAULT_QUALITY_PROFILE
        self.apply_quality_profile(quality_profile or self.DEFAULT_QUALITY_PROFILE)
    
    def apply_quality_profile(self, profile_name: str):
        """Apply preset thresholds based on selected quality profile"""
        preset = self.QUALITY_PROFILES.get(profile_name)
        if not preset:
            preset = self.QUALITY_PROFILES[self.DEFAULT_QUALITY_PROFILE]
            profile_name = self.DEFAULT_QUALITY_PROFILE
        self.current_quality_profile = profile_name
        self.UNION_COMPATIBILITY_THRESHOLD = preset["UNION_COMPATIBILITY_THRESHOLD"]
        self.JOIN_RETENTION_THRESHOLD = preset["JOIN_RETENTION_THRESHOLD"]
        self.MAX_MATCHES_PER_ROW = preset["MAX_MATCHES_PER_ROW"]
        self.COLUMN_DELETE_THRESHOLD = preset.get("COLUMN_DELETE_THRESHOLD", 0.5)
    
    # ==================== THRESHOLDS ====================
    
    # Union thresholds
    UNION_THRESHOLD = 0.72  # Minimum score for column compatibility
    UNION_NAME_WEIGHT = 0.85  # Weight for name similarity in hybrid score
    UNION_MODEL_WEIGHT = 0.15  # Weight for model probability in hybrid score
    UNION_COMPATIBILITY_THRESHOLD = 0.7  # Minimum hybrid score for column pair
    
    # Join thresholds
    JOIN_ROW_THRESHOLD = 0.44  # Minimum XGBoost probability for row pair
    JOIN_RETENTION_THRESHOLD = 0.5  # Minimum retention ratio (matched_rows / denominator)
    MAX_MATCHES_PER_ROW = 5  # Maximum matches per row in many-to-many join (0 = unlimited)
    COLUMN_DELETE_THRESHOLD = 0.5  # Ratio threshold for dropping high-null columns
    
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
