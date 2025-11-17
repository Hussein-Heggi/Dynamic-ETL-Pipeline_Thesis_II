"""
Data Contracts - Updated for LLM-driven architecture with validation reports
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Literal
import pandas as pd


# ============================================================================
# CONTRACT A: QueryAnalyzer Output (NEW)
# ============================================================================

class FeatureSpec(BaseModel):
    """Features extracted from query"""
    native: List[str] = Field(default_factory=list, description="API-provided features")
    enrichment: List[str] = Field(default_factory=list, description="Calculated features")


class LLMAPIRequest(BaseModel):
    """API request proposed by LLM"""
    api_name: str
    endpoint_name: str
    parameters: Dict[str, Any]
    reasoning: Optional[str] = None


class LLMResponse(BaseModel):
    """Complete LLM output from QueryAnalyzer"""
    features: FeatureSpec
    api_requests: List[LLMAPIRequest]
    tickers: List[str] = Field(default_factory=list)


# ============================================================================
# CONTRACT B: Validated API Request (UPDATED)
# ============================================================================

class APIRequest(BaseModel):
    """A single API request with validation metadata"""
    api_name: str
    endpoint_name: str
    parameters: Dict[str, Any]
    
    # Validation fields
    semantic_score: Optional[float] = Field(None, description="FAISS similarity score (0-1)")
    validation_status: Literal["PENDING", "VALID", "WARNING", "ERROR"] = "PENDING"
    validation_errors: List[str] = Field(default_factory=list)
    validation_warnings: List[str] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    """Plan for executing API requests"""
    ranked_requests: List[APIRequest]


# ============================================================================
# CONTRACT C: Output Validation (NEW)
# ============================================================================

class ValidationReport(BaseModel):
    """Post-fetch validation report for a dataset"""
    api_name: str
    endpoint_name: str
    ticker: Optional[str] = None
    
    found_features: List[str] = Field(default_factory=list)
    fuzzy_matched_features: List[Dict[str, Any]] = Field(default_factory=list)
    missing_features: List[str] = Field(default_factory=list)
    
    actual_columns: List[str] = Field(default_factory=list)
    validation_passed: bool = True


# ============================================================================
# CONTRACT D: Execution Results (UPDATED)
# ============================================================================

class APIResult(BaseModel):
    """Result from a single API call"""
    api_name: str
    endpoint_name: str
    status: Literal["SUCCESS", "FAILED"]
    data: Any = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    used_parameters: Dict[str, Any] = Field(default_factory=dict)
    response_code: Optional[int] = None
    error_message: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


class ExecutionResults(BaseModel):
    """Complete results from pipeline execution"""
    results: List[APIResult]
    failed_requests: List[APIRequest] = Field(default_factory=list)
    overall_status: Literal["COMPLETE", "PARTIAL", "FAILED"]
    execution_time_ms: int = 0
    
    class Config:
        arbitrary_types_allowed = True