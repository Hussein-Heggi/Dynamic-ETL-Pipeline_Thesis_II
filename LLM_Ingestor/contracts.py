"""
Data Contracts - With proceed flag and semantic_keywords
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Literal
import pandas as pd


class FeatureSpec(BaseModel):
    native: List[str] = Field(default_factory=list)
    enrichment: List[str] = Field(default_factory=list)


class LLMAPIRequest(BaseModel):
    api_name: str
    endpoint_name: str
    parameters: Dict[str, Any]
    reasoning: Optional[str] = None


class LLMResponse(BaseModel):
    proceed: bool = Field(default=True)
    features: FeatureSpec
    semantic_keywords: List[str] = Field(default_factory=list)
    api_requests: List[LLMAPIRequest]
    tickers: List[str] = Field(default_factory=list)


class APIRequest(BaseModel):
    api_name: str
    endpoint_name: str
    parameters: Dict[str, Any]
    semantic_score: Optional[float] = None
    validation_status: Literal["PENDING", "VALID", "WARNING", "ERROR"] = "PENDING"
    validation_errors: List[str] = Field(default_factory=list)
    validation_warnings: List[str] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    ranked_requests: List[APIRequest]


class ValidationReport(BaseModel):
    api_name: str
    endpoint_name: str
    ticker: Optional[str] = None
    found_features: List[str] = Field(default_factory=list)
    fuzzy_matched_features: List[Dict[str, Any]] = Field(default_factory=list)
    missing_features: List[str] = Field(default_factory=list)
    actual_columns: List[str] = Field(default_factory=list)
    validation_passed: bool = True


class APIResult(BaseModel):
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
    results: List[APIResult]
    failed_requests: List[APIRequest] = Field(default_factory=list)
    overall_status: Literal["COMPLETE", "PARTIAL", "FAILED"]
    execution_time_ms: int = 0
    
    class Config:
        arbitrary_types_allowed = True