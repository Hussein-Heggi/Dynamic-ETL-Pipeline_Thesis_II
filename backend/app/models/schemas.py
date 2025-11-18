from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class PipelineStatus(str, Enum):
    PENDING = "pending"
    INGESTION = "ingestion"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    COMPLETED = "completed"
    FAILED = "failed"

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query for stock data")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Additional options")

class PipelineRunResponse(BaseModel):
    run_id: str
    status: PipelineStatus
    message: str
    created_at: datetime

class PipelineStatusResponse(BaseModel):
    run_id: str
    status: PipelineStatus
    progress: int  # 0-100
    current_stage: str
    message: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

class DataFrameInfo(BaseModel):
    index: int
    shape: tuple[int, int]
    columns: List[str]
    preview: List[Dict[str, Any]]  # First 5 rows

class PipelineResultsResponse(BaseModel):
    run_id: str
    status: PipelineStatus
    dataframes: List[DataFrameInfo]
    enrichment_features: List[str]
    validation_report: Optional[Dict[str, Any]] = None
    transformation_report: Optional[Dict[str, Any]] = None

class HistoryItem(BaseModel):
    run_id: str
    query: str
    status: PipelineStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    duration: Optional[float] = None  # in seconds

class WebSocketMessage(BaseModel):
    type: str  # "progress", "log", "error", "complete"
    stage: Optional[str] = None
    progress: Optional[int] = None
    message: str
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None
