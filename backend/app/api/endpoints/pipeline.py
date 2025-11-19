from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from typing import List
from pathlib import Path

from app.models.schemas import (
    QueryRequest,
    PipelineRunResponse,
    PipelineStatusResponse,
    PipelineResultsResponse,
    HistoryItem,
    PipelineStatus
)
from app.services.pipeline_service import pipeline_service

router = APIRouter()

@router.post("/pipeline/run", response_model=PipelineRunResponse)
async def run_pipeline(request: QueryRequest, background_tasks: BackgroundTasks):
    """Start a new pipeline run"""
    try:
        run_id = pipeline_service.create_run(request.query, request.options)

        # Run pipeline in background
        background_tasks.add_task(
            pipeline_service.run_pipeline,
            run_id,
            request.query,
            request.options or {}
        )

        run_info = pipeline_service.get_run_status(run_id)

        return PipelineRunResponse(
            run_id=run_id,
            status=PipelineStatus.PENDING,
            message="Pipeline started successfully",
            created_at=run_info["started_at"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipeline/status/{run_id}", response_model=PipelineStatusResponse)
async def get_pipeline_status(run_id: str):
    """Get the status of a pipeline run"""
    run_info = pipeline_service.get_run_status(run_id)

    if not run_info:
        raise HTTPException(status_code=404, detail="Pipeline run not found")

    return PipelineStatusResponse(
        run_id=run_info["run_id"],
        query=run_info["query"],
        status=run_info["status"],
        progress=run_info["progress"],
        current_stage=run_info["current_stage"],
        message=run_info["message"],
        started_at=run_info["started_at"],
        completed_at=run_info["completed_at"],
        error=run_info["error"],
        stage_flags=run_info.get("stage_flags", {})
    )

@router.get("/pipeline/results/{run_id}", response_model=PipelineResultsResponse)
async def get_pipeline_results(run_id: str):
    """Get the results of a completed pipeline run"""
    results = pipeline_service.get_results(run_id)

    if not results:
        run_info = pipeline_service.get_run_status(run_id)
        if not run_info:
            raise HTTPException(status_code=404, detail="Pipeline run not found")
        elif run_info["status"] != PipelineStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Pipeline is not completed yet. Current status: {run_info['status']}"
            )
        else:
            raise HTTPException(status_code=404, detail="Results not found")

    return PipelineResultsResponse(**results)

@router.get("/pipeline/download/{run_id}/{filename}")
async def download_file(run_id: str, filename: str):
    """Download a specific file from a pipeline run"""
    run_info = pipeline_service.get_run_status(run_id)

    if not run_info:
        raise HTTPException(status_code=404, detail="Pipeline run not found")

    run_dir = Path(run_info["run_dir"])
    file_path = run_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Security check: ensure file is within run directory
    if not str(file_path.resolve()).startswith(str(run_dir.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )

@router.get("/pipeline/history", response_model=List[HistoryItem])
async def get_pipeline_history():
    """Get the history of all pipeline runs"""
    history = pipeline_service.get_history()
    return [HistoryItem(**item) for item in history]
