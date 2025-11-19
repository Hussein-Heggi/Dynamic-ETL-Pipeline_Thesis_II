import sys
import os
import uuid
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path to import pipeline modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from LLM_Ingestor.ingestor import Ingestor
from validator import Validator
from transform.transform import transform_pipeline
from app.models.schemas import PipelineStatus

class PipelineService:
    def __init__(self):
        self.temp_dir = Path(__file__).parent.parent / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        self.runs: Dict[str, Dict[str, Any]] = {}  # In-memory storage for run metadata

    def create_run(self, query: str) -> str:
        """Create a new pipeline run and return run_id"""
        run_id = str(uuid.uuid4())
        run_dir = self.temp_dir / run_id
        run_dir.mkdir(exist_ok=True)

        self.runs[run_id] = {
            "run_id": run_id,
            "query": query,
            "status": PipelineStatus.PENDING,
            "progress": 0,
            "current_stage": "pending",
            "message": "Pipeline created",
            "started_at": datetime.utcnow(),
            "completed_at": None,
            "error": None,
            "run_dir": str(run_dir),
            "stage_flags": {
                "ingestion": False,
                "validation": False,
                "transformation": False,
                "completed": False
            }
        }

        return run_id

    def get_run_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a pipeline run"""
        return self.runs.get(run_id)

    def update_run_status(
        self,
        run_id: str,
        status: PipelineStatus,
        progress: int,
        message: str,
        current_stage: str = None,
        error: str = None
    ):
        """Update the status of a pipeline run"""
        if run_id in self.runs:
            self.runs[run_id]["status"] = status
            self.runs[run_id]["progress"] = progress
            self.runs[run_id]["message"] = message
            if current_stage:
                self.runs[run_id]["current_stage"] = current_stage
            if error:
                self.runs[run_id]["error"] = error
            if status in [PipelineStatus.COMPLETED, PipelineStatus.FAILED]:
                self.runs[run_id]["completed_at"] = datetime.utcnow()

    def set_stage_flag(self, run_id: str, stage: str, value: bool = True):
        """Mark a stage as completed (or reset it)"""
        if run_id in self.runs and stage in self.runs[run_id]["stage_flags"]:
            self.runs[run_id]["stage_flags"][stage] = value

    async def _run_blocking(self, func, *args, **kwargs):
        """Run a blocking function in a thread to avoid blocking the event loop"""
        return await asyncio.to_thread(func, *args, **kwargs)

    def _write_text_file(self, path: Path, content: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)

    def _write_json_file(self, path: Path, data: Any):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    async def run_pipeline(self, run_id: str, query: str, websocket_callback=None):
        """Execute the ETL pipeline"""
        run_dir = Path(self.runs[run_id]["run_dir"])

        try:
            # Stage 1: Ingestion
            await self._send_update(
                run_id,
                websocket_callback,
                "progress",
                PipelineStatus.INGESTION,
                10,
                "Starting data ingestion..."
            )
            self.update_run_status(
                run_id,
                PipelineStatus.INGESTION,
                10,
                "Running ingestion...",
                "ingestion"
            )

            ingestor = Ingestor()
            dataframes, enrichment_features, key_features, validation_reports = await self._run_blocking(
                ingestor.process,
                query
            )

            await self._run_blocking(
                self._write_text_file,
                run_dir / "enrichment_features.txt",
                ",".join(enrichment_features)
            )

            self.set_stage_flag(run_id, "ingestion")

            await self._send_update(
                run_id,
                websocket_callback,
                "progress",
                PipelineStatus.INGESTION,
                30,
                f"Ingestion completed. Retrieved {len(dataframes)} dataframe(s)"
            )

            # Stage 2: Validation
            await self._send_update(
                run_id,
                websocket_callback,
                "progress",
                PipelineStatus.VALIDATION,
                40,
                "Starting validation..."
            )
            self.update_run_status(
                run_id,
                PipelineStatus.VALIDATION,
                40,
                "Running validation...",
                "validation"
            )

            validator = Validator()
            val_out, validation_report = await self._run_blocking(
                validator.process,
                dataframes
            )

            await self._run_blocking(
                validator.save_outputs,
                val_out,
                str(run_dir / "val_outputs"),
                "result"
            )

            await self._run_blocking(
                self._write_json_file,
                run_dir / "validation_report.json",
                validation_report
            )

            self.set_stage_flag(run_id, "validation")

            await self._send_update(
                run_id,
                websocket_callback,
                "progress",
                PipelineStatus.VALIDATION,
                60,
                "Validation completed"
            )

            # Stage 3: Transformation
            await self._send_update(
                run_id,
                websocket_callback,
                "progress",
                PipelineStatus.TRANSFORMATION,
                70,
                "Starting transformation..."
            )
            self.update_run_status(
                run_id,
                PipelineStatus.TRANSFORMATION,
                70,
                "Running transformation...",
                "transformation"
            )

            trans_out, transformation_report = await self._run_blocking(
                transform_pipeline,
                val_out,
                enrichment_features
            )

            for i, df in enumerate(trans_out):
                await self._run_blocking(df.to_csv, run_dir / f"df_{i}.csv", index=False)

            await self._run_blocking(
                self._write_json_file,
                run_dir / "transformation_report.json",
                transformation_report
            )

            self.set_stage_flag(run_id, "transformation")

            await self._send_update(
                run_id,
                websocket_callback,
                "progress",
                PipelineStatus.TRANSFORMATION,
                90,
                "Transformation completed"
            )

            # Stage 4: Completion
            self.update_run_status(
                run_id,
                PipelineStatus.COMPLETED,
                100,
                "Pipeline completed successfully",
                "completed"
            )

            self.set_stage_flag(run_id, "completed")
            await self._send_update(
                run_id,
                websocket_callback,
                "complete",
                PipelineStatus.COMPLETED,
                100,
                "Pipeline completed successfully",
                data={"run_id": run_id}
            )

        except Exception as e:
            error_msg = str(e)
            self.update_run_status(
                run_id,
                PipelineStatus.FAILED,
                0,
                f"Pipeline failed: {error_msg}",
                "failed",
                error=error_msg
            )

            await self._send_update(
                run_id,
                websocket_callback,
                "error",
                PipelineStatus.FAILED,
                0,
                f"Pipeline failed: {error_msg}"
            )
            raise

    async def _send_update(
        self,
        run_id: str,
        websocket_callback,
        msg_type: str,
        stage: PipelineStatus,
        progress: int,
        message: str,
        data: Dict[str, Any] = None
    ):
        """Send update via WebSocket if callback is provided"""
        if websocket_callback:
            payload = dict(data) if data else {}
            if run_id in self.runs:
                payload.setdefault("stage_flags", self.runs[run_id]["stage_flags"].copy())
            payload.setdefault("run_id", run_id)
            stage_flags = payload.get("stage_flags")
            await websocket_callback({
                "type": msg_type,
                "stage": stage.value,
                "progress": progress,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
                "stage_flags": stage_flags,
                "data": payload
            })

    def get_results(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get the results of a completed pipeline run"""
        if run_id not in self.runs:
            return None

        run_info = self.runs[run_id]
        if run_info["status"] not in [PipelineStatus.COMPLETED]:
            return None

        run_dir = Path(run_info["run_dir"])

        # Load dataframes
        dataframes_info = []
        for csv_file in sorted(run_dir.glob("df_*.csv")):
            import pandas as pd
            df = pd.read_csv(csv_file)
            dataframes_info.append({
                "index": int(csv_file.stem.split("_")[1]),
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "preview": df.head(5).to_dict(orient="records")
            })

        # Load enrichment features
        enrichment_features = []
        enrichment_file = run_dir / "enrichment_features.txt"
        if enrichment_file.exists():
            with open(enrichment_file, "r") as f:
                enrichment_features = f.read().split(",")

        # Load reports
        validation_report = None
        val_report_file = run_dir / "validation_report.json"
        if val_report_file.exists():
            with open(val_report_file, "r") as f:
                validation_report = json.load(f)

        transformation_report = None
        trans_report_file = run_dir / "transformation_report.json"
        if trans_report_file.exists():
            with open(trans_report_file, "r") as f:
                transformation_report = json.load(f)

        return {
            "run_id": run_id,
            "status": run_info["status"],
            "dataframes": dataframes_info,
            "enrichment_features": enrichment_features,
            "validation_report": validation_report,
            "transformation_report": transformation_report
        }

    def get_history(self) -> List[Dict[str, Any]]:
        """Get history of all pipeline runs"""
        history = []
        for run_id, run_info in self.runs.items():
            duration = None
            if run_info["completed_at"]:
                duration = (run_info["completed_at"] - run_info["started_at"]).total_seconds()

            history.append({
                "run_id": run_id,
                "query": run_info["query"],
                "status": run_info["status"],
                "created_at": run_info["started_at"],
                "completed_at": run_info["completed_at"],
                "duration": duration
            })

        return sorted(history, key=lambda x: x["created_at"], reverse=True)

# Global instance
pipeline_service = PipelineService()
