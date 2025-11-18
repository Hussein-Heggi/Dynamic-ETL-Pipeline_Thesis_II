from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict
import asyncio
import json

from app.services.pipeline_service import pipeline_service

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, run_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[run_id] = websocket

    def disconnect(self, run_id: str):
        if run_id in self.active_connections:
            del self.active_connections[run_id]

    async def send_message(self, run_id: str, message: dict):
        if run_id in self.active_connections:
            try:
                await self.active_connections[run_id].send_json(message)
            except:
                self.disconnect(run_id)

manager = ConnectionManager()

@router.websocket("/pipeline/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for real-time pipeline updates"""
    await manager.connect(run_id, websocket)

    try:
        # Check if run exists
        run_info = pipeline_service.get_run_status(run_id)
        if not run_info:
            await websocket.send_json({
                "type": "error",
                "message": "Pipeline run not found"
            })
            await websocket.close()
            return

        # Define callback for pipeline updates
        async def websocket_callback(message: dict):
            await manager.send_message(run_id, message)

        # If pipeline is pending, start it
        if run_info["status"] == "pending":
            asyncio.create_task(
                pipeline_service.run_pipeline(
                    run_id,
                    run_info["query"],
                    websocket_callback
                )
            )

        # Keep connection alive and listen for messages
        while True:
            try:
                # Wait for client messages (ping/pong)
                data = await websocket.receive_text()

                # Send current status on request
                if data == "status":
                    current_status = pipeline_service.get_run_status(run_id)
                    await websocket.send_json({
                        "type": "status",
                        "status": current_status["status"],
                        "progress": current_status["progress"],
                        "message": current_status["message"]
                    })
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                break

    finally:
        manager.disconnect(run_id)
