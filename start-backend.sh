#!/bin/bash

echo "Starting FastAPI Backend..."
echo "================================"

# Navigate to backend directory
cd "$(dirname "$0")/backend"

# Activate virtual environment
source ../../.venv/bin/activate

# Start uvicorn server
echo "Backend will be available at: http://localhost:8000"
echo "API Docs at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
