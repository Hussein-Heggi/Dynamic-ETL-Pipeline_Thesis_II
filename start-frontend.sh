#!/bin/bash

echo "Starting Next.js Frontend..."
echo "================================"

# Navigate to frontend directory
cd "$(dirname "$0")/frontend"

# Start development server
echo "Frontend will be available at: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

npm run dev
