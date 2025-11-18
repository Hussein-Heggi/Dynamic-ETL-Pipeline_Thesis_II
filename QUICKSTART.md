# Quick Start Guide - Dynamic ETL Pipeline Web App

## What We Built

A complete full-stack web application for your ETL pipeline with:

### Frontend (Next.js + TypeScript)
- **Home Page**: Dashboard with stats and recent runs
- **Query Page**: Natural language input for stock data queries
- **Results Page**: Real-time pipeline execution monitoring with WebSocket
- **History Page**: View all past pipeline runs

### Backend (FastAPI + Python)
- REST API for pipeline operations
- WebSocket for real-time updates
- Background task execution
- File-based result storage

## Running the Application

### Option 1: Using Startup Scripts (Easiest)

Open **two terminals**:

**Terminal 1 - Backend:**
```bash
cd "/home/g7/Desktop/Thesis/Thesis II/Dynamic-ETL-Pipeline_Thesis_II"
./start-backend.sh
```

**Terminal 2 - Frontend:**
```bash
cd "/home/g7/Desktop/Thesis/Thesis II/Dynamic-ETL-Pipeline_Thesis_II"
./start-frontend.sh
```

### Option 2: Manual Start

**Terminal 1 - Backend:**
```bash
cd "/home/g7/Desktop/Thesis/Thesis II/Dynamic-ETL-Pipeline_Thesis_II/backend"
source ../../.venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd "/home/g7/Desktop/Thesis/Thesis II/Dynamic-ETL-Pipeline_Thesis_II/frontend"
npm run dev
```

## Accessing the Application

Once both servers are running:

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## How to Use

### 1. Create a Query

1. Go to http://localhost:3000
2. Click "Create New Query" or navigate to "/query"
3. Enter a natural language query, for example:
   ```
   Show me Apple stock prices from last 30 days with SMA 10 days
   ```
4. Click "Run Pipeline"

### 2. Monitor Execution

- You'll be automatically redirected to the Results page
- Watch real-time progress through the pipeline stages:
  - Ingestion → Validation → Transformation → Completed
- View live logs from the pipeline
- Progress bar updates in real-time via WebSocket

### 3. View Results

Once completed, explore the results in different tabs:

- **Data Preview**: View the processed dataframes in tables
- **Validation Report**: See the validation results
- **Transformation Report**: View transformation details
- **Downloads**: Download CSV and JSON files

### 4. Check History

- Navigate to "/history" to see all past runs
- Click "View" on any completed run to see its results
- View stats like success rate and average duration

## Features

### Real-Time Updates
- WebSocket connection provides live progress updates
- See exactly what stage the pipeline is in
- View logs as they happen

### Comprehensive Data Display
- Preview of processed dataframes
- Full JSON reports for validation and transformation
- Download all results as files

### User-Friendly Interface
- Clean, modern UI with shadcn/ui components
- Light mode design (as requested)
- Responsive layout for different screen sizes
- Status badges and icons for quick understanding

### API Integration
- Fully integrated with your existing pipeline:
  - LLM_Ingestor
  - Validator
  - Transformer
- Automatic background processing
- File-based result storage

## Project Structure

```
Dynamic-ETL-Pipeline_Thesis_II/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── api/endpoints/       # API routes
│   │   ├── services/            # Pipeline service
│   │   ├── models/              # Pydantic schemas
│   │   └── temp/                # Results storage
│   └── requirements.txt
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx            # Home page
│   │   ├── query/page.tsx      # Query input page
│   │   ├── results/[runId]/    # Results page
│   │   └── history/page.tsx    # History page
│   ├── components/
│   │   ├── Navbar.tsx          # Navigation
│   │   └── ui/                 # shadcn/ui components
│   └── lib/
│       ├── api/                # API client
│       └── websocket.ts        # WebSocket client
│
├── start-backend.sh            # Backend startup script
├── start-frontend.sh           # Frontend startup script
└── your existing pipeline files...
```

## API Endpoints

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/pipeline/run` | Start new pipeline |
| GET | `/api/v1/pipeline/status/{run_id}` | Get status |
| GET | `/api/v1/pipeline/results/{run_id}` | Get results |
| GET | `/api/v1/pipeline/download/{run_id}/{file}` | Download file |
| GET | `/api/v1/pipeline/history` | Get history |

### WebSocket

- `WS /ws/pipeline/{run_id}` - Real-time pipeline updates

## Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation
- **WebSockets** - Real-time communication

### Frontend
- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **shadcn/ui** - UI components
- **TanStack Query** - Data fetching
- **Axios** - HTTP client
- **Recharts** - Data visualization
- **date-fns** - Date formatting
- **lucide-react** - Icons

## Troubleshooting

### Backend won't start
- Make sure you're in the correct directory
- Activate the virtual environment: `source ../../.venv/bin/activate`
- Check if port 8000 is available: `lsof -i :8000`

### Frontend won't start
- Make sure you ran `npm install` in the frontend directory
- Check if port 3000 is available: `lsof -i :3000`
- Try clearing npm cache: `npm cache clean --force`

### CORS Errors
- Ensure backend is running on port 8000
- Check `.env.local` in frontend directory
- Verify CORS settings in `backend/app/main.py`

### WebSocket not connecting
- Make sure WebSocket URL is correct in `.env.local`
- Check browser console for errors
- Verify backend WebSocket endpoint is running

### Pipeline errors
- Check backend terminal for Python errors
- Verify your pipeline dependencies are installed in `.venv`
- Check the transformation report for detailed error messages

## What's Next?

The application is fully functional! Here are some ideas for enhancements:

1. **Add Charts**: Implement stock price visualizations with Recharts
2. **Database Storage**: Replace file storage with SQLite/PostgreSQL
3. **User Authentication**: Add login system for multi-user support
4. **Export Options**: Add PDF export for reports
5. **Notifications**: Email alerts when pipeline completes
6. **Comparison View**: Compare multiple pipeline runs side-by-side
7. **Dark Mode**: Add theme toggle
8. **Advanced Filters**: Filter history by date, status, etc.

## Testing the Application

### Test the Backend API

```bash
# Check health
curl http://localhost:8000/health

# Start a pipeline (replace with your query)
curl -X POST http://localhost:8000/api/v1/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me Apple stock prices from last 30 days with SMA 10 days"}'

# Get status (replace RUN_ID)
curl http://localhost:8000/api/v1/pipeline/status/RUN_ID
```

### Test the Frontend

1. Open http://localhost:3000
2. Create a query
3. Watch the real-time progress
4. View the results
5. Check the history page

## Support

For issues or questions:
1. Check the browser console for frontend errors
2. Check the backend terminal for Python errors
3. Review the API docs at http://localhost:8000/docs
4. Check the transformation and validation reports for pipeline errors

Enjoy your Dynamic ETL Pipeline Web Application!
