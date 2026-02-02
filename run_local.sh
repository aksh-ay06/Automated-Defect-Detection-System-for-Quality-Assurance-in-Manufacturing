#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate

echo "Starting FastAPI on port 8000..."
uvicorn src.api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

sleep 2

echo "Starting Streamlit on port 8501..."
streamlit run src/dashboard.py --server.port 8501 --server.address 0.0.0.0 &
DASH_PID=$!

trap "kill $API_PID $DASH_PID 2>/dev/null" EXIT

echo ""
echo "API:       http://localhost:8000"
echo "Dashboard: http://localhost:8501"
echo "Press Ctrl+C to stop both services."

wait
