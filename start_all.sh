#!/bin/bash

# Ensure conda environment exists and activate
env_name="goldensignalsai"
env_file="environment.yaml"
if ! conda info --envs | grep -q "$env_name"; then
  echo "Conda environment $env_name not found. Creating from $env_file..."
  conda env create -f "$env_file"
fi
echo "Activating conda environment: $env_name"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $env_name

LOGDIR="logs"
mkdir -p "$LOGDIR"

# Ensure backend dependencies are installed
if [ -f pyproject.toml ]; then
  if ! command -v poetry >/dev/null; then
    echo "Poetry not found. Installing poetry..."
    pip install poetry
  fi
  echo "Installing backend dependencies with poetry..."
  poetry install || { echo "Poetry install failed!"; exit 1; }
elif [ -f requirements.txt ]; then
  echo "Installing backend dependencies with pip..."
  pip install -r requirements.txt || { echo "pip install failed!"; exit 1; }
fi

# Ensure .env exists
if [ ! -f .env ] && [ -f .env.example ]; then
  echo ".env not found. Copying from .env.example. Please update secrets if needed."
  cp .env.example .env
fi

# Configurable backend port
DEFAULT_BACKEND_PORT=8001
MAX_PORT=8100
BACKEND_PORT=$DEFAULT_BACKEND_PORT

# Function to check and free port
free_port() {
  local port=$1
  local pid
  pid=$(lsof -ti tcp:$port)
  if [ ! -z "$pid" ]; then
    echo "Port $port is in use by PID $pid. Killing..."
    kill -9 $pid
    sleep 1
  fi
}

# Try to find a free port for backend
while [ $BACKEND_PORT -le $MAX_PORT ]; do
  if lsof -i :$BACKEND_PORT -sTCP:LISTEN >/dev/null; then
    echo "Port $BACKEND_PORT is in use. Attempting to free it."
    free_port $BACKEND_PORT
    if lsof -i :$BACKEND_PORT -sTCP:LISTEN >/dev/null; then
      echo "Failed to free port $BACKEND_PORT. Trying next port."
      BACKEND_PORT=$((BACKEND_PORT+1))
      continue
    fi
  fi
  break
done

if [ $BACKEND_PORT -gt $MAX_PORT ]; then
  echo "ERROR: Could not find a free port for backend in range $DEFAULT_BACKEND_PORT-$MAX_PORT. Exiting."
  exit 1
fi

# Kill any orphaned uvicorn/python backend processes
echo "Killing any orphaned uvicorn/python processes..."
pkill -f "uvicorn presentation.api.main:app" 2>/dev/null
pkill -f uvicorn 2>/dev/null
pkill -f python 2>/dev/null
sleep 1

# Kill any existing frontend (react-scripts) processes and free port 3000
echo "Killing any existing frontend (react-scripts) processes..."
pkill -f "react-scripts start" 2>/dev/null
sleep 1
if lsof -i :3000 -sTCP:LISTEN >/dev/null; then
  echo "Port 3000 is in use. Attempting to free it."
  pid=$(lsof -ti tcp:3000)
  kill -9 $pid
  sleep 1
fi

# Check and install frontend dependencies if needed
cd presentation/frontend
if [ ! -d node_modules ]; then
  echo "Installing frontend dependencies (npm install)..."
  npm install || { echo "npm install failed!"; exit 1; }
fi
cd ../..

# Start backend (FastAPI) and log output
echo "Starting backend (FastAPI) on port $BACKEND_PORT..."
conda run -n goldensignalsai uvicorn presentation.api.main:app --host 0.0.0.0 --port $BACKEND_PORT --reload > "$LOGDIR/backend.log" 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to be ready (check port)
for i in {1..30}; do
  if lsof -i :$BACKEND_PORT -sTCP:LISTEN >/dev/null; then
    echo "Backend is up on port $BACKEND_PORT!"
    break
  fi
  echo "Waiting for backend to start on port $BACKEND_PORT... ($i)"
  sleep 1
done

# Start frontend (React) and log output
cd presentation/frontend
if [ -f node_modules/.bin/react-scripts ]; then
  echo "Starting frontend (React) on port 3000..."
  npm run start > "../../$LOGDIR/frontend.log" 2>&1 &
  FRONTEND_PID=$!
  echo "Frontend PID: $FRONTEND_PID"
else
  echo "node_modules not found, please run 'npm install' in presentation/frontend first."
fi
cd ../..

# Health check for backend and frontend
sleep 5
backend_ok=0
frontend_ok=0
if curl -s "http://localhost:$BACKEND_PORT/docs" | grep -qi "swagger"; then
  backend_ok=1
fi
if curl -s "http://localhost:3000" | grep -qi "<!DOCTYPE html>"; then
  frontend_ok=1
fi

# Print summary
if [ $backend_ok -eq 1 ]; then
  echo "[OK] Backend is running on port $BACKEND_PORT."
else
  echo "[ERROR] Backend did not start correctly. Check $LOGDIR/backend.log."
fi
if [ $frontend_ok -eq 1 ]; then
  echo "[OK] Frontend is running on port 3000."
else
  echo "[ERROR] Frontend did not start correctly. Check $LOGDIR/frontend.log."
fi

echo "--- Process Info ---"
echo "Backend log:   $LOGDIR/backend.log"
echo "Frontend log:  $LOGDIR/frontend.log"
echo "Backend running on port: $BACKEND_PORT"
echo "To view logs: tail -f $LOGDIR/backend.log $LOGDIR/frontend.log"
echo "To stop: kill $BACKEND_PID $FRONTEND_PID"
