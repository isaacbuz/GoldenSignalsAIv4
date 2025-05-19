#!/usr/bin/env bash

# GoldenSignalsAI Application Runner
# This script manages dependencies, ports, and launches backend, frontend, and API microservices.
# It logs all actions and cleans up on exit.

APP_PORT=8000
FRONTEND_PORT=3000
API_PORT=8080
LOGFILE="run_golden_signals.log"
ENV_NAME="goldensignalsai"
REQUIREMENTS="requirements.txt"
FRONTEND_DIR="presentation/frontend"
API_DIR="presentation/api"

# Logging function
echo_log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

# Trap for cleanup
cleanup() {
    echo_log "Shutting down all services and cleaning up."
    if [[ -n "$SERVER_PID" ]]; then
        kill $SERVER_PID 2>/dev/null
        echo_log "Stopped FastAPI server (PID $SERVER_PID)"
    fi
    if [[ -n "$FRONTEND_PID" ]]; then
        kill $FRONTEND_PID 2>/dev/null
        echo_log "Stopped Dash frontend (PID $FRONTEND_PID)"
    fi
    if [[ -n "$API_PID" ]]; then
        kill $API_PID 2>/dev/null
        echo_log "Stopped API microservice (PID $API_PID)"
    fi
}
trap cleanup EXIT

# Activate conda environment
echo_log "Activating conda environment: $ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
if [[ $? -ne 0 ]]; then
    echo_log "Failed to activate conda environment: $ENV_NAME"
    exit 1
fi

# Set PYTHONPATH to project root for all subprocesses
export PYTHONPATH="$(pwd):$PYTHONPATH"
echo_log "PYTHONPATH set to: $PYTHONPATH"

# Install backend dependencies if needed
if [[ -f "$REQUIREMENTS" ]]; then
    echo_log "Installing/updating backend dependencies from $REQUIREMENTS"
    pip install -r "$REQUIREMENTS" | tee -a "$LOGFILE"
else
    echo_log "No requirements.txt found. Skipping backend dependency installation."
fi

# Ensure memory_profiler is installed (for backend)
if ! python -c 'import memory_profiler' 2>/dev/null; then
    echo_log "Installing missing Python package: memory_profiler"
    pip install memory_profiler | tee -a "$LOGFILE"
else
    echo_log "memory_profiler already installed."
fi

# Install frontend dependencies if needed
if [[ -d "$FRONTEND_DIR" ]]; then
    if [[ -f "$FRONTEND_DIR/package.json" ]]; then
        echo_log "Installing/updating frontend npm dependencies."
        cd "$FRONTEND_DIR"
        npm install | tee -a "../../$LOGFILE"
        cd - >/dev/null
        # Overwrite src/App.jsx with robust minimal version every run
        echo_log "Ensuring robust minimal App.jsx for frontend."
        cat > "$FRONTEND_DIR/src/App.jsx" <<'EOF'
import React, { useEffect, useState } from 'react';
import { API_URL, WS_URL } from './config';

function App() {
  const [apiHealth, setApiHealth] = useState('');
  const [ohlcv, setOhlcv] = useState(null);
  const [wsStatus, setWsStatus] = useState('disconnected');
  const [wsMsg, setWsMsg] = useState('');

  // Test API health
  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then(r => r.ok ? r.text() : r.statusText)
      .then(setApiHealth)
      .catch(() => setApiHealth('API not reachable'));
  }, []);

  // Test OHLCV endpoint
  useEffect(() => {
    fetch(`${API_URL}/api/ohlcv?symbol=AAPL&timeframe=D`)
      .then(r => r.json())
      .then(setOhlcv)
      .catch(() => setOhlcv(null));
  }, []);

  // Test WebSocket
  useEffect(() => {
    let ws;
    setWsStatus('connecting');
    try {
      ws = new window.WebSocket(`${WS_URL}/ws/ohlcv?symbol=AAPL&timeframe=D`);
      ws.onopen = () => setWsStatus('connected');
      ws.onmessage = (e) => setWsMsg(e.data);
      ws.onerror = () => setWsStatus('error');
      ws.onclose = () => setWsStatus('closed');
    } catch {
      setWsStatus('error');
    }
    return () => ws && ws.close();
  }, []);

  return (
    <div style={{ maxWidth: 600, margin: '40px auto', fontFamily: 'sans-serif' }}>
      <h2>GoldenSignalsAI Frontend (Vite+React18)</h2>
      <div style={{marginBottom: 16}}>
        <b>API Health:</b> {apiHealth || 'Loading...'}
      </div>
      <div style={{marginBottom: 16}}>
        <b>OHLCV Test Data:</b><br/>
        <pre style={{background: '#f4f4f4', padding: 8, borderRadius: 4, fontSize: 13}}>
          {ohlcv ? JSON.stringify(ohlcv, null, 2) : 'Loading or failed.'}
        </pre>
      </div>
      <div style={{marginBottom: 16}}>
        <b>WebSocket Status:</b> {wsStatus}<br/>
        <b>Latest WS Message:</b> <span style={{wordBreak: 'break-all'}}>{wsMsg}</span>
      </div>
      <div style={{marginTop: 32, color: '#888', fontSize: 13}}>
        <b>Backend URL:</b> {API_URL}<br/>
        <b>WS URL:</b> {WS_URL}
      </div>
    </div>
  );
}

export default App;
EOF
    fi
    if [[ -f "$FRONTEND_DIR/requirements.txt" ]]; then
        echo_log "Installing/updating frontend Python dependencies."
        pip install -r "$FRONTEND_DIR/requirements.txt" | tee -a "$LOGFILE"
    fi
else
    echo_log "No frontend directory found. Skipping frontend dependency installation."
fi

# Install API microservice dependencies if needed
if [[ -d "$API_DIR" && -f "$API_DIR/requirements.txt" ]]; then
    echo_log "Installing/updating API microservice dependencies."
    pip install -r "$API_DIR/requirements.txt" | tee -a "$LOGFILE"
fi

# Function to check and clear port
clear_port() {
    local PORT=$1
    if lsof -i :$PORT -t >/dev/null; then
        PID_TO_KILL=$(lsof -i :$PORT -t)
        echo_log "Port $PORT is in use by PID $PID_TO_KILL. Attempting to terminate."
        kill -9 $PID_TO_KILL
        sleep 2
        if lsof -i :$PORT -t >/dev/null; then
            echo_log "Failed to free port $PORT. Exiting."
            exit 1
        fi
        echo_log "Port $PORT is now free."
    else
        echo_log "Port $PORT is free."
    fi
}

# Clear all needed ports
clear_port $APP_PORT
clear_port $FRONTEND_PORT
clear_port $API_PORT

# Spin up FastAPI backend server
LOGCMD="uvicorn main:app --host 0.0.0.0 --port $APP_PORT"
echo_log "Starting FastAPI backend server: $LOGCMD"
$LOGCMD &
SERVER_PID=$!
echo_log "FastAPI backend running with PID $SERVER_PID on port $APP_PORT"

# Spin up Vite+React frontend
if [[ -f "$FRONTEND_DIR/package.json" ]]; then
    echo_log "Starting Vite+React frontend on port $FRONTEND_PORT"
    cd "$FRONTEND_DIR"
    npm run dev -- --port $FRONTEND_PORT &
    FRONTEND_PID=$!
    cd - >/dev/null
    echo_log "Vite+React frontend running with PID $FRONTEND_PID on port $FRONTEND_PORT"
else
    echo_log "No frontend found at $FRONTEND_DIR. Skipping frontend launch."
fi

# Spin up API microservice if present
if [[ -f "$API_DIR/main.py" ]]; then
    echo_log "Starting API microservice on port $API_PORT"
    uvicorn presentation.api.main:app --host 0.0.0.0 --port $API_PORT &
    API_PID=$!
    echo_log "API microservice running with PID $API_PID on port $API_PORT"
else
    echo_log "No API microservice found at $API_DIR/main.py. Skipping API microservice launch."
fi

# Print summary of running services
echo_log "--- Service Summary ---"
echo_log "Backend:   http://localhost:$APP_PORT"
echo_log "Frontend:  http://localhost:$FRONTEND_PORT"
echo_log "API:       http://localhost:$API_PORT"
echo_log "------------------------"

# Wait for all background jobs
wait
