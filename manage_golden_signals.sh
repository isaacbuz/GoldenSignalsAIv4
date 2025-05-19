#!/usr/bin/env bash

# Robust Manager for GoldenSignalsAI: Controls backend, frontend, API, and logs
# Usage: ./manage_golden_signals.sh [start|stop|restart|status|logs]

APP_PORT=8000
FRONTEND_PORT=3000
API_PORT=8080
LOGFILE="run_golden_signals.log"
ENV_NAME="goldensignalsai"
REQUIREMENTS="requirements.txt"
FRONTEND_DIR="presentation/frontend"
API_DIR="presentation/api"

# PID files for tracking
BACKEND_PID_FILE=".backend.pid"
FRONTEND_PID_FILE=".frontend.pid"
API_PID_FILE=".api.pid"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

# Start backend
start_backend() {
    log "Starting FastAPI backend on port $APP_PORT..."
    conda run -n "$ENV_NAME" uvicorn main:app --host 0.0.0.0 --port $APP_PORT &
    echo $! > "$BACKEND_PID_FILE"
    log "FastAPI backend started with PID $(cat $BACKEND_PID_FILE)"
}

# Start frontend
start_frontend() {
    log "Starting React frontend on port $FRONTEND_PORT..."
    cd "$FRONTEND_DIR"
    npm start &
    echo $! > "../../$FRONTEND_PID_FILE"
    cd - >/dev/null
    log "Frontend started with PID $(cat $FRONTEND_PID_FILE)"
}

# Start API microservice
start_api() {
    log "Starting API microservice on port $API_PORT..."
    conda run -n "$ENV_NAME" uvicorn presentation.api.main:app --host 0.0.0.0 --port $API_PORT &
    echo $! > "$API_PID_FILE"
    log "API microservice started with PID $(cat $API_PID_FILE)"
}

# Stop all services
stop_all() {
    for pidfile in "$BACKEND_PID_FILE" "$FRONTEND_PID_FILE" "$API_PID_FILE"; do
        if [[ -f "$pidfile" ]]; then
            PID=$(cat "$pidfile")
            if kill -0 $PID 2>/dev/null; then
                kill $PID
                log "Stopped process $PID from $pidfile"
            fi
            rm -f "$pidfile"
        fi
    done
}

# Status of all services
status() {
    for pidfile in "$BACKEND_PID_FILE" "$FRONTEND_PID_FILE" "$API_PID_FILE"; do
        if [[ -f "$pidfile" ]]; then
            PID=$(cat "$pidfile")
            if kill -0 $PID 2>/dev/null; then
                echo "$pidfile: RUNNING (PID $PID)"
            else
                echo "$pidfile: NOT RUNNING (stale PID $PID)"
            fi
        else
            echo "$pidfile: NOT RUNNING"
        fi
    done
}

# Tail logs in real time
logs() {
    tail -f "$LOGFILE"
}

case "$1" in
    start)
        stop_all
        start_backend
        start_frontend
        start_api
        ;;
    stop)
        stop_all
        ;;
    restart)
        stop_all
        start_backend
        start_frontend
        start_api
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        ;;
esac
