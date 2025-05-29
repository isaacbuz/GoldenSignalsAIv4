#!/bin/bash

# File: manage_servers.sh
# Usage: ./manage_servers.sh [start|stop|restart|status]
# Controls both FastAPI backend and Next.js frontend servers

PID_FILE=".servers.pid"
BACKEND_PORT=8000
FRONTEND_DIR="frontend"

start_servers() {
  if [ -f "$PID_FILE" ]; then
    echo "Servers may already be running. Use './manage_servers.sh status' or './manage_servers.sh stop' first."
    exit 1
  fi

  echo "Starting FastAPI backend on port $BACKEND_PORT..."
  uvicorn main:app --reload --port $BACKEND_PORT > backend.log 2>&1 &
  BACKEND_PID=$!
  echo "Backend PID: $BACKEND_PID"

  echo "Starting Next.js frontend..."
  cd "$FRONTEND_DIR" && npm run dev > ../frontend.log 2>&1 &
  FRONTEND_PID=$!
  cd - > /dev/null
  echo "Frontend PID: $FRONTEND_PID"

  echo "$BACKEND_PID
$FRONTEND_PID" > "$PID_FILE"
  echo "Servers started."
}

stop_servers() {
  if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found. Servers may not be running."
    exit 1
  fi
  PIDS=($(cat "$PID_FILE"))
  echo "Stopping servers with PIDs: ${PIDS[*]}"
  for pid in "${PIDS[@]}"; do
    if kill -0 $pid 2>/dev/null; then
      kill $pid
      echo "Stopped process $pid"
    else
      echo "Process $pid not running."
    fi
  done
  rm -f "$PID_FILE"
  echo "Servers stopped."
}

status_servers() {
  if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found. Servers may not be running."
    exit 0
  fi
  PIDS=($(cat "$PID_FILE"))
  for pid in "${PIDS[@]}"; do
    if kill -0 $pid 2>/dev/null; then
      echo "Process $pid is running."
    else
      echo "Process $pid is NOT running."
    fi
  done
}

case "$1" in
  start)
    start_servers
    ;;
  stop)
    stop_servers
    ;;
  restart)
    stop_servers
    start_servers
    ;;
  status)
    status_servers
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status}"
    exit 1
    ;;
esac
