#!/bin/bash

# GoldenSignalsAI Complete System Stop Script
# This script stops all components of the system

echo "🛑 Stopping GoldenSignalsAI System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a port is in use
port_in_use() {
    lsof -i :$1 >/dev/null 2>&1
}

# Read PIDs from file if it exists
if [ -f .pids ]; then
    echo "📋 Reading saved PIDs..."
    source .pids

    if [ ! -z "$BACKEND_PID" ]; then
        echo "   Stopping Backend (PID: $BACKEND_PID)..."
        kill $BACKEND_PID 2>/dev/null
    fi

    if [ ! -z "$FRONTEND_PID" ]; then
        echo "   Stopping Frontend (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID 2>/dev/null
    fi

    if [ ! -z "$GATEWAY_PID" ]; then
        echo "   Stopping MCP Gateway (PID: $GATEWAY_PID)..."
        kill $GATEWAY_PID 2>/dev/null
    fi

    rm -f .pids
fi

# Kill any remaining processes on our ports
echo "🧹 Cleaning up any remaining processes..."
for port in 3000 8000 8001 8002 8003 8004 8005; do
    if port_in_use $port; then
        echo "   Killing process on port $port..."
        lsof -ti :$port | xargs kill -9 2>/dev/null
    fi
done

# Kill any remaining Node processes (frontend)
echo "   Stopping any remaining Node processes..."
pkill -f "npm run dev" 2>/dev/null
pkill -f "vite" 2>/dev/null

# Kill any remaining Python processes (backend)
echo "   Stopping any remaining Python processes..."
pkill -f "simple_backend.py" 2>/dev/null
pkill -f "mcp_gateway.py" 2>/dev/null
pkill -f "mcp_servers" 2>/dev/null

# Optional: Stop Redis (comment out if you want to keep Redis running)
# echo "🔴 Stopping Redis..."
# redis-cli shutdown 2>/dev/null

echo ""
echo "========================================="
echo -e "${GREEN}✅ All services stopped!${NC}"
echo "========================================="
echo ""
echo "💡 To restart, run: ./start_all.sh"
echo ""
