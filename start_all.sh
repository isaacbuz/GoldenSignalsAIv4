#!/bin/bash

# GoldenSignalsAI Complete System Startup Script
# This script starts all components of the system

echo "🚀 Starting GoldenSignalsAI Complete System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
port_in_use() {
    lsof -i :$1 >/dev/null 2>&1
}

# Function to wait for a service to be ready
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=0
    
    echo "⏳ Waiting for $name to be ready..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url" > /dev/null; then
            echo -e "${GREEN}✅ $name is ready!${NC}"
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    echo -e "${RED}❌ $name failed to start${NC}"
    return 1
}

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi

if ! command_exists node; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    exit 1
fi

if ! command_exists redis-cli; then
    echo -e "${RED}❌ Redis is not installed${NC}"
    exit 1
fi

if ! command_exists psql; then
    echo -e "${YELLOW}⚠️  PostgreSQL client is not installed (optional)${NC}"
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env from env.example..."
    cp env.example .env
    echo -e "${YELLOW}⚠️  Please edit .env and add your API keys${NC}"
fi

# Start Redis if not running
if ! pgrep -x "redis-server" > /dev/null; then
    echo "🔴 Starting Redis..."
    redis-server --daemonize yes
    sleep 2
else
    echo -e "${GREEN}✅ Redis is already running${NC}"
fi

# Check if PostgreSQL is running (optional)
if command_exists pg_isready; then
    if pg_isready -q; then
        echo -e "${GREEN}✅ PostgreSQL is running${NC}"
    else
        echo -e "${YELLOW}⚠️  PostgreSQL is not running (using SQLite fallback)${NC}"
    fi
fi

# Kill any existing processes on our ports
echo "🧹 Cleaning up existing processes..."
for port in 3000 8000 8001 8002 8003 8004 8005; do
    if port_in_use $port; then
        echo "   Killing process on port $port..."
        lsof -ti :$port | xargs kill -9 2>/dev/null
    fi
done

sleep 2

# Create necessary directories
mkdir -p logs data/market_cache data/rate_limit_cache

# Start Backend Services
echo ""
echo "🔧 Starting Backend Services..."

# Start Simple Backend (main API)
echo "   Starting Simple Backend (port 8000)..."
python3 simple_backend.py > logs/simple_backend.log 2>&1 &
BACKEND_PID=$!
echo "   PID: $BACKEND_PID"

# Wait for backend to be ready
wait_for_service "http://localhost:8000" "Simple Backend"

# Optional: Start MCP Gateway (uncomment when ready)
# echo "   Starting MCP Gateway (port 8000)..."
# python3 mcp_servers/mcp_gateway.py > logs/mcp_gateway.log 2>&1 &
# GATEWAY_PID=$!
# echo "   PID: $GATEWAY_PID"

# Optional: Start individual MCP servers (uncomment when ready)
# echo "   Starting Trading Signals MCP Server (port 8001)..."
# python3 mcp_servers/trading_signals_server.py > logs/trading_signals.log 2>&1 &

# echo "   Starting Market Data MCP Server (port 8002)..."
# python3 mcp_servers/market_data_server.py > logs/market_data.log 2>&1 &

# Start Frontend
echo ""
echo "🎨 Starting Frontend..."
cd frontend
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo "   PID: $FRONTEND_PID"

# Wait for frontend to be ready
sleep 5
wait_for_service "http://localhost:3000" "Frontend"

# Summary
echo ""
echo "========================================="
echo -e "${GREEN}🎉 GoldenSignalsAI is running!${NC}"
echo "========================================="
echo ""
echo "📊 Access Points:"
echo "   • Frontend: http://localhost:3000"
echo "   • Backend API: http://localhost:8000"
echo "   • API Docs: http://localhost:8000/docs"
echo ""
echo "📝 Logs:"
echo "   • Backend: logs/simple_backend.log"
echo "   • Frontend: logs/frontend.log"
echo ""
echo "🛑 To stop all services, run: ./stop_all.sh"
echo ""
echo "💡 Tips:"
echo "   • Check .env for API key configuration"
echo "   • Monitor logs for any errors"
echo "   • Use 'redis-cli' to inspect cache"
echo ""

# Save PIDs for stop script
echo "BACKEND_PID=$BACKEND_PID" > .pids
echo "FRONTEND_PID=$FRONTEND_PID" >> .pids

# Keep script running and show logs
echo "📜 Showing combined logs (Ctrl+C to exit)..."
echo "========================================="
tail -f logs/*.log 