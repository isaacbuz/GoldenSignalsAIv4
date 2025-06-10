#!/bin/bash

# GoldenSignalsAI V3 - Unified UI Startup Script
# This script ensures consistent port usage by killing existing processes

set -e

echo "🚀 Starting GoldenSignalsAI V3 UI..."
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ports
BACKEND_PORT=8000
FRONTEND_PORT=3000

echo -e "${BLUE}📋 Cleaning up existing processes...${NC}"

# Kill existing processes on backend port
echo -e "${YELLOW}🔪 Killing processes on port ${BACKEND_PORT} (Backend)...${NC}"
if command -v node >/dev/null 2>&1; then
    node scripts/kill-port.js ${BACKEND_PORT} || echo "✅ Port ${BACKEND_PORT} already available"
else
    echo "⚠️  Node.js not found, using system commands..."
    if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
        lsof -ti:${BACKEND_PORT} | xargs kill -9 2>/dev/null || echo "✅ Port ${BACKEND_PORT} already available"
    fi
fi

# Kill existing processes on frontend port
echo -e "${YELLOW}🔪 Killing processes on port ${FRONTEND_PORT} (Frontend)...${NC}"
if command -v node >/dev/null 2>&1; then
    node scripts/kill-port.js ${FRONTEND_PORT} || echo "✅ Port ${FRONTEND_PORT} already available"
else
    echo "⚠️  Node.js not found, using system commands..."
    if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
        lsof -ti:${FRONTEND_PORT} | xargs kill -9 2>/dev/null || echo "✅ Port ${FRONTEND_PORT} already available"
    fi
fi

echo -e "${GREEN}✅ Ports cleaned successfully!${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found. Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${BLUE}🐍 Activating Python virtual environment...${NC}"
source venv/bin/activate

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${YELLOW}📦 Installing frontend dependencies...${NC}"
    cd frontend
    npm install
    cd ..
fi

echo -e "${GREEN}🎯 Starting backend server on port ${BACKEND_PORT}...${NC}"
# Start backend in background
cd src
python main_simple.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Check if backend started successfully
if kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${GREEN}✅ Backend server started successfully (PID: $BACKEND_PID)${NC}"
else
    echo -e "${RED}❌ Backend server failed to start${NC}"
    exit 1
fi

echo -e "${GREEN}🎨 Starting frontend development server on port ${FRONTEND_PORT}...${NC}"
# Start frontend
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Wait a moment for frontend to start
sleep 5

# Check if frontend started successfully
if kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${GREEN}✅ Frontend development server started successfully (PID: $FRONTEND_PID)${NC}"
else
    echo -e "${RED}❌ Frontend development server failed to start${NC}"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 GoldenSignalsAI V3 UI is now running!${NC}"
echo "==========================================="
echo -e "${BLUE}📊 Backend API:${NC}    http://localhost:${BACKEND_PORT}"
echo -e "${BLUE}🎨 Frontend UI:${NC}    http://localhost:${FRONTEND_PORT}"
echo -e "${BLUE}📚 API Docs:${NC}       http://localhost:${BACKEND_PORT}/docs"
echo -e "${BLUE}❤️  Health Check:${NC}  http://localhost:${BACKEND_PORT}/health"
echo ""
echo -e "${YELLOW}💡 Press Ctrl+C to stop all services${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down services...${NC}"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}✅ All services stopped${NC}"
    exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup INT

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
