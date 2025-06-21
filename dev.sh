#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Port configuration
BACKEND_PORT=8000
FRONTEND_PORT=3000

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -i :$port > /dev/null 2>&1; then
        return 0 # Port is in use
    else
        return 1 # Port is free
    fi
}

# Function to kill process on a port
kill_port() {
    local port=$1
    if check_port $port; then
        echo -e "${YELLOW}Port $port is in use. Attempting to free it...${NC}"
        lsof -ti :$port | xargs kill -9 2>/dev/null
        sleep 2
        if check_port $port; then
            echo -e "${RED}Failed to free port $port. Please free it manually and try again.${NC}"
            exit 1
        else
            echo -e "${GREEN}Successfully freed port $port${NC}"
        fi
    fi
}

# Function to handle cleanup
cleanup() {
    echo -e "\n${RED}Shutting down development environment...${NC}"
    
    # Kill processes on our ports
    kill_port $BACKEND_PORT
    kill_port $FRONTEND_PORT
    
    # Kill any remaining background processes
    kill $(jobs -p) 2>/dev/null
    exit 0
}

# Set up trap for cleanup
trap cleanup SIGINT SIGTERM

# Check and free ports before starting
echo -e "${BLUE}Checking ports...${NC}"
kill_port $BACKEND_PORT
kill_port $FRONTEND_PORT

# Start backend
echo -e "${BLUE}Starting backend server on port $BACKEND_PORT...${NC}"
source .venv/bin/activate
python -m uvicorn src.main:app --host 0.0.0.0 --port $BACKEND_PORT --reload --log-level debug &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start frontend
echo -e "${BLUE}Starting frontend development server on port $FRONTEND_PORT...${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!

# Wait for both processes
echo -e "${GREEN}Development environment is running!${NC}"
echo -e "${GREEN}Backend: http://localhost:$BACKEND_PORT${NC}"
echo -e "${GREEN}Frontend: http://localhost:$FRONTEND_PORT${NC}"
echo -e "${GREEN}Press Ctrl+C to stop all services${NC}"

# Keep script running and handle cleanup
wait 