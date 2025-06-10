#!/bin/bash

# GoldenSignalsAI V3 Development Utilities
# Usage: ./dev-utils.sh [command]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if required tools are installed
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python
    if command -v python3 &> /dev/null; then
        log_success "Python3 is installed"
    else
        log_error "Python3 is required but not installed"
        exit 1
    fi
    
    # Check Node.js
    if command -v node &> /dev/null; then
        log_success "Node.js is installed"
    else
        log_error "Node.js is required but not installed"
        exit 1
    fi
    
    # Check npm
    if command -v npm &> /dev/null; then
        log_success "npm is installed"
    else
        log_error "npm is required but not installed"
        exit 1
    fi
    
    # Check if virtual environment exists
    if [ -d "venv" ]; then
        log_success "Python virtual environment exists"
    else
        log_warning "Python virtual environment not found"
        log_info "Creating virtual environment..."
        python3 -m venv venv
        log_success "Virtual environment created"
    fi
}

# Setup the project
setup() {
    log_info "Setting up GoldenSignalsAI V3..."
    
    check_dependencies
    
    # Activate virtual environment and install Python dependencies
    log_info "Installing Python dependencies..."
    source venv/bin/activate
    pip install -r requirements.txt 2>/dev/null || log_warning "requirements.txt not found, skipping Python deps"
    
    # Install Node.js dependencies
    log_info "Installing Node.js dependencies..."
    cd frontend
    npm install
    cd ..
    
    log_success "Setup complete!"
}

# Kill processes on specific ports
kill_port() {
    local port=$1
    log_info "Killing processes on port $port..."
    
    if lsof -ti:$port > /dev/null 2>&1; then
        lsof -ti:$port | xargs kill -9
        log_success "Killed processes on port $port"
    else
        log_info "No processes found on port $port"
    fi
}

# Clean development environment
clean() {
    log_info "Cleaning development environment..."
    
    # Kill development servers
    kill_port 8000
    kill_port 3000
    
    # Clean Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    
    # Clean Node.js cache
    if [ -d "frontend/node_modules/.cache" ]; then
        rm -rf frontend/node_modules/.cache
        log_success "Cleared Vite cache"
    fi
    
    log_success "Environment cleaned!"
}

# Start backend server
start_backend() {
    log_info "Starting backend server..."
    
    # Check if virtual environment is activated
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        source venv/bin/activate
    fi
    
    cd src
    python main_simple.py &
    BACKEND_PID=$!
    cd ..
    
    # Wait a moment and check if server started
    sleep 3
    if curl -s http://localhost:8000/health > /dev/null; then
        log_success "Backend server started on port 8000 (PID: $BACKEND_PID)"
        echo $BACKEND_PID > .backend.pid
    else
        log_error "Failed to start backend server"
        exit 1
    fi
}

# Start frontend server
start_frontend() {
    log_info "Starting frontend server..."
    
    cd frontend
    npx vite --port 3000 &
    FRONTEND_PID=$!
    cd ..
    
    # Wait a moment and check if server started
    sleep 3
    if curl -s http://localhost:3000 > /dev/null; then
        log_success "Frontend server started on port 3000 (PID: $FRONTEND_PID)"
        echo $FRONTEND_PID > .frontend.pid
    else
        log_error "Failed to start frontend server"
        exit 1
    fi
}

# Start both servers
start_all() {
    log_info "Starting all services..."
    
    # Clean any existing processes
    kill_port 8000
    kill_port 3000
    
    # Start backend
    start_backend
    
    # Start frontend
    start_frontend
    
    log_success "All services started!"
    log_info "Backend: http://localhost:8000"
    log_info "Frontend: http://localhost:3000"
    log_info "API Docs: http://localhost:8000/docs"
}

# Stop all services
stop_all() {
    log_info "Stopping all services..."
    
    # Kill using saved PIDs if available
    if [ -f ".backend.pid" ]; then
        BACKEND_PID=$(cat .backend.pid)
        kill $BACKEND_PID 2>/dev/null || true
        rm .backend.pid
    fi
    
    if [ -f ".frontend.pid" ]; then
        FRONTEND_PID=$(cat .frontend.pid)
        kill $FRONTEND_PID 2>/dev/null || true
        rm .frontend.pid
    fi
    
    # Fallback to port killing
    kill_port 8000
    kill_port 3000
    
    log_success "All services stopped!"
}

# Show status
status() {
    echo "🔍 GoldenSignalsAI V3 Status"
    echo "============================"
    
    # Check backend
    if curl -s http://localhost:8000/health > /dev/null; then
        log_success "Backend: Running on port 8000"
    else
        log_error "Backend: Not running"
    fi
    
    # Check frontend
    if curl -s http://localhost:3000 > /dev/null; then
        log_success "Frontend: Running on port 3000"
    else
        log_error "Frontend: Not running"
    fi
}

# Show logs
logs() {
    local service=$1
    
    if [ "$service" = "backend" ] && [ -f ".backend.pid" ]; then
        BACKEND_PID=$(cat .backend.pid)
        tail -f /proc/$BACKEND_PID/fd/1 2>/dev/null || log_error "Backend logs not available"
    elif [ "$service" = "frontend" ] && [ -f ".frontend.pid" ]; then
        FRONTEND_PID=$(cat .frontend.pid)
        tail -f /proc/$FRONTEND_PID/fd/1 2>/dev/null || log_error "Frontend logs not available"
    else
        log_error "Specify 'backend' or 'frontend' for logs, or service not running"
    fi
}

# Test API endpoints
test_api() {
    log_info "Testing API endpoints..."
    
    if ! curl -s http://localhost:8000/health > /dev/null; then
        log_error "Backend is not running"
        exit 1
    fi
    
    # Test health endpoint
    HEALTH=$(curl -s http://localhost:8000/health | jq -r '.status // "unknown"')
    log_success "Health: $HEALTH"
    
    # Test signals endpoint
    if curl -s http://localhost:8000/api/v1/signals/AAPL > /dev/null; then
        SIGNAL=$(curl -s http://localhost:8000/api/v1/signals/AAPL | jq -r '.signal // "N/A"')
        log_success "Signals: AAPL signal is $SIGNAL"
    else
        log_error "Signals API failed"
    fi
    
    # Test agents endpoint
    if curl -s http://localhost:8000/api/v1/agents/performance > /dev/null; then
        AGENTS=$(curl -s http://localhost:8000/api/v1/agents/performance | jq '.agents | length')
        log_success "Agents: $AGENTS agents active"
    else
        log_error "Agents API failed"
    fi
}

# Main command handler
case "${1:-help}" in
    setup)
        setup
        ;;
    clean)
        clean
        ;;
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    restart)
        stop_all
        start_all
        ;;
    status)
        status
        ;;
    backend)
        start_backend
        ;;
    frontend)
        start_frontend
        ;;
    logs)
        logs $2
        ;;
    test)
        test_api
        ;;
    help|*)
        echo "GoldenSignalsAI V3 Development Utilities"
        echo ""
        echo "Usage: ./dev-utils.sh [command]"
        echo ""
        echo "Commands:"
        echo "  setup      - Setup the development environment"
        echo "  start      - Start both backend and frontend servers"
        echo "  stop       - Stop all services"
        echo "  restart    - Restart all services"
        echo "  status     - Show service status"
        echo "  clean      - Clean development environment"
        echo "  backend    - Start only backend server"
        echo "  frontend   - Start only frontend server"
        echo "  logs       - Show logs (logs [backend|frontend])"
        echo "  test       - Test API endpoints"
        echo "  help       - Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./dev-utils.sh setup     # Initial setup"
        echo "  ./dev-utils.sh start     # Start everything"
        echo "  ./dev-utils.sh status    # Check status"
        echo "  ./dev-utils.sh logs backend  # View backend logs"
        ;;
esac 