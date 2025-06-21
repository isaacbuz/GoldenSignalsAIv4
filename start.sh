#!/bin/bash

# GoldenSignalsAI Master Startup Script
# This script manages all services for the GoldenSignalsAI platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_ROOT"

# Default values
MODE="dev"
SERVICES="all"
DETACHED=false

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to kill process on port
kill_port() {
    local port=$1
    local pids=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pids" ]; then
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed"
        exit 1
    fi
    
    # Check virtual environment
    if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
        print_warning "Python virtual environment not found. Creating one..."
        python3 -m venv .venv
    fi
    
    print_success "Prerequisites checked"
}

# Function to activate virtual environment
activate_venv() {
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        source venv/bin/activate
    else
        print_error "No virtual environment found"
        exit 1
    fi
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Python dependencies
    activate_venv
    pip install -r requirements.txt 2>/dev/null || print_warning "Some Python packages failed to install"
    
    # Frontend dependencies
    if [ -d "frontend" ]; then
        cd frontend
        npm install
        cd ..
    fi
    
    print_success "Dependencies installed"
}

# Function to start backend
start_backend() {
    print_status "Starting backend server..."
    
    # Kill existing backend process
    kill_port 8000
    
    activate_venv
    
    if [ "$DETACHED" = true ]; then
        nohup python simple_backend.py > backend.log 2>&1 &
        echo $! > backend.pid
        print_success "Backend started in background (PID: $(cat backend.pid))"
    else
        python simple_backend.py &
        BACKEND_PID=$!
        print_success "Backend started (PID: $BACKEND_PID)"
    fi
    
    # Wait for backend to be ready
    sleep 3
    if check_port 8000; then
        print_success "Backend is running on http://localhost:8000"
    else
        print_error "Backend failed to start"
        return 1
    fi
}

# Function to start frontend
start_frontend() {
    print_status "Starting frontend..."
    
    # Kill existing frontend process
    kill_port 3000
    
    if [ ! -d "frontend" ]; then
        print_error "Frontend directory not found"
        return 1
    fi
    
    cd frontend
    
    if [ "$DETACHED" = true ]; then
        nohup npm run dev > ../frontend.log 2>&1 &
        echo $! > ../frontend.pid
        print_success "Frontend started in background (PID: $(cat ../frontend.pid))"
    else
        npm run dev &
        FRONTEND_PID=$!
        print_success "Frontend started (PID: $FRONTEND_PID)"
    fi
    
    cd ..
    
    # Wait for frontend to be ready
    sleep 5
    if check_port 3000; then
        print_success "Frontend is running on http://localhost:3000"
    else
        print_error "Frontend failed to start"
        return 1
    fi
}

# Function to start databases
start_databases() {
    print_status "Starting databases..."
    
    # Check if PostgreSQL is running
    if command -v pg_isready &> /dev/null && pg_isready -q; then
        print_success "PostgreSQL is already running"
    else
        print_warning "PostgreSQL is not running. Please start it manually or use Docker"
    fi
    
    # Check if Redis is running
    if command -v redis-cli &> /dev/null && redis-cli ping &> /dev/null; then
        print_success "Redis is already running"
    else
        print_warning "Redis is not running. Please start it manually or use Docker"
    fi
}

# Function to start all services
start_all() {
    print_status "Starting all services..."
    
    check_prerequisites
    
    if [ "$MODE" = "dev" ]; then
        start_databases
        start_backend
        start_frontend
    elif [ "$MODE" = "docker" ]; then
        start_docker
    fi
    
    print_success "All services started successfully!"
    echo ""
    echo "Access the application at:"
    echo "  Frontend: http://localhost:3000"
    echo "  Backend API: http://localhost:8000"
    echo "  API Docs: http://localhost:8000/docs"
}

# Function to start with Docker
start_docker() {
    print_status "Starting services with Docker Compose..."
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    docker-compose up -d
    
    print_success "Docker services started"
}

# Function to stop all services
stop_all() {
    print_status "Stopping all services..."
    
    # Stop backend
    if [ -f "backend.pid" ]; then
        kill $(cat backend.pid) 2>/dev/null || true
        rm backend.pid
    fi
    kill_port 8000
    
    # Stop frontend
    if [ -f "frontend.pid" ]; then
        kill $(cat frontend.pid) 2>/dev/null || true
        rm frontend.pid
    fi
    kill_port 3000
    
    # Stop Docker services if running
    if [ "$MODE" = "docker" ] && command -v docker-compose &> /dev/null; then
        docker-compose down
    fi
    
    print_success "All services stopped"
}

# Function to restart services
restart_all() {
    stop_all
    sleep 2
    start_all
}

# Function to show status
show_status() {
    print_status "Service Status:"
    echo ""
    
    # Backend status
    if check_port 8000; then
        print_success "Backend: Running on port 8000"
    else
        print_error "Backend: Not running"
    fi
    
    # Frontend status
    if check_port 3000; then
        print_success "Frontend: Running on port 3000"
    else
        print_error "Frontend: Not running"
    fi
    
    # Database status
    if command -v pg_isready &> /dev/null && pg_isready -q; then
        print_success "PostgreSQL: Running"
    else
        print_warning "PostgreSQL: Not running or not installed"
    fi
    
    if command -v redis-cli &> /dev/null && redis-cli ping &> /dev/null; then
        print_success "Redis: Running"
    else
        print_warning "Redis: Not running or not installed"
    fi
}

# Function to show logs
show_logs() {
    local service=$1
    
    case $service in
        backend)
            if [ -f "backend.log" ]; then
                tail -f backend.log
            else
                print_error "No backend logs found"
            fi
            ;;
        frontend)
            if [ -f "frontend.log" ]; then
                tail -f frontend.log
            else
                print_error "No frontend logs found"
            fi
            ;;
        *)
            print_error "Unknown service: $service"
            ;;
    esac
}

# Function to show help
show_help() {
    echo "GoldenSignalsAI Master Startup Script"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  start       Start all services (default)"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  status      Show service status"
    echo "  logs        Show logs (requires service name)"
    echo "  install     Install dependencies"
    echo "  help        Show this help message"
    echo ""
    echo "Options:"
    echo "  --mode      Startup mode: dev (default), docker, prod"
    echo "  --services  Services to start: all (default), backend, frontend"
    echo "  --detached  Run services in background"
    echo ""
    echo "Examples:"
    echo "  $0                          # Start all services in dev mode"
    echo "  $0 start --mode docker      # Start with Docker"
    echo "  $0 start --services backend # Start only backend"
    echo "  $0 logs backend            # Show backend logs"
    echo "  $0 status                  # Check service status"
}

# Parse command line arguments
COMMAND=${1:-start}
shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --services)
            SERVICES="$2"
            shift 2
            ;;
        --detached|-d)
            DETACHED=true
            shift
            ;;
        *)
            if [ "$COMMAND" = "logs" ]; then
                SERVICE_NAME="$1"
            fi
            shift
            ;;
    esac
done

# Main execution
case $COMMAND in
    start)
        if [ "$SERVICES" = "all" ]; then
            start_all
        elif [ "$SERVICES" = "backend" ]; then
            check_prerequisites
            start_backend
        elif [ "$SERVICES" = "frontend" ]; then
            check_prerequisites
            start_frontend
        else
            print_error "Unknown service: $SERVICES"
            exit 1
        fi
        
        if [ "$DETACHED" = false ]; then
            echo ""
            print_status "Press Ctrl+C to stop all services..."
            wait
        fi
        ;;
    stop)
        stop_all
        ;;
    restart)
        restart_all
        ;;
    status)
        show_status
        ;;
    logs)
        if [ -z "$SERVICE_NAME" ]; then
            print_error "Please specify a service name"
            echo "Usage: $0 logs [backend|frontend]"
            exit 1
        fi
        show_logs "$SERVICE_NAME"
        ;;
    install)
        install_dependencies
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac 