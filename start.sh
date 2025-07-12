#!/bin/bash

# GoldenSignalsAI Startup Script
# This script helps you set up and run the application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Check if running on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    print_status "Running on macOS"
else
    print_status "Running on Linux/Unix"
fi

print_step "GoldenSignalsAI Setup & Startup"

# Check prerequisites
print_step "Checking Prerequisites"

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

python_version=$(python3 --version | cut -d' ' -f2)
print_status "Python version: $python_version"

# Check Node.js
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js 16 or higher."
    exit 1
fi

node_version=$(node --version)
print_status "Node.js version: $node_version"

# Check npm
if ! command -v npm &> /dev/null; then
    print_error "npm is not installed. Please install npm."
    exit 1
fi

npm_version=$(npm --version)
print_status "npm version: $npm_version"

# Setup Backend
print_step "Setting up Backend"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Setup environment variables
if [ ! -f ".env" ]; then
    print_warning "No .env file found. Creating from template..."
    cp config.example.env .env
    print_warning "Please edit .env file with your actual configuration values"
fi

# Setup Frontend
print_step "Setting up Frontend"

cd frontend

# Install Node.js dependencies
if [ ! -d "node_modules" ]; then
    print_status "Installing Node.js dependencies..."
    npm install
else
    print_status "Node.js dependencies already installed"
fi

# Build frontend (optional)
if [ "$1" = "--build" ]; then
    print_status "Building frontend..."
    npm run build
fi

cd ..

# Setup Database (SQLite for development)
print_step "Setting up Database"

# Create logs directory
mkdir -p logs

# Create data directory
mkdir -p data

print_status "Database setup complete (using SQLite for development)"

# Start Services
print_step "Starting Services"

# Function to start backend
start_backend() {
    print_status "Starting backend server..."
    cd src
    python main.py &
    BACKEND_PID=$!
    cd ..
    print_status "Backend started with PID: $BACKEND_PID"
}

# Function to start frontend
start_frontend() {
    print_status "Starting frontend development server..."
    cd frontend
    npm run dev &
    FRONTEND_PID=$!
    cd ..
    print_status "Frontend started with PID: $FRONTEND_PID"
}

# Start backend
start_backend

# Wait a bit for backend to start
sleep 3

# Start frontend
start_frontend

# Wait a bit for frontend to start
sleep 3

print_step "Application Started Successfully!"

print_status "🚀 GoldenSignalsAI is now running:"
print_status "   • Backend API: http://localhost:8000"
print_status "   • Frontend App: http://localhost:3000"
print_status "   • API Docs: http://localhost:8000/docs"
print_status "   • WebSocket: ws://localhost:8000/ws"

print_step "Useful Commands"
echo "Backend logs: tail -f logs/app.log"
echo "Stop services: pkill -f 'python main.py' && pkill -f 'npm run dev'"
echo "Restart backend: cd src && python main.py"
echo "Restart frontend: cd frontend && npm run dev"

print_step "Next Steps"
echo "1. Open http://localhost:3000 in your browser"
echo "2. Try generating signals for symbols like AAPL, GOOGL, TSLA"
echo "3. Check the WebSocket connection for real-time updates"
echo "4. Visit http://localhost:8000/docs for API documentation"

# Function to handle cleanup on exit
cleanup() {
    print_step "Shutting down services..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    print_status "Services stopped"
}

# Trap cleanup function on script exit
trap cleanup EXIT

# Keep script running
if [ "$1" = "--daemon" ]; then
    print_status "Running in daemon mode. Press Ctrl+C to stop."
    while true; do
        sleep 10
    done
else
    print_status "Press Ctrl+C to stop all services"
    wait
fi 