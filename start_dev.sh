#!/bin/bash

# GoldenSignalsAI Development Startup Script
# This script starts both the optimized backend and frontend in development mode

echo "🚀 Starting GoldenSignalsAI Development Environment"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is required but not installed."
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    exit 1
fi

# Check if we're in the correct directory
if [ ! -f "standalone_backend_optimized.py" ]; then
    echo "❌ standalone_backend_optimized.py not found. Please run from the project root."
    exit 1
fi

if [ ! -d "frontend" ]; then
    echo "❌ frontend directory not found. Please run from the project root."
    exit 1
fi

# Function to cleanup background processes
cleanup() {
    echo "🛑 Shutting down development servers..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
        echo "✅ Backend stopped"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
        echo "✅ Frontend stopped"
    fi
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

# Install backend dependencies if needed
echo "🔧 Checking backend dependencies..."
python3 -c "import fastapi, uvicorn, yfinance, pandas, numpy, cachetools" 2>/dev/null || {
    echo "📦 Installing backend dependencies..."
    pip3 install fastapi uvicorn yfinance pandas numpy cachetools websockets python-multipart
}

# Start the optimized backend
echo "🚀 Starting Optimized Backend (Real Market Data)..."
python3 standalone_backend_optimized.py &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 5

# Check if backend is running
if curl -s http://localhost:8000/docs > /dev/null 2>&1; then
    echo "✅ Backend is running at http://localhost:8000"
    echo "📊 API Documentation: http://localhost:8000/docs"
else
    echo "❌ Backend failed to start"
    exit 1
fi

# Install frontend dependencies if needed
echo "🔧 Checking frontend dependencies..."
cd frontend
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

# Start the frontend
echo "🚀 Starting Frontend..."
npm run dev &
FRONTEND_PID=$!

# Wait for frontend to start
echo "⏳ Waiting for frontend to start..."
sleep 5

# Check if frontend is running
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Frontend is running at http://localhost:3000"
else
    echo "❌ Frontend failed to start"
    exit 1
fi

echo ""
echo "🎉 GoldenSignalsAI Development Environment is Ready!"
echo "=================================================="
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "🔌 WebSocket: ws://localhost:8000/ws/signals"
echo ""
echo "Features Available:"
echo "✅ Real market data from yfinance"
echo "✅ Advanced signal generation with technical indicators"
echo "✅ WebSocket live updates"
echo "✅ Performance monitoring and caching"
echo "✅ Signal quality reporting"
echo "✅ Backtesting capabilities"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "=================================================="

# Keep the script running
wait
