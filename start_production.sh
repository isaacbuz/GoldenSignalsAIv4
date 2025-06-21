#!/bin/bash
# GoldenSignalsAI V2 - Production Startup Script
# This script starts both backend and frontend services

# Start production environment with optimized backend

echo "🚀 Starting GoldenSignalsAI Production Environment (Optimized)"

# Kill any existing processes
echo "Cleaning up existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:3000 | xargs kill -9 2>/dev/null

# Wait for ports to be free
sleep 2

# Start the optimized backend
echo "Starting optimized backend on port 8000..."
cd /Users/isaacbuz/Documents/Projects/FinTech/GoldenSignalsAI_V2
python standalone_backend_optimized.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start the frontend
echo "Starting frontend on port 3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 3

echo "✅ Production environment started!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "📊 Access points:"
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "Performance Stats: http://localhost:8000/api/v1/performance"
echo ""
echo "To stop: ./stop_production.sh"

# Keep script running
wait 