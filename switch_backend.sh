#!/bin/bash

# Backend Switcher Script for GoldenSignalsAI
# Allows easy switching between mock and full backend

echo "🔧 GoldenSignalsAI Backend Switcher"
echo "=================================="

# Function to stop current backend
stop_backend() {
    echo "🛑 Stopping current backend..."
    pkill -f "python3.*backend" 2>/dev/null || true
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    sleep 2
}

# Function to start mock backend
start_mock() {
    echo "🚀 Starting Mock Backend..."
    python3 simple_mock_backend.py &
    sleep 3
    if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        echo "✅ Mock Backend running at http://localhost:8000"
        echo "📊 Features: Fast startup, predictable data, no external dependencies"
    else
        echo "❌ Failed to start mock backend"
        exit 1
    fi
}

# Function to start full backend
start_full() {
    echo "🚀 Starting Full Backend (with real market data)..."

    # Check dependencies
    python3 -c "import yfinance, pandas, numpy, cachetools" 2>/dev/null || {
        echo "📦 Installing required dependencies..."
        pip3 install yfinance pandas numpy cachetools
    }

    python3 standalone_backend_optimized.py &
    sleep 5
    if curl -s http://localhost:8000/docs > /dev/null 2>&1; then
        echo "✅ Full Backend running at http://localhost:8000"
        echo "📊 Features: Real market data, advanced signals, WebSocket, caching"
    else
        echo "❌ Failed to start full backend"
        exit 1
    fi
}

# Function to show current backend
show_current() {
    if curl -s http://localhost:8000/docs 2>/dev/null | grep -q "Mock Backend"; then
        echo "📊 Current: Mock Backend"
    elif curl -s http://localhost:8000/docs 2>/dev/null | grep -q "Optimized Backend"; then
        echo "📊 Current: Full Backend (Optimized)"
    else
        echo "❌ No backend currently running"
    fi
}

# Parse command line arguments
case "$1" in
    "mock")
        stop_backend
        start_mock
        ;;
    "full")
        stop_backend
        start_full
        ;;
    "status")
        show_current
        ;;
    "stop")
        stop_backend
        echo "✅ Backend stopped"
        ;;
    *)
        echo "Usage: $0 {mock|full|status|stop}"
        echo ""
        echo "Commands:"
        echo "  mock   - Start mock backend (fast, predictable data)"
        echo "  full   - Start full backend (real market data)"
        echo "  status - Show current backend status"
        echo "  stop   - Stop current backend"
        echo ""
        show_current
        exit 1
        ;;
esac
