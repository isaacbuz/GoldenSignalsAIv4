#!/bin/bash

# GoldenSignalsAI V3 Status Check Script
# This script checks the health of both frontend and backend services

echo "🔍 GoldenSignalsAI V3 Status Check"
echo "=================================="

# Check Backend
echo ""
echo "🔧 Backend Status (Port 8000):"
echo "------------------------------"

if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend is running"
    BACKEND_STATUS=$(curl -s http://localhost:8000/health | jq -r '.status // "unknown"')
    echo "   Status: $BACKEND_STATUS"
    
    # Test API endpoints
    echo "   Testing API endpoints..."
    
    # Test signals endpoint
    if curl -s http://localhost:8000/api/v1/signals/AAPL > /dev/null; then
        echo "   ✅ Signals API working"
    else
        echo "   ❌ Signals API failed"
    fi
    
    # Test market data endpoint
    if curl -s http://localhost:8000/api/v1/market-data/AAPL > /dev/null; then
        echo "   ✅ Market Data API working"
    else
        echo "   ❌ Market Data API failed"
    fi
    
    # Test agent performance endpoint
    if curl -s http://localhost:8000/api/v1/agents/performance > /dev/null; then
        echo "   ✅ Agent Performance API working"
    else
        echo "   ❌ Agent Performance API failed"
    fi
    
else
    echo "❌ Backend is not running"
    echo "   To start: cd src && python main_simple.py"
fi

# Check Frontend
echo ""
echo "🎨 Frontend Status (Port 3000):"
echo "-------------------------------"

if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Frontend is running"
    echo "   URL: http://localhost:3000"
else
    echo "❌ Frontend is not running"
    echo "   To start: cd frontend && npm run dev"
fi

# Check processes
echo ""
echo "🔄 Running Processes:"
echo "--------------------"

# Check for backend processes
BACKEND_PROCESSES=$(ps aux | grep -E "(python.*main_simple|uvicorn)" | grep -v grep | wc -l)
if [ $BACKEND_PROCESSES -gt 0 ]; then
    echo "✅ Backend processes found: $BACKEND_PROCESSES"
    ps aux | grep -E "(python.*main_simple|uvicorn)" | grep -v grep | head -3
else
    echo "❌ No backend processes found"
fi

echo ""

# Check for frontend processes  
FRONTEND_PROCESSES=$(ps aux | grep -E "(vite|npm.*dev)" | grep -v grep | wc -l)
if [ $FRONTEND_PROCESSES -gt 0 ]; then
    echo "✅ Frontend processes found: $FRONTEND_PROCESSES"
    ps aux | grep -E "(vite|npm.*dev)" | grep -v grep | head -3
else
    echo "❌ No frontend processes found"
fi

# Port usage
echo ""
echo "🌐 Port Usage:"
echo "-------------"
echo "Port 8000 (Backend):"
lsof -i :8000 2>/dev/null | head -5 || echo "   No processes on port 8000"

echo ""
echo "Port 3000 (Frontend):"
lsof -i :3000 2>/dev/null | head -5 || echo "   No processes on port 3000"

# Quick API test
echo ""
echo "🚀 Quick API Test:"
echo "-----------------"

if curl -s http://localhost:8000/health > /dev/null && curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Both services are operational!"
    echo ""
    echo "📊 Sample API Response:"
    echo "----------------------"
    curl -s http://localhost:8000/api/v1/signals/AAPL | jq '{symbol, signal, confidence}' 2>/dev/null || echo "jq not available for JSON formatting"
else
    echo "❌ One or both services are not responding"
fi

echo ""
echo "🎯 Quick Start Commands:"
echo "========================" 
echo "Backend:  cd src && python main_simple.py &"
echo "Frontend: cd frontend && npm run dev &"
echo "Status:   ./check-status.sh"
echo "" 