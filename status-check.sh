#!/bin/bash

# GoldenSignalsAI V3 Comprehensive Status Check Script
echo "🔍 GoldenSignalsAI V3 System Status Check"
echo "=========================================="

# Check Backend Status
echo ""
echo "🎯 Backend Service (Port 8000):"
echo "-------------------------------"

if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend is running"

    # Get detailed health status
    HEALTH_STATUS=$(curl -s http://localhost:8000/health | jq -r '.status // "unknown"')
    echo "   📊 Status: $HEALTH_STATUS"

    # Test key endpoints
    echo "   🔍 Testing API endpoints..."

    # Test signals endpoint
    if curl -s http://localhost:8000/api/v1/signals/AAPL > /dev/null; then
        SIGNAL_TYPE=$(curl -s http://localhost:8000/api/v1/signals/AAPL | jq -r '.signal // "N/A"')
        echo "   ✅ Signals API: $SIGNAL_TYPE signal for AAPL"
    else
        echo "   ❌ Signals API failed"
    fi

    # Test agents performance endpoint
    if curl -s http://localhost:8000/api/v1/agents/performance > /dev/null; then
        AGENT_COUNT=$(curl -s http://localhost:8000/api/v1/agents/performance | jq '.agents | length')
        echo "   ✅ Agents API: $AGENT_COUNT agents active"
    else
        echo "   ❌ Agents API failed"
    fi

    # Test market data endpoint
    if curl -s http://localhost:8000/api/v1/market-data/AAPL > /dev/null; then
        PRICE=$(curl -s http://localhost:8000/api/v1/market-data/AAPL | jq -r '.price // "N/A"')
        echo "   ✅ Market Data API: AAPL at \$$PRICE"
    else
        echo "   ❌ Market Data API failed"
    fi

    # Test market summary endpoint
    if curl -s http://localhost:8000/api/v1/market-summary > /dev/null; then
        echo "   ✅ Market Summary API working"
    else
        echo "   ❌ Market Summary API failed"
    fi

else
    echo "❌ Backend is not running"
    echo "   💡 Try: cd src && python main_simple.py"
fi

# Check Frontend Status
echo ""
echo "🎨 Frontend Service (Port 3000):"
echo "--------------------------------"

if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Frontend is running"

    # Check if it's serving the React app
    if curl -s http://localhost:3000 | grep -q "React"; then
        echo "   📱 React app is serving correctly"
    else
        echo "   ⚠️  Frontend is running but may have issues"
    fi

    echo "   🌐 URL: http://localhost:3000"

else
    echo "❌ Frontend is not running"
    echo "   💡 Try: cd frontend && npx vite --port 3000"
fi

# Check Integration
echo ""
echo "🔗 Frontend-Backend Integration:"
echo "-------------------------------"

if curl -s http://localhost:8000/health > /dev/null && curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Both services are running"
    echo "   📡 CORS should be properly configured"
    echo "   🔄 API calls from frontend should work"

    # Test a few key integration points
    echo "   🧪 Testing key integration endpoints..."

    # Test latest signals endpoint
    if curl -s "http://localhost:8000/api/v1/signals/latest?limit=5" > /dev/null; then
        SIGNAL_COUNT=$(curl -s "http://localhost:8000/api/v1/signals/latest?limit=5" | jq length 2>/dev/null || echo "0")
        echo "   ✅ Latest signals: $SIGNAL_COUNT signals available"
    fi

else
    echo "❌ Integration issues detected"
    echo "   🔧 Make sure both services are running"
fi

# Final Summary
echo ""
echo "📋 Summary:"
echo "----------"

BACKEND_OK=$(curl -s http://localhost:8000/health > /dev/null && echo "✅" || echo "❌")
FRONTEND_OK=$(curl -s http://localhost:3000 > /dev/null && echo "✅" || echo "❌")

echo "Backend:  $BACKEND_OK http://localhost:8000"
echo "Frontend: $FRONTEND_OK http://localhost:3000"

if [ "$BACKEND_OK" = "✅" ] && [ "$FRONTEND_OK" = "✅" ]; then
    echo ""
    echo "🎉 All systems operational!"
    echo "🚀 Navigate to http://localhost:3000 to use the app"
else
    echo ""
    echo "⚠️  Some services need attention"
fi
