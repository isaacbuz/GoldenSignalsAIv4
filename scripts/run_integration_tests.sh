#!/bin/bash

# Run Integration Tests for RAG-Agent-MCP System
# Issue #195: Integration-1: RAG-Agent-MCP Integration Testing

set -e

echo "🧪 RAG-Agent-MCP Integration Test Runner"
echo "======================================="

# Check if running in Docker or locally
if [ "$1" == "--docker" ]; then
    echo "🐳 Running tests in Docker..."
    
    # Build images
    echo "📦 Building Docker images..."
    docker-compose -f docker-compose.integration-test.yml build
    
    # Start services
    echo "🚀 Starting MCP servers..."
    docker-compose -f docker-compose.integration-test.yml up -d \
        market-data-mcp \
        rag-query-mcp \
        agent-comm-mcp \
        risk-analytics-mcp \
        execution-mcp
    
    # Wait for services to be ready
    echo "⏳ Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    echo "🏥 Checking service health..."
    for port in 8190 8191 8192 8193 8194; do
        if curl -f http://localhost:$port/ > /dev/null 2>&1; then
            echo "  ✅ Service on port $port is healthy"
        else
            echo "  ❌ Service on port $port is not responding"
        fi
    done
    
    # Run tests
    echo "🧪 Running integration tests..."
    docker-compose -f docker-compose.integration-test.yml run --rm integration-tests
    
    # Copy test results
    echo "📄 Copying test results..."
    docker cp $(docker-compose -f docker-compose.integration-test.yml ps -q integration-tests):/app/test_results ./test_results
    
    # Stop services
    echo "🛑 Stopping services..."
    docker-compose -f docker-compose.integration-test.yml down
    
else
    echo "💻 Running tests locally..."
    
    # Check Python environment
    if [ ! -d ".venv" ]; then
        echo "❌ Virtual environment not found. Please create one first."
        exit 1
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Install test dependencies
    echo "📦 Installing test dependencies..."
    pip install -q pytest pytest-asyncio pytest-cov pytest-timeout aiohttp
    
    # Run unit tests first
    echo "🧪 Running unit tests..."
    pytest tests/unit/ -v --tb=short || true
    
    # Run integration tests
    echo "🧪 Running integration tests..."
    pytest tests/integration/test_rag_agent_mcp_integration.py -v --tb=short
    
    # Generate coverage report
    if [ "$1" == "--coverage" ]; then
        echo "📊 Generating coverage report..."
        pytest tests/ --cov=agents --cov=mcp_servers --cov-report=html --cov-report=term
        echo "📄 Coverage report saved to htmlcov/index.html"
    fi
fi

echo ""
echo "✅ Integration tests completed!"
echo ""

# Show test summary
if [ -f "test_results/integration-test-results.xml" ]; then
    echo "📊 Test Results Summary:"
    python -c "
import xml.etree.ElementTree as ET
tree = ET.parse('test_results/integration-test-results.xml')
root = tree.getroot()
testsuite = root.find('testsuite') or root
total = testsuite.get('tests', '0')
failures = testsuite.get('failures', '0')
errors = testsuite.get('errors', '0')
time = testsuite.get('time', '0')
passed = int(total) - int(failures) - int(errors)
print(f'  Total Tests: {total}')
print(f'  Passed: {passed} ✅')
print(f'  Failed: {failures} ❌')
print(f'  Errors: {errors} ⚠️')
print(f'  Time: {float(time):.2f}s')
"
fi 