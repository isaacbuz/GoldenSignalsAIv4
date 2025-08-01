#!/bin/bash

# Smoke tests for GoldenSignalsAI deployment
# Usage: ./smoke-tests.sh <BASE_URL>

set -e

BASE_URL=${1:-"http://localhost:8000"}
TIMEOUT=10
MAX_RETRIES=3
VERBOSE=${VERBOSE:-false}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Test function with retry logic
test_endpoint() {
    local endpoint=$1
    local expected_status=$2
    local description=$3
    local retry_count=0

    log_info "Testing: $description"

    while [ $retry_count -lt $MAX_RETRIES ]; do
        response=$(curl -s -o /dev/null -w "%{http_code}" -m $TIMEOUT "$BASE_URL$endpoint" || echo "000")

        if [ "$response" = "$expected_status" ]; then
            log_info "✅ $description - Status: $response"
            return 0
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $MAX_RETRIES ]; then
                log_warning "Retry $retry_count/$MAX_RETRIES for $endpoint (got $response, expected $expected_status)"
                sleep 2
            fi
        fi
    done

    log_error "❌ $description - Expected: $expected_status, Got: $response"
    return 1
}

# Test WebSocket connection
test_websocket() {
    local ws_url=$(echo "$BASE_URL" | sed 's/http/ws/g')"/ws"
    log_info "Testing WebSocket connection: $ws_url"

    # Use Python to test WebSocket
    python3 -c "
import asyncio
import websockets
import json

async def test_ws():
    try:
        async with websockets.connect('$ws_url', timeout=5) as websocket:
            # Send subscription message
            await websocket.send(json.dumps({
                'type': 'subscribe',
                'channels': ['signals']
            }))

            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            print('WebSocket connection successful')
            return True
    except Exception as e:
        print(f'WebSocket connection failed: {e}')
        return False

asyncio.run(test_ws())
" && log_info "✅ WebSocket connection test passed" || log_error "❌ WebSocket connection test failed"
}

# Main test execution
main() {
    log_info "Starting smoke tests for $BASE_URL"
    log_info "========================================="

    # Track failures
    FAILED_TESTS=0

    # Core API endpoints
    test_endpoint "/" 200 "Health check endpoint" || ((FAILED_TESTS++))
    test_endpoint "/api/v1/signals" 200 "Signals endpoint" || ((FAILED_TESTS++))
    test_endpoint "/api/v1/market-data/SPY" 200 "Market data endpoint" || ((FAILED_TESTS++))
    test_endpoint "/api/v1/performance" 200 "Performance metrics endpoint" || ((FAILED_TESTS++))
    test_endpoint "/api/v1/market/opportunities" 200 "Market opportunities endpoint" || ((FAILED_TESTS++))

    # Signal-specific endpoints
    test_endpoint "/api/v1/signals/SPY/insights" 200 "Signal insights endpoint" || ((FAILED_TESTS++))
    test_endpoint "/api/v1/signals/quality-report" 200 "Signal quality report" || ((FAILED_TESTS++))

    # Monitoring endpoints
    test_endpoint "/api/v1/monitoring/performance" 200 "Monitoring performance" || ((FAILED_TESTS++))
    test_endpoint "/api/v1/monitoring/active-signals" 200 "Active signals" || ((FAILED_TESTS++))
    test_endpoint "/api/v1/monitoring/recommendations" 200 "Recommendations" || ((FAILED_TESTS++))

    # Pipeline endpoints
    test_endpoint "/api/v1/pipeline/stats" 200 "Pipeline statistics" || ((FAILED_TESTS++))

    # Test WebSocket if not in CI environment
    if [ -z "$CI" ]; then
        test_websocket || ((FAILED_TESTS++))
    fi

    # Error handling tests
    test_endpoint "/api/v1/market-data/INVALID123" 404 "Invalid symbol handling" || ((FAILED_TESTS++))
    test_endpoint "/api/v1/nonexistent" 404 "404 error handling" || ((FAILED_TESTS++))

    # Response time test
    log_info "Testing response times..."
    start_time=$(date +%s%N)
    curl -s "$BASE_URL/api/v1/signals" > /dev/null
    end_time=$(date +%s%N)
    response_time=$(( ($end_time - $start_time) / 1000000 ))

    if [ $response_time -lt 1000 ]; then
        log_info "✅ Response time test passed: ${response_time}ms"
    else
        log_warning "⚠️ Response time test slow: ${response_time}ms"
    fi

    # Summary
    log_info "========================================="
    if [ $FAILED_TESTS -eq 0 ]; then
        log_info "✅ All smoke tests passed!"
        exit 0
    else
        log_error "❌ $FAILED_TESTS tests failed!"
        exit 1
    fi
}

# Check dependencies
check_dependencies() {
    command -v curl >/dev/null 2>&1 || { log_error "curl is required but not installed."; exit 1; }
    command -v python3 >/dev/null 2>&1 || { log_error "python3 is required but not installed."; exit 1; }
}

# Run checks and tests
check_dependencies
main
