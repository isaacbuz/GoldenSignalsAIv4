#!/bin/bash

# Master Test Runner Script for GoldenSignalsAI V2
# This script runs all tests across the codebase with comprehensive logging

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create test logs directory
TEST_LOG_DIR="test_logs"
mkdir -p $TEST_LOG_DIR

# Generate timestamp for this test run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$TEST_LOG_DIR/test_run_$TIMESTAMP.log"
SUMMARY_FILE="$TEST_LOG_DIR/test_summary_$TIMESTAMP.txt"

# Initialize counters
TOTAL_PASSED=0
TOTAL_FAILED=0
TOTAL_MODULES=0
FAILED_MODULES=""

# Function to log messages
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Function to run a test command and capture results
run_test() {
    local module_name=$1
    local test_command=$2
    local working_dir=${3:-.}

    log "\n${BLUE}=================================================================================${NC}"
    log "${BLUE}Running $module_name${NC}"
    log "${BLUE}Command: $test_command${NC}"
    log "${BLUE}Directory: $working_dir${NC}"
    log "${BLUE}=================================================================================${NC}"

    # Change to working directory
    cd "$working_dir" || {
        log "${RED}Failed to change to directory: $working_dir${NC}"
        return 1
    }

    # Run the test command
    START_TIME=$(date +%s)
    eval "$test_command" >> "$LOG_FILE" 2>&1
    TEST_EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    # Return to original directory
    cd - > /dev/null 2>&1

    # Update counters
    TOTAL_MODULES=$((TOTAL_MODULES + 1))

    if [ $TEST_EXIT_CODE -eq 0 ]; then
        log "${GREEN}✓ $module_name PASSED (Duration: ${DURATION}s)${NC}"
        TOTAL_PASSED=$((TOTAL_PASSED + 1))
    else
        log "${RED}✗ $module_name FAILED (Duration: ${DURATION}s)${NC}"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
        FAILED_MODULES="$FAILED_MODULES\n  - $module_name"
    fi

    return $TEST_EXIT_CODE
}

# Function to check prerequisites
check_prerequisites() {
    log "${YELLOW}Checking prerequisites...${NC}"

    # Check Python
    if ! command -v python3 &> /dev/null; then
        log "${RED}Python 3 is not installed${NC}"
        return 1
    fi

    # Check Node.js
    if ! command -v node &> /dev/null; then
        log "${RED}Node.js is not installed${NC}"
        return 1
    fi

    # Check npm
    if ! command -v npm &> /dev/null; then
        log "${RED}npm is not installed${NC}"
        return 1
    fi

    # Check if virtual environment is activated
    if [ -z "$VIRTUAL_ENV" ]; then
        log "${YELLOW}Warning: Python virtual environment is not activated${NC}"
        log "${YELLOW}Attempting to activate .venv...${NC}"
        if [ -f ".venv/bin/activate" ]; then
            source .venv/bin/activate
            log "${GREEN}Virtual environment activated${NC}"
        else
            log "${RED}Virtual environment not found. Please create one first.${NC}"
            return 1
        fi
    fi

    log "${GREEN}All prerequisites met${NC}"
    return 0
}

# Main test execution
main() {
    log "${BLUE}========================================${NC}"
    log "${BLUE}GoldenSignalsAI V2 - Master Test Runner${NC}"
    log "${BLUE}Started at: $(date)${NC}"
    log "${BLUE}========================================${NC}"

    # Check prerequisites
    if ! check_prerequisites; then
        log "${RED}Prerequisites check failed. Exiting.${NC}"
        exit 1
    fi

    # Backend Tests
    log "\n${YELLOW}=== BACKEND TESTS ===${NC}"

    # Unit Tests
    run_test "Backend Unit Tests" "python -m pytest tests/unit -v --tb=short --no-header" "."

    # Integration Tests
    run_test "Backend Integration Tests" "python -m pytest tests/integration -v --tb=short --no-header" "."

    # Agent Tests
    run_test "Agent Tests" "python -m pytest tests/agents -v --tb=short --no-header" "."

    # Performance Tests (excluding slow tests)
    run_test "Performance Tests" "python -m pytest tests/performance -v --tb=short --no-header -m 'not slow'" "."

    # AlphaPy Tests
    run_test "AlphaPy Tests" "python -m pytest tests/AlphaPy -v --tb=short --no-header" "."

    # Root Tests
    run_test "Root Tests" "python -m pytest tests/root_tests -v --tb=short --no-header" "."

    # Comprehensive System Test
    run_test "Comprehensive System Test" "python tests/test_comprehensive_system.py" "."

    # Frontend Tests
    log "\n${YELLOW}=== FRONTEND TESTS ===${NC}"

    # Check if frontend directory exists
    if [ -d "frontend" ]; then
        # Install dependencies if needed
        if [ ! -d "frontend/node_modules" ]; then
            log "${YELLOW}Installing frontend dependencies...${NC}"
            (cd frontend && npm install) >> "$LOG_FILE" 2>&1
        fi

        # Run frontend tests
        run_test "Frontend Unit Tests" "npm test -- --run" "frontend"

        # Run E2E tests if Cypress is installed
        if [ -f "frontend/cypress.config.ts" ]; then
            run_test "Frontend E2E Tests" "npm run test:e2e:headless" "frontend"
        fi
    else
        log "${YELLOW}Frontend directory not found. Skipping frontend tests.${NC}"
    fi

    # ML Model Tests
    log "\n${YELLOW}=== ML MODEL TESTS ===${NC}"

    # ML Models Tests
    if [ -d "ml_models/tests" ]; then
        run_test "ML Model Tests" "python -m pytest ml_models/tests -v --tb=short --no-header" "."
    fi

    # ML Training Tests
    if [ -d "ml_training" ]; then
        run_test "ML Training Tests" "python -m pytest ml_training -v --tb=short --no-header -k test" "."
    fi

    # Infrastructure Tests
    log "\n${YELLOW}=== INFRASTRUCTURE TESTS ===${NC}"

    # Config Validation
    run_test "Config Validation" "python -c 'import yaml; yaml.safe_load(open(\"config.yaml\"))' && echo 'Config is valid'" "."

    # Database Connection Test
    if [ -f "check_databases.py" ]; then
        run_test "Database Connection Test" "python check_databases.py" "."
    fi

    # Generate Summary
    log "\n${BLUE}========================================${NC}"
    log "${BLUE}TEST EXECUTION SUMMARY${NC}"
    log "${BLUE}========================================${NC}"
    log "Total Test Modules: $TOTAL_MODULES"
    log "${GREEN}Passed: $TOTAL_PASSED${NC}"
    log "${RED}Failed: $TOTAL_FAILED${NC}"

    if [ $TOTAL_FAILED -gt 0 ]; then
        log "\n${RED}Failed Modules:${NC}"
        echo -e "$FAILED_MODULES" | tee -a "$LOG_FILE"
    fi

    # Calculate success rate
    if [ $TOTAL_MODULES -gt 0 ]; then
        SUCCESS_RATE=$(awk "BEGIN {printf \"%.2f\", ($TOTAL_PASSED/$TOTAL_MODULES)*100}")
        log "\nSuccess Rate: ${SUCCESS_RATE}%"
    fi

    log "\n${BLUE}Completed at: $(date)${NC}"
    log "${BLUE}Full log available at: $LOG_FILE${NC}"

    # Create summary file
    {
        echo "GoldenSignalsAI V2 Test Summary"
        echo "==============================="
        echo "Date: $(date)"
        echo "Total Modules: $TOTAL_MODULES"
        echo "Passed: $TOTAL_PASSED"
        echo "Failed: $TOTAL_FAILED"
        echo "Success Rate: ${SUCCESS_RATE}%"
        if [ $TOTAL_FAILED -gt 0 ]; then
            echo -e "\nFailed Modules:$FAILED_MODULES"
        fi
    } > "$SUMMARY_FILE"

    log "${BLUE}Summary saved to: $SUMMARY_FILE${NC}"

    # Exit with appropriate code
    if [ $TOTAL_FAILED -eq 0 ]; then
        log "\n${GREEN}All tests passed! 🎉${NC}"
        exit 0
    else
        log "\n${RED}Some tests failed. Please check the logs.${NC}"
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [options]"
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --backend      Run only backend tests"
        echo "  --frontend     Run only frontend tests"
        echo "  --ml           Run only ML tests"
        echo "  --quick        Run quick tests only (skip slow tests)"
        exit 0
        ;;
    --backend)
        # Run only backend tests
        log "${YELLOW}Running backend tests only${NC}"
        # Add backend-only logic here
        ;;
    --frontend)
        # Run only frontend tests
        log "${YELLOW}Running frontend tests only${NC}"
        # Add frontend-only logic here
        ;;
    --ml)
        # Run only ML tests
        log "${YELLOW}Running ML tests only${NC}"
        # Add ML-only logic here
        ;;
    --quick)
        # Run quick tests only
        export PYTEST_ADDOPTS="-m 'not slow'"
        log "${YELLOW}Running quick tests only (skipping slow tests)${NC}"
        ;;
esac

# Run main function
main
