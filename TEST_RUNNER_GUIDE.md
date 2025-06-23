# GoldenSignalsAI V2 - Comprehensive Test Runner Guide

## Overview

The GoldenSignalsAI V2 project includes a comprehensive test runner that orchestrates all tests across the entire codebase with detailed logging and reporting capabilities.

## Quick Start

### Running All Tests

```bash
# Using the Python test runner
python test_runner.py

# Using Make
make test-all
```

### Running Specific Module Tests

```bash
# Backend tests only
python test_runner.py --module backend
# or
make test-backend

# Frontend tests only
python test_runner.py --module frontend
# or
make test-frontend

# ML tests only
python test_runner.py --module ml
# or
make test-ml

# Infrastructure tests only
python test_runner.py --module infrastructure
# or
make test-infrastructure
```

## Test Organization

### Backend Tests
- **Unit Tests**: `tests/unit/` - Test individual components in isolation
- **Integration Tests**: `tests/integration/` - Test component interactions
- **Agent Tests**: `tests/agents/` - Test AI agent functionality
- **Performance Tests**: `tests/performance/` - Test system performance
- **System Tests**: `tests/test_comprehensive_system.py` - End-to-end system tests

### Frontend Tests
- **Unit Tests**: Component and hook tests using Vitest
- **Integration Tests**: Tests with all providers and state management
- **E2E Tests**: Full user flow tests with Cypress

### ML Tests
- **Model Tests**: `ml_models/tests/` - Test ML model functionality
- **Training Tests**: `ml_training/` - Test training pipelines

### Infrastructure Tests
- **Config Validation**: Validates YAML configuration files
- **Database Tests**: Tests database connections and operations

## Features

### 1. Comprehensive Logging
- All test output is logged to `test_logs/test_run_YYYYMMDD_HHMMSS.log`
- Color-coded console output for easy reading
- Detailed error reporting for failed tests

### 2. Test Statistics
- Tracks passed, failed, skipped, and error counts
- Calculates success rates
- Measures execution time for each test suite

### 3. JSON Summary Reports
- Generates `test_logs/test_summary_YYYYMMDD_HHMMSS.json`
- Contains detailed results for each test suite
- Can be used for CI/CD integration

### 4. Prerequisite Checking
- Verifies Python installation
- Checks Node.js availability
- Validates virtual environment
- Confirms dependencies are installed

## Make Commands

```bash
# View all test commands
make test-help

# Run quick tests (exclude slow tests)
make test-quick

# Run tests with coverage report
make test-coverage

# Show last test run summary
make test-report

# Clean test artifacts
make test-clean

# Run specific test file
make test-file FILE=tests/unit/test_example.py

# List available test suites
make test-list
```

## Test Runner Options

### Command Line Arguments

```bash
# List all available test suites
python test_runner.py --list

# Run multiple specific modules
python test_runner.py --module backend frontend

# Run with custom Python interpreter
/path/to/python test_runner.py
```

### Environment Variables

```bash
# Skip slow tests
export PYTEST_ADDOPTS="-m 'not slow'"

# Increase test verbosity
export PYTEST_ADDOPTS="-vv"
```

## Output Examples

### Successful Test Run
```
2024-01-20 10:30:45 - INFO - Checking prerequisites...
2024-01-20 10:30:45 - SUCCESS - ✓ Python 3
2024-01-20 10:30:45 - SUCCESS - ✓ Node.js
2024-01-20 10:30:45 - SUCCESS - ✓ Virtual Environment
2024-01-20 10:30:45 - SUCCESS - ✓ Python Dependencies
2024-01-20 10:30:45 - SUCCESS - ✓ Frontend Dependencies

================================================================================
Running Backend - Unit Tests
Command: python -m pytest tests/unit -v --tb=short
================================================================================
2024-01-20 10:30:50 - SUCCESS - ✓ Unit Tests PASSED (5.23s)
```

### Test Summary
```
================================================================================
TEST EXECUTION SUMMARY
================================================================================

Backend Tests:
  ✓ Unit Tests: 45 passed, 0 failed, 2 skipped (5.23s)
  ✓ Integration Tests: 23 passed, 0 failed, 0 skipped (12.45s)
  ✗ Agent Tests: 18 passed, 2 failed, 1 skipped (8.67s)

Frontend Tests:
  ✓ Unit & Integration Tests: 67 passed, 0 failed, 3 skipped (15.34s)
  ✓ Component Tests: 34 passed, 0 failed, 0 skipped (7.89s)

--------------------------------------------------------------------------------
Overall Statistics:
  Total Tests Passed: 187
  Total Tests Failed: 2
  Total Tests Skipped: 6
  Total Errors: 0
  Total Duration: 49.58s
  Success Rate: 98.94%
```

## Troubleshooting

### Common Issues

1. **"Python virtual environment is not activated"**
   ```bash
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate  # On Windows
   ```

2. **"Frontend dependencies not found"**
   ```bash
   cd frontend && npm install
   ```

3. **"Module 'pytest' not found"**
   ```bash
   pip install -r requirements-test.txt
   ```

4. **Frontend tests fail with "vitest not found"**
   ```bash
   cd frontend && npm install --save-dev vitest
   ```

### Debug Mode

To see more detailed output during test execution:

```bash
# Set debug logging
export LOG_LEVEL=DEBUG
python test_runner.py

# Or use pytest verbose mode
pytest -vv tests/
```

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Run All Tests
  run: |
    python test_runner.py
    
- name: Upload Test Results
  if: always()
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: test_logs/
```

### GitLab CI Example

```yaml
test:
  script:
    - python test_runner.py
  artifacts:
    when: always
    paths:
      - test_logs/
    reports:
      junit: test_logs/test_summary_*.json
```

## Best Practices

1. **Run tests before committing**
   ```bash
   make test-quick  # Quick validation
   ```

2. **Run full test suite before merging**
   ```bash
   make test-all  # Comprehensive testing
   ```

3. **Check coverage regularly**
   ```bash
   make test-coverage
   ```

4. **Clean test artifacts periodically**
   ```bash
   make test-clean
   ```

5. **Review test reports for trends**
   ```bash
   make test-report
   ```

## Adding New Tests

### Backend Test
1. Create test file in appropriate directory (`tests/unit/`, `tests/integration/`, etc.)
2. Follow pytest conventions (files named `test_*.py` or `*_test.py`)
3. Tests will be automatically discovered

### Frontend Test
1. Create test file with `.test.ts` or `.test.tsx` extension
2. Place in `__tests__` directory or next to component
3. Use Vitest or React Testing Library

### E2E Test
1. Create test file in `frontend/cypress/e2e/`
2. Use `.cy.ts` extension
3. Follow Cypress best practices

## Performance Considerations

- The test runner uses a 5-minute timeout for each test suite
- Slow tests should be marked with `@pytest.mark.slow`
- Use `make test-quick` for rapid feedback during development
- Full test suite may take 5-10 minutes depending on system

## Maintenance

### Updating Test Runner
The test runner is located at `test_runner.py`. To add new test suites:

1. Add new TestResult to appropriate method
2. Update command list
3. Add parsing logic if needed

### Log Rotation
Test logs accumulate in `test_logs/`. Consider periodic cleanup:

```bash
# Keep only last 10 test runs
ls -t test_logs/test_run_*.log | tail -n +11 | xargs rm -f
``` 