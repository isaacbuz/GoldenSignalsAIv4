# GoldenSignalsAI V2 Test Quick Reference Guide

## Running All Tests
```bash
# Run complete test suite
python test_runner.py

# Alternative: Run all tests directly
python run_all_tests.py
```

## Running Specific Test Categories

### Data Quality Tests
```bash
python -m pytest tests/unit/test_data_quality.py -v
```
Tests: Data validation, outlier detection, normalization, feature engineering

### Signal Generation Tests
```bash
python -m pytest tests/unit/test_signal_generation.py -v
```
Tests: Signal quality, filtering, risk adjustment, execution readiness

### Backtesting Tests
```bash
python -m pytest tests/unit/test_backtesting_validation.py -v
```
Tests: Realistic execution, walk-forward optimization, stress testing

### Monitoring & Feedback Tests
```bash
python -m pytest tests/unit/test_monitoring_feedback.py -v
```
Tests: Real-time monitoring, anomaly detection, adaptive learning

### Model Optimization Tests
```bash
python -m pytest tests/unit/test_model_optimization.py -v
```
Tests: Hyperparameter tuning, feature selection, ensemble methods

### Risk Management Tests
```bash
python -m pytest tests/unit/test_domain_risk_management.py -v
```
Tests: Technical analysis, risk controls, market awareness

## Running All New Quality Tests
```bash
python -m pytest tests/unit/test_data_quality.py tests/unit/test_signal_generation.py tests/unit/test_backtesting_validation.py tests/unit/test_monitoring_feedback.py tests/unit/test_model_optimization.py tests/unit/test_domain_risk_management.py -v
```

## Test Coverage Report
```bash
# Run tests with coverage
python -m pytest tests/unit --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Quick Validation Commands

### Before Deploying
```bash
# Essential pre-deployment tests
python -m pytest tests/unit/test_signal_generation.py tests/unit/test_domain_risk_management.py -v
```

### After Data Source Changes
```bash
# Data quality validation
python -m pytest tests/unit/test_data_quality.py -v
```

### After Model Updates
```bash
# Model and signal validation
python -m pytest tests/unit/test_model_optimization.py tests/unit/test_signal_generation.py -v
```

### Performance Monitoring
```bash
# Check system performance
python -m pytest tests/performance -v
```

## Debugging Failed Tests

### Verbose Output
```bash
python -m pytest tests/unit/test_name.py -vv
```

### Show Local Variables
```bash
python -m pytest tests/unit/test_name.py --showlocals
```

### Run Specific Test
```bash
python -m pytest tests/unit/test_data_quality.py::TestDataQuality::test_missing_value_detection -v
```

### Skip Slow Tests
```bash
python -m pytest tests -m "not slow" -v
```

## Common Issues

### Import Errors
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Backend Not Running
```bash
# Start backend before running integration tests
python standalone_backend_optimized.py &
```

### Test Database
```bash
# Reset test database
python setup_local_db.sh
```

## Test Markers

- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.integration` - Requires external services
- `@pytest.mark.unit` - Fast, isolated tests
- `@pytest.mark.critical` - Must pass for deployment

## CI/CD Integration
```yaml
# Example GitHub Actions
- name: Run Quality Tests
  run: |
    python -m pytest tests/unit -v --junitxml=test-results.xml
``` 