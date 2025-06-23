# --------------------------------------------------------------------------------
# Developer helpers
# --------------------------------------------------------------------------------

install:
	npm ci --prefix frontend
	poetry install --no-root --no-interaction --sync --with=dev

lint:
	poetry run black --check . && poetry run flake8 . && poetry run mypy src

dev:
	docker-compose up --build

setup:
	bash scripts/setup_local_env.sh

# Testing Commands
# ================

.PHONY: test test-all test-backend test-frontend test-ml test-infrastructure test-quick test-coverage test-report

# Run all tests with comprehensive logging
test-all:
	@echo "Running all tests..."
	@python test_runner.py

# Run backend tests only
test-backend:
	@echo "Running backend tests..."
	@python test_runner.py --module backend

# Run frontend tests only
test-frontend:
	@echo "Running frontend tests..."
	@python test_runner.py --module frontend

# Run ML tests only
test-ml:
	@echo "Running ML tests..."
	@python test_runner.py --module ml

# Run infrastructure tests only
test-infrastructure:
	@echo "Running infrastructure tests..."
	@python test_runner.py --module infrastructure

# Run quick tests (exclude slow tests)
test-quick:
	@echo "Running quick tests..."
	@pytest -m "not slow" -v

# Run tests with coverage
test-coverage:
	@echo "Running tests with coverage..."
	@pytest --cov=src --cov=agents --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/"

# Generate test report from last run
test-report:
	@echo "Generating test report..."
	@python -c "import json; import glob; files = sorted(glob.glob('test_logs/test_summary_*.json')); \
	data = json.load(open(files[-1])) if files else {}; \
	print('\n=== LAST TEST RUN SUMMARY ==='); \
	print(f\"Success Rate: {data.get('summary', {}).get('success_rate', 0):.2f}%\"); \
	print(f\"Passed: {data.get('summary', {}).get('total_passed', 0)}\"); \
	print(f\"Failed: {data.get('summary', {}).get('total_failed', 0)}\"); \
	print(f\"Duration: {data.get('execution_time', {}).get('duration', 0):.2f}s\")"

# Clean test artifacts
test-clean:
	@echo "Cleaning test artifacts..."
	@rm -rf .pytest_cache
	@rm -rf htmlcov
	@rm -rf .coverage
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Test artifacts cleaned"

# Run specific test file
test-file:
	@echo "Usage: make test-file FILE=path/to/test_file.py"
	@pytest $(FILE) -v

# List available test suites
test-list:
	@python test_runner.py --list

# Default test command
test: test-quick

# Help for test commands
test-help:
	@echo "Available test commands:"
	@echo "  make test              - Run quick tests (default)"
	@echo "  make test-all          - Run all tests with logging"
	@echo "  make test-backend      - Run backend tests only"
	@echo "  make test-frontend     - Run frontend tests only"
	@echo "  make test-ml           - Run ML tests only"
	@echo "  make test-infrastructure - Run infrastructure tests"
	@echo "  make test-quick        - Run quick tests (exclude slow)"
	@echo "  make test-coverage     - Run tests with coverage report"
	@echo "  make test-report       - Show last test run summary"
	@echo "  make test-clean        - Clean test artifacts"
	@echo "  make test-file FILE=x  - Run specific test file"
	@echo "  make test-list         - List available test suites"


# Testing targets
test:
	pytest tests/unit -v

test-all:
	pytest -v

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term

test-watch:
	ptw -- -v

test-unit:
	pytest tests/unit -v -m "not slow"

test-integration:
	pytest tests/integration -v

test-e2e:
	pytest tests/e2e -v

test-parallel:
	pytest -n auto -v

test-benchmark:
	pytest --benchmark-only

clean-test:
	rm -rf .pytest_cache htmlcov .coverage coverage.xml


# Testing targets
test:
	pytest tests/unit -v

test-all:
	pytest -v

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term

test-watch:
	ptw -- -v

test-unit:
	pytest tests/unit -v -m "not slow"

test-integration:
	pytest tests/integration -v

test-e2e:
	pytest tests/e2e -v

test-parallel:
	pytest -n auto -v

test-benchmark:
	pytest --benchmark-only

clean-test:
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
