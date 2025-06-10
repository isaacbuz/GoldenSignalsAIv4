# --------------------------------------------------------------------------------
# Developer helpers
# --------------------------------------------------------------------------------

install:
	npm ci --prefix frontend
	poetry install --no-root --no-interaction --sync --with=dev

lint:
	poetry run black --check . && poetry run flake8 . && poetry run mypy src

test:
	poetry run pytest --cov=src --cov-report=term-missing

dev:
	docker-compose up --build

test-frontend:
	npm run test --prefix frontend

test-backend:
	pytest

setup:
	bash scripts/setup_local_env.sh
