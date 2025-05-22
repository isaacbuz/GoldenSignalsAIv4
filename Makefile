install:
	npm install --prefix presentation/frontend
	pip install -r constraints.txt

dev:
	docker-compose up --build

test-frontend:
	npm run test --prefix presentation/frontend

test-backend:
	pytest
