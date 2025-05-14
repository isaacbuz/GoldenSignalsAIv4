# create_part4.py
# Purpose: Creates files in the .github/workflows/ and k8s/ directories for the GoldenSignalsAI project,
# including CI/CD pipeline and Kubernetes deployment configurations. Incorporates improvements
# for automated testing, building, and deployment, optimized for options trading application scalability.

import logging
from pathlib import Path

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


def create_part4():
    """Create files in .github/workflows/ and k8s/."""
    # Define the base directory as the current working directory
    base_dir = Path.cwd()

    logger.info({"message": f"Creating CI/CD and Kubernetes files in {base_dir}"})

    # Define CI/CD and Kubernetes files
    config_files = {
        ".github/workflows/ci.yml": """# .github/workflows/ci.yml
# Purpose: Defines a GitHub Actions workflow for continuous integration, running
# linting, testing, and building Docker images for the GoldenSignalsAI project.
# Ensures code quality and deployment readiness for options trading application.

name: CI Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -
      - name: Install dependencies
        run: |
          poetry install --no-root
      - name: Run Black
        run: |
          poetry run black --check .
      - name: Run Flake8
        run: |
          poetry run flake8 --max-line-length=88

  test:
    runs-on: ubuntu-latest
    needs: lint
    services:
      redis:
        image: redis:6.2
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -
      - name: Install dependencies
        run: |
          poetry install --no-root
      - name: Run tests
        run: |
          poetry run pytest --cov=./ --cov-report=xml
        env:
          ALPHA_VANTAGE_API_KEY: ${{ secrets.ALPHA_VANTAGE_API_KEY }}
          NEWS_API_KEY: ${{ secrets.NEWS_API_KEY }}
          TWITTER_BEARER_TOKEN: ${{ secrets.TWITTER_BEARER_TOKEN }}
          TWILIO_ACCOUNT_SID: ${{ secrets.TWILIO_ACCOUNT_SID }}
          TWILIO_AUTH_TOKEN: ${{ secrets.TWILIO_AUTH_TOKEN }}
          TWILIO_PHONE_NUMBER: ${{ secrets.TWILIO_PHONE_NUMBER }}
          WHATSAPP_PHONE_NUMBER: ${{ secrets.WHATSAPP_PHONE_NUMBER }}
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v3
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      - name: Login to DockerHub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      - name: Build and push FastAPI backend
        uses: docker/build-push-action@v4
        with:
          context: .
          file: ./Dockerfile
          push: true
          tags: user/goldensignalsai-backend:latest
      - name: Build and push React frontend
        uses: docker/build-push-action@v4
        with:
          context: .
          file: ./Dockerfile.frontend
          push: true
          tags: user/goldensignalsai-frontend:latest
""",
        "k8s/deployment.yaml": """# k8s/deployment.yaml
# Purpose: Defines Kubernetes deployment and service configurations for the
# GoldenSignalsAI application, including FastAPI backend, React frontend, Redis,
# and dashboard. Optimized for scalability and reliability in options trading
# workflows, with resource limits and health checks.

apiVersion: apps/v1
kind: Deployment
metadata:
  name: goldensignalsai-backend
  labels:
    app: goldensignalsai-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: goldensignalsai-backend
  template:
    metadata:
      labels:
        app: goldensignalsai-backend
    spec:
      containers:
      - name: backend
        image: user/goldensignalsai-backend:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            cpu: "2"
            memory: "2Gi"
          requests:
            cpu: "1"
            memory: "1Gi"
        env:
        - name: REDIS_HOST
          value: "redis-service"
        - name: REDIS_PORT
          value: "6379"
        - name: ALPHA_VANTAGE_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: alpha-vantage-api-key
        - name: NEWS_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: news-api-key
        - name: TWITTER_BEARER_TOKEN
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: twitter-bearer-token
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: backend-service
spec:
  selector:
    app: goldensignalsai-backend
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: ClusterIP

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: goldensignalsai-frontend
  labels:
    app: goldensignalsai-frontend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: goldensignalsai-frontend
  template:
    metadata:
      labels:
        app: goldensignalsai-frontend
    spec:
      containers:
      - name: frontend
        image: user/goldensignalsai-frontend:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            cpu: "1"
            memory: "1Gi"
          requests:
            cpu: "0.5"
            memory: "512Mi"
        livenessProbe:
          httpGet:
            path: /
            port: 8080
          initialDelaySeconds: 15
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: frontend-service
spec:
  selector:
    app: goldensignalsai-frontend
  ports:
  - protocol: TCP
    port: 8080
    targetPort: 8080
  type: LoadBalancer

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: goldensignalsai-dashboard
  labels:
    app: goldensignalsai-dashboard
spec:
  replicas: 1
  selector:
    matchLabels:
      app: goldensignalsai-dashboard
  template:
    metadata:
      labels:
        app: goldensignalsai-dashboard
    spec:
      containers:
      - name: dashboard
        image: user/goldensignalsai-backend:latest
        command: ["python", "-m", "monitoring.agent_dashboard"]
        ports:
        - containerPort: 8050
        resources:
          limits:
            cpu: "1"
            memory: "1Gi"
          requests:
            cpu: "0.5"
            memory: "512Mi"
        livenessProbe:
          httpGet:
            path: /
            port: 8050
          initialDelaySeconds: 15
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 8050
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: dashboard-service
spec:
  selector:
    app: goldensignalsai-dashboard
  ports:
  - protocol: TCP
    port: 8050
    targetPort: 8050
  type: LoadBalancer

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  labels:
    app: redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:6.2
        ports:
        - containerPort: 6379
        resources:
          limits:
            cpu: "1"
            memory: "512Mi"
          requests:
            cpu: "0.5"
            memory: "256Mi"
        livenessProbe:
          tcpSocket:
            port: 6379
          initialDelaySeconds: 15
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
spec:
  selector:
    app: redis
  ports:
  - protocol: TCP
    port: 6379
    targetPort: 6379
  type: ClusterIP
""",
    }

    # Write CI/CD and Kubernetes files
    for file_path, content in config_files.items():
        file_path = base_dir / file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info({"message": f"Created file: {file_path}"})

    print("Part 4: .github/workflows/ and k8s/ created successfully")


if __name__ == "__main__":
    create_part4()
