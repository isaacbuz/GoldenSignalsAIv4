# create_part1.py
# Purpose: Creates root directory files for the GoldenSignalsAI project, including
# configuration files, Docker setups, and the main entry point for services.
# Incorporates improvements for options trading with comprehensive configuration.

import logging
from pathlib import Path

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


def create_part1():
    """Create root directory files."""
    # Define the base directory as the current working directory
    base_dir = Path.cwd()

    logger.info({"message": f"Creating root directory files in {base_dir}"})

    # Define root directory files
    root_files = {
        "run_services.py": """# run_services.py
# Purpose: Main entry point to start all services for GoldenSignalsAI, including
# the FastAPI backend, React frontend, and Dash dashboard for options trading.

import subprocess
import logging

# Configure logging with JSON-like format
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

def start_services():
    \"\"\"Start all services for GoldenSignalsAI.\"\"\"
    logger.info({"message": "Starting GoldenSignalsAI services"})
    try:
        # Start Redis server
        subprocess.Popen(["redis-server"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info({"message": "Redis server started"})

        # Start FastAPI backend
        subprocess.Popen(
            ["uvicorn", "presentation.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info({"message": "FastAPI backend started on port 8000"})

        # Start React frontend
        subprocess.Popen(
            ["npm", "start"],
            cwd="presentation/frontend",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True  # shell=True for npm command on Windows/Mac
        )
        logger.info({"message": "React frontend started"})

        # Start Dash dashboard
        subprocess.Popen(
            ["python", "-m", "monitoring.agent_dashboard"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info({"message": "Dash dashboard started on port 8050"})

    except Exception as e:
        logger.error({"message": f"Failed to start services: {str(e)}"})
        raise

if __name__ == "__main__":
    start_services()
""",
        "pyproject.toml": """[tool.poetry]
name = "goldensignalsai"
version = "0.1.0"
description = "An AI-powered trading signal generator with a focus on options trading."
authors = ["Your Name <your.email@example.com>"]

[tool.poetry.dependencies]
python = "^3.10"
pandas = "^2.0.0"
numpy = "^1.24.0"
yfinance = "^0.2.0"
requests = "^2.28.0"
vaderSentiment = "^3.3.2"
pydantic = "^2.0.0"
fastapi = "^0.95.0"
uvicorn = "^0.21.0"
dash = "^2.9.0"
plotly = "^5.14.0"
redis = "^4.5.0"
celery = "^5.2.0"
mlflow = "^2.2.0"
torch = "^2.0.0"
scikit-learn = "^1.2.0"
statsmodels = "^0.14.0"
twilio = "^8.0.0"
python-telegram-bot = "^20.0.0"
tweepy = "^4.14.0"
pytest = "^7.3.0"
pytest-cov = "^4.0.0"
httpx = "^0.24.0"
black = "^23.3.0"
flake8 = "^6.0.0"

[tool.poetry.dev-dependencies]

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
""",
        "poetry.lock": """# poetry.lock
# Placeholder for Poetry lock file. Run `poetry lock` to generate the actual lock file.
""",
        "config.yaml": """# config.yaml
# Purpose: Configuration file for GoldenSignalsAI, defining parameters for agents,
# notifications, data sources, and Redis connections, with options trading specifics.

drift_threshold: 0.5

model_weights:
  lstm: 0.3
  gru: 0.2
  transformer: 0.2
  cnn: 0.1
  xgboost: 0.1
  lightgbm: 0.05
  catboost: 0.05

redis:
  cluster_enabled: false
  cluster_nodes:
    - host: "localhost"
      port: 6379

agents:
  breakout:
    window: 20
    threshold: 0.05
  options_flow:
    iv_skew_threshold: 0.1
  reversion:
    mean_reversion_window: 20
  options_chain:
    volume_threshold: 1000
    oi_threshold: 5000
  news_sentiment:
    sentiment_threshold: 0.3
  social_media_sentiment:
    sentiment_threshold: 0.3
  options_risk:
    max_delta: 0.7
    max_gamma: 0.1
    max_theta: -0.05
  regime:
    regime_window: 30
  portfolio:
    risk_profile: "balanced"
  backtest_research:
    max_strategies: 10

notifications:
  default_channels: ["sms", "whatsapp", "telegram", "x"]
  alert_threshold: 0.8
  escalation_timeout: 300
""",
        "Dockerfile": """# Dockerfile
# Purpose: Docker configuration for the FastAPI backend of GoldenSignalsAI.

FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-root

COPY . .

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "presentation.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
""",
        "Dockerfile.frontend": """# Dockerfile.frontend
# Purpose: Docker configuration for the React frontend of GoldenSignalsAI.

FROM node:18

WORKDIR /app

COPY presentation/frontend/package.json presentation/frontend/package-lock.json ./
RUN npm install

COPY presentation/frontend .

EXPOSE 8080

CMD ["npm", "start"]
""",
        "docker-compose.yml": """version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      - redis

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8080:8080"

  redis:
    image: redis:6.2
    ports:
      - "6379:6379"

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8050:8050"
    command: python -m monitoring.agent_dashboard
    depends_on:
      - backend
""",
        "README.md": """# GoldenSignalsAI

An AI-powered trading signal generator with a focus on options trading, utilizing multi-agent systems, machine learning, and real-time data processing.

## Setup Instructions

1. **Install Poetry**:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install Dependencies**:
   ```bash
   poetry install
   ```

3. **Set Environment Variables**:
   Export the following environment variables (or add them to a `.env` file):
   ```bash
   export ALPHA_VANTAGE_API_KEY="your_alpha_vantage_api_key"
   export NEWS_API_KEY="your_news_api_key"
   export TWITTER_BEARER_TOKEN="your_twitter_bearer_token"
   export TWILIO_ACCOUNT_SID="your_twilio_account_sid"
   export TWILIO_AUTH_TOKEN="your_twilio_auth_token"
   export TWILIO_PHONE_NUMBER="your_twilio_phone_number"
   export WHATSAPP_PHONE_NUMBER="your_whatsapp_phone_number"
   export TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
   export SLACK_WEBHOOK_URL="your_slack_webhook_url"
   export SECRET_KEY="your_secret_key"
   ```

4. **Run Services**:
   Start all services using the main entry point:
   ```bash
   poetry run python run_services.py
   ```
   Alternatively, use Docker Compose:
   ```bash
   docker-compose up --build
   ```

## Services

- **FastAPI Backend**: Runs on `http://localhost:8000`
- **React Frontend**: Runs on `http://localhost:8080`
- **Dash Dashboard**: Runs on `http://localhost:8050`

## Running Tests

Run tests with coverage:
```bash
poetry run pytest --cov=./ --cov-report=xml
```

## Deployment

The project includes Kubernetes configurations in `k8s/` and a CI pipeline in `.github/workflows/ci.yml` for automated testing and deployment.
""",
        ".pre-commit-config.yaml": """repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/PyCQA/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88]
""",
    }

    # Write root directory files
    for file_path, content in root_files.items():
        file_path = base_dir / file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info({"message": f"Created file: {file_path}"})

    print("Part 1: Root directory files created successfully")


if __name__ == "__main__":
    create_part1()
