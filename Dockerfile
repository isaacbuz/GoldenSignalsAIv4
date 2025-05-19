<<<<<<< HEAD
# Base Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-dev
COPY . .
CMD ["bash"]
=======
# Dockerfile
# Purpose: Docker configuration for the FastAPI backend of GoldenSignalsAI.

FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-root

COPY . .

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "presentation.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
>>>>>>> b3d312fc9c631d3b59f644472ad576448be06c0b
