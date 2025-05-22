# Dockerfile
# Purpose: Docker configuration for the FastAPI backend of GoldenSignalsAI.

# Multi-stage Dockerfile for FastAPI backend (GoldenSignalsAI)
# Stage 1: Build dependencies
FROM python:3.10-slim as builder
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install --upgrade pip && pip install poetry && poetry config virtualenvs.create false && poetry install --no-root

# Stage 2: Production image
FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY . .
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["poetry", "run", "uvicorn", "presentation.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
