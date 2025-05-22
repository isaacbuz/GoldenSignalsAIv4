# Dockerfiles and Containerization

This directory contains all Dockerfiles and container-related configuration for GoldenSignalsAI.

## Structure
- `backend.Dockerfile`: Backend (FastAPI) container
- `frontend.Dockerfile`: Frontend (React) container
- `analytics.Dockerfile`: Analytics/Dash container
- `docker-compose.yml`: Orchestrates multi-container setup

## Usage
Build and run the full stack:
```bash
docker-compose up --build
```

Build a specific image:
```bash
docker build -f backend.Dockerfile -t goldensignalsai-backend .
```

## Notes
- All secrets/configs should be provided via environment variables or Docker secrets.
- Do not commit actual secrets or API keys to this directory.
