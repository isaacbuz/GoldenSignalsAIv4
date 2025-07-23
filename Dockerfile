# Dockerfile
# Purpose: Docker configuration for the FastAPI backend of GoldenSignalsAI.

# Multi-stage Dockerfile for FastAPI backend (GoldenSignalsAI)
# Stage 1: Build dependencies
FROM python:3.13-slim as builder

# Install build dependencies and TA-Lib
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget build-essential && \
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    wget -O config.guess 'http://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.guess;hb=HEAD' && \
    wget -O config.sub 'http://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.sub;hb=HEAD' && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.4.0-src.tar.gz ta-lib/

WORKDIR /app
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies (keeping build tools for compilation)
RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root

# Clean up build dependencies after installation
RUN apt-get remove -y build-essential wget && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Stage 2: Production image
FROM python:3.13-slim

# Copy TA-Lib from the builder stage
COPY --from=builder /usr/lib/libta_lib.so.0 /usr/lib/libta_lib.so.0
COPY --from=builder /usr/lib/libta_lib.a /usr/lib/libta_lib.a

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["poetry", "run", "uvicorn", "presentation.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
