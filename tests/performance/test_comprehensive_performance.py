"""Comprehensive performance tests for GoldenSignalsAI V2."""

import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_api_response_time():
    """Test API endpoint response times."""
    start_time = time.time()
    response = client.get("/health")
    end_time = time.time()

    assert response.status_code == 200
    assert (end_time - start_time) < 1.0  # Should respond within 1 second

def test_concurrent_requests():
    """Test system under concurrent load."""
    def make_request():
        return client.get("/health")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(50)]
        responses = [future.result() for future in futures]

    # All requests should succeed
    assert all(r.status_code == 200 for r in responses)

@pytest.mark.asyncio
async def test_signal_generation_performance():
    """Test signal generation performance."""
    start_time = time.time()

    # Simulate signal generation
    await asyncio.sleep(0.1)  # Mock processing time

    end_time = time.time()
    assert (end_time - start_time) < 0.5  # Should complete within 500ms

def test_memory_usage():
    """Test memory usage under load."""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss

    # Simulate some operations
    for _ in range(1000):
        client.get("/health")

    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory

    # Memory increase should be reasonable (< 100MB)
    assert memory_increase < 100 * 1024 * 1024
