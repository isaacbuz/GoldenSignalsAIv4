import pytest
from src.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_full_system_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_signal_pipeline_integration():
    response = client.get("/api/v1/signals")
    assert response.status_code == 200
