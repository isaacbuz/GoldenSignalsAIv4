import pytest
from fastapi.testclient import TestClient
from presentation.api.main import app

client = TestClient(app)

def test_openapi_contract():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "paths" in response.json()
    assert "components" in response.json()

def test_api_security_headers():
    response = client.get("/api/some-protected-endpoint")
    assert "x-frame-options" in response.headers or "X-Frame-Options" in response.headers
    assert "x-content-type-options" in response.headers or "X-Content-Type-Options" in response.headers
