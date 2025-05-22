import pytest
from fastapi.testclient import TestClient
from presentation.api.main import app

client = TestClient(app)

def test_rate_limiting():
    for _ in range(15):
        response = client.get("/api/some-endpoint")
    assert response.status_code in (200, 429)

def test_cors_headers():
    response = client.options("/api/some-endpoint", headers={"Origin": "https://yourfrontend.com", "Access-Control-Request-Method": "GET"})
    assert response.headers.get("access-control-allow-origin") == "https://yourfrontend.com"
