from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_analyze_endpoint():
    response = client.get("/api/analyze?ticker=AAPL")
    assert response.status_code == 200
    data = response.json()
    assert "signal" in data
    assert "confidence" in data