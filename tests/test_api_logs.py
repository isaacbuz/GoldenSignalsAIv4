from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_logs_endpoint():
    response = client.get("/api/logs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)