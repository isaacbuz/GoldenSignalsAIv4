from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_train_endpoint():
    response = client.post("/api/train")
    assert response.status_code == 200
    assert "status" in response.json()
