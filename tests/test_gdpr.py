import pytest
from fastapi.testclient import TestClient
from presentation.api.main import app

client = TestClient(app)

def test_gdpr_request_data():
    response = client.post("/gdpr/request-data")
    assert response.status_code == 200
    assert "message" in response.json()

def test_gdpr_delete_data():
    response = client.post("/gdpr/delete-data")
    assert response.status_code == 200
    assert "message" in response.json()
