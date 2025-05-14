import pytest
from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_app_exists():
    assert app is not None, "Main application should be defined"

def test_app_configuration():
    assert hasattr(app, 'title'), "Application should have a title"
    assert app.title == "GoldenSignalsAI", "Application title should match project name"

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to GoldenSignalsAI"}
