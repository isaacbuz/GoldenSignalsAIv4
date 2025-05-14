# presentation/tests/test_api.py
# Purpose: End-to-end tests for the FastAPI backend of GoldenSignalsAI, ensuring API
# endpoints work as expected for options trading workflows.

import httpx
import pytest
from fastapi.testclient import TestClient

from presentation.api.main import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_health_check():
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_login():
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/token", data={"username": "user1", "password": "password1"}
        )
        assert response.status_code == 200
        assert "access_token" in response.json()
