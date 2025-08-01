"""
Integration tests for API endpoints
"""

import pytest
from httpx import AsyncClient
from datetime import datetime

@pytest.mark.integration
class TestSignalAPI:
    """Test signal API endpoints"""

    @pytest.mark.asyncio
    async def test_generate_signal_endpoint(self, async_client: AsyncClient):
        """Test POST /api/v1/signals/generate"""
        response = await async_client.post(
            "/api/v1/signals/generate",
            json={"symbol": "AAPL"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "signal" in data
        assert data["signal"]["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_get_signals_endpoint(self, async_client: AsyncClient):
        """Test GET /api/v1/signals"""
        response = await async_client.get(
            "/api/v1/signals",
            params={"symbol": "AAPL", "limit": 10}
        )

        assert response.status_code == 200
        data = response.json()
        assert "signals" in data
        assert isinstance(data["signals"], list)

    @pytest.mark.asyncio
    async def test_health_check(self, async_client: AsyncClient):
        """Test GET /health"""
        response = await async_client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
