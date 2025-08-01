"""
Tests for health check endpoints.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys

# Mock the problematic imports
sys.modules['src.domain.backtesting.backtest_metrics'] = MagicMock()
sys.modules['src.domain.backtesting.advanced_backtest_engine'] = MagicMock()

@pytest.fixture
def client():
    """Create test client with mocked app."""
    # Create a minimal FastAPI app for testing
    app = FastAPI()

    # Import health router after mocking
    from src.api.v1.health import router as health_router
    app.include_router(health_router)

    return TestClient(app)

class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_basic_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health/")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["service"] == "GoldenSignalsAI"
        assert data["version"] == "2.0.0"

    @patch('src.api.v1.health.get_db')
    @patch('src.api.v1.health.get_redis')
    def test_detailed_health_check_all_healthy(self, mock_redis, mock_db, client):
        """Test detailed health check when all components are healthy."""
        # Mock database
        mock_db_instance = Mock()
        mock_db_instance.execute.return_value = Mock()
        mock_db.return_value = mock_db_instance

        # Mock Redis
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance

        response = client.get("/health/detailed")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "components" in data
        assert "system" in data

        # Check component statuses
        assert data["components"]["database"]["status"] == "healthy"
        assert data["components"]["redis"]["status"] == "healthy"

        # Check system metrics
        assert "cpu_percent" in data["system"]
        assert "memory_percent" in data["system"]
        assert "disk_percent" in data["system"]

    @patch('src.api.v1.health.get_db')
    @patch('src.api.v1.health.get_redis')
    def test_detailed_health_check_database_unhealthy(self, mock_redis, mock_db, client):
        """Test detailed health check when database is unhealthy."""
        # Mock database failure
        mock_db_instance = Mock()
        mock_db_instance.execute.side_effect = Exception("Database connection failed")
        mock_db.return_value = mock_db_instance

        # Mock Redis healthy
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance

        response = client.get("/health/detailed")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "degraded"
        assert data["components"]["database"]["status"] == "unhealthy"
        assert "error" in data["components"]["database"]
        assert data["components"]["redis"]["status"] == "healthy"

    @patch('src.api.v1.health.AgentOrchestrator')
    def test_agents_health_check(self, mock_orchestrator, client):
        """Test agents health check endpoint."""
        # Mock orchestrator with agents
        mock_instance = Mock()
        mock_instance.agents = {
            "rsi_agent": Mock(),
            "macd_agent": Mock(),
            "sentiment_agent": Mock()
        }
        mock_orchestrator.return_value = mock_instance

        response = client.get("/health/agents")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "agents" in data
        assert len(data["agents"]) == 3

        for agent_name, agent_status in data["agents"].items():
            assert agent_status["status"] == "healthy"
            assert agent_status["initialized"] is True

    @patch('src.api.v1.health.get_db')
    @patch('src.api.v1.health.get_redis')
    def test_readiness_check_ready(self, mock_redis, mock_db, client):
        """Test readiness check when service is ready."""
        # Mock healthy database
        mock_db_instance = Mock()
        mock_db_instance.execute.return_value = Mock()
        mock_db.return_value = mock_db_instance

        # Mock healthy Redis
        mock_redis_instance = Mock()
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance

        response = client.get("/health/ready")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ready"
        assert "timestamp" in data

    @patch('src.api.v1.health.get_db')
    def test_readiness_check_not_ready(self, mock_db, client):
        """Test readiness check when service is not ready."""
        # Mock database failure
        mock_db_instance = Mock()
        mock_db_instance.execute.side_effect = Exception("Database not ready")
        mock_db.return_value = mock_db_instance

        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["detail"] == "Service not ready"

    def test_liveness_check(self, client):
        """Test liveness check endpoint."""
        response = client.get("/health/live")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "alive"
        assert "timestamp" in data
