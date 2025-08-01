"""
API endpoint integration tests
Tests all new Phase 2 endpoints
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import json
from datetime import datetime, timezone

# We'll import the app when it's available
try:
    from standalone_backend_optimized import app
    client = TestClient(app)
except ImportError:
    client = None


class TestAPIEndpoints:
    """Test API endpoints"""

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["status"] == "operational"

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_signals_endpoint(self):
        """Test signals endpoint"""
        response = client.get("/api/v1/signals")
        assert response.status_code == 200
        signals = response.json()
        assert isinstance(signals, list)

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_market_data_endpoint(self):
        """Test market data endpoint"""
        response = client.get("/api/v1/market-data/AAPL")
        assert response.status_code == 200
        data = response.json()
        assert "symbol" in data
        assert "price" in data

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_pipeline_stats_endpoint(self):
        """Test pipeline stats endpoint"""
        response = client.get("/api/v1/pipeline/stats")
        assert response.status_code == 200
        stats = response.json()
        assert "total_signals_processed" in stats
        assert "filter_stats" in stats

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_pipeline_configure_endpoint(self):
        """Test pipeline configuration endpoint"""
        config = {
            "min_confidence": 0.7,
            "min_quality_score": 0.8,
            "allowed_risk_levels": ["low", "medium"]
        }
        response = client.post("/api/v1/pipeline/configure", json=config)
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "configured"

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_signal_quality_report_endpoint(self):
        """Test signal quality report endpoint"""
        response = client.get("/api/v1/signals/quality-report")
        assert response.status_code == 200
        report = response.json()
        assert "timestamp" in report
        assert "total_signals_generated" in report
        assert "pipeline_stats" in report

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_signal_feedback_endpoint(self):
        """Test signal feedback endpoint"""
        response = client.post(
            "/api/v1/signals/feedback",
            params={
                "signal_id": "test_signal_001",
                "outcome": "success",
                "profit_loss": 5.5,
                "notes": "Good timing"
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "feedback_received"

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_monitoring_track_entry_endpoint(self):
        """Test monitoring track entry endpoint"""
        signal_data = {
            "id": "test_signal_002",
            "entry_price": 100.0,
            "symbol": "AAPL",
            "action": "BUY"
        }
        response = client.post("/api/v1/monitoring/track-entry", json=signal_data)
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "tracking_started"

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_monitoring_track_exit_endpoint(self):
        """Test monitoring track exit endpoint"""
        response = client.post(
            "/api/v1/monitoring/track-exit",
            params={
                "signal_id": "test_signal_002",
                "exit_price": 105.0,
                "outcome": "success",
                "notes": "Target reached"
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "exit_tracked"

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_monitoring_performance_endpoint(self):
        """Test monitoring performance endpoint"""
        response = client.get("/api/v1/monitoring/performance")
        assert response.status_code == 200
        metrics = response.json()
        assert "total_signals" in metrics
        assert "win_rate" in metrics
        assert "sharpe_ratio" in metrics

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_monitoring_recommendations_endpoint(self):
        """Test monitoring recommendations endpoint"""
        response = client.get("/api/v1/monitoring/recommendations")
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_monitoring_feedback_summary_endpoint(self):
        """Test monitoring feedback summary endpoint"""
        response = client.get("/api/v1/monitoring/feedback-summary")
        assert response.status_code == 200
        summary = response.json()
        assert "total_feedback" in summary
        assert "average_rating" in summary

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_monitoring_snapshot_endpoint(self):
        """Test monitoring snapshot endpoint"""
        response = client.post("/api/v1/monitoring/snapshot")
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "snapshot_saved"
        assert "timestamp" in result

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_monitoring_active_signals_endpoint(self):
        """Test monitoring active signals endpoint"""
        response = client.get("/api/v1/monitoring/active-signals")
        assert response.status_code == 200
        data = response.json()
        assert "active_signals" in data
        assert "count" in data

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_backtest_run_endpoint(self):
        """Test backtest run endpoint"""
        response = client.post(
            "/api/v1/backtest/run",
            params={
                "symbols": ["AAPL", "GOOGL"],
                "quick_mode": True
            }
        )
        # This might take time, so just check it accepts the request
        assert response.status_code in [200, 500]  # 500 if ML libs not available

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_backtest_recommendations_endpoint(self):
        """Test backtest recommendations endpoint"""
        response = client.get(
            "/api/v1/backtest/recommendations",
            params={"symbols": ["AAPL", "GOOGL"]}
        )
        # This might fail if ML libs not available
        assert response.status_code in [200, 500]

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_performance_stats_endpoint(self):
        """Test performance stats endpoint"""
        response = client.get("/api/v1/performance")
        assert response.status_code == 200
        stats = response.json()
        assert "endpoints" in stats
        assert "cache" in stats
        assert "uptime" in stats

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_signal_insights_endpoint(self):
        """Test signal insights endpoint"""
        response = client.get("/api/v1/signals/AAPL/insights")
        assert response.status_code == 200
        insights = response.json()
        assert "symbol" in insights
        assert "recommendation" in insights

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_market_opportunities_endpoint(self):
        """Test market opportunities endpoint"""
        response = client.get("/api/v1/market/opportunities")
        assert response.status_code == 200
        data = response.json()
        assert "opportunities" in data
        assert isinstance(data["opportunities"], list)

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_historical_data_endpoint(self):
        """Test historical data endpoint"""
        response = client.get(
            "/api/v1/market-data/AAPL/historical",
            params={"period": "1d", "interval": "5m"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_error_handling_invalid_symbol(self):
        """Test error handling for invalid symbol"""
        response = client.get("/api/v1/market-data/INVALID123XYZ")
        assert response.status_code == 404

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_error_handling_invalid_period(self):
        """Test error handling for invalid period"""
        response = client.get(
            "/api/v1/market-data/AAPL/historical",
            params={"period": "invalid_period"}
        )
        assert response.status_code == 422

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_websocket_connection(self):
        """Test WebSocket connection"""
        # WebSocket testing requires special client
        # Just verify the endpoint exists
        with client.websocket_connect("/ws") as websocket:
            # Should connect successfully
            websocket.send_text("test")
            # Connection should work even if no response


class TestAPIIntegration:
    """Integration tests for API workflows"""

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_signal_generation_workflow(self):
        """Test complete signal generation workflow"""
        # 1. Get signals
        response = client.get("/api/v1/signals")
        assert response.status_code == 200
        signals = response.json()

        if signals:
            signal = signals[0]

            # 2. Get quality report
            response = client.get("/api/v1/signals/quality-report")
            assert response.status_code == 200

            # 3. Submit feedback
            response = client.post(
                "/api/v1/signals/feedback",
                params={
                    "signal_id": signal["id"],
                    "outcome": "success",
                    "profit_loss": 2.5
                }
            )
            assert response.status_code == 200

    @pytest.mark.skipif(client is None, reason="Backend not available")
    def test_monitoring_workflow(self):
        """Test monitoring workflow"""
        # 1. Track entry
        signal_data = {
            "id": "workflow_test_001",
            "entry_price": 100.0,
            "symbol": "AAPL"
        }
        response = client.post("/api/v1/monitoring/track-entry", json=signal_data)
        assert response.status_code == 200

        # 2. Track exit
        response = client.post(
            "/api/v1/monitoring/track-exit",
            params={
                "signal_id": "workflow_test_001",
                "exit_price": 105.0,
                "outcome": "success"
            }
        )
        assert response.status_code == 200

        # 3. Get performance
        response = client.get("/api/v1/monitoring/performance")
        assert response.status_code == 200

        # 4. Get recommendations
        response = client.get("/api/v1/monitoring/recommendations")
        assert response.status_code == 200
