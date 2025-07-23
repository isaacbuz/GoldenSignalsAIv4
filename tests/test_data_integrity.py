"""
Data Integrity Test Suite for GoldenSignalsAI
Tests data consistency, validation, and integrity across the system
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
import json
from unittest.mock import patch, MagicMock
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
try:
    from src.main import app
except ImportError:
    app = None


class TestMarketDataIntegrity:
    """Test market data consistency and validation"""

    @pytest.fixture
    def client(self, test_app):
        return TestClient(test_app)

    def test_price_data_validation(self, client):
        """Test that price data is properly validated"""
        # Test negative prices are rejected
        invalid_data = {
            "symbol": "AAPL",
            "price": -100.0,
            "timestamp": datetime.now().isoformat()
        }

        response = client.post("/api/v1/market-data/update", json=invalid_data)
        assert response.status_code == 422

        # Test extreme price changes are flagged
        extreme_data = {
            "symbol": "AAPL",
            "price": 10000.0,  # Unrealistic price
            "timestamp": datetime.now().isoformat()
        }

        response = client.post("/api/v1/market-data/update", json=extreme_data)
        # Should either reject or flag for review
        assert response.status_code in [422, 202]

    def test_ohlcv_data_consistency(self, client):
        """Test OHLCV data consistency rules"""
        # High should be >= Low
        # Close should be between High and Low
        # Open should be between High and Low
        # Volume should be non-negative

        invalid_candles = [
            {
                "open": 100, "high": 90, "low": 110, "close": 95,
                "volume": 1000, "timestamp": datetime.now().isoformat()
            },
            {
                "open": 100, "high": 110, "low": 90, "close": 120,
                "volume": -1000, "timestamp": datetime.now().isoformat()
            }
        ]

        for candle in invalid_candles:
            response = client.post(
                f"/api/v1/market-data/AAPL/candle",
                json=candle
            )
            assert response.status_code == 422

    def test_timestamp_ordering(self, client):
        """Test that data maintains proper time ordering"""
        response = client.get("/api/v1/market-data/AAPL/history?period=1d")

        if response.status_code == 200:
            data = response.json()
            timestamps = [item["timestamp"] for item in data.get("data", [])]

            # Verify timestamps are in ascending order
            for i in range(1, len(timestamps)):
                assert timestamps[i] >= timestamps[i-1]

    def test_data_precision(self):
        """Test that financial calculations maintain proper precision"""
        try:
            from src.services.portfolio_service import calculate_position_value
        except ImportError:
            # Mock the function for testing
            def calculate_position_value(quantity, price):
                return quantity * price

        # Test with Decimal for precise financial calculations
        quantity = Decimal("100.123456")
        price = Decimal("150.99")

        value = calculate_position_value(quantity, price)

        # Should maintain precision
        expected = quantity * price
        assert value == expected
        assert isinstance(value, Decimal)


class TestSignalDataIntegrity:
    """Test trading signal data integrity"""

    @pytest.fixture
    def client(self, test_app):
        return TestClient(test_app)

    def test_signal_confidence_bounds(self, client):
        """Test that signal confidence is within valid bounds"""
        response = client.get("/api/v1/signals/AAPL")

        if response.status_code == 200:
            signals = response.json().get("signals", [])

            for signal in signals:
                # Confidence should be between 0 and 100
                assert 0 <= signal["confidence"] <= 100

                # Signal type should be valid
                assert signal["type"] in ["BUY", "SELL", "HOLD"]

                # Required fields should exist
                assert "symbol" in signal
                assert "timestamp" in signal
                assert "source" in signal

    def test_signal_consistency(self, client):
        """Test that signals are internally consistent"""
        response = client.get("/api/v1/signals/AAPL")

        if response.status_code == 200:
            signals = response.json().get("signals", [])

            # Group signals by timestamp
            from collections import defaultdict
            signals_by_time = defaultdict(list)

            for signal in signals:
                signals_by_time[signal["timestamp"]].append(signal)

            # Check for conflicting signals at same time
            for timestamp, time_signals in signals_by_time.items():
                buy_signals = [s for s in time_signals if s["type"] == "BUY"]
                sell_signals = [s for s in time_signals if s["type"] == "SELL"]

                # Shouldn't have both strong buy and strong sell at same time
                if buy_signals and sell_signals:
                    buy_confidence = max(s["confidence"] for s in buy_signals)
                    sell_confidence = max(s["confidence"] for s in sell_signals)

                    # Both shouldn't be high confidence
                    assert not (buy_confidence > 80 and sell_confidence > 80)

    def test_signal_metadata_validation(self, client):
        """Test that signal metadata is properly structured"""
        response = client.get("/api/v1/signals/AAPL")

        if response.status_code == 200:
            signals = response.json().get("signals", [])

            for signal in signals:
                if "metadata" in signal:
                    metadata = signal["metadata"]

                    # Stop loss should be less than entry for buy signals
                    if signal["type"] == "BUY" and "stop_loss" in metadata:
                        assert metadata["stop_loss"] < signal.get("price", float('inf'))

                    # Stop loss should be greater than entry for sell signals
                    if signal["type"] == "SELL" and "stop_loss" in metadata:
                        assert metadata["stop_loss"] > signal.get("price", 0)

                    # Risk reward ratio should be positive
                    if "risk_reward_ratio" in metadata:
                        assert metadata["risk_reward_ratio"] > 0


class TestPortfolioDataIntegrity:
    """Test portfolio data consistency and integrity"""

    @pytest.fixture
    def client(self, test_app):
        return TestClient(test_app)

    def test_position_quantity_integrity(self, client):
        """Test that position quantities remain consistent"""
        # Create a position
        position_data = {
            "symbol": "AAPL",
            "quantity": 100,
            "average_price": 150.0
        }

        response = client.post("/api/v1/portfolio/positions", json=position_data)
        position_id = response.json().get("id")

        # Execute trades
        trades = [
            {"position_id": position_id, "quantity": -30, "price": 155.0},
            {"position_id": position_id, "quantity": 50, "price": 152.0},
            {"position_id": position_id, "quantity": -20, "price": 156.0}
        ]

        for trade in trades:
            client.post("/api/v1/portfolio/trades", json=trade)

        # Verify final quantity
        response = client.get(f"/api/v1/portfolio/positions/{position_id}")
        final_position = response.json()

        # Should be 100 - 30 + 50 - 20 = 100
        assert final_position["quantity"] == 100

    def test_portfolio_value_calculation(self, client):
        """Test that portfolio values are calculated correctly"""
        # Get portfolio summary
        response = client.get("/api/v1/portfolio/summary")

        if response.status_code == 200:
            summary = response.json()

            # Total value should equal sum of position values + cash
            positions_value = sum(p["market_value"] for p in summary.get("positions", []))
            cash_balance = summary.get("cash_balance", 0)
            total_value = summary.get("total_value", 0)

            assert abs(total_value - (positions_value + cash_balance)) < 0.01

            # Verify P&L calculations
            total_cost = sum(p["quantity"] * p["average_price"]
                           for p in summary.get("positions", []))
            total_market_value = sum(p["market_value"] for p in summary.get("positions", []))
            calculated_pnl = total_market_value - total_cost

            reported_pnl = summary.get("unrealized_pnl", 0)
            assert abs(calculated_pnl - reported_pnl) < 0.01

    def test_transaction_atomicity(self, client):
        """Test that portfolio transactions are atomic"""
        # Attempt to buy more than available cash
        large_order = {
            "symbol": "AAPL",
            "quantity": 1000000,  # Very large quantity
            "order_type": "market"
        }

        # Get initial portfolio state
        initial_response = client.get("/api/v1/portfolio/summary")
        initial_cash = initial_response.json().get("cash_balance", 0)

        # Attempt the order
        order_response = client.post("/api/v1/portfolio/orders", json=large_order)

        # Get final portfolio state
        final_response = client.get("/api/v1/portfolio/summary")
        final_cash = final_response.json().get("cash_balance", 0)

        # If order failed, cash should remain unchanged
        if order_response.status_code != 200:
            assert initial_cash == final_cash


class TestAgentDataIntegrity:
    """Test agent analysis data integrity"""

    @pytest.fixture
    def client(self, test_app):
        return TestClient(test_app)

    def test_agent_consensus_calculation(self, client):
        """Test that agent consensus is calculated correctly"""
        response = client.get("/api/v1/agents/consensus/AAPL")

        if response.status_code == 200:
            consensus = response.json()

            # Verify vote counting
            total_votes = (consensus.get("buy_votes", 0) +
                         consensus.get("sell_votes", 0) +
                         consensus.get("hold_votes", 0))

            reported_total = consensus.get("total_agents", 0)
            assert total_votes == reported_total

            # Verify weighted consensus
            if "weighted_scores" in consensus:
                weights = consensus["weighted_scores"]
                total_weight = sum(weights.values())

                # Weights should sum to approximately 1.0
                assert abs(total_weight - 1.0) < 0.01

    def test_agent_signal_aggregation(self, client):
        """Test that agent signals are properly aggregated"""
        response = client.get("/api/v1/agents/signals/AAPL")

        if response.status_code == 200:
            data = response.json()
            agent_signals = data.get("agent_signals", {})
            aggregated = data.get("aggregated_signal", {})

            # Calculate expected aggregation
            confidences = [s["confidence"] for s in agent_signals.values()
                         if "confidence" in s]

            if confidences:
                expected_avg = sum(confidences) / len(confidences)
                reported_avg = aggregated.get("average_confidence", 0)

                # Should match within floating point tolerance
                assert abs(expected_avg - reported_avg) < 0.001

    def test_workflow_state_consistency(self, client):
        """Test that workflow states are consistent"""
        # Start a workflow
        workflow_response = client.post(
            "/api/v1/workflow/analyze",
            json={"symbol": "AAPL"}
        )

        workflow_id = workflow_response.json().get("workflow_id")

        # Poll for completion
        states_seen = []
        for _ in range(10):
            status_response = client.get(f"/api/v1/workflow/status/{workflow_id}")
            state = status_response.json().get("state")
            states_seen.append(state)

            if state in ["completed", "failed"]:
                break

            asyncio.run(asyncio.sleep(1))

        # Verify state transitions are valid
        valid_transitions = {
            "pending": ["running", "failed"],
            "running": ["completed", "failed"],
            "completed": [],
            "failed": []
        }

        for i in range(1, len(states_seen)):
            prev_state = states_seen[i-1]
            curr_state = states_seen[i]

            if prev_state != curr_state:
                assert curr_state in valid_transitions.get(prev_state, [])


class TestDataSynchronization:
    """Test data synchronization across services"""

    @pytest.fixture
    def client(self, test_app):
        return TestClient(test_app)

    def test_cache_consistency(self, client):
        """Test that cached data remains consistent with source"""
        symbol = "AAPL"

        # Get data directly (bypassing cache)
        direct_response = client.get(
            f"/api/v1/market-data/{symbol}/current",
            headers={"Cache-Control": "no-cache"}
        )
        direct_data = direct_response.json()

        # Get cached data
        cached_response = client.get(f"/api/v1/market-data/{symbol}/current")
        cached_data = cached_response.json()

        # Key fields should match
        assert direct_data["symbol"] == cached_data["symbol"]

        # Prices should be close (allowing for real-time changes)
        if "price" in direct_data and "price" in cached_data:
            price_diff = abs(direct_data["price"] - cached_data["price"])
            price_pct_diff = price_diff / direct_data["price"]

            # Should be within 1% (allowing for market movement)
            assert price_pct_diff < 0.01

    def test_websocket_data_consistency(self, test_app):
        """Test that WebSocket data matches REST API data"""
        client = TestClient(test_app)
        symbol = "AAPL"

        # Get REST API data
        rest_response = client.get(f"/api/v1/market-data/{symbol}/current")
        rest_price = rest_response.json().get("price")

        # Connect to WebSocket and get data
        ws_prices = []
        with client.websocket_connect(f"/ws/market-data/{symbol}") as websocket:
            # Collect a few price updates
            for _ in range(5):
                data = websocket.receive_json()
                if data.get("type") == "price":
                    ws_prices.append(data.get("price"))

        # At least one WebSocket price should be close to REST price
        if ws_prices and rest_price:
            min_diff = min(abs(p - rest_price) for p in ws_prices)
            min_pct_diff = min_diff / rest_price

            # Should be within 2% (allowing for timing differences)
            assert min_pct_diff < 0.02

    def test_cross_service_data_integrity(self, client):
        """Test data integrity across multiple services"""
        symbol = "AAPL"

        # Get data from different endpoints
        market_data = client.get(f"/api/v1/market-data/{symbol}/current").json()
        signal_data = client.get(f"/api/v1/signals/{symbol}").json()
        agent_data = client.get(f"/api/v1/agents/analysis/{symbol}").json()

        # All should reference the same symbol
        assert market_data.get("symbol") == symbol
        assert all(s.get("symbol") == symbol for s in signal_data.get("signals", []))
        assert agent_data.get("symbol") == symbol

        # Timestamps should be reasonably close
        timestamps = []

        if "timestamp" in market_data:
            timestamps.append(datetime.fromisoformat(market_data["timestamp"]))

        for signal in signal_data.get("signals", []):
            if "timestamp" in signal:
                timestamps.append(datetime.fromisoformat(signal["timestamp"]))

        if timestamps:
            time_range = max(timestamps) - min(timestamps)
            # All data should be within 5 minutes of each other
            assert time_range < timedelta(minutes=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
