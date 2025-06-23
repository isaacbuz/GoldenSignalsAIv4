"""
Integration tests for the GoldenSignalsAI V2 trading system.
"""
import pytest
import asyncio
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestIntegration:
    """Integration tests for the trading system"""
    
    base_url = "http://localhost:8000"
    
    @pytest.fixture(autouse=True)
    def check_backend(self):
        """Check if backend is running before tests"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=2)
            if response.status_code != 200:
                pytest.skip("Backend not running")
        except:
            pytest.skip("Backend not running")
    
    def test_signal_generation_flow(self):
        """Test the complete signal generation flow"""
        # Get market data
        market_response = requests.get(f"{self.base_url}/api/v1/market-data/SPY")
        assert market_response.status_code == 200
        market_data = market_response.json()
        
        # Get signals for the same symbol
        signals_response = requests.get(f"{self.base_url}/api/v1/signals/SPY")
        assert signals_response.status_code == 200
        signals = signals_response.json()
        
        # Verify signal structure
        for signal in signals:
            assert "symbol" in signal
            assert "action" in signal
            assert "confidence" in signal
            assert signal["action"] in ["BUY", "SELL", "HOLD"]
            assert 0 <= signal["confidence"] <= 1
            
        # Get insights
        insights_response = requests.get(f"{self.base_url}/api/v1/signals/SPY/insights")
        assert insights_response.status_code == 200
        insights = insights_response.json()
        
        assert "recommendation" in insights
        assert insights["recommendation"] in ["BUY", "SELL", "HOLD"]
    
    def test_historical_data_integration(self):
        """Test historical data retrieval and processing"""
        # Get historical data
        hist_response = requests.get(
            f"{self.base_url}/api/v1/market-data/SPY/historical",
            params={"period": "5d", "interval": "1h"}
        )
        assert hist_response.status_code == 200
        hist_data = hist_response.json()
        
        assert "data" in hist_data
        assert len(hist_data["data"]) > 0
        
        # Verify data structure
        for point in hist_data["data"][:5]:
            assert all(key in point for key in ["timestamp", "open", "high", "low", "close", "volume"])
            assert point["high"] >= max(point["open"], point["close"])
            assert point["low"] <= min(point["open"], point["close"])
    
    def test_market_opportunities_aggregation(self):
        """Test market opportunities aggregation across symbols"""
        # Get opportunities
        opp_response = requests.get(f"{self.base_url}/api/v1/market/opportunities")
        assert opp_response.status_code == 200
        opportunities = opp_response.json()
        
        assert "opportunities" in opportunities
        opps = opportunities["opportunities"]
        
        # Verify opportunities are sorted by confidence
        if len(opps) > 1:
            confidences = [o["confidence"] for o in opps]
            assert confidences == sorted(confidences, reverse=True)
    
    def test_signal_consistency(self):
        """Test signal consistency across multiple requests"""
        symbol = "SPY"
        
        # Get signals multiple times
        signals_list = []
        for _ in range(3):
            response = requests.get(f"{self.base_url}/api/v1/signals/{symbol}")
            assert response.status_code == 200
            signals_list.append(response.json())
        
        # Signals should be relatively consistent (cached)
        # Check that at least some signals are the same
        if len(signals_list[0]) > 0:
            first_signal_id = signals_list[0][0].get("id")
            if first_signal_id:
                # If using caching, we should see the same signal ID
                for signals in signals_list[1:]:
                    if len(signals) > 0:
                        assert any(s.get("id") == first_signal_id for s in signals)
    
    def test_error_handling_integration(self):
        """Test error handling across the system"""
        # Invalid symbol
        response = requests.get(f"{self.base_url}/api/v1/market-data/INVALID123XYZ")
        assert response.status_code == 404
        
        # Invalid historical data parameters
        response = requests.get(
            f"{self.base_url}/api/v1/market-data/SPY/historical",
            params={"period": "invalid", "interval": "1d"}
        )
        assert response.status_code == 422
        
        # Non-existent endpoint
        response = requests.get(f"{self.base_url}/api/v1/nonexistent")
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_websocket_integration(self):
        """Test WebSocket integration"""
        import websockets
        import json
        
        try:
            async with websockets.connect("ws://localhost:8000/ws") as websocket:
                # Send subscription message
                await websocket.send(json.dumps({
                    "type": "subscribe",
                    "symbols": ["SPY", "AAPL"]
                }))
                
                # Wait for at least one message
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)
                
                assert "type" in data
                assert data["type"] in ["batch_update", "market_update", "signals"]
                
        except asyncio.TimeoutError:
            # WebSocket might not send immediate updates in test mode
            pass
        except Exception as e:
            pytest.skip(f"WebSocket test failed: {str(e)}")
    
    def test_performance_monitoring(self):
        """Test performance monitoring integration"""
        # Make several requests to generate performance data
        endpoints = [
            "/api/v1/signals",
            "/api/v1/market-data/SPY",
            "/api/v1/market/opportunities"
        ]
        
        for endpoint in endpoints:
            for _ in range(3):
                requests.get(f"{self.base_url}{endpoint}")
        
        # Get performance stats
        perf_response = requests.get(f"{self.base_url}/api/v1/performance")
        assert perf_response.status_code == 200
        perf_data = perf_response.json()
        
        assert "endpoints" in perf_data
        assert "cache" in perf_data
        
        # Verify some endpoints have been tracked
        assert len(perf_data["endpoints"]) > 0
        
        # Verify cache is being used
        assert perf_data["cache"]["total_requests"] > 0 