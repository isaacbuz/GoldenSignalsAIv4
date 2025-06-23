"""
Integration tests for signals API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from datetime import datetime, timezone

from src.main import app
from src.services.signal_generation_engine import TradingSignal


class TestSignalsAPI:
    """Integration tests for signals API"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_signals(self):
        """Create mock signals for testing"""
        return [
            {
                'id': 'AAPL_123',
                'symbol': 'AAPL',
                'action': 'BUY',
                'confidence': 0.75,
                'price': 150.50,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'reason': 'RSI oversold; MACD bullish',
                'indicators': {'rsi': 28.5, 'macd': 1.2},
                'risk_level': 'medium',
                'entry_price': 150.50,
                'stop_loss': 147.00,
                'take_profit': 156.00,
                'metadata': {},
                'quality_score': 0.85
            },
            {
                'id': 'GOOGL_124',
                'symbol': 'GOOGL',
                'action': 'SELL',
                'confidence': 0.68,
                'price': 2500.00,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'reason': 'RSI overbought; Bearish divergence',
                'indicators': {'rsi': 75.0, 'macd': -0.5},
                'risk_level': 'high',
                'entry_price': 2500.00,
                'stop_loss': 2550.00,
                'take_profit': 2400.00,
                'metadata': {},
                'quality_score': 0.80
            }
        ]
    
    def test_generate_signals_default_symbols(self, client):
        """Test signal generation with default symbols"""
        with patch('src.application.services.ApplicationServices.generate_signals') as mock_generate:
            mock_generate.return_value = self.mock_signals()
            
            response = client.get("/api/v1/signals/generate")
            
            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'success'
            assert data['count'] == 2
            assert len(data['signals']) == 2
            
            # Verify default symbols were used
            mock_generate.assert_called_once_with(['AAPL', 'GOOGL', 'MSFT'])
    
    def test_generate_signals_custom_symbols(self, client):
        """Test signal generation with custom symbols"""
        with patch('src.application.services.ApplicationServices.generate_signals') as mock_generate:
            mock_generate.return_value = self.mock_signals()
            
            response = client.get("/api/v1/signals/generate?symbols=TSLA&symbols=NVDA")
            
            assert response.status_code == 200
            mock_generate.assert_called_once_with(['TSLA', 'NVDA'])
    
    def test_generate_signals_error_handling(self, client):
        """Test error handling in signal generation"""
        with patch('src.application.services.ApplicationServices.generate_signals') as mock_generate:
            mock_generate.side_effect = Exception("Service unavailable")
            
            response = client.get("/api/v1/signals/generate")
            
            assert response.status_code == 500
            assert "Service unavailable" in response.json()['detail']
    
    def test_get_signal_performance(self, client):
        """Test getting signal performance"""
        mock_performance = {
            'signal_id': 'AAPL_123',
            'profit_loss': 250.00,
            'profit_loss_percent': 5.2,
            'holding_period_hours': 48,
            'status': 'closed'
        }
        
        with patch('src.application.services.ApplicationServices.get_signal_performance') as mock_perf:
            mock_perf.return_value = mock_performance
            
            response = client.get("/api/v1/signals/AAPL_123/performance")
            
            assert response.status_code == 200
            assert response.json() == mock_performance
    
    def test_get_signal_performance_not_found(self, client):
        """Test signal performance not found"""
        with patch('src.application.services.ApplicationServices.get_signal_performance') as mock_perf:
            mock_perf.return_value = {"error": "Signal not found"}
            
            response = client.get("/api/v1/signals/INVALID_ID/performance")
            
            assert response.status_code == 404
    
    def test_analyze_signal_risk(self, client):
        """Test risk analysis endpoint"""
        mock_risk_analysis = {
            'total_risk': 0.02,
            'position_size': 0.1,
            'max_loss': 200.00,
            'risk_reward_ratio': 2.5,
            'recommendations': ['Position size appropriate', 'Consider tighter stop loss']
        }
        
        with patch('src.application.services.ApplicationServices.analyze_risk') as mock_risk:
            mock_risk.return_value = mock_risk_analysis
            
            response = client.post(
                "/api/v1/signals/analyze-risk",
                json=self.mock_signals()
            )
            
            assert response.status_code == 200
            assert response.json() == mock_risk_analysis
    
    def test_validate_data_quality(self, client):
        """Test data quality validation endpoint"""
        mock_validation = {
            'symbol': 'AAPL',
            'is_valid': True,
            'issues': [],
            'score': 0.95,
            'source': 'yahoo'
        }
        
        with patch('src.application.services.ApplicationServices.validate_data_quality') as mock_validate:
            mock_validate.return_value = mock_validation
            
            response = client.get("/api/v1/signals/validate/AAPL")
            
            assert response.status_code == 200
            assert response.json() == mock_validation
    
    def test_validate_data_quality_invalid_symbol(self, client):
        """Test data quality validation with invalid symbol"""
        with patch('src.application.services.ApplicationServices.validate_data_quality') as mock_validate:
            mock_validate.return_value = {
                'symbol': 'INVALID',
                'is_valid': False,
                'issues': ['Symbol not found'],
                'score': 0.0
            }
            
            response = client.get("/api/v1/signals/validate/INVALID")
            
            assert response.status_code == 200
            data = response.json()
            assert data['is_valid'] is False
            assert 'Symbol not found' in data['issues']
    
    @pytest.mark.parametrize("endpoint,method", [
        ("/api/v1/signals/generate", "GET"),
        ("/api/v1/signals/AAPL_123/performance", "GET"),
        ("/api/v1/signals/analyze-risk", "POST"),
        ("/api/v1/signals/validate/AAPL", "GET"),
    ])
    def test_api_endpoints_exist(self, client, endpoint, method):
        """Test that all API endpoints are accessible"""
        # Mock the services to avoid actual execution
        with patch('src.application.services.get_services'):
            if method == "GET":
                response = client.get(endpoint)
            else:
                response = client.post(endpoint, json=[])
            
            # Should not return 404
            assert response.status_code != 404 