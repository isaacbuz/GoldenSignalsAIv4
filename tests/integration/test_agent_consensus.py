"""
Integration Tests for Agent Consensus System
Tests the multi-agent signal generation and consensus mechanism
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.main import DatabaseSignalGenerator
from src.models.signal import Signal, SignalAction, RiskLevel, SignalStatus
from src.models.agent import Agent
from src.services.redis_cache_service import RedisCacheService


class TestAgentConsensus:
    """Test suite for agent consensus mechanism"""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session"""
        session = MagicMock()

        # Mock agents with different weights
        mock_agents = [
            Agent(
                id=1,
                name="RSI_Agent",
                agent_type="rsi",
                consensus_weight=1.2,
                accuracy=0.75,
                is_active=True
            ),
            Agent(
                id=2,
                name="MACD_Agent",
                agent_type="macd",
                consensus_weight=1.0,
                accuracy=0.70,
                is_active=True
            ),
            Agent(
                id=3,
                name="Volume_Agent",
                agent_type="volume",
                consensus_weight=0.8,
                accuracy=0.65,
                is_active=True
            ),
            Agent(
                id=4,
                name="Momentum_Agent",
                agent_type="momentum",
                consensus_weight=1.1,
                accuracy=0.72,
                is_active=True
            ),
            Agent(
                id=5,
                name="Sentiment_Agent",
                agent_type="sentiment",
                consensus_weight=0.9,
                accuracy=0.68,
                is_active=True
            )
        ]

        session.query(Agent).filter(Agent.is_active == True).all.return_value = mock_agents
        return session

    @pytest.fixture
    def signal_generator(self):
        """Create signal generator instance"""
        generator = DatabaseSignalGenerator()
        # Mock cache service
        generator.cache_service = MagicMock(spec=RedisCacheService)
        generator.cache_service.get_agent_result.return_value = None
        return generator

    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data"""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        prices = np.random.randn(30).cumsum() + 150

        data = pd.DataFrame({
            'Open': prices + np.random.randn(30) * 0.5,
            'High': prices + np.abs(np.random.randn(30)),
            'Low': prices - np.abs(np.random.randn(30)),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 30)
        }, index=dates)

        return data

    @pytest.mark.asyncio
    async def test_consensus_all_agents_agree_buy(self, signal_generator, mock_db_session, mock_market_data):
        """Test consensus when all agents agree on BUY signal"""
        with patch('yfinance.Ticker') as mock_ticker:
            # Setup mock data - bullish scenario
            mock_ticker.return_value.history.return_value = mock_market_data
            mock_market_data['Close'].iloc[-5:] = [145, 147, 149, 151, 153]  # Uptrend

            # Generate signal
            signal = await signal_generator.generate_signal('AAPL', mock_db_session)

            # Assertions
            assert signal.action == SignalAction.BUY
            assert signal.confidence >= 0.8  # High confidence when all agree
            assert signal.consensus_strength >= 0.8
            assert signal.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]

            # Verify agent votes were recorded
            assert 'RSI_Agent' in signal.agent_votes
            assert 'MACD_Agent' in signal.agent_votes
            assert len(signal.agent_votes) == 5

    @pytest.mark.asyncio
    async def test_consensus_mixed_signals(self, signal_generator, mock_db_session, mock_market_data):
        """Test consensus with mixed agent signals"""
        with patch('yfinance.Ticker') as mock_ticker:
            # Setup mock data - sideways market
            mock_ticker.return_value.history.return_value = mock_market_data
            mock_market_data['Close'].iloc[-5:] = [150, 149, 150, 149, 150]  # Sideways

            # Generate signal
            signal = await signal_generator.generate_signal('AAPL', mock_db_session)

            # Assertions
            assert signal.action in [SignalAction.HOLD, SignalAction.BUY, SignalAction.SELL]
            assert signal.confidence < 0.8  # Lower confidence with mixed signals
            assert signal.risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH]

    @pytest.mark.asyncio
    async def test_consensus_strong_sell_signal(self, signal_generator, mock_db_session, mock_market_data):
        """Test consensus when agents indicate SELL"""
        with patch('yfinance.Ticker') as mock_ticker:
            # Setup mock data - bearish scenario
            mock_ticker.return_value.history.return_value = mock_market_data
            mock_market_data['Close'].iloc[-5:] = [155, 153, 151, 149, 147]  # Downtrend
            mock_market_data['Volume'].iloc[-2:] *= 2  # High volume on decline

            # Generate signal
            signal = await signal_generator.generate_signal('AAPL', mock_db_session)

            # Assertions
            assert signal.action == SignalAction.SELL
            assert signal.stop_loss is not None
            assert signal.target_price is not None
            assert signal.reasoning is not None

    @pytest.mark.asyncio
    async def test_agent_weight_influence(self, signal_generator, mock_db_session, mock_market_data):
        """Test that agent weights properly influence consensus"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_market_data

            # Modify agent weights to test influence
            agents = mock_db_session.query(Agent).filter(Agent.is_active == True).all.return_value

            # Give RSI agent very high weight
            agents[0].consensus_weight = 3.0  # RSI_Agent

            # Generate signal
            signal = await signal_generator.generate_signal('AAPL', mock_db_session)

            # RSI agent's vote should heavily influence the outcome
            assert signal.agent_votes['RSI_Agent']['weight'] == 3.0

    @pytest.mark.asyncio
    async def test_consensus_with_disabled_agents(self, signal_generator, mock_db_session, mock_market_data):
        """Test consensus when some agents are disabled"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_market_data

            # Disable some agents
            agents = mock_db_session.query(Agent).filter(Agent.is_active == True).all.return_value
            # Only return 3 active agents
            mock_db_session.query(Agent).filter(Agent.is_active == True).all.return_value = agents[:3]

            # Generate signal
            signal = await signal_generator.generate_signal('AAPL', mock_db_session)

            # Should still generate valid signal with fewer agents
            assert signal is not None
            assert len(signal.agent_votes) == 3
            assert signal.action in [SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD]

    @pytest.mark.asyncio
    async def test_risk_assessment_calculation(self, signal_generator, mock_db_session, mock_market_data):
        """Test risk level calculation based on volatility"""
        with patch('yfinance.Ticker') as mock_ticker:
            # High volatility scenario
            mock_market_data['Close'] = mock_market_data['Close'] * (1 + np.random.randn(30) * 0.1)
            mock_ticker.return_value.history.return_value = mock_market_data

            # Generate signal
            signal = await signal_generator.generate_signal('TSLA', mock_db_session)

            # High volatility should result in higher risk
            assert signal.risk_level == RiskLevel.HIGH
            assert 'volatility' in signal.indicators

    @pytest.mark.asyncio
    async def test_consensus_caching(self, signal_generator, mock_db_session, mock_market_data):
        """Test that consensus results are properly cached"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_market_data

            # First call - should generate and cache
            signal1 = await signal_generator.generate_signal('AAPL', mock_db_session)

            # Verify cache was set
            signal_generator.cache_service.set_agent_result.assert_called_once()

            # Setup cache to return cached result
            signal_generator.cache_service.get_agent_result.return_value = {
                'result': {
                    'id': str(signal1.id),
                    'action': signal1.action.value,
                    'confidence': signal1.confidence
                }
            }

            # Mock the database to return the signal
            mock_db_session.query(Signal).filter.return_value.first.return_value = signal1

            # Second call - should use cache
            signal2 = await signal_generator.generate_signal('AAPL', mock_db_session)

            # Should not call market data again
            assert mock_ticker.return_value.history.call_count == 1

    @pytest.mark.asyncio
    async def test_consensus_with_extreme_market_conditions(self, signal_generator, mock_db_session):
        """Test consensus during extreme market conditions"""
        with patch('yfinance.Ticker') as mock_ticker:
            # Create extreme market data - 20% drop
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            prices = np.linspace(150, 120, 30)  # Steady decline

            extreme_data = pd.DataFrame({
                'Open': prices + np.random.randn(30) * 0.5,
                'High': prices + 1,
                'Low': prices - 1,
                'Close': prices,
                'Volume': np.linspace(1000000, 5000000, 30)  # Increasing volume
            }, index=dates)

            mock_ticker.return_value.history.return_value = extreme_data

            # Generate signal
            signal = await signal_generator.generate_signal('AAPL', mock_db_session)

            # Should generate strong sell signal
            assert signal.action == SignalAction.SELL
            assert signal.confidence >= 0.7
            assert signal.risk_level == RiskLevel.HIGH
            assert 'momentum' in signal.reasoning.lower()

    @pytest.mark.asyncio
    async def test_consensus_error_handling(self, signal_generator, mock_db_session):
        """Test error handling in consensus mechanism"""
        with patch('yfinance.Ticker') as mock_ticker:
            # Simulate market data fetch failure
            mock_ticker.return_value.history.return_value = pd.DataFrame()  # Empty dataframe

            # Should handle gracefully and use mock data
            signal = await signal_generator.generate_signal('INVALID', mock_db_session)

            # Should still generate a valid signal
            assert signal is not None
            assert signal.action in [SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD]
            assert signal.confidence > 0

    def test_technical_indicator_calculations(self, signal_generator):
        """Test individual technical indicator calculations"""
        # Test RSI calculation
        prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        rsi = signal_generator.calculate_rsi(prices, window=5)
        assert 0 <= rsi <= 100

        # Test MACD calculation
        prices = pd.Series(np.random.randn(30).cumsum() + 100)
        macd_data = signal_generator.calculate_macd(prices)
        assert 'macd' in macd_data
        assert 'signal' in macd_data
        assert 'histogram' in macd_data

    @pytest.mark.asyncio
    async def test_consensus_performance_metrics(self, signal_generator, mock_db_session, mock_market_data):
        """Test that performance metrics are properly calculated"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_market_data

            # Generate multiple signals
            signals = []
            for symbol in ['AAPL', 'GOOGL', 'MSFT']:
                signal = await signal_generator.generate_signal(symbol, mock_db_session)
                signals.append(signal)

            # Verify all signals have required metrics
            for signal in signals:
                assert signal.confidence > 0
                assert signal.consensus_strength > 0
                assert signal.price > 0
                assert signal.indicators is not None
                assert len(signal.agent_votes) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
