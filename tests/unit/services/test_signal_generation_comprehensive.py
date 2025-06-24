"""
Comprehensive tests for signal generation engine
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, AsyncMock

from src.services.signal_generation_engine import SignalGenerationEngine, TradingSignal
from src.services.data_quality_validator import DataQualityValidator, DataQualityReport


class TestSignalGenerationEngine:
    """Test suite for SignalGenerationEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create a signal generation engine instance"""
        return SignalGenerationEngine()
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing"""
        dates = pd.date_range(end=datetime.now(timezone.utc), periods=100, freq='D')
        data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(100, 200, 100),
            'Low': np.random.uniform(100, 200, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.randint(1000000, 10000000, 100)
        })
        
        # Ensure high is highest and low is lowest
        data['High'] = data[['Open', 'High', 'Low', 'Close']].max(axis=1)
        data['Low'] = data[['Open', 'High', 'Low', 'Close']].min(axis=1)
        
        return data.set_index('Date')
    
    @pytest.fixture
    def quality_report(self):
        """Create a mock quality report"""
        return DataQualityReport(
            symbol="AAPL",
            is_valid=True,
            completeness=1.0,
            accuracy=1.0,
            timeliness=1.0,
            consistency=1.0,
            issues=[],
            recommendations=[]
        )
    
    @pytest.mark.asyncio
    async def test_generate_signals_success(self, engine, sample_market_data, quality_report):
        """Test successful signal generation"""
        # Mock the data validator
        with patch.object(engine.data_validator, 'get_market_data_with_fallback') as mock_get_data:
            # Setup mock data
            mock_get_data.return_value = (sample_market_data, 'yahoo')
            
            # Mock validate_market_data
            with patch.object(engine.data_validator, 'validate_market_data') as mock_validate:
                mock_validate.return_value = quality_report
                
                # Generate signals
                signals = await engine.generate_signals(['AAPL', 'GOOGL'])
                
                # Assertions
                assert isinstance(signals, list)
                assert all(isinstance(s, TradingSignal) for s in signals)
                assert mock_get_data.call_count == 2
    
    @pytest.mark.asyncio
    async def test_generate_signals_no_data(self, engine):
        """Test signal generation with no data available"""
        with patch.object(engine.data_validator, 'get_market_data_with_fallback') as mock_get_data:
            mock_get_data.return_value = (None, None)
            
            signals = await engine.generate_signals(['INVALID'])
            
            assert signals == []
    
    @pytest.mark.asyncio
    async def test_calculate_indicators(self, engine, sample_market_data):
        """Test technical indicator calculation"""
        indicators = await engine._calculate_indicators(sample_market_data)
        
        # Check all indicators are calculated
        expected_indicators = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_histogram',
            'rsi', 'bb_upper', 'bb_lower', 'bb_percent',
            'volume_sma', 'volume_ratio', 'atr',
            'stoch_k', 'stoch_d'
        ]
        
        for indicator in expected_indicators:
            assert indicator in indicators
            assert len(indicators[indicator]) == len(sample_market_data)
    
    @pytest.mark.asyncio
    async def test_rsi_calculation(self, engine):
        """Test RSI calculation accuracy"""
        # Create simple price series for predictable RSI
        prices = pd.Series([44, 44.25, 44.50, 43.75, 44.75, 45.50, 45.25, 46, 47, 47.50])
        
        rsi = await engine._calculate_rsi(prices, period=9)
        
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(prices)
        # RSI should be between 0 and 100
        assert all(0 <= val <= 100 for val in rsi.dropna())
    
    def test_engineer_features(self, engine, sample_market_data):
        """Test feature engineering"""
        # Calculate indicators first
        indicators = engine._calculate_indicators_sync(sample_market_data)
        
        features = engine._engineer_features(sample_market_data, indicators)
        
        # Check feature columns exist
        expected_features = [
            'price_change', 'high_low_ratio', 'close_open_ratio',
            'volume_change', 'price_volume_trend', 'sma_cross',
            'macd_cross', 'rsi_oversold', 'rsi_overbought',
            'bb_position', 'momentum_5', 'momentum_10',
            'volatility', 'atr_ratio'
        ]
        
        for feature in expected_features:
            assert feature in features.columns
    
    @pytest.mark.asyncio
    async def test_analyze_and_generate_signal(self, engine, sample_market_data, quality_report):
        """Test signal analysis and generation"""
        indicators = await engine._calculate_indicators(sample_market_data)
        features = engine._engineer_features(sample_market_data, indicators)
        
        signal = await engine._analyze_and_generate_signal(
            "AAPL", sample_market_data, indicators, features, quality_report
        )
        
        if signal:  # Signal might be None if conditions aren't met
            assert isinstance(signal, TradingSignal)
            assert signal.symbol == "AAPL"
            assert signal.action in ["BUY", "SELL", "HOLD"]
            assert 0 <= signal.confidence <= 1
            assert signal.risk_level in ["low", "medium", "high"]
    
    def test_signal_to_dict(self, engine):
        """Test signal dictionary conversion"""
        signal = TradingSignal(
            id="TEST_123",
            symbol="AAPL",
            action="BUY",
            confidence=0.75,
            price=150.0,
            timestamp=datetime.now(timezone.utc),
            reason="Test signal",
            indicators={"rsi": 30.0},
            risk_level="medium",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0
        )
        
        signal_dict = signal.to_dict()
        
        assert signal_dict['id'] == "TEST_123"
        assert signal_dict['symbol'] == "AAPL"
        assert signal_dict['action'] == "BUY"
        assert signal_dict['confidence'] == 0.75
        assert isinstance(signal_dict['timestamp'], str)
    
    @pytest.mark.asyncio
    async def test_signal_caching(self, engine, sample_market_data, quality_report):
        """Test signal caching mechanism"""
        with patch.object(engine.data_validator, 'get_market_data_with_fallback') as mock_get_data:
            mock_get_data.return_value = (sample_market_data, 'yahoo')
            
            with patch.object(engine.data_validator, 'validate_market_data') as mock_validate:
                mock_validate.return_value = quality_report
                
                # First call
                signal1 = await engine._generate_signal_for_symbol('AAPL')
                
                # Second call (should use cache)
                signal2 = await engine._generate_signal_for_symbol('AAPL')
                
                # Only one call to get data (second should use cache)
                assert mock_get_data.call_count == 1
    
    def test_train_ml_model(self, engine, sample_market_data):
        """Test ML model training"""
        # Create labels
        labels = pd.Series(
            np.random.choice(['BUY', 'SELL', 'HOLD'], size=len(sample_market_data)),
            index=sample_market_data.index
        )
        
        # Train model
        engine.train_ml_model(sample_market_data, labels)
        
        # Check model is trained
        assert engine.ml_model is not None
        assert hasattr(engine.ml_model, 'predict')
        assert hasattr(engine.ml_model, 'predict_proba')
    
    @pytest.mark.asyncio
    async def test_error_handling(self, engine):
        """Test error handling in signal generation"""
        with patch.object(engine.data_validator, 'get_market_data_with_fallback') as mock_get_data:
            # Simulate an error
            mock_get_data.side_effect = Exception("Network error")
            
            signals = await engine.generate_signals(['AAPL'])
            
            # Should return empty list on error
            assert signals == []
    
    @pytest.mark.asyncio
    async def test_concurrent_signal_generation(self, engine):
        """Test concurrent signal generation for multiple symbols"""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        
        with patch.object(engine, '_generate_signal_for_symbol') as mock_generate:
            # Create different mock signals
            mock_signals = []
            for symbol in symbols:
                signal = TradingSignal(
                    id=f"{symbol}_123",
                    symbol=symbol,
                    action="BUY",
                    confidence=0.7,
                    price=100.0,
                    timestamp=datetime.now(timezone.utc),
                    reason="Test",
                    indicators={},
                    risk_level="medium",
                    entry_price=100.0,
                    stop_loss=95.0,
                    take_profit=110.0
                )
                mock_signals.append(signal)
            
            mock_generate.side_effect = mock_signals
            
            signals = await engine.generate_signals(symbols)
            
            assert len(signals) == len(symbols)
            assert all(s.symbol == symbols[i] for i, s in enumerate(signals)) 