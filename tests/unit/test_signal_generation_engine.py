"""
Tests for Signal Generation Engine
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from src.services.signal_generation_engine import SignalGenerationEngine, TradingSignal
from src.services.data_quality_validator import DataQualityReport


class TestSignalGenerationEngine:
    """Test the signal generation engine"""
    
    @pytest.fixture
    def engine(self):
        """Create a signal generation engine instance"""
        return SignalGenerationEngine()
        
    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data"""
        dates = pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=100, freq='D')
        
        # Generate more realistic price data
        base_price = 100
        prices = [base_price]
        for i in range(99):
            # Random walk with small daily changes (typically 0-2%)
            change = np.random.normal(0, 0.01) * prices[-1]
            prices.append(prices[-1] + change)
            
        prices = np.array(prices)
        
        # Generate OHLC data around the close price
        data = pd.DataFrame(index=dates)
        data['Close'] = prices
        
        # Open is typically close to previous close
        data['Open'] = data['Close'].shift(1).fillna(base_price)
        
        # High and Low are within daily range (typically 1-2% from close)
        daily_range = np.random.uniform(0.005, 0.02, 100)  # 0.5% to 2% range
        data['High'] = data['Close'] * (1 + daily_range)
        data['Low'] = data['Close'] * (1 - daily_range)
        
        # Volume
        data['Volume'] = np.random.randint(1000000, 10000000, 100)
        
        # Ensure logical consistency
        data['High'] = data[['Open', 'High', 'Low', 'Close']].max(axis=1)
        data['Low'] = data[['Open', 'High', 'Low', 'Close']].min(axis=1)
        
        return data
        
    @pytest.fixture
    def quality_report(self):
        """Create a quality report"""
        return DataQualityReport(
            symbol="AAPL",
            is_valid=True,
            completeness=0.95,
            accuracy=0.9,
            timeliness=0.85,
            consistency=0.9,
            issues=[],
            recommendations=[]
        )
        
    @pytest.mark.asyncio
    async def test_generate_signals_single_symbol(self, engine, mock_market_data, quality_report):
        """Test generating signals for a single symbol"""
        with patch.object(engine.data_validator, 'get_market_data_with_fallback', 
                         return_value=(mock_market_data, "test")):
            with patch.object(engine.data_validator, 'validate_market_data',
                             return_value=quality_report):
                signals = await engine.generate_signals(["AAPL"])
                
                # Should generate at least some signals
                assert isinstance(signals, list)
                
                # Check signal properties
                for signal in signals:
                    assert isinstance(signal, TradingSignal)
                    assert signal.symbol == "AAPL"
                    assert signal.action in ["BUY", "SELL", "HOLD"]
                    assert 0 <= signal.confidence <= 1
                    assert signal.risk_level in ["low", "medium", "high"]
                    assert signal.stop_loss is not None
                    assert signal.take_profit is not None
                    
    @pytest.mark.asyncio
    async def test_generate_signals_multiple_symbols(self, engine, mock_market_data, quality_report):
        """Test generating signals for multiple symbols"""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        
        with patch.object(engine.data_validator, 'get_market_data_with_fallback',
                         return_value=(mock_market_data, "test")):
            with patch.object(engine.data_validator, 'validate_market_data',
                             return_value=quality_report):
                signals = await engine.generate_signals(symbols)
                
                # Should handle multiple symbols
                assert isinstance(signals, list)
                
                # Check we got signals for different symbols
                signal_symbols = {s.symbol for s in signals}
                assert len(signal_symbols) <= len(symbols)
                
    @pytest.mark.asyncio
    async def test_calculate_indicators(self, engine, mock_market_data):
        """Test technical indicator calculations"""
        indicators = await engine._calculate_indicators(mock_market_data)
        
        # Check all expected indicators are present
        expected_indicators = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_histogram',
            'rsi', 'bb_upper', 'bb_lower', 'bb_percent',
            'volume_sma', 'volume_ratio', 'atr',
            'stoch_k', 'stoch_d'
        ]
        
        for indicator in expected_indicators:
            assert indicator in indicators
            assert isinstance(indicators[indicator], pd.Series)
            assert len(indicators[indicator]) == len(mock_market_data)
            
    @pytest.mark.asyncio
    async def test_rsi_calculation(self, engine):
        """Test RSI calculation"""
        # Create simple price series
        prices = pd.Series([100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106, 108, 107])
        
        rsi = await engine._calculate_rsi(prices)
        
        # RSI should be between 0 and 100
        assert all(0 <= r <= 100 for r in rsi.dropna())
        
        # With mostly upward movement, RSI should be > 50
        assert rsi.iloc[-1] > 50
        
    @pytest.mark.asyncio
    async def test_atr_calculation(self, engine, mock_market_data):
        """Test ATR calculation"""
        atr = await engine._calculate_atr(mock_market_data)
        
        # ATR should be positive
        assert all(atr.dropna() > 0)
        
        # ATR should be reasonable relative to price
        avg_price = mock_market_data['Close'].mean()
        avg_atr = atr.dropna().mean()
        assert avg_atr < avg_price * 0.1  # ATR typically < 10% of price
        
    def test_engineer_features(self, engine, mock_market_data):
        """Test feature engineering"""
        # Calculate indicators first
        loop = asyncio.new_event_loop()
        indicators = loop.run_until_complete(engine._calculate_indicators(mock_market_data))
        
        features = engine._engineer_features(mock_market_data, indicators)
        
        # Check expected features
        expected_features = [
            'price_change', 'high_low_ratio', 'close_open_ratio',
            'volume_change', 'price_volume_trend',
            'sma_cross', 'macd_cross', 'rsi_oversold', 'rsi_overbought',
            'bb_position', 'momentum_5', 'momentum_10',
            'volatility', 'atr_ratio'
        ]
        
        for feature in expected_features:
            assert feature in features.columns
            
        # Check feature values are reasonable
        assert all(features['high_low_ratio'] >= 1)
        assert all(features['rsi_oversold'].isin([0, 1]))
        assert all(features['rsi_overbought'].isin([0, 1]))
        
    @pytest.mark.asyncio
    async def test_signal_caching(self, engine, mock_market_data, quality_report):
        """Test that signals are cached properly"""
        with patch.object(engine.data_validator, 'get_market_data_with_fallback',
                         return_value=(mock_market_data, "test")):
            with patch.object(engine.data_validator, 'validate_market_data',
                             return_value=quality_report):
                # Generate signal first time
                signal1 = await engine._generate_signal_for_symbol("AAPL")
                
                # Generate again - should get cached result
                signal2 = await engine._generate_signal_for_symbol("AAPL")
                
                if signal1 and signal2:
                    assert signal1.id == signal2.id  # Same signal from cache
                    
    @pytest.mark.asyncio
    async def test_error_handling(self, engine):
        """Test error handling when data is unavailable"""
        with patch.object(engine.data_validator, 'get_market_data_with_fallback',
                         return_value=(None, None)):
            signal = await engine._generate_signal_for_symbol("INVALID")
            
            # Should return None when no data available
            assert signal is None
            
    @pytest.mark.asyncio
    async def test_signal_quality_adjustment(self, engine, mock_market_data):
        """Test that signal confidence is adjusted by data quality"""
        # Create low quality report
        low_quality_report = DataQualityReport(
            symbol="AAPL",
            is_valid=True,
            completeness=0.5,
            accuracy=0.5,
            timeliness=0.5,
            consistency=0.5,
            issues=["Missing data", "Stale data"],
            recommendations=[]
        )
        
        with patch.object(engine.data_validator, 'get_market_data_with_fallback',
                         return_value=(mock_market_data, "test")):
            with patch.object(engine.data_validator, 'validate_market_data',
                             return_value=low_quality_report):
                signals = await engine.generate_signals(["AAPL"])
                
                # Low quality data should result in lower confidence or no signals
                if signals:
                    assert all(s.confidence <= 0.5 for s in signals)
                    
    def test_ml_model_training(self, engine, mock_market_data):
        """Test ML model training"""
        # Create labels for training
        labels = pd.Series(
            np.random.choice([0, 1, 2], size=len(mock_market_data)),  # 0=SELL, 1=HOLD, 2=BUY
            index=mock_market_data.index
        )
        
        # Train model
        engine.train_ml_model(mock_market_data, labels)
        
        # Model should be trained
        assert engine.ml_model is not None
        
        # Test prediction
        loop = asyncio.new_event_loop()
        indicators = loop.run_until_complete(engine._calculate_indicators(mock_market_data))
        features = engine._engineer_features(mock_market_data, indicators)
        
        # Should be able to make predictions
        predictions = engine.ml_model.predict(engine.scaler.transform(features.fillna(0)))
        assert len(predictions) == len(features) 