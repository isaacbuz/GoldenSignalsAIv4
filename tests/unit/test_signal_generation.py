"""
Tests for signal generation quality in GoldenSignalsAI V2.
Based on best practices for robust signal generation logic.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestSignalGeneration:
    """Test signal generation logic and quality"""
    
    @pytest.fixture
    def market_data_with_indicators(self):
        """Generate market data with technical indicators"""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        
        # Generate realistic price movement
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 100)
        })
        
        # Calculate technical indicators
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
        data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + 2 * bb_std
        data['bb_lower'] = data['bb_middle'] - 2 * bb_std
        
        # ATR for volatility
        high = data['close'] * 1.01  # Simulate high
        low = data['close'] * 0.99   # Simulate low
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - data['close'].shift()),
            'lc': abs(low - data['close'].shift())
        }).max(axis=1)
        data['atr'] = tr.rolling(window=14).mean()
        
        return data
    
    def test_signal_thresholding(self, market_data_with_indicators):
        """Test signal generation with confidence thresholds"""
        data = market_data_with_indicators.copy()
        
        # Generate mock ML predictions with confidence scores
        np.random.seed(42)
        data['ml_prediction'] = np.random.choice(['buy', 'sell', 'hold'], size=len(data))
        data['ml_confidence'] = np.random.rand(len(data))
        
        # Apply confidence threshold
        confidence_threshold = 0.7
        high_confidence_signals = data[data['ml_confidence'] >= confidence_threshold]
        
        # Verify threshold filtering
        assert all(high_confidence_signals['ml_confidence'] >= confidence_threshold)
        assert len(high_confidence_signals) < len(data), "Should filter out low confidence signals"
        
        # Test dynamic thresholding based on volatility
        volatility_normalized = data['atr'] / data['close']
        dynamic_threshold = 0.6 + (volatility_normalized * 0.2).clip(0, 0.3)
        
        dynamic_signals = data[data['ml_confidence'] >= dynamic_threshold]
        assert len(dynamic_signals) >= len(high_confidence_signals), "Dynamic threshold should be more flexible"
    
    def test_signal_filtering_rules(self, market_data_with_indicators):
        """Test multi-indicator signal filtering"""
        data = market_data_with_indicators.dropna()
        
        # Rule 1: Buy signal only if RSI < 30 and MACD crosses above signal
        buy_signals = data[
            (data['rsi'] < 30) & 
            (data['macd'] > data['macd_signal']) &
            (data['macd'].shift(1) <= data['macd_signal'].shift(1))  # Crossover
        ]
        
        # Rule 2: Sell signal only if RSI > 70 and price above BB upper
        sell_signals = data[
            (data['rsi'] > 70) &
            (data['close'] > data['bb_upper'])
        ]
        
        # Verify signal quality
        if len(buy_signals) > 0:
            assert all(buy_signals['rsi'] < 30), "All buy signals should have RSI < 30"
        
        if len(sell_signals) > 0:
            assert all(sell_signals['rsi'] > 70), "All sell signals should have RSI > 70"
        
        # Test signal frequency limiting (avoid overtrading)
        min_holding_period = 5  # hours
        filtered_buy_signals = []
        last_signal_time = pd.Timestamp('2000-01-01')
        
        for idx, row in buy_signals.iterrows():
            if row['timestamp'] - last_signal_time >= pd.Timedelta(hours=min_holding_period):
                filtered_buy_signals.append(row)
                last_signal_time = row['timestamp']
        
        # Should have fewer signals after filtering
        assert len(filtered_buy_signals) <= len(buy_signals)
    
    def test_context_aware_signal_generation(self, market_data_with_indicators):
        """Test signals that consider market context"""
        data = market_data_with_indicators.dropna()
        
        # Identify market trend
        data['trend'] = np.where(data['sma_20'] > data['sma_50'], 'uptrend', 'downtrend')
        
        # Generate signals based on trend context
        # In uptrend: more aggressive buy signals
        uptrend_buy_signals = data[
            (data['trend'] == 'uptrend') &
            (data['rsi'] < 50) &  # Less strict RSI in uptrend
            (data['close'] > data['sma_20'])
        ]
        
        # In downtrend: more conservative buy signals
        downtrend_buy_signals = data[
            (data['trend'] == 'downtrend') &
            (data['rsi'] < 20) &  # More strict RSI in downtrend
            (data['close'] < data['bb_lower'])  # Must be oversold
        ]
        
        # Verify context-aware rules
        if len(uptrend_buy_signals) > 0:
            assert all(uptrend_buy_signals['trend'] == 'uptrend')
            assert all(uptrend_buy_signals['rsi'] < 50)
        
        if len(downtrend_buy_signals) > 0:
            assert all(downtrend_buy_signals['trend'] == 'downtrend')
            assert all(downtrend_buy_signals['rsi'] < 20)
    
    def test_signal_quality_scoring(self, market_data_with_indicators):
        """Test signal quality scoring mechanism"""
        data = market_data_with_indicators.dropna()
        
        def calculate_signal_quality_score(row):
            """Calculate quality score for a signal"""
            score = 0.0
            
            # Technical indicator alignment
            if row['rsi'] < 30:
                score += 0.2
            elif row['rsi'] > 70:
                score += 0.2
            
            # Trend alignment
            if row['sma_20'] > row['sma_50']:  # Uptrend
                if row['close'] > row['sma_20']:
                    score += 0.2
            else:  # Downtrend
                if row['close'] < row['sma_20']:
                    score += 0.2
            
            # MACD confirmation
            if abs(row['macd_histogram']) > 0.5:  # Strong momentum
                score += 0.2
            
            # Bollinger Band position
            bb_position = (row['close'] - row['bb_lower']) / (row['bb_upper'] - row['bb_lower'])
            if bb_position < 0.2 or bb_position > 0.8:  # Near extremes
                score += 0.2
            
            # Volume confirmation (simulate)
            avg_volume = data['volume'].rolling(window=20).mean().loc[row.name]
            if pd.notna(avg_volume) and row['volume'] > avg_volume * 1.5:
                score += 0.2
            
            return min(score, 1.0)  # Cap at 1.0
        
        # Calculate quality scores for all signals
        data['signal_quality'] = data.apply(calculate_signal_quality_score, axis=1)
        
        # Filter high-quality signals
        high_quality_threshold = 0.6
        high_quality_signals = data[data['signal_quality'] >= high_quality_threshold]
        
        # Verify scoring
        assert 'signal_quality' in data.columns
        assert all(0 <= score <= 1.0 for score in data['signal_quality'])
        assert len(high_quality_signals) < len(data) * 0.5, "High quality signals should be selective"
    
    def test_risk_adjusted_signals(self, market_data_with_indicators):
        """Test signals with integrated risk management"""
        data = market_data_with_indicators.dropna()
        
        # Calculate stop-loss and take-profit levels using ATR
        atr_multiplier_stop = 2.0
        atr_multiplier_target = 3.0
        
        # For buy signals
        data['stop_loss'] = data['close'] - (data['atr'] * atr_multiplier_stop)
        data['take_profit'] = data['close'] + (data['atr'] * atr_multiplier_target)
        
        # Calculate risk-reward ratio
        data['risk'] = data['close'] - data['stop_loss']
        data['reward'] = data['take_profit'] - data['close']
        data['risk_reward_ratio'] = data['reward'] / data['risk']
        
        # Filter signals with good risk-reward ratio
        min_risk_reward = 1.5
        good_rr_signals = data[data['risk_reward_ratio'] >= min_risk_reward]
        
        # Verify risk management
        assert all(good_rr_signals['risk_reward_ratio'] >= min_risk_reward)
        assert all(data['stop_loss'] < data['close']), "Stop loss should be below entry"
        assert all(data['take_profit'] > data['close']), "Take profit should be above entry"
        
        # Test position sizing based on risk
        account_balance = 100000
        risk_per_trade = 0.02  # 2% risk per trade
        
        data['position_size'] = (account_balance * risk_per_trade) / data['risk']
        data['position_value'] = data['position_size'] * data['close']
        
        # Verify position sizing doesn't exceed account balance
        assert all(data['position_value'] <= account_balance)
    
    def test_signal_validation_pipeline(self):
        """Test complete signal validation pipeline"""
        # Create a mock signal
        signal = {
            'symbol': 'SPY',
            'action': 'buy',
            'confidence': 0.85,
            'price': 450.00,
            'timestamp': pd.Timestamp.now(),
            'indicators': {
                'rsi': 25,
                'macd_histogram': 0.8,
                'volume_ratio': 1.8  # Current volume / average volume
            },
            'risk_metrics': {
                'stop_loss': 445.00,
                'take_profit': 460.00,
                'position_size': 100
            }
        }
        
        def validate_signal(signal):
            """Comprehensive signal validation"""
            errors = []
            warnings = []
            
            # Required fields validation
            required_fields = ['symbol', 'action', 'confidence', 'price', 'timestamp']
            for field in required_fields:
                if field not in signal:
                    errors.append(f"Missing required field: {field}")
            
            # Action validation
            if signal.get('action') not in ['buy', 'sell', 'hold']:
                errors.append("Invalid action type")
            
            # Confidence validation
            confidence = signal.get('confidence', 0)
            if not 0 <= confidence <= 1:
                errors.append("Confidence must be between 0 and 1")
            elif confidence < 0.6:
                warnings.append("Low confidence signal")
            
            # Price validation
            price = signal.get('price', 0)
            if price <= 0:
                errors.append("Invalid price")
            
            # Risk metrics validation
            risk_metrics = signal.get('risk_metrics', {})
            if 'stop_loss' in risk_metrics and 'take_profit' in risk_metrics:
                stop_loss = risk_metrics['stop_loss']
                take_profit = risk_metrics['take_profit']
                
                if signal['action'] == 'buy':
                    if stop_loss >= price:
                        errors.append("Stop loss must be below entry price for buy")
                    if take_profit <= price:
                        errors.append("Take profit must be above entry price for buy")
                elif signal['action'] == 'sell':
                    if stop_loss <= price:
                        errors.append("Stop loss must be above entry price for sell")
                    if take_profit >= price:
                        errors.append("Take profit must be below entry price for sell")
            
            # Indicator validation
            indicators = signal.get('indicators', {})
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                if not 0 <= rsi <= 100:
                    errors.append("RSI must be between 0 and 100")
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings
            }
        
        # Test valid signal
        validation_result = validate_signal(signal)
        assert validation_result['valid'], f"Signal should be valid: {validation_result['errors']}"
        
        # Test invalid signals
        invalid_signal = signal.copy()
        invalid_signal['confidence'] = 1.5  # Invalid confidence
        invalid_result = validate_signal(invalid_signal)
        assert not invalid_result['valid']
        assert "Confidence must be between 0 and 1" in invalid_result['errors']
    
    def test_signal_execution_readiness(self):
        """Test if signals are ready for execution"""
        # Mock order book data
        order_book = {
            'bids': [
                {'price': 449.95, 'volume': 1000},
                {'price': 449.90, 'volume': 2000},
                {'price': 449.85, 'volume': 1500}
            ],
            'asks': [
                {'price': 450.05, 'volume': 1200},
                {'price': 450.10, 'volume': 1800},
                {'price': 450.15, 'volume': 2200}
            ],
            'spread': 0.10,
            'mid_price': 450.00
        }
        
        signal = {
            'symbol': 'SPY',
            'action': 'buy',
            'price': 450.00,
            'size': 500,
            'max_slippage': 0.05  # 0.05% max slippage
        }
        
        def check_execution_readiness(signal, order_book):
            """Check if signal can be executed given current market conditions"""
            readiness = {
                'executable': True,
                'issues': [],
                'estimated_fill_price': None
            }
            
            # Check spread
            spread_pct = order_book['spread'] / order_book['mid_price']
            if spread_pct > 0.001:  # 0.1% spread threshold
                readiness['issues'].append(f"Wide spread: {spread_pct:.3%}")
            
            # Check liquidity for market order
            if signal['action'] == 'buy':
                # Check ask side liquidity
                total_volume = sum(ask['volume'] for ask in order_book['asks'])
                if total_volume < signal['size']:
                    readiness['executable'] = False
                    readiness['issues'].append("Insufficient ask liquidity")
                else:
                    # Calculate estimated fill price
                    remaining_size = signal['size']
                    total_cost = 0
                    for ask in order_book['asks']:
                        fill_size = min(remaining_size, ask['volume'])
                        total_cost += fill_size * ask['price']
                        remaining_size -= fill_size
                        if remaining_size == 0:
                            break
                    readiness['estimated_fill_price'] = total_cost / signal['size']
            
            # Check slippage
            if readiness['estimated_fill_price']:
                slippage = abs(readiness['estimated_fill_price'] - signal['price']) / signal['price']
                if slippage > signal['max_slippage']:
                    readiness['executable'] = False
                    readiness['issues'].append(f"Excessive slippage: {slippage:.3%}")
            
            return readiness
        
        # Test execution readiness
        readiness = check_execution_readiness(signal, order_book)
        assert 'executable' in readiness
        assert 'estimated_fill_price' in readiness
        
        # Test with insufficient liquidity
        large_signal = signal.copy()
        large_signal['size'] = 10000  # Larger than available liquidity
        large_readiness = check_execution_readiness(large_signal, order_book)
        assert not large_readiness['executable']
        assert "Insufficient ask liquidity" in large_readiness['issues'] 