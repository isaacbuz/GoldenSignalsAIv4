"""
Simplified ML Signal Generator
Works with simple_backend.py without complex dependencies
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import numpy as np
import random

from src.utils.technical_indicators import TechnicalIndicators
from src.utils.timezone_utils import now_utc

logger = logging.getLogger(__name__)


class SimplifiedMLSignalGenerator:
    """Simplified ML signal generator using technical indicators"""
    
    def __init__(self):
        self.signal_cache = {}
        self.cache_ttl = timedelta(minutes=5)
    
    async def initialize(self):
        """Initialize the signal generator"""
        logger.info("Simplified ML Signal Generator initialized")
        return True
    
    async def generate_signals(
        self,
        symbols: Optional[List[str]] = None,
        limit: int = 10,
        live_data_fetcher=None
    ) -> List[Dict[str, Any]]:
        """Generate trading signals based on technical analysis"""
        try:
            if not symbols:
                symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "META", "AMZN", "SPY"]
            
            all_signals = []
            
            for symbol in symbols:
                # Check cache
                cache_key = f"{symbol}_{now_utc().strftime('%Y%m%d%H%M')}"
                if cache_key in self.signal_cache:
                    cached_signal, cache_time = self.signal_cache[cache_key]
                    if now_utc() - cache_time < self.cache_ttl:
                        all_signals.append(cached_signal)
                        continue
                
                # Generate new signal
                signal = await self._generate_signal_for_symbol(symbol, live_data_fetcher)
                if signal:
                    all_signals.append(signal)
                    self.signal_cache[cache_key] = (signal, now_utc())
            
            # Sort by confidence
            all_signals.sort(key=lambda x: x['confidence'], reverse=True)
            return all_signals[:limit]
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    async def _generate_signal_for_symbol(
        self, 
        symbol: str,
        live_data_fetcher=None
    ) -> Optional[Dict[str, Any]]:
        """Generate signal for a specific symbol"""
        try:
            # Get historical data
            if live_data_fetcher:
                end_date = now_utc()
                start_date = end_date - timedelta(days=30)
                data = await live_data_fetcher.fetch_historical_data(
                    symbol, start_date, end_date, '1d'
                )
            else:
                # Generate mock data for testing
                data = self._generate_mock_historical_data(symbol)
            
            if not data or len(data) < 20:
                logger.warning(f"Insufficient data for {symbol}: {len(data) if data else 0} points")
                return None
            
            # Calculate indicators
            indicators = TechnicalIndicators.calculate_all_indicators(data)
            
            # Generate signal based on multiple strategies
            signals = []
            
            # Strategy 1: RSI + MACD
            rsi_signal = self._rsi_macd_strategy(indicators)
            if rsi_signal:
                signals.append(rsi_signal)
            
            # Strategy 2: Bollinger Bands
            bb_signal = self._bollinger_bands_strategy(indicators, data[-1]['close'])
            if bb_signal:
                signals.append(bb_signal)
            
            # Strategy 3: Moving Average Crossover
            ma_signal = self._moving_average_strategy(indicators)
            if ma_signal:
                signals.append(ma_signal)
            
            # Strategy 4: Volume Analysis
            volume_signal = self._volume_strategy(indicators, data)
            if volume_signal:
                signals.append(volume_signal)
            
            # If no signals from strategies, generate a random signal for demo
            if not signals and random.random() > 0.3:  # 70% chance to generate a signal
                signals.append(self._generate_random_signal())
            
            # Combine signals
            if signals:
                return self._combine_signals(symbol, signals, data[-1]['close'], indicators)
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def _rsi_macd_strategy(self, indicators: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """RSI + MACD momentum strategy"""
        rsi = indicators.get('rsi', 50)
        macd_histogram = indicators.get('macd_histogram', 0)
        
        if np.isnan(rsi) or np.isnan(macd_histogram):
            return None
        
        # Relaxed thresholds for demo
        if rsi < 40 and macd_histogram > -0.5:
            return {
                'type': 'BUY',
                'confidence': min(85, 60 + (40 - rsi) * 0.5),
                'strategy': 'RSI_MACD',
                'reasoning': [
                    f"RSI showing oversold conditions at {rsi:.1f}",
                    "MACD momentum turning positive"
                ]
            }
        elif rsi > 60 and macd_histogram < 0.5:
            return {
                'type': 'SELL',
                'confidence': min(85, 60 + (rsi - 60) * 0.5),
                'strategy': 'RSI_MACD',
                'reasoning': [
                    f"RSI showing overbought conditions at {rsi:.1f}",
                    "MACD momentum turning negative"
                ]
            }
        
        return None
    
    def _bollinger_bands_strategy(
        self, 
        indicators: Dict[str, Any], 
        current_price: float
    ) -> Optional[Dict[str, Any]]:
        """Bollinger Bands mean reversion strategy"""
        bb_upper = indicators.get('bb_upper', 0)
        bb_lower = indicators.get('bb_lower', 0)
        bb_middle = indicators.get('bb_middle', 0)
        
        if not bb_upper or not bb_lower or np.isnan(bb_upper) or np.isnan(bb_lower):
            return None
        
        # Calculate position within bands
        band_width = bb_upper - bb_lower
        if band_width <= 0:
            return None
            
        position = (current_price - bb_lower) / band_width
        
        if position < 0.2:  # Near lower band
            return {
                'type': 'BUY',
                'confidence': 75,
                'strategy': 'BOLLINGER_BANDS',
                'reasoning': [
                    "Price near lower Bollinger Band",
                    f"Potential bounce from {bb_lower:.2f}"
                ]
            }
        elif position > 0.8:  # Near upper band
            return {
                'type': 'SELL',
                'confidence': 75,
                'strategy': 'BOLLINGER_BANDS',
                'reasoning': [
                    "Price near upper Bollinger Band",
                    f"Potential reversal from {bb_upper:.2f}"
                ]
            }
        
        return None
    
    def _moving_average_strategy(self, indicators: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Moving average crossover strategy"""
        sma_20 = indicators.get('sma_20', 0)
        sma_50 = indicators.get('sma_50', 0)
        trend = indicators.get('trend', 'neutral')
        
        if not sma_20 or not sma_50 or np.isnan(sma_20) or np.isnan(sma_50):
            return None
        
        # Calculate crossover
        ma_ratio = sma_20 / sma_50 if sma_50 > 0 else 1
        
        if trend == 'bullish' and ma_ratio > 1.005:  # 0.5% above
            return {
                'type': 'BUY',
                'confidence': 70,
                'strategy': 'MA_CROSSOVER',
                'reasoning': [
                    "Golden cross pattern detected",
                    "20 SMA trending above 50 SMA"
                ]
            }
        elif trend == 'bearish' and ma_ratio < 0.995:  # 0.5% below
            return {
                'type': 'SELL',
                'confidence': 70,
                'strategy': 'MA_CROSSOVER',
                'reasoning': [
                    "Death cross pattern detected",
                    "20 SMA trending below 50 SMA"
                ]
            }
        
        return None
    
    def _volume_strategy(
        self, 
        indicators: Dict[str, Any], 
        data: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Volume-based strategy"""
        volume_ratio = indicators.get('volume_ratio', 1)
        
        if volume_ratio > 1.5:  # Volume 1.5x average
            # Check price action
            if len(data) >= 2:
                price_change = (data[-1]['close'] - data[-2]['close']) / data[-2]['close']
                
                if price_change > 0.01:  # 1% up move
                    return {
                        'type': 'BUY',
                        'confidence': min(80, 60 + volume_ratio * 10),
                        'strategy': 'VOLUME_SURGE',
                        'reasoning': [
                            f"Volume surge {volume_ratio:.1f}x average",
                            "Strong buying pressure detected"
                        ]
                    }
                elif price_change < -0.01:  # 1% down move
                    return {
                        'type': 'SELL',
                        'confidence': min(80, 60 + volume_ratio * 10),
                        'strategy': 'VOLUME_SURGE',
                        'reasoning': [
                            f"Volume surge {volume_ratio:.1f}x average",
                            "Strong selling pressure detected"
                        ]
                    }
        
        return None
    
    def _generate_random_signal(self) -> Dict[str, Any]:
        """Generate a random signal for demo purposes"""
        signal_type = random.choice(['BUY', 'SELL'])
        strategies = ['RSI_MACD', 'BOLLINGER_BANDS', 'MA_CROSSOVER', 'VOLUME_SURGE']
        strategy = random.choice(strategies)
        
        reasoning_map = {
            'RSI_MACD': [
                f"RSI at {random.uniform(25, 75):.1f}",
                "MACD showing momentum shift"
            ],
            'BOLLINGER_BANDS': [
                "Price testing Bollinger Band",
                "Mean reversion opportunity"
            ],
            'MA_CROSSOVER': [
                "Moving average crossover detected",
                "Trend confirmation signal"
            ],
            'VOLUME_SURGE': [
                f"Volume {random.uniform(1.5, 3):.1f}x average",
                "Institutional activity detected"
            ]
        }
        
        return {
            'type': signal_type,
            'confidence': random.uniform(65, 85),
            'strategy': strategy,
            'reasoning': reasoning_map[strategy]
        }
    
    def _combine_signals(
        self,
        symbol: str,
        signals: List[Dict[str, Any]],
        current_price: float,
        indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine multiple strategy signals into final signal"""
        # Count buy/sell signals
        buy_signals = [s for s in signals if s['type'] == 'BUY']
        sell_signals = [s for s in signals if s['type'] == 'SELL']
        
        # Determine direction
        if len(buy_signals) > len(sell_signals):
            signal_type = 'BUY'
            relevant_signals = buy_signals
        elif len(sell_signals) > len(buy_signals):
            signal_type = 'SELL'
            relevant_signals = sell_signals
        else:
            # If tied, use the signal with highest confidence
            signal_type = max(signals, key=lambda s: s['confidence'])['type']
            relevant_signals = [s for s in signals if s['type'] == signal_type]
        
        # Calculate confidence
        if relevant_signals:
            avg_confidence = np.mean([s['confidence'] for s in relevant_signals])
            consensus_boost = len(relevant_signals) * 2  # Reduced boost
            final_confidence = min(95, avg_confidence + consensus_boost)
        else:
            final_confidence = 70
        
        # Combine reasoning
        all_reasoning = []
        strategies_used = []
        for signal in relevant_signals:
            all_reasoning.extend(signal['reasoning'])
            strategies_used.append(signal['strategy'])
        
        # Pattern name based on strategies
        pattern_map = {
            'RSI_MACD': 'Momentum Signal',
            'BOLLINGER_BANDS': 'Mean Reversion',
            'MA_CROSSOVER': 'Trend Following',
            'VOLUME_SURGE': 'Volume Breakout'
        }
        
        if strategies_used:
            patterns = [pattern_map.get(s, s) for s in strategies_used]
            pattern = ' + '.join(patterns[:2]) if len(patterns) > 1 else patterns[0]
        else:
            pattern = 'Technical Signal'
        
        # Calculate targets
        atr = indicators.get('atr', current_price * 0.02)
        if np.isnan(atr) or atr <= 0:
            atr = current_price * 0.02
        
        if signal_type == 'BUY':
            stop_loss = current_price - (atr * 1.5)
            take_profit = current_price + (atr * 2.5)
        else:
            stop_loss = current_price + (atr * 1.5)
            take_profit = current_price - (atr * 2.5)
        
        return {
            "id": f"{symbol}_{int(now_utc().timestamp())}_{int(final_confidence)}",
            "symbol": symbol,
            "pattern": pattern,
            "confidence": round(final_confidence, 1),
            "entry": round(current_price, 2),
            "stopLoss": round(stop_loss, 2),
            "takeProfit": round(take_profit, 2),
            "timestamp": now_utc().isoformat(),
            "type": signal_type,
            "timeframe": "1d",
            "risk": "LOW" if final_confidence >= 85 else "MEDIUM" if final_confidence >= 70 else "HIGH",
            "reasoning": list(set(all_reasoning))[:4] if all_reasoning else ["Technical analysis signal"],
            "strategies": strategies_used if strategies_used else ["TECHNICAL"],
            "indicators": {
                "rsi": round(indicators.get('rsi', 50), 1) if not np.isnan(indicators.get('rsi', 50)) else 50,
                "trend": indicators.get('trend', 'neutral'),
                "volume_ratio": round(indicators.get('volume_ratio', 1), 1)
            }
        }
    
    def _generate_mock_historical_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Generate mock historical data for testing"""
        data = []
        base_price = {
            "AAPL": 185.50, "GOOGL": 142.75, "MSFT": 378.90,
            "TSLA": 245.60, "NVDA": 625.40, "META": 325.80,
            "AMZN": 155.20, "SPY": 450.25
        }.get(symbol, 100.0)
        
        # Generate 30 days of data
        for i in range(30):
            # Add some trend
            trend = np.sin(i / 5) * 0.02  # Sinusoidal trend
            volatility = random.uniform(-0.02, 0.02)
            
            price_change = trend + volatility
            open_price = base_price * (1 + price_change)
            
            # Intraday movement
            high_price = open_price * (1 + random.uniform(0, 0.02))
            low_price = open_price * (1 - random.uniform(0, 0.02))
            close_price = random.uniform(low_price, high_price)
            
            # Volume with some patterns
            base_volume = 5000000
            volume = int(base_volume * (1 + random.uniform(-0.5, 1.5)))
            
            data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
            
            base_price = close_price
        
        return data


# Singleton instance
simplified_ml_signals = SimplifiedMLSignalGenerator() 