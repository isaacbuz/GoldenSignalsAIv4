"""
Momentum Agent - GoldenSignalsAI V3

Identifies momentum-based trading opportunities using price momentum,
volume momentum, and trend analysis.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseAgent
from ..models.signals import Signal, SignalType, SignalStrength


class MomentumAgent(BaseAgent):
    """
    Agent specializing in momentum-based trading strategies
    """
    
    def __init__(self, name: str, db_manager, redis_manager):
        super().__init__(name, db_manager, redis_manager)
        self.lookback_periods = [5, 10, 20, 50]
        self.momentum_threshold = 0.02  # 2% momentum threshold
    
    async def analyze(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[Signal]:
        """
        Perform momentum analysis on the given symbol
        
        Args:
            symbol: Stock symbol
            market_data: Current market data
            historical_data: Historical price data
            
        Returns:
            Momentum-based signal or None
        """
        try:
            if not historical_data or len(historical_data) < 50:
                logger.warning(f"Insufficient historical data for {symbol} momentum analysis")
                return None
            
            # Convert to DataFrame
            df = self._prepare_dataframe(historical_data)
            
            # Calculate momentum indicators
            momentum_data = await self._calculate_momentum_indicators(df)
            
            # Generate signal
            signal = await self._generate_momentum_signal(symbol, market_data, momentum_data)
            
            return signal
            
        except Exception as e:
            logger.error(f"Momentum analysis failed for {symbol}: {str(e)}")
            return None
    
    def _prepare_dataframe(self, historical_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert historical data to pandas DataFrame"""
        try:
            data = []
            for point in historical_data:
                data.append({
                    'timestamp': point.get('timestamp', datetime.utcnow()),
                    'close': float(point.get('close', point.get('price', 0))),
                    'volume': int(point.get('volume', 0)),
                    'high': float(point.get('high', point.get('price', 0))),
                    'low': float(point.get('low', point.get('price', 0)))
                })
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to prepare DataFrame: {str(e)}")
            raise
    
    async def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate various momentum indicators"""
        try:
            indicators = {}
            
            # Price momentum for different periods
            for period in self.lookback_periods:
                momentum = (df['close'] / df['close'].shift(period) - 1) * 100
                indicators[f'price_momentum_{period}d'] = momentum
            
            # Volume momentum
            for period in [5, 10, 20]:
                vol_momentum = df['volume'] / df['volume'].rolling(window=period).mean()
                indicators[f'volume_momentum_{period}d'] = vol_momentum
            
            # Rate of Change (ROC)
            for period in [5, 10, 20]:
                roc = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
                indicators[f'roc_{period}d'] = roc
            
            # Momentum Oscillator
            momentum_osc = df['close'] - df['close'].shift(10)
            indicators['momentum_oscillator'] = momentum_osc
            
            # Price Velocity (rate of price change acceleration)
            price_velocity = df['close'].diff()
            price_acceleration = price_velocity.diff()
            indicators['price_velocity'] = price_velocity
            indicators['price_acceleration'] = price_acceleration
            
            # Relative Strength vs Market (using self as proxy for now)
            rs_5d = df['close'].pct_change(5) * 100
            rs_20d = df['close'].pct_change(20) * 100
            indicators['relative_strength_5d'] = rs_5d
            indicators['relative_strength_20d'] = rs_20d
            
            # High-Low momentum
            hl_momentum = (df['high'] - df['low']) / df['close']
            indicators['hl_momentum'] = hl_momentum.rolling(window=10).mean()
            
            # Trend strength
            for period in [5, 10, 20]:
                trend_strength = abs(df['close'].rolling(window=period).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                ))
                indicators[f'trend_strength_{period}d'] = trend_strength
            
            return indicators
            
        except Exception as e:
            logger.error(f"Failed to calculate momentum indicators: {str(e)}")
            raise
    
    async def _generate_momentum_signal(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        momentum_data: Dict[str, Any]
    ) -> Optional[Signal]:
        """Generate trading signal based on momentum analysis"""
        try:
            current_price = market_data.get('price', 0)
            if current_price <= 0:
                return None
            
            # Get latest momentum values
            momentum_scores = {}
            
            # Price momentum analysis
            price_momentum_score = 0
            for period in self.lookback_periods:
                key = f'price_momentum_{period}d'
                if key in momentum_data and not momentum_data[key].empty:
                    momentum = momentum_data[key].iloc[-1]
                    if period <= 10:  # Short-term momentum
                        weight = 0.3
                    elif period <= 20:  # Medium-term momentum
                        weight = 0.25
                    else:  # Long-term momentum
                        weight = 0.2
                    
                    if momentum > self.momentum_threshold * 100:  # Convert to percentage
                        price_momentum_score += weight
                    elif momentum < -self.momentum_threshold * 100:
                        price_momentum_score -= weight
            
            momentum_scores['price_momentum'] = price_momentum_score
            
            # Volume momentum analysis
            volume_momentum_score = 0
            for period in [5, 10, 20]:
                key = f'volume_momentum_{period}d'
                if key in momentum_data and not momentum_data[key].empty:
                    vol_momentum = momentum_data[key].iloc[-1]
                    weight = 0.3 if period == 5 else 0.2
                    
                    if vol_momentum > 1.2:  # 20% above average volume
                        volume_momentum_score += weight
                    elif vol_momentum < 0.8:  # 20% below average volume
                        volume_momentum_score -= weight * 0.5  # Less penalty for low volume
            
            momentum_scores['volume_momentum'] = volume_momentum_score
            
            # ROC analysis
            roc_score = 0
            for period in [5, 10, 20]:
                key = f'roc_{period}d'
                if key in momentum_data and not momentum_data[key].empty:
                    roc = momentum_data[key].iloc[-1]
                    weight = 0.25 if period <= 10 else 0.15
                    
                    if roc > 2:  # 2% positive ROC
                        roc_score += weight
                    elif roc < -2:  # 2% negative ROC
                        roc_score -= weight
            
            momentum_scores['roc'] = roc_score
            
            # Momentum oscillator
            momentum_osc_score = 0
            if 'momentum_oscillator' in momentum_data and not momentum_data['momentum_oscillator'].empty:
                momentum_osc = momentum_data['momentum_oscillator'].iloc[-1]
                momentum_osc_sma = momentum_data['momentum_oscillator'].rolling(window=10).mean().iloc[-1]
                
                if momentum_osc > momentum_osc_sma and momentum_osc > 0:
                    momentum_osc_score = 0.2
                elif momentum_osc < momentum_osc_sma and momentum_osc < 0:
                    momentum_osc_score = -0.2
            
            momentum_scores['momentum_oscillator'] = momentum_osc_score
            
            # Price acceleration
            acceleration_score = 0
            if 'price_acceleration' in momentum_data and not momentum_data['price_acceleration'].empty:
                acceleration = momentum_data['price_acceleration'].iloc[-1]
                velocity = momentum_data['price_velocity'].iloc[-1]
                
                if acceleration > 0 and velocity > 0:  # Accelerating upward
                    acceleration_score = 0.15
                elif acceleration < 0 and velocity < 0:  # Accelerating downward
                    acceleration_score = -0.15
            
            momentum_scores['acceleration'] = acceleration_score
            
            # Trend strength
            trend_strength_score = 0
            for period in [5, 10, 20]:
                key = f'trend_strength_{period}d'
                if key in momentum_data and not momentum_data[key].empty:
                    trend_strength = momentum_data[key].iloc[-1]
                    weight = 0.1
                    
                    # Strong trend adds to momentum
                    if trend_strength > current_price * 0.01:  # 1% of price as threshold
                        trend_strength_score += weight
            
            momentum_scores['trend_strength'] = trend_strength_score
            
            # Calculate overall momentum score
            total_momentum_score = sum(momentum_scores.values())
            
            # Determine signal type and confidence
            if total_momentum_score > 0.4:
                signal_type = SignalType.BUY
                confidence = min(total_momentum_score, 0.9)
            elif total_momentum_score < -0.4:
                signal_type = SignalType.SELL
                confidence = min(abs(total_momentum_score), 0.9)
            else:
                signal_type = SignalType.HOLD
                confidence = 0.3
            
            # Adjust confidence based on volume confirmation
            volume_confirmation = momentum_scores.get('volume_momentum', 0)
            if volume_confirmation > 0:
                confidence *= 1.1  # Boost confidence with volume
            elif volume_confirmation < 0:
                confidence *= 0.9  # Reduce confidence without volume
            
            confidence = max(0.1, min(0.9, confidence))
            
            # Determine signal strength
            if confidence >= 0.7:
                strength = SignalStrength.STRONG
            elif confidence >= 0.5:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
            
            # Generate reasoning
            reasoning = self._generate_momentum_reasoning(signal_type, momentum_scores, total_momentum_score)
            
            # Create signal
            signal = await self.create_signal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                strength=strength,
                current_price=current_price,
                reasoning=reasoning,
                features={
                    "total_momentum_score": total_momentum_score,
                    "price_momentum": momentum_scores.get('price_momentum', 0),
                    "volume_momentum": momentum_scores.get('volume_momentum', 0),
                    "roc_score": momentum_scores.get('roc', 0),
                    "momentum_oscillator": momentum_scores.get('momentum_oscillator', 0),
                    "acceleration": momentum_scores.get('acceleration', 0),
                    "trend_strength": momentum_scores.get('trend_strength', 0),
                    "short_term_momentum": momentum_data.get('price_momentum_5d', pd.Series()).iloc[-1] if 'price_momentum_5d' in momentum_data else 0,
                    "medium_term_momentum": momentum_data.get('price_momentum_20d', pd.Series()).iloc[-1] if 'price_momentum_20d' in momentum_data else 0
                }
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Failed to generate momentum signal for {symbol}: {str(e)}")
            return None
    
    def _generate_momentum_reasoning(
        self,
        signal_type: SignalType,
        momentum_scores: Dict[str, float],
        total_score: float
    ) -> str:
        """Generate human-readable reasoning for momentum signal"""
        try:
            reasons = []
            
            # Price momentum reasoning
            price_momentum = momentum_scores.get('price_momentum', 0)
            if price_momentum > 0.2:
                reasons.append("Strong positive price momentum")
            elif price_momentum < -0.2:
                reasons.append("Strong negative price momentum")
            elif abs(price_momentum) > 0.1:
                reasons.append("Moderate price momentum")
            
            # Volume momentum reasoning
            volume_momentum = momentum_scores.get('volume_momentum', 0)
            if volume_momentum > 0.2:
                reasons.append("High volume supporting momentum")
            elif volume_momentum < -0.1:
                reasons.append("Low volume weakening momentum")
            
            # ROC reasoning
            roc_score = momentum_scores.get('roc', 0)
            if roc_score > 0.2:
                reasons.append("Positive rate of change")
            elif roc_score < -0.2:
                reasons.append("Negative rate of change")
            
            # Acceleration reasoning
            acceleration = momentum_scores.get('acceleration', 0)
            if acceleration > 0.1:
                reasons.append("Price acceleration detected")
            elif acceleration < -0.1:
                reasons.append("Price deceleration detected")
            
            # Trend strength reasoning
            trend_strength = momentum_scores.get('trend_strength', 0)
            if trend_strength > 0.05:
                reasons.append("Strong trending behavior")
            
            if not reasons:
                reasons.append("Mixed momentum signals")
            
            return f"Momentum analysis {signal_type.value} (score: {total_score:.2f}): " + "; ".join(reasons)
            
        except Exception as e:
            logger.error(f"Failed to generate momentum reasoning: {str(e)}")
            return f"Momentum analysis {signal_type.value} signal" 