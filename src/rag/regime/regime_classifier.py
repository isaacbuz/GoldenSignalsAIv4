"""
Market Regime Classification System
Identifies and classifies current market regime for context-aware trading
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import talib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

@dataclass
class MarketRegime:
    """Represents a market regime"""
    regime_type: str
    confidence: float
    volatility: str  # low, medium, high
    trend: str  # bullish, bearish, sideways
    characteristics: Dict[str, float]
    start_date: Optional[datetime]
    expected_duration: Optional[int]  # days

class RegimeClassifier:
    """Classifies market regimes using multiple indicators"""
    
    def __init__(self):
        self.regime_history = []
        self.scaler = StandardScaler()
        self.regime_models = {}
        self.current_regime = None
        
        # Define regime characteristics
        self.regime_definitions = {
            "bull_quiet": {
                "trend": "bullish",
                "volatility": "low",
                "characteristics": {"returns": 0.05, "volatility": 0.1, "volume": 0.8}
            },
            "bull_volatile": {
                "trend": "bullish", 
                "volatility": "high",
                "characteristics": {"returns": 0.08, "volatility": 0.25, "volume": 1.2}
            },
            "bear_quiet": {
                "trend": "bearish",
                "volatility": "low", 
                "characteristics": {"returns": -0.03, "volatility": 0.12, "volume": 0.7}
            },
            "bear_volatile": {
                "trend": "bearish",
                "volatility": "high",
                "characteristics": {"returns": -0.08, "volatility": 0.35, "volume": 1.5}
            },
            "sideways": {
                "trend": "sideways",
                "volatility": "medium",
                "characteristics": {"returns": 0.0, "volatility": 0.15, "volume": 0.9}
            },
            "transition": {
                "trend": "mixed",
                "volatility": "high",
                "characteristics": {"returns": 0.0, "volatility": 0.3, "volume": 1.3}
            }
        }
    
    async def classify_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """Classify current market regime"""
        # Calculate regime indicators
        indicators = self._calculate_indicators(market_data)
        
        # Score each regime type
        regime_scores = {}
        for regime_type, definition in self.regime_definitions.items():
            score = self._score_regime(indicators, definition["characteristics"])
            regime_scores[regime_type] = score
        
        # Select best matching regime
        best_regime = max(regime_scores.items(), key=lambda x: x[1])
        regime_type = best_regime[0]
        confidence = best_regime[1]
        
        # Create regime object
        regime_def = self.regime_definitions[regime_type]
        regime = MarketRegime(
            regime_type=regime_type,
            confidence=confidence,
            volatility=regime_def["volatility"],
            trend=regime_def["trend"],
            characteristics=indicators,
            start_date=self._detect_regime_start(market_data, regime_type),
            expected_duration=self._estimate_duration(regime_type)
        )
        
        self.current_regime = regime
        self.regime_history.append(regime)
        
        return regime
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate regime indicators"""
        indicators = {}
        
        # Returns
        returns = data['close'].pct_change()
        indicators['returns'] = returns.mean()
        indicators['returns_std'] = returns.std()
        
        # Volatility (using ATR)
        indicators['volatility'] = talib.ATR(
            data['high'].values,
            data['low'].values, 
            data['close'].values,
            timeperiod=14
        )[-1] / data['close'].iloc[-1]
        
        # Volume
        indicators['volume'] = data['volume'].iloc[-20:].mean() / data['volume'].mean()
        
        # Trend strength (ADX)
        indicators['trend_strength'] = talib.ADX(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            timeperiod=14
        )[-1]
        
        # Market breadth (simplified)
        indicators['breadth'] = self._calculate_breadth(data)
        
        # Fear & Greed indicators
        indicators['rsi'] = talib.RSI(data['close'].values)[-1]
        indicators['vix_proxy'] = self._calculate_vix_proxy(data)
        
        return indicators
    
    def _score_regime(self, indicators: Dict[str, float], 
                     target_characteristics: Dict[str, float]) -> float:
        """Score how well indicators match regime characteristics"""
        score = 0.0
        weights = {"returns": 0.3, "volatility": 0.3, "volume": 0.2, "others": 0.2}
        
        # Compare key metrics
        for key, target_value in target_characteristics.items():
            if key in indicators:
                actual_value = indicators[key]
                # Calculate similarity (inverse of normalized difference)
                diff = abs(actual_value - target_value) / (abs(target_value) + 0.001)
                similarity = 1 / (1 + diff)
                score += similarity * weights.get(key, weights["others"])
        
        return score
    
    def _detect_regime_start(self, data: pd.DataFrame, regime_type: str) -> datetime:
        """Detect when current regime started"""
        # Simplified detection - look for significant change
        lookback = min(60, len(data))
        
        for i in range(lookback, 0, -1):
            subset = data.iloc[-i:]
            indicators = self._calculate_indicators(subset)
            
            # Check if indicators match current regime
            score = self._score_regime(indicators, 
                                     self.regime_definitions[regime_type]["characteristics"])
            
            if score < 0.7:  # Regime was different
                return data.index[-i + 1]
        
        return data.index[-lookback]
    
    def _estimate_duration(self, regime_type: str) -> int:
        """Estimate expected regime duration in days"""
        # Based on historical averages
        avg_durations = {
            "bull_quiet": 120,
            "bull_volatile": 45,
            "bear_quiet": 60,
            "bear_volatile": 30,
            "sideways": 90,
            "transition": 21
        }
        
        return avg_durations.get(regime_type, 60)
    
    def _calculate_breadth(self, data: pd.DataFrame) -> float:
        """Calculate market breadth indicator"""
        # Simplified - in production use advance/decline data
        returns = data['close'].pct_change()
        positive_days = (returns > 0).sum()
        total_days = len(returns.dropna())
        
        return positive_days / total_days if total_days > 0 else 0.5
    
    def _calculate_vix_proxy(self, data: pd.DataFrame) -> float:
        """Calculate VIX proxy using price data"""
        # Simplified VIX proxy using realized volatility
        returns = data['close'].pct_change()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        return volatility * 100  # Convert to percentage
    
    def get_regime_context(self) -> Dict[str, Any]:
        """Get current regime context for trading decisions"""
        if not self.current_regime:
            return {"regime": "unknown", "adjustments": {}}
        
        regime = self.current_regime
        
        # Define regime-specific adjustments
        adjustments = {
            "bull_quiet": {
                "position_size": 1.2,
                "stop_loss": 0.02,
                "take_profit": 0.05,
                "holding_period": "medium"
            },
            "bull_volatile": {
                "position_size": 0.8,
                "stop_loss": 0.03,
                "take_profit": 0.08,
                "holding_period": "short"
            },
            "bear_quiet": {
                "position_size": 0.6,
                "stop_loss": 0.015,
                "take_profit": 0.03,
                "holding_period": "short"
            },
            "bear_volatile": {
                "position_size": 0.4,
                "stop_loss": 0.02,
                "take_profit": 0.04,
                "holding_period": "very_short"
            },
            "sideways": {
                "position_size": 0.8,
                "stop_loss": 0.02,
                "take_profit": 0.03,
                "holding_period": "short"
            },
            "transition": {
                "position_size": 0.5,
                "stop_loss": 0.025,
                "take_profit": 0.04,
                "holding_period": "very_short"
            }
        }
        
        return {
            "regime": regime.regime_type,
            "confidence": regime.confidence,
            "trend": regime.trend,
            "volatility": regime.volatility,
            "adjustments": adjustments.get(regime.regime_type, {}),
            "characteristics": regime.characteristics,
            "duration_estimate": regime.expected_duration
        }
    
    async def predict_regime_change(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Predict potential regime changes"""
        current = self.current_regime
        if not current:
            return {"change_probability": 0, "next_regime": None}
        
        # Calculate regime stability indicators
        indicators = self._calculate_indicators(data)
        current_score = self._score_regime(indicators, 
                                         self.regime_definitions[current.regime_type]["characteristics"])
        
        # Check scores for other regimes
        alternative_scores = {}
        for regime_type, definition in self.regime_definitions.items():
            if regime_type != current.regime_type:
                score = self._score_regime(indicators, definition["characteristics"])
                alternative_scores[regime_type] = score
        
        # Find best alternative
        best_alternative = max(alternative_scores.items(), key=lambda x: x[1])
        
        # Calculate change probability
        change_probability = 0.0
        if best_alternative[1] > current_score:
            change_probability = (best_alternative[1] - current_score) / current_score
        
        # Time-based probability adjustment
        if current.start_date:
            days_in_regime = (datetime.now() - current.start_date).days
            if days_in_regime > current.expected_duration:
                change_probability *= 1.5
        
        return {
            "change_probability": min(change_probability, 1.0),
            "next_regime": best_alternative[0] if change_probability > 0.3 else None,
            "confidence": best_alternative[1] if change_probability > 0.3 else 0,
            "trigger_indicators": self._identify_change_triggers(indicators, current)
        }
    
    def _identify_change_triggers(self, indicators: Dict[str, float], 
                                 current_regime: MarketRegime) -> List[str]:
        """Identify what might trigger regime change"""
        triggers = []
        
        # Check volatility changes
        current_vol = current_regime.characteristics.get("volatility", 0)
        if abs(indicators["volatility"] - current_vol) / current_vol > 0.3:
            triggers.append("volatility_shift")
        
        # Check trend changes
        if indicators["trend_strength"] < 25:  # Weak trend
            triggers.append("trend_weakening")
        
        # Check extreme RSI
        if indicators["rsi"] > 70 or indicators["rsi"] < 30:
            triggers.append("overbought_oversold")
        
        # Check volume anomalies
        if indicators["volume"] > 1.5 or indicators["volume"] < 0.5:
            triggers.append("volume_anomaly")
        
        return triggers
