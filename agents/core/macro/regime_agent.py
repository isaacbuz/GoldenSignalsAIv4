"""
Regime Agent - Detects and analyzes market regimes (bull/bear/sideways) for trading signals.
Uses multiple indicators to classify market conditions and regime changes.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from agents.common.base.base_agent import BaseAgent
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class RegimeAgent(BaseAgent):
    """Agent that detects market regimes and generates regime-based trading signals."""

    def __init__(
        self,
        name: str = "Regime",
        lookback_period: int = 252,  # 1 year
        regime_change_threshold: float = 0.7,
        trend_strength_threshold: float = 0.6,
        volatility_regime_threshold: float = 1.5,
        correlation_window: int = 60,
        min_regime_duration: int = 10
    ):
        """
        Initialize Regime agent.

        Args:
            name: Agent name
            lookback_period: Period for regime analysis
            regime_change_threshold: Threshold for regime change detection
            trend_strength_threshold: Minimum trend strength for directional regime
            volatility_regime_threshold: Volatility threshold for regime classification
            correlation_window: Window for correlation analysis
            min_regime_duration: Minimum bars for regime persistence
        """
        super().__init__(name=name, agent_type="regime")
        self.lookback_period = lookback_period
        self.regime_change_threshold = regime_change_threshold
        self.trend_strength_threshold = trend_strength_threshold
        self.volatility_regime_threshold = volatility_regime_threshold
        self.correlation_window = correlation_window
        self.min_regime_duration = min_regime_duration

    def calculate_trend_indicators(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate various trend strength indicators."""
        try:
            if len(prices) < 50:
                return {}

            indicators = {}

            # Moving average slopes
            for period in [20, 50, 100, 200]:
                if len(prices) >= period:
                    ma = prices.rolling(window=period).mean()
                    if len(ma.dropna()) >= 5:
                        # Calculate slope of moving average
                        y = ma.dropna().tail(period//4).values  # Use last quarter of the MA
                        x = np.arange(len(y))
                        if len(y) > 1:
                            slope, _, r_value, _, _ = stats.linregress(x, y)
                            indicators[f'ma_{period}_slope'] = slope
                            indicators[f'ma_{period}_r_squared'] = r_value ** 2

            # Price position relative to moving averages
            for period in [20, 50, 200]:
                if len(prices) >= period:
                    ma = prices.rolling(window=period).mean()
                    current_price = prices.iloc[-1]
                    current_ma = ma.iloc[-1]

                    if current_ma > 0:
                        indicators[f'price_vs_ma_{period}'] = (current_price - current_ma) / current_ma

            # Trend strength using ADX-like calculation
            if len(prices) >= 20:
                high_low_range = abs(prices.diff())
                directional_movement_up = np.where(prices.diff() > 0, prices.diff(), 0)
                directional_movement_down = np.where(prices.diff() < 0, abs(prices.diff()), 0)

                dm_up_smooth = pd.Series(directional_movement_up).rolling(window=14).mean()
                dm_down_smooth = pd.Series(directional_movement_down).rolling(window=14).mean()
                tr_smooth = high_low_range.rolling(window=14).mean()

                di_up = 100 * dm_up_smooth / tr_smooth
                di_down = 100 * dm_down_smooth / tr_smooth

                dx = 100 * abs(di_up - di_down) / (di_up + di_down)
                adx = dx.rolling(window=14).mean()

                if not adx.isna().iloc[-1]:
                    indicators['trend_strength'] = adx.iloc[-1] / 100.0

            return indicators

        except Exception as e:
            logger.error(f"Trend indicators calculation failed: {str(e)}")
            return {}

    def calculate_volatility_regime(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate volatility regime indicators."""
        try:
            if len(returns) < 20:
                return {}

            # Rolling volatilities
            vol_windows = [10, 20, 60]
            volatilities = {}

            for window in vol_windows:
                if len(returns) >= window:
                    vol = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
                    volatilities[f'vol_{window}d'] = vol.iloc[-1]

            # Volatility regime classification
            if 'vol_20d' in volatilities and 'vol_60d' in volatilities:
                vol_ratio = volatilities['vol_20d'] / volatilities['vol_60d']

                if vol_ratio > self.volatility_regime_threshold:
                    vol_regime = 'high_volatility'
                elif vol_ratio < (1 / self.volatility_regime_threshold):
                    vol_regime = 'low_volatility'
                else:
                    vol_regime = 'normal_volatility'
            else:
                vol_regime = 'unknown'

            # Volatility clustering
            abs_returns = abs(returns)
            if len(abs_returns) >= 20:
                clustering = abs_returns.rolling(window=5).mean().rolling(window=10).std()
                vol_clustering = clustering.iloc[-1] if not clustering.isna().iloc[-1] else 0
            else:
                vol_clustering = 0

            return {
                'volatilities': volatilities,
                'vol_regime': vol_regime,
                'vol_ratio': vol_ratio if 'vol_ratio' in locals() else 1.0,
                'vol_clustering': vol_clustering
            }

        except Exception as e:
            logger.error(f"Volatility regime calculation failed: {str(e)}")
            return {}

    def calculate_market_breadth(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate market breadth indicators if sector/index data available."""
        try:
            breadth_indicators = {}

            # If we have sector performance data
            sector_data = market_data.get('sector_performance', {})
            if sector_data:
                sector_returns = list(sector_data.values())
                if sector_returns:
                    # Percentage of sectors with positive returns
                    positive_sectors = sum(1 for r in sector_returns if r > 0)
                    breadth_indicators['positive_sector_pct'] = positive_sectors / len(sector_returns)

                    # Average sector performance
                    breadth_indicators['avg_sector_performance'] = np.mean(sector_returns)

                    # Sector performance dispersion
                    breadth_indicators['sector_dispersion'] = np.std(sector_returns)

            # Advance-Decline ratio simulation (if not available)
            # This would normally come from market breadth data
            breadth_indicators['advance_decline_ratio'] = 0.55  # Placeholder

            return breadth_indicators

        except Exception as e:
            logger.error(f"Market breadth calculation failed: {str(e)}")
            return {}

    def detect_regime_using_hmm(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Detect regimes using Hidden Markov Model approach with Gaussian Mixture."""
        try:
            if len(features) < 30 or features.empty:
                return {'regime': 'unknown', 'regime_probability': 0.0}

            # Prepare features for regime detection
            feature_cols = ['returns', 'volatility', 'trend_strength']
            available_features = []

            for col in feature_cols:
                if col in features.columns and not features[col].isna().all():
                    available_features.append(col)

            if not available_features:
                return {'regime': 'unknown', 'regime_probability': 0.0}

            # Use available features
            X = features[available_features].dropna()

            if len(X) < 10:
                return {'regime': 'unknown', 'regime_probability': 0.0}

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Fit Gaussian Mixture Model with 3 components (bull, bear, sideways)
            gmm = GaussianMixture(n_components=3, random_state=42, max_iter=100)
            gmm.fit(X_scaled)

            # Predict current regime
            current_features = X_scaled[-1].reshape(1, -1)
            regime_probs = gmm.predict_proba(current_features)[0]
            current_regime_idx = np.argmax(regime_probs)
            max_prob = regime_probs[current_regime_idx]

            # Map regime index to regime name based on characteristics
            regime_means = gmm.means_

            # Analyze regime characteristics
            regimes = []
            for i, mean in enumerate(regime_means):
                # Assume first feature is returns, second is volatility
                avg_return = mean[0] if len(mean) > 0 else 0
                avg_volatility = mean[1] if len(mean) > 1 else 0

                if avg_return > 0.1 and avg_volatility < 0:
                    regime_type = 'bull'
                elif avg_return < -0.1 and avg_volatility > 0:
                    regime_type = 'bear'
                else:
                    regime_type = 'sideways'

                regimes.append(regime_type)

            current_regime = regimes[current_regime_idx]

            return {
                'regime': current_regime,
                'regime_probability': max_prob,
                'all_regime_probs': {
                    regimes[i]: regime_probs[i] for i in range(len(regimes))
                },
                'regime_stability': max_prob  # Higher probability = more stable regime
            }

        except Exception as e:
            logger.error(f"HMM regime detection failed: {str(e)}")
            return {'regime': 'unknown', 'regime_probability': 0.0}

    def classify_regime_simple(self, trend_indicators: Dict[str, float], vol_regime: Dict[str, Any], returns: pd.Series) -> Dict[str, Any]:
        """Simple rule-based regime classification."""
        try:
            if len(returns) < 20:
                return {'regime': 'unknown', 'confidence': 0.0}

            # Calculate recent performance
            recent_return = returns.tail(20).mean() * 252  # Annualized
            trend_strength = trend_indicators.get('trend_strength', 0)
            vol_regime_type = vol_regime.get('vol_regime', 'normal_volatility')

            # Price vs moving averages
            price_vs_ma_20 = trend_indicators.get('price_vs_ma_20', 0)
            price_vs_ma_50 = trend_indicators.get('price_vs_ma_50', 0)
            price_vs_ma_200 = trend_indicators.get('price_vs_ma_200', 0)

            # Regime classification logic
            bull_signals = 0
            bear_signals = 0
            sideways_signals = 0

            # Return-based signals
            if recent_return > 0.10:  # >10% annualized
                bull_signals += 2
            elif recent_return < -0.10:  # <-10% annualized
                bear_signals += 2
            else:
                sideways_signals += 1

            # Trend strength signals
            if trend_strength > self.trend_strength_threshold:
                if recent_return > 0:
                    bull_signals += 2
                else:
                    bear_signals += 2
            else:
                sideways_signals += 1

            # Moving average signals
            ma_signals = [price_vs_ma_20, price_vs_ma_50, price_vs_ma_200]
            positive_ma = sum(1 for sig in ma_signals if sig > 0.02)  # >2% above MA
            negative_ma = sum(1 for sig in ma_signals if sig < -0.02)  # >2% below MA

            if positive_ma >= 2:
                bull_signals += positive_ma
            elif negative_ma >= 2:
                bear_signals += negative_ma
            else:
                sideways_signals += 1

            # Volatility regime adjustment
            if vol_regime_type == 'high_volatility':
                bear_signals += 1  # High vol often associated with bear markets
            elif vol_regime_type == 'low_volatility':
                bull_signals += 1  # Low vol often associated with bull markets

            # Determine regime
            total_signals = bull_signals + bear_signals + sideways_signals

            if bull_signals > bear_signals and bull_signals > sideways_signals:
                regime = 'bull'
                confidence = bull_signals / total_signals
            elif bear_signals > bull_signals and bear_signals > sideways_signals:
                regime = 'bear'
                confidence = bear_signals / total_signals
            else:
                regime = 'sideways'
                confidence = sideways_signals / total_signals

            return {
                'regime': regime,
                'confidence': confidence,
                'bull_signals': bull_signals,
                'bear_signals': bear_signals,
                'sideways_signals': sideways_signals,
                'recent_return_annualized': recent_return,
                'trend_strength': trend_strength
            }

        except Exception as e:
            logger.error(f"Simple regime classification failed: {str(e)}")
            return {'regime': 'unknown', 'confidence': 0.0}

    def detect_regime_changes(self, historical_regimes: List[str], current_regime: str) -> Dict[str, Any]:
        """Detect regime changes and their characteristics."""
        try:
            if not historical_regimes or len(historical_regimes) < self.min_regime_duration:
                return {'regime_change': False, 'regime_stability': 'unknown'}

            # Check for recent regime changes
            recent_regimes = historical_regimes[-self.min_regime_duration:]

            # Count regime consistency
            regime_counts = {}
            for regime in recent_regimes:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1

            most_common_regime = max(regime_counts, key=regime_counts.get)
            regime_stability = regime_counts[most_common_regime] / len(recent_regimes)

            # Detect regime change
            if len(historical_regimes) >= 2:
                previous_regime = historical_regimes[-2]
                regime_change = current_regime != previous_regime

                # Count recent changes
                changes_in_window = 0
                for i in range(1, min(len(historical_regimes), 20)):
                    if historical_regimes[-i] != historical_regimes[-i-1]:
                        changes_in_window += 1
            else:
                regime_change = False
                changes_in_window = 0

            # Stability classification
            if regime_stability > 0.8:
                stability_class = 'stable'
            elif regime_stability > 0.6:
                stability_class = 'moderate'
            else:
                stability_class = 'unstable'

            return {
                'regime_change': regime_change,
                'regime_stability': regime_stability,
                'stability_class': stability_class,
                'recent_changes': changes_in_window,
                'dominant_recent_regime': most_common_regime
            }

        except Exception as e:
            logger.error(f"Regime change detection failed: {str(e)}")
            return {'regime_change': False, 'regime_stability': 'unknown'}

    def generate_regime_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on regime analysis."""
        try:
            prices = pd.Series(data.get('prices', []))
            market_data = data.get('market_data', {})
            historical_regimes = data.get('historical_regimes', [])

            if len(prices) < 50:
                return {
                    'action': 'hold',
                    'confidence': 0.0,
                    'signal_type': 'insufficient_data',
                    'reasoning': 'Insufficient price data for regime analysis'
                }

            # Calculate returns
            returns = prices.pct_change().dropna()

            # Calculate indicators
            trend_indicators = self.calculate_trend_indicators(prices)
            vol_regime = self.calculate_volatility_regime(returns)
            breadth_indicators = self.calculate_market_breadth(market_data)

            # Classify regime
            regime_classification = self.classify_regime_simple(trend_indicators, vol_regime, returns)

            # Detect regime changes
            current_regime = regime_classification['regime']
            regime_change_analysis = self.detect_regime_changes(historical_regimes, current_regime)

            # Generate signals
            action = "hold"
            confidence = 0.0
            signal_type = f"{current_regime}_regime"
            reasoning = []

            # Base signal from regime
            if current_regime == 'bull':
                action = "buy"
                confidence += 0.5
                reasoning.append(f"Bull regime detected (confidence: {regime_classification['confidence']:.2f})")
            elif current_regime == 'bear':
                action = "sell"
                confidence += 0.5
                reasoning.append(f"Bear regime detected (confidence: {regime_classification['confidence']:.2f})")
            else:
                action = "hold"
                confidence += 0.2
                reasoning.append(f"Sideways regime detected")

            # Regime change signals
            if regime_change_analysis.get('regime_change'):
                confidence += 0.3
                reasoning.append("Recent regime change detected")
                signal_type = f"regime_change_to_{current_regime}"

            # Stability adjustments
            stability = regime_change_analysis.get('regime_stability', 0)
            if stability > 0.8:
                confidence *= 1.2  # Boost confidence for stable regimes
                reasoning.append("High regime stability")
            elif stability < 0.4:
                confidence *= 0.7  # Reduce confidence for unstable regimes
                reasoning.append("Low regime stability - high uncertainty")

            # Volatility regime adjustments
            vol_regime_type = vol_regime.get('vol_regime', 'normal')
            if vol_regime_type == 'high_volatility':
                confidence *= 0.8  # Reduce confidence in high vol environments
                reasoning.append("High volatility regime")
            elif vol_regime_type == 'low_volatility':
                confidence *= 1.1  # Boost confidence in low vol environments
                reasoning.append("Low volatility regime")

            # Trend strength confirmation
            trend_strength = trend_indicators.get('trend_strength', 0)
            if trend_strength > self.trend_strength_threshold:
                confidence *= 1.1
                reasoning.append(f"Strong trend detected (strength: {trend_strength:.2f})")

            return {
                'action': action,
                'confidence': min(1.0, confidence),
                'signal_type': signal_type,
                'reasoning': reasoning,
                'current_regime': current_regime
            }

        except Exception as e:
            logger.error(f"Regime signal generation failed: {str(e)}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'signal_type': 'error',
                'reasoning': [str(e)]
            }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and generate regime-based signals."""
        try:
            if "prices" not in data:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {"error": "Price data not provided"}
                }

            # Generate signals
            signal_data = self.generate_regime_signals(data)

            # Get comprehensive analysis
            prices = pd.Series(data["prices"])
            returns = prices.pct_change().dropna()
            market_data = data.get("market_data", {})
            historical_regimes = data.get("historical_regimes", [])

            trend_indicators = self.calculate_trend_indicators(prices)
            vol_regime = self.calculate_volatility_regime(returns)
            breadth_indicators = self.calculate_market_breadth(market_data)
            regime_classification = self.classify_regime_simple(trend_indicators, vol_regime, returns)
            regime_change_analysis = self.detect_regime_changes(historical_regimes, regime_classification['regime'])

            return {
                "action": signal_data['action'],
                "confidence": signal_data['confidence'],
                "metadata": {
                    "signal_type": signal_data['signal_type'],
                    "reasoning": signal_data['reasoning'],
                    "current_regime": signal_data['current_regime'],
                    "regime_classification": regime_classification,
                    "trend_indicators": trend_indicators,
                    "volatility_regime": vol_regime,
                    "breadth_indicators": breadth_indicators,
                    "regime_change_analysis": regime_change_analysis
                }
            }

        except Exception as e:
            logger.error(f"Regime signal processing failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }
