"""
RSI (Relative Strength Index) technical analysis agent with advanced features.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.ml.models.market_data import MarketData
from src.ml.models.signals import Signal, SignalSource, SignalStrength, SignalType

from ....base import AgentConfig, BaseAgent

logger = logging.getLogger(__name__)

class RSIAgent(BaseAgent):
    """Agent that generates trading signals based on RSI with trend-adjusted thresholds."""

    def __init__(
        self,
        config: AgentConfig,
        db_manager,
        redis_manager,
        period: int = 14,
        overbought: float = 70,
        oversold: float = 30,
        trend_factor: bool = True
    ):
        """
        Initialize RSI agent.

        Args:
            config: Agent configuration
            db_manager: Database manager
            redis_manager: Redis manager
            period: RSI calculation period
            overbought: Overbought threshold
            oversold: Oversold threshold
            trend_factor: Whether to adjust thresholds based on trend
        """
        super().__init__(config=config, db_manager=db_manager, redis_manager=redis_manager)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        self.trend_factor = trend_factor

    def calculate_rsi(self, prices: pd.Series) -> Optional[float]:
        """Calculate RSI with trend-adjusted thresholds."""
        try:
            if len(prices) < self.period + 1:
                return None

            # Calculate returns and separate gains/losses
            deltas = prices.diff()
            gains = deltas.where(deltas > 0, 0.0)
            losses = -deltas.where(deltas < 0, 0.0)

            # Calculate average gains and losses
            avg_gain = gains.rolling(window=self.period).mean()
            avg_loss = losses.rolling(window=self.period).mean()

            # Calculate RS and RSI
            # Handle the case where avg_loss is 0 (all gains)
            if avg_loss.iloc[-1] == 0:
                if avg_gain.iloc[-1] > 0:
                    rsi_value = 100.0  # All gains = RSI 100
                else:
                    rsi_value = 50.0   # No movement = RSI 50
            else:
                rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
                rsi_value = 100 - (100 / (1 + rs))

            # Adjust thresholds based on trend if enabled
            if self.trend_factor:
                trend = (prices.iloc[-1] / prices.iloc[-self.period] - 1) * 100
                self.overbought = min(80, 70 + abs(trend))
                self.oversold = max(20, 30 - abs(trend))

            return float(rsi_value)

        except Exception as e:
            logger.error(f"RSI calculation failed: {str(e)}")
            return None

    async def analyze(self, market_data: MarketData) -> Signal:
        """
        Analyze market data and generate RSI-based trading signal.

        Args:
            market_data: Market data for analysis

        Returns:
            Signal: Trading signal with RSI analysis
        """
        try:
            # Extract close prices from market data
            prices = None

            # Check if market_data has close_prices attribute directly
            if hasattr(market_data, 'close_prices'):
                prices = pd.Series(market_data.close_prices)
            # Check if it's in a data dict (for test compatibility)
            elif hasattr(market_data, 'data') and isinstance(market_data.data, dict) and 'close_prices' in market_data.data:
                prices = pd.Series(market_data.data['close_prices'])
            # Check OHLCV data
            elif hasattr(market_data, 'ohlcv_1h') and market_data.ohlcv_1h:
                prices = pd.Series([bar.close for bar in market_data.ohlcv_1h])
            elif hasattr(market_data, 'ohlcv_1d') and market_data.ohlcv_1d:
                prices = pd.Series([bar.close for bar in market_data.ohlcv_1d])
            else:
                # Try to get from historical data
                historical = await self.get_historical_market_data(
                    market_data.symbol,
                    timeframe="1h",
                    limit=self.period + 10
                )
                if historical:
                    prices = pd.Series([d['close'] for d in historical])
                else:
                    # Use current price as a single data point
                    prices = pd.Series([market_data.current_price])

            rsi = self.calculate_rsi(prices)

            if rsi is None:
                # Return neutral signal if RSI can't be calculated
                return Signal(
                    symbol=market_data.symbol,
                    signal_type=SignalType.HOLD,
                    confidence=0.0,
                    strength=SignalStrength.WEAK,
                    source=SignalSource.TECHNICAL_ANALYSIS,
                    current_price=market_data.current_price,
                    reasoning="Insufficient data for RSI calculation",
                    features={"rsi": None, "error": "Insufficient data"}
                )

            # Determine signal based on RSI
            current_price = float(prices.iloc[-1])

            if rsi > self.overbought:
                signal_type = SignalType.SELL
                confidence = min((rsi - self.overbought) / (100 - self.overbought), 1.0)
                strength = SignalStrength.STRONG if confidence > 0.8 else SignalStrength.MEDIUM
                reasoning = f"RSI {rsi:.1f} indicates overbought condition (>{self.overbought})"
            elif rsi < self.oversold:
                signal_type = SignalType.BUY
                confidence = min((self.oversold - rsi) / self.oversold, 1.0)
                strength = SignalStrength.STRONG if confidence > 0.8 else SignalStrength.MEDIUM
                reasoning = f"RSI {rsi:.1f} indicates oversold condition (<{self.oversold})"
            else:
                signal_type = SignalType.HOLD
                mid_point = (self.overbought + self.oversold) / 2
                confidence = 1.0 - abs(rsi - mid_point) / (mid_point - self.oversold)
                strength = SignalStrength.WEAK
                reasoning = f"RSI {rsi:.1f} in neutral zone"

            # Adjust confidence based on trend consistency
            if len(prices) >= self.period:
                trend = prices.pct_change(self.period).iloc[-1]
                if (signal_type == SignalType.BUY and trend > 0) or \
                   (signal_type == SignalType.SELL and trend < 0):
                    confidence *= 0.8  # Reduce confidence when against trend

            # Calculate target and stop loss
            atr = self._calculate_atr(prices)
            if signal_type == SignalType.BUY:
                target_price = current_price * (1 + 0.02)  # 2% target
                stop_loss = current_price * (1 - 0.01)     # 1% stop
            elif signal_type == SignalType.SELL:
                target_price = current_price * (1 - 0.02)  # 2% target
                stop_loss = current_price * (1 + 0.01)     # 1% stop
            else:
                target_price = current_price
                stop_loss = current_price

            return Signal(
                symbol=market_data.symbol,
                signal_type=signal_type,
                confidence=max(min(confidence, 1.0), 0.0),
                strength=strength,
                source=SignalSource.TECHNICAL_ANALYSIS,
                current_price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                reasoning=reasoning,
                features={
                    "rsi": rsi,
                    "period": self.period,
                    "overbought": self.overbought,
                    "oversold": self.oversold,
                    "trend_adjusted": self.trend_factor
                },
                indicators={"rsi": rsi}
            )

        except Exception as e:
            logger.error(f"RSI analysis failed: {str(e)}")
            return Signal(
                symbol=market_data.symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                strength=SignalStrength.WEAK,
                source=SignalSource.TECHNICAL_ANALYSIS,
                current_price=market_data.current_price if hasattr(market_data, 'current_price') else 0.0,
                reasoning=f"Analysis failed: {str(e)}",
                features={"error": str(e)}
            )

    def get_required_data_types(self) -> List[str]:
        """
        Returns list of required data types for RSI analysis.

        Returns:
            List of data type strings
        """
        return ["price", "close_prices", "historical_prices"]

    def _calculate_atr(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range for position sizing."""
        try:
            if len(prices) < period + 1:
                return prices.std() * 0.02  # Fallback to 2% of std dev

            high = prices.rolling(window=2).max()
            low = prices.rolling(window=2).min()
            close_prev = prices.shift(1)

            tr1 = high - low
            tr2 = abs(high - close_prev)
            tr3 = abs(low - close_prev)

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean().iloc[-1]

            return float(atr) if not pd.isna(atr) else prices.std() * 0.02
        except:
            return prices.std() * 0.02

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and generate RSI signals with confidence levels."""
        try:
            if "close_prices" not in data:
                raise ValueError("Close prices not found in market data")

            prices = pd.Series(data["close_prices"])
            rsi = self.calculate_rsi(prices)

            if rsi is None:
                return {
                    "action": "hold",
                    "confidence": 0.0,
                    "metadata": {
                        "rsi": None,
                        "error": "Insufficient data for RSI calculation"
                    }
                }

            # Calculate confidence based on RSI distance from thresholds
            if rsi > self.overbought:
                action = "sell"
                confidence = min((rsi - self.overbought) / (100 - self.overbought), 1.0)
            elif rsi < self.oversold:
                action = "buy"
                confidence = min((self.oversold - rsi) / self.oversold, 1.0)
            else:
                action = "hold"
                # Confidence decreases as RSI approaches middle range
                mid_point = (self.overbought + self.oversold) / 2
                confidence = 1.0 - abs(rsi - mid_point) / (mid_point - self.oversold)

            # Adjust confidence based on trend consistency
            if len(prices) >= self.period:
                trend = prices.pct_change(self.period).iloc[-1]
                if (action == "buy" and trend > 0) or (action == "sell" and trend < 0):
                    confidence *= 0.8  # Reduce confidence when against trend

            return {
                "action": action,
                "confidence": max(min(confidence, 1.0), 0.0),
                "metadata": {
                    "rsi": rsi,
                    "period": self.period,
                    "overbought": self.overbought,
                    "oversold": self.oversold,
                    "trend_adjusted": self.trend_factor
                }
            }

        except Exception as e:
            logger.error(f"RSI signal processing failed: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "metadata": {"error": str(e)}
            }

    def process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Process and potentially modify a trading signal."""
        # Default implementation: return signal as-is
        return signal
