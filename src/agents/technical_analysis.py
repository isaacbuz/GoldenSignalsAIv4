"""
Technical Analysis Agent - GoldenSignalsAI V3

Performs technical analysis using various indicators like RSI, MACD, 
Bollinger Bands, Moving Averages, and trend analysis.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseAgent, AgentConfig
from ..models.signals import Signal, SignalType, SignalStrength, SignalSource
from ..models.market_data import MarketData


# -------------------------------------------------------------
# TechnicalAnalysisAgent
# -------------------------------------------------------------

class TechnicalAnalysisAgent(BaseAgent):
    """
    Agent specializing in technical analysis and chart patterns
    """
    
    def __init__(self, name: str, db_manager, redis_manager):
        config = AgentConfig(
            name=name,
            version="1.0.0",
            enabled=True,
            weight=0.25,
            confidence_threshold=0.7,
            timeout=30,
            max_retries=3,
            learning_rate=0.01
        )
        super().__init__(config, db_manager, redis_manager)
        self.indicators = {
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "bb_std": 2,
            "sma_short": 20,
            "sma_long": 50
        }
    
    def get_required_data_types(self) -> List[str]:
        """
        Returns list of required data types for this agent
        
        Returns:
            List of data type strings (e.g., ['price', 'volume', 'news'])
        """
        return ['price', 'volume', 'ohlcv']
    
    # ------------------------------------------------------------------
    # Public API required by BaseAgent (signature: analyze(market_data))
    # ------------------------------------------------------------------

    async def analyze(self, *args, **kwargs) -> Optional[Signal]:  # type: ignore[override]
        """Adapter wrapper so the abstract signature is satisfied.

        BaseAgent expects :py:meth:`analyze(market_data)`; our real analysis
        method needs the *symbol*, *market_data* dict, and *historical_data*.
        This wrapper extracts what we need from the `MarketData` model and then
        delegates to the internal ``_analyze_extended`` method.
        """

        # Accept both the original orchestrator-style call and the
        # BaseAgent.execute_with_monitoring call.

        symbol: Optional[str] = None
        market_data: Optional[Any] = None
        historical_data: Optional[List[Dict[str, Any]]] = None

        # Positional args handling
        if len(args) == 1:
            # Called via BaseAgent.execute_with_monitoring(market_data)
            market_data = args[0]
            symbol = getattr(market_data, "symbol", None)
        elif len(args) >= 3:
            # Called via orchestrator.analyze(symbol, market_data_dict, historical_data)
            symbol = args[0]
            market_data = args[1]
            historical_data = args[2]
        else:
            # Fallback to kwargs
            symbol = kwargs.get("symbol")
            market_data = kwargs.get("market_data")
            historical_data = kwargs.get("historical_data")

        if symbol is None or market_data is None:
            logger.error("TechnicalAnalysisAgent.analyze missing required arguments")
            return None

        # If market_data is provided as pydantic model, convert to dict
        if isinstance(market_data, MarketData):
            md_dict: Dict[str, Any] = {
                "price": market_data.current_price,
                "volume": getattr(market_data, "volume", None),
            }
            historical_data = historical_data or getattr(market_data, "ohlcv_1h", [])
        else:
            md_dict = market_data  # assume dict-like

        historical_data = historical_data or []

        return await self._analyze_extended(symbol, md_dict, historical_data)

    # ------------------------------------------------------------------
    # Internal extended analysis (called by orchestrator directly as well)
    # ------------------------------------------------------------------

    async def _analyze_extended(self, symbol: str, market_data: Dict[str, Any], historical_data: List[Dict[str, Any]]) -> Optional[Signal]:
        """
        Perform technical analysis on the given market data
        
        Args:
            symbol: Symbol of the market data
            market_data: Current market data for analysis
            historical_data: List of historical data points
            
        Returns:
            Technical analysis signal
        """
        try:
            # symbol is already provided
            if not historical_data or len(historical_data) < 50:
                logger.warning(f"Insufficient historical data for {symbol} technical analysis")
                return None

            # If current_price not in market_data dict, fallback to last close
            current_price = market_data.get("price") if isinstance(market_data, dict) else getattr(market_data, "current_price", None)

            # Convert to DataFrame for analysis
            df = self._prepare_dataframe(historical_data)
            
            # Calculate technical indicators
            indicators = await self._calculate_indicators(df)
            
            # Generate signal based on indicators
            signal = await self._generate_signal(symbol, current_price or float(df['close'].iloc[-1]), indicators)
            
            return signal
            
        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}: {str(e)}")
            return None
    
    def _prepare_dataframe(self, historical_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert historical data to pandas DataFrame"""
        try:
            # Assuming historical data has OHLCV format
            data = []
            for point in historical_data:
                data.append({
                    'timestamp': point.get('timestamp', datetime.utcnow()),
                    'open': float(point.get('open', point.get('price', 0))),
                    'high': float(point.get('high', point.get('price', 0))),
                    'low': float(point.get('low', point.get('price', 0))),
                    'close': float(point.get('close', point.get('price', 0))),
                    'volume': int(point.get('volume', 0))
                })
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to prepare DataFrame: {str(e)}")
            raise
    
    async def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators"""
        try:
            indicators = {}
            
            # RSI (Relative Strength Index)
            indicators['rsi'] = self._calculate_rsi(df['close'], self.indicators['rsi_period'])
            
            # MACD
            macd_data = self._calculate_macd(
                df['close'],
                self.indicators['macd_fast'],
                self.indicators['macd_slow'],
                self.indicators['macd_signal']
            )
            indicators.update(macd_data)
            
            # Bollinger Bands
            bb_data = self._calculate_bollinger_bands(
                df['close'],
                self.indicators['bb_period'],
                self.indicators['bb_std']
            )
            indicators.update(bb_data)
            
            # Moving Averages
            indicators['sma_short'] = df['close'].rolling(window=self.indicators['sma_short']).mean()
            indicators['sma_long'] = df['close'].rolling(window=self.indicators['sma_long']).mean()
            
            # Price vs Moving Averages
            indicators['price_vs_sma_short'] = df['close'] / indicators['sma_short']
            indicators['price_vs_sma_long'] = df['close'] / indicators['sma_long']
            
            # Volume indicators
            indicators['volume_sma'] = df['volume'].rolling(window=20).mean()
            indicators['volume_ratio'] = df['volume'] / indicators['volume_sma']
            
            # Trend indicators
            indicators['trend_short'] = self._calculate_trend(df['close'], 5)
            indicators['trend_medium'] = self._calculate_trend(df['close'], 20)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Failed to calculate indicators: {str(e)}")
            raise
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"Failed to calculate RSI: {str(e)}")
            return pd.Series(index=prices.index, dtype=float)
    
    def _calculate_macd(self, prices: pd.Series, fast: int, slow: int, signal: int) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return {
                'macd_line': macd_line,
                'macd_signal': signal_line,
                'macd_histogram': histogram
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate MACD: {str(e)}")
            return {
                'macd_line': pd.Series(index=prices.index, dtype=float),
                'macd_signal': pd.Series(index=prices.index, dtype=float),
                'macd_histogram': pd.Series(index=prices.index, dtype=float)
            }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int, std_dev: float) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            rolling_mean = prices.rolling(window=period).mean()
            rolling_std = prices.rolling(window=period).std()
            
            upper_band = rolling_mean + (rolling_std * std_dev)
            lower_band = rolling_mean - (rolling_std * std_dev)
            
            # Calculate position within bands
            bb_position = (prices - lower_band) / (upper_band - lower_band)
            
            return {
                'bb_upper': upper_band,
                'bb_middle': rolling_mean,
                'bb_lower': lower_band,
                'bb_position': bb_position,
                'bb_width': (upper_band - lower_band) / rolling_mean
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate Bollinger Bands: {str(e)}")
            return {
                'bb_upper': pd.Series(index=prices.index, dtype=float),
                'bb_middle': pd.Series(index=prices.index, dtype=float),
                'bb_lower': pd.Series(index=prices.index, dtype=float),
                'bb_position': pd.Series(index=prices.index, dtype=float),
                'bb_width': pd.Series(index=prices.index, dtype=float)
            }
    
    def _calculate_trend(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate trend direction over given period"""
        try:
            # Linear regression slope over period
            def slope(series):
                if len(series) < 2:
                    return 0
                x = np.arange(len(series))
                y = series.values
                return np.polyfit(x, y, 1)[0]
            
            return prices.rolling(window=period).apply(slope, raw=False)
            
        except Exception as e:
            logger.error(f"Failed to calculate trend: {str(e)}")
            return pd.Series(index=prices.index, dtype=float)
    
    async def _generate_signal(
        self,
        symbol: str,
        current_price: float,
        indicators: Dict[str, Any]
    ) -> Optional[Signal]:
        """Generate trading signal based on technical indicators"""
        try:
            # Get latest indicator values
            latest = {k: v.iloc[-1] if isinstance(v, pd.Series) else v 
                     for k, v in indicators.items()}
            
            # Calculate signal type and confidence
            signal_type, confidence = self._determine_signal_type(latest)
            
            if signal_type == SignalType.HOLD:
                return None
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                signal_type,
                latest.get('rsi', 50),
                latest.get('macd_line', 0),
                latest.get('bb_position', 0.5),
                latest.get('trend_medium', 0),
                latest.get('price_vs_sma_long', 1.0)
            )
            
            # Create signal
            signal = Signal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                source=SignalSource.TECHNICAL_ANALYSIS,
                current_price=current_price,
                reasoning=reasoning,
                indicators=latest,
                time_horizon=60  # 1 hour horizon for technical signals
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Failed to generate signal: {str(e)}")
            return None
    
    def _generate_reasoning(
        self,
        signal_type: SignalType,
        rsi: float,
        macd: float,
        bb_position: float,
        trend: float,
        price_vs_sma: float
    ) -> str:
        """Generate human-readable reasoning for the signal"""
        try:
            reasons = []
            
            # RSI reasoning
            if rsi < 30:
                reasons.append("RSI indicates oversold conditions")
            elif rsi > 70:
                reasons.append("RSI indicates overbought conditions")
            
            # MACD reasoning
            if macd > 0:
                reasons.append("MACD shows bullish momentum")
            elif macd < 0:
                reasons.append("MACD shows bearish momentum")
            
            # Bollinger Bands reasoning
            if bb_position < 0.2:
                reasons.append("Price near lower Bollinger Band")
            elif bb_position > 0.8:
                reasons.append("Price near upper Bollinger Band")
            
            # Trend reasoning
            if trend > 0:
                reasons.append("Short-term trend is positive")
            elif trend < 0:
                reasons.append("Short-term trend is negative")
            
            # Moving average reasoning
            if price_vs_sma > 1.02:
                reasons.append("Price above short-term moving average")
            elif price_vs_sma < 0.98:
                reasons.append("Price below short-term moving average")
            
            if not reasons:
                reasons.append("Mixed technical signals")
            
            return f"Technical analysis {signal_type.value}: " + "; ".join(reasons)
            
        except Exception as e:
            logger.error(f"Failed to generate reasoning: {str(e)}")
            return f"Technical analysis {signal_type.value} signal" 