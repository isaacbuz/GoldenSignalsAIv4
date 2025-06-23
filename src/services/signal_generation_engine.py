"""
Signal Generation Engine for GoldenSignalsAI V2
Implements high-quality signal generation based on best practices
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.services.data_quality_validator import DataQualityValidator, DataQualityReport
from src.utils.timezone_utils import now_utc, make_aware
from src.utils.performance import measure_performance, AsyncBatchProcessor, performance_cache

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Represents a high-quality trading signal"""
    id: str
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0-1
    price: float
    timestamp: datetime
    reason: str
    indicators: Dict[str, float]
    risk_level: str  # low, medium, high
    entry_price: float
    stop_loss: float
    take_profit: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'action': self.action,
            'confidence': self.confidence,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'reason': self.reason,
            'indicators': self.indicators,
            'risk_level': self.risk_level,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'metadata': self.metadata,
            'quality_score': self.quality_score
        }


class SignalGenerationEngine:
    """
    High-quality signal generation engine with:
    - Multiple indicator analysis
    - ML-based prediction
    - Quality scoring
    - Risk management
    """
    
    def __init__(self) -> None:
        self.data_validator = DataQualityValidator()
        self.scaler = StandardScaler()
        self.ml_model = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.signal_cache = {}
        self.cache_ttl = 60  # seconds
        
    async def generate_signals(self, symbols: List[str]) -> List[TradingSignal]:
        """Generate high-quality signals for multiple symbols"""
        with measure_performance("signal_generation_total"):
            # Use batch processor for better performance
            batch_processor = AsyncBatchProcessor(
                batch_size=10,
                max_concurrent=self.executor._max_workers
            )
            
            results = await batch_processor.process_all(
                symbols,
                self._generate_signal_for_symbol
            )
            
            signals = []
            for result in results:
                if isinstance(result, TradingSignal):
                    signals.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Signal generation error: {result}")
                    
            logger.info(f"Generated {len(signals)} signals from {len(symbols)} symbols")
            return signals
        
    async def _generate_signal_for_symbol(self, symbol: str) -> Optional[TradingSignal]:
        """Generate signal for a single symbol"""
        # Check cache
        if symbol in self.signal_cache:
            cached_signal, cache_time = self.signal_cache[symbol]
            if (now_utc() - cache_time).seconds < self.cache_ttl:
                logger.debug(f"Using cached signal for {symbol}")
                return cached_signal
        
        try:
            with measure_performance(f"signal_generation_{symbol}"):
                # Get high-quality data
                data, source = await self.data_validator.get_market_data_with_fallback(symbol)
                
                if data is None or data.empty:
                    logger.warning(f"No data available for {symbol}")
                    return None
                
            # Validate data quality
            quality_report = self.data_validator.validate_market_data(data, symbol)
            
            if not quality_report.is_valid:
                logger.warning(f"Data quality issues for {symbol}: {quality_report.issues}")
                # Clean data if possible
                data = self.data_validator.clean_data(data)
                
            # Generate technical indicators
            indicators = await self._calculate_indicators(data)
            
            # Engineer features
            features = self._engineer_features(data, indicators)
            
            # Generate signal
            signal = await self._analyze_and_generate_signal(
                symbol, data, indicators, features, quality_report
            )
            
            # Cache the signal
            if signal:
                self.signal_cache[symbol] = (signal, now_utc())
                
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
            
    async def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate comprehensive technical indicators"""
        indicators = {}
        
        # Price-based indicators
        indicators['sma_20'] = data['Close'].rolling(window=20).mean()
        indicators['sma_50'] = data['Close'].rolling(window=50).mean()
        indicators['ema_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        indicators['ema_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
        indicators['macd_signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
        indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # RSI
        indicators['rsi'] = await self._calculate_rsi(data['Close'])
        
        # Bollinger Bands
        bb_sma = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        indicators['bb_upper'] = bb_sma + (bb_std * 2)
        indicators['bb_lower'] = bb_sma - (bb_std * 2)
        indicators['bb_percent'] = (data['Close'] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        # Volume indicators
        indicators['volume_sma'] = data['Volume'].rolling(window=20).mean()
        indicators['volume_ratio'] = data['Volume'] / indicators['volume_sma']
        
        # ATR (Average True Range)
        indicators['atr'] = await self._calculate_atr(data)
        
        # Stochastic
        indicators['stoch_k'], indicators['stoch_d'] = await self._calculate_stochastic(data)
        
        return indicators
        
    def _calculate_indicators_sync(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Synchronous version of calculate_indicators for training"""
        indicators = {}
        
        # Price-based indicators
        indicators['sma_20'] = data['Close'].rolling(window=20).mean()
        indicators['sma_50'] = data['Close'].rolling(window=50).mean()
        indicators['ema_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        indicators['ema_26'] = data['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
        indicators['macd_signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
        indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # RSI
        indicators['rsi'] = self._calculate_rsi_sync(data['Close'])
        
        # Bollinger Bands
        bb_sma = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        indicators['bb_upper'] = bb_sma + (bb_std * 2)
        indicators['bb_lower'] = bb_sma - (bb_std * 2)
        indicators['bb_percent'] = (data['Close'] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        # Volume indicators
        indicators['volume_sma'] = data['Volume'].rolling(window=20).mean()
        indicators['volume_ratio'] = data['Volume'] / indicators['volume_sma']
        
        # ATR (Average True Range)
        indicators['atr'] = self._calculate_atr_sync(data)
        
        # Stochastic
        indicators['stoch_k'], indicators['stoch_d'] = self._calculate_stochastic_sync(data)
        
        return indicators
        
    async def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    async def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
        
    async def _calculate_stochastic(self, data: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        low_min = data['Low'].rolling(window=period).min()
        high_max = data['High'].rolling(window=period).max()
        
        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=3).mean()
        
        return k_percent, d_percent
        
    def _calculate_rsi_sync(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Synchronous RSI calculation"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_atr_sync(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Synchronous ATR calculation"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
        
    def _calculate_stochastic_sync(self, data: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Synchronous Stochastic calculation"""
        low_min = data['Low'].rolling(window=period).min()
        high_max = data['High'].rolling(window=period).max()
        
        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=3).mean()
        
        return k_percent, d_percent
        
    def _engineer_features(self, data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> pd.DataFrame:
        """Engineer features for ML model"""
        features = pd.DataFrame(index=data.index)
        
        # Price features
        features['price_change'] = data['Close'].pct_change()
        features['high_low_ratio'] = data['High'] / data['Low']
        features['close_open_ratio'] = data['Close'] / data['Open']
        
        # Volume features
        features['volume_change'] = data['Volume'].pct_change()
        features['price_volume_trend'] = features['price_change'] * data['Volume']
        
        # Indicator features
        features['sma_cross'] = (indicators['sma_20'] > indicators['sma_50']).astype(int)
        features['macd_cross'] = (indicators['macd'] > indicators['macd_signal']).astype(int)
        features['rsi_oversold'] = (indicators['rsi'] < 30).astype(int)
        features['rsi_overbought'] = (indicators['rsi'] > 70).astype(int)
        features['bb_position'] = indicators['bb_percent']
        
        # Momentum features
        features['momentum_5'] = data['Close'].pct_change(5)
        features['momentum_10'] = data['Close'].pct_change(10)
        
        # Volatility features
        features['volatility'] = data['Close'].pct_change().rolling(20).std()
        features['atr_ratio'] = indicators['atr'] / data['Close']
        
        return features.fillna(0)
        
    async def _analyze_and_generate_signal(
        self, 
        symbol: str,
        data: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        features: pd.DataFrame,
        quality_report: DataQualityReport
    ) -> Optional[TradingSignal]:
        """Analyze data and generate trading signal"""
        # Get latest values
        latest = data.iloc[-1]
        latest_indicators = {k: v.iloc[-1] for k, v in indicators.items() if not v.empty}
        latest_features = features.iloc[-1]
        
        # Initialize signal parameters
        action = "HOLD"
        confidence = 0.0
        reasons = []
        
        # Technical analysis signals
        # RSI Signal
        if latest_indicators.get('rsi', 50) < 30:
            confidence += 0.2
            reasons.append("RSI oversold")
            action = "BUY"
        elif latest_indicators.get('rsi', 50) > 70:
            confidence += 0.2
            reasons.append("RSI overbought")
            action = "SELL"
            
        # MACD Signal
        if latest_features.get('macd_cross') == 1:
            confidence += 0.15
            reasons.append("MACD bullish crossover")
            if action != "SELL":
                action = "BUY"
        elif latest_features.get('macd_cross') == 0 and latest_indicators.get('macd', 0) < latest_indicators.get('macd_signal', 0):
            confidence += 0.15
            reasons.append("MACD bearish crossover")
            if action != "BUY":
                action = "SELL"
                
        # Bollinger Bands Signal
        bb_percent = latest_indicators.get('bb_percent', 0.5)
        if bb_percent < 0.2:
            confidence += 0.15
            reasons.append("Near lower Bollinger Band")
            if action != "SELL":
                action = "BUY"
        elif bb_percent > 0.8:
            confidence += 0.15
            reasons.append("Near upper Bollinger Band")
            if action != "BUY":
                action = "SELL"
                
        # Moving Average Signal
        if latest_features.get('sma_cross') == 1:
            confidence += 0.1
            reasons.append("Golden cross (SMA)")
            if action == "HOLD":
                action = "BUY"
                
        # Volume Signal
        if latest_indicators.get('volume_ratio', 1) > 2:
            confidence += 0.1
            reasons.append("High volume surge")
            
        # Stochastic Signal
        stoch_k = latest_indicators.get('stoch_k', 50)
        if stoch_k < 20:
            confidence += 0.1
            reasons.append("Stochastic oversold")
            if action != "SELL":
                action = "BUY"
        elif stoch_k > 80:
            confidence += 0.1
            reasons.append("Stochastic overbought")
            if action != "BUY":
                action = "SELL"
                
        # ML prediction (if model is trained)
        if self.ml_model is not None:
            try:
                ml_features = latest_features.values.reshape(1, -1)
                ml_features_scaled = self.scaler.transform(ml_features)
                ml_prediction = self.ml_model.predict_proba(ml_features_scaled)[0]
                
                ml_confidence = max(ml_prediction)
                ml_action = ["SELL", "HOLD", "BUY"][np.argmax(ml_prediction)]
                
                # Blend ML prediction with technical analysis
                if ml_confidence > 0.6:
                    confidence = (confidence + ml_confidence) / 2
                    if ml_action != "HOLD":
                        action = ml_action
                        reasons.append(f"ML prediction: {ml_action} ({ml_confidence:.2f})")
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")
                
        # Quality adjustment
        quality_factor = quality_report.overall_score
        confidence *= quality_factor
        
        # Risk level determination
        volatility = latest_features.get('volatility', 0.02)
        if volatility < 0.01:
            risk_level = "low"
        elif volatility < 0.025:
            risk_level = "medium"
        else:
            risk_level = "high"
            
        # Only generate signal if confidence is sufficient
        if confidence < 0.3 or action == "HOLD":
            return None
            
        # Calculate entry, stop loss, and take profit
        current_price = float(latest['Close'])
        atr = float(latest_indicators.get('atr', current_price * 0.02))
        
        if action == "BUY":
            entry_price = current_price
            stop_loss = current_price - (2 * atr)
            take_profit = current_price + (3 * atr)
        else:  # SELL
            entry_price = current_price
            stop_loss = current_price + (2 * atr)
            take_profit = current_price - (3 * atr)
            
        # Create signal
        signal = TradingSignal(
            id=f"{symbol}_{int(now_utc().timestamp()*1000)}",
            symbol=symbol,
            action=action,
            confidence=min(confidence, 1.0),
            price=current_price,
            timestamp=now_utc(),
            reason="; ".join(reasons),
            indicators={k: float(v) if not pd.isna(v) else 0.0 for k, v in latest_indicators.items()},
            risk_level=risk_level,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'data_source': quality_report.symbol,
                'data_quality_score': quality_report.overall_score,
                'volume': int(latest.get('Volume', 0)),
                'features': {k: float(v) if not pd.isna(v) else 0.0 for k, v in latest_features.items()}
            },
            quality_score=quality_factor
        )
        
        return signal
        
    def train_ml_model(self, historical_data: pd.DataFrame, labels: pd.Series) -> None:
        """Train the ML model on historical data"""
        try:
            # Prepare features
            # Using synchronous calculation for training
            indicators = self._calculate_indicators_sync(historical_data)
            features = self._engineer_features(historical_data, indicators)
            
            # Remove NaN values
            valid_idx = ~(features.isna().any(axis=1) | labels.isna())
            features_clean = features[valid_idx]
            labels_clean = labels[valid_idx]
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features_clean)
            
            # Train model
            self.ml_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.ml_model.fit(features_scaled, labels_clean)
            
            logger.info(f"ML model trained with accuracy: {self.ml_model.score(features_scaled, labels_clean):.2f}")
            
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
            self.ml_model = None 