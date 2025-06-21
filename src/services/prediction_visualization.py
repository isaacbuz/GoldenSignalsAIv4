import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import yfinance as yf
import talib
from enum import Enum

logger = logging.getLogger(__name__)

class PredictionTimeframe(Enum):
    """Prediction timeframes"""
    MINUTES_5 = "5m"
    MINUTES_15 = "15m"
    MINUTES_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    
    @property
    def minutes(self) -> int:
        """Get timeframe in minutes"""
        mapping = {
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
            "1w": 10080
        }
        return mapping.get(self.value, 60)

@dataclass
class PredictionPoint:
    """Single prediction point"""
    timestamp: datetime
    price: float
    confidence: float
    upper_bound: float
    lower_bound: float
    
@dataclass
class TrendPrediction:
    """Complete trend prediction"""
    symbol: str
    current_price: float
    prediction_points: List[PredictionPoint]
    trend_direction: str  # "bullish", "bearish", "neutral"
    strength: float  # 0-100
    confidence: float  # 0-100
    support_levels: List[float]
    resistance_levels: List[float]
    key_levels: List[Dict[str, Any]]
    momentum_score: float
    volatility: float
    prediction_method: str
    timeframe: str
    metadata: Dict[str, Any]

class PredictionVisualizationService:
    """Advanced prediction visualization service"""
    
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'polynomial': None,  # Initialized per use
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        logger.info("Prediction Visualization Service initialized")
        
    def generate_predictions(
        self,
        symbol: str,
        historical_data: pd.DataFrame,
        timeframe: PredictionTimeframe,
        prediction_periods: int = 20
    ) -> TrendPrediction:
        """Generate comprehensive price predictions with trend visualization"""
        try:
            # Prepare data
            df = self._prepare_data(historical_data)
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(df)
            
            # Generate multiple predictions
            predictions = {
                'linear': self._linear_prediction(df, prediction_periods),
                'polynomial': self._polynomial_prediction(df, prediction_periods),
                'technical': self._technical_prediction(df, indicators, prediction_periods),
                'ml_ensemble': self._ml_ensemble_prediction(df, indicators, prediction_periods),
                'monte_carlo': self._monte_carlo_prediction(df, prediction_periods)
            }
            
            # Combine predictions
            combined_prediction = self._combine_predictions(predictions, df)
            
            # Calculate support/resistance
            support_resistance = self._calculate_support_resistance(df)
            
            # Analyze trend
            trend_analysis = self._analyze_trend(df, combined_prediction)
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                combined_prediction, df
            )
            
            # Build prediction points
            prediction_points = self._build_prediction_points(
                combined_prediction,
                confidence_intervals,
                timeframe
            )
            
            return TrendPrediction(
                symbol=symbol,
                current_price=float(df['close'].iloc[-1]),
                prediction_points=prediction_points,
                trend_direction=trend_analysis['direction'],
                strength=trend_analysis['strength'],
                confidence=trend_analysis['confidence'],
                support_levels=support_resistance['support'],
                resistance_levels=support_resistance['resistance'],
                key_levels=self._identify_key_levels(df, support_resistance),
                momentum_score=self._calculate_momentum_score(df, indicators),
                volatility=self._calculate_volatility(df),
                prediction_method="ensemble",
                timeframe=timeframe.value,
                metadata={
                    'models_used': list(predictions.keys()),
                    'indicators': list(indicators.keys()),
                    'data_points': len(df),
                    'prediction_timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            raise
            
    def _prepare_data(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean historical data"""
        df = historical_data.copy()
        
        # Ensure required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
                
        # Sort by timestamp
        df = df.sort_index()
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
        
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate technical indicators for prediction"""
        indicators = {}
        
        # Trend indicators
        indicators['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        indicators['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        indicators['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        indicators['ema_26'] = talib.EMA(df['close'], timeperiod=26)
        
        # MACD
        macd, signal, hist = talib.MACD(df['close'])
        indicators['macd'] = macd
        indicators['macd_signal'] = signal
        indicators['macd_hist'] = hist
        
        # RSI
        indicators['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        indicators['bb_upper'] = upper
        indicators['bb_middle'] = middle
        indicators['bb_lower'] = lower
        
        # ATR for volatility
        indicators['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Volume indicators
        indicators['obv'] = talib.OBV(df['close'], df['volume'])
        indicators['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        
        # Momentum
        indicators['roc'] = talib.ROC(df['close'], timeperiod=10)
        indicators['mom'] = talib.MOM(df['close'], timeperiod=10)
        
        # Pattern recognition helpers
        indicators['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        indicators['r1'] = 2 * indicators['pivot'] - df['low']
        indicators['s1'] = 2 * indicators['pivot'] - df['high']
        
        return indicators
        
    def _linear_prediction(
        self, 
        df: pd.DataFrame, 
        periods: int
    ) -> np.ndarray:
        """Linear regression prediction"""
        # Use last N periods for training
        n_periods = min(100, len(df))
        recent_data = df.tail(n_periods)
        
        X = np.arange(len(recent_data)).reshape(-1, 1)
        y = recent_data['close'].values
        
        # Fit model
        self.models['linear'].fit(X, y)
        
        # Predict future
        future_X = np.arange(len(recent_data), len(recent_data) + periods).reshape(-1, 1)
        predictions = self.models['linear'].predict(future_X)
        
        return predictions
        
    def _polynomial_prediction(
        self,
        df: pd.DataFrame,
        periods: int,
        degree: int = 3
    ) -> np.ndarray:
        """Polynomial regression prediction"""
        n_periods = min(100, len(df))
        recent_data = df.tail(n_periods)
        
        X = np.arange(len(recent_data)).reshape(-1, 1)
        y = recent_data['close'].values
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        
        # Fit model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Predict future
        future_X = np.arange(len(recent_data), len(recent_data) + periods).reshape(-1, 1)
        future_X_poly = poly.transform(future_X)
        predictions = model.predict(future_X_poly)
        
        return predictions
        
    def _technical_prediction(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        periods: int
    ) -> np.ndarray:
        """Prediction based on technical analysis"""
        predictions = []
        last_close = df['close'].iloc[-1]
        
        # Calculate trend strength
        sma_20 = indicators['sma_20'].iloc[-1]
        sma_50 = indicators['sma_50'].iloc[-1]
        
        # Determine trend direction
        if last_close > sma_20 > sma_50:
            trend_multiplier = 1.002  # Bullish
        elif last_close < sma_20 < sma_50:
            trend_multiplier = 0.998  # Bearish
        else:
            trend_multiplier = 1.0  # Neutral
            
        # Use ATR for volatility adjustment
        atr = indicators['atr'].iloc[-1]
        volatility_factor = atr / last_close
        
        # Generate predictions
        price = last_close
        for i in range(periods):
            # Apply trend with some randomness
            random_factor = np.random.normal(1.0, volatility_factor * 0.5)
            price = price * trend_multiplier * random_factor
            
            # Apply mean reversion pressure
            if indicators['rsi'].iloc[-1] > 70:  # Overbought
                price *= 0.995
            elif indicators['rsi'].iloc[-1] < 30:  # Oversold
                price *= 1.005
                
            predictions.append(price)
            
        return np.array(predictions)
        
    def _ml_ensemble_prediction(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        periods: int
    ) -> np.ndarray:
        """Machine learning ensemble prediction"""
        # Prepare features
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Add indicators as features
        for name, series in indicators.items():
            if len(series) == len(df):
                df[f'ind_{name}'] = series
                feature_cols.append(f'ind_{name}')
                
        # Create lagged features
        for i in range(1, 6):
            df[f'close_lag_{i}'] = df['close'].shift(i)
            df[f'volume_lag_{i}'] = df['volume'].shift(i)
            feature_cols.extend([f'close_lag_{i}', f'volume_lag_{i}'])
            
        # Remove NaN rows
        df_clean = df.dropna()
        
        # Prepare training data
        X = df_clean[feature_cols].values[:-1]
        y = df_clean['close'].values[1:]
        
        # Train model
        self.models['random_forest'].fit(X, y)
        
        # Generate predictions iteratively
        predictions = []
        last_features = df_clean[feature_cols].iloc[-1].values.reshape(1, -1)
        
        for _ in range(periods):
            pred = self.models['random_forest'].predict(last_features)[0]
            predictions.append(pred)
            
            # Update features for next prediction
            # This is simplified - in production, update all features properly
            last_features[0, 3] = pred  # Update close price
            
        return np.array(predictions)
        
    def _monte_carlo_prediction(
        self,
        df: pd.DataFrame,
        periods: int,
        n_simulations: int = 1000
    ) -> np.ndarray:
        """Monte Carlo simulation for price prediction"""
        returns = df['close'].pct_change().dropna()
        
        # Calculate drift and volatility
        drift = returns.mean()
        volatility = returns.std()
        
        # Run simulations
        last_price = df['close'].iloc[-1]
        simulations = []
        
        for _ in range(n_simulations):
            prices = [last_price]
            for _ in range(periods):
                shock = np.random.normal(drift, volatility)
                price = prices[-1] * (1 + shock)
                prices.append(price)
            simulations.append(prices[1:])
            
        # Return mean of simulations
        return np.mean(simulations, axis=0)
        
    def _combine_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        df: pd.DataFrame
    ) -> np.ndarray:
        """Combine multiple predictions with weighted average"""
        # Define weights based on historical accuracy (simplified)
        weights = {
            'linear': 0.15,
            'polynomial': 0.15,
            'technical': 0.25,
            'ml_ensemble': 0.30,
            'monte_carlo': 0.15
        }
        
        # Ensure all predictions have same length
        min_length = min(len(pred) for pred in predictions.values())
        
        # Weighted average
        combined = np.zeros(min_length)
        for method, pred in predictions.items():
            combined += weights[method] * pred[:min_length]
            
        return combined
        
    def _calculate_support_resistance(
        self,
        df: pd.DataFrame,
        lookback: int = 100
    ) -> Dict[str, List[float]]:
        """Calculate support and resistance levels"""
        recent_data = df.tail(lookback)
        
        # Find local minima and maxima
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        
        # Identify peaks and troughs
        support_levels = []
        resistance_levels = []
        
        for i in range(2, len(highs) - 2):
            # Resistance (local maxima)
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                resistance_levels.append(highs[i])
                
            # Support (local minima)
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                support_levels.append(lows[i])
                
        # Cluster nearby levels
        support_levels = self._cluster_levels(support_levels)
        resistance_levels = self._cluster_levels(resistance_levels)
        
        # Add psychological levels
        current_price = df['close'].iloc[-1]
        psychological_levels = self._get_psychological_levels(current_price)
        
        return {
            'support': sorted(support_levels)[:5],  # Top 5 support levels
            'resistance': sorted(resistance_levels, reverse=True)[:5],  # Top 5 resistance levels
            'psychological': psychological_levels
        }
        
    def _cluster_levels(self, levels: List[float], threshold: float = 0.01) -> List[float]:
        """Cluster nearby price levels"""
        if not levels:
            return []
            
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if (level - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
                
        if current_cluster:
            clusters.append(np.mean(current_cluster))
            
        return clusters
        
    def _get_psychological_levels(self, price: float) -> List[float]:
        """Get psychological price levels (round numbers)"""
        levels = []
        
        # Determine scale
        if price < 10:
            step = 0.5
        elif price < 100:
            step = 5
        elif price < 1000:
            step = 50
        else:
            step = 100
            
        # Get levels around current price
        base = (price // step) * step
        for i in range(-2, 3):
            level = base + i * step
            if level > 0:
                levels.append(level)
                
        return levels
        
    def _analyze_trend(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze trend direction and strength"""
        current_price = df['close'].iloc[-1]
        predicted_price = predictions[-1]
        
        # Calculate trend metrics
        price_change = (predicted_price - current_price) / current_price
        
        # Determine direction
        if price_change > 0.02:
            direction = "bullish"
        elif price_change < -0.02:
            direction = "bearish"
        else:
            direction = "neutral"
            
        # Calculate strength (0-100)
        strength = min(100, abs(price_change) * 1000)
        
        # Calculate confidence based on prediction consistency
        prediction_std = np.std(predictions)
        confidence = max(0, 100 - (prediction_std / current_price * 100))
        
        return {
            'direction': direction,
            'strength': strength,
            'confidence': confidence,
            'price_change': price_change
        }
        
    def _calculate_confidence_intervals(
        self,
        predictions: np.ndarray,
        df: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals for predictions"""
        # Use historical volatility
        returns = df['close'].pct_change().dropna()
        volatility = returns.std()
        
        # Calculate standard error
        n = len(df)
        std_error = volatility * np.sqrt(np.arange(1, len(predictions) + 1))
        
        # Z-score for confidence level
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Calculate bounds
        upper_bound = predictions + z_score * std_error
        lower_bound = predictions - z_score * std_error
        
        return {
            'upper': upper_bound,
            'lower': lower_bound,
            'std_error': std_error
        }
        
    def _build_prediction_points(
        self,
        predictions: np.ndarray,
        confidence_intervals: Dict[str, np.ndarray],
        timeframe: PredictionTimeframe
    ) -> List[PredictionPoint]:
        """Build prediction points with timestamps"""
        points = []
        current_time = datetime.now()
        
        for i in range(len(predictions)):
            # Calculate timestamp
            minutes_ahead = (i + 1) * timeframe.minutes
            timestamp = current_time + timedelta(minutes=minutes_ahead)
            
            # Calculate confidence decay
            confidence = 100 * np.exp(-i * 0.05)  # Exponential decay
            
            point = PredictionPoint(
                timestamp=timestamp,
                price=float(predictions[i]),
                confidence=confidence,
                upper_bound=float(confidence_intervals['upper'][i]),
                lower_bound=float(confidence_intervals['lower'][i])
            )
            points.append(point)
            
        return points
        
    def _identify_key_levels(
        self,
        df: pd.DataFrame,
        support_resistance: Dict[str, List[float]]
    ) -> List[Dict[str, Any]]:
        """Identify key price levels with metadata"""
        key_levels = []
        current_price = df['close'].iloc[-1]
        
        # Add support levels
        for level in support_resistance['support']:
            distance_pct = abs(current_price - level) / current_price * 100
            key_levels.append({
                'price': level,
                'type': 'support',
                'strength': self._calculate_level_strength(df, level),
                'distance_pct': distance_pct,
                'touches': self._count_level_touches(df, level)
            })
            
        # Add resistance levels
        for level in support_resistance['resistance']:
            distance_pct = abs(current_price - level) / current_price * 100
            key_levels.append({
                'price': level,
                'type': 'resistance',
                'strength': self._calculate_level_strength(df, level),
                'distance_pct': distance_pct,
                'touches': self._count_level_touches(df, level)
            })
            
        # Sort by distance from current price
        key_levels.sort(key=lambda x: x['distance_pct'])
        
        return key_levels[:10]  # Return top 10 closest levels
        
    def _calculate_level_strength(
        self,
        df: pd.DataFrame,
        level: float,
        threshold: float = 0.01
    ) -> float:
        """Calculate strength of a price level"""
        touches = self._count_level_touches(df, level, threshold)
        recency = self._calculate_level_recency(df, level, threshold)
        
        # Combine touches and recency
        strength = (touches * 0.7 + recency * 0.3) * 10
        return min(100, strength)
        
    def _count_level_touches(
        self,
        df: pd.DataFrame,
        level: float,
        threshold: float = 0.01
    ) -> int:
        """Count how many times price touched a level"""
        touches = 0
        
        for _, row in df.iterrows():
            # Check if high or low is near the level
            if (abs(row['high'] - level) / level < threshold or
                abs(row['low'] - level) / level < threshold):
                touches += 1
                
        return touches
        
    def _calculate_level_recency(
        self,
        df: pd.DataFrame,
        level: float,
        threshold: float = 0.01
    ) -> float:
        """Calculate how recent a level was tested"""
        for i in range(len(df) - 1, -1, -1):
            row = df.iloc[i]
            if (abs(row['high'] - level) / level < threshold or
                abs(row['low'] - level) / level < threshold):
                # Return normalized recency score
                return (i / len(df)) * 100
                
        return 0
        
    def _calculate_momentum_score(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, pd.Series]
    ) -> float:
        """Calculate overall momentum score"""
        scores = []
        
        # RSI momentum
        rsi = indicators['rsi'].iloc[-1]
        if rsi > 50:
            scores.append((rsi - 50) * 2)  # 0-100 scale
        else:
            scores.append((50 - rsi) * -2)  # -100-0 scale
            
        # MACD momentum
        macd_hist = indicators['macd_hist'].iloc[-1]
        macd_score = np.clip(macd_hist * 10, -100, 100)
        scores.append(macd_score)
        
        # Price vs moving averages
        price = df['close'].iloc[-1]
        sma_20 = indicators['sma_20'].iloc[-1]
        ma_score = ((price - sma_20) / sma_20) * 100
        scores.append(np.clip(ma_score, -100, 100))
        
        # Average momentum score
        return np.mean(scores)
        
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate current volatility"""
        returns = df['close'].pct_change().dropna()
        
        # Calculate different volatility measures
        std_volatility = returns.std() * np.sqrt(252) * 100  # Annualized
        
        # Recent volatility (last 20 periods)
        recent_volatility = returns.tail(20).std() * np.sqrt(252) * 100
        
        # Weight recent volatility more
        return 0.7 * recent_volatility + 0.3 * std_volatility
        
    def get_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect candlestick patterns for visualization"""
        patterns = []
        
        # Use TA-Lib pattern recognition
        pattern_functions = {
            'HAMMER': talib.CDLHAMMER,
            'DOJI': talib.CDLDOJI,
            'ENGULFING': talib.CDLENGULFING,
            'MORNING_STAR': talib.CDLMORNINGSTAR,
            'EVENING_STAR': talib.CDLEVENINGSTAR,
            'THREE_WHITE_SOLDIERS': talib.CDL3WHITESOLDIERS,
            'THREE_BLACK_CROWS': talib.CDL3BLACKCROWS,
            'SHOOTING_STAR': talib.CDLSHOOTINGSTAR,
            'HARAMI': talib.CDLHARAMI,
            'DARK_CLOUD_COVER': talib.CDLDARKCLOUDCOVER
        }
        
        for pattern_name, pattern_func in pattern_functions.items():
            try:
                result = pattern_func(
                    df['open'].values,
                    df['high'].values,
                    df['low'].values,
                    df['close'].values
                )
                
                # Find where pattern is detected
                for i, val in enumerate(result):
                    if val != 0:  # Pattern detected
                        patterns.append({
                            'type': pattern_name,
                            'index': i,
                            'timestamp': df.index[i],
                            'price': df['close'].iloc[i],
                            'direction': 'bullish' if val > 0 else 'bearish',
                            'strength': abs(val)
                        })
            except Exception as e:
                logger.warning(f"Error detecting {pattern_name}: {e}")
                
        return patterns 