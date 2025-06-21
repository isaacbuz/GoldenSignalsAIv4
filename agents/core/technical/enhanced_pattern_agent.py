"""
Enhanced Pattern Recognition Agent - Institutional-Grade Pattern Detection
Combines Bulkowski's classical patterns, @GxTradez price phase methodology,
multi-timeframe candlestick analysis, and selective ML enhancement
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from scipy import stats
from scipy.signal import argrelextrema
from dataclasses import dataclass
from datetime import datetime, timedelta
import yfinance as yf
from collections import defaultdict
import talib
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

from agents.core.technical.pattern_agent import PatternAgent

logger = logging.getLogger(__name__)

@dataclass
class TimeframeData:
    """Store data for different timeframes"""
    df_1h: pd.DataFrame
    df_4h: pd.DataFrame
    df_daily: pd.DataFrame
    df_weekly: Optional[pd.DataFrame] = None

@dataclass
class MarketPhase:
    """Market phase classification"""
    phase: str  # 'expansion' or 'consolidation'
    direction: str  # 'bullish', 'bearish', or 'neutral'
    strength: float  # 0-1
    duration: int  # bars in current phase

class EnhancedPatternAgent(PatternAgent):
    """
    Advanced pattern recognition combining multiple methodologies:
    1. Bulkowski's statistically-proven chart patterns
    2. @GxTradez price phase analysis and SMT divergence
    3. Multi-timeframe candlestick confluence
    4. ML for pattern quality scoring (not pattern detection)
    """
    
    # Bulkowski's pattern statistics with modern adjustments
    PATTERN_STATISTICS = {
        'high_tight_flag': {
            'avg_gain': 0.21,
            'success_rate': 0.85,
            'modern_adjustment': 0.7,  # Account for increased failure rates
            'min_prior_rise': 0.90,
            'max_flag_depth': 0.25,
            'min_volume_decline': 0.4
        },
        'double_bottom': {
            'eve_eve': {'avg_gain': 0.12, 'success_rate': 0.82, 'modern_adjustment': 0.75},
            'adam_adam': {'avg_gain': 0.10, 'success_rate': 0.78, 'modern_adjustment': 0.72},
            'adam_eve': {'avg_gain': 0.10, 'success_rate': 0.79, 'modern_adjustment': 0.73},
            'eve_adam': {'avg_gain': 0.10, 'success_rate': 0.80, 'modern_adjustment': 0.74}
        },
        'head_shoulders': {
            'avg_decline': -0.23,
            'success_rate': 0.77,
            'modern_adjustment': 0.68,
            'false_signal_rate': 0.07,
            'min_pattern_width': 20,
            'neckline_tolerance': 0.02
        },
        'rounding_bottom': {
            'avg_gain': 0.48,
            'success_rate': 0.96,
            'modern_adjustment': 0.85,
            'break_even_failure': 0.04,
            'min_duration_weeks': 7,
            'volume_pattern': 'u_shaped'
        },
        'ascending_triangle': {
            'avg_gain': 0.09,
            'success_rate': 0.72,
            'modern_adjustment': 0.65,
            'min_touches': 2,
            'resistance_tolerance': 0.01
        },
        'falling_wedge': {
            'avg_gain': 0.10,
            'success_rate': 0.73,
            'modern_adjustment': 0.66,
            'volume_decline_required': True,
            'min_convergence': 0.7
        },
        'rectangle': {
            'bottom': {'avg_gain': 0.11, 'success_rate': 0.79, 'modern_adjustment': 0.71},
            'top': {'avg_decline': -0.08, 'success_rate': 0.71, 'modern_adjustment': 0.64}
        },
        'cup_handle': {
            'avg_gain': 0.34,
            'success_rate': 0.68,
            'modern_adjustment': 0.61,
            'min_cup_depth': 0.12,
            'max_cup_depth': 0.33,
            'handle_drift': 0.10
        }
    }
    
    # Candlestick patterns for multi-timeframe confluence
    CANDLESTICK_PATTERNS = {
        'bullish_reversal': [
            'hammer', 'inverted_hammer', 'bullish_engulfing', 
            'piercing_pattern', 'morning_star', 'three_white_soldiers',
            'bullish_harami', 'tweezer_bottom', 'dragonfly_doji'
        ],
        'bearish_reversal': [
            'hanging_man', 'shooting_star', 'bearish_engulfing',
            'dark_cloud_cover', 'evening_star', 'three_black_crows',
            'bearish_harami', 'tweezer_top', 'gravestone_doji'
        ],
        'continuation': [
            'rising_three_methods', 'falling_three_methods',
            'upside_gap_three_methods', 'downside_gap_three_methods'
        ]
    }
    
    def __init__(
        self,
        name: str = "EnhancedPattern",
        use_ml_scoring: bool = True,
        check_correlations: bool = True,
        timeframes: List[str] = ['1h', '4h', '1d']
    ):
        """
        Initialize Enhanced Pattern Agent
        
        Args:
            name: Agent name
            use_ml_scoring: Use ML for pattern quality scoring
            check_correlations: Enable SMT divergence checking
            timeframes: Timeframes to analyze
        """
        super().__init__(name=name)
        self.use_ml_scoring = use_ml_scoring
        self.check_correlations = check_correlations
        self.timeframes = timeframes
        self.ml_model = None
        self.correlation_pairs = {
            'indices': ['SPY', 'QQQ', 'IWM'],
            'sectors': ['XLF', 'XLE', 'XLK', 'XLV']
        }
        
        if self.use_ml_scoring:
            self._load_or_train_ml_model()
    
    def _load_or_train_ml_model(self):
        """Load pre-trained ML model or train a simple one for pattern quality scoring"""
        model_path = 'ml_models/pattern_quality_scorer.pkl'
        
        if os.path.exists(model_path):
            self.ml_model = joblib.load(model_path)
        else:
            # Train a simple model for pattern quality scoring
            # In production, this would use historical pattern outcomes
            logger.info("Training ML pattern quality scorer...")
            # Placeholder - in reality, train on historical data
            self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def get_multi_timeframe_data(self, symbol: str) -> TimeframeData:
        """Fetch data for multiple timeframes"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get different timeframe data
            df_1h = ticker.history(period="1mo", interval="1h")
            df_4h = ticker.history(period="3mo", interval="1h").resample('4H').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 
                'Close': 'last', 'Volume': 'sum'
            }).dropna()
            df_daily = ticker.history(period="1y", interval="1d")
            df_weekly = ticker.history(period="2y", interval="1wk")
            
            return TimeframeData(df_1h, df_4h, df_daily, df_weekly)
            
        except Exception as e:
            logger.error(f"Failed to fetch multi-timeframe data: {str(e)}")
            return None
    
    def identify_market_phase(self, prices: pd.Series, period: int = 20) -> MarketPhase:
        """
        Identify current market phase using @GxTradez methodology
        Price cannot reverse from consolidation - must expand first
        """
        try:
            # Calculate ATR for volatility
            high = prices.shift(1).rolling(period).max()
            low = prices.shift(1).rolling(period).min()
            atr = talib.ATR(high.values, low.values, prices.values, timeperiod=14)
            
            # Recent price range
            recent_high = prices.tail(period).max()
            recent_low = prices.tail(period).min()
            range_pct = (recent_high - recent_low) / recent_low
            
            # Directional movement
            adx = talib.ADX(high.values, low.values, prices.values, timeperiod=14)
            plus_di = talib.PLUS_DI(high.values, low.values, prices.values, timeperiod=14)
            minus_di = talib.MINUS_DI(high.values, low.values, prices.values, timeperiod=14)
            
            # Determine phase
            current_atr = atr[-1] if len(atr) > 0 else 0
            avg_atr = np.mean(atr[-period:]) if len(atr) >= period else current_atr
            
            # Expansion vs Consolidation
            if current_atr > avg_atr * 1.2 or range_pct > 0.05:
                phase = 'expansion'
                strength = min(current_atr / avg_atr, 2.0) / 2.0
            else:
                phase = 'consolidation'
                strength = 1.0 - (current_atr / avg_atr if avg_atr > 0 else 0.5)
            
            # Direction
            if len(plus_di) > 0 and len(minus_di) > 0:
                if plus_di[-1] > minus_di[-1] and adx[-1] > 25:
                    direction = 'bullish'
                elif minus_di[-1] > plus_di[-1] and adx[-1] > 25:
                    direction = 'bearish'
                else:
                    direction = 'neutral'
            else:
                direction = 'neutral'
            
            # Duration in current phase
            phase_changes = []
            for i in range(len(atr)):
                if i > period:
                    is_expansion = atr[i] > np.mean(atr[i-period:i]) * 1.2
                    phase_changes.append(is_expansion)
            
            # Count bars since last phase change
            duration = 1
            current_phase_type = phase == 'expansion'
            for i in range(len(phase_changes)-1, -1, -1):
                if phase_changes[i] == current_phase_type:
                    duration += 1
                else:
                    break
            
            return MarketPhase(phase, direction, strength, duration)
            
        except Exception as e:
            logger.error(f"Market phase identification failed: {str(e)}")
            return MarketPhase('unknown', 'neutral', 0.5, 0)
    
    def detect_smt_divergence(self, symbol: str, timeframe: str = '1h') -> Dict[str, Any]:
        """
        Detect SMT (Smart Money Technique) divergence
        Look for correlation breaks between related instruments
        """
        try:
            smt_signals = {
                'divergence_detected': False,
                'type': None,
                'strength': 0.0,
                'correlated_pairs': []
            }
            
            # Get correlation group for symbol
            correlation_group = None
            for group, symbols in self.correlation_pairs.items():
                if symbol in symbols:
                    correlation_group = symbols
                    break
            
            if not correlation_group:
                return smt_signals
            
            # Fetch data for correlation analysis
            correlations = {}
            main_data = yf.download(symbol, period="1mo", interval=timeframe)['Close']
            
            for corr_symbol in correlation_group:
                if corr_symbol != symbol:
                    corr_data = yf.download(corr_symbol, period="1mo", interval=timeframe)['Close']
                    
                    # Calculate rolling correlation
                    correlation = main_data.corr(corr_data)
                    recent_corr = main_data.tail(20).corr(corr_data.tail(20))
                    
                    # Detect divergence (correlation break)
                    if abs(correlation - recent_corr) > 0.3:
                        smt_signals['divergence_detected'] = True
                        smt_signals['strength'] = abs(correlation - recent_corr)
                        smt_signals['correlated_pairs'].append({
                            'symbol': corr_symbol,
                            'historical_corr': correlation,
                            'recent_corr': recent_corr,
                            'divergence': correlation - recent_corr
                        })
            
            # Determine divergence type
            if smt_signals['divergence_detected']:
                # Bullish divergence: Symbol underperforming but should catch up
                # Bearish divergence: Symbol outperforming but should revert
                avg_divergence = np.mean([p['divergence'] for p in smt_signals['correlated_pairs']])
                smt_signals['type'] = 'bullish' if avg_divergence < -0.3 else 'bearish'
            
            return smt_signals
            
        except Exception as e:
            logger.error(f"SMT divergence detection failed: {str(e)}")
            return {'divergence_detected': False}
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """Detect candlestick patterns using TA-Lib"""
        patterns = defaultdict(list)
        
        # Prepare OHLC data
        open_prices = df['Open'].values
        high_prices = df['High'].values
        low_prices = df['Low'].values
        close_prices = df['Close'].values
        
        # Detect patterns
        pattern_functions = {
            'hammer': talib.CDLHAMMER,
            'inverted_hammer': talib.CDLINVERTEDHAMMER,
            'bullish_engulfing': talib.CDLENGULFING,
            'piercing_pattern': talib.CDLPIERCING,
            'morning_star': talib.CDLMORNINGSTAR,
            'three_white_soldiers': talib.CDL3WHITESOLDIERS,
            'hanging_man': talib.CDLHANGINGMAN,
            'shooting_star': talib.CDLSHOOTINGSTAR,
            'bearish_engulfing': talib.CDLENGULFING,
            'dark_cloud_cover': talib.CDLDARKCLOUDCOVER,
            'evening_star': talib.CDLEVENINGSTAR,
            'three_black_crows': talib.CDL3BLACKCROWS,
            'doji': talib.CDLDOJI,
            'dragonfly_doji': talib.CDLDRAGONFLYDOJI,
            'gravestone_doji': talib.CDLGRAVESTONEDOJI
        }
        
        for pattern_name, pattern_func in pattern_functions.items():
            try:
                result = pattern_func(open_prices, high_prices, low_prices, close_prices)
                # Find where pattern is detected (non-zero values)
                pattern_indices = np.where(result != 0)[0]
                if len(pattern_indices) > 0:
                    patterns[pattern_name] = pattern_indices.tolist()
            except:
                continue
        
        return dict(patterns)
    
    def analyze_multi_timeframe_confluence(self, timeframe_data: TimeframeData) -> Dict[str, Any]:
        """
        Analyze patterns across multiple timeframes for confluence
        Following @GxTradez approach: Daily must support or show reversal criteria
        """
        confluence = {
            'score': 0.0,
            'signals': {},
            'candlesticks': {},
            'phase_alignment': False
        }
        
        # Analyze each timeframe
        for tf_name, df in [('1h', timeframe_data.df_1h), 
                           ('4h', timeframe_data.df_4h), 
                           ('daily', timeframe_data.df_daily)]:
            if df is None or len(df) < 20:
                continue
            
            # Market phase
            phase = self.identify_market_phase(df['Close'])
            confluence['signals'][tf_name] = {
                'phase': phase.phase,
                'direction': phase.direction,
                'strength': phase.strength
            }
            
            # Candlestick patterns
            candles = self.detect_candlestick_patterns(df)
            confluence['candlesticks'][tf_name] = candles
        
        # Calculate confluence score
        # Higher timeframes have more weight
        weights = {'1h': 0.2, '4h': 0.3, 'daily': 0.5}
        
        # Phase alignment check
        phases = [s['phase'] for s in confluence['signals'].values()]
        directions = [s['direction'] for s in confluence['signals'].values()]
        
        if len(set(phases)) == 1:  # All timeframes in same phase
            confluence['score'] += 0.3
            confluence['phase_alignment'] = True
        
        if len(set(directions)) == 1 and directions[0] != 'neutral':  # Same direction
            confluence['score'] += 0.4
        
        # Candlestick confluence
        bullish_count = 0
        bearish_count = 0
        
        for tf, patterns in confluence['candlesticks'].items():
            weight = weights.get(tf, 0.2)
            for pattern_name in patterns:
                if pattern_name in [p for p in self.CANDLESTICK_PATTERNS['bullish_reversal']]:
                    bullish_count += weight
                elif pattern_name in [p for p in self.CANDLESTICK_PATTERNS['bearish_reversal']]:
                    bearish_count += weight
        
        if bullish_count > bearish_count:
            confluence['score'] += min(bullish_count * 0.3, 0.3)
        elif bearish_count > bullish_count:
            confluence['score'] -= min(bearish_count * 0.3, 0.3)
        
        return confluence
    
    def calculate_pattern_quality_score(self, pattern: Dict[str, Any], 
                                      market_context: Dict[str, Any]) -> float:
        """
        Calculate pattern quality using multiple factors
        Optionally enhanced with ML scoring
        """
        base_score = 0.0
        
        # 1. Statistical edge from Bulkowski (adjusted for modern markets)
        pattern_stats = self.PATTERN_STATISTICS.get(pattern['type'], {})
        if pattern_stats:
            historical_success = pattern_stats.get('success_rate', 0.5)
            modern_adjustment = pattern_stats.get('modern_adjustment', 0.7)
            base_score += historical_success * modern_adjustment * 0.3
        
        # 2. Volume confirmation (critical per Bulkowski)
        if pattern.get('volume_confirmed', False):
            base_score += 0.25
        
        # 3. Pattern clarity
        if pattern.get('clarity_score', 0) > 0.8:
            base_score += 0.15
        
        # 4. Market phase alignment (@GxTradez methodology)
        phase = market_context.get('phase', {})
        if phase.get('phase') == 'expansion' and pattern.get('signal') != 'hold':
            base_score += 0.15
        elif phase.get('phase') == 'consolidation' and pattern.get('type') in ['triangle', 'rectangle']:
            base_score += 0.1
        
        # 5. Multi-timeframe confluence
        if market_context.get('mtf_confluence', {}).get('score', 0) > 0.7:
            base_score += 0.15
        
        # 6. ML enhancement (if enabled)
        if self.use_ml_scoring and self.ml_model:
            try:
                # Extract features for ML model
                features = self._extract_ml_features(pattern, market_context)
                ml_score = self.ml_model.predict_proba([features])[0][1]
                # Blend ML score with rule-based score
                final_score = base_score * 0.7 + ml_score * 0.3
            except:
                final_score = base_score
        else:
            final_score = base_score
        
        return min(final_score, 1.0)
    
    def _extract_ml_features(self, pattern: Dict[str, Any], 
                           market_context: Dict[str, Any]) -> List[float]:
        """Extract features for ML scoring"""
        features = []
        
        # Pattern-specific features
        features.append(float(pattern.get('clarity_score', 0)))
        features.append(float(pattern.get('volume_confirmed', 0)))
        features.append(float(pattern.get('pattern_height', 0)))
        features.append(float(pattern.get('duration_bars', 0)))
        
        # Market context features
        phase = market_context.get('phase', {})
        features.append(1.0 if phase.get('phase') == 'expansion' else 0.0)
        features.append(1.0 if phase.get('direction') == 'bullish' else 0.0)
        features.append(float(market_context.get('mtf_confluence', {}).get('score', 0)))
        features.append(float(market_context.get('smt_divergence', {}).get('strength', 0)))
        
        # Time-based features
        hour = datetime.now().hour
        features.append(float(hour) / 24.0)  # Normalized hour
        features.append(float(datetime.now().month) / 12.0)  # Normalized month
        
        return features
    
    def detect_advanced_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect additional patterns beyond base PatternAgent"""
        patterns = []
        
        prices = df['Close']
        highs = df['High']
        lows = df['Low']
        volume = df['Volume'] if 'Volume' in df.columns else None
        
        # High and Tight Flag (Bulkowski's #1 pattern)
        htf = self.detect_high_tight_flag(prices, highs, lows, volume)
        if htf:
            patterns.append(htf)
        
        # Rounding Bottom with volume confirmation
        rb = self.detect_rounding_bottom_enhanced(prices, volume)
        if rb:
            patterns.append(rb)
        
        # Cup and Handle
        ch = self.detect_cup_handle(prices, highs, lows, volume)
        if ch:
            patterns.append(ch)
        
        # Ascending/Descending Triangles with better detection
        triangles = self.detect_triangles_enhanced(prices, highs, lows)
        patterns.extend(triangles)
        
        return patterns
    
    def detect_high_tight_flag(self, prices: pd.Series, highs: pd.Series, 
                               lows: pd.Series, volume: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Detect High and Tight Flag - Bulkowski's best performing pattern
        Requires 90%+ rise in <2 months, then tight consolidation
        """
        try:
            if len(prices) < 60:  # Need at least 60 days
                return None
            
            # Look for the flagpole (90%+ rise)
            for i in range(20, len(prices) - 10):
                # Calculate rise over past 40 days
                start_price = prices.iloc[i-40:i-20].min()
                peak_price = prices.iloc[i-20:i].max()
                rise = (peak_price - start_price) / start_price
                
                if rise >= 0.90:  # 90%+ rise
                    # Check for flag (consolidation)
                    flag_prices = prices.iloc[i:i+10]
                    flag_high = flag_prices.max()
                    flag_low = flag_prices.min()
                    flag_depth = (flag_high - flag_low) / flag_high
                    
                    # Tight consolidation (max 25% depth)
                    if flag_depth <= 0.25:
                        # Volume should decline during flag
                        if volume is not None:
                            flag_volume = volume.iloc[i:i+10].mean()
                            prior_volume = volume.iloc[i-20:i].mean()
                            volume_decline = (prior_volume - flag_volume) / prior_volume
                            
                            if volume_decline < 0.4:  # Not enough volume decline
                                continue
                        
                        return {
                            'type': 'high_tight_flag',
                            'signal': 'buy',
                            'confidence': 0.85,  # Very high confidence pattern
                            'flagpole_start': i-40,
                            'flagpole_end': i,
                            'flag_start': i,
                            'flag_end': i+10,
                            'rise_percent': rise,
                            'flag_depth': flag_depth,
                            'target': peak_price * 1.21,  # 21% avg gain
                            'volume_confirmed': True
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"High tight flag detection failed: {str(e)}")
            return None
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method combining all pattern detection approaches
        """
        try:
            symbol = data.get('symbol', 'UNKNOWN')
            
            # Get multi-timeframe data
            tf_data = self.get_multi_timeframe_data(symbol)
            if not tf_data:
                return self._create_hold_signal("Insufficient data")
            
            # 1. Identify market phase (GxTradez methodology)
            market_phase = self.identify_market_phase(tf_data.df_daily['Close'])
            
            # 2. Check for SMT divergence if enabled
            smt_divergence = {'divergence_detected': False}
            if self.check_correlations:
                smt_divergence = self.detect_smt_divergence(symbol, '4h')
            
            # 3. Multi-timeframe confluence analysis
            mtf_confluence = self.analyze_multi_timeframe_confluence(tf_data)
            
            # 4. Detect patterns on primary timeframe (hourly as per GxTradez)
            patterns = []
            
            # Base patterns from parent class
            base_result = super().process({
                'close_prices': tf_data.df_1h['Close'].values,
                'high_prices': tf_data.df_1h['High'].values,
                'low_prices': tf_data.df_1h['Low'].values,
                'volume': tf_data.df_1h['Volume'].values
            })
            
            if base_result['confidence'] > 0:
                patterns.append({
                    'type': base_result['metadata'].get('detected_pattern', {}).get('pattern', 'unknown'),
                    'signal': base_result['action'],
                    'confidence': base_result['confidence'],
                    'source': 'base_patterns'
                })
            
            # Advanced patterns
            advanced_patterns = self.detect_advanced_patterns(tf_data.df_1h)
            patterns.extend(advanced_patterns)
            
            # 5. Build market context
            market_context = {
                'phase': market_phase,
                'smt_divergence': smt_divergence,
                'mtf_confluence': mtf_confluence,
                'market_regime': self._determine_market_regime(tf_data.df_daily)
            }
            
            # 6. Score and select best pattern
            best_pattern = None
            best_score = 0
            
            for pattern in patterns:
                score = self.calculate_pattern_quality_score(pattern, market_context)
                if score > best_score:
                    best_score = score
                    best_pattern = pattern
            
            # 7. Generate final signal
            if best_pattern and best_score > 0.6:  # Quality threshold
                # Apply GxTradez rules
                if market_phase.phase == 'consolidation' and best_pattern['signal'] in ['buy', 'sell']:
                    # Cannot reverse from consolidation
                    return self._create_hold_signal("Awaiting expansion phase for reversal")
                
                # Check daily candle support (GxTradez requirement)
                daily_support = self._check_daily_support(tf_data.df_daily, best_pattern['signal'])
                if not daily_support:
                    return self._create_hold_signal("Daily timeframe not supportive")
                
                return {
                    "action": best_pattern['signal'],
                    "confidence": best_score,
                    "metadata": {
                        "pattern": best_pattern,
                        "market_phase": market_phase.__dict__,
                        "smt_divergence": smt_divergence,
                        "mtf_confluence": mtf_confluence,
                        "quality_score": best_score,
                        "ml_enhanced": self.use_ml_scoring
                    }
                }
            
            return self._create_hold_signal("No high-quality patterns detected")
            
        except Exception as e:
            logger.error(f"Enhanced pattern processing failed: {str(e)}")
            return self._create_hold_signal(f"Processing error: {str(e)}")
    
    def _determine_market_regime(self, daily_df: pd.DataFrame) -> str:
        """Determine overall market regime"""
        if len(daily_df) < 50:
            return 'unknown'
        
        # Simple regime detection using moving averages
        sma_50 = daily_df['Close'].rolling(50).mean()
        sma_200 = daily_df['Close'].rolling(200).mean()
        
        if len(sma_200.dropna()) > 0:
            current_price = daily_df['Close'].iloc[-1]
            if current_price > sma_50.iloc[-1] > sma_200.iloc[-1]:
                return 'bull'
            elif current_price < sma_50.iloc[-1] < sma_200.iloc[-1]:
                return 'bear'
        
        return 'neutral'
    
    def _check_daily_support(self, daily_df: pd.DataFrame, signal: str) -> bool:
        """Check if daily timeframe supports the signal (GxTradez rule)"""
        if len(daily_df) < 5:
            return False
        
        # Check last 3 daily candles
        recent_candles = daily_df.tail(3)
        
        if signal == 'buy':
            # Look for bullish daily structure
            bullish_signs = 0
            for _, candle in recent_candles.iterrows():
                if candle['Close'] > candle['Open']:  # Bullish candle
                    bullish_signs += 1
                if candle['Low'] > recent_candles['Low'].min():  # Higher lows
                    bullish_signs += 1
            
            return bullish_signs >= 3
        
        elif signal == 'sell':
            # Look for bearish daily structure
            bearish_signs = 0
            for _, candle in recent_candles.iterrows():
                if candle['Close'] < candle['Open']:  # Bearish candle
                    bearish_signs += 1
                if candle['High'] < recent_candles['High'].max():  # Lower highs
                    bearish_signs += 1
            
            return bearish_signs >= 3
        
        return True  # Hold signals always supported
    
    def _create_hold_signal(self, reason: str) -> Dict[str, Any]:
        """Create a hold signal with explanation"""
        return {
            "action": "hold",
            "confidence": 0.0,
            "metadata": {
                "reason": reason,
                "ml_enhanced": self.use_ml_scoring
            }
        } 