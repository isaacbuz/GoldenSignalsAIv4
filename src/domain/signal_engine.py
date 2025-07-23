"""
Signal Engine: Professional-grade options trading signal generator.
Combines AI predictions, technical analysis, and options-specific analysis.
"""
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import talib
from loguru import logger

from src.data.data_fetcher import MarketDataFetcher
from src.domain.analytics.performance_tracker import PerformanceTracker
from src.domain.risk_management.position_sizer import PositionSizer
from src.domain.trading.indicators import Indicators
from src.domain.trading.options_analysis import OptionsAnalysis
from src.domain.volatility_agent import VolatilityAgent
from src.ml.models.factory import ModelFactory
from src.ml.models.signal import TradingSignal


class SignalType(Enum):
    CALL = "CALL"
    PUT = "PUT"
    IRON_CONDOR = "IRON_CONDOR"
    BUTTERFLY = "BUTTERFLY"
    CALENDAR = "CALENDAR"
    DIAGONAL = "DIAGONAL"
    STRADDLE = "STRADDLE"
    STRANGLE = "STRANGLE"
    BULL_CALL_SPREAD = "BULL_CALL_SPREAD"
    BEAR_PUT_SPREAD = "BEAR_PUT_SPREAD"
    PUT_RATIO_SPREAD = "PUT_RATIO_SPREAD"
    CALL_RATIO_SPREAD = "CALL_RATIO_SPREAD"

class SignalStrength(Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"

class SignalEngine:
    def __init__(
        self, 
        factory: ModelFactory, 
        data_fetcher: MarketDataFetcher,
        user_id: str = None,
        risk_profile: str = "balanced"
    ):
        self.factory = factory
        self.data_fetcher = data_fetcher
        self.user_id = user_id
        self.risk_profile = risk_profile
        self.options_analysis = OptionsAnalysis(data_fetcher)
        self.position_sizer = PositionSizer()
        self.performance_tracker = PerformanceTracker()
        self._last_signal_time = {}
        self._signal_cache = {}
        self._cache_duration = timedelta(minutes=5)
        self._signal_history = deque(maxlen=100)
        self.volatility_agent = VolatilityAgent()
        
        logger.info({
            "message": "SignalEngine initialized",
            "user_id": user_id,
            "risk_profile": risk_profile
        })

    async def generate_trade_signal(self, symbol: str, account_size: float = 100000.0) -> TradingSignal:
        """Generate professional-grade options trading signal."""
        # Check cache first
        if self._is_cache_valid(symbol):
            return self._signal_cache[symbol]

        try:
            # Step 1: Fetch market data
            market_data = await self.data_fetcher.fetch_market_data(symbol)
            if market_data.empty:
                raise ValueError(f"No data found for symbol: {symbol}")

            # Step 2: Analyze market conditions
            market_analysis = self._analyze_market_conditions(symbol, market_data)
            
            # Step 3: Generate AI prediction
            ai_score = await self._compute_ai_score(symbol, market_data)
            
            # Step 4: Compute technical signals
            indicator_signals = self._compute_technical_signals(market_data)
            
            # Step 5: Analyze sentiment
            sentiment_score = self._analyze_sentiment(market_data.get('news_articles', []))

            # Step 5b: Volatility agent
            volatility_breakdown = self.volatility_agent.analyze(market_data)
            
            # Step 6: Identify potential setups
            setups = self._identify_setups(market_data, market_analysis)
            
            # Step 7: Validate each setup
            validated_setups = []
            for setup in setups:
                if self._validate_setup(setup, market_data):
                    validated_setups.append(setup)

            if not validated_setups:
                return self._create_default_signal(symbol, "No valid setups found")

            # Step 8: Rank setups by probability
            ranked_setups = self._rank_setups(validated_setups)
            
            # Step 9: Generate detailed signal for best setup
            best_setup = ranked_setups[0]
            signal = self._generate_detailed_signal(best_setup, market_data)
            
            # Step 10: Add risk management parameters
            signal.update(self._add_risk_parameters(signal, account_size))
            
            # Step 11: Add trade management guidelines
            signal.update(self._add_trade_management(signal))
            
            # Step 12: Calculate final confidence
            final_score, confidence = self._calculate_signal_confidence(
                ai_score, indicator_signals, sentiment_score
            )
            
            # Create comprehensive trading signal
            trading_signal = TradingSignal(
                symbol=symbol,
                action=signal["signal_type"].value,
                confidence=confidence,
                ai_score=ai_score,
                indicator_score=sum(indicator_signals.values()) / len(indicator_signals) if indicator_signals else 0.0,
                final_score=final_score,
                timestamp=datetime.now().isoformat(),
                risk_profile=self.risk_profile,
                indicators=list(indicator_signals.keys()),
                metadata={
                    "setup_details": signal["setup_details"],
                    "options_details": signal["options_details"],
                    "market_context": signal["market_context"],
                    "risk_metrics": signal["risk_metrics"],
                    "trade_management": signal["trade_management"],
                    "breakdown": [volatility_breakdown]
                }
            )
            
            # Update cache and history
            self._signal_cache[symbol] = trading_signal
            self._last_signal_time[symbol] = datetime.now()
            self._signal_history.append(trading_signal)
            
            logger.info(f"Generated signal for {symbol}: {trading_signal}")
            return trading_signal
            
        except Exception as e:
            logger.error(f"Signal generation error for {symbol}: {e}")
            return self._create_default_signal(symbol, str(e))

    async def _compute_ai_score(self, symbol: str, market_data: pd.DataFrame) -> float:
        """Compute AI-driven prediction score."""
        try:
            X_new_lstm = np.array(market_data["Close"].tail(60)).reshape(1, 60, 1)
            X_new_tree = market_data.tail(10)
            return await self.factory.get_ai_model().predict(symbol, X_new_lstm, X_new_tree) or 0.0
        except Exception as e:
            logger.warning(f"AI score computation failed: {e}")
            return 0.0

    def _compute_technical_signals(self, market_data: pd.DataFrame) -> Dict:
        """Compute technical indicator signals."""
        indicators = Indicators(market_data)
        return indicators.compute_regime_adjusted_signal(["RSI", "MACD"])

    def _analyze_sentiment(self, news_articles: List) -> float:
        """Analyze market sentiment from news."""
        return self.factory.get_sentiment_model().analyze(news_articles)

    def _calculate_signal_confidence(
        self, 
        ai_score: float, 
        indicator_signals: Dict, 
        sentiment_score: float
    ) -> tuple:
        """Calculate signal confidence with weighted scoring."""
        weights = {"ai": 0.5, "indicators": 0.3, "sentiment": 0.2}
        
        indicator_score = (
            sum(indicator_signals.values()) / len(indicator_signals)
            if indicator_signals else 0.0
        )

        final_score = (
            weights["ai"] * ai_score +
            weights["indicators"] * indicator_score +
            weights["sentiment"] * sentiment_score
        )

        # Risk profile adjustment
        confidence_multiplier = {
            "conservative": 0.7,
            "balanced": 1.0,
            "aggressive": 1.3
        }.get(self.risk_profile, 1.0)

        confidence = min(abs(final_score) * confidence_multiplier, 1.0)
        return final_score, confidence

    def _create_default_signal(
        self, 
        symbol: str, 
        error: Optional[str] = None
    ) -> TradingSignal:
        """Create a default hold signal."""
        return TradingSignal(
            symbol=symbol,
            action="HOLD",
            confidence=0.0,
            ai_score=0.0,
            indicator_score=0.0,
            final_score=0.0,
            timestamp=datetime.now().isoformat(),
            risk_profile=self.risk_profile,
            indicators=[],
            metadata={'error': error} if error else {}
        )

    def _validate_signal(self, signal: Dict, market_data: Dict) -> Tuple[bool, float]:
        """Enhanced signal validation with multiple factors."""
        validation_scores = []
        
        # Technical Validation (30%)
        technical_score = self._validate_technical_indicators(market_data)
        validation_scores.append((technical_score, 0.3))
        
        # Options-Specific Validation (30%)
        options_score = self._validate_options_metrics(signal, market_data)
        validation_scores.append((options_score, 0.3))
        
        # Market Context Validation (20%)
        market_score = self._validate_market_context(market_data)
        validation_scores.append((market_score, 0.2))
        
        # Volume Profile Validation (20%)
        volume_score = self._validate_volume_profile(market_data)
        validation_scores.append((volume_score, 0.2))
        
        # Calculate weighted score
        final_score = sum(score * weight for score, weight in validation_scores)
        
        return final_score >= 0.95, final_score

    def _validate_technical_indicators(self, market_data: Dict) -> float:
        """Validate technical indicators alignment."""
        df = pd.DataFrame(market_data)
        
        # Calculate technical indicators
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        
        # Trend Indicators
        sma_20 = talib.SMA(close, timeperiod=20)
        sma_50 = talib.SMA(close, timeperiod=50)
        ema_20 = talib.EMA(close, timeperiod=20)
        
        # Momentum Indicators
        rsi = talib.RSI(close, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(close)
        stoch_k, stoch_d = talib.STOCH(high, low, close)
        
        # Volume Indicators
        obv = talib.OBV(close, volume)
        ad = talib.AD(high, low, close, volume)
        
        # Volatility Indicators
        atr = talib.ATR(high, low, close, timeperiod=14)
        bbands_upper, bbands_middle, bbands_lower = talib.BBANDS(close)
        
        # Calculate alignment scores
        trend_score = self._calculate_trend_score(close, sma_20, sma_50, ema_20)
        momentum_score = self._calculate_momentum_score(rsi, macd, macd_signal, stoch_k, stoch_d)
        volume_score = self._calculate_volume_score(volume, obv, ad)
        volatility_score = self._calculate_volatility_score(close, atr, bbands_upper, bbands_lower)
        
        # Weighted average of all scores
        weights = {
            'trend': 0.3,
            'momentum': 0.3,
            'volume': 0.2,
            'volatility': 0.2
        }
        
        final_score = (
            trend_score * weights['trend'] +
            momentum_score * weights['momentum'] +
            volume_score * weights['volume'] +
            volatility_score * weights['volatility']
        )
        
        return final_score

    def _calculate_trend_score(self, close: np.ndarray, sma_20: np.ndarray, 
                             sma_50: np.ndarray, ema_20: np.ndarray) -> float:
        """Calculate trend alignment score."""
        # Check if price is above both SMAs
        price_above_sma = (close[-1] > sma_20[-1]) and (close[-1] > sma_50[-1])
        
        # Check if SMAs are aligned (20 above 50)
        sma_aligned = sma_20[-1] > sma_50[-1]
        
        # Check if EMA is above price (for pullback entries)
        ema_above_price = ema_20[-1] > close[-1]
        
        # Calculate trend strength
        trend_strength = abs(sma_20[-1] - sma_50[-1]) / sma_50[-1]
        
        # Combine factors
        score = 0.0
        if price_above_sma:
            score += 0.4
        if sma_aligned:
            score += 0.3
        if ema_above_price:
            score += 0.2
        score += min(trend_strength * 10, 0.1)  # Cap trend strength contribution
        
        return score

    def _calculate_momentum_score(self, rsi: np.ndarray, macd: np.ndarray,
                                macd_signal: np.ndarray, stoch_k: np.ndarray,
                                stoch_d: np.ndarray) -> float:
        """Calculate momentum alignment score."""
        # RSI conditions
        rsi_score = 0.0
        if 40 <= rsi[-1] <= 60:  # Neutral zone
            rsi_score = 0.3
        elif (rsi[-1] > rsi[-2] and rsi[-1] < 70) or (rsi[-1] < rsi[-2] and rsi[-1] > 30):
            rsi_score = 0.4
        
        # MACD conditions
        macd_score = 0.0
        if macd[-1] > macd_signal[-1] and macd[-1] > 0:
            macd_score = 0.3
        elif macd[-1] > macd_signal[-1]:
            macd_score = 0.2
        
        # Stochastic conditions
        stoch_score = 0.0
        if stoch_k[-1] > stoch_d[-1] and stoch_k[-1] < 80:
            stoch_score = 0.3
        elif stoch_k[-1] > stoch_d[-1]:
            stoch_score = 0.2
        
        return (rsi_score + macd_score + stoch_score) / 3

    def _calculate_volume_score(self, volume: np.ndarray, obv: np.ndarray,
                              ad: np.ndarray) -> float:
        """Calculate volume alignment score."""
        # Volume trend
        vol_ma = np.mean(volume[-5:])
        vol_trend = volume[-1] > vol_ma
        
        # OBV trend
        obv_trend = obv[-1] > obv[-2]
        
        # Accumulation/Distribution
        ad_trend = ad[-1] > ad[-2]
        
        score = 0.0
        if vol_trend:
            score += 0.4
        if obv_trend:
            score += 0.3
        if ad_trend:
            score += 0.3
            
        return score

    def _calculate_volatility_score(self, close: np.ndarray, atr: np.ndarray,
                                  bbands_upper: np.ndarray, bbands_lower: np.ndarray) -> float:
        """Calculate volatility alignment score."""
        # ATR relative to price
        atr_ratio = atr[-1] / close[-1]
        
        # Bollinger Band position
        bb_position = (close[-1] - bbands_lower[-1]) / (bbands_upper[-1] - bbands_lower[-1])
        
        # Volatility trend
        vol_trend = atr[-1] > atr[-2]
        
        score = 0.0
        if 0.01 <= atr_ratio <= 0.03:  # Ideal volatility range
            score += 0.4
        elif 0.03 < atr_ratio <= 0.05:  # Acceptable range
            score += 0.2
            
        if 0.2 <= bb_position <= 0.8:  # Price not at extremes
            score += 0.4
        elif 0.1 <= bb_position <= 0.9:  # Acceptable range
            score += 0.2
            
        if not vol_trend:  # Prefer decreasing volatility
            score += 0.2
            
        return score

    def _validate_options_metrics(self, signal: Dict, market_data: Dict) -> float:
        """Validate options-specific metrics."""
        # Get options chain data
        options_chain = self.data_fetcher.fetch_options_chain(signal['symbol'])
        
        if options_chain.empty:
            return 0.0
            
        # Calculate options metrics
        iv_percentile = self._calculate_iv_percentile(options_chain)
        iv_rank = self._calculate_iv_rank(options_chain)
        put_call_ratio = self._calculate_put_call_ratio(options_chain)
        volume_oi_ratio = self._calculate_volume_oi_ratio(options_chain)
        
        # Score each metric
        iv_score = self._score_iv_metrics(iv_percentile, iv_rank)
        pcr_score = self._score_put_call_ratio(put_call_ratio)
        volume_score = self._score_volume_oi(volume_oi_ratio)
        
        # Weighted average
        weights = {
            'iv': 0.4,
            'pcr': 0.3,
            'volume': 0.3
        }
        
        return (
            iv_score * weights['iv'] +
            pcr_score * weights['pcr'] +
            volume_score * weights['volume']
        )

    def _calculate_iv_percentile(self, options_chain: pd.DataFrame) -> float:
        """Calculate IV percentile."""
        ivs = options_chain['implied_volatility'].values
        return np.percentile(ivs, 50)  # Median IV

    def _calculate_iv_rank(self, options_chain: pd.DataFrame) -> float:
        """Calculate IV rank."""
        ivs = options_chain['implied_volatility'].values
        return (ivs[-1] - min(ivs)) / (max(ivs) - min(ivs))

    def _calculate_put_call_ratio(self, options_chain: pd.DataFrame) -> float:
        """Calculate put-call ratio."""
        put_volume = options_chain[options_chain['type'] == 'put']['volume'].sum()
        call_volume = options_chain[options_chain['type'] == 'call']['volume'].sum()
        return put_volume / call_volume if call_volume > 0 else float('inf')

    def _calculate_volume_oi_ratio(self, options_chain: pd.DataFrame) -> float:
        """Calculate volume to open interest ratio."""
        return options_chain['volume'].sum() / options_chain['open_interest'].sum()

    def _score_iv_metrics(self, iv_percentile: float, iv_rank: float) -> float:
        """Score IV metrics."""
        # Prefer moderate IV
        iv_percentile_score = 1.0 - abs(0.5 - iv_percentile) * 2
        iv_rank_score = 1.0 - abs(0.5 - iv_rank) * 2
        
        return (iv_percentile_score + iv_rank_score) / 2

    def _score_put_call_ratio(self, pcr: float) -> float:
        """Score put-call ratio."""
        if 0.7 <= pcr <= 1.3:  # Neutral range
            return 1.0
        elif 0.5 <= pcr <= 1.5:  # Acceptable range
            return 0.7
        else:
            return 0.3

    def _score_volume_oi(self, ratio: float) -> float:
        """Score volume to open interest ratio."""
        if 0.5 <= ratio <= 2.0:  # Ideal range
            return 1.0
        elif 0.2 <= ratio <= 5.0:  # Acceptable range
            return 0.7
        else:
            return 0.3

    def _validate_market_context(self, market_data: Dict) -> float:
        """Validate market context and conditions."""
        # Get market indices data
        indices = self.data_fetcher.fetch_market_indices()
        
        # Calculate market trend
        market_trend = self._calculate_market_trend(indices)
        
        # Calculate sector strength
        sector_strength = self._calculate_sector_strength(signal['symbol'])
        
        # Calculate market breadth
        market_breadth = self._calculate_market_breadth(indices)
        
        # Weighted average
        weights = {
            'trend': 0.4,
            'sector': 0.3,
            'breadth': 0.3
        }
        
        return (
            market_trend * weights['trend'] +
            sector_strength * weights['sector'] +
            market_breadth * weights['breadth']
        )

    def _validate_volume_profile(self, market_data: Dict) -> float:
        """Validate volume profile and liquidity."""
        df = pd.DataFrame(market_data)
        
        # Calculate volume profile
        volume_profile = self._calculate_volume_profile(df)
        
        # Calculate liquidity metrics
        liquidity = self._calculate_liquidity_metrics(df)
        
        # Calculate volume trend
        volume_trend = self._calculate_volume_trend(df)
        
        # Weighted average
        weights = {
            'profile': 0.4,
            'liquidity': 0.3,
            'trend': 0.3
        }
        
        return (
            volume_profile * weights['profile'] +
            liquidity * weights['liquidity'] +
            volume_trend * weights['trend']
        )

    def _calculate_position_size(self, signal: Dict, account_size: float) -> Dict:
        """Calculate optimal position size based on signal confidence and risk metrics."""
        return self.position_sizer.calculate_position(
            signal=signal,
            account_size=account_size,
            risk_metrics=self._get_risk_metrics(signal)
        )

    def _get_risk_metrics(self, signal: Dict) -> Dict:
        """Calculate risk metrics for position sizing."""
        return {
            "volatility": self._calculate_volatility(signal),
            "correlation": self._calculate_correlation(signal),
            "max_loss": self._calculate_max_loss(signal)
        }

    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached signal is still valid."""
        if symbol not in self._last_signal_time:
            return False
        return datetime.now() - self._last_signal_time[symbol] < self._cache_duration

    def _analyze_market_conditions(self, symbol: str, market_data: pd.DataFrame) -> Dict:
        """Analyze current market conditions for signal generation."""
        return {
            "trend": self._analyze_trend(market_data),
            "volatility": self._analyze_volatility(market_data),
            "support_resistance": self._find_support_resistance(market_data),
            "options_skew": self._analyze_options_skew(symbol),
            "earnings_impact": self._check_earnings_impact(symbol),
            "market_regime": self._determine_market_regime(market_data)
        }

    def _identify_setups(self, market_data: pd.DataFrame, market_analysis: Dict) -> List[Dict]:
        """Identify potential trading setups based on market conditions."""
        setups = []
        
        # Trend-based setups
        if market_analysis["trend"]["strength"] > 0.7:
            setups.extend(self._identify_trend_setups(market_data, market_analysis))
            
        # Volatility-based setups
        if market_analysis["volatility"]["is_high"]:
            setups.extend(self._identify_volatility_setups(market_data, market_analysis))
            
        # Earnings-based setups
        if market_analysis["earnings_impact"]["upcoming"]:
            setups.extend(self._identify_earnings_setups(market_data, market_analysis))
            
        # Support/Resistance setups
        setups.extend(self._identify_sr_setups(market_data, market_analysis))
        
        return setups

    def _validate_setup(self, setup: Dict, market_data: pd.DataFrame) -> bool:
        """Validate a trading setup with multiple criteria."""
        # Technical validation
        tech_score = self._validate_technical_indicators(market_data)
        
        # Options validation
        options_score = self._validate_options_metrics(setup, market_data)
        
        # Market context validation
        market_score = self._validate_market_context(market_data)
        
        # Volume validation
        volume_score = self._validate_volume_profile(market_data)
        
        # Combine scores with weighted importance
        final_score = (
            tech_score * 0.3 +
            options_score * 0.3 +
            market_score * 0.2 +
            volume_score * 0.2
        )
        
        return final_score >= 0.95

    def _generate_detailed_signal(self, setup: Dict, market_data: pd.DataFrame) -> Dict:
        """Generate detailed trading signal with all necessary information."""
        signal = {
            "symbol": setup["symbol"],
            "signal_type": setup["type"],
            "strength": self._determine_signal_strength(setup["confidence"]),
            "confidence": setup["confidence"],
            "timestamp": datetime.now().isoformat(),
            "setup_details": {
                "entry_price": setup["entry_price"],
                "target_price": setup["target_price"],
                "stop_loss": setup["stop_loss"],
                "timeframe": setup["timeframe"],
                "probability": setup["probability"]
            },
            "options_details": self._generate_options_details(setup),
            "market_context": self._generate_market_context(setup),
            "risk_metrics": self._calculate_risk_metrics(setup),
            "trade_management": self._generate_trade_management(setup)
        }
        
        return signal

    def _generate_options_details(self, setup: Dict) -> Dict:
        """Generate detailed options information for the signal."""
        return {
            "strike_selection": self._select_optimal_strikes(setup),
            "expiration_selection": self._select_optimal_expiration(setup),
            "greeks": self._calculate_greeks(setup),
            "iv_analysis": self._analyze_implied_volatility(setup),
            "options_chain": self._analyze_options_chain(setup)
        }

    def _generate_market_context(self, setup: Dict) -> Dict:
        """Generate market context information for the signal."""
        return {
            "trend_analysis": self._analyze_trend_context(setup),
            "volatility_regime": self._analyze_volatility_regime(setup),
            "sector_strength": self._analyze_sector_strength(setup),
            "market_breadth": self._analyze_market_breadth(setup),
            "correlation_analysis": self._analyze_correlations(setup)
        }

    def _generate_trade_management(self, setup: Dict) -> Dict:
        """Generate trade management guidelines."""
        return {
            "entry_rules": self._generate_entry_rules(setup),
            "exit_rules": self._generate_exit_rules(setup),
            "adjustment_rules": self._generate_adjustment_rules(setup),
            "risk_management": self._generate_risk_rules(setup),
            "position_sizing": self._generate_position_sizing(setup)
        }

    def _determine_signal_strength(self, confidence: float) -> str:
        """Determine signal strength based on confidence score."""
        if confidence >= 0.85:
            return SignalStrength.STRONG
        elif confidence >= 0.70:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK

    def _update_signal_history(self, symbol: str, signal: Dict) -> None:
        """Update signal history for performance tracking."""
        if symbol not in self._signal_history:
            self._signal_history[symbol] = []
            
        self._signal_history[symbol].append({
            "signal": signal,
            "timestamp": datetime.now(),
            "status": "pending"
        })
        
        # Keep only last 100 signals
        self._signal_history[symbol] = self._signal_history[symbol][-100:]

    def _calculate_market_trend(self, indices: Dict) -> float:
        """Calculate overall market trend score."""
        # Get major indices
        sp500 = indices.get('SPY', pd.DataFrame())
        nasdaq = indices.get('QQQ', pd.DataFrame())
        russell = indices.get('IWM', pd.DataFrame())
        
        if sp500.empty or nasdaq.empty or russell.empty:
            return 0.5  # Neutral if data unavailable
            
        # Calculate 20-day moving averages
        sp500_ma = sp500['Close'].rolling(20).mean()
        nasdaq_ma = nasdaq['Close'].rolling(20).mean()
        russell_ma = russell['Close'].rolling(20).mean()
        
        # Calculate trend scores
        sp500_score = 1.0 if sp500['Close'].iloc[-1] > sp500_ma.iloc[-1] else 0.0
        nasdaq_score = 1.0 if nasdaq['Close'].iloc[-1] > nasdaq_ma.iloc[-1] else 0.0
        russell_score = 1.0 if russell['Close'].iloc[-1] > russell_ma.iloc[-1] else 0.0
        
        # Weighted average (SPY has higher weight)
        return (sp500_score * 0.5 + nasdaq_score * 0.3 + russell_score * 0.2)

    def _calculate_sector_strength(self, symbol: str) -> float:
        """Calculate sector strength score."""
        # Get sector data
        sector_data = self.data_fetcher.fetch_sector_data(symbol)
        
        if sector_data.empty:
            return 0.5  # Neutral if data unavailable
            
        # Calculate sector performance
        sector_perf = sector_data['Close'].pct_change(20)
        sector_rank = sector_perf.rank(pct=True)
        
        # Get symbol's sector
        symbol_sector = self.data_fetcher.get_symbol_sector(symbol)
        
        if symbol_sector not in sector_rank:
            return 0.5
            
        # Convert rank to score (0-1)
        return sector_rank[symbol_sector]

    def _calculate_market_breadth(self, indices: Dict) -> float:
        """Calculate market breadth score."""
        # Get advance-decline data
        adv_dec = self.data_fetcher.fetch_advance_decline()
        
        if adv_dec.empty:
            return 0.5  # Neutral if data unavailable
            
        # Calculate advance-decline ratio
        adv_dec_ratio = adv_dec['Advances'] / adv_dec['Declines']
        
        # Calculate 10-day moving average
        adv_dec_ma = adv_dec_ratio.rolling(10).mean()
        
        # Score based on current ratio vs moving average
        current_ratio = adv_dec_ratio.iloc[-1]
        ma_ratio = adv_dec_ma.iloc[-1]
        
        if current_ratio > ma_ratio * 1.1:  # Strong breadth
            return 1.0
        elif current_ratio > ma_ratio:  # Positive breadth
            return 0.7
        elif current_ratio > ma_ratio * 0.9:  # Neutral breadth
            return 0.5
        else:  # Weak breadth
            return 0.3

    def _calculate_volume_profile(self, df: pd.DataFrame) -> float:
        """Calculate volume profile score."""
        # Calculate price levels
        price_range = df['High'].max() - df['Low'].min()
        num_levels = 10
        level_size = price_range / num_levels
        
        # Create price levels
        levels = np.arange(df['Low'].min(), df['High'].max(), level_size)
        
        # Calculate volume at each level
        volume_profile = []
        for i in range(len(levels) - 1):
            level_volume = df[
                (df['Close'] >= levels[i]) & 
                (df['Close'] < levels[i + 1])
            ]['Volume'].sum()
            volume_profile.append(level_volume)
            
        # Normalize volume profile
        total_volume = sum(volume_profile)
        if total_volume == 0:
            return 0.5
            
        volume_profile = [v / total_volume for v in volume_profile]
        
        # Calculate current price level
        current_price = df['Close'].iloc[-1]
        current_level = int((current_price - df['Low'].min()) / level_size)
        
        # Score based on volume at current level
        if current_level >= len(volume_profile):
            return 0.5
            
        volume_score = volume_profile[current_level]
        
        # Convert to 0-1 score
        return min(volume_score * 5, 1.0)  # Scale up and cap at 1.0

    def _calculate_liquidity_metrics(self, df: pd.DataFrame) -> float:
        """Calculate liquidity metrics score."""
        # Calculate average daily volume
        avg_volume = df['Volume'].mean()
        
        # Calculate average spread
        avg_spread = (df['High'] - df['Low']).mean() / df['Close'].mean()
        
        # Calculate volume trend
        volume_ma = df['Volume'].rolling(20).mean()
        volume_trend = df['Volume'].iloc[-1] / volume_ma.iloc[-1]
        
        # Score each metric
        volume_score = min(avg_volume / 1000000, 1.0)  # Cap at 1M shares
        spread_score = 1.0 - min(avg_spread * 100, 0.5)  # Penalize wide spreads
        trend_score = min(volume_trend, 2.0) / 2.0  # Cap at 2x average
        
        # Weighted average
        weights = {
            'volume': 0.4,
            'spread': 0.3,
            'trend': 0.3
        }
        
        return (
            volume_score * weights['volume'] +
            spread_score * weights['spread'] +
            trend_score * weights['trend']
        )

    def _calculate_volume_trend(self, df: pd.DataFrame) -> float:
        """Calculate volume trend score."""
        # Calculate 5-day and 20-day volume moving averages
        vol_ma5 = df['Volume'].rolling(5).mean()
        vol_ma20 = df['Volume'].rolling(20).mean()
        
        # Calculate volume trend
        short_trend = df['Volume'].iloc[-1] / vol_ma5.iloc[-1]
        long_trend = df['Volume'].iloc[-1] / vol_ma20.iloc[-1]
        
        # Score based on both trends
        if short_trend > 1.2 and long_trend > 1.1:  # Strong uptrend
            return 1.0
        elif short_trend > 1.1 and long_trend > 1.0:  # Moderate uptrend
            return 0.7
        elif short_trend > 1.0 and long_trend > 0.9:  # Slight uptrend
            return 0.5
        else:  # Downtrend
            return 0.3

    def _identify_trend_setups(self, market_data: pd.DataFrame, market_analysis: Dict) -> List[Dict]:
        """Identify trend-based options setups."""
        setups = []
        trend = market_analysis["trend"]
        
        if trend["direction"] == "up" and trend["strength"] > 0.7:
            # Bullish setups
            setups.extend([
                self._create_call_setup(market_data, trend),
                self._create_bull_call_spread_setup(market_data, trend),
                self._create_diagonal_setup(market_data, trend)
            ])
        elif trend["direction"] == "down" and trend["strength"] > 0.7:
            # Bearish setups
            setups.extend([
                self._create_put_setup(market_data, trend),
                self._create_bear_put_spread_setup(market_data, trend),
                self._create_diagonal_setup(market_data, trend)
            ])
            
        return setups

    def _identify_volatility_setups(self, market_data: pd.DataFrame, market_analysis: Dict) -> List[Dict]:
        """Identify volatility-based options setups."""
        setups = []
        volatility = market_analysis["volatility"]
        
        if volatility["is_high"]:
            # High volatility setups
            setups.extend([
                self._create_iron_condor_setup(market_data, volatility),
                self._create_butterfly_setup(market_data, volatility),
                self._create_straddle_setup(market_data, volatility)
            ])
        else:
            # Low volatility setups
            setups.extend([
                self._create_calendar_spread_setup(market_data, volatility),
                self._create_diagonal_setup(market_data, volatility)
            ])
            
        return setups

    def _identify_earnings_setups(self, market_data: pd.DataFrame, market_analysis: Dict) -> List[Dict]:
        """Identify earnings-based options setups."""
        setups = []
        earnings = market_analysis["earnings_impact"]
        
        if earnings["upcoming"]:
            # Earnings setups
            setups.extend([
                self._create_earnings_straddle_setup(market_data, earnings),
                self._create_earnings_iron_condor_setup(market_data, earnings),
                self._create_earnings_butterfly_setup(market_data, earnings)
            ])
            
        return setups

    def _identify_sr_setups(self, market_data: pd.DataFrame, market_analysis: Dict) -> List[Dict]:
        """Identify support/resistance based options setups."""
        setups = []
        sr_levels = market_analysis["support_resistance"]
        current_price = market_data["Close"].iloc[-1]
        
        # Find nearest support and resistance
        nearest_support = max([s for s in sr_levels["support"] if s < current_price], default=None)
        nearest_resistance = min([r for r in sr_levels["resistance"] if r > current_price], default=None)
        
        if nearest_support and nearest_resistance:
            # Range-bound setups
            setups.extend([
                self._create_iron_condor_setup(market_data, {
                    "support": nearest_support,
                    "resistance": nearest_resistance
                }),
                self._create_butterfly_setup(market_data, {
                    "support": nearest_support,
                    "resistance": nearest_resistance
                })
            ])
            
        return setups

    def _create_call_setup(self, market_data: pd.DataFrame, trend: Dict) -> Dict:
        """Create a call options setup."""
        return {
            "type": SignalType.CALL,
            "symbol": market_data["Symbol"].iloc[0],
            "confidence": trend["strength"],
            "entry_price": market_data["Close"].iloc[-1],
            "target_price": self._calculate_target_price(market_data, "up"),
            "stop_loss": self._calculate_stop_loss(market_data, "up"),
            "timeframe": "1-2 weeks",
            "probability": self._calculate_probability(market_data, "up"),
            "setup_rules": {
                "entry": "Buy calls when price pulls back to 20 EMA",
                "exit": "Take profit at target or if trend weakens",
                "adjustment": "Roll up if price moves strongly in favor"
            }
        }

    def _create_put_setup(self, market_data: pd.DataFrame, trend: Dict) -> Dict:
        """Create a put options setup."""
        return {
            "type": SignalType.PUT,
            "symbol": market_data["Symbol"].iloc[0],
            "confidence": trend["strength"],
            "entry_price": market_data["Close"].iloc[-1],
            "target_price": self._calculate_target_price(market_data, "down"),
            "stop_loss": self._calculate_stop_loss(market_data, "down"),
            "timeframe": "1-2 weeks",
            "probability": self._calculate_probability(market_data, "down"),
            "setup_rules": {
                "entry": "Buy puts when price rallies to resistance",
                "exit": "Take profit at target or if trend weakens",
                "adjustment": "Roll down if price moves strongly in favor"
            }
        }

    def _create_iron_condor_setup(self, market_data: pd.DataFrame, volatility: Dict) -> Dict:
        """Create an iron condor setup."""
        return {
            "type": SignalType.IRON_CONDOR,
            "symbol": market_data["Symbol"].iloc[0],
            "confidence": 0.85,
            "entry_price": market_data["Close"].iloc[-1],
            "target_price": None,  # Iron condors have defined risk
            "stop_loss": None,  # Iron condors have defined risk
            "timeframe": "30-45 days",
            "probability": 0.75,
            "setup_rules": {
                "entry": "Sell OTM put spread and OTM call spread",
                "exit": "Close at 50% max profit or if price approaches short strikes",
                "adjustment": "Roll untested side if price moves towards it"
            }
        }

    def _create_butterfly_setup(self, market_data: pd.DataFrame, volatility: Dict) -> Dict:
        """Create a butterfly setup."""
        return {
            "type": SignalType.BUTTERFLY,
            "symbol": market_data["Symbol"].iloc[0],
            "confidence": 0.80,
            "entry_price": market_data["Close"].iloc[-1],
            "target_price": None,  # Butterflies have defined risk
            "stop_loss": None,  # Butterflies have defined risk
            "timeframe": "30-45 days",
            "probability": 0.70,
            "setup_rules": {
                "entry": "Buy butterfly at expected price target",
                "exit": "Close at 50% max profit or if price moves away from center",
                "adjustment": "Roll to new center if price moves significantly"
            }
        }

    def _create_calendar_spread_setup(self, market_data: pd.DataFrame, volatility: Dict) -> Dict:
        """Create a calendar spread setup."""
        return {
            "type": SignalType.CALENDAR,
            "symbol": market_data["Symbol"].iloc[0],
            "confidence": 0.75,
            "entry_price": market_data["Close"].iloc[-1],
            "target_price": None,  # Calendar spreads profit from time decay
            "stop_loss": None,  # Calendar spreads have defined risk
            "timeframe": "30-45 days",
            "probability": 0.65,
            "setup_rules": {
                "entry": "Sell near-term option, buy longer-term option",
                "exit": "Close when near-term option expires or if volatility increases",
                "adjustment": "Roll to new strikes if price moves significantly"
            }
        }

    def _create_diagonal_setup(self, market_data: pd.DataFrame, trend: Dict) -> Dict:
        """Create a diagonal spread setup."""
        return {
            "type": SignalType.DIAGONAL,
            "symbol": market_data["Symbol"].iloc[0],
            "confidence": trend["strength"],
            "entry_price": market_data["Close"].iloc[-1],
            "target_price": self._calculate_target_price(market_data, trend["direction"]),
            "stop_loss": self._calculate_stop_loss(market_data, trend["direction"]),
            "timeframe": "45-60 days",
            "probability": self._calculate_probability(market_data, trend["direction"]),
            "setup_rules": {
                "entry": "Sell near-term option, buy longer-term option at different strike",
                "exit": "Close when near-term option expires or if trend reverses",
                "adjustment": "Roll to new strikes if price moves significantly"
            }
        }

    def _create_straddle_setup(self, market_data: pd.DataFrame, volatility: Dict) -> Dict:
        """Create a straddle setup."""
        return {
            "type": SignalType.STRADDLE,
            "symbol": market_data["Symbol"].iloc[0],
            "confidence": 0.85,
            "entry_price": market_data["Close"].iloc[-1],
            "target_price": None,  # Straddles profit from large moves in either direction
            "stop_loss": None,  # Straddles have defined risk
            "timeframe": "7-14 days",
            "probability": 0.60,
            "setup_rules": {
                "entry": "Buy both ATM call and put",
                "exit": "Close when price makes a significant move or if volatility decreases",
                "adjustment": "Roll to new strikes if price moves significantly"
            }
        }

    def _calculate_target_price(self, market_data: pd.DataFrame, direction: str) -> float:
        """Calculate target price based on ATR and direction."""
        atr = talib.ATR(market_data["High"], market_data["Low"], 
                        market_data["Close"], timeperiod=14).iloc[-1]
        current_price = market_data["Close"].iloc[-1]
        
        if direction == "up":
            return round(current_price + (atr * 2), 2)
        else:
            return round(current_price - (atr * 2), 2)

    def _calculate_stop_loss(self, market_data: pd.DataFrame, direction: str) -> float:
        """Calculate stop loss based on ATR and direction."""
        atr = talib.ATR(market_data["High"], market_data["Low"], 
                        market_data["Close"], timeperiod=14).iloc[-1]
        current_price = market_data["Close"].iloc[-1]
        
        if direction == "up":
            return round(current_price - atr, 2)
        else:
            return round(current_price + atr, 2)

    def _calculate_probability(self, market_data: pd.DataFrame, direction: str) -> float:
        """Calculate probability of success based on historical patterns."""
        # This is a simplified version - in practice, you'd want to use more sophisticated
        # statistical analysis and machine learning models
        if direction == "up":
            return 0.75
        else:
            return 0.70

    def _analyze_trend(self, market_data: pd.DataFrame) -> Dict:
        """Analyze market trend and strength."""
        # Calculate moving averages
        sma_20 = talib.SMA(market_data["Close"], timeperiod=20)
        sma_50 = talib.SMA(market_data["Close"], timeperiod=50)
        ema_20 = talib.EMA(market_data["Close"], timeperiod=20)
        
        # Calculate trend direction
        current_price = market_data["Close"].iloc[-1]
        direction = "up" if current_price > sma_20.iloc[-1] > sma_50.iloc[-1] else "down"
        
        # Calculate trend strength
        price_to_sma20 = abs(current_price - sma_20.iloc[-1]) / sma_20.iloc[-1]
        sma20_to_sma50 = abs(sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
        
        strength = min((price_to_sma20 + sma20_to_sma50) * 10, 1.0)
        
        return {
            "direction": direction,
            "strength": strength,
            "sma_20": sma_20.iloc[-1],
            "sma_50": sma_50.iloc[-1],
            "ema_20": ema_20.iloc[-1]
        }

    def _analyze_volatility(self, market_data: pd.DataFrame) -> Dict:
        """Analyze market volatility conditions."""
        # Calculate ATR
        atr = talib.ATR(market_data["High"], market_data["Low"], 
                        market_data["Close"], timeperiod=14)
        
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(market_data["Close"])
        
        # Calculate historical volatility
        returns = market_data["Close"].pct_change()
        hist_vol = returns.std() * np.sqrt(252)  # Annualized
        
        # Determine if volatility is high
        current_price = market_data["Close"].iloc[-1]
        bb_width = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]
        is_high = bb_width > 0.05 or hist_vol > 0.3
        
        return {
            "is_high": is_high,
            "atr": atr.iloc[-1],
            "bb_width": bb_width,
            "historical_volatility": hist_vol,
            "bb_upper": bb_upper.iloc[-1],
            "bb_lower": bb_lower.iloc[-1]
        }

    def _find_support_resistance(self, market_data: pd.DataFrame) -> Dict:
        """Find key support and resistance levels."""
        # Calculate pivot points
        high = market_data["High"].iloc[-1]
        low = market_data["Low"].iloc[-1]
        close = market_data["Close"].iloc[-1]
        
        # Calculate pivot point
        pivot = (high + low + close) / 3
        
        # Calculate support and resistance levels
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        
        # Find additional levels using price action
        price_levels = self._find_price_levels(market_data)
        
        return {
            "support": [s2, s1] + price_levels["support"],
            "resistance": [r1, r2] + price_levels["resistance"],
            "pivot": pivot
        }

    def _find_price_levels(self, market_data: pd.DataFrame) -> Dict:
        """Find additional support and resistance levels using price action."""
        # Use the last 100 days of data
        recent_data = market_data.tail(100)
        
        # Find local minima and maxima
        window = 5
        local_min = []
        local_max = []
        
        for i in range(window, len(recent_data) - window):
            if all(recent_data["Low"].iloc[i] <= recent_data["Low"].iloc[i-j] for j in range(1, window+1)) and \
               all(recent_data["Low"].iloc[i] <= recent_data["Low"].iloc[i+j] for j in range(1, window+1)):
                local_min.append(recent_data["Low"].iloc[i])
                
            if all(recent_data["High"].iloc[i] >= recent_data["High"].iloc[i-j] for j in range(1, window+1)) and \
               all(recent_data["High"].iloc[i] >= recent_data["High"].iloc[i+j] for j in range(1, window+1)):
                local_max.append(recent_data["High"].iloc[i])
        
        # Cluster nearby levels
        support_levels = self._cluster_price_levels(local_min)
        resistance_levels = self._cluster_price_levels(local_max)
        
        return {
            "support": support_levels,
            "resistance": resistance_levels
        }

    def _cluster_price_levels(self, levels: List[float], threshold: float = 0.02) -> List[float]:
        """Cluster nearby price levels."""
        if not levels:
            return []
            
        # Sort levels
        levels = sorted(levels)
        
        # Cluster levels that are within threshold of each other
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if (level - current_cluster[-1]) / current_cluster[-1] <= threshold:
                current_cluster.append(level)
            else:
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]
                
        clusters.append(sum(current_cluster) / len(current_cluster))
        
        return clusters

    def _analyze_options_skew(self, symbol: str) -> Dict:
        """Analyze options implied volatility skew."""
        # Get options chain data
        options_chain = self.data_fetcher.fetch_options_chain(symbol)
        
        if options_chain.empty:
            return {
                "skew": "neutral",
                "put_skew": 0.0,
                "call_skew": 0.0
            }
            
        # Calculate IV for different strikes
        atm_strike = options_chain["strike"].iloc[len(options_chain)//2]
        
        # Calculate put skew (OTM puts vs ATM)
        otm_puts = options_chain[
            (options_chain["type"] == "put") & 
            (options_chain["strike"] < atm_strike)
        ]
        put_skew = otm_puts["implied_volatility"].mean() - \
                   options_chain[options_chain["strike"] == atm_strike]["implied_volatility"].iloc[0]
        
        # Calculate call skew (OTM calls vs ATM)
        otm_calls = options_chain[
            (options_chain["type"] == "call") & 
            (options_chain["strike"] > atm_strike)
        ]
        call_skew = otm_calls["implied_volatility"].mean() - \
                    options_chain[options_chain["strike"] == atm_strike]["implied_volatility"].iloc[0]
        
        # Determine overall skew
        if put_skew > 0.1:
            skew = "put_skewed"
        elif call_skew > 0.1:
            skew = "call_skewed"
        else:
            skew = "neutral"
            
        return {
            "skew": skew,
            "put_skew": put_skew,
            "call_skew": call_skew
        }

    def _check_earnings_impact(self, symbol: str) -> Dict:
        """Check for upcoming earnings and their potential impact."""
        # Get earnings data
        earnings_data = self.data_fetcher.fetch_earnings_data(symbol)
        
        if earnings_data.empty:
            return {
                "upcoming": False,
                "date": None,
                "expected_move": None
            }
            
        # Check for upcoming earnings
        next_earnings = earnings_data[earnings_data["date"] > datetime.now()].iloc[0]
        
        if next_earnings.empty:
            return {
                "upcoming": False,
                "date": None,
                "expected_move": None
            }
            
        # Calculate expected move based on options pricing
        expected_move = self._calculate_expected_move(symbol, next_earnings["date"])
        
        return {
            "upcoming": True,
            "date": next_earnings["date"],
            "expected_move": expected_move
        }

    def _calculate_expected_move(self, symbol: str, earnings_date: datetime) -> float:
        """Calculate expected move based on options pricing."""
        # Get options chain for earnings expiration
        options_chain = self.data_fetcher.fetch_options_chain(symbol, expiration=earnings_date)
        
        if options_chain.empty:
            return None
            
        # Calculate ATM straddle price
        atm_strike = options_chain["strike"].iloc[len(options_chain)//2]
        atm_call = options_chain[
            (options_chain["type"] == "call") & 
            (options_chain["strike"] == atm_strike)
        ]
        atm_put = options_chain[
            (options_chain["type"] == "put") & 
            (options_chain["strike"] == atm_strike)
        ]
        
        if atm_call.empty or atm_put.empty:
            return None
            
        # Expected move is approximately the price of the ATM straddle
        expected_move = (atm_call["last_price"].iloc[0] + atm_put["last_price"].iloc[0]) / \
                       atm_strike
        
        return expected_move

    def _determine_market_regime(self, market_data: pd.DataFrame) -> Dict:
        """Determine the current market regime."""
        # Calculate volatility regime
        returns = market_data["Close"].pct_change()
        volatility = returns.std() * np.sqrt(252)
        
        # Calculate trend regime
        sma_20 = talib.SMA(market_data["Close"], timeperiod=20)
        sma_50 = talib.SMA(market_data["Close"], timeperiod=50)
        
        # Calculate momentum regime
        rsi = talib.RSI(market_data["Close"], timeperiod=14)
        
        # Determine regime
        if volatility > 0.3:
            regime = "high_volatility"
        elif volatility < 0.15:
            regime = "low_volatility"
        else:
            regime = "normal_volatility"
            
        if sma_20.iloc[-1] > sma_50.iloc[-1]:
            trend = "uptrend"
        else:
            trend = "downtrend"
            
        if rsi.iloc[-1] > 70:
            momentum = "overbought"
        elif rsi.iloc[-1] < 30:
            momentum = "oversold"
        else:
            momentum = "neutral"
            
        return {
            "regime": regime,
            "trend": trend,
            "momentum": momentum,
            "volatility": volatility
        }

    def get_signal_history(self) -> List[Dict]:
        """Get recent signal history."""
        return list(self._signal_history)

    def get_performance_metrics(self) -> Dict:
        """Get signal performance metrics."""
        return self.performance_tracker.get_performance_metrics()
