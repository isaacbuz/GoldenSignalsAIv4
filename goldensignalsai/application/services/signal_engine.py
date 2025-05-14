# application/services/signal_engine.py
# Purpose: Generates trading signals by combining AI predictions with technical indicators,
# tailored for options trading with regime-adjusted signals.

# application/services/signal_engine.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from loguru import logger

from goldensignalsai.domain.trading.indicators import Indicators
from goldensignalsai.application.ai_service.orchestrator import Orchestrator
from goldensignalsai.application.services.signal_types import SignalType, TradingSignal

class SignalEngine:
    """Advanced trading signal generation engine with multi-factor analysis."""

    def __init__(
        self, 
        data_fetcher, 
        user_id: str, 
        risk_profile: str = "balanced",
        strategies: Optional[List] = None
    ):
        """
        Initialize the SignalEngine with comprehensive configuration.

        Args:
            data_fetcher: Data source for market information
            user_id (str): User identifier for personalized signals
            risk_profile (str): Risk tolerance level
            strategies (Optional[List]): Additional signal generation strategies
        """
        self.data_fetcher = data_fetcher
        self.user_id = user_id
        self.risk_profile = risk_profile
        self.orchestrator = Orchestrator(data_fetcher)
        self.strategies = strategies or []

        logger.info(
            {
                "message": "SignalEngine initialized",
                "user_id": user_id,
                "risk_profile": risk_profile,
            }
        )

    async def generate_signal(self, symbol: str) -> TradingSignal:
        """
        Generate a comprehensive trading signal for a given symbol.

        Args:
            symbol (str): Trading symbol

        Returns:
            TradingSignal: Detailed trading signal with multi-factor analysis
        """
        logger.info(f"Generating signal for {symbol}")
        
        try:
            # Fetch comprehensive market data
            market_data = await self.orchestrator.consume_stream(symbol)
            
            if not market_data:
                logger.warning(f"No market data available for {symbol}")
                return self._create_default_signal(symbol)

            # Extract market components
            stock_data = market_data.get("stock_data", pd.DataFrame())
            options_data = market_data.get("options_data", {})
            news_articles = market_data.get("news_articles", [])

            # Multi-factor signal generation
            ai_score = await self._compute_ai_score(symbol, stock_data)
            indicator_signals = self._compute_technical_signals(stock_data)
            sentiment_score = self._analyze_sentiment(news_articles)

            # Signal scoring and confidence calculation
            final_score, confidence = self._calculate_signal_confidence(
                ai_score, indicator_signals, sentiment_score
            )

            # Determine signal type and action
            signal_type = self._determine_signal_type(final_score)
            
            # Create comprehensive trading signal
            signal = TradingSignal(
                symbol=symbol,
                timestamp=pd.Timestamp.now(),
                signal_type=signal_type,
                confidence=confidence,
                price=stock_data['close'].iloc[-1] if not stock_data.empty else 0.0,
                additional_info={
                    'ai_score': ai_score,
                    'indicator_signals': indicator_signals,
                    'sentiment_score': sentiment_score,
                    'risk_profile': self.risk_profile
                }
            )

            logger.info(f"Generated signal for {symbol}: {signal}")
            return signal

        except Exception as e:
            logger.error(f"Signal generation error for {symbol}: {e}")
            return self._create_default_signal(symbol, error=str(e))

    def _compute_ai_score(self, symbol: str, stock_data: pd.DataFrame) -> float:
        """Compute AI-driven prediction score."""
        try:
            X_new_lstm = np.array(stock_data["close"].tail(60)).reshape(1, 60, 1)
            X_new_tree = stock_data.tail(10)
            return await self.orchestrator.ensemble_predict(symbol, X_new_lstm, X_new_tree) or 0.0
        except Exception as e:
            logger.warning(f"AI score computation failed: {e}")
            return 0.0

    def _compute_technical_signals(self, stock_data: pd.DataFrame) -> Dict:
        """Compute technical indicator signals."""
        indicators = Indicators(stock_data)
        return indicators.compute_regime_adjusted_signal(["RSI", "MACD"])

    def _analyze_sentiment(self, news_articles: List) -> float:
        """Analyze market sentiment from news."""
        return self.orchestrator.analyze_sentiment(news_articles)

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

    def _determine_signal_type(self, final_score: float) -> SignalType:
        """Determine signal type based on score."""
        if final_score > 0.3:
            return SignalType.BUY
        elif final_score < -0.3:
            return SignalType.SELL
        return SignalType.HOLD

    def _create_default_signal(
        self, 
        symbol: str, 
        error: Optional[str] = None
    ) -> TradingSignal:
        """Create a default hold signal."""
        return TradingSignal(
            symbol=symbol,
            timestamp=pd.Timestamp.now(),
            signal_type=SignalType.HOLD,
            confidence=0.0,
            price=0.0,
            additional_info={'error': error} if error else {}
        )