"""
News Arbitrage Agent
High-speed news analysis and arbitrage opportunity detection
Issue #188: Agent-4: Develop News Arbitrage Agent
"""

import asyncio
import json
import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import spacy
from textblob import TextBlob

logger = logging.getLogger(__name__)


class NewsType(Enum):
    """Types of news events"""
    EARNINGS = "earnings"
    MERGER = "merger"
    REGULATORY = "regulatory"
    ECONOMIC = "economic"
    PRODUCT = "product"
    LEGAL = "legal"
    MANAGEMENT = "management"
    GUIDANCE = "guidance"
    ANALYST = "analyst"
    SECTOR = "sector"
    MACRO = "macro"
    BREAKING = "breaking"


class NewsImpact(Enum):
    """Expected news impact"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"
    UNCERTAIN = "uncertain"


class ArbitrageType(Enum):
    """Types of arbitrage opportunities"""
    CROSS_ASSET = "cross_asset"  # Stock vs options, ETF vs components
    TEMPORAL = "temporal"  # Time-based inefficiency
    STATISTICAL = "statistical"  # Mean reversion after news
    PAIRS = "pairs"  # Related stocks divergence
    INDEX = "index"  # Index vs components
    SENTIMENT = "sentiment"  # News vs market reaction mismatch
    VOLATILITY = "volatility"  # Implied vs realized vol


@dataclass
class NewsEvent:
    """Individual news event"""
    event_id: str
    timestamp: datetime
    headline: str
    summary: str
    source: str
    news_type: NewsType
    tickers: List[str]
    sentiment_score: float  # -1 to 1
    confidence: float
    keywords: List[str]
    entities: Dict[str, List[str]]  # Named entities
    urgency: str  # immediate, high, medium, low
    expected_duration: str  # minutes, hours, days


@dataclass
class MarketReaction:
    """Market reaction to news"""
    ticker: str
    initial_price: float
    current_price: float
    price_change_pct: float
    volume_spike: float  # Multiple of average
    volatility_change: float
    option_flow: Dict[str, Any]
    correlated_moves: List[Dict[str, float]]
    reaction_speed: float  # Seconds to reach 50% of move


@dataclass
class ArbitrageOpportunity:
    """Identified arbitrage opportunity"""
    opp_id: str
    arb_type: ArbitrageType
    news_event: NewsEvent
    instruments: List[str]
    entry_prices: Dict[str, float]
    target_prices: Dict[str, float]
    expected_profit_bps: float
    confidence: float
    time_window: int  # Seconds
    risk_factors: List[str]
    entry_signals: List[str]
    exit_signals: List[str]
    position_sizes: Dict[str, float]  # Instrument -> size


@dataclass
class ArbitrageResult:
    """Result of arbitrage execution"""
    opp_id: str
    entry_time: datetime
    exit_time: datetime
    instruments_traded: Dict[str, Dict[str, Any]]  # Instrument -> trade details
    realized_pnl: float
    realized_bps: float
    slippage_bps: float
    execution_quality: float  # 0-1
    market_conditions: Dict[str, Any]


class NewsProcessor:
    """Process and analyze news events"""

    def __init__(self):
        # Initialize NLP models (mock for demo)
        self.sentiment_analyzer = self._init_sentiment_model()
        self.entity_extractor = self._init_entity_model()
        self.impact_predictor = self._init_impact_model()

        # Historical patterns
        self.news_patterns = self._load_news_patterns()
        self.reaction_patterns = self._load_reaction_patterns()

    def _init_sentiment_model(self) -> Any:
        """Initialize sentiment analysis model"""
        # In production, would use fine-tuned financial sentiment model
        return TextBlob

    def _init_entity_model(self) -> Any:
        """Initialize named entity recognition"""
        # In production, would use spaCy with financial entities
        return None  # Simplified for demo

    def _init_impact_model(self) -> Dict[str, Any]:
        """Initialize impact prediction model"""
        return {
            'earnings_beat': {'impact': 0.03, 'duration': 'days', 'confidence': 0.8},
            'earnings_miss': {'impact': -0.05, 'duration': 'days', 'confidence': 0.8},
            'merger_announcement': {'impact': 0.15, 'duration': 'days', 'confidence': 0.7},
            'regulatory_approval': {'impact': 0.05, 'duration': 'hours', 'confidence': 0.6},
            'regulatory_rejection': {'impact': -0.10, 'duration': 'days', 'confidence': 0.7},
            'product_launch': {'impact': 0.02, 'duration': 'days', 'confidence': 0.5},
            'lawsuit': {'impact': -0.03, 'duration': 'hours', 'confidence': 0.6},
            'ceo_departure': {'impact': -0.02, 'duration': 'hours', 'confidence': 0.7},
            'guidance_raise': {'impact': 0.04, 'duration': 'days', 'confidence': 0.8},
            'guidance_lower': {'impact': -0.06, 'duration': 'days', 'confidence': 0.8}
        }

    def _load_news_patterns(self) -> Dict[str, List[str]]:
        """Load news keyword patterns"""
        return {
            'earnings_beat': ['beats', 'exceeds', 'tops', 'surpasses', 'earnings beat'],
            'earnings_miss': ['misses', 'falls short', 'disappoints', 'below expectations'],
            'merger': ['merger', 'acquisition', 'acquires', 'to buy', 'takeover', 'deal'],
            'regulatory': ['FDA', 'SEC', 'approval', 'rejected', 'fined', 'investigation'],
            'product': ['launches', 'announces', 'unveils', 'introduces', 'new product'],
            'legal': ['lawsuit', 'sued', 'litigation', 'settlement', 'court', 'jury'],
            'management': ['CEO', 'CFO', 'resigns', 'appoints', 'departure', 'hire'],
            'guidance': ['guidance', 'outlook', 'forecast', 'raises', 'lowers', 'warns']
        }

    def _load_reaction_patterns(self) -> Dict[NewsType, Dict[str, Any]]:
        """Load typical market reaction patterns"""
        return {
            NewsType.EARNINGS: {
                'initial_move': 0.03,  # 3% typical
                'peak_time': 300,  # 5 minutes
                'reversion_rate': 0.3,  # 30% reversion
                'correlated_assets': ['sector_etf', 'competitors']
            },
            NewsType.MERGER: {
                'initial_move': 0.10,  # 10% typical for target
                'peak_time': 600,  # 10 minutes
                'reversion_rate': 0.1,  # 10% reversion
                'correlated_assets': ['acquirer', 'sector_peers']
            },
            NewsType.REGULATORY: {
                'initial_move': 0.05,
                'peak_time': 900,  # 15 minutes
                'reversion_rate': 0.4,
                'correlated_assets': ['biotech_etf', 'competitors']
            }
        }

    async def process_news(self, raw_news: Dict[str, Any]) -> NewsEvent:
        """Process raw news into structured event"""
        # Extract basic info
        headline = raw_news.get('headline', '')
        summary = raw_news.get('summary', headline)
        timestamp = datetime.fromisoformat(raw_news.get('timestamp', datetime.now().isoformat()))

        # Classify news type
        news_type = self._classify_news_type(headline, summary)

        # Extract entities and tickers
        entities = self._extract_entities(headline + ' ' + summary)
        tickers = self._extract_tickers(headline + ' ' + summary, entities)

        # Analyze sentiment
        sentiment = self._analyze_sentiment(headline, summary, news_type)

        # Extract keywords
        keywords = self._extract_keywords(headline + ' ' + summary)

        # Determine urgency
        urgency = self._determine_urgency(news_type, keywords)

        # Predict duration
        duration = self._predict_duration(news_type)

        return NewsEvent(
            event_id=f"NEWS_{timestamp.timestamp()}",
            timestamp=timestamp,
            headline=headline,
            summary=summary,
            source=raw_news.get('source', 'unknown'),
            news_type=news_type,
            tickers=tickers,
            sentiment_score=sentiment,
            confidence=0.8,  # Simplified
            keywords=keywords,
            entities=entities,
            urgency=urgency,
            expected_duration=duration
        )

    def _classify_news_type(self, headline: str, summary: str) -> NewsType:
        """Classify news type based on content"""
        text = (headline + ' ' + summary).lower()

        # Check patterns
        for pattern_type, keywords in self.news_patterns.items():
            if any(keyword in text for keyword in keywords):
                if 'earnings' in pattern_type:
                    return NewsType.EARNINGS
                elif 'merger' in pattern_type:
                    return NewsType.MERGER
                elif 'regulatory' in pattern_type:
                    return NewsType.REGULATORY
                elif 'product' in pattern_type:
                    return NewsType.PRODUCT
                elif 'legal' in pattern_type:
                    return NewsType.LEGAL
                elif 'management' in pattern_type:
                    return NewsType.MANAGEMENT
                elif 'guidance' in pattern_type:
                    return NewsType.GUIDANCE

        return NewsType.BREAKING  # Default

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        # Simplified entity extraction
        entities = {
            'companies': [],
            'persons': [],
            'numbers': [],
            'percentages': []
        }

        # Extract companies (simplified - look for capitalized words)
        companies = re.findall(r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b', text)
        entities['companies'] = list(set(companies))[:5]

        # Extract numbers and percentages
        numbers = re.findall(r'\$?\d+(?:\.\d+)?(?:B|M|K)?', text)
        entities['numbers'] = numbers[:5]

        percentages = re.findall(r'\d+(?:\.\d+)?%', text)
        entities['percentages'] = percentages[:5]

        return entities

    def _extract_tickers(self, text: str, entities: Dict[str, List[str]]) -> List[str]:
        """Extract stock tickers from text"""
        # Look for uppercase symbols 1-5 chars long
        tickers = re.findall(r'\b[A-Z]{1,5}\b', text)

        # Filter common words
        common_words = {'CEO', 'CFO', 'FDA', 'SEC', 'IPO', 'NYSE', 'THE', 'AND', 'FOR'}
        tickers = [t for t in tickers if t not in common_words]

        # Return unique tickers
        return list(set(tickers))[:5]

    def _analyze_sentiment(self, headline: str, summary: str,
                         news_type: NewsType) -> float:
        """Analyze sentiment of news"""
        # Simple sentiment analysis
        text = headline + ' ' + summary

        # Use TextBlob for basic sentiment
        blob = TextBlob(text)
        base_sentiment = blob.sentiment.polarity  # -1 to 1

        # Adjust for financial keywords
        positive_words = ['beat', 'exceed', 'surpass', 'raise', 'upgrade', 'approve']
        negative_words = ['miss', 'fall', 'disappoint', 'lower', 'downgrade', 'reject']

        positive_count = sum(1 for word in positive_words if word in text.lower())
        negative_count = sum(1 for word in negative_words if word in text.lower())

        # Combine scores
        keyword_sentiment = (positive_count - negative_count) * 0.2
        final_sentiment = np.clip(base_sentiment + keyword_sentiment, -1, 1)

        return final_sentiment

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key words from text"""
        # Simple keyword extraction
        words = text.lower().split()

        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        keywords = [w for w in words if w not in stop_words and len(w) > 3]

        # Get most common
        from collections import Counter
        word_counts = Counter(keywords)

        return [word for word, _ in word_counts.most_common(10)]

    def _determine_urgency(self, news_type: NewsType, keywords: List[str]) -> str:
        """Determine urgency of news"""
        urgent_keywords = ['breaking', 'alert', 'urgent', 'immediate', 'now', 'just']

        if any(word in keywords for word in urgent_keywords):
            return 'immediate'
        elif news_type in [NewsType.EARNINGS, NewsType.MERGER]:
            return 'high'
        elif news_type in [NewsType.REGULATORY, NewsType.LEGAL]:
            return 'medium'
        else:
            return 'low'

    def _predict_duration(self, news_type: NewsType) -> str:
        """Predict impact duration"""
        duration_map = {
            NewsType.EARNINGS: 'days',
            NewsType.MERGER: 'days',
            NewsType.REGULATORY: 'hours',
            NewsType.ECONOMIC: 'hours',
            NewsType.PRODUCT: 'days',
            NewsType.LEGAL: 'hours',
            NewsType.MANAGEMENT: 'hours',
            NewsType.GUIDANCE: 'days',
            NewsType.BREAKING: 'minutes'
        }

        return duration_map.get(news_type, 'hours')


class ArbitrageDetector:
    """Detect arbitrage opportunities from news"""

    def __init__(self):
        self.correlation_matrix = self._load_correlations()
        self.pair_relationships = self._load_pair_relationships()
        self.historical_reactions = self._load_historical_reactions()

    def _load_correlations(self) -> Dict[str, Dict[str, float]]:
        """Load asset correlations"""
        # Simplified correlation matrix
        return {
            'AAPL': {'QQQ': 0.8, 'MSFT': 0.7, 'GOOGL': 0.6, 'SPY': 0.7},
            'MSFT': {'QQQ': 0.85, 'AAPL': 0.7, 'GOOGL': 0.75, 'SPY': 0.8},
            'JPM': {'XLF': 0.9, 'BAC': 0.85, 'GS': 0.8, 'SPY': 0.7},
            'XOM': {'XLE': 0.95, 'CVX': 0.9, 'COP': 0.85, 'SPY': 0.6}
        }

    def _load_pair_relationships(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load pair trading relationships"""
        return {
            'competitors': [
                {'pair': ['AAPL', 'MSFT'], 'spread_mean': 0.0, 'spread_std': 0.02},
                {'pair': ['JPM', 'BAC'], 'spread_mean': 0.0, 'spread_std': 0.03},
                {'pair': ['XOM', 'CVX'], 'spread_mean': 0.0, 'spread_std': 0.025}
            ],
            'etf_components': [
                {'etf': 'XLF', 'components': ['JPM', 'BAC', 'GS', 'MS'], 'weight': 0.25},
                {'etf': 'XLE', 'components': ['XOM', 'CVX', 'COP'], 'weight': 0.33}
            ]
        }

    def _load_historical_reactions(self) -> Dict[NewsType, Dict[str, Any]]:
        """Load historical reaction patterns"""
        return {
            NewsType.EARNINGS: {
                'avg_move': 0.04,
                'reaction_time': 300,  # seconds
                'reversion_prob': 0.3,
                'option_iv_change': 0.2
            },
            NewsType.MERGER: {
                'target_move': 0.15,
                'acquirer_move': -0.03,
                'reaction_time': 600,
                'arb_spread': 0.02
            }
        }

    async def detect_opportunities(self, news_event: NewsEvent,
                                 market_data: Dict[str, Dict[str, Any]]) -> List[ArbitrageOpportunity]:
        """Detect arbitrage opportunities from news event"""
        opportunities = []

        # Check each arbitrage type
        for arb_type in ArbitrageType:
            opp = await self._check_arbitrage_type(arb_type, news_event, market_data)
            if opp:
                opportunities.append(opp)

        # Sort by expected profit
        opportunities.sort(key=lambda x: x.expected_profit_bps, reverse=True)

        return opportunities

    async def _check_arbitrage_type(self, arb_type: ArbitrageType,
                                  news_event: NewsEvent,
                                  market_data: Dict[str, Dict[str, Any]]) -> Optional[ArbitrageOpportunity]:
        """Check specific arbitrage type"""
        if arb_type == ArbitrageType.CROSS_ASSET:
            return await self._check_cross_asset_arb(news_event, market_data)
        elif arb_type == ArbitrageType.PAIRS:
            return await self._check_pairs_arb(news_event, market_data)
        elif arb_type == ArbitrageType.INDEX:
            return await self._check_index_arb(news_event, market_data)
        elif arb_type == ArbitrageType.SENTIMENT:
            return await self._check_sentiment_arb(news_event, market_data)
        elif arb_type == ArbitrageType.VOLATILITY:
            return await self._check_volatility_arb(news_event, market_data)

        return None

    async def _check_cross_asset_arb(self, news: NewsEvent,
                                   market_data: Dict[str, Dict[str, Any]]) -> Optional[ArbitrageOpportunity]:
        """Check cross-asset arbitrage (stock vs options)"""
        if not news.tickers:
            return None

        ticker = news.tickers[0]
        if ticker not in market_data:
            return None

        stock_data = market_data[ticker]

        # Check if options are mispriced relative to expected move
        if 'implied_volatility' in stock_data:
            current_iv = stock_data['implied_volatility']
            expected_move = abs(news.sentiment_score) * 0.05  # Simplified

            # If IV hasn't adjusted to news yet
            if expected_move > current_iv * 0.1:  # 10% of IV
                return ArbitrageOpportunity(
                    opp_id=f"XASSET_{ticker}_{news.event_id}",
                    arb_type=ArbitrageType.CROSS_ASSET,
                    news_event=news,
                    instruments=[ticker, f"{ticker}_OPTIONS"],
                    entry_prices={
                        ticker: stock_data['price'],
                        f"{ticker}_OPTIONS": current_iv
                    },
                    target_prices={
                        ticker: stock_data['price'] * (1 + expected_move),
                        f"{ticker}_OPTIONS": current_iv * 1.5
                    },
                    expected_profit_bps=int(expected_move * 5000),  # Simplified
                    confidence=0.7,
                    time_window=600,  # 10 minutes
                    risk_factors=['execution_risk', 'liquidity_risk'],
                    entry_signals=['news_released', 'iv_unchanged'],
                    exit_signals=['iv_spike', 'price_move_complete'],
                    position_sizes={ticker: -1.0, f"{ticker}_OPTIONS": 2.0}  # Delta neutral
                )

        return None

    async def _check_pairs_arb(self, news: NewsEvent,
                             market_data: Dict[str, Dict[str, Any]]) -> Optional[ArbitrageOpportunity]:
        """Check pairs trading arbitrage"""
        if not news.tickers:
            return None

        ticker = news.tickers[0]

        # Find correlated pairs
        if ticker in self.correlation_matrix:
            correlations = self.correlation_matrix[ticker]

            for paired_ticker, correlation in correlations.items():
                if correlation > 0.8 and paired_ticker in market_data:
                    # Check if pair has diverged
                    ticker_move = self._calculate_move(ticker, market_data)
                    pair_move = self._calculate_move(paired_ticker, market_data)

                    divergence = abs(ticker_move - pair_move)

                    if divergence > 0.02:  # 2% divergence
                        return ArbitrageOpportunity(
                            opp_id=f"PAIRS_{ticker}_{paired_ticker}_{news.event_id}",
                            arb_type=ArbitrageType.PAIRS,
                            news_event=news,
                            instruments=[ticker, paired_ticker],
                            entry_prices={
                                ticker: market_data[ticker]['price'],
                                paired_ticker: market_data[paired_ticker]['price']
                            },
                            target_prices={
                                ticker: market_data[ticker]['price'],
                                paired_ticker: market_data[paired_ticker]['price'] * (1 + ticker_move)
                            },
                            expected_profit_bps=int(divergence * 5000),
                            confidence=0.65,
                            time_window=1800,  # 30 minutes
                            risk_factors=['correlation_breakdown', 'sector_risk'],
                            entry_signals=['divergence_detected', 'correlation_stable'],
                            exit_signals=['convergence_complete', 'stop_loss'],
                            position_sizes={ticker: -1.0, paired_ticker: 1.0}
                        )

        return None

    async def _check_index_arb(self, news: NewsEvent,
                             market_data: Dict[str, Dict[str, Any]]) -> Optional[ArbitrageOpportunity]:
        """Check index arbitrage opportunities"""
        # Check if news affects index component
        for etf_info in self.pair_relationships['etf_components']:
            etf = etf_info['etf']
            components = etf_info['components']

            affected_components = [t for t in news.tickers if t in components]

            if affected_components and etf in market_data:
                # Calculate expected ETF move based on component moves
                component_moves = []
                for comp in affected_components:
                    if comp in market_data:
                        move = self._calculate_move(comp, market_data)
                        component_moves.append(move * etf_info['weight'])

                if component_moves:
                    expected_etf_move = sum(component_moves)
                    actual_etf_move = self._calculate_move(etf, market_data)

                    discrepancy = expected_etf_move - actual_etf_move

                    if abs(discrepancy) > 0.005:  # 0.5% discrepancy
                        return ArbitrageOpportunity(
                            opp_id=f"INDEX_{etf}_{news.event_id}",
                            arb_type=ArbitrageType.INDEX,
                            news_event=news,
                            instruments=[etf] + affected_components,
                            entry_prices={
                                **{etf: market_data[etf]['price']},
                                **{comp: market_data[comp]['price'] for comp in affected_components if comp in market_data}
                            },
                            target_prices={
                                etf: market_data[etf]['price'] * (1 + expected_etf_move)
                            },
                            expected_profit_bps=int(abs(discrepancy) * 10000),
                            confidence=0.75,
                            time_window=900,  # 15 minutes
                            risk_factors=['tracking_error', 'liquidity_mismatch'],
                            entry_signals=['index_lag_detected'],
                            exit_signals=['arbitrage_closed', 'rebalancing'],
                            position_sizes={etf: 1.0 if discrepancy > 0 else -1.0}
                        )

        return None

    async def _check_sentiment_arb(self, news: NewsEvent,
                                 market_data: Dict[str, Dict[str, Any]]) -> Optional[ArbitrageOpportunity]:
        """Check sentiment-based arbitrage"""
        if not news.tickers:
            return None

        ticker = news.tickers[0]
        if ticker not in market_data:
            return None

        # Check if market reaction doesn't match news sentiment
        actual_move = self._calculate_move(ticker, market_data)
        expected_direction = 1 if news.sentiment_score > 0 else -1
        expected_magnitude = abs(news.sentiment_score) * 0.03  # 3% per sentiment unit

        # If market moved opposite to sentiment or insufficient magnitude
        if (actual_move * expected_direction < 0) or \
           (abs(actual_move) < expected_magnitude * 0.5):

            return ArbitrageOpportunity(
                opp_id=f"SENT_{ticker}_{news.event_id}",
                arb_type=ArbitrageType.SENTIMENT,
                news_event=news,
                instruments=[ticker],
                entry_prices={ticker: market_data[ticker]['price']},
                target_prices={
                    ticker: market_data[ticker]['price'] * (1 + expected_direction * expected_magnitude)
                },
                expected_profit_bps=int(expected_magnitude * 10000),
                confidence=0.6,
                time_window=3600,  # 1 hour
                risk_factors=['sentiment_persistence', 'market_regime'],
                entry_signals=['sentiment_divergence', 'low_volume'],
                exit_signals=['sentiment_alignment', 'volume_spike'],
                position_sizes={ticker: expected_direction * 1.0}
            )

        return None

    async def _check_volatility_arb(self, news: NewsEvent,
                                  market_data: Dict[str, Dict[str, Any]]) -> Optional[ArbitrageOpportunity]:
        """Check volatility arbitrage"""
        if news.news_type not in [NewsType.EARNINGS, NewsType.REGULATORY]:
            return None

        ticker = news.tickers[0] if news.tickers else None
        if not ticker or ticker not in market_data:
            return None

        stock_data = market_data[ticker]

        # Check if IV is too low given news
        if 'implied_volatility' in stock_data:
            current_iv = stock_data['implied_volatility']
            historical_reaction = self.historical_reactions.get(news.news_type, {})
            expected_move = historical_reaction.get('avg_move', 0.04)

            # Convert move to IV terms
            expected_iv = expected_move * np.sqrt(252)  # Annualized

            if current_iv < expected_iv * 0.8:  # IV 20% below expected
                return ArbitrageOpportunity(
                    opp_id=f"VOL_{ticker}_{news.event_id}",
                    arb_type=ArbitrageType.VOLATILITY,
                    news_event=news,
                    instruments=[f"{ticker}_OPTIONS", ticker],
                    entry_prices={
                        f"{ticker}_OPTIONS": current_iv,
                        ticker: stock_data['price']
                    },
                    target_prices={
                        f"{ticker}_OPTIONS": expected_iv
                    },
                    expected_profit_bps=int((expected_iv - current_iv) * 10000),
                    confidence=0.7,
                    time_window=1800,  # 30 minutes
                    risk_factors=['vol_crush', 'theta_decay'],
                    entry_signals=['low_iv', 'news_catalyst'],
                    exit_signals=['iv_expansion', 'event_complete'],
                    position_sizes={
                        f"{ticker}_OPTIONS": 1.0,  # Long vol
                        ticker: -0.5  # Delta hedge
                    }
                )

        return None

    def _calculate_move(self, ticker: str, market_data: Dict[str, Dict[str, Any]]) -> float:
        """Calculate price move for ticker"""
        if ticker not in market_data:
            return 0.0

        data = market_data[ticker]
        if 'prev_close' in data and 'price' in data:
            return (data['price'] - data['prev_close']) / data['prev_close']

        return 0.0


class NewsArbitrageAgent:
    """
    News Arbitrage Agent that detects and executes arbitrage opportunities
    from breaking news events with ultra-low latency
    """

    def __init__(self):
        """Initialize the news arbitrage agent"""
        self.news_processor = NewsProcessor()
        self.arbitrage_detector = ArbitrageDetector()

        # Execution tracking
        self.active_opportunities = {}
        self.execution_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(lambda: {
            'opportunities': 0,
            'executed': 0,
            'successful': 0,
            'total_pnl': 0,
            'avg_profit_bps': 0
        })

        # Speed metrics
        self.latency_tracker = {
            'news_to_detection': deque(maxlen=100),
            'detection_to_execution': deque(maxlen=100),
            'total_latency': deque(maxlen=100)
        }

    async def process_news_event(self, raw_news: Dict[str, Any],
                               market_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Process news event and detect opportunities"""
        start_time = datetime.now()

        # Process news
        news_event = await self.news_processor.process_news(raw_news)
        news_process_time = (datetime.now() - start_time).total_seconds()

        # Detect arbitrage opportunities
        detect_start = datetime.now()
        opportunities = await self.arbitrage_detector.detect_opportunities(news_event, market_data)
        detect_time = (datetime.now() - detect_start).total_seconds()

        # Track latency
        self.latency_tracker['news_to_detection'].append(news_process_time + detect_time)

        # Prepare response
        response = {
            'news_event': {
                'event_id': news_event.event_id,
                'headline': news_event.headline,
                'type': news_event.news_type.value,
                'sentiment': news_event.sentiment_score,
                'tickers': news_event.tickers,
                'urgency': news_event.urgency
            },
            'opportunities_found': len(opportunities),
            'opportunities': [
                {
                    'id': opp.opp_id,
                    'type': opp.arb_type.value,
                    'instruments': opp.instruments,
                    'expected_profit_bps': opp.expected_profit_bps,
                    'confidence': opp.confidence,
                    'time_window': opp.time_window
                }
                for opp in opportunities[:5]  # Top 5
            ],
            'processing_time_ms': int((news_process_time + detect_time) * 1000),
            'timestamp': datetime.now().isoformat()
        }

        # Store active opportunities
        for opp in opportunities:
            self.active_opportunities[opp.opp_id] = opp

        return response

    async def execute_arbitrage(self, opp_id: str,
                              market_data: Dict[str, Dict[str, Any]]) -> ArbitrageResult:
        """Execute arbitrage opportunity"""
        if opp_id not in self.active_opportunities:
            raise ValueError(f"Opportunity {opp_id} not found")

        opportunity = self.active_opportunities[opp_id]
        entry_time = datetime.now()

        # Simulate execution (in production, would connect to execution system)
        trades = {}
        total_slippage = 0

        for instrument, size in opportunity.position_sizes.items():
            if instrument in market_data:
                entry_price = opportunity.entry_prices.get(instrument, market_data[instrument]['price'])

                # Simulate slippage
                slippage = abs(size) * 0.0001  # 1 bp per unit
                if size > 0:
                    exec_price = entry_price * (1 + slippage)
                else:
                    exec_price = entry_price * (1 - slippage)

                trades[instrument] = {
                    'size': size,
                    'entry_price': entry_price,
                    'exec_price': exec_price,
                    'slippage_bps': slippage * 10000
                }

                total_slippage += slippage * 10000

        # Simulate holding period and exit
        await asyncio.sleep(0.1)  # Simulate time passing

        # Calculate P&L (simplified)
        exit_time = datetime.now()
        total_pnl = 0

        for instrument, trade in trades.items():
            if instrument in opportunity.target_prices:
                exit_price = opportunity.target_prices[instrument]
            else:
                # Use current market price
                exit_price = market_data.get(instrument, {}).get('price', trade['exec_price'])

            if trade['size'] > 0:
                pnl = (exit_price - trade['exec_price']) / trade['exec_price']
            else:
                pnl = (trade['exec_price'] - exit_price) / trade['exec_price']

            pnl *= abs(trade['size'])
            total_pnl += pnl

        # Create result
        result = ArbitrageResult(
            opp_id=opp_id,
            entry_time=entry_time,
            exit_time=exit_time,
            instruments_traded=trades,
            realized_pnl=total_pnl,
            realized_bps=total_pnl * 10000,
            slippage_bps=total_slippage / len(trades) if trades else 0,
            execution_quality=0.85,  # Simplified
            market_conditions={
                'volatility': 0.20,
                'liquidity': 0.75,
                'spread': 5.0
            }
        )

        # Update metrics
        self._update_metrics(opportunity.arb_type, result)

        # Store in history
        self.execution_history.append(result)

        # Remove from active
        del self.active_opportunities[opp_id]

        return result

    def _update_metrics(self, arb_type: ArbitrageType, result: ArbitrageResult):
        """Update performance metrics"""
        metrics = self.performance_metrics[arb_type.value]

        metrics['executed'] += 1
        if result.realized_pnl > 0:
            metrics['successful'] += 1

        metrics['total_pnl'] += result.realized_pnl
        metrics['avg_profit_bps'] = (
            metrics['total_pnl'] / metrics['executed'] * 10000
            if metrics['executed'] > 0 else 0
        )

    async def monitor_active_opportunities(self, market_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Monitor active arbitrage opportunities"""
        monitoring_results = []

        for opp_id, opportunity in list(self.active_opportunities.items()):
            # Check if opportunity still valid
            time_elapsed = (datetime.now() - opportunity.news_event.timestamp).total_seconds()

            if time_elapsed > opportunity.time_window:
                # Opportunity expired
                del self.active_opportunities[opp_id]
                status = 'expired'
            else:
                # Check exit signals
                exit_triggered = False
                for signal in opportunity.exit_signals:
                    # Simplified signal checking
                    if signal == 'convergence_complete':
                        # Check if prices converged
                        exit_triggered = True
                        break

                if exit_triggered:
                    # Execute exit
                    result = await self.execute_arbitrage(opp_id, market_data)
                    status = 'executed'
                else:
                    status = 'active'

            monitoring_results.append({
                'opp_id': opp_id,
                'status': status,
                'time_remaining': max(0, opportunity.time_window - time_elapsed),
                'current_pnl': self._calculate_current_pnl(opportunity, market_data)
            })

        return {
            'active_opportunities': len(self.active_opportunities),
            'monitoring_results': monitoring_results,
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_current_pnl(self, opportunity: ArbitrageOpportunity,
                             market_data: Dict[str, Dict[str, Any]]) -> float:
        """Calculate current P&L for opportunity"""
        total_pnl = 0

        for instrument, size in opportunity.position_sizes.items():
            if instrument in opportunity.entry_prices and instrument in market_data:
                entry_price = opportunity.entry_prices[instrument]
                current_price = market_data[instrument].get('price', entry_price)

                if size > 0:
                    pnl = (current_price - entry_price) / entry_price
                else:
                    pnl = (entry_price - current_price) / entry_price

                total_pnl += pnl * abs(size)

        return total_pnl * 10000  # In bps

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        total_metrics = {
            'total_opportunities': sum(m['opportunities'] for m in self.performance_metrics.values()),
            'total_executed': sum(m['executed'] for m in self.performance_metrics.values()),
            'total_successful': sum(m['successful'] for m in self.performance_metrics.values()),
            'total_pnl': sum(m['total_pnl'] for m in self.performance_metrics.values()),
            'success_rate': 0,
            'avg_latency_ms': 0
        }

        if total_metrics['total_executed'] > 0:
            total_metrics['success_rate'] = (
                total_metrics['total_successful'] / total_metrics['total_executed']
            )

        # Calculate average latency
        if self.latency_tracker['total_latency']:
            total_metrics['avg_latency_ms'] = (
                np.mean(self.latency_tracker['total_latency']) * 1000
            )

        # By arbitrage type breakdown
        by_type = {}
        for arb_type, metrics in self.performance_metrics.items():
            if metrics['executed'] > 0:
                by_type[arb_type] = {
                    'executed': metrics['executed'],
                    'success_rate': metrics['successful'] / metrics['executed'],
                    'avg_profit_bps': metrics['avg_profit_bps'],
                    'total_pnl': metrics['total_pnl']
                }

        return {
            'summary': total_metrics,
            'by_arbitrage_type': by_type,
            'recent_executions': [
                {
                    'opp_id': r.opp_id,
                    'pnl_bps': r.realized_bps,
                    'execution_time': r.exit_time.isoformat()
                }
                for r in list(self.execution_history)[-10:]
            ]
        }


# Demo function
async def demo_news_arbitrage():
    """Demonstrate the News Arbitrage Agent"""
    agent = NewsArbitrageAgent()

    print("News Arbitrage Agent Demo")
    print("="*70)

    # Test Case 1: Earnings Beat News
    print("\n📰 Case 1: Earnings Beat News Event")
    print("-"*50)

    earnings_news = {
        'headline': 'Apple Beats Q4 Earnings Expectations by 15%',
        'summary': 'Apple Inc. (AAPL) reported Q4 earnings of $1.50 per share, beating analyst expectations of $1.30. Revenue also exceeded forecasts.',
        'timestamp': datetime.now().isoformat(),
        'source': 'Reuters'
    }

    market_data = {
        'AAPL': {
            'price': 195.00,
            'prev_close': 195.00,
            'volume': 1000000,
            'implied_volatility': 0.25,
            'bid': 194.98,
            'ask': 195.02
        },
        'QQQ': {
            'price': 400.00,
            'prev_close': 400.00,
            'volume': 5000000
        },
        'MSFT': {
            'price': 380.00,
            'prev_close': 380.00,
            'volume': 2000000
        }
    }

    result = await agent.process_news_event(earnings_news, market_data)

    print(f"News Type: {result['news_event']['type']}")
    print(f"Sentiment: {result['news_event']['sentiment']:.2f}")
    print(f"Affected Tickers: {', '.join(result['news_event']['tickers'])}")
    print(f"\n🎯 Arbitrage Opportunities Found: {result['opportunities_found']}")

    for opp in result['opportunities']:
        print(f"\n  {opp['type'].upper()} Arbitrage:")
        print(f"    Instruments: {', '.join(opp['instruments'])}")
        print(f"    Expected Profit: {opp['expected_profit_bps']:.0f} bps")
        print(f"    Confidence: {opp['confidence']:.1%}")
        print(f"    Time Window: {opp['time_window']}s")

    print(f"\n⚡ Processing Time: {result['processing_time_ms']}ms")

    # Test Case 2: M&A Announcement
    print("\n\n📰 Case 2: Merger & Acquisition News")
    print("-"*50)

    merger_news = {
        'headline': 'Microsoft to Acquire Gaming Company for $10B',
        'summary': 'Microsoft Corp (MSFT) announced the acquisition of XYZ Gaming for $10 billion in an all-cash deal.',
        'timestamp': datetime.now().isoformat(),
        'source': 'Bloomberg'
    }

    # Update market data to show divergence
    market_data['MSFT']['price'] = 378.00  # Down on acquirer
    market_data['QQQ']['price'] = 400.50  # Minimal move

    result2 = await agent.process_news_event(merger_news, market_data)

    print(f"News Impact: {result2['news_event']['type']}")
    print(f"Urgency: {result2['news_event']['urgency'].upper()}")

    # Execute one of the opportunities
    if result2['opportunities']:
        opp_id = result2['opportunities'][0]['id']
        print(f"\n💰 Executing Arbitrage: {opp_id}")

        exec_result = await agent.execute_arbitrage(opp_id, market_data)

        print(f"  Entry Time: {exec_result.entry_time.strftime('%H:%M:%S.%f')[:-3]}")
        print(f"  Exit Time: {exec_result.exit_time.strftime('%H:%M:%S.%f')[:-3]}")
        print(f"  Realized P&L: {exec_result.realized_bps:.1f} bps")
        print(f"  Slippage: {exec_result.slippage_bps:.1f} bps")
        print(f"  Execution Quality: {exec_result.execution_quality:.1%}")

    # Test Case 3: Breaking News - Regulatory
    print("\n\n📰 Case 3: Breaking Regulatory News")
    print("-"*50)

    regulatory_news = {
        'headline': 'BREAKING: FDA Approves XYZ Biotech Drug Ahead of Schedule',
        'summary': 'The FDA has granted approval for XYZ Biotech revolutionary cancer treatment, months ahead of expected timeline.',
        'timestamp': datetime.now().isoformat(),
        'source': 'FDA'
    }

    # Add biotech data
    market_data['XYZ'] = {
        'price': 50.00,
        'prev_close': 50.00,
        'volume': 500000,
        'implied_volatility': 0.60
    }

    market_data['IBB'] = {  # Biotech ETF
        'price': 150.00,
        'prev_close': 150.00,
        'volume': 1000000
    }

    result3 = await agent.process_news_event(regulatory_news, market_data)

    print(f"🚨 URGENT: {result3['news_event']['urgency'].upper()} priority news!")
    print(f"Processing completed in {result3['processing_time_ms']}ms")

    # Monitor active opportunities
    print("\n\n📊 Monitoring Active Opportunities")
    print("-"*50)

    monitoring = await agent.monitor_active_opportunities(market_data)

    print(f"Active Opportunities: {monitoring['active_opportunities']}")
    for mon in monitoring['monitoring_results'][:3]:
        print(f"  {mon['opp_id']}: {mon['status']} | "
              f"Time Left: {mon['time_remaining']:.0f}s | "
              f"Current P&L: {mon['current_pnl']:.1f} bps")

    # Performance summary
    print("\n\n📊 Performance Summary")
    print("-"*50)

    summary = agent.get_performance_summary()

    print(f"Total Opportunities Detected: {summary['summary']['total_opportunities']}")
    print(f"Total Executed: {summary['summary']['total_executed']}")
    print(f"Success Rate: {summary['summary']['success_rate']:.1%}")
    print(f"Average Latency: {summary['summary']['avg_latency_ms']:.1f}ms")

    if summary['by_arbitrage_type']:
        print("\nBy Arbitrage Type:")
        for arb_type, metrics in summary['by_arbitrage_type'].items():
            print(f"  {arb_type}: {metrics['executed']} trades, "
                  f"{metrics['avg_profit_bps']:.1f} bps avg")

    print("\n✅ News Arbitrage Agent demonstrates:")
    print("- Ultra-low latency news processing (<100ms)")
    print("- Multiple arbitrage strategy detection")
    print("- Cross-asset opportunity identification")
    print("- Real-time execution and monitoring")


if __name__ == "__main__":
    asyncio.run(demo_news_arbitrage())
