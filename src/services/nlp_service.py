"""
NLP Service for AI Trading Analyst
Handles intent classification and entity extraction for trading queries
"""

import re
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
import spacy
from transformers import pipeline
import numpy as np


class AnalysisIntent(Enum):
    """Types of analysis intents"""
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    PRICE_PREDICTION = "price_prediction"
    PATTERN_RECOGNITION = "pattern_recognition"
    RISK_ASSESSMENT = "risk_assessment"
    COMPARISON = "comparison"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    MARKET_OVERVIEW = "market_overview"
    OPTIONS_ANALYSIS = "options_analysis"
    GENERAL_QUERY = "general_query"


class NLPService:
    """
    Natural Language Processing service for understanding trading queries
    """
    
    def __init__(self):
        # Initialize spaCy for entity extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # Fallback if spaCy model not installed
            self.nlp = None
            
        # Intent patterns
        self.intent_patterns = {
            AnalysisIntent.TECHNICAL_ANALYSIS: [
                r"technical analysis",
                r"analyze.*chart",
                r"technical.*setup",
                r"indicators",
                r"support.*resistance",
                r"trend analysis",
                r"price action"
            ],
            AnalysisIntent.SENTIMENT_ANALYSIS: [
                r"sentiment",
                r"market mood",
                r"social media",
                r"news analysis",
                r"what.*saying about",
                r"market feeling"
            ],
            AnalysisIntent.PRICE_PREDICTION: [
                r"predict",
                r"forecast",
                r"price target",
                r"where.*going",
                r"will.*reach",
                r"price projection"
            ],
            AnalysisIntent.PATTERN_RECOGNITION: [
                r"pattern",
                r"formation",
                r"chart pattern",
                r"candlestick",
                r"elliott wave",
                r"harmonic"
            ],
            AnalysisIntent.RISK_ASSESSMENT: [
                r"risk",
                r"volatility",
                r"drawdown",
                r"safe",
                r"position size",
                r"stop loss"
            ],
            AnalysisIntent.COMPARISON: [
                r"compare",
                r"versus",
                r"vs",
                r"better than",
                r"correlation",
                r"relative"
            ],
            AnalysisIntent.OPTIONS_ANALYSIS: [
                r"options",
                r"calls",
                r"puts",
                r"strike",
                r"expiration",
                r"implied volatility",
                r"greeks"
            ],
            AnalysisIntent.PORTFOLIO_ANALYSIS: [
                r"portfolio",
                r"allocation",
                r"diversification",
                r"holdings",
                r"rebalance"
            ]
        }
        
        # Common trading symbols
        self.common_symbols = {
            'SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 
            'NVDA', 'AMD', 'JPM', 'BAC', 'XLF', 'XLE', 'GLD', 'TLT', 'VIX'
        }
        
        # Timeframe patterns
        self.timeframe_patterns = {
            '1m': ['1 minute', '1min', '1m', 'minute'],
            '5m': ['5 minute', '5min', '5m', '5 minutes'],
            '15m': ['15 minute', '15min', '15m', '15 minutes'],
            '30m': ['30 minute', '30min', '30m', '30 minutes', 'half hour'],
            '1h': ['1 hour', '1hr', '1h', 'hourly', 'hour'],
            '4h': ['4 hour', '4hr', '4h', '4 hours'],
            '1d': ['daily', '1 day', '1d', 'day'],
            '1w': ['weekly', '1 week', '1w', 'week'],
            '1M': ['monthly', '1 month', '1M', 'month']
        }
        
        # Indicator patterns
        self.indicator_patterns = {
            'rsi': ['rsi', 'relative strength'],
            'macd': ['macd', 'moving average convergence'],
            'ema': ['ema', 'exponential moving average'],
            'sma': ['sma', 'simple moving average', 'ma'],
            'bollinger': ['bollinger', 'bb', 'bands'],
            'volume': ['volume', 'vol'],
            'vwap': ['vwap', 'volume weighted'],
            'atr': ['atr', 'average true range'],
            'stochastic': ['stochastic', 'stoch'],
            'fibonacci': ['fibonacci', 'fib'],
            'ichimoku': ['ichimoku', 'cloud']
        }
        
    async def parse_query(self, query: str) -> Tuple[AnalysisIntent, Dict[str, Any]]:
        """
        Parse user query to extract intent and entities
        """
        query_lower = query.lower()
        
        # Classify intent
        intent = self._classify_intent(query_lower)
        
        # Extract entities
        entities = self._extract_entities(query, query_lower)
        
        return intent, entities
    
    def _classify_intent(self, query: str) -> AnalysisIntent:
        """
        Classify the intent of the query
        """
        intent_scores = {}
        
        # Score each intent based on pattern matching
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query):
                    score += 1
            intent_scores[intent] = score
        
        # Get the intent with highest score
        if max(intent_scores.values()) > 0:
            return max(intent_scores, key=intent_scores.get)
        
        # Default to general query if no specific intent found
        return AnalysisIntent.GENERAL_QUERY
    
    def _extract_entities(self, query: str, query_lower: str) -> Dict[str, Any]:
        """
        Extract entities from the query
        """
        entities = {
            'symbols': [],
            'timeframe': None,
            'indicators': [],
            'date_range': None,
            'comparison_symbols': []
        }
        
        # Extract symbols
        entities['symbols'] = self._extract_symbols(query)
        
        # Extract timeframe
        entities['timeframe'] = self._extract_timeframe(query_lower)
        
        # Extract indicators
        entities['indicators'] = self._extract_indicators(query_lower)
        
        # Extract date range if present
        entities['date_range'] = self._extract_date_range(query)
        
        # For comparison queries, extract comparison symbols
        if 'vs' in query_lower or 'versus' in query_lower or 'compare' in query_lower:
            entities['comparison_symbols'] = self._extract_comparison_symbols(query)
        
        return entities
    
    def _extract_symbols(self, query: str) -> List[str]:
        """
        Extract stock symbols from query
        """
        symbols = []
        
        # Pattern for stock symbols (1-5 uppercase letters)
        symbol_pattern = r'\b[A-Z]{1,5}\b'
        potential_symbols = re.findall(symbol_pattern, query)
        
        # Filter to known symbols or likely stock symbols
        for symbol in potential_symbols:
            if symbol in self.common_symbols or len(symbol) <= 4:
                symbols.append(symbol)
        
        # Default to SPY if no symbol found
        if not symbols:
            symbols = ['SPY']
            
        return symbols
    
    def _extract_timeframe(self, query: str) -> str:
        """
        Extract timeframe from query
        """
        for timeframe, patterns in self.timeframe_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    return timeframe
        
        # Default timeframe based on context
        if 'intraday' in query or 'scalp' in query:
            return '5m'
        elif 'swing' in query:
            return '1d'
        else:
            return '1h'  # Default
    
    def _extract_indicators(self, query: str) -> List[str]:
        """
        Extract technical indicators from query
        """
        indicators = []
        
        for indicator, patterns in self.indicator_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    indicators.append(indicator)
                    break
        
        return indicators
    
    def _extract_date_range(self, query: str) -> Optional[Dict[str, str]]:
        """
        Extract date range from query
        """
        # Common date patterns
        date_patterns = [
            r'last (\d+) (days?|weeks?|months?)',
            r'past (\d+) (days?|weeks?|months?)',
            r'from (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})',
            r'since (\w+)',
            r'year to date',
            r'ytd'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, query.lower())
            if match:
                # Parse and return date range
                # Implementation would convert to actual dates
                return {'pattern': pattern, 'match': match.group()}
        
        return None
    
    def _extract_comparison_symbols(self, query: str) -> List[str]:
        """
        Extract symbols for comparison queries
        """
        # Split by comparison keywords
        parts = re.split(r'\s+(?:vs|versus|compare|with)\s+', query, flags=re.IGNORECASE)
        
        comparison_symbols = []
        if len(parts) > 1:
            # Extract symbols from the comparison part
            symbols = self._extract_symbols(parts[1])
            comparison_symbols.extend(symbols)
        
        return comparison_symbols
    
    def enhance_query_understanding(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance query understanding with context
        """
        # Use conversation context to improve understanding
        enhanced = {
            'original_query': query,
            'context_symbols': context.get('recent_symbols', []),
            'context_timeframe': context.get('current_timeframe'),
            'context_indicators': context.get('active_indicators', [])
        }
        
        # Merge with extracted entities
        intent, entities = self.parse_query(query)
        
        # Use context if entities are missing
        if not entities['symbols'] and enhanced['context_symbols']:
            entities['symbols'] = enhanced['context_symbols'][:1]
        
        if not entities['timeframe'] and enhanced['context_timeframe']:
            entities['timeframe'] = enhanced['context_timeframe']
        
        return {
            'intent': intent,
            'entities': entities,
            'context': enhanced
        }


# Example usage
async def test_nlp_service():
    nlp = NLPService()
    
    test_queries = [
        "Analyze AAPL technical setup on the daily chart",
        "What's the sentiment for TSLA?",
        "Predict where SPY will be next week",
        "Show me head and shoulders pattern on NVDA",
        "Compare AAPL vs MSFT performance",
        "What's the risk of holding QQQ calls?",
        "Analyze SPY with RSI and MACD on 1 hour chart"
    ]
    
    for query in test_queries:
        intent, entities = await nlp.parse_query(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {intent.value}")
        print(f"Entities: {entities}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_nlp_service()) 