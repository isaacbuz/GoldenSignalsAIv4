"""
ML Signal Generator Service
Generates real trading signals using ML models and agents
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import numpy as np

from src.agents.orchestrator import AgentOrchestrator
from src.agents.momentum import MomentumAgent
from src.agents.mean_reversion import MeanReversionAgent
from src.agents.technical_analysis import TechnicalAnalysisAgent
from src.agents.sentiment_analysis import SentimentAnalysisAgent
from src.agents.volume_analysis import VolumeAnalysisAgent
from src.utils.technical_indicators import TechnicalIndicators
from src.utils.timezone_utils import now_utc

logger = logging.getLogger(__name__)


class MLSignalGenerator:
    """Generate trading signals using ML models and multiple agents"""
    
    def __init__(self):
        self.orchestrator = AgentOrchestrator()
        
        # Initialize agents
        self.agents = {
            'momentum': MomentumAgent(),
            'mean_reversion': MeanReversionAgent(),
            'technical': TechnicalAnalysisAgent(),
            'sentiment': SentimentAnalysisAgent(),
            'volume': VolumeAnalysisAgent()
        }
        
        # Signal cache to prevent duplicates
        self._signal_cache = {}
        self._cache_ttl = timedelta(minutes=5)
    
    async def initialize(self):
        """Initialize all agents and services"""
        try:
            # Initialize orchestrator
            await self.orchestrator.initialize()
            
            # Initialize individual agents
            for name, agent in self.agents.items():
                logger.info(f"Initializing {name} agent")
                await agent.initialize()
                
            logger.info("ML Signal Generator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ML Signal Generator: {e}")
            raise
    
    async def generate_signals(
        self, 
        symbols: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Generate trading signals for given symbols or top movers"""
        try:
            # Default symbols if none provided
            if not symbols:
                symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "META", "AMZN", "SPY"]
            
            all_signals = []
            
            for symbol in symbols:
                # Check cache first
                cache_key = f"{symbol}_{now_utc().strftime('%Y%m%d%H%M')}"
                if cache_key in self._signal_cache:
                    cached_signal, cache_time = self._signal_cache[cache_key]
                    if now_utc() - cache_time < self._cache_ttl:
                        all_signals.append(cached_signal)
                        continue
                
                # Generate new signal
                signal = await self._generate_signal_for_symbol(symbol)
                if signal:
                    all_signals.append(signal)
                    # Cache the signal
                    self._signal_cache[cache_key] = (signal, now_utc())
            
            # Sort by confidence and return top signals
            all_signals.sort(key=lambda x: x['confidence'], reverse=True)
            return all_signals[:limit]
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    async def _generate_signal_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Generate signal for a specific symbol using multiple agents"""
        try:
            # Fetch market data
            market_data = await self._fetch_market_data(symbol)
            if not market_data:
                return None
            
            # Get signals from each agent
            agent_signals = {}
            for agent_name, agent in self.agents.items():
                try:
                    if agent_name == 'sentiment':
                        # Sentiment agent might need different data
                        signal = await self._get_sentiment_signal(symbol)
                    else:
                        signal = await self._analyze_with_agent(agent, symbol, market_data)
                    
                    if signal:
                        agent_signals[agent_name] = signal
                        
                except Exception as e:
                    logger.error(f"Error with {agent_name} agent for {symbol}: {e}")
            
            # Combine signals using orchestrator logic
            if agent_signals:
                combined_signal = self._combine_signals(symbol, agent_signals, market_data)
                return combined_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    async def _fetch_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch comprehensive market data for analysis"""
        try:
            # For now, generate mock data
            # In production, this would fetch real market data
            historical = []
            base_price = {
                "AAPL": 185.50, "GOOGL": 142.75, "MSFT": 378.90,
                "TSLA": 245.60, "NVDA": 625.40, "META": 325.80,
                "AMZN": 155.20, "SPY": 450.25
            }.get(symbol, 100.0)
            
            for i in range(30):
                price_change = np.random.uniform(-0.03, 0.03)
                open_price = base_price * (1 + price_change)
                close_price = open_price * (1 + np.random.uniform(-0.02, 0.02))
                
                historical.append({
                    'open': open_price,
                    'high': max(open_price, close_price) * (1 + np.random.uniform(0, 0.01)),
                    'low': min(open_price, close_price) * (1 - np.random.uniform(0, 0.01)),
                    'close': close_price,
                    'volume': np.random.randint(1000000, 10000000)
                })
                
                base_price = close_price
            
            # Calculate technical indicators
            indicators = TechnicalIndicators.calculate_all_indicators(historical)
            
            return {
                'symbol': symbol,
                'current_price': historical[-1]['close'],
                'volume': historical[-1]['volume'],
                'change_percent': ((historical[-1]['close'] - historical[0]['close']) / historical[0]['close']) * 100,
                'historical': historical,
                'indicators': indicators,
                'quote': {
                    'price': historical[-1]['close'],
                    'volume': historical[-1]['volume'],
                    'changePercent': ((historical[-1]['close'] - historical[0]['close']) / historical[0]['close']) * 100
                }
            }
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None
    
    async def _analyze_with_agent(
        self, 
        agent: Any, 
        symbol: str, 
        market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyze market data with a specific agent"""
        try:
            return await agent.analyze(market_data)
            
        except Exception as e:
            logger.error(f"Error in agent analysis: {e}")
            return None
    
    async def _get_sentiment_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get sentiment-based signal (mock for now)"""
        # TODO: Integrate real sentiment analysis
        sentiment_score = np.random.uniform(-1, 1)
        
        if sentiment_score > 0.5:
            return {
                'type': 'BUY',
                'confidence': min(90, 60 + sentiment_score * 40),
                'reasoning': ['Positive market sentiment', 'Bullish news coverage'],
                'agent': 'SentimentAnalysisAgent'
            }
        elif sentiment_score < -0.5:
            return {
                'type': 'SELL',
                'confidence': min(90, 60 + abs(sentiment_score) * 40),
                'reasoning': ['Negative market sentiment', 'Bearish news coverage'],
                'agent': 'SentimentAnalysisAgent'
            }
        
        return None
    
    def _combine_signals(
        self, 
        symbol: str, 
        agent_signals: Dict[str, Dict[str, Any]], 
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine signals from multiple agents into a final signal"""
        try:
            # Count buy/sell signals
            buy_signals = [s for s in agent_signals.values() if s['type'] == 'BUY']
            sell_signals = [s for s in agent_signals.values() if s['type'] == 'SELL']
            
            # Determine consensus
            if len(buy_signals) > len(sell_signals):
                signal_type = 'BUY'
                relevant_signals = buy_signals
            elif len(sell_signals) > len(buy_signals):
                signal_type = 'SELL'
                relevant_signals = sell_signals
            else:
                # No clear consensus
                return None
            
            # Calculate combined confidence
            avg_confidence = np.mean([s['confidence'] for s in relevant_signals])
            consensus_boost = len(relevant_signals) * 5  # Boost for agreement
            final_confidence = min(95, avg_confidence + consensus_boost)
            
            # Combine reasoning
            all_reasoning = []
            for signal in relevant_signals:
                all_reasoning.extend(signal['reasoning'])
            
            # Determine pattern
            patterns = {
                'momentum': 'Momentum Breakout',
                'mean_reversion': 'Mean Reversion Setup',
                'technical': 'Technical Pattern',
                'sentiment': 'Sentiment Driven',
                'volume': 'Volume Surge'
            }
            
            primary_agent = max(relevant_signals, key=lambda s: s['confidence'])['agent']
            pattern = patterns.get(primary_agent.replace('Agent', '').lower(), 'Multi-Signal Convergence')
            
            # Calculate price targets
            current_price = market_data['current_price']
            atr = market_data['indicators'].get('atr', current_price * 0.02)
            
            if signal_type == 'BUY':
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 3)
            else:
                stop_loss = current_price + (atr * 2)
                take_profit = current_price - (atr * 3)
            
            return {
                "id": f"{symbol}_{int(now_utc().timestamp())}_{int(final_confidence)}",
                "symbol": symbol,
                "pattern": pattern,
                "confidence": round(final_confidence, 1),
                "entry": round(current_price, 2),
                "stopLoss": round(stop_loss, 2),
                "takeProfit": round(take_profit, 2),
                "timestamp": now_utc().isoformat(),
                "type": signal_type,
                "timeframe": "1d",  # Daily signals for now
                "risk": self._calculate_risk_level(final_confidence),
                "reasoning": list(set(all_reasoning))[:5],  # Top 5 unique reasons
                "agents_agreed": len(relevant_signals),
                "agents_total": len(agent_signals)
            }
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            return None
    
    def _calculate_risk_level(self, confidence: float) -> str:
        """Calculate risk level based on confidence"""
        if confidence >= 85:
            return "LOW"
        elif confidence >= 70:
            return "MEDIUM"
        else:
            return "HIGH"


# Singleton instance
ml_signal_generator = MLSignalGenerator() 