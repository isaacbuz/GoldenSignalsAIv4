"""
Hybrid Agent Orchestrator
Manages agents with independent/collaborative signals and sentiment tracking
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import base classes
from agents.common.data_bus import AgentDataBus
from agents.common.hybrid_agent_base import HybridAgent, SentimentAggregator
from agents.core.technical.hybrid_bollinger_agent import HybridBollingerAgent
from agents.core.technical.hybrid_macd_agent import HybridMACDAgent
from agents.core.technical.hybrid_pattern_agent import HybridPatternAgent

# Import hybrid agents from their actual locations
from agents.core.technical.hybrid_rsi_agent import HybridRSIAgent
from agents.core.technical.hybrid_volume_agent import HybridVolumeAgent
from agents.hybrid.hybrid_sentiment_flow_agent import HybridSentimentFlowAgent

# Import meta agents
from agents.meta.enhanced_ml_meta_agent import EnhancedMLMetaAgent

logger = logging.getLogger(__name__)

class HybridOrchestrator:
    """
    Orchestrates hybrid agents with sentiment tracking and performance optimization
    """

    def __init__(self, symbols: Optional[List[str]] = None):
        self.symbols = symbols or ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']

        # Core infrastructure
        self.data_bus = AgentDataBus()
        self.sentiment_aggregator = SentimentAggregator()
        self.ml_meta_agent = EnhancedMLMetaAgent(self.data_bus)

        # Agent registry
        self.agents = {}
        self.agent_performance = {}

        # Execution
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.running = False

        # Caching
        self.signal_cache = {}
        self.sentiment_cache = {}

        # Initialize agents
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all hybrid agents"""
        logger.info("Initializing hybrid trading agents...")

        # Technical agents with hybrid capability
        self.agents['rsi'] = HybridRSIAgent(self.data_bus)
        self.agents['volume'] = HybridVolumeAgent(self.data_bus)
        self.agents['pattern'] = HybridPatternAgent(self.data_bus)
        self.agents['macd'] = HybridMACDAgent(self.data_bus)
        self.agents['bollinger'] = HybridBollingerAgent(self.data_bus)
        self.agents['sentiment_flow'] = HybridSentimentFlowAgent(self.data_bus)

        # Initialize performance tracking for each agent
        for agent_name in self.agents:
            self.agent_performance[agent_name] = {
                'signals_generated': 0,
                'last_signal': None,
                'sentiment_history': [],
                'divergence_count': 0,
                'performance_metrics': {}
            }

        # Initialize ML meta agent
        self.meta_agent = EnhancedMLMetaAgent(self.data_bus)

        logger.info(f"Initialized {len(self.agents)} hybrid agents with data bus")

    def generate_signals_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """Generate comprehensive signals with sentiment analysis"""
        logger.info(f"Generating hybrid signals for {symbol}")

        # Run all agents in parallel
        agent_futures = []
        for agent_name, agent in self.agents.items():
            future = self.executor.submit(self._run_hybrid_agent, agent_name, agent, symbol)
            agent_futures.append((agent_name, future))

        # Collect results
        agent_signals = []
        agent_breakdown = {}
        sentiment_updates = {}

        for agent_name, future in agent_futures:
            try:
                result = future.result(timeout=30)

                # Extract signal and sentiment
                signal = result['signal']
                sentiment = result['sentiment']

                agent_signals.append(signal)

                # Store detailed breakdown
                agent_breakdown[agent_name] = {
                    "final": {
                        "action": signal.get("action", "HOLD"),
                        "confidence": signal.get("confidence", 0.0),
                        "reasoning": signal.get("metadata", {}).get("reasoning", "")
                    },
                    "components": signal.get("metadata", {}).get("signal_components", {}),
                    "sentiment": sentiment
                }

                # Update sentiment aggregator
                sentiment_updates[agent_name] = sentiment

                # Track performance
                self._update_agent_tracking(agent_name, signal, sentiment)

            except Exception as e:
                logger.error(f"Error getting signal from {agent_name}: {e}")
                agent_breakdown[agent_name] = {
                    "final": {"action": "ERROR", "confidence": 0.0, "reasoning": str(e)},
                    "components": {},
                    "sentiment": {"final": "neutral"}
                }

        # Update sentiment aggregator
        for agent_name, sentiment in sentiment_updates.items():
            self.sentiment_aggregator.update_sentiment(agent_name, sentiment)

        # Get market sentiment
        market_sentiment = self.sentiment_aggregator.get_market_sentiment()

        # Use ML Meta Agent for final signal
        ml_signal = self.ml_meta_agent.optimize_ensemble(
            agent_signals,
            market_data={
                'symbol': symbol,
                'sentiment': market_sentiment,
                'agent_count': len(agent_signals)
            }
        )

        # Build comprehensive response
        comprehensive_signal = {
            'symbol': symbol,
            'action': ml_signal['action'],
            'confidence': ml_signal['confidence'],
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'reasoning': ml_signal['metadata']['reasoning'],
                'ml_optimization': ml_signal['metadata'].get('ml_optimization', {}),
                'agent_breakdown': agent_breakdown,
                'total_agents': len(self.agents),
                'market_sentiment': market_sentiment,
                'divergence_analysis': self._analyze_divergence(agent_breakdown)
            }
        }

        # Cache results
        self.signal_cache[symbol] = comprehensive_signal
        self.sentiment_cache[symbol] = market_sentiment

        return comprehensive_signal

    def _run_hybrid_agent(self, agent_name: str, agent: HybridAgent, symbol: str) -> Dict[str, Any]:
        """Run a hybrid agent and extract both signal and sentiment"""
        try:
            # Generate hybrid signal
            signal = agent.generate_signal(symbol)

            # Get sentiment data
            sentiment = agent.current_sentiment.copy()

            # Get performance metrics
            metrics = agent.get_performance_metrics()

            return {
                'signal': signal,
                'sentiment': sentiment,
                'metrics': metrics
            }

        except Exception as e:
            logger.error(f"Error running hybrid agent {agent_name}: {e}")
            return {
                'signal': {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "metadata": {"agent": agent_name, "error": str(e)}
                },
                'sentiment': {'final': 'neutral'},
                'metrics': {}
            }

    def _analyze_divergence(self, agent_breakdown: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze divergence patterns across agents"""
        divergences = {
            'count': 0,
            'strong_divergences': [],
            'opportunities': []
        }

        # Count divergences
        for agent_name, data in agent_breakdown.items():
            components = data.get('components', {})
            if components:
                div = components.get('divergence', {})
                if div.get('type') != 'none':
                    divergences['count'] += 1

                    if div.get('type') == 'strong':
                        divergences['strong_divergences'].append({
                            'agent': agent_name,
                            'independent': div.get('independent_action'),
                            'collaborative': div.get('collaborative_action')
                        })

        # Identify opportunities
        if divergences['count'] >= 3:
            divergences['opportunities'].append("Multiple agents showing divergence - potential turning point")

        if divergences['strong_divergences']:
            divergences['opportunities'].append("Strong divergence detected - contrarian opportunity")

        return divergences

    def _update_agent_tracking(self, agent_name: str, signal: Dict[str, Any], sentiment: Dict[str, str]):
        """Update agent performance tracking"""
        tracking = self.agent_performance[agent_name]

        tracking['signals_generated'] += 1
        tracking['last_signal'] = {
            'action': signal.get('action'),
            'confidence': signal.get('confidence'),
            'timestamp': datetime.now()
        }

        # Track sentiment history
        tracking['sentiment_history'].append({
            'sentiment': sentiment,
            'timestamp': datetime.now()
        })

        # Keep only recent history
        if len(tracking['sentiment_history']) > 100:
            tracking['sentiment_history'].pop(0)

        # Count divergences
        components = signal.get('metadata', {}).get('signal_components', {})
        if components.get('divergence', {}).get('type') != 'none':
            tracking['divergence_count'] += 1

    def update_agent_performance(self, agent_name: str, signal_id: str, outcome: float):
        """Update agent performance based on signal outcome"""
        if agent_name in self.agents:
            agent = self.agents[agent_name]
            agent.update_performance(signal_id, outcome)

            # Update ML meta agent
            self.ml_meta_agent.update_performance(agent_name, {}, outcome)

    def get_sentiment_analysis(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive sentiment analysis"""
        if symbol and symbol in self.sentiment_cache:
            symbol_sentiment = self.sentiment_cache[symbol]
        else:
            symbol_sentiment = None

        # Get agent sentiments
        agent_sentiments = {}
        for agent_name, agent in self.agents.items():
            agent_sentiments[agent_name] = {
                'current': agent.current_sentiment,
                'metrics': agent.get_performance_metrics()
            }

        # Get market sentiment
        market_sentiment = self.sentiment_aggregator.get_market_sentiment()

        return {
            'market_sentiment': market_sentiment,
            'symbol_sentiment': symbol_sentiment,
            'agent_sentiments': agent_sentiments,
            'sentiment_trends': self._calculate_sentiment_trends(),
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_sentiment_trends(self) -> Dict[str, Any]:
        """Calculate sentiment trends over time"""
        trends = {
            'bullish_momentum': 0,
            'bearish_momentum': 0,
            'sentiment_shifts': []
        }

        # Analyze recent sentiment changes
        for agent_name, tracking in self.agent_performance.items():
            history = tracking['sentiment_history']
            if len(history) >= 2:
                recent = history[-1]['sentiment']['final']
                previous = history[-2]['sentiment']['final']

                if recent != previous:
                    trends['sentiment_shifts'].append({
                        'agent': agent_name,
                        'from': previous,
                        'to': recent,
                        'timestamp': history[-1]['timestamp']
                    })

                # Calculate momentum
                if 'bullish' in recent:
                    trends['bullish_momentum'] += 1
                elif 'bearish' in recent:
                    trends['bearish_momentum'] += 1

        return trends

    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        dashboard = {
            'agents': {},
            'market_performance': {},
            'divergence_analysis': {},
            'ml_optimization': {}
        }

        # Agent performance
        for agent_name, agent in self.agents.items():
            metrics = agent.get_performance_metrics()
            tracking = self.agent_performance[agent_name]

            dashboard['agents'][agent_name] = {
                'performance': metrics,
                'signals_generated': tracking['signals_generated'],
                'divergence_count': tracking['divergence_count'],
                'divergence_rate': tracking['divergence_count'] / max(tracking['signals_generated'], 1),
                'last_signal': tracking['last_signal']
            }

        # Market performance
        total_signals = sum(t['signals_generated'] for t in self.agent_performance.values())
        total_divergences = sum(t['divergence_count'] for t in self.agent_performance.values())

        dashboard['market_performance'] = {
            'total_signals': total_signals,
            'total_divergences': total_divergences,
            'average_divergence_rate': total_divergences / max(total_signals, 1),
            'active_symbols': len(self.symbols),
            'sentiment': self.sentiment_aggregator.get_market_sentiment()
        }

        # ML optimization metrics
        dashboard['ml_optimization'] = self.ml_meta_agent.get_performance_report()

        return dashboard

    def start_continuous_analysis(self, interval_seconds: int = 300):
        """Start continuous signal generation with sentiment tracking"""
        self.running = True

        def analysis_loop():
            while self.running:
                try:
                    logger.info("Running hybrid signal generation cycle...")

                    for symbol in self.symbols:
                        signal = self.generate_signals_for_symbol(symbol)

                        # Log summary
                        logger.info(
                            f"{symbol}: {signal['action']} "
                            f"(confidence: {signal['confidence']:.2f}, "
                            f"sentiment: {signal['metadata']['market_sentiment']['overall']})"
                        )

                    # Save ML model periodically
                    self.ml_meta_agent.save_model('ml_models/hybrid_meta_model.json')

                except Exception as e:
                    logger.error(f"Error in analysis loop: {e}")

                time.sleep(interval_seconds)

        # Start in background thread
        analysis_thread = threading.Thread(target=analysis_loop, daemon=True)
        analysis_thread.start()
        logger.info(f"Started hybrid analysis with {interval_seconds}s interval")

    def stop(self):
        """Stop the orchestrator"""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("Hybrid orchestrator stopped")


# Example usage
if __name__ == "__main__":
    # Initialize orchestrator
    orchestrator = HybridOrchestrator(symbols=['AAPL', 'GOOGL', 'TSLA'])

    # Generate signals
    print("🤖 Hybrid Orchestrator Test\n" + "="*50)

    signal = orchestrator.generate_signals_for_symbol('AAPL')

    print(f"\n📊 Signal for AAPL")
    print(f"Action: {signal['action']}")
    print(f"Confidence: {signal['confidence']:.2%}")
    print(f"Market Sentiment: {signal['metadata']['market_sentiment']['overall']}")
    print(f"Divergences: {signal['metadata']['divergence_analysis']['count']}")

    # Get sentiment analysis
    sentiment = orchestrator.get_sentiment_analysis()
    print(f"\n🎭 Sentiment Analysis")
    print(f"Market: {sentiment['market_sentiment']}")

    # Get performance dashboard
    dashboard = orchestrator.get_performance_dashboard()
    print(f"\n📈 Performance Dashboard")
    print(json.dumps(dashboard['market_performance'], indent=2))
