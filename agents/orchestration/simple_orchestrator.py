"""
Simple Agent Orchestrator
Manages multiple trading agents and coordinates signal generation
"""

import asyncio
import json
import logging
import os

# Import our agents
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.core.flow.order_flow_agent import OrderFlowAgent
from agents.core.market.market_profile_agent import MarketProfileAgent
from agents.core.options.simple_options_flow_agent import SimpleOptionsFlowAgent
from agents.core.sentiment.simple_sentiment_agent import SimpleSentimentAgent
from agents.core.technical.adx_agent import ADXAgent
from agents.core.technical.atr_agent import ATRAgent

# Phase 2 agents
from agents.core.technical.bollinger_bands_agent import BollingerBandsAgent
from agents.core.technical.ema_agent import EMAAgent
from agents.core.technical.fibonacci_agent import FibonacciAgent

# Phase 3 agents
from agents.core.technical.ichimoku_agent import IchimokuAgent
from agents.core.technical.ma_crossover_agent import MACrossoverAgent
from agents.core.technical.macd_agent import MACDAgent
from agents.core.technical.parabolic_sar_agent import ParabolicSARAgent

# Phase 1 agents
from agents.core.technical.simple_working_agent import SimpleRSIAgent
from agents.core.technical.std_dev_agent import StandardDeviationAgent
from agents.core.technical.stochastic_agent import StochasticAgent
from agents.core.technical.volume_spike_agent import VolumeSpikeAgent
from agents.core.technical.vwap_agent import VWAPAgent

# Phase 4 agents
from agents.core.volume.volume_profile_agent import VolumeProfileAgent

# Meta agents
from agents.meta.simple_consensus_agent import SimpleConsensusAgent

logger = logging.getLogger(__name__)

class SimpleOrchestrator:
    """Orchestrates multiple trading agents and generates consensus signals"""

    def __init__(self, symbols: Optional[List[str]] = None):
        self.symbols = symbols or ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        self.agents = {}
        self.consensus_agent = SimpleConsensusAgent()
        self.executor = ThreadPoolExecutor(max_workers=20)  # Increased for Phase 4 agents
        self.running = False
        self.signal_cache = {}
        self.performance_metrics = {}

        # Initialize agents
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all trading agents"""
        logger.info("Initializing trading agents...")

        # Phase 1 Technical agents
        self.agents['rsi'] = SimpleRSIAgent(period=14)
        self.agents['macd'] = MACDAgent()
        self.agents['volume'] = VolumeSpikeAgent()
        self.agents['ma_cross'] = MACrossoverAgent(fast_ma=20, slow_ma=50)

        # Phase 2 Technical agents
        self.agents['bollinger'] = BollingerBandsAgent(period=20, std_dev=2.0)
        self.agents['stochastic'] = StochasticAgent(k_period=14, d_period=3)
        self.agents['ema'] = EMAAgent(ema_periods=[8, 13, 21, 34, 55, 89])
        self.agents['atr'] = ATRAgent(atr_period=14, risk_multiplier=2.0)
        self.agents['vwap'] = VWAPAgent()

        # Phase 3 Advanced agents
        self.agents['ichimoku'] = IchimokuAgent()
        self.agents['fibonacci'] = FibonacciAgent()
        self.agents['adx'] = ADXAgent()
        self.agents['parabolic_sar'] = ParabolicSARAgent()
        self.agents['std_dev'] = StandardDeviationAgent()

        # Phase 4 Market Analysis agents
        self.agents['volume_profile'] = VolumeProfileAgent()
        self.agents['market_profile'] = MarketProfileAgent()
        self.agents['order_flow'] = OrderFlowAgent()
        self.agents['sentiment'] = SimpleSentimentAgent()
        self.agents['options_flow'] = SimpleOptionsFlowAgent()

        # Initialize performance tracking
        for agent_name in self.agents:
            self.performance_metrics[agent_name] = {
                'total_signals': 0,
                'correct_signals': 0,
                'accuracy': 0.0,
                'avg_confidence': 0.0,
                'signal_history': []
            }

        logger.info(f"Initialized {len(self.agents)} trading agents (Phase 1-4)")

    def run_agent(self, agent_name: str, agent: Any, symbol: str) -> Dict[str, Any]:
        """Run a single agent and return its signal"""
        try:
            signal = agent.generate_signal(symbol)

            # Track performance (simplified)
            if signal.get('confidence', 0) > 0:
                metrics = self.performance_metrics[agent_name]
                metrics['total_signals'] += 1

                # Update average confidence
                history = metrics['signal_history']
                history.append(signal.get('confidence', 0))
                if len(history) > 100:  # Keep last 100 signals
                    history.pop(0)
                metrics['avg_confidence'] = sum(history) / len(history) if history else 0

            return signal

        except Exception as e:
            logger.error(f"Error running {agent_name} for {symbol}: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "metadata": {
                    "agent": agent_name,
                    "error": str(e)
                }
            }

    def generate_signals_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """Generate signals from all agents for a symbol"""
        logger.info(f"Generating signals for {symbol} with {len(self.agents)} agents")

        # Run all agents in parallel
        agent_futures = []
        for agent_name, agent in self.agents.items():
            future = self.executor.submit(self.run_agent, agent_name, agent, symbol)
            agent_futures.append((agent_name, future))

        # Collect results
        agent_signals = []
        agent_breakdown = {}

        for agent_name, future in agent_futures:
            try:
                signal = future.result(timeout=30)
                agent_signals.append(signal)

                # Store individual agent results for breakdown
                agent_breakdown[agent_name] = {
                    "action": signal.get("action", "HOLD"),
                    "confidence": signal.get("confidence", 0.0),
                    "reasoning": signal.get("metadata", {}).get("reasoning", "")
                }

            except Exception as e:
                logger.error(f"Timeout or error getting signal from {agent_name}: {e}")
                agent_breakdown[agent_name] = {
                    "action": "ERROR",
                    "confidence": 0.0,
                    "reasoning": str(e)
                }

        # Generate consensus
        consensus_signal = self.consensus_agent.combine_signals(agent_signals)

        # Add symbol and agent breakdown to consensus
        consensus_signal['symbol'] = symbol
        consensus_signal['timestamp'] = datetime.now().isoformat()
        consensus_signal['metadata']['agent_breakdown'] = agent_breakdown
        consensus_signal['metadata']['total_agents'] = len(self.agents)

        # Cache the signal
        self.signal_cache[symbol] = consensus_signal

        return consensus_signal

    def generate_all_signals(self) -> List[Dict[str, Any]]:
        """Generate signals for all symbols"""
        all_signals = []

        for symbol in self.symbols:
            try:
                signal = self.generate_signals_for_symbol(symbol)
                all_signals.append(signal)
            except Exception as e:
                logger.error(f"Error generating signals for {symbol}: {e}")

        return all_signals

    def get_latest_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest cached signal for a symbol"""
        return self.signal_cache.get(symbol)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all agents"""
        total_signals = sum(m['total_signals'] for m in self.performance_metrics.values())

        # Calculate average accuracy (mock for now)
        for agent_name, metrics in self.performance_metrics.items():
            if metrics['total_signals'] > 0:
                # Mock accuracy based on confidence
                metrics['accuracy'] = 65 + (metrics['avg_confidence'] * 20)  # 65-85% range

        return {
            "agents": self.performance_metrics,
            "summary": {
                "total_agents": len(self.agents),
                "total_signals": total_signals,
                "active_symbols": len(self.symbols),
                "phase_1_agents": 4,
                "phase_2_agents": 5,
                "phase_3_agents": 5,
                "phase_4_agents": 5
            },
            "timestamp": datetime.now().isoformat()
        }

    def start_signal_generation(self, interval_seconds: int = 300):
        """Start continuous signal generation"""
        self.running = True

        def generation_loop():
            while self.running:
                try:
                    logger.info("Running signal generation cycle...")
                    signals = self.generate_all_signals()
                    logger.info(f"Generated {len(signals)} signals")

                    # Log summary
                    for signal in signals:
                        action_counts = {}
                        for agent, data in signal['metadata'].get('agent_breakdown', {}).items():
                            action = data['action']
                            action_counts[action] = action_counts.get(action, 0) + 1

                        logger.info(
                            f"{signal['symbol']}: {signal['action']} "
                            f"(confidence: {signal['confidence']:.2f}, "
                            f"breakdown: {action_counts})"
                        )

                except Exception as e:
                    logger.error(f"Error in signal generation loop: {e}")

                # Wait for next cycle
                time.sleep(interval_seconds)

        # Start in background thread
        generation_thread = threading.Thread(target=generation_loop, daemon=True)
        generation_thread.start()
        logger.info(f"Started signal generation with {interval_seconds}s interval")

    def stop(self):
        """Stop the orchestrator"""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("Orchestrator stopped")

    def to_json(self, signal: Dict[str, Any]) -> str:
        """Convert signal to JSON format for API"""
        # Transform to match frontend expectations
        return {
            "signal_id": f"{signal.get('symbol', 'UNKNOWN')}_{int(time.time())}",
            "symbol": signal.get('symbol', 'UNKNOWN'),
            "signal_type": signal.get('action', 'HOLD'),
            "confidence": signal.get('confidence', 0.0),
            "strength": self._get_strength(signal.get('confidence', 0.0)),
            "current_price": signal.get('metadata', {}).get('indicators', {}).get('price', 0),
            "reasoning": signal.get('metadata', {}).get('reasoning', ''),
            "indicators": signal.get('metadata', {}).get('consensus_details', {}),
            "timestamp": signal.get('timestamp', datetime.now().isoformat())
        }

    def _get_strength(self, confidence: float) -> str:
        """Convert confidence to strength label"""
        if confidence >= 0.8:
            return "STRONG"
        elif confidence >= 0.6:
            return "MODERATE"
        else:
            return "WEAK"


# Example usage
if __name__ == "__main__":
    # Initialize orchestrator
    orchestrator = SimpleOrchestrator(symbols=['AAPL', 'GOOGL', 'TSLA'])

    # Generate signals once
    print("🤖 Enhanced Orchestrator Test (Phase 1-4: 19 Agents)\n" + "="*50)
    signals = orchestrator.generate_all_signals()

    for signal in signals:
        print(f"\n📊 {signal['symbol']}")
        print(f"Action: {signal['action']}")
        print(f"Confidence: {signal['confidence']:.2%}")
        print(f"Reasoning: {signal['metadata']['reasoning']}")
        print(f"Contributing Agents: {len(signal['metadata'].get('agent_breakdown', {}))}")

    # Show performance
    print("\n📈 Performance Metrics")
    print(json.dumps(orchestrator.get_performance_metrics(), indent=2))
