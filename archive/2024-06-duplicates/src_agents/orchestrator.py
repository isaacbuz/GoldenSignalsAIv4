"""
Agent Orchestrator - GoldenSignalsAI V3

Multi-agent coordination, consensus building, and signal fusion.
Manages the entire agent ecosystem and orchestrates signal generation.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import uuid
import yaml
import importlib
from pathlib import Path
import logging
import json
from abc import ABC, abstractmethod

from .base import BaseAgent
from .technical_analysis import TechnicalAnalysisAgent
from .sentiment_analysis import SentimentAnalysisAgent
from .momentum import MomentumAgent
from .mean_reversion import MeanReversionAgent
from .volume_analysis import VolumeAnalysisAgent
from src.services.signal_service import SignalService
from src.services.market_data_service import MarketDataService
from src.websocket.manager import WebSocketManager
from src.models.signals import Signal, SignalType, SignalStrength
from src.utils.explain import ExplanationEngine
from src.utils.gatekeeper import SignalGatekeeper


class AgentOrchestrator:
    """
    Orchestrates multiple AI agents for comprehensive signal generation
    """
    
    def __init__(
        self,
        signal_service: SignalService,
        market_data_service: MarketDataService,
        websocket_manager: WebSocketManager
    ):
        self.signal_service = signal_service
        self.market_data_service = market_data_service
        self.websocket_manager = websocket_manager
        
        # Agent registry
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_weights: Dict[str, float] = {}
        self.agent_performance: Dict[str, Dict[str, Any]] = {}
        
        # Orchestration settings
        self.consensus_threshold = 0.6
        self.min_agents_required = 3
        self.signal_cooldown = 300  # 5 minutes between signals for same symbol
        
        # Tracking
        self.active_symbols: set = set()
        self.last_signals: Dict[str, datetime] = {}
        self._running = False
        self._initialized = False
        
        self.signal_gate = SignalGatekeeper(min_confidence=0.6, max_risk_score=0.8)
    
    @property
    def is_running(self) -> bool:
        """Check if orchestrator is running"""
        return self._running
    
    async def initialize(self) -> None:
        """Initialize the agent orchestrator and all agents"""
        try:
            # Initialize agents
            await self._initialize_agents()
            
            # Load agent performance history
            await self._load_agent_performance()
            
            self._initialized = True
            logger.info(f"AgentOrchestrator initialized with {len(self.agents)} agents")
            
            # --------------------------------------------------------------
            # Load legacy adapters defined in config/legacy_agents.yml
            # --------------------------------------------------------------
            legacy_cfg_path = Path(__file__).resolve().parents[2] / "config" / "legacy_agents.yml"
            if legacy_cfg_path.exists():
                try:
                    with open(legacy_cfg_path, "r") as f:
                        legacy_cfg = yaml.safe_load(f) or {}
                    for entry in legacy_cfg.get("legacy_agents", []):
                        if not entry.get("enabled", True):
                            continue
                        adapter_path = entry["adapter"]
                        weight = float(entry.get("weight", 0.05))
                        module_name, class_name = adapter_path.rsplit(".", 1)
                        module = importlib.import_module(module_name)
                        adapter_cls = getattr(module, class_name)
                        adapter_instance = adapter_cls(
                            db_manager=self.signal_service.db_manager,
                            redis_manager=self.signal_service.redis_manager,
                        )
                        await adapter_instance.initialize()
                        self.agents[adapter_instance.config.name] = adapter_instance
                        self.agent_weights[adapter_instance.config.name] = weight
                        logger.info(f"Loaded legacy adapter {adapter_instance.config.name} (weight={weight})")
                except Exception as e:
                    logger.error(f"Failed to load legacy adapters: {str(e)}")
            else:
                logger.warning("No legacy agent manifest found; skipping legacy adapters")
            
        except Exception as e:
            logger.error(f"Failed to initialize AgentOrchestrator: {str(e)}")
            raise
    
    async def _initialize_agents(self) -> None:
        """Initialize all trading agents"""
        try:
            # Technical Analysis Agent
            technical_agent = TechnicalAnalysisAgent(
                name="technical_analysis",
                db_manager=self.signal_service.db_manager,
                redis_manager=self.signal_service.redis_manager
            )
            await technical_agent.initialize()
            self.agents["technical_analysis"] = technical_agent
            self.agent_weights["technical_analysis"] = 0.25
            
            # Sentiment Analysis Agent
            sentiment_agent = SentimentAnalysisAgent(
                name="sentiment_analysis",
                db_manager=self.signal_service.db_manager,
                redis_manager=self.signal_service.redis_manager
            )
            await sentiment_agent.initialize()
            self.agents["sentiment_analysis"] = sentiment_agent
            self.agent_weights["sentiment_analysis"] = 0.20
            
            # Momentum Agent
            momentum_agent = MomentumAgent(
                name="momentum",
                db_manager=self.signal_service.db_manager,
                redis_manager=self.signal_service.redis_manager
            )
            await momentum_agent.initialize()
            self.agents["momentum"] = momentum_agent
            self.agent_weights["momentum"] = 0.25
            
            # Mean Reversion Agent
            mean_reversion_agent = MeanReversionAgent(
                name="mean_reversion",
                db_manager=self.signal_service.db_manager,
                redis_manager=self.signal_service.redis_manager
            )
            await mean_reversion_agent.initialize()
            self.agents["mean_reversion"] = mean_reversion_agent
            self.agent_weights["mean_reversion"] = 0.20
            
            # Volume Analysis Agent
            volume_agent = VolumeAnalysisAgent(
                name="volume_analysis",
                db_manager=self.signal_service.db_manager,
                redis_manager=self.signal_service.redis_manager
            )
            await volume_agent.initialize()
            self.agents["volume_analysis"] = volume_agent
            self.agent_weights["volume_analysis"] = 0.10
            
            logger.info("All trading agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {str(e)}")
            raise
    
    async def _load_agent_performance(self) -> None:
        """Load historical performance data for all agents"""
        try:
            for agent_name in self.agents.keys():
                # This would load from database in production
                self.agent_performance[agent_name] = {
                    "total_signals": 0,
                    "correct_signals": 0,
                    "accuracy": 0.5,  # Start with neutral
                    "avg_confidence": 0.5,
                    "last_updated": datetime.utcnow()
                }
            
            logger.info("Agent performance data loaded")
            
        except Exception as e:
            logger.error(f"Failed to load agent performance: {str(e)}")
    
    async def start_signal_generation(self) -> None:
        """Start the main signal generation loop"""
        if not self._initialized:
            raise RuntimeError("AgentOrchestrator not initialized")
        
        self._running = True
        logger.info("Starting agent orchestrator signal generation")
        
        while self._running:
            try:
                # Get symbols to analyze
                symbols_to_analyze = await self._get_symbols_to_analyze()
                
                if symbols_to_analyze:
                    # Process symbols in parallel
                    tasks = [
                        self._analyze_symbol(symbol) 
                        for symbol in symbols_to_analyze
                    ]
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # Update agent performance
                await self._update_agent_performance()
                
                # Wait before next iteration
                await asyncio.sleep(30)  # 30-second intervals
                
            except Exception as e:
                logger.error(f"Error in signal generation loop: {str(e)}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _get_symbols_to_analyze(self) -> List[str]:
        """Get list of symbols to analyze based on subscriptions and watchlists"""
        try:
            # Get subscribed symbols from Redis
            subscribed_symbols = await self.signal_service.redis_manager.get_subscribed_symbols()
            
            # Add default watchlist if no subscriptions
            if not subscribed_symbols:
                subscribed_symbols = ["AAPL", "TSLA", "GOOGL", "AMZN", "MSFT", "NVDA", "SPY", "QQQ"]
            
            # Filter out symbols that are in cooldown
            current_time = datetime.utcnow()
            eligible_symbols = []
            
            for symbol in subscribed_symbols:
                last_signal_time = self.last_signals.get(symbol)
                if not last_signal_time or (current_time - last_signal_time).total_seconds() > self.signal_cooldown:
                    eligible_symbols.append(symbol)
            
            return eligible_symbols[:10]  # Limit to 10 symbols per iteration
            
        except Exception as e:
            logger.error(f"Failed to get symbols to analyze: {str(e)}")
            return []
    
    async def _analyze_symbol(self, symbol: str) -> Optional[Signal]:
        """
        Analyze a symbol using all agents and generate consensus signal
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            Consensus signal or None
        """
        try:
            # Get market data for the symbol
            market_data = await self.market_data_service.get_quote(symbol)
            if not market_data:
                logger.warning(f"No market data available for {symbol}")
                return None
            
            # Get historical data for analysis
            historical_data = await self.market_data_service.get_historical_data(
                symbol, period="5d", interval="1h"
            )
            
            # Collect signals from all agents
            agent_signals = []
            
            for agent_name, agent in self.agents.items():
                try:
                    signal = await agent.analyze(symbol, market_data, historical_data)
                    if signal:
                        agent_signals.append((agent_name, signal))
                        logger.debug(f"Agent {agent_name} signal for {symbol}: {signal.signal_type} ({signal.confidence:.2f})")
                        
                except Exception as e:
                    logger.warning(f"Agent {agent_name} failed for {symbol}: {str(e)}")
                    continue
            
            # Generate consensus signal
            if len(agent_signals) >= self.min_agents_required:
                consensus_signal = await self._build_consensus(symbol, agent_signals, market_data)
                if consensus_signal:
                    # Store and broadcast the signal
                    await self._publish_signal(consensus_signal)
                    self.last_signals[symbol] = datetime.utcnow()
                    return consensus_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to analyze symbol {symbol}: {str(e)}")
            return None
    
    async def _build_consensus(
        self,
        symbol: str,
        agent_signals: List[Tuple[str, Any]],
        market_data: Dict[str, Any]
    ) -> Optional[Signal]:
        """
        Build consensus signal from multiple agent outputs
        
        Args:
            symbol: Stock symbol
            agent_signals: List of (agent_name, signal) tuples
            market_data: Current market data
            
        Returns:
            Consensus signal or None
        """
        try:
            if not agent_signals:
                return None
            
            # Separate signals by type
            buy_signals = []
            sell_signals = []
            hold_signals = []
            
            for agent_name, signal in agent_signals:
                agent_weight = self.agent_weights.get(agent_name, 1.0)
                weighted_confidence = signal.confidence * agent_weight
                
                if signal.signal_type == SignalType.BUY:
                    buy_signals.append((agent_name, signal, weighted_confidence))
                elif signal.signal_type == SignalType.SELL:
                    sell_signals.append((agent_name, signal, weighted_confidence))
                else:
                    hold_signals.append((agent_name, signal, weighted_confidence))
            
            # Calculate weighted votes
            total_weight = sum(self.agent_weights.get(name, 1.0) for name, _ in agent_signals)
            buy_score = sum(conf for _, _, conf in buy_signals) / total_weight if total_weight > 0 else 0
            sell_score = sum(conf for _, _, conf in sell_signals) / total_weight if total_weight > 0 else 0
            hold_score = sum(conf for _, _, conf in hold_signals) / total_weight if total_weight > 0 else 0
            
            # Determine consensus
            max_score = max(buy_score, sell_score, hold_score)
            
            if max_score < self.consensus_threshold:
                logger.debug(f"No consensus reached for {symbol} (max score: {max_score:.2f})")
                return None
            
            # Determine signal type and strength
            if buy_score == max_score:
                signal_type = SignalType.BUY
                consensus_signals = buy_signals
            elif sell_score == max_score:
                signal_type = SignalType.SELL
                consensus_signals = sell_signals
            else:
                signal_type = SignalType.HOLD
                consensus_signals = hold_signals
            
            # Calculate consensus confidence
            consensus_confidence = max_score
            
            # Determine signal strength
            if consensus_confidence >= 0.8:
                strength = SignalStrength.STRONG
            elif consensus_confidence >= 0.6:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
            
            # Calculate risk metrics
            risk_score = await self._calculate_risk_score(symbol, market_data, consensus_signals)
            
            # Create consensus signal
            consensus_signal = Signal(
                signal_id=str(uuid.uuid4()),
                symbol=symbol,
                signal_type=signal_type,
                confidence=consensus_confidence,
                strength=strength,
                source="agent_orchestrator",
                current_price=market_data.get("price"),
                target_price=await self._calculate_target_price(symbol, signal_type, market_data),
                stop_loss=await self._calculate_stop_loss(symbol, signal_type, market_data),
                risk_score=risk_score,
                reasoning=await self._generate_reasoning(symbol, consensus_signals, signal_type),
                features={
                    "agent_votes": {
                        "buy_score": buy_score,
                        "sell_score": sell_score,
                        "hold_score": hold_score,
                        "participating_agents": len(agent_signals),
                        "total_agents": len(self.agents)
                    }
                },
                expires_at=datetime.utcnow() + timedelta(hours=2)
            )
            
            # Attach explainability tree
            try:
                agent_outputs_raw = {
                    name: {
                        "signal": sig.signal_type.value,
                        "confidence": sig.confidence,
                        "strength": sig.strength.value,
                    }
                    for name, sig, _ in agent_signals
                }
                explanation_tree = ExplanationEngine().generate(
                    agent_outputs=agent_outputs_raw,
                    meta={
                        "signal_type": signal_type.value,
                        "confidence": consensus_confidence,
                    },
                )
                consensus_signal.features["explanation"] = explanation_tree
            except Exception as ex:
                logger.warning(f"Failed to build explanation tree: {ex}")
            
            logger.info(
                f"Consensus signal generated for {symbol}: {signal_type.value} "
                f"(confidence: {consensus_confidence:.2f}, strength: {strength.value})"
            )
            
            return consensus_signal
            
        except Exception as e:
            logger.error(f"Failed to build consensus for {symbol}: {str(e)}")
            return None
    
    async def _calculate_risk_score(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        consensus_signals: List[Tuple[str, Any, float]]
    ) -> float:
        """Calculate risk score for the consensus signal"""
        try:
            # Base risk from market data
            price = market_data.get("price", 0)
            volume = market_data.get("volume", 0)
            change_percent = abs(market_data.get("change_percent", 0))
            
            # Volatility risk (higher change = higher risk)
            volatility_risk = min(change_percent / 10.0, 1.0)  # Cap at 1.0
            
            # Volume risk (lower volume = higher risk)
            avg_volume = 1000000  # This should come from historical data
            volume_risk = max(0, 1.0 - (volume / avg_volume)) if volume > 0 else 1.0
            
            # Agent disagreement risk
            agreement_risk = 1.0 - (len(consensus_signals) / len(self.agents))
            
            # Combine risk factors
            total_risk = (volatility_risk * 0.4 + volume_risk * 0.3 + agreement_risk * 0.3)
            
            return min(max(total_risk, 0.0), 1.0)  # Clamp between 0 and 1
            
        except Exception as e:
            logger.error(f"Failed to calculate risk score: {str(e)}")
            return 0.5  # Default moderate risk
    
    async def _calculate_target_price(
        self,
        symbol: str,
        signal_type: SignalType,
        market_data: Dict[str, Any]
    ) -> Optional[float]:
        """Calculate target price based on signal type and market data"""
        try:
            current_price = market_data.get("price", 0)
            if current_price <= 0:
                return None
            
            # Simple target price calculation (this could be much more sophisticated)
            if signal_type == SignalType.BUY:
                return current_price * 1.05  # 5% upside target
            elif signal_type == SignalType.SELL:
                return current_price * 0.95  # 5% downside target
            else:
                return current_price  # Hold at current price
                
        except Exception as e:
            logger.error(f"Failed to calculate target price: {str(e)}")
            return None
    
    async def _calculate_stop_loss(
        self,
        symbol: str,
        signal_type: SignalType,
        market_data: Dict[str, Any]
    ) -> Optional[float]:
        """Calculate stop loss based on signal type and risk management"""
        try:
            current_price = market_data.get("price", 0)
            if current_price <= 0:
                return None
            
            # Simple stop loss calculation (2% risk)
            if signal_type == SignalType.BUY:
                return current_price * 0.98  # 2% downside stop
            elif signal_type == SignalType.SELL:
                return current_price * 1.02  # 2% upside stop
            else:
                return None  # No stop loss for hold
                
        except Exception as e:
            logger.error(f"Failed to calculate stop loss: {str(e)}")
            return None
    
    async def _generate_reasoning(
        self,
        symbol: str,
        consensus_signals: List[Tuple[str, Any, float]],
        signal_type: SignalType
    ) -> str:
        """Generate human-readable reasoning for the consensus signal"""
        try:
            agent_names = [name for name, _, _ in consensus_signals]
            agent_count = len(consensus_signals)
            total_agents = len(self.agents)
            
            reasoning = f"Consensus {signal_type.value} signal from {agent_count}/{total_agents} agents: "
            reasoning += ", ".join(agent_names)
            
            # Add specific insights from top contributing agents
            if consensus_signals:
                top_signal = max(consensus_signals, key=lambda x: x[2])
                reasoning += f". Primary driver: {top_signal[0]} with {top_signal[2]:.1%} confidence."
            
            return reasoning
            
        except Exception as e:
            logger.error(f"Failed to generate reasoning: {str(e)}")
            return f"Consensus {signal_type.value} signal from multiple agents"
    
    async def _publish_signal(self, signal: Signal) -> None:
        """Publish signal to storage and WebSocket subscribers"""
        try:
            # Gatekeeper check
            if not self.signal_gate.allow(signal):
                reason = self.signal_gate.reason(signal)
                logger.info(f"Signal {signal.signal_id} for {signal.symbol} blocked by gatekeeper: {reason}")
                return
            
            # Store signal in database
            await self.signal_service.store_signal(signal)
            
            # Broadcast via WebSocket
            await self.websocket_manager.broadcast_to_symbol_subscribers(
                signal.symbol,
                {
                    "type": "signal",
                    "signal": {
                        "signal_id": signal.signal_id,
                        "symbol": signal.symbol,
                        "signal_type": signal.signal_type.value,
                        "confidence": signal.confidence,
                        "strength": signal.strength.value,
                        "current_price": signal.current_price,
                        "target_price": signal.target_price,
                        "stop_loss": signal.stop_loss,
                        "risk_score": signal.risk_score,
                        "reasoning": signal.reasoning,
                        "created_at": signal.created_at.isoformat()
                    }
                }
            )
            
            logger.info(f"Published signal {signal.signal_id} for {signal.symbol}")
            
        except Exception as e:
            logger.error(f"Failed to publish signal: {str(e)}")
    
    async def _update_agent_performance(self) -> None:
        """Update agent performance metrics periodically"""
        try:
            # This would update based on actual signal outcomes
            # For now, just update the timestamp
            for agent_name in self.agent_performance:
                self.agent_performance[agent_name]["last_updated"] = datetime.utcnow()
            
            # Adjust agent weights based on performance
            await self._adjust_agent_weights()
            
        except Exception as e:
            logger.error(f"Failed to update agent performance: {str(e)}")
    
    async def _adjust_agent_weights(self) -> None:
        """Dynamically adjust agent weights based on performance"""
        try:
            # This would implement dynamic weight adjustment
            # For now, keep weights static
            pass
            
        except Exception as e:
            logger.error(f"Failed to adjust agent weights: {str(e)}")
    
    async def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get current orchestrator status and metrics"""
        try:
            return {
                "running": self._running,
                "active_agents": len(self.agents),
                "active_symbols": len(self.active_symbols),
                "agent_weights": self.agent_weights,
                "agent_performance": self.agent_performance,
                "consensus_threshold": self.consensus_threshold,
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get orchestrator status: {str(e)}")
            return {"error": str(e)}
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the orchestrator"""
        try:
            self._running = False
            
            # Shutdown all agents
            for agent in self.agents.values():
                await agent.shutdown()
            
            logger.info("AgentOrchestrator shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during AgentOrchestrator shutdown: {str(e)}") 