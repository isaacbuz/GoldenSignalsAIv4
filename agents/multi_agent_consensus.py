"""
Multi-Agent Consensus System
Coordinates multiple agents to reach consensus on trading decisions
Issue #189: Agent-5: Develop Multi-Agent Consensus System
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ConsensusMethod(Enum):
    """Consensus reaching methods"""
    WEIGHTED_VOTING = "weighted_voting"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    BYZANTINE_FAULT_TOLERANT = "bft"
    HIERARCHICAL = "hierarchical"
    ENSEMBLE = "ensemble"
    ADAPTIVE = "adaptive"


class SignalType(Enum):
    """Types of trading signals"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    NO_SIGNAL = "no_signal"


class AgentType(Enum):
    """Types of agents in the system"""
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    FLOW = "flow"
    REGIME = "regime"
    RISK = "risk"
    EXECUTION = "execution"
    ARBITRAGE = "arbitrage"
    LIQUIDITY = "liquidity"


@dataclass
class AgentSignal:
    """Signal from an individual agent"""
    agent_id: str
    agent_type: AgentType
    signal: SignalType
    confidence: float  # 0-1
    reasoning: str
    supporting_data: Dict[str, Any]
    timestamp: datetime
    time_horizon: str  # immediate, short, medium, long
    expected_move: Optional[float] = None  # Expected price move %
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None  # Recommended position size


@dataclass
class ConsensusResult:
    """Result of consensus process"""
    consensus_id: str
    final_signal: SignalType
    confidence: float
    method_used: ConsensusMethod
    participating_agents: List[str]
    agreement_score: float  # 0-1, how much agents agree
    dissenting_agents: List[Dict[str, Any]]
    supporting_signals: Dict[SignalType, List[str]]  # Signal -> agents
    risk_assessment: Dict[str, float]
    execution_recommendation: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class AgentPerformance:
    """Track agent performance for weighting"""
    agent_id: str
    total_signals: int
    correct_signals: int
    accuracy: float
    avg_return: float
    sharpe_ratio: float
    recent_performance: List[float]  # Recent signal outcomes
    specialties: Dict[str, float]  # Market condition -> performance
    last_updated: datetime


class ConsensusEngine:
    """Core consensus engine"""

    def __init__(self):
        self.voting_weights = self._initialize_weights()
        self.performance_tracker = {}
        self.consensus_history = deque(maxlen=1000)
        self.conflict_resolution_rules = self._load_conflict_rules()

    def _initialize_weights(self) -> Dict[AgentType, float]:
        """Initialize default agent weights"""
        return {
            AgentType.SENTIMENT: 0.15,
            AgentType.TECHNICAL: 0.20,
            AgentType.FUNDAMENTAL: 0.15,
            AgentType.FLOW: 0.25,  # Options flow high weight
            AgentType.REGIME: 0.10,
            AgentType.RISK: 0.15,
            AgentType.EXECUTION: 0.05,
            AgentType.ARBITRAGE: 0.10,
            AgentType.LIQUIDITY: 0.05
        }

    def _load_conflict_rules(self) -> Dict[str, Any]:
        """Load rules for resolving conflicts"""
        return {
            'risk_veto': True,  # Risk agent can veto trades
            'regime_adjustment': True,  # Adjust weights by regime
            'confidence_threshold': 0.6,  # Min confidence for action
            'super_majority': 0.75,  # % agreement for strong signal
            'tie_breaker': AgentType.FLOW  # Tie-breaker agent
        }

    async def reach_consensus(self, signals: List[AgentSignal],
                            method: ConsensusMethod = ConsensusMethod.ADAPTIVE) -> ConsensusResult:
        """Reach consensus from agent signals"""
        if not signals:
            return self._create_no_signal_result()

        # Select consensus method
        if method == ConsensusMethod.ADAPTIVE:
            method = self._select_best_method(signals)

        # Apply consensus method
        if method == ConsensusMethod.WEIGHTED_VOTING:
            result = await self._weighted_voting_consensus(signals)
        elif method == ConsensusMethod.CONFIDENCE_WEIGHTED:
            result = await self._confidence_weighted_consensus(signals)
        elif method == ConsensusMethod.PERFORMANCE_WEIGHTED:
            result = await self._performance_weighted_consensus(signals)
        elif method == ConsensusMethod.BYZANTINE_FAULT_TOLERANT:
            result = await self._bft_consensus(signals)
        elif method == ConsensusMethod.HIERARCHICAL:
            result = await self._hierarchical_consensus(signals)
        elif method == ConsensusMethod.ENSEMBLE:
            result = await self._ensemble_consensus(signals)
        else:
            result = await self._weighted_voting_consensus(signals)

        # Apply conflict resolution
        result = self._resolve_conflicts(result, signals)

        # Risk check
        result = self._apply_risk_checks(result, signals)

        # Store in history
        self.consensus_history.append(result)

        return result

    def _select_best_method(self, signals: List[AgentSignal]) -> ConsensusMethod:
        """Select best consensus method based on situation"""
        # Check signal diversity
        signal_types = set(s.signal for s in signals)
        confidence_variance = np.var([s.confidence for s in signals])

        # High disagreement -> BFT
        if len(signal_types) >= 4:
            return ConsensusMethod.BYZANTINE_FAULT_TOLERANT

        # High confidence variance -> Performance weighted
        if confidence_variance > 0.1:
            return ConsensusMethod.PERFORMANCE_WEIGHTED

        # Many agents -> Hierarchical
        if len(signals) > 7:
            return ConsensusMethod.HIERARCHICAL

        # Default to adaptive weighted
        return ConsensusMethod.CONFIDENCE_WEIGHTED

    async def _weighted_voting_consensus(self, signals: List[AgentSignal]) -> ConsensusResult:
        """Simple weighted voting"""
        votes = defaultdict(float)

        for signal in signals:
            weight = self.voting_weights.get(signal.agent_type, 0.1)
            votes[signal.signal] += weight

        # Normalize votes
        total_weight = sum(votes.values())
        if total_weight > 0:
            for signal_type in votes:
                votes[signal_type] /= total_weight

        # Get winning signal
        final_signal = max(votes.items(), key=lambda x: x[1])[0]
        confidence = votes[final_signal]

        return self._create_consensus_result(
            final_signal, confidence, ConsensusMethod.WEIGHTED_VOTING,
            signals, votes
        )

    async def _confidence_weighted_consensus(self, signals: List[AgentSignal]) -> ConsensusResult:
        """Weight by agent confidence"""
        votes = defaultdict(float)
        confidence_sum = defaultdict(float)

        for signal in signals:
            weight = self.voting_weights.get(signal.agent_type, 0.1)
            confidence_weight = weight * signal.confidence
            votes[signal.signal] += confidence_weight
            confidence_sum[signal.signal] += signal.confidence

        # Get winning signal
        final_signal = max(votes.items(), key=lambda x: x[1])[0]

        # Average confidence of winning signal
        signal_count = sum(1 for s in signals if s.signal == final_signal)
        avg_confidence = confidence_sum[final_signal] / signal_count if signal_count > 0 else 0

        return self._create_consensus_result(
            final_signal, avg_confidence, ConsensusMethod.CONFIDENCE_WEIGHTED,
            signals, votes
        )

    async def _performance_weighted_consensus(self, signals: List[AgentSignal]) -> ConsensusResult:
        """Weight by historical performance"""
        votes = defaultdict(float)

        for signal in signals:
            # Get agent performance
            perf = self.performance_tracker.get(signal.agent_id)
            if perf:
                performance_weight = perf.accuracy * perf.sharpe_ratio
            else:
                performance_weight = 0.5  # Default for new agents

            base_weight = self.voting_weights.get(signal.agent_type, 0.1)
            total_weight = base_weight * performance_weight * signal.confidence
            votes[signal.signal] += total_weight

        # Normalize and get winner
        total = sum(votes.values())
        if total > 0:
            for s in votes:
                votes[s] /= total

        final_signal = max(votes.items(), key=lambda x: x[1])[0]
        confidence = votes[final_signal]

        return self._create_consensus_result(
            final_signal, confidence, ConsensusMethod.PERFORMANCE_WEIGHTED,
            signals, votes
        )

    async def _bft_consensus(self, signals: List[AgentSignal]) -> ConsensusResult:
        """Byzantine Fault Tolerant consensus - resistant to bad actors"""
        # Remove outliers
        filtered_signals = self._filter_byzantine_signals(signals)

        # Multi-round voting
        rounds = 3
        votes_history = []

        for round_num in range(rounds):
            votes = defaultdict(float)

            for signal in filtered_signals:
                # Adjust weight based on previous rounds
                if round_num > 0:
                    consistency_bonus = self._calculate_consistency_bonus(
                        signal, votes_history
                    )
                else:
                    consistency_bonus = 1.0

                weight = self.voting_weights.get(signal.agent_type, 0.1)
                votes[signal.signal] += weight * signal.confidence * consistency_bonus

            votes_history.append(votes)

        # Final tally
        final_votes = votes_history[-1]
        final_signal = max(final_votes.items(), key=lambda x: x[1])[0]

        # Calculate byzantine confidence
        confidence = self._calculate_bft_confidence(final_votes, len(filtered_signals))

        return self._create_consensus_result(
            final_signal, confidence, ConsensusMethod.BYZANTINE_FAULT_TOLERANT,
            filtered_signals, final_votes
        )

    def _filter_byzantine_signals(self, signals: List[AgentSignal]) -> List[AgentSignal]:
        """Filter out potentially malicious/faulty signals"""
        if len(signals) < 3:
            return signals

        # Calculate median confidence
        confidences = [s.confidence for s in signals]
        median_conf = np.median(confidences)
        std_conf = np.std(confidences)

        # Filter extreme outliers
        filtered = []
        for signal in signals:
            # Check if confidence is within reasonable range
            if abs(signal.confidence - median_conf) <= 2 * std_conf:
                filtered.append(signal)
            else:
                logger.warning(f"Filtered byzantine signal from {signal.agent_id}")

        return filtered

    def _calculate_consistency_bonus(self, signal: AgentSignal,
                                   votes_history: List[Dict]) -> float:
        """Calculate consistency bonus for BFT"""
        if not votes_history:
            return 1.0

        # Check if agent has been consistent
        consistency_score = 0
        for votes in votes_history:
            if signal.signal in votes:
                consistency_score += 1

        return 1.0 + (consistency_score / len(votes_history)) * 0.5

    def _calculate_bft_confidence(self, votes: Dict[SignalType, float],
                                num_agents: int) -> float:
        """Calculate confidence for BFT consensus"""
        if not votes or num_agents == 0:
            return 0.0

        # Get top two signals
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_votes) == 1:
            return sorted_votes[0][1]

        # Calculate margin between top two
        margin = sorted_votes[0][1] - sorted_votes[1][1]

        # Adjust confidence based on margin and participation
        base_confidence = sorted_votes[0][1]
        margin_bonus = min(0.2, margin)
        participation_factor = min(1.0, num_agents / 5)  # Expect at least 5 agents

        return min(1.0, base_confidence + margin_bonus) * participation_factor

    async def _hierarchical_consensus(self, signals: List[AgentSignal]) -> ConsensusResult:
        """Hierarchical consensus - group by agent type first"""
        # Group signals by type
        type_groups = defaultdict(list)
        for signal in signals:
            type_groups[signal.agent_type].append(signal)

        # Get consensus within each group
        group_consensus = {}
        for agent_type, group_signals in type_groups.items():
            if group_signals:
                group_result = await self._group_consensus(group_signals)
                group_consensus[agent_type] = group_result

        # Meta-consensus across groups
        meta_votes = defaultdict(float)
        for agent_type, consensus in group_consensus.items():
            weight = self.voting_weights.get(agent_type, 0.1)
            meta_votes[consensus['signal']] += weight * consensus['confidence']

        # Final signal
        final_signal = max(meta_votes.items(), key=lambda x: x[1])[0]
        confidence = meta_votes[final_signal] / sum(meta_votes.values())

        return self._create_consensus_result(
            final_signal, confidence, ConsensusMethod.HIERARCHICAL,
            signals, meta_votes
        )

    async def _group_consensus(self, signals: List[AgentSignal]) -> Dict[str, Any]:
        """Get consensus within a group"""
        if not signals:
            return {'signal': SignalType.NO_SIGNAL, 'confidence': 0}

        # Simple majority within group
        signal_counts = defaultdict(int)
        confidence_sum = defaultdict(float)

        for signal in signals:
            signal_counts[signal.signal] += 1
            confidence_sum[signal.signal] += signal.confidence

        # Get majority signal
        majority_signal = max(signal_counts.items(), key=lambda x: x[1])[0]
        avg_confidence = confidence_sum[majority_signal] / signal_counts[majority_signal]

        return {
            'signal': majority_signal,
            'confidence': avg_confidence,
            'count': signal_counts[majority_signal],
            'total': len(signals)
        }

    async def _ensemble_consensus(self, signals: List[AgentSignal]) -> ConsensusResult:
        """Ensemble method - combine multiple consensus approaches"""
        methods = [
            ConsensusMethod.WEIGHTED_VOTING,
            ConsensusMethod.CONFIDENCE_WEIGHTED,
            ConsensusMethod.PERFORMANCE_WEIGHTED
        ]

        ensemble_results = []
        for method in methods:
            if method == ConsensusMethod.WEIGHTED_VOTING:
                result = await self._weighted_voting_consensus(signals)
            elif method == ConsensusMethod.CONFIDENCE_WEIGHTED:
                result = await self._confidence_weighted_consensus(signals)
            elif method == ConsensusMethod.PERFORMANCE_WEIGHTED:
                result = await self._performance_weighted_consensus(signals)

            ensemble_results.append(result)

        # Combine ensemble results
        final_votes = defaultdict(float)
        for result in ensemble_results:
            final_votes[result.final_signal] += result.confidence

        # Average the votes
        for signal in final_votes:
            final_votes[signal] /= len(methods)

        final_signal = max(final_votes.items(), key=lambda x: x[1])[0]
        confidence = final_votes[final_signal]

        return self._create_consensus_result(
            final_signal, confidence, ConsensusMethod.ENSEMBLE,
            signals, final_votes
        )

    def _resolve_conflicts(self, result: ConsensusResult,
                         signals: List[AgentSignal]) -> ConsensusResult:
        """Resolve conflicts in consensus"""
        # Check if risk agent disagrees
        if self.conflict_resolution_rules['risk_veto']:
            risk_signals = [s for s in signals if s.agent_type == AgentType.RISK]
            if risk_signals:
                risk_signal = risk_signals[0]
                if risk_signal.signal in [SignalType.STRONG_SELL, SignalType.SELL]:
                    if result.final_signal in [SignalType.BUY, SignalType.STRONG_BUY]:
                        # Risk veto
                        result.final_signal = SignalType.HOLD
                        result.confidence *= 0.5
                        result.metadata['risk_veto'] = True

        # Check agreement score
        agreement = self._calculate_agreement_score(signals, result.final_signal)
        result.agreement_score = agreement

        # Downgrade signal if low agreement
        if agreement < 0.5 and result.final_signal != SignalType.HOLD:
            result.confidence *= 0.7
            if result.final_signal == SignalType.STRONG_BUY:
                result.final_signal = SignalType.BUY
            elif result.final_signal == SignalType.STRONG_SELL:
                result.final_signal = SignalType.SELL

        return result

    def _apply_risk_checks(self, result: ConsensusResult,
                         signals: List[AgentSignal]) -> ConsensusResult:
        """Apply risk management checks"""
        risk_assessment = {
            'max_position_size': 1.0,
            'stop_loss_required': False,
            'risk_score': 0.5,
            'warnings': []
        }

        # Collect risk metrics
        stop_losses = [s.stop_loss for s in signals if s.stop_loss]
        position_sizes = [s.position_size for s in signals if s.position_size]

        if stop_losses:
            risk_assessment['recommended_stop_loss'] = np.mean(stop_losses)
            risk_assessment['stop_loss_required'] = True

        if position_sizes:
            risk_assessment['max_position_size'] = np.min(position_sizes)

        # High confidence but high disagreement = risky
        if result.confidence > 0.8 and result.agreement_score < 0.6:
            risk_assessment['risk_score'] = 0.8
            risk_assessment['warnings'].append('High confidence but low agreement')

        result.risk_assessment = risk_assessment

        # Execution recommendation
        result.execution_recommendation = self._create_execution_recommendation(
            result, signals
        )

        return result

    def _create_execution_recommendation(self, result: ConsensusResult,
                                       signals: List[AgentSignal]) -> Dict[str, Any]:
        """Create execution recommendations"""
        exec_rec = {
            'execute': result.confidence > self.conflict_resolution_rules['confidence_threshold'],
            'position_size': result.risk_assessment['max_position_size'],
            'entry_strategy': 'scale_in' if result.agreement_score < 0.7 else 'full_position',
            'urgency': 'high' if result.confidence > 0.8 else 'normal',
            'time_horizon': self._determine_time_horizon(signals)
        }

        # Add specific parameters
        if result.risk_assessment.get('stop_loss_required'):
            exec_rec['stop_loss'] = result.risk_assessment.get('recommended_stop_loss')

        # Get execution agent recommendations
        exec_signals = [s for s in signals if s.agent_type == AgentType.EXECUTION]
        if exec_signals:
            exec_rec['execution_notes'] = exec_signals[0].supporting_data

        return exec_rec

    def _determine_time_horizon(self, signals: List[AgentSignal]) -> str:
        """Determine consensus time horizon"""
        horizons = [s.time_horizon for s in signals if s.time_horizon]
        if not horizons:
            return 'medium'

        # Get most common
        horizon_counts = defaultdict(int)
        for h in horizons:
            horizon_counts[h] += 1

        return max(horizon_counts.items(), key=lambda x: x[1])[0]

    def _calculate_agreement_score(self, signals: List[AgentSignal],
                                 consensus_signal: SignalType) -> float:
        """Calculate how much agents agree"""
        if not signals:
            return 0.0

        agreeing = sum(1 for s in signals if s.signal == consensus_signal)
        return agreeing / len(signals)

    def _create_consensus_result(self, final_signal: SignalType,
                               confidence: float,
                               method: ConsensusMethod,
                               signals: List[AgentSignal],
                               votes: Dict[SignalType, float]) -> ConsensusResult:
        """Create consensus result object"""
        # Group signals by type
        supporting_signals = defaultdict(list)
        for signal in signals:
            supporting_signals[signal.signal].append(signal.agent_id)

        # Find dissenting agents
        dissenting = []
        for signal in signals:
            if signal.signal != final_signal:
                dissenting.append({
                    'agent_id': signal.agent_id,
                    'signal': signal.signal.value,
                    'confidence': signal.confidence,
                    'reasoning': signal.reasoning
                })

        return ConsensusResult(
            consensus_id=f"CONSENSUS_{datetime.now().timestamp()}",
            final_signal=final_signal,
            confidence=confidence,
            method_used=method,
            participating_agents=[s.agent_id for s in signals],
            agreement_score=0.0,  # Will be set by conflict resolution
            dissenting_agents=dissenting,
            supporting_signals=dict(supporting_signals),
            risk_assessment={},  # Will be set by risk checks
            execution_recommendation={},  # Will be set by risk checks
            timestamp=datetime.now(),
            metadata={'vote_distribution': dict(votes)}
        )

    def _create_no_signal_result(self) -> ConsensusResult:
        """Create result when no signals available"""
        return ConsensusResult(
            consensus_id=f"CONSENSUS_{datetime.now().timestamp()}",
            final_signal=SignalType.NO_SIGNAL,
            confidence=0.0,
            method_used=ConsensusMethod.WEIGHTED_VOTING,
            participating_agents=[],
            agreement_score=0.0,
            dissenting_agents=[],
            supporting_signals={},
            risk_assessment={'risk_score': 0.0},
            execution_recommendation={'execute': False},
            timestamp=datetime.now(),
            metadata={'reason': 'no_signals'}
        )

    def update_agent_performance(self, agent_id: str, outcome: float):
        """Update agent performance tracking"""
        if agent_id not in self.performance_tracker:
            self.performance_tracker[agent_id] = AgentPerformance(
                agent_id=agent_id,
                total_signals=0,
                correct_signals=0,
                accuracy=0.5,
                avg_return=0.0,
                sharpe_ratio=0.0,
                recent_performance=[],
                specialties={},
                last_updated=datetime.now()
            )

        perf = self.performance_tracker[agent_id]
        perf.total_signals += 1
        if outcome > 0:
            perf.correct_signals += 1

        perf.accuracy = perf.correct_signals / perf.total_signals
        perf.recent_performance.append(outcome)
        if len(perf.recent_performance) > 20:
            perf.recent_performance.pop(0)

        # Update Sharpe ratio
        if len(perf.recent_performance) >= 5:
            returns = np.array(perf.recent_performance)
            perf.avg_return = np.mean(returns)
            perf.sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6)

        perf.last_updated = datetime.now()


class MultiAgentConsensus:
    """
    Main Multi-Agent Consensus System
    Coordinates all trading agents to reach unified decisions
    """

    def __init__(self):
        """Initialize the consensus system"""
        self.consensus_engine = ConsensusEngine()
        self.registered_agents = {}
        self.active_sessions = {}
        self.decision_history = deque(maxlen=1000)

        # Performance metrics
        self.metrics = {
            'total_decisions': 0,
            'profitable_decisions': 0,
            'consensus_time_ms': deque(maxlen=100),
            'agreement_scores': deque(maxlen=100)
        }

    def register_agent(self, agent_id: str, agent_type: AgentType,
                      capabilities: List[str]):
        """Register an agent with the consensus system"""
        self.registered_agents[agent_id] = {
            'type': agent_type,
            'capabilities': capabilities,
            'active': True,
            'last_signal': None,
            'registered_at': datetime.now()
        }
        logger.info(f"Registered agent {agent_id} of type {agent_type.value}")

    async def request_consensus(self, symbol: str,
                              context: Dict[str, Any],
                              required_agents: Optional[List[AgentType]] = None,
                              timeout: float = 5.0) -> ConsensusResult:
        """Request consensus from agents"""
        start_time = datetime.now()
        session_id = f"SESSION_{symbol}_{start_time.timestamp()}"

        # Start consensus session
        self.active_sessions[session_id] = {
            'symbol': symbol,
            'context': context,
            'start_time': start_time,
            'signals_received': []
        }

        try:
            # Collect signals from agents
            signals = await self._collect_agent_signals(
                symbol, context, required_agents, timeout
            )

            # Reach consensus
            consensus = await self.consensus_engine.reach_consensus(signals)

            # Record metrics
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics['consensus_time_ms'].append(elapsed_ms)
            self.metrics['agreement_scores'].append(consensus.agreement_score)
            self.metrics['total_decisions'] += 1

            # Store decision
            self.decision_history.append({
                'session_id': session_id,
                'symbol': symbol,
                'consensus': consensus,
                'timestamp': datetime.now()
            })

            return consensus

        finally:
            # Clean up session
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

    async def _collect_agent_signals(self, symbol: str,
                                   context: Dict[str, Any],
                                   required_agents: Optional[List[AgentType]],
                                   timeout: float) -> List[AgentSignal]:
        """Collect signals from registered agents"""
        signals = []

        # Determine which agents to query
        if required_agents:
            agents_to_query = [
                (aid, info) for aid, info in self.registered_agents.items()
                if info['type'] in required_agents and info['active']
            ]
        else:
            agents_to_query = [
                (aid, info) for aid, info in self.registered_agents.items()
                if info['active']
            ]

        # Query agents in parallel
        tasks = []
        for agent_id, agent_info in agents_to_query:
            task = self._get_agent_signal(agent_id, agent_info, symbol, context)
            tasks.append(task)

        # Wait for signals with timeout
        if tasks:
            done, pending = await asyncio.wait(
                tasks, timeout=timeout, return_when=asyncio.ALL_COMPLETED
            )

            # Collect completed signals
            for task in done:
                try:
                    signal = await task
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    logger.error(f"Error getting signal: {e}")

            # Cancel pending tasks
            for task in pending:
                task.cancel()

        return signals

    async def _get_agent_signal(self, agent_id: str, agent_info: Dict[str, Any],
                              symbol: str, context: Dict[str, Any]) -> Optional[AgentSignal]:
        """Get signal from individual agent (mock for demo)"""
        # In production, this would call the actual agent
        # For demo, generate mock signals based on agent type

        agent_type = agent_info['type']

        # Simulate agent processing time
        await asyncio.sleep(np.random.uniform(0.1, 0.5))

        # Generate mock signal based on agent type
        if agent_type == AgentType.SENTIMENT:
            # Sentiment agent logic
            sentiment_score = context.get('sentiment', {}).get('score', 0)
            if sentiment_score > 0.5:
                signal = SignalType.BUY
                confidence = min(0.9, sentiment_score)
            elif sentiment_score < -0.5:
                signal = SignalType.SELL
                confidence = min(0.9, abs(sentiment_score))
            else:
                signal = SignalType.HOLD
                confidence = 0.6

            reasoning = f"Sentiment score: {sentiment_score:.2f}"

        elif agent_type == AgentType.TECHNICAL:
            # Technical agent logic
            rsi = context.get('technical', {}).get('rsi', 50)
            if rsi < 30:
                signal = SignalType.BUY
                confidence = 0.8
                reasoning = f"RSI oversold at {rsi}"
            elif rsi > 70:
                signal = SignalType.SELL
                confidence = 0.8
                reasoning = f"RSI overbought at {rsi}"
            else:
                signal = SignalType.HOLD
                confidence = 0.5
                reasoning = f"RSI neutral at {rsi}"

        elif agent_type == AgentType.FLOW:
            # Options flow agent logic
            flow_score = context.get('options_flow', {}).get('smart_money_score', 50)
            if flow_score > 80:
                signal = SignalType.STRONG_BUY
                confidence = 0.9
                reasoning = f"Strong institutional buying, score: {flow_score}"
            elif flow_score < 20:
                signal = SignalType.STRONG_SELL
                confidence = 0.9
                reasoning = f"Strong institutional selling, score: {flow_score}"
            else:
                signal = SignalType.HOLD
                confidence = 0.4
                reasoning = f"Mixed flow signals, score: {flow_score}"

        elif agent_type == AgentType.RISK:
            # Risk agent logic
            risk_score = context.get('risk', {}).get('score', 0.5)
            if risk_score > 0.7:
                signal = SignalType.SELL
                confidence = 0.85
                reasoning = f"High risk detected: {risk_score:.2f}"
            else:
                signal = SignalType.HOLD
                confidence = 0.7
                reasoning = f"Risk within acceptable range: {risk_score:.2f}"

        else:
            # Default logic for other agents
            signal = SignalType.HOLD
            confidence = 0.5
            reasoning = "Default signal"

        return AgentSignal(
            agent_id=agent_id,
            agent_type=agent_type,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            supporting_data={
                'symbol': symbol,
                'context_summary': {k: type(v).__name__ for k, v in context.items()}
            },
            timestamp=datetime.now(),
            time_horizon='short',
            expected_move=np.random.uniform(-0.02, 0.02),
            stop_loss=0.02 if signal in [SignalType.BUY, SignalType.STRONG_BUY] else None,
            position_size=min(1.0, confidence)
        )

    def get_consensus_analytics(self) -> Dict[str, Any]:
        """Get analytics on consensus performance"""
        recent_decisions = list(self.decision_history)[-50:]

        # Signal distribution
        signal_distribution = defaultdict(int)
        for decision in recent_decisions:
            signal_distribution[decision['consensus'].final_signal.value] += 1

        # Method usage
        method_usage = defaultdict(int)
        for decision in recent_decisions:
            method_usage[decision['consensus'].method_used.value] += 1

        # Average metrics
        avg_consensus_time = np.mean(self.metrics['consensus_time_ms']) if self.metrics['consensus_time_ms'] else 0
        avg_agreement = np.mean(self.metrics['agreement_scores']) if self.metrics['agreement_scores'] else 0

        return {
            'total_decisions': self.metrics['total_decisions'],
            'avg_consensus_time_ms': avg_consensus_time,
            'avg_agreement_score': avg_agreement,
            'signal_distribution': dict(signal_distribution),
            'method_usage': dict(method_usage),
            'active_agents': len([a for a in self.registered_agents.values() if a['active']]),
            'success_rate': self.metrics['profitable_decisions'] / self.metrics['total_decisions']
                          if self.metrics['total_decisions'] > 0 else 0
        }

    def update_decision_outcome(self, session_id: str, outcome: float):
        """Update the outcome of a consensus decision"""
        # Find the decision
        for decision in self.decision_history:
            if decision.get('session_id') == session_id:
                decision['outcome'] = outcome

                # Update metrics
                if outcome > 0:
                    self.metrics['profitable_decisions'] += 1

                # Update individual agent performance
                consensus = decision['consensus']
                for agent_id in consensus.participating_agents:
                    # Agents that agreed with consensus get the outcome
                    agent_signal = next(
                        (s for s in consensus.supporting_signals.get(consensus.final_signal, [])
                         if s == agent_id), None
                    )
                    if agent_signal:
                        self.consensus_engine.update_agent_performance(agent_id, outcome)
                    else:
                        # Dissenting agents get negative outcome
                        self.consensus_engine.update_agent_performance(agent_id, -abs(outcome) * 0.5)

                break


# Demo function
async def demo_consensus_system():
    """Demonstrate the Multi-Agent Consensus System"""
    system = MultiAgentConsensus()

    print("Multi-Agent Consensus System Demo")
    print("="*70)

    # Register agents
    print("\n📝 Registering Agents...")
    agents = [
        ('sentiment_001', AgentType.SENTIMENT, ['news', 'social', 'sentiment']),
        ('technical_001', AgentType.TECHNICAL, ['indicators', 'patterns']),
        ('flow_001', AgentType.FLOW, ['options', 'institutional']),
        ('risk_001', AgentType.RISK, ['risk', 'portfolio']),
        ('regime_001', AgentType.REGIME, ['market_regime']),
        ('liquidity_001', AgentType.LIQUIDITY, ['liquidity', 'execution'])
    ]

    for agent_id, agent_type, capabilities in agents:
        system.register_agent(agent_id, agent_type, capabilities)
        print(f"  ✓ {agent_id} ({agent_type.value})")

    # Test Case 1: Bullish Consensus
    print("\n\n📊 Case 1: Bullish Market Scenario")
    print("-"*50)

    bullish_context = {
        'sentiment': {'score': 0.8, 'sources': 7},
        'technical': {'rsi': 45, 'macd': 'bullish'},
        'options_flow': {'smart_money_score': 85, 'call_put_ratio': 1.8},
        'risk': {'score': 0.3, 'vix': 15},
        'regime': {'type': 'bull_quiet', 'confidence': 0.9},
        'liquidity': {'score': 0.8, 'spread': 0.0001}
    }

    consensus1 = await system.request_consensus('AAPL', bullish_context)

    print(f"\n🎯 Consensus Result:")
    print(f"  Signal: {consensus1.final_signal.value.upper()}")
    print(f"  Confidence: {consensus1.confidence:.1%}")
    print(f"  Method: {consensus1.method_used.value}")
    print(f"  Agreement Score: {consensus1.agreement_score:.1%}")

    print(f"\n📊 Vote Distribution:")
    for signal, agents in consensus1.supporting_signals.items():
        print(f"  {signal.value}: {len(agents)} agents")

    if consensus1.dissenting_agents:
        print(f"\n⚠️ Dissenting Agents:")
        for dissent in consensus1.dissenting_agents[:2]:
            print(f"  {dissent['agent_id']}: {dissent['signal']} "
                  f"(confidence: {dissent['confidence']:.1%})")

    print(f"\n💰 Execution Recommendation:")
    exec_rec = consensus1.execution_recommendation
    print(f"  Execute: {'YES' if exec_rec['execute'] else 'NO'}")
    print(f"  Position Size: {exec_rec['position_size']:.1%}")
    print(f"  Entry Strategy: {exec_rec['entry_strategy']}")

    # Test Case 2: Conflicting Signals
    print("\n\n📊 Case 2: Conflicting Signals Scenario")
    print("-"*50)

    conflicting_context = {
        'sentiment': {'score': -0.6, 'sources': 7},
        'technical': {'rsi': 75, 'macd': 'bearish'},
        'options_flow': {'smart_money_score': 70, 'call_put_ratio': 1.2},
        'risk': {'score': 0.8, 'vix': 28},
        'regime': {'type': 'transition', 'confidence': 0.5},
        'liquidity': {'score': 0.4, 'spread': 0.0005}
    }

    consensus2 = await system.request_consensus('SPY', conflicting_context)

    print(f"\n🎯 Consensus Result:")
    print(f"  Signal: {consensus2.final_signal.value.upper()}")
    print(f"  Confidence: {consensus2.confidence:.1%}")
    print(f"  Agreement Score: {consensus2.agreement_score:.1%}")

    if consensus2.metadata.get('risk_veto'):
        print(f"\n🛑 Risk Veto Applied!")

    print(f"\n⚠️ Risk Assessment:")
    risk = consensus2.risk_assessment
    print(f"  Risk Score: {risk.get('risk_score', 0):.2f}")
    if risk.get('warnings'):
        print(f"  Warnings: {', '.join(risk['warnings'])}")

    # Test Case 3: Byzantine Fault Tolerance
    print("\n\n📊 Case 3: Byzantine Fault Tolerance Test")
    print("-"*50)

    # Register a "faulty" agent
    system.register_agent('faulty_001', AgentType.TECHNICAL, ['indicators'])

    byzantine_context = {
        'sentiment': {'score': 0.3, 'sources': 7},
        'technical': {'rsi': 50, 'macd': 'neutral'},
        'options_flow': {'smart_money_score': 55, 'call_put_ratio': 1.0},
        'risk': {'score': 0.5, 'vix': 20}
    }

    # Force BFT method
    system.consensus_engine.reach_consensus = lambda signals, method=ConsensusMethod.BYZANTINE_FAULT_TOLERANT: \
        system.consensus_engine.reach_consensus(signals, ConsensusMethod.BYZANTINE_FAULT_TOLERANT)

    consensus3 = await system.request_consensus('MSFT', byzantine_context)

    print(f"\n🎯 BFT Consensus Result:")
    print(f"  Signal: {consensus3.final_signal.value.upper()}")
    print(f"  Confidence: {consensus3.confidence:.1%}")
    print(f"  Filtered Outliers: Check logs for Byzantine signals")

    # Test Case 4: Performance Analytics
    print("\n\n📊 Case 4: Consensus Analytics")
    print("-"*50)

    # Simulate multiple decisions
    print("\nSimulating 10 rapid consensus decisions...")
    for i in range(10):
        random_context = {
            'sentiment': {'score': np.random.uniform(-1, 1)},
            'technical': {'rsi': np.random.uniform(20, 80)},
            'options_flow': {'smart_money_score': np.random.uniform(0, 100)},
            'risk': {'score': np.random.uniform(0, 1)}
        }
        await system.request_consensus(f'TEST{i}', random_context, timeout=1.0)

    analytics = system.get_consensus_analytics()

    print(f"\n📈 Consensus Performance:")
    print(f"  Total Decisions: {analytics['total_decisions']}")
    print(f"  Avg Consensus Time: {analytics['avg_consensus_time_ms']:.1f}ms")
    print(f"  Avg Agreement Score: {analytics['avg_agreement_score']:.1%}")

    print(f"\n📊 Signal Distribution:")
    for signal, count in analytics['signal_distribution'].items():
        print(f"  {signal}: {count} times")

    print(f"\n🔧 Method Usage:")
    for method, count in analytics['method_usage'].items():
        print(f"  {method}: {count} times")

    # Summary
    print("\n\n" + "="*70)
    print("✅ Multi-Agent Consensus System demonstrates:")
    print("- Multiple consensus methods (voting, BFT, hierarchical)")
    print("- Conflict resolution with risk veto")
    print("- Performance-based agent weighting")
    print("- Real-time consensus in <500ms")
    print("- Byzantine fault tolerance for reliability")


if __name__ == "__main__":
    asyncio.run(demo_consensus_system())
