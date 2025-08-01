#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Multi-Agent Consensus System
Issue #210: Agent System Unit Testing - Consensus Engine
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

# Import the consensus system
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.multi_agent_consensus import (
    ConsensusMethod, SignalType, AgentType, AgentSignal,
    ConsensusResult, AgentPerformance, ConsensusEngine,
    MultiAgentConsensus
)


class TestConsensusEngine:
    """Test ConsensusEngine class"""

    @pytest.fixture
    def consensus_engine(self):
        """Create a consensus engine instance"""
        return ConsensusEngine()

    @pytest.fixture
    def sample_signals(self):
        """Create sample agent signals for testing"""
        return [
            AgentSignal(
                agent_id="sentiment_001",
                agent_type=AgentType.SENTIMENT,
                signal=SignalType.BUY,
                confidence=0.8,
                reasoning="Positive sentiment",
                supporting_data={"score": 0.8},
                timestamp=datetime.now(),
                time_horizon="short"
            ),
            AgentSignal(
                agent_id="technical_001",
                agent_type=AgentType.TECHNICAL,
                signal=SignalType.BUY,
                confidence=0.7,
                reasoning="RSI oversold",
                supporting_data={"rsi": 25},
                timestamp=datetime.now(),
                time_horizon="short"
            ),
            AgentSignal(
                agent_id="flow_001",
                agent_type=AgentType.FLOW,
                signal=SignalType.STRONG_BUY,
                confidence=0.9,
                reasoning="Heavy call buying",
                supporting_data={"flow_score": 85},
                timestamp=datetime.now(),
                time_horizon="short"
            ),
            AgentSignal(
                agent_id="risk_001",
                agent_type=AgentType.RISK,
                signal=SignalType.HOLD,
                confidence=0.6,
                reasoning="Moderate risk",
                supporting_data={"risk_score": 0.5},
                timestamp=datetime.now(),
                time_horizon="short"
            )
        ]

    def test_initialization(self, consensus_engine):
        """Test consensus engine initialization"""
        assert consensus_engine.voting_weights is not None
        assert len(consensus_engine.voting_weights) > 0
        assert consensus_engine.performance_tracker == {}
        assert len(consensus_engine.consensus_history) == 0
        assert consensus_engine.conflict_resolution_rules is not None

    @pytest.mark.asyncio
    async def test_weighted_voting_consensus(self, consensus_engine, sample_signals):
        """Test weighted voting consensus method"""
        result = await consensus_engine._weighted_voting_consensus(sample_signals)

        assert isinstance(result, ConsensusResult)
        assert result.final_signal in [s.signal for s in sample_signals]
        assert 0 <= result.confidence <= 1
        assert result.method_used == ConsensusMethod.WEIGHTED_VOTING
        assert len(result.participating_agents) == len(sample_signals)

    @pytest.mark.asyncio
    async def test_confidence_weighted_consensus(self, consensus_engine, sample_signals):
        """Test confidence weighted consensus method"""
        result = await consensus_engine._confidence_weighted_consensus(sample_signals)

        assert isinstance(result, ConsensusResult)
        assert result.method_used == ConsensusMethod.CONFIDENCE_WEIGHTED
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_performance_weighted_consensus(self, consensus_engine, sample_signals):
        """Test performance weighted consensus with no history"""
        result = await consensus_engine._performance_weighted_consensus(sample_signals)

        assert isinstance(result, ConsensusResult)
        assert result.method_used == ConsensusMethod.PERFORMANCE_WEIGHTED
        # With no history, should use default weights
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_bft_consensus(self, consensus_engine, sample_signals):
        """Test Byzantine Fault Tolerant consensus"""
        # Add a byzantine signal
        byzantine_signal = AgentSignal(
            agent_id="byzantine_001",
            agent_type=AgentType.TECHNICAL,
            signal=SignalType.STRONG_SELL,
            confidence=0.99,  # Extreme confidence
            reasoning="Malicious signal",
            supporting_data={},
            timestamp=datetime.now(),
            time_horizon="short"
        )
        signals_with_byzantine = sample_signals + [byzantine_signal]

        result = await consensus_engine._bft_consensus(signals_with_byzantine)

        assert isinstance(result, ConsensusResult)
        assert result.method_used == ConsensusMethod.BYZANTINE_FAULT_TOLERANT
        # Byzantine signal should be filtered out or have reduced impact
        assert result.final_signal != SignalType.STRONG_SELL

    @pytest.mark.asyncio
    async def test_hierarchical_consensus(self, consensus_engine, sample_signals):
        """Test hierarchical consensus method"""
        result = await consensus_engine._hierarchical_consensus(sample_signals)

        assert isinstance(result, ConsensusResult)
        assert result.method_used == ConsensusMethod.HIERARCHICAL
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_ensemble_consensus(self, consensus_engine, sample_signals):
        """Test ensemble consensus method"""
        result = await consensus_engine._ensemble_consensus(sample_signals)

        assert isinstance(result, ConsensusResult)
        assert result.method_used == ConsensusMethod.ENSEMBLE
        # Ensemble should combine multiple methods
        assert result.confidence > 0

    def test_risk_veto(self, consensus_engine):
        """Test risk agent veto functionality"""
        # Create conflicting signals
        signals = [
            AgentSignal(
                agent_id="technical_001",
                agent_type=AgentType.TECHNICAL,
                signal=SignalType.STRONG_BUY,
                confidence=0.9,
                reasoning="Strong buy signal",
                supporting_data={},
                timestamp=datetime.now(),
                time_horizon="short"
            ),
            AgentSignal(
                agent_id="risk_001",
                agent_type=AgentType.RISK,
                signal=SignalType.STRONG_SELL,
                confidence=0.95,
                reasoning="High risk detected",
                supporting_data={},
                timestamp=datetime.now(),
                time_horizon="short"
            )
        ]

        # Create initial bullish result
        initial_result = ConsensusResult(
            consensus_id="test_001",
            final_signal=SignalType.STRONG_BUY,
            confidence=0.9,
            method_used=ConsensusMethod.WEIGHTED_VOTING,
            participating_agents=["technical_001", "risk_001"],
            agreement_score=0.5,
            dissenting_agents=[],
            supporting_signals={},
            risk_assessment={},
            execution_recommendation={},
            timestamp=datetime.now(),
            metadata={}
        )

        # Apply conflict resolution
        resolved_result = consensus_engine._resolve_conflicts(initial_result, signals)

        # Risk veto should downgrade the signal
        assert resolved_result.final_signal == SignalType.HOLD
        assert resolved_result.confidence <= initial_result.confidence
        assert resolved_result.metadata.get('risk_veto') is True

    def test_agreement_score_calculation(self, consensus_engine):
        """Test agreement score calculation"""
        signals = [
            AgentSignal(
                agent_id=f"agent_{i}",
                agent_type=AgentType.TECHNICAL,
                signal=SignalType.BUY if i < 3 else SignalType.SELL,
                confidence=0.8,
                reasoning="Test",
                supporting_data={},
                timestamp=datetime.now(),
                time_horizon="short"
            )
            for i in range(5)
        ]

        score = consensus_engine._calculate_agreement_score(signals, SignalType.BUY)
        assert score == 3/5  # 3 out of 5 agents agree

    @pytest.mark.asyncio
    async def test_empty_signals(self, consensus_engine):
        """Test consensus with no signals"""
        result = await consensus_engine.reach_consensus([])

        assert result.final_signal == SignalType.NO_SIGNAL
        assert result.confidence == 0.0
        assert len(result.participating_agents) == 0

    def test_update_agent_performance(self, consensus_engine):
        """Test agent performance tracking"""
        agent_id = "test_agent_001"

        # First update
        consensus_engine.update_agent_performance(agent_id, 0.05)

        assert agent_id in consensus_engine.performance_tracker
        perf = consensus_engine.performance_tracker[agent_id]
        assert perf.total_signals == 1
        assert perf.correct_signals == 1
        assert perf.accuracy == 1.0

        # Second update with loss
        consensus_engine.update_agent_performance(agent_id, -0.03)

        perf = consensus_engine.performance_tracker[agent_id]
        assert perf.total_signals == 2
        assert perf.correct_signals == 1
        assert perf.accuracy == 0.5


class TestMultiAgentConsensus:
    """Test MultiAgentConsensus system"""

    @pytest.fixture
    def consensus_system(self):
        """Create a multi-agent consensus system"""
        return MultiAgentConsensus()

    def test_agent_registration(self, consensus_system):
        """Test agent registration"""
        agent_id = "test_agent_001"
        agent_type = AgentType.SENTIMENT
        capabilities = ["news", "social"]

        consensus_system.register_agent(agent_id, agent_type, capabilities)

        assert agent_id in consensus_system.registered_agents
        agent_info = consensus_system.registered_agents[agent_id]
        assert agent_info['type'] == agent_type
        assert agent_info['capabilities'] == capabilities
        assert agent_info['active'] is True

    @pytest.mark.asyncio
    async def test_request_consensus_with_timeout(self, consensus_system):
        """Test consensus request with timeout"""
        # Register a slow agent
        consensus_system.register_agent("slow_agent", AgentType.TECHNICAL, ["slow"])

        # Mock the agent signal collection to simulate timeout
        async def slow_signal_collection(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout
            return []

        with patch.object(consensus_system, '_collect_agent_signals', slow_signal_collection):
            result = await consensus_system.request_consensus(
                "AAPL", {}, timeout=0.1
            )

            # Should still return a result even with timeout
            assert isinstance(result, ConsensusResult)
            assert result.final_signal == SignalType.NO_SIGNAL

    @pytest.mark.asyncio
    async def test_full_consensus_flow(self, consensus_system):
        """Test full consensus flow with multiple agents"""
        # Register agents
        agents = [
            ("sentiment_001", AgentType.SENTIMENT, ["news"]),
            ("technical_001", AgentType.TECHNICAL, ["indicators"]),
            ("flow_001", AgentType.FLOW, ["options"]),
            ("risk_001", AgentType.RISK, ["risk"])
        ]

        for agent_id, agent_type, capabilities in agents:
            consensus_system.register_agent(agent_id, agent_type, capabilities)

        # Mock agent signals
        mock_signals = [
            AgentSignal(
                agent_id=agent_id,
                agent_type=agent_type,
                signal=SignalType.BUY,
                confidence=0.8,
                reasoning="Test reasoning",
                supporting_data={},
                timestamp=datetime.now(),
                time_horizon="short"
            )
            for agent_id, agent_type, _ in agents
        ]

        with patch.object(consensus_system, '_collect_agent_signals',
                         new_callable=AsyncMock, return_value=mock_signals):
            result = await consensus_system.request_consensus(
                "AAPL", {"test": "context"}
            )

            assert isinstance(result, ConsensusResult)
            assert result.final_signal != SignalType.NO_SIGNAL
            assert len(result.participating_agents) == len(agents)
            assert result.agreement_score > 0

    def test_get_consensus_analytics(self, consensus_system):
        """Test consensus analytics generation"""
        # Add some mock decisions to history
        for i in range(5):
            consensus_system.decision_history.append({
                'session_id': f'session_{i}',
                'symbol': 'AAPL',
                'consensus': ConsensusResult(
                    consensus_id=f'consensus_{i}',
                    final_signal=SignalType.BUY if i % 2 == 0 else SignalType.SELL,
                    confidence=0.8,
                    method_used=ConsensusMethod.WEIGHTED_VOTING,
                    participating_agents=['agent_1', 'agent_2'],
                    agreement_score=0.8,
                    dissenting_agents=[],
                    supporting_signals={},
                    risk_assessment={},
                    execution_recommendation={},
                    timestamp=datetime.now(),
                    metadata={}
                ),
                'timestamp': datetime.now()
            })

        consensus_system.metrics['total_decisions'] = 5
        consensus_system.metrics['profitable_decisions'] = 3

        analytics = consensus_system.get_consensus_analytics()

        assert analytics['total_decisions'] == 5
        assert analytics['success_rate'] == 0.6
        assert 'signal_distribution' in analytics
        assert 'method_usage' in analytics

    def test_update_decision_outcome(self, consensus_system):
        """Test updating decision outcomes"""
        session_id = "test_session_001"

        # Add a decision
        consensus_system.decision_history.append({
            'session_id': session_id,
            'consensus': ConsensusResult(
                consensus_id='test_consensus',
                final_signal=SignalType.BUY,
                confidence=0.8,
                method_used=ConsensusMethod.WEIGHTED_VOTING,
                participating_agents=['agent_1', 'agent_2'],
                agreement_score=1.0,
                dissenting_agents=[],
                supporting_signals={SignalType.BUY: ['agent_1', 'agent_2']},
                risk_assessment={},
                execution_recommendation={},
                timestamp=datetime.now(),
                metadata={}
            )
        })

        # Update with positive outcome
        consensus_system.update_decision_outcome(session_id, 0.05)

        assert consensus_system.metrics['profitable_decisions'] == 1

        # Verify the decision was updated
        decision = next(d for d in consensus_system.decision_history
                       if d.get('session_id') == session_id)
        assert decision['outcome'] == 0.05


class TestEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.fixture
    def consensus_engine(self):
        return ConsensusEngine()

    @pytest.mark.asyncio
    async def test_all_agents_disagree(self, consensus_engine):
        """Test when all agents have different signals"""
        signals = [
            AgentSignal(
                agent_id=f"agent_{i}",
                agent_type=list(AgentType)[i % len(AgentType)],
                signal=list(SignalType)[i % len(SignalType)],
                confidence=0.7,
                reasoning="Test",
                supporting_data={},
                timestamp=datetime.now(),
                time_horizon="short"
            )
            for i in range(5)
        ]

        result = await consensus_engine.reach_consensus(signals)

        assert isinstance(result, ConsensusResult)
        assert result.agreement_score < 0.5  # Low agreement

    @pytest.mark.asyncio
    async def test_single_agent_consensus(self, consensus_engine):
        """Test consensus with only one agent"""
        signal = AgentSignal(
            agent_id="single_agent",
            agent_type=AgentType.TECHNICAL,
            signal=SignalType.BUY,
            confidence=0.9,
            reasoning="Single agent test",
            supporting_data={},
            timestamp=datetime.now(),
            time_horizon="short"
        )

        result = await consensus_engine.reach_consensus([signal])

        assert result.final_signal == SignalType.BUY
        assert result.confidence > 0
        assert result.agreement_score == 1.0  # 100% agreement with self

    @pytest.mark.asyncio
    async def test_high_confidence_minority(self, consensus_engine):
        """Test when minority has very high confidence"""
        signals = [
            # Majority with low confidence
            AgentSignal(
                agent_id=f"majority_{i}",
                agent_type=AgentType.TECHNICAL,
                signal=SignalType.SELL,
                confidence=0.3,
                reasoning="Low confidence sell",
                supporting_data={},
                timestamp=datetime.now(),
                time_horizon="short"
            )
            for i in range(3)
        ] + [
            # Minority with high confidence
            AgentSignal(
                agent_id="minority_1",
                agent_type=AgentType.FLOW,
                signal=SignalType.STRONG_BUY,
                confidence=0.95,
                reasoning="High confidence buy",
                supporting_data={},
                timestamp=datetime.now(),
                time_horizon="short"
            )
        ]

        result = await consensus_engine._confidence_weighted_consensus(signals)

        # High confidence minority might win in confidence-weighted
        assert isinstance(result, ConsensusResult)
        assert result.confidence > 0


def test_agent_types_coverage():
    """Test that all agent types are covered"""
    agent_types = list(AgentType)
    assert len(agent_types) >= 5  # At least 5 agent types
    assert AgentType.SENTIMENT in agent_types
    assert AgentType.TECHNICAL in agent_types
    assert AgentType.FLOW in agent_types
    assert AgentType.RISK in agent_types


def test_signal_types_coverage():
    """Test that all signal types are covered"""
    signal_types = list(SignalType)
    assert SignalType.STRONG_BUY in signal_types
    assert SignalType.BUY in signal_types
    assert SignalType.HOLD in signal_types
    assert SignalType.SELL in signal_types
    assert SignalType.STRONG_SELL in signal_types
    assert SignalType.NO_SIGNAL in signal_types


def test_consensus_methods_coverage():
    """Test that all consensus methods are covered"""
    methods = list(ConsensusMethod)
    assert ConsensusMethod.WEIGHTED_VOTING in methods
    assert ConsensusMethod.CONFIDENCE_WEIGHTED in methods
    assert ConsensusMethod.PERFORMANCE_WEIGHTED in methods
    assert ConsensusMethod.BYZANTINE_FAULT_TOLERANT in methods
    assert ConsensusMethod.HIERARCHICAL in methods
    assert ConsensusMethod.ENSEMBLE in methods
    assert ConsensusMethod.ADAPTIVE in methods


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--cov=agents.multi_agent_consensus", "--cov-report=html"])
