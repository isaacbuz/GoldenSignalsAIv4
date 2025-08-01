"""
Tests for agent functionality in GoldenSignalsAI V2.
"""
import pytest
from typing import Dict, Any
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agents.common.base.base_agent import BaseAgent, RiskAverseAgent, AggressiveAgent


class TestBaseAgentImplementations:
    """Test base agent implementations"""

    def test_risk_averse_agent(self):
        """Test RiskAverseAgent functionality"""
        agent = RiskAverseAgent()

        # Test with sample signal
        signal = {
            "symbol": "SPY",
            "action": "buy",
            "confidence": 0.8,
            "metadata": {"source": "technical"}
        }

        result = agent.process_signal(signal)

        # Verify signal is modified
        assert "risk_adjusted" in result
        assert result["risk_adjusted"] is True

        # Verify original signal data is preserved
        assert result["symbol"] == "SPY"
        assert result["action"] == "buy"
        assert result["confidence"] == 0.8

    def test_aggressive_agent(self):
        """Test AggressiveAgent functionality"""
        agent = AggressiveAgent()

        # Test with sample signal
        signal = {
            "symbol": "AAPL",
            "action": "sell",
            "confidence": 0.6,
            "metadata": {"source": "sentiment"}
        }

        result = agent.process_signal(signal)

        # Verify signal is modified
        assert "aggressive_mode" in result
        assert result["aggressive_mode"] is True

        # Verify original signal data is preserved
        assert result["symbol"] == "AAPL"
        assert result["action"] == "sell"
        assert result["confidence"] == 0.6

    def test_custom_agent_implementation(self):
        """Test custom agent implementation"""

        class ConservativeAgent(BaseAgent):
            """Custom conservative agent"""

            def process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
                """Apply conservative modifications"""
                signal = signal.copy()

                # Reduce confidence for conservative approach
                signal["confidence"] = signal.get("confidence", 1.0) * 0.7
                signal["conservative"] = True

                # Change action to hold if confidence is too low
                if signal["confidence"] < 0.5:
                    signal["action"] = "hold"

                return signal

        agent = ConservativeAgent()

        # Test with high confidence signal
        high_conf_signal = {
            "symbol": "MSFT",
            "action": "buy",
            "confidence": 0.9
        }

        result = agent.process_signal(high_conf_signal)
        assert result["confidence"] == 0.9 * 0.7  # 0.63
        assert result["action"] == "buy"  # Still buy since confidence > 0.5
        assert result["conservative"] is True

        # Test with low confidence signal
        low_conf_signal = {
            "symbol": "TSLA",
            "action": "sell",
            "confidence": 0.6
        }

        result = agent.process_signal(low_conf_signal)
        assert result["confidence"] == 0.6 * 0.7  # 0.42
        assert result["action"] == "hold"  # Changed to hold since confidence < 0.5
        assert result["conservative"] is True

    def test_agent_inheritance(self):
        """Test that agents properly inherit from BaseAgent"""
        assert issubclass(RiskAverseAgent, BaseAgent)
        assert issubclass(AggressiveAgent, BaseAgent)

        # Verify they have the required method
        assert hasattr(RiskAverseAgent, 'process_signal')
        assert hasattr(AggressiveAgent, 'process_signal')

    def test_signal_immutability(self):
        """Test that agents don't modify the original signal"""
        agent = RiskAverseAgent()

        original_signal = {
            "symbol": "GOOGL",
            "action": "buy",
            "confidence": 0.75
        }

        # Create a copy to test
        signal_copy = original_signal.copy()

        # Process the signal
        result = agent.process_signal(signal_copy)

        # Original signal should remain unchanged
        assert original_signal == {
            "symbol": "GOOGL",
            "action": "buy",
            "confidence": 0.75
        }
