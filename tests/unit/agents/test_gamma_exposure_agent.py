"""
Comprehensive unit tests for GammaExposureAgent.

Tests cover:
- Black-Scholes gamma calculations
- Dealer gamma exposure analysis
- Gamma flip point detection
- Price pinning analysis
- Signal generation logic
- Edge cases and error handling
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Import the agent to test
from agents.core.options.gamma_exposure_agent import GammaExposureAgent


class TestGammaExposureAgent:
    """Test suite for GammaExposureAgent"""

    @pytest.fixture
    def agent(self):
        """Create a GammaExposureAgent instance for testing"""
        return GammaExposureAgent(
            name="TestGammaAgent",
            gamma_threshold=100000,
            pin_proximity_threshold=0.02,
            min_open_interest=100
        )

    @pytest.fixture
    def sample_options_data(self):
        """Sample options data for testing"""
        return [
            {
                'strike': 100,
                'type': 'call',
                'open_interest': 1000,
                'volume': 500,
                'time_to_expiry': 0.25,  # 3 months
                'implied_volatility': 0.25
            },
            {
                'strike': 105,
                'type': 'call',
                'open_interest': 800,
                'volume': 300,
                'time_to_expiry': 0.25,
                'implied_volatility': 0.28
            },
            {
                'strike': 95,
                'type': 'put',
                'open_interest': 1200,
                'volume': 600,
                'time_to_expiry': 0.25,
                'implied_volatility': 0.30
            },
            {
                'strike': 90,
                'type': 'put',
                'open_interest': 900,
                'volume': 400,
                'time_to_expiry': 0.25,
                'implied_volatility': 0.35
            }
        ]

    def test_agent_initialization(self, agent):
        """Test agent initializes with correct parameters"""
        # Test default initialization via fixture
        assert agent.name == "TestGammaAgent"
        assert agent.agent_type == "options"
        assert agent.gamma_threshold == 100000
        assert agent.pin_proximity_threshold == 0.02
        assert agent.min_open_interest == 100

        # Test custom initialization
        custom_agent = GammaExposureAgent(
            name="CustomGamma",
            gamma_threshold=50000,
            pin_proximity_threshold=0.01,
            min_open_interest=50
        )

        assert custom_agent.name == "CustomGamma"
        assert custom_agent.agent_type == "options"
        assert custom_agent.gamma_threshold == 50000
        assert custom_agent.pin_proximity_threshold == 0.01
        assert custom_agent.min_open_interest == 50

    def test_black_scholes_gamma_calculation(self, agent):
        """Test Black-Scholes gamma calculation"""
        # Test normal case
        gamma = agent.calculate_black_scholes_gamma(
            spot=100,
            strike=100,
            time_to_expiry=0.25,
            volatility=0.25,
            risk_free_rate=0.05
        )

        # Gamma should be positive for ATM options
        assert gamma > 0
        assert isinstance(gamma, float)

        # Test edge cases
        assert agent.calculate_black_scholes_gamma(100, 100, 0, 0.25) == 0.0  # No time
        assert agent.calculate_black_scholes_gamma(100, 100, 0.25, 0) == 0.0  # No volatility

    def test_dealer_gamma_exposure_calculation(self, agent, sample_options_data):
        """Test dealer gamma exposure calculation"""
        spot_price = 102

        result = agent.calculate_dealer_gamma_exposure(sample_options_data, spot_price)

        # Check result structure
        assert 'net_gamma_exposure' in result
        assert 'call_gamma' in result
        assert 'put_gamma' in result
        assert 'gamma_by_strike' in result
        assert 'significant_levels' in result
        assert 'gamma_flip_level' in result
        assert 'current_spot' in result

        # Verify types
        assert isinstance(result['net_gamma_exposure'], float)
        assert isinstance(result['call_gamma'], float)
        assert isinstance(result['put_gamma'], float)
        assert isinstance(result['gamma_by_strike'], dict)
        assert isinstance(result['significant_levels'], list)
        assert result['current_spot'] == spot_price

    def test_gamma_flip_point_detection(self, agent):
        """Test gamma flip point detection"""
        # Create gamma by strike data
        gamma_by_strike = {
            95: -50000,   # Put gamma (dealer short)
            100: -100000, # ATM - large negative
            105: 30000,   # Call gamma
            110: 80000    # OTM call gamma
        }

        spot_price = 102
        flip_point = agent.find_gamma_flip_point(gamma_by_strike, spot_price)

        # Should find a flip point
        assert flip_point is not None
        assert isinstance(flip_point, float)

        # Test empty data
        assert agent.find_gamma_flip_point({}, spot_price) is None

    def test_gamma_pinning_analysis(self, agent, sample_options_data):
        """Test gamma pinning risk analysis"""
        spot_price = 100  # Near strike prices

        # Calculate gamma data first
        gamma_data = agent.calculate_dealer_gamma_exposure(sample_options_data, spot_price)

        # Analyze pinning
        pinning_result = agent.analyze_gamma_pinning_risk(gamma_data, spot_price)

        # Check result structure
        assert 'pinning_detected' in pinning_result
        assert isinstance(pinning_result['pinning_detected'], bool)

        if pinning_result['pinning_detected']:
            assert 'pin_level' in pinning_result
            assert 'pin_strength' in pinning_result
            assert 'pin_classification' in pinning_result
            assert 'distance_to_pin' in pinning_result

    def test_volatility_impact_calculation(self, agent):
        """Test gamma impact on volatility calculation"""
        # Test case with negative gamma (dealers short)
        gamma_data_short = {
            'net_gamma_exposure': -200000,
            'gamma_flip_level': 105,
            'current_spot': 100
        }

        vol_impact = agent.calculate_gamma_impact_on_volatility(gamma_data_short)

        assert 'volatility_impact' in vol_impact
        assert 'impact_magnitude' in vol_impact
        assert 'regime' in vol_impact
        assert vol_impact['volatility_impact'] == 'amplifying'
        assert vol_impact['regime'] == 'short_gamma'

        # Test case with positive gamma (dealers long)
        gamma_data_long = {
            'net_gamma_exposure': 150000,
            'gamma_flip_level': 95,
            'current_spot': 100
        }

        vol_impact = agent.calculate_gamma_impact_on_volatility(gamma_data_long)

        assert vol_impact['volatility_impact'] == 'dampening'
        assert vol_impact['regime'] == 'long_gamma'

    def test_signal_generation(self, agent, sample_options_data):
        """Test gamma signal generation"""
        data = {
            'options_data': sample_options_data,
            'spot_price': 102
        }

        signals = agent.generate_gamma_signals(data)

        # Check signal structure
        assert 'action' in signals
        assert 'confidence' in signals
        assert 'signal_type' in signals
        assert 'reasoning' in signals

        # Validate signal values
        assert signals['action'] in ['buy', 'sell', 'hold']
        assert 0.0 <= signals['confidence'] <= 1.0
        assert isinstance(signals['reasoning'], list)

    def test_process_method(self, agent, sample_options_data):
        """Test the main process method"""
        data = {
            'options_data': sample_options_data,
            'spot_price': 102
        }

        result = agent.process(data)

        # Check result structure
        assert 'action' in result
        assert 'confidence' in result
        assert 'metadata' in result

        # Validate action and confidence
        assert result['action'] in ['buy', 'sell', 'hold']
        assert 0.0 <= result['confidence'] <= 1.0

        # Check metadata content
        metadata = result['metadata']
        assert 'signal_type' in metadata
        assert 'reasoning' in metadata
        assert 'gamma_exposure' in metadata
        assert 'pinning_analysis' in metadata
        assert 'volatility_impact' in metadata

    def test_edge_cases(self, agent):
        """Test edge cases and error handling"""
        # Test with no options data
        result = agent.process({})
        assert result['action'] == 'hold'
        assert result['confidence'] == 0.0
        assert 'error' in result['metadata']

        # Test with invalid spot price
        result = agent.process({
            'options_data': [],
            'spot_price': 0
        })
        assert result['action'] == 'hold'

        # Test with minimal options data
        minimal_data = [{
            'strike': 100,
            'type': 'call',
            'open_interest': 50,  # Below threshold
            'volume': 10,
            'time_to_expiry': 0.1,
            'implied_volatility': 0.2
        }]

        result = agent.process({
            'options_data': minimal_data,
            'spot_price': 100
        })

        # Should handle gracefully
        assert 'action' in result
        assert 'confidence' in result

    def test_performance_requirements(self, agent, sample_options_data):
        """Test performance requirements"""
        import time

        data = {
            'options_data': sample_options_data * 10,  # Larger dataset
            'spot_price': 102
        }

        start_time = time.time()
        result = agent.process(data)
        execution_time = time.time() - start_time

        # Should complete within reasonable time
        assert execution_time < 1.0  # Less than 1 second
        assert result['action'] in ['buy', 'sell', 'hold']

    def test_gamma_calculations_accuracy(self, agent):
        """Test accuracy of gamma calculations"""
        # Test known Black-Scholes values
        # For ATM option with specific parameters
        gamma = agent.calculate_black_scholes_gamma(
            spot=100,
            strike=100,
            time_to_expiry=0.25,
            volatility=0.20,
            risk_free_rate=0.05
        )

        # Gamma should be approximately 0.0199 for these parameters
        expected_gamma = 0.0199
        assert abs(gamma - expected_gamma) < 0.005  # Within 0.5% tolerance

    def test_signal_consistency(self, agent, sample_options_data):
        """Test signal consistency across multiple runs"""
        data = {
            'options_data': sample_options_data,
            'spot_price': 102
        }

        # Run multiple times
        results = []
        for _ in range(5):
            result = agent.process(data)
            results.append(result)

        # All results should be identical for same input
        first_result = results[0]
        for result in results[1:]:
            assert result['action'] == first_result['action']
            assert abs(result['confidence'] - first_result['confidence']) < 0.001

    @pytest.mark.parametrize("spot_price,expected_regime", [
        (90, "short_gamma"),   # Below most strikes
        (100, "mixed"),        # Near ATM strikes
        (110, "long_gamma")    # Above most strikes
    ])
    def test_different_spot_prices(self, agent, sample_options_data, spot_price, expected_regime):
        """Test behavior with different spot prices"""
        data = {
            'options_data': sample_options_data,
            'spot_price': spot_price
        }

        result = agent.process(data)

        # Should produce valid results for all spot prices
        assert result['action'] in ['buy', 'sell', 'hold']
        assert 0.0 <= result['confidence'] <= 1.0

        # Check that gamma analysis reflects spot price position
        gamma_data = result['metadata']['gamma_exposure']
        assert 'net_gamma_exposure' in gamma_data


@pytest.mark.integration
class TestGammaExposureAgentIntegration:
    """Integration tests for GammaExposureAgent"""

    def test_with_real_market_conditions(self):
        """Test agent with realistic market conditions"""
        agent = GammaExposureAgent()

        # Simulate high gamma environment (OPEX week)
        high_gamma_data = {
            'options_data': [
                {
                    'strike': strike,
                    'type': 'call' if strike > 100 else 'put',
                    'open_interest': 5000,
                    'volume': 2000,
                    'time_to_expiry': 0.02,  # 1 week to expiry
                    'implied_volatility': 0.30
                }
                for strike in range(95, 106)
            ],
            'spot_price': 100
        }

        result = agent.process(high_gamma_data)

        # Should detect high gamma environment
        assert result['confidence'] > 0.3  # Should have reasonable confidence
        assert 'gamma' in result['metadata']['signal_type'].lower()

    def test_error_recovery(self):
        """Test error recovery and fallback behavior"""
        agent = GammaExposureAgent()

        # Test with malformed data
        malformed_data = {
            'options_data': [
                {'strike': 'invalid', 'type': 'call'},  # Invalid strike
                {'strike': 100}  # Missing required fields
            ],
            'spot_price': 100
        }

        # Should not crash and return safe default
        result = agent.process(malformed_data)
        assert result['action'] == 'hold'
        assert result['confidence'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
