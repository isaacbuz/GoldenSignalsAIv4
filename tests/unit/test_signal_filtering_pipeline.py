"""
Tests for Signal Filtering Pipeline
"""

import pytest
from datetime import datetime, timedelta
from typing import List

from src.services.signal_generation_engine import TradingSignal
from src.services.signal_filtering_pipeline import (
    SignalFilteringPipeline,
    ConfidenceFilter,
    QualityScoreFilter,
    RiskFilter,
    VolumeFilter,
    TechnicalConsistencyFilter,
    DuplicateSignalFilter,
    MarketConditionFilter
)


class TestSignalFilters:
    """Test individual signal filters"""

    @pytest.fixture
    def sample_signal(self):
        """Create a sample signal for testing"""
        return TradingSignal(
            id="TEST_001",
            symbol="AAPL",
            action="BUY",
            confidence=0.75,
            price=150.0,
            timestamp=datetime.now(),
            reason="Test signal",
            indicators={
                "rsi": 25,
                "macd": 0.5,
                "macd_signal": 0.3,
                "bb_percent": 0.15,
                "stoch_k": 18,
                "volume_ratio": 2.5
            },
            risk_level="medium",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            quality_score=0.85
        )

    def test_confidence_filter(self, sample_signal):
        """Test confidence filter"""
        # Test pass case
        filter_pass = ConfidenceFilter(min_confidence=0.7)
        assert filter_pass.filter(sample_signal) is True

        # Test fail case
        filter_fail = ConfidenceFilter(min_confidence=0.8)
        assert filter_fail.filter(sample_signal) is False

        # Check stats
        stats = filter_fail.get_stats()
        assert stats['total_signals'] == 1
        assert stats['filtered_signals'] == 1

    def test_quality_score_filter(self, sample_signal):
        """Test quality score filter"""
        # Test pass case
        filter_pass = QualityScoreFilter(min_quality_score=0.8)
        assert filter_pass.filter(sample_signal) is True

        # Test fail case
        filter_fail = QualityScoreFilter(min_quality_score=0.9)
        assert filter_fail.filter(sample_signal) is False

    def test_risk_filter(self, sample_signal):
        """Test risk filter"""
        # Test pass case
        filter_pass = RiskFilter(allowed_risk_levels=["low", "medium"])
        assert filter_pass.filter(sample_signal) is True

        # Test fail case
        filter_fail = RiskFilter(allowed_risk_levels=["low"])
        assert filter_fail.filter(sample_signal) is False

    def test_volume_filter(self, sample_signal):
        """Test volume filter"""
        # Test pass case
        filter_pass = VolumeFilter(min_volume_ratio=2.0)
        assert filter_pass.filter(sample_signal) is True

        # Test fail case
        filter_fail = VolumeFilter(min_volume_ratio=3.0)
        assert filter_fail.filter(sample_signal) is False

    def test_technical_consistency_filter(self, sample_signal):
        """Test technical consistency filter"""
        # The sample signal has RSI oversold, MACD bullish, BB oversold, Stoch oversold
        # All consistent with BUY signal
        filter_obj = TechnicalConsistencyFilter(min_consistent_indicators=2)
        assert filter_obj.filter(sample_signal) is True

        # Test with higher requirement
        filter_strict = TechnicalConsistencyFilter(min_consistent_indicators=5)
        assert filter_strict.filter(sample_signal) is False

    def test_duplicate_signal_filter(self, sample_signal):
        """Test duplicate signal filter"""
        filter_obj = DuplicateSignalFilter(time_window_minutes=30)

        # First signal should pass
        assert filter_obj.filter(sample_signal) is True

        # Duplicate signal should fail
        duplicate = TradingSignal(
            id="TEST_002",
            symbol="AAPL",
            action="BUY",
            confidence=0.75,
            price=150.5,  # Similar price
            timestamp=datetime.now(),
            reason="Test signal",
            indicators=sample_signal.indicators,
            risk_level="medium",
            entry_price=150.5,
            stop_loss=145.5,
            take_profit=160.5,
            quality_score=0.85
        )
        assert filter_obj.filter(duplicate) is False

        # Different action should pass
        sell_signal = TradingSignal(
            id="TEST_003",
            symbol="AAPL",
            action="SELL",
            confidence=0.75,
            price=150.0,
            timestamp=datetime.now(),
            reason="Test signal",
            indicators=sample_signal.indicators,
            risk_level="medium",
            entry_price=150.0,
            stop_loss=155.0,
            take_profit=140.0,
            quality_score=0.85
        )
        assert filter_obj.filter(sell_signal) is True

    def test_market_condition_filter(self, sample_signal):
        """Test market condition filter"""
        filter_obj = MarketConditionFilter()
        # Currently always passes
        assert filter_obj.filter(sample_signal) is True


class TestSignalFilteringPipeline:
    """Test the complete filtering pipeline"""

    @pytest.fixture
    def pipeline(self):
        """Create a signal filtering pipeline"""
        return SignalFilteringPipeline()

    @pytest.fixture
    def sample_signals(self):
        """Create multiple sample signals for testing"""
        signals = []

        # High quality BUY signal
        signals.append(TradingSignal(
            id="HQ_BUY_001",
            symbol="AAPL",
            action="BUY",
            confidence=0.85,
            price=150.0,
            timestamp=datetime.now(),
            reason="Strong buy signal",
            indicators={
                "rsi": 25,
                "macd": 0.5,
                "macd_signal": 0.3,
                "bb_percent": 0.15,
                "stoch_k": 18,
                "volume_ratio": 3.0
            },
            risk_level="low",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            quality_score=0.9
        ))

        # Low confidence signal
        signals.append(TradingSignal(
            id="LC_SELL_001",
            symbol="GOOGL",
            action="SELL",
            confidence=0.3,  # Too low
            price=2800.0,
            timestamp=datetime.now(),
            reason="Weak sell signal",
            indicators={
                "rsi": 75,
                "macd": -0.5,
                "macd_signal": -0.3,
                "bb_percent": 0.85,
                "stoch_k": 82,
                "volume_ratio": 1.5
            },
            risk_level="medium",
            entry_price=2800.0,
            stop_loss=2850.0,
            take_profit=2700.0,
            quality_score=0.8
        ))

        # High risk signal
        signals.append(TradingSignal(
            id="HR_BUY_001",
            symbol="TSLA",
            action="BUY",
            confidence=0.7,
            price=250.0,
            timestamp=datetime.now(),
            reason="Risky buy",
            indicators={
                "rsi": 35,
                "macd": 0.2,
                "macd_signal": 0.1,
                "bb_percent": 0.3,
                "stoch_k": 25,
                "volume_ratio": 0.8  # Too low
            },
            risk_level="high",  # Not allowed by default
            entry_price=250.0,
            stop_loss=240.0,
            take_profit=270.0,
            quality_score=0.75
        ))

        # Low quality signal
        signals.append(TradingSignal(
            id="LQ_SELL_001",
            symbol="MSFT",
            action="SELL",
            confidence=0.6,
            price=380.0,
            timestamp=datetime.now(),
            reason="Low quality data",
            indicators={
                "rsi": 65,
                "macd": -0.1,
                "macd_signal": 0.0,
                "bb_percent": 0.7,
                "stoch_k": 70,
                "volume_ratio": 1.5
            },
            risk_level="medium",
            entry_price=380.0,
            stop_loss=390.0,
            take_profit=360.0,
            quality_score=0.5  # Too low
        ))

        # Inconsistent indicators
        signals.append(TradingSignal(
            id="INC_BUY_001",
            symbol="AMZN",
            action="BUY",
            confidence=0.6,
            price=140.0,
            timestamp=datetime.now(),
            reason="Inconsistent indicators",
            indicators={
                "rsi": 75,  # Overbought - inconsistent with BUY
                "macd": -0.5,  # Bearish - inconsistent with BUY
                "macd_signal": -0.3,
                "bb_percent": 0.9,  # Near upper band - inconsistent with BUY
                "stoch_k": 85,  # Overbought - inconsistent with BUY
                "volume_ratio": 2.0
            },
            risk_level="medium",
            entry_price=140.0,
            stop_loss=135.0,
            take_profit=150.0,
            quality_score=0.8
        ))

        return signals

    def test_pipeline_initialization(self, pipeline):
        """Test pipeline is initialized with default filters"""
        assert len(pipeline.filters) == 7  # 7 default filters

        filter_names = [f.name for f in pipeline.filters]
        expected_filters = [
            "ConfidenceFilter",
            "QualityScoreFilter",
            "RiskFilter",
            "VolumeFilter",
            "TechnicalConsistencyFilter",
            "DuplicateSignalFilter",
            "MarketConditionFilter"
        ]

        for expected in expected_filters:
            assert expected in filter_names

    def test_filter_signals(self, pipeline, sample_signals):
        """Test filtering multiple signals"""
        filtered = pipeline.filter_signals(sample_signals)

        # Only the first signal should pass all filters
        assert len(filtered) == 1
        assert filtered[0].id == "HQ_BUY_001"

        # Check pipeline stats
        stats = pipeline.get_pipeline_stats()
        assert stats['total_signals_processed'] == 5
        assert stats['signals_passed'] == 1
        assert stats['signals_filtered'] == 4

    def test_add_remove_filters(self, pipeline):
        """Test adding and removing filters"""
        initial_count = len(pipeline.filters)

        # Add a new filter
        new_filter = ConfidenceFilter(min_confidence=0.9)
        pipeline.add_filter(new_filter)
        assert len(pipeline.filters) == initial_count + 1

        # Remove a filter
        pipeline.remove_filter("ConfidenceFilter")
        # Should remove all confidence filters
        assert all(f.name != "ConfidenceFilter" for f in pipeline.filters)

    def test_adjust_filter_parameters(self, pipeline):
        """Test dynamic parameter adjustment"""
        # Get initial confidence threshold
        conf_filter = next(f for f in pipeline.filters if isinstance(f, ConfidenceFilter))
        initial_threshold = conf_filter.min_confidence

        # Simulate high false positive rate
        performance_metrics = {
            'false_positive_rate': 0.4,
            'signal_accuracy': 0.3
        }

        pipeline.adjust_filter_parameters(performance_metrics)

        # Confidence threshold should increase
        assert conf_filter.min_confidence > initial_threshold

    def test_create_custom_pipeline(self):
        """Test creating custom pipeline with specific config"""
        config = {
            'confidence_filter': True,
            'min_confidence': 0.8,
            'quality_filter': False,  # Disable quality filter
            'risk_filter': True,
            'allowed_risk_levels': ["low"],
            'volume_filter': True,
            'min_volume_ratio': 2.5,
            'technical_consistency_filter': False,  # Disable
            'duplicate_filter': True,
            'duplicate_window_minutes': 60
        }

        pipeline = SignalFilteringPipeline()
        custom_pipeline = pipeline.create_custom_pipeline(config)

        # Check filters match config
        filter_names = [f.name for f in custom_pipeline.filters]
        assert "QualityScoreFilter" not in filter_names
        assert "TechnicalConsistencyFilter" not in filter_names

        # Check parameters
        conf_filter = next(f for f in custom_pipeline.filters if isinstance(f, ConfidenceFilter))
        assert conf_filter.min_confidence == 0.8

        risk_filter = next(f for f in custom_pipeline.filters if isinstance(f, RiskFilter))
        assert risk_filter.allowed_risk_levels == ["low"]

    def test_filter_performance(self, pipeline):
        """Test filter performance with many signals"""
        # Create 1000 signals
        large_signal_set = []
        for i in range(1000):
            signal = TradingSignal(
                id=f"PERF_{i}",
                symbol="AAPL",
                action="BUY" if i % 2 == 0 else "SELL",
                confidence=0.5 + (i % 50) / 100,
                price=150.0 + i / 100,
                timestamp=datetime.now() - timedelta(minutes=i),
                reason="Performance test",
                indicators={
                    "rsi": 20 + i % 60,
                    "volume_ratio": 1.0 + i % 3
                },
                risk_level="low" if i % 3 == 0 else "medium" if i % 3 == 1 else "high",
                entry_price=150.0,
                stop_loss=145.0,
                take_profit=160.0,
                quality_score=0.7 + (i % 30) / 100
            )
            large_signal_set.append(signal)

        # Filter should handle large sets efficiently
        import time
        start_time = time.time()
        filtered = pipeline.filter_signals(large_signal_set)
        elapsed_time = time.time() - start_time

        # Should complete quickly (< 1 second for 1000 signals)
        assert elapsed_time < 1.0

        # Should filter out many signals
        assert len(filtered) < len(large_signal_set) * 0.5
