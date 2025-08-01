"""
Integration tests for the complete signal generation pipeline
Tests the flow from data → signal generation → filtering → monitoring
"""
import pytest
import asyncio
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from src.services.signal_generation_engine import SignalGenerationEngine, TradingSignal
from src.services.signal_filtering_pipeline import SignalFilteringPipeline
from src.services.signal_monitoring_service import SignalMonitoringService
from src.services.data_quality_validator import DataQualityValidator


class TestSignalPipelineIntegration:
    """Integration tests for full signal pipeline"""

    @pytest.fixture
    async def signal_engine(self):
        """Create signal generation engine"""
        return SignalGenerationEngine()

    @pytest.fixture
    def filter_pipeline(self):
        """Create signal filtering pipeline"""
        return SignalFilteringPipeline()

    @pytest.fixture
    def monitoring_service(self):
        """Create monitoring service"""
        return SignalMonitoringService(db_path=":memory:")

    @pytest.fixture
    def data_validator(self):
        """Create data quality validator"""
        return DataQualityValidator()

    @pytest.mark.asyncio
    async def test_full_signal_pipeline(self, signal_engine, filter_pipeline, monitoring_service):
        """Test complete signal pipeline from generation to monitoring"""
        # Step 1: Generate signals
        symbols = ["AAPL", "GOOGL", "MSFT"]
        signals = await signal_engine.generate_signals(symbols)

        assert len(signals) > 0
        assert all(isinstance(s, TradingSignal) for s in signals)

        # Step 2: Filter signals
        filtered_signals = filter_pipeline.filter_signals(signals)

        assert len(filtered_signals) <= len(signals)
        pipeline_stats = filter_pipeline.get_pipeline_stats()
        assert pipeline_stats["total_signals_processed"] > 0

        # Step 3: Monitor signals
        for signal in filtered_signals[:5]:  # Track first 5 signals
            monitoring_service.track_signal_entry(
                signal.id,
                signal.entry_price,
                symbol=signal.symbol,
                action=signal.action,
                risk_level=signal.risk_level
            )

            # Simulate exit
            exit_price = signal.entry_price * (1.02 if signal.action == "BUY" else 0.98)
            monitoring_service.track_signal_exit(
                signal.id,
                exit_price,
                "success" if exit_price > signal.entry_price else "failure"
            )

        # Verify metrics
        metrics = monitoring_service.get_performance_metrics()
        assert metrics.total_signals >= min(5, len(filtered_signals))

    @pytest.mark.asyncio
    async def test_data_quality_integration(self, data_validator, signal_engine):
        """Test data quality validation integration with signal generation"""
        # Get validated data
        symbol = "SPY"
        data, source = await data_validator.get_market_data_with_fallback(symbol)

        if data is not None and not data.empty:
            # Validate data quality
            quality_report = data_validator.validate_data(data, symbol)

            # Generate signals only if quality is good
            if quality_report.accuracy > 0.7:
                signals = await signal_engine.generate_signals([symbol])
                assert len(signals) > 0

                # Check that signal quality reflects data quality
                for signal in signals:
                    assert signal.confidence <= 1.0
                    assert signal.quality_score <= quality_report.accuracy

    @pytest.mark.asyncio
    async def test_filter_parameter_adjustment(self, signal_engine, filter_pipeline, monitoring_service):
        """Test dynamic filter parameter adjustment based on performance"""
        # Generate initial signals
        signals = await signal_engine.generate_signals(["AAPL", "GOOGL"])

        # Filter with default parameters
        initial_filtered = filter_pipeline.filter_signals(signals)
        initial_count = len(initial_filtered)

        # Simulate poor performance
        for i in range(10):
            monitoring_service.track_signal_entry(f"test_{i}", 100)
            monitoring_service.track_signal_exit(f"test_{i}", 95, "failure")

        # Get performance metrics
        metrics = monitoring_service.get_performance_metrics()

        # Adjust filter parameters based on poor performance
        performance_data = {
            "false_positive_rate": 0.7,  # High false positive rate
            "signal_accuracy": metrics.win_rate
        }
        filter_pipeline.adjust_filter_parameters(performance_data)

        # Filter again with adjusted parameters
        adjusted_filtered = filter_pipeline.filter_signals(signals)
        adjusted_count = len(adjusted_filtered)

        # Should filter more aggressively after poor performance
        assert adjusted_count <= initial_count

    @pytest.mark.asyncio
    async def test_ml_model_integration(self, signal_engine):
        """Test ML model training and prediction integration"""
        # Create training data
        dates = pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=100, freq='D')
        training_data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(95, 115, 100),
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)

        # Train ML model
        await signal_engine.train_ml_model(training_data, "TEST")

        # Generate signals using trained model
        signals = await signal_engine.generate_signals(["TEST"])

        # Verify ML-enhanced signals
        assert len(signals) > 0
        for signal in signals:
            assert "ml_prediction" in signal.indicators
            assert signal.indicators["ml_prediction"] in [0, 1]

    @pytest.mark.asyncio
    async def test_signal_lifecycle_tracking(self, signal_engine, filter_pipeline, monitoring_service):
        """Test complete signal lifecycle from generation to outcome"""
        # Generate signal
        signals = await signal_engine.generate_signals(["AAPL"])
        assert len(signals) > 0

        signal = signals[0]

        # Filter signal
        filtered = filter_pipeline.filter_signals([signal])

        if filtered:
            # Track lifecycle
            monitoring_service.track_signal_entry(
                signal.id,
                signal.entry_price,
                symbol=signal.symbol,
                action=signal.action,
                confidence=signal.confidence
            )

            # Collect feedback
            monitoring_service.collect_feedback(
                signal.id,
                user_rating=4,
                outcome="profitable",
                notes="Good timing"
            )

            # Exit signal
            outcome = monitoring_service.track_signal_exit(
                signal.id,
                signal.take_profit,
                "success"
            )

            assert outcome is not None
            assert outcome.profit_loss > 0

            # Get feedback summary
            feedback_summary = monitoring_service.get_signal_feedback_summary()
            assert feedback_summary["total_feedback"] > 0

    @pytest.mark.asyncio
    async def test_multi_symbol_batch_processing(self, signal_engine, filter_pipeline):
        """Test batch processing of multiple symbols"""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

        # Generate signals for all symbols
        start_time = datetime.now(tz=timezone.utc)
        all_signals = await signal_engine.generate_signals(symbols)
        generation_time = (datetime.now(tz=timezone.utc) - start_time).total_seconds()

        # Should generate signals for multiple symbols efficiently
        assert len(all_signals) >= len(symbols)  # At least one signal per symbol
        assert generation_time < 10  # Should complete within 10 seconds

        # Filter all signals at once
        start_time = datetime.now(tz=timezone.utc)
        filtered_signals = filter_pipeline.filter_signals(all_signals)
        filter_time = (datetime.now(tz=timezone.utc) - start_time).total_seconds()

        assert filter_time < 1  # Filtering should be fast

        # Check signal distribution
        symbol_counts = {}
        for signal in filtered_signals:
            symbol_counts[signal.symbol] = symbol_counts.get(signal.symbol, 0) + 1

        # Should have signals from multiple symbols
        assert len(symbol_counts) > 1

    @pytest.mark.asyncio
    async def test_error_recovery_integration(self, signal_engine, filter_pipeline):
        """Test error handling and recovery in the pipeline"""
        # Include invalid symbol
        symbols = ["AAPL", "INVALID_SYMBOL_XYZ", "GOOGL"]

        # Should handle errors gracefully
        signals = await signal_engine.generate_signals(symbols)

        # Should still generate signals for valid symbols
        valid_symbols = {"AAPL", "GOOGL"}
        generated_symbols = {s.symbol for s in signals}

        assert len(generated_symbols & valid_symbols) > 0

        # Filter should handle any malformed signals
        filtered = filter_pipeline.filter_signals(signals)
        assert isinstance(filtered, list)

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, monitoring_service):
        """Test performance monitoring and reporting"""
        # Simulate a trading session
        session_signals = [
            ("sig1", "AAPL", 150, 155, "success"),
            ("sig2", "GOOGL", 2800, 2750, "failure"),
            ("sig3", "MSFT", 400, 410, "success"),
            ("sig4", "AAPL", 155, 152, "failure"),
            ("sig5", "TSLA", 1000, 1050, "success"),
        ]

        for sig_id, symbol, entry, exit, outcome in session_signals:
            monitoring_service.track_signal_entry(sig_id, entry, symbol=symbol)
            monitoring_service.track_signal_exit(sig_id, exit, outcome)

        # Get comprehensive metrics
        metrics = monitoring_service.get_performance_metrics()

        # Verify metrics calculation
        assert metrics.total_signals == 5
        assert metrics.successful_signals == 3
        assert metrics.win_rate == 0.6

        # Check symbol breakdown
        assert "AAPL" in metrics.by_symbol
        assert metrics.by_symbol["AAPL"]["total_signals"] == 2

        # Generate recommendations
        recommendations = monitoring_service.generate_improvement_recommendations()
        assert len(recommendations) > 0

        # Save snapshot
        monitoring_service.save_performance_snapshot()

        # Verify snapshot was saved
        conn = monitoring_service._get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM performance_snapshots")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1
