"""
Unit tests for SignalMonitoringService
"""
import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from src.services.signal_monitoring_service import (
    SignalMonitoringService, SignalOutcome, PerformanceMetrics
)


class TestSignalMonitoringService:
    """Test cases for SignalMonitoringService"""
    
    @pytest.fixture
    def monitoring_service(self):
        """Create test monitoring service"""
        # Use in-memory database for testing
        service = SignalMonitoringService(db_path=":memory:")
        return service
    
    def test_initialization(self, monitoring_service):
        """Test service initialization"""
        assert monitoring_service is not None
        assert monitoring_service.signal_outcomes == {}
        assert monitoring_service.signal_feedback == {}
    
    def test_track_signal_entry(self, monitoring_service):
        """Test tracking signal entry"""
        signal_id = "test_signal_001"
        entry_price = 100.0
        
        monitoring_service.track_signal_entry(signal_id, entry_price)
        
        assert signal_id in monitoring_service.signal_outcomes
        outcome = monitoring_service.signal_outcomes[signal_id]
        assert outcome.signal_id == signal_id
        assert outcome.entry_price == entry_price
        assert outcome.exit_price is None
        assert outcome.exit_time is None
    
    def test_track_signal_exit(self, monitoring_service):
        """Test tracking signal exit"""
        signal_id = "test_signal_002"
        entry_price = 100.0
        exit_price = 105.0
        
        # Track entry first
        monitoring_service.track_signal_entry(signal_id, entry_price)
        
        # Track exit
        outcome = monitoring_service.track_signal_exit(
            signal_id, exit_price, "test_complete"
        )
        
        assert outcome is not None
        assert outcome.exit_price == exit_price
        assert outcome.profit_loss == 5.0
        assert outcome.profit_loss_pct == 5.0
        assert outcome.outcome == "test_complete"
        assert outcome.exit_time is not None
    
    def test_calculate_performance_metrics(self, monitoring_service):
        """Test performance metrics calculation"""
        # Add some test signals
        signals = [
            ("sig1", 100, 105, "success"),  # 5% profit
            ("sig2", 100, 95, "failure"),   # 5% loss
            ("sig3", 100, 110, "success"),  # 10% profit
        ]
        
        for sig_id, entry, exit, outcome in signals:
            monitoring_service.track_signal_entry(sig_id, entry)
            monitoring_service.track_signal_exit(sig_id, exit, outcome)
        
        metrics = monitoring_service.get_performance_metrics()
        
        assert metrics.total_signals == 3
        assert metrics.successful_signals == 2
        assert metrics.failed_signals == 1
        assert metrics.win_rate == pytest.approx(0.667, rel=0.01)
        assert metrics.average_profit == pytest.approx(7.5, rel=0.01)
        assert metrics.average_loss == pytest.approx(-5.0, rel=0.01)
        assert metrics.profit_factor == pytest.approx(1.5, rel=0.01)
    
    def test_performance_by_symbol(self, monitoring_service):
        """Test performance breakdown by symbol"""
        # Add signals for different symbols
        monitoring_service.track_signal_entry("sig1", 100, symbol="AAPL")
        monitoring_service.track_signal_exit("sig1", 105, "success")
        
        monitoring_service.track_signal_entry("sig2", 200, symbol="GOOGL")
        monitoring_service.track_signal_exit("sig2", 210, "success")
        
        monitoring_service.track_signal_entry("sig3", 100, symbol="AAPL")
        monitoring_service.track_signal_exit("sig3", 95, "failure")
        
        metrics = monitoring_service.get_performance_metrics()
        
        assert "AAPL" in metrics.by_symbol
        assert "GOOGL" in metrics.by_symbol
        assert metrics.by_symbol["AAPL"]["total_signals"] == 2
        assert metrics.by_symbol["GOOGL"]["total_signals"] == 1
    
    def test_improvement_recommendations(self, monitoring_service):
        """Test improvement recommendation generation"""
        # Add some failing signals
        for i in range(5):
            monitoring_service.track_signal_entry(f"fail_{i}", 100)
            monitoring_service.track_signal_exit(f"fail_{i}", 95, "failure")
        
        recommendations = monitoring_service.generate_improvement_recommendations()
        
        assert len(recommendations) > 0
        assert any("win rate" in rec.lower() for rec in recommendations)
    
    def test_collect_feedback(self, monitoring_service):
        """Test feedback collection"""
        signal_id = "test_signal_003"
        monitoring_service.collect_feedback(
            signal_id,
            user_rating=4,
            outcome="profitable",
            notes="Good signal timing"
        )
        
        assert signal_id in monitoring_service.signal_feedback
        feedback = monitoring_service.signal_feedback[signal_id]
        assert feedback["user_rating"] == 4
        assert feedback["outcome"] == "profitable"
        assert feedback["notes"] == "Good signal timing"
    
    def test_save_performance_snapshot(self, monitoring_service):
        """Test saving performance snapshot"""
        # Add some signals
        monitoring_service.track_signal_entry("sig1", 100)
        monitoring_service.track_signal_exit("sig1", 105, "success")
        
        # Save snapshot
        monitoring_service.save_performance_snapshot()
        
        # Verify snapshot was saved
        conn = monitoring_service._get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM performance_snapshots")
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 1
    
    def test_get_signal_feedback_summary(self, monitoring_service):
        """Test feedback summary generation"""
        # Add feedback for multiple signals
        monitoring_service.collect_feedback("sig1", 5, "success", "Excellent")
        monitoring_service.collect_feedback("sig2", 3, "partial", "OK")
        monitoring_service.collect_feedback("sig3", 1, "failure", "Poor timing")
        
        summary = monitoring_service.get_signal_feedback_summary()
        
        assert summary["total_feedback"] == 3
        assert summary["average_rating"] == 3.0
        assert summary["outcome_distribution"]["success"] == 1
        assert summary["outcome_distribution"]["partial"] == 1
        assert summary["outcome_distribution"]["failure"] == 1
    
    def test_timeframe_filtering(self, monitoring_service):
        """Test performance metrics with timeframe filtering"""
        # Add signals at different times
        now = datetime.now(timezone.utc)
        
        # Recent signal
        monitoring_service.track_signal_entry("recent", 100)
        monitoring_service.track_signal_exit("recent", 105, "success")
        
        # Old signal (simulate by modifying the entry time)
        old_signal = SignalOutcome(
            signal_id="old",
            entry_price=100,
            entry_time=now - timedelta(days=10)
        )
        monitoring_service.signal_outcomes["old"] = old_signal
        monitoring_service.track_signal_exit("old", 95, "failure")
        
        # Get metrics for last 7 days
        metrics = monitoring_service.get_performance_metrics(
            timeframe=timedelta(days=7)
        )
        
        # Should only include recent signal
        assert metrics.total_signals == 1
        assert metrics.successful_signals == 1
    
    def test_sharpe_ratio_calculation(self, monitoring_service):
        """Test Sharpe ratio calculation"""
        # Add multiple signals with varied returns
        returns = [5, -2, 3, -1, 4, 2, -3, 6, 1, -2]
        
        for i, ret in enumerate(returns):
            entry = 100
            exit = entry * (1 + ret / 100)
            monitoring_service.track_signal_entry(f"sig_{i}", entry)
            monitoring_service.track_signal_exit(
                f"sig_{i}", exit, "success" if ret > 0 else "failure"
            )
        
        metrics = monitoring_service.get_performance_metrics()
        
        # Sharpe ratio should be calculated
        assert metrics.sharpe_ratio != 0
        assert isinstance(metrics.sharpe_ratio, float)
    
    def test_max_drawdown_calculation(self, monitoring_service):
        """Test max drawdown calculation"""
        # Simulate a drawdown scenario
        cumulative = 100
        signals = [
            ("sig1", 100, 110),  # +10
            ("sig2", 110, 115),  # +5
            ("sig3", 115, 105),  # -10
            ("sig4", 105, 95),   # -10
            ("sig5", 95, 100),   # +5
        ]
        
        for sig_id, entry, exit in signals:
            monitoring_service.track_signal_entry(sig_id, entry)
            monitoring_service.track_signal_exit(
                sig_id, exit, "success" if exit > entry else "failure"
            )
        
        metrics = monitoring_service.get_performance_metrics()
        
        # Max drawdown should be calculated
        assert metrics.max_drawdown < 0
        assert isinstance(metrics.max_drawdown, float) 