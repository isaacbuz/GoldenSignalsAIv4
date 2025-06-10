"""
Tests for the signal logging utility.
"""
import pytest
import os
import json
from datetime import datetime, timedelta
import shutil
from agents.common.utils.signal_logger import SignalLogger

@pytest.fixture
def temp_log_dir(tmpdir):
    """Create a temporary directory for log files."""
    log_dir = os.path.join(str(tmpdir), "test_signals")
    yield log_dir
    # Cleanup
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

@pytest.fixture
def sample_signals():
    """Generate sample trading signals."""
    base_time = datetime.now()
    return [
        {
            "action": "buy",
            "confidence": 0.8,
            "metadata": {"price": 100}
        },
        {
            "action": "hold",
            "confidence": 0.5,
            "metadata": {"price": 101}
        },
        {
            "action": "sell",
            "confidence": 0.7,
            "metadata": {"price": 102}
        }
    ]

def test_logger_initialization(temp_log_dir):
    """Test SignalLogger initialization."""
    logger = SignalLogger(log_dir=temp_log_dir)
    assert logger.log_dir == temp_log_dir
    assert os.path.exists(temp_log_dir)
    assert logger.current_log is None
    assert len(logger.signals) == 0

def test_new_log_creation(temp_log_dir):
    """Test creating new log file."""
    logger = SignalLogger(log_dir=temp_log_dir)
    logger.start_new_log("test_strategy")
    
    assert logger.current_log is not None
    assert os.path.exists(logger.current_log)
    assert logger.current_log.startswith(os.path.join(temp_log_dir, "test_strategy_"))
    assert logger.current_log.endswith(".json")

def test_signal_logging(temp_log_dir, sample_signals):
    """Test logging trading signals."""
    logger = SignalLogger(log_dir=temp_log_dir)
    logger.start_new_log("test_strategy")
    
    # Log multiple signals
    for signal in sample_signals:
        logger.log_signal(signal)
    
    assert len(logger.signals) == len(sample_signals)
    
    # Verify file contents
    with open(logger.current_log, "r") as f:
        saved_data = json.load(f)
    assert len(saved_data["signals"]) == len(sample_signals)
    assert all("timestamp" in s for s in saved_data["signals"])

def test_signal_analysis(temp_log_dir, sample_signals):
    """Test signal analysis functionality."""
    logger = SignalLogger(log_dir=temp_log_dir)
    logger.start_new_log("test_strategy")
    
    # Create signals with specific timestamps
    base_time = datetime.now()
    for i, signal in enumerate(sample_signals):
        logger.signals.append({
            "timestamp": (base_time + timedelta(hours=i)).isoformat(),
            "signal": signal
        })
    
    # Test analysis without date filters
    analysis = logger.analyze_signals()
    assert analysis["total_signals"] == len(sample_signals)
    assert "buy" in analysis["signal_distribution"]
    assert "sell" in analysis["signal_distribution"]
    assert "hold" in analysis["signal_distribution"]
    assert isinstance(analysis["average_confidence"], float)
    
    # Test analysis with date filter
    filtered_analysis = logger.analyze_signals(
        start_date=(base_time + timedelta(hours=1)).isoformat()
    )
    assert filtered_analysis["total_signals"] == len(sample_signals) - 1

def test_error_handling(temp_log_dir):
    """Test error handling scenarios."""
    logger = SignalLogger(log_dir=temp_log_dir)
    
    # Test logging without starting new log
    with pytest.raises(ValueError):
        logger.log_signal({"action": "buy"})
    
    # Test analysis with no signals
    analysis = logger.analyze_signals()
    assert "error" in analysis
    
    # Test loading non-existent log
    assert not logger.load_log("nonexistent.json")

def test_log_loading(temp_log_dir, sample_signals):
    """Test loading existing log files."""
    logger = SignalLogger(log_dir=temp_log_dir)
    logger.start_new_log("test_strategy")
    
    # Log some signals
    for signal in sample_signals:
        logger.log_signal(signal)
    
    # Create new logger instance and load log
    new_logger = SignalLogger(log_dir=temp_log_dir)
    log_file = os.path.basename(logger.current_log)
    assert new_logger.load_log(log_file)
    assert len(new_logger.signals) == len(sample_signals)

def test_recent_signals(temp_log_dir, sample_signals):
    """Test retrieving recent signals."""
    logger = SignalLogger(log_dir=temp_log_dir)
    logger.start_new_log("test_strategy")
    
    # Log signals
    for signal in sample_signals:
        logger.log_signal(signal)
    
    # Test getting recent signals
    recent = logger.get_recent_signals(2)
    assert len(recent) == 2
    assert recent[-1]["signal"]["action"] == sample_signals[-1]["action"]
    
    # Test getting more signals than available
    all_recent = logger.get_recent_signals(10)
    assert len(all_recent) == len(sample_signals)

def test_signal_transitions(temp_log_dir):
    """Test signal transition matrix calculation."""
    logger = SignalLogger(log_dir=temp_log_dir)
    logger.start_new_log("test_strategy")
    
    # Create sequence of signals
    signals = [
        {"action": "buy", "confidence": 0.8},
        {"action": "hold", "confidence": 0.5},
        {"action": "hold", "confidence": 0.5},
        {"action": "sell", "confidence": 0.7},
        {"action": "buy", "confidence": 0.8}
    ]
    
    for signal in signals:
        logger.log_signal(signal)
    
    analysis = logger.analyze_signals()
    transitions = analysis["signal_transitions"]
    
    # Verify transitions exist
    assert "buy" in transitions
    assert "hold" in transitions
    assert "sell" in transitions 