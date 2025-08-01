import pytest
from unittest.mock import Mock, patch
from src.services.signal_generation_engine import SignalGenerationEngine

def test_signal_generation_engine_initialization():
    engine = SignalGenerationEngine()
    assert engine is not None

@pytest.mark.asyncio
async def test_generate_signals():
    engine = SignalGenerationEngine()
    signals = await engine.generate_signals("AAPL")
    assert isinstance(signals, list)
