from archive.legacy_backend_agents.indicators.rsi_agent import RSI

def test_rsi_run_returns_valid_signal():
    agent = RSI()
    result = agent.run("AAPL")
    assert "signal" in result
    assert result["signal"] in ["buy", "sell", "hold"]
    assert 0 <= result["confidence"] <= 1