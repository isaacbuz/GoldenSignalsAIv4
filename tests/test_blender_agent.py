from backend.agents.blender_agent import blend_signals

def test_blender_output_format():
    inputs = [
        {"signal": "buy", "confidence": 0.8, "explanation": "RSI low"},
        {"signal": "buy", "confidence": 0.6, "explanation": "MACD cross"}
    ]
    result = blend_signals(inputs)
    assert result["signal"] == "buy"
    assert 0 <= result["confidence"] <= 1
    assert "explanation" in result