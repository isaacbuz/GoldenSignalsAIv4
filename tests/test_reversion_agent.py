import pytest
import pandas as pd
from agents.predictive.reversion import ReversionAgent

# --- Helper: generate synthetic price/volume data ---
def make_test_df(length=50, uptrend=False, volume_spike_at=None):
    prices = [100 + i*0.5 if uptrend else 100 - i*0.2 for i in range(length)]
    volumes = [1000]*length
    if volume_spike_at is not None and 0 <= volume_spike_at < length:
        volumes[volume_spike_at] = 5000
    df = pd.DataFrame({
        'Close': prices,
        'Volume': volumes
    })
    return df

# --- Test RSI Factor ---
def test_rsi_factor():
    df = make_test_df(length=20, uptrend=True)
    rsi = ReversionAgent.rsi_factor(df)
    assert 0 <= rsi <= 100

# --- Test Momentum Factor ---
def test_momentum_factor():
    df = make_test_df(length=15, uptrend=True)
    assert ReversionAgent.momentum_factor(df, window=10) is True
    df_down = make_test_df(length=15, uptrend=False)
    assert ReversionAgent.momentum_factor(df_down, window=10) is False

# --- Test Volume Spike Factor ---
def test_volume_spike_factor():
    df = make_test_df(length=25, uptrend=False, volume_spike_at=24)
    assert ReversionAgent.volume_spike_factor(df, window=20, spike_ratio=2.0) is True
    df_no_spike = make_test_df(length=25, uptrend=False)
    assert ReversionAgent.volume_spike_factor(df_no_spike, window=20, spike_ratio=2.0) is False

# --- Test MACD Factor ---
def test_macd_factor():
    df = make_test_df(length=40, uptrend=True)
    macd = ReversionAgent.macd_factor(df)
    assert isinstance(macd, float)

# --- Test Bollinger Bands Factor ---
def test_bollinger_bands_factor():
    df = make_test_df(length=25, uptrend=True)
    pos = ReversionAgent.bollinger_bands_factor(df, window=20, num_std=2.0)
    assert pos in {'above', 'below', 'within'}

# --- Test Agent with all factors ---
def test_reversion_agent_multifactor():
    df = make_test_df(length=50, uptrend=True, volume_spike_at=49)
    agent = ReversionAgent(
        trade_horizon="day",
        custom_factors={
            "rsi": ReversionAgent.rsi_factor,
            "momentum": ReversionAgent.momentum_factor,
            "volume_spike": ReversionAgent.volume_spike_factor,
            "macd": ReversionAgent.macd_factor,
            "bollinger": ReversionAgent.bollinger_bands_factor
        }
    )
    signal = agent.process({"stock_data": df})
    assert signal["action"] in {"buy", "sell", "hold"}
    meta = signal["metadata"]
    assert "rsi" in meta and "macd" in meta and "bollinger" in meta
    assert "volume_spike" in meta and "momentum" in meta

# --- Test Backtest ---
def test_reversion_agent_backtest():
    df = make_test_df(length=60, uptrend=True)
    agent = ReversionAgent(trade_horizon="day")
    results = agent.backtest(df)
    assert set(results.keys()) == {"trades", "win_rate", "pnl"}
    assert results["trades"] >= 0
    assert 0.0 <= results["win_rate"] <= 1.0
