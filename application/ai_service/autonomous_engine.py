"""
autonomous_engine.py
Purpose: Implements the AutonomousEngine for GoldenSignalsAI, responsible for analyzing market data, making trade decisions, and managing risk profiles using multi-timeframe indicators and ensemble logic.
"""

from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
import pandas as pd
import ta

class Action(Enum):
    LONG = auto()
    SHORT = auto()
    HOLD = auto()

@dataclass
class TradeDecision:
    symbol: str
    action: Action
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    timeframe: str
    rationale: dict
    risk_profile: str

class AutonomousEngine:
    def __init__(self, validator):
        self.validator = validator
        self.confluence_thresholds = {'conservative': 0.8, 'balanced': 0.7, 'aggressive': 0.6}
        self.timeframes = ["1m", "5m", "15m", "1h"]

    async def analyze_and_decide(self, symbol: str, data: dict, risk_profile: str = "balanced"):
        decision = self._indicator_based_decision(symbol, data, risk_profile)
        if decision.action != Action.HOLD:
            return decision
        return self._ensemble_based_decision(symbol, data, risk_profile)

    def _indicator_based_decision(self, symbol: str, data: dict, risk_profile: str):
        indicators = self._calculate_multi_timeframe_indicators(data)
        score = self._score_confluence(indicators, risk_profile)
        position_size = self._calculate_position_size(indicators['15m'], risk_profile)
        return self._make_decision(symbol, score, indicators, position_size, risk_profile)

    def _ensemble_based_decision(self, symbol: str, data: dict, risk_profile: str):
        latest_df = data['15m']
        predictions = self.validator.predict(latest_df)
        avg_prediction = np.mean(predictions)
        if avg_prediction > 0.7:
            entry = latest_df['close'].iloc[-1]
            stop_loss = entry * (1 - 0.02)
            take_profit = entry * (1 + 0.04)
            return TradeDecision(symbol=symbol, action=Action.LONG, confidence=avg_prediction, entry_price=entry,
                                 stop_loss=stop_loss, take_profit=take_profit, timeframe='15m',
                                 rationale={'ensemble_prediction': float(avg_prediction)}, risk_profile=risk_profile)
        elif avg_prediction < 0.3:
            entry = latest_df['close'].iloc[-1]
            stop_loss = entry * (1 + 0.02)
            take_profit = entry * (1 - 0.04)
            return TradeDecision(symbol=symbol, action=Action.SHORT, confidence=1 - avg_prediction, entry_price=entry,
                                 stop_loss=stop_loss, take_profit=take_profit, timeframe='15m',
                                 rationale={'ensemble_prediction': float(avg_prediction)}, risk_profile=risk_profile)
        return TradeDecision(symbol=symbol, action=Action.HOLD, confidence=0, entry_price=0, stop_loss=0,
                             take_profit=0, timeframe='', rationale={}, risk_profile=risk_profile)

    def _calculate_multi_timeframe_indicators(self, data):
        indicators = {}
        for tf, df in data.items():
            df = ta.add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume")
            df['MA_Confluence'] = self._ma_confluence(df)
            df['BB_Score'] = self._bollinger_score(df)
            df['MACD_Strength'] = self._macd_strength(df)
            df['VWAP_Score'] = self._vwap_score(df)
            df['Volume_Spike'] = self._volume_spike_score(df)
            indicators[tf] = df.iloc[-1]
        return indicators

    def _ma_confluence(self, df):
        ma10 = df['trend_sma_fast']
        ma50 = df['trend_sma_slow']
        ma200 = df['trend_sma_long']
        score = 0
        if ma10 > ma50: score += 1
        if ma50 > ma200: score += 1
        if ma10 > ma200: score += 1
        return score / 3

    def _bollinger_score(self, df):
        close = df['close']
        upper_band = df['volatility_bbh']
        lower_band = df['volatility_bbl']
        if close >= upper_band:
            return -1
        elif close <= lower_band:
            return 1
        return 2 * ((close - lower_band) / (upper_band - lower_band)) - 1

    def _macd_strength(self, df):
        macd_line = df['trend_macd']
        signal_line = df['trend_macd_signal']
        histogram = df['trend_macd_diff']
        if macd_line > signal_line and histogram > 0:
            return min(1, histogram / (macd_line * 0.1))
        elif macd_line < signal_line and histogram < 0:
            return max(-1, histogram / (macd_line * 0.1))
        return 0

    def _volume_spike_score(self, df):
        current_volume = df['volume']
        avg_volume = df['volume'].rolling(window=20).mean()
        ratio = current_volume / avg_volume
        if ratio > 2.5:
            return 1
        elif ratio > 1.8:
            return 0.5
        return 0

    def _vwap_score(self, df):
        close = df['close']
        vwap = df['volume_vwap']
        if close < vwap * 0.995:
            return 1
        elif close > vwap * 1.005:
            return -1
        return 0

    def _score_confluence(self, indicators, risk_profile):
        time_weights = {'1m': 0.2, '5m': 0.3, '15m': 0.3, '1h': 0.2}
        total_score = 0
        for tf, data in indicators.items():
            tf_score = 0
            tf_score += 0.3 * data['MA_Confluence']
            tf_score += 0.1 * self._normalize_rsi(data['momentum_rsi'])
            tf_score += 0.2 * data['MACD_Strength']
            tf_score += 0.2 * data['BB_Score']
            tf_score += 0.1 * data['Volume_Spike']
            tf_score += 0.1 * data['VWAP_Score']
            total_score += tf_score * time_weights[tf]
        return total_score

    def _normalize_rsi(self, rsi):
        if rsi < 30:
            return 1
        elif rsi > 70:
            return -1
        return (50 - rsi) / 50

    def _calculate_position_size(self, indicators, risk_profile):
        return 10

    def _make_decision(self, symbol, score, indicators, position_size, risk_profile):
        latest = indicators['15m']
        threshold = self.confluence_thresholds[risk_profile]
        if score >= threshold:
            entry = latest['close']
            stop_loss = entry * (1 - 0.02)
            take_profit = entry * (1 + 0.04)
            return TradeDecision(
                symbol=symbol, action=Action.LONG, confidence=score, entry_price=entry,
                stop_loss=stop_loss, take_profit=take_profit, timeframe='15m',
                rationale={
                    'MA_Confluence': latest['MA_Confluence'], 'RSI': latest['momentum_rsi'],
                    'MACD': latest['MACD_Strength'], 'VWAP': latest['VWAP_Score'],
                    'Volume': latest['Volume_Spike']
                }, risk_profile=risk_profile
            )
        elif score <= -threshold:
            entry = latest['close']
            stop_loss = entry * (1 + 0.02)
            take_profit = entry * (1 - 0.04)
            return TradeDecision(
                symbol=symbol, action=Action.SHORT, confidence=abs(score), entry_price=entry,
                stop_loss=stop_loss, take_profit=take_profit, timeframe='15m',
                rationale={
                    'MA_Confluence': latest['MA_Confluence'], 'RSI': latest['momentum_rsi'],
                    'MACD': latest['MACD_Strength'], 'VWAP': latest['VWAP_Score'],
                    'Volume': latest['Volume_Spike']
                }, risk_profile=risk_profile
            )
        return TradeDecision(symbol=symbol, action=Action.HOLD, confidence=0, entry_price=0,
                             stop_loss=0, take_profit=0, timeframe='', rationale={},
                             risk_profile=risk_profile)
