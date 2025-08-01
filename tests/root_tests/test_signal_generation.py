#!/usr/bin/env python3
"""
🎯 GoldenSignalsAI - Streamlined Signal Generation System
Pure focus on generating high-quality entry/exit signals for options trading
"""

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List, Optional
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SignalGenerator:
    """Core signal generation engine - clean and focused"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.load_models()

    def load_models(self):
        """Load trained ML models"""
        model_dir = Path("ml_training/models")

        try:
            # Load core models for signal generation
            self.models['signal_classifier'] = joblib.load(model_dir / 'signal_classifier_model.pkl')
            self.scalers['signal_classifier'] = joblib.load(model_dir / 'signal_classifier_scaler.pkl')

            self.models['direction'] = joblib.load(model_dir / 'direction_model.pkl')
            self.scalers['direction'] = joblib.load(model_dir / 'direction_scaler.pkl')

            self.models['risk'] = joblib.load(model_dir / 'risk_model.pkl')
            self.scalers['risk'] = joblib.load(model_dir / 'risk_scaler.pkl')

            logger.info("✅ ML models loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not load all models: {e}")
            logger.info("Using rule-based signals as fallback")

    def fetch_market_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """Fetch latest market data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)

            if data.empty:
                raise ValueError(f"No data found for {symbol}")

            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate essential technical indicators for signal generation"""
        df = data.copy()

        # Core indicators for options signals
        df['Returns'] = df['Close'].pct_change()
        df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))

        # Moving averages - matching training features
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['EMA_5'] = df['Close'].ewm(span=5).mean()
        df['EMA_10'] = df['Close'].ewm(span=10).mean()
        df['EMA_20'] = df['Close'].ewm(span=20).mean()

        # RSI - Key for oversold/overbought
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD - Momentum shifts
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

        # Bollinger Bands - Volatility expansion
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # Volume analysis
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        # Volatility for risk assessment
        df['Volatility_20'] = df['Returns'].rolling(20).std() * np.sqrt(252)

        # ATR - Average True Range
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()

        # Price patterns
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Open_Pct'] = (df['Close'] - df['Open']) / df['Open']

        # Support/Resistance levels
        df['Resistance'] = df['High'].rolling(20).max()
        df['Support'] = df['Low'].rolling(20).min()
        df['SR_Position'] = (df['Close'] - df['Support']) / (df['Resistance'] - df['Support'])

        return df.dropna()

    def generate_ml_signal(self, features: pd.DataFrame) -> Dict:
        """Generate signal using ML models"""
        if not self.models:
            return self.generate_rule_based_signal(features)

        try:
            # Prepare features for ML models - matching training features exactly
            feature_cols = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'Returns', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_20',
                'RSI', 'MACD', 'MACD_Signal',
                'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width',
                'Volume_Ratio', 'Volatility_20', 'High_Low_Pct'
            ]

            # Check if we have enough data
            if len(features) == 0:
                return self.generate_rule_based_signal(features)

            # Get available features from our data
            available_cols = [col for col in feature_cols if col in features.columns]

            # If we're missing required features, fall back to rule-based
            if len(available_cols) < len(feature_cols):
                logger.warning(f"Missing features for ML model. Using rule-based signals.")
                return self.generate_rule_based_signal(features)

            # Get latest features
            latest_features = features[available_cols].iloc[-1:].fillna(0)

            # Signal classification (Bear: 0, Neutral: 1, Bull: 2)
            X_scaled = self.scalers['signal_classifier'].transform(latest_features)
            signal_proba = self.models['signal_classifier'].predict_proba(X_scaled)[0]
            signal_class = np.argmax(signal_proba)

            # Direction prediction
            X_dir_scaled = self.scalers['direction'].transform(latest_features)
            direction_proba = self.models['direction'].predict(X_dir_scaled)[0]

            # Risk assessment
            X_risk_scaled = self.scalers['risk'].transform(latest_features)
            risk_score = self.models['risk'].predict(X_risk_scaled)[0]

            # Generate signal
            if signal_class == 2 and direction_proba > 0.6:  # Bull signal
                signal_type = "BUY_CALL"
                confidence = signal_proba[2] * direction_proba
            elif signal_class == 0 and direction_proba < 0.4:  # Bear signal
                signal_type = "BUY_PUT"
                confidence = signal_proba[0] * (1 - direction_proba)
            else:
                signal_type = "HOLD"
                confidence = signal_proba[1]

            return {
                "signal": signal_type,
                "confidence": float(confidence),
                "ml_probabilities": {
                    "bear": float(signal_proba[0]),
                    "neutral": float(signal_proba[1]),
                    "bull": float(signal_proba[2])
                },
                "direction_probability": float(direction_proba),
                "risk_score": float(risk_score)
            }

        except Exception as e:
            logger.error(f"ML signal generation failed: {e}")
            return self.generate_rule_based_signal(features)

    def generate_rule_based_signal(self, data: pd.DataFrame) -> Dict:
        """Fallback rule-based signal generation"""
        if len(data) == 0:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "active_indicators": [],
                "rule_based": True,
                "error": "No data available"
            }

        latest = data.iloc[-1]
        signals = []
        confidence = 0.5

        # RSI signals
        if latest['RSI'] < 30:
            signals.append("RSI_OVERSOLD")
            confidence += 0.1
        elif latest['RSI'] > 70:
            signals.append("RSI_OVERBOUGHT")
            confidence += 0.1

        # MACD signals
        if latest['MACD'] > latest['MACD_Signal'] and latest['MACD_Histogram'] > 0:
            signals.append("MACD_BULLISH")
            confidence += 0.1
        elif latest['MACD'] < latest['MACD_Signal'] and latest['MACD_Histogram'] < 0:
            signals.append("MACD_BEARISH")
            confidence += 0.1

        # Bollinger Band signals
        if latest['BB_Position'] < 0.2:
            signals.append("BB_OVERSOLD")
            confidence += 0.1
        elif latest['BB_Position'] > 0.8:
            signals.append("BB_OVERBOUGHT")
            confidence += 0.1

        # Moving average signals
        if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
            signals.append("MA_BULLISH")
            confidence += 0.1
        elif latest['Close'] < latest['SMA_20'] < latest['SMA_50']:
            signals.append("MA_BEARISH")
            confidence += 0.1

        # Determine final signal
        bull_signals = [s for s in signals if 'BULLISH' in s or 'OVERSOLD' in s]
        bear_signals = [s for s in signals if 'BEARISH' in s or 'OVERBOUGHT' in s]

        if len(bull_signals) >= 2:
            signal_type = "BUY_CALL"
        elif len(bear_signals) >= 2:
            signal_type = "BUY_PUT"
        else:
            signal_type = "HOLD"

        return {
            "signal": signal_type,
            "confidence": min(confidence, 1.0),
            "active_indicators": signals,
            "rule_based": True
        }

    def calculate_levels(self, data: pd.DataFrame) -> Dict:
        """Calculate entry, exit, and stop levels"""
        latest = data.iloc[-1]
        atr = data['Close'].rolling(14).apply(
            lambda x: np.mean(np.abs(np.diff(x)))
        ).iloc[-1]

        # Entry zone (current price +/- 0.5 ATR)
        entry_buffer = atr * 0.5
        entry_zone = [
            float(latest['Close'] - entry_buffer),
            float(latest['Close'] + entry_buffer)
        ]

        # Exit target (1.5x ATR from entry)
        exit_target = float(latest['Close'] + (atr * 1.5))

        # Stop loss (1x ATR from entry)
        stop_loss = float(latest['Close'] - atr)

        return {
            "current_price": float(latest['Close']),
            "entry_zone": entry_zone,
            "exit_target": exit_target,
            "stop_loss": stop_loss,
            "atr": float(atr)
        }

    def generate_signal(self, symbol: str) -> Dict:
        """Main signal generation method"""
        logger.info(f"\n🎯 Generating signal for {symbol}...")

        # Fetch data
        data = self.fetch_market_data(symbol)
        if data.empty:
            return {
                "symbol": symbol,
                "status": "error",
                "message": "Unable to fetch market data"
            }

        # Calculate indicators
        data_with_indicators = self.calculate_indicators(data)

        # Generate signal
        signal_info = self.generate_ml_signal(data_with_indicators)

        # Calculate levels
        levels = self.calculate_levels(data_with_indicators)

        # Build complete signal
        latest = data_with_indicators.iloc[-1]

        signal = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "signal": signal_info["signal"],
            "confidence": signal_info["confidence"],
            "timeframe": "1-5 days",
            "entry_zone": levels["entry_zone"],
            "exit_target": levels["exit_target"],
            "stop_loss": levels["stop_loss"],
            "current_price": levels["current_price"],
            "technical_indicators": {
                "rsi": float(latest['RSI']),
                "macd": float(latest['MACD']),
                "bb_position": float(latest['BB_Position']),
                "volume_ratio": float(latest['Volume_Ratio']),
                "volatility": float(latest['Volatility_20'])
            },
            "risk_reward_ratio": abs(
                (levels["exit_target"] - levels["current_price"]) /
                (levels["current_price"] - levels["stop_loss"])
            ),
            "reasoning": self._generate_reasoning(signal_info, latest)
        }

        # Add ML-specific info if available
        if 'ml_probabilities' in signal_info:
            signal['ml_analysis'] = signal_info

        return signal

    def _generate_reasoning(self, signal_info: Dict, latest_data: pd.Series) -> str:
        """Generate human-readable reasoning for the signal"""
        reasons = []

        # RSI reasoning
        if latest_data['RSI'] < 30:
            reasons.append("RSI oversold (<30)")
        elif latest_data['RSI'] > 70:
            reasons.append("RSI overbought (>70)")

        # MACD reasoning
        if latest_data['MACD_Histogram'] > 0:
            reasons.append("MACD bullish crossover")
        elif latest_data['MACD_Histogram'] < 0:
            reasons.append("MACD bearish crossover")

        # Bollinger Band reasoning
        if latest_data['BB_Position'] < 0.2:
            reasons.append("Price near lower Bollinger Band")
        elif latest_data['BB_Position'] > 0.8:
            reasons.append("Price near upper Bollinger Band")

        # Volume reasoning
        if latest_data['Volume_Ratio'] > 1.5:
            reasons.append("High volume spike")

        # ML reasoning if available
        if 'ml_probabilities' in signal_info:
            ml_probs = signal_info['ml_probabilities']
            dominant = max(ml_probs, key=ml_probs.get)
            reasons.append(f"ML model indicates {dominant} market ({ml_probs[dominant]:.0%} confidence)")

        return " + ".join(reasons) if reasons else "Mixed signals, proceed with caution"

async def test_live_signals():
    """Test signal generation with live data"""
    generator = SignalGenerator()

    # Test symbols
    symbols = ['AAPL', 'SPY', 'TSLA', 'GOOGL', 'NVDA']

    print("="*60)
    print("🎯 GoldenSignalsAI - Live Signal Generation")
    print("="*60)

    for symbol in symbols:
        signal = generator.generate_signal(symbol)

        if signal.get('status') == 'error':
            print(f"\n❌ {symbol}: {signal['message']}")
            continue

        print(f"\n{'='*60}")
        print(f"📊 {symbol} - ${signal['current_price']:.2f}")
        print(f"{'='*60}")
        print(f"🎯 Signal: {signal['signal']}")
        print(f"🔥 Confidence: {signal['confidence']:.1%}")
        print(f"⏱️ Timeframe: {signal['timeframe']}")
        print(f"\n📍 Entry Zone: ${signal['entry_zone'][0]:.2f} - ${signal['entry_zone'][1]:.2f}")
        print(f"🎯 Exit Target: ${signal['exit_target']:.2f}")
        print(f"🛑 Stop Loss: ${signal['stop_loss']:.2f}")
        print(f"📊 Risk/Reward: {signal['risk_reward_ratio']:.2f}:1")

        print(f"\n📈 Technical Indicators:")
        tech = signal['technical_indicators']
        print(f"   RSI: {tech['rsi']:.1f}")
        print(f"   MACD: {tech['macd']:.4f}")
        print(f"   BB Position: {tech['bb_position']:.1%}")
        print(f"   Volume Ratio: {tech['volume_ratio']:.2f}x")
        print(f"   Volatility: {tech['volatility']:.1%}")

        if 'ml_analysis' in signal:
            ml = signal['ml_analysis']['ml_probabilities']
            print(f"\n🤖 ML Analysis:")
            print(f"   Bear: {ml['bear']:.1%}")
            print(f"   Neutral: {ml['neutral']:.1%}")
            print(f"   Bull: {ml['bull']:.1%}")

        print(f"\n💡 Reasoning: {signal['reasoning']}")

    print(f"\n{'='*60}")
    print("✅ Signal generation complete!")
    print("="*60)

def export_signals_json(signals: List[Dict], filename: str = "signals.json"):
    """Export signals to JSON for frontend consumption"""
    with open(filename, 'w') as f:
        json.dump(signals, f, indent=2, default=str)
    logger.info(f"📄 Signals exported to {filename}")

if __name__ == "__main__":
    # Run signal generation test
    asyncio.run(test_live_signals())
