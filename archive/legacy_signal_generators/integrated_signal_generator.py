#!/usr/bin/env python3
"""
🎯 GoldenSignalsAI - Integrated Signal Generator
Combines real ML models with the blending system
"""

import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from pathlib import Path
from typing import Dict, Optional
import logging
from ml_signal_blender import MLSignalBlender, ModelOutputs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedSignalGenerator:
    """Production signal generator with ML model blending"""
    
    def __init__(self):
        self.blender = MLSignalBlender()
        self.models = {}
        self.scalers = {}
        self.load_models()
        
    def load_models(self):
        """Load trained ML models"""
        model_dir = Path("ml_training/models")
        
        try:
            # Load models if they exist
            if (model_dir / 'signal_classifier_model.pkl').exists():
                self.models['signal_classifier'] = joblib.load(model_dir / 'signal_classifier_model.pkl')
                self.scalers['signal_classifier'] = joblib.load(model_dir / 'signal_classifier_scaler.pkl')
            
            if (model_dir / 'direction_model.pkl').exists():
                self.models['direction'] = joblib.load(model_dir / 'direction_model.pkl')
                self.scalers['direction'] = joblib.load(model_dir / 'direction_scaler.pkl')
            
            if (model_dir / 'risk_model.pkl').exists():
                self.models['risk'] = joblib.load(model_dir / 'risk_model.pkl')
                self.scalers['risk'] = joblib.load(model_dir / 'risk_scaler.pkl')
                
            logger.info(f"✅ Loaded {len(self.models)} ML models")
        except Exception as e:
            logger.warning(f"Could not load all models: {e}")
    
    def generate_integrated_signal(self, symbol: str, data: Optional[pd.DataFrame] = None) -> Dict:
        """Generate signal using ML models and blending system"""
        
        # Fetch data if not provided
        if data is None:
            data = self.fetch_and_prepare_data(symbol)
            if data is None:
                return self.generate_fallback_signal(symbol)
        
        # Get model outputs
        model_outputs = self.get_model_predictions(data)
        
        # Get technical signals
        technical_signals = self.calculate_technical_signals(data)
        
        # Blend all signals
        blended_signal = self.blender.blend_models(model_outputs, technical_signals)
        
        # Add price levels
        price_levels = self.calculate_price_levels(data, blended_signal)
        
        # Build final signal
        return {
            'symbol': symbol,
            'timestamp': pd.Timestamp.now().isoformat(),
            'signal': blended_signal['signal'],
            'confidence': blended_signal['confidence'],
            'strength': blended_signal['strength'],
            'risk_level': blended_signal['risk_level'],
            'consensus': blended_signal['model_consensus'],
            **price_levels,
            'reasoning': blended_signal['reasoning'],
            'model_details': blended_signal['detailed_scores']
        }
    
    def fetch_and_prepare_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch and prepare data for ML models"""
        try:
            # For demo, use simulated data
            # In production, this would fetch real data
            dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='D')
            
            # Generate realistic price data
            np.random.seed(hash(symbol) % 1000)
            returns = np.random.normal(0.0005, 0.02, 100)
            prices = 100 * np.exp(np.cumsum(returns))
            
            data = pd.DataFrame({
                'Close': prices,
                'Open': prices * (1 + np.random.normal(0, 0.005, 100)),
                'High': prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
                'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
                'Volume': np.random.lognormal(17, 0.5, 100)
            }, index=dates)
            
            # Calculate features
            return self.calculate_features(data)
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None
    
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features needed by ML models"""
        df = data.copy()
        
        # Basic features
        df['Returns'] = df['Close'].pct_change()
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['EMA_10'] = df['Close'].ewm(span=10).mean()
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['SMA_20']
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Volume
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Volatility
        df['Volatility_20'] = df['Returns'].rolling(20).std() * np.sqrt(252)
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        
        return df.dropna()
    
    def get_model_predictions(self, data: pd.DataFrame) -> ModelOutputs:
        """Get predictions from all ML models"""
        
        # Prepare features
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Returns', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_20',
            'RSI', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width',
            'Volume_Ratio', 'Volatility_20', 'High_Low_Pct'
        ]
        
        # Get available features
        available_cols = [col for col in feature_cols if col in data.columns]
        features = data[available_cols].iloc[-1:].fillna(0)
        
        # Default outputs (for demo or if models not loaded)
        signal_probs = {'bear': 0.3, 'neutral': 0.4, 'bull': 0.3}
        direction_prob = 0.5
        risk_score = 0.5
        
        # Get real predictions if models are loaded
        if self.models:
            try:
                # Signal classifier
                if 'signal_classifier' in self.models:
                    X_scaled = self.scalers['signal_classifier'].transform(features)
                    probs = self.models['signal_classifier'].predict_proba(X_scaled)[0]
                    signal_probs = {'bear': probs[0], 'neutral': probs[1], 'bull': probs[2]}
                
                # Direction predictor
                if 'direction' in self.models:
                    X_scaled = self.scalers['direction'].transform(features)
                    direction_prob = self.models['direction'].predict(X_scaled)[0]
                
                # Risk model
                if 'risk' in self.models:
                    X_scaled = self.scalers['risk'].transform(features)
                    risk_score = self.models['risk'].predict(X_scaled)[0]
                    # Normalize to 0-1 range
                    risk_score = np.clip(risk_score, 0, 1)
                    
            except Exception as e:
                logger.warning(f"Error getting model predictions: {e}")
        
        # For demo, add some variation based on technical indicators
        else:
            latest = data.iloc[-1]
            
            # Adjust based on RSI
            if latest['RSI'] < 30:
                signal_probs['bull'] += 0.2
                signal_probs['bear'] -= 0.1
                direction_prob = 0.7
            elif latest['RSI'] > 70:
                signal_probs['bear'] += 0.2
                signal_probs['bull'] -= 0.1
                direction_prob = 0.3
            
            # Normalize probabilities
            total = sum(signal_probs.values())
            signal_probs = {k: v/total for k, v in signal_probs.items()}
            
            # Risk based on volatility
            risk_score = np.clip(latest['Volatility_20'] * 2, 0.1, 0.9)
        
        return ModelOutputs(
            signal_probabilities=signal_probs,
            direction_probability=direction_prob,
            risk_score=risk_score,
            confidence_scores={
                'signal': max(signal_probs.values()),
                'direction': abs(direction_prob - 0.5) * 2,
                'risk': 1 - risk_score
            }
        )
    
    def calculate_technical_signals(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicator signals"""
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        signals = {
            'rsi': latest['RSI'],
            'volume_spike': latest['Volume_Ratio'] > 1.5,
            'confirmations': 0
        }
        
        # MACD signal
        if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            signals['macd_signal'] = 'bullish'
            signals['confirmations'] += 1
        elif latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
            signals['macd_signal'] = 'bearish'
            signals['confirmations'] += 1
        else:
            signals['macd_signal'] = 'neutral'
        
        # RSI confirmation
        if latest['RSI'] < 30 or latest['RSI'] > 70:
            signals['confirmations'] += 1
        
        # Volume confirmation
        if signals['volume_spike']:
            signals['confirmations'] += 1
        
        return signals
    
    def calculate_price_levels(self, data: pd.DataFrame, signal: Dict) -> Dict:
        """Calculate entry, exit, and stop levels"""
        latest = data.iloc[-1]
        atr = (data['High'] - data['Low']).rolling(14).mean().iloc[-1]
        
        current_price = latest['Close']
        
        # Adjust levels based on signal type and risk
        risk_multiplier = 1.0
        if signal['risk_level'] == 'high':
            risk_multiplier = 1.5
        elif signal['risk_level'] == 'low':
            risk_multiplier = 0.8
        
        if signal['signal'] == 'BUY_CALL':
            entry_zone = [current_price - atr * 0.3, current_price + atr * 0.3]
            exit_target = current_price + atr * 2.0 * risk_multiplier
            stop_loss = current_price - atr * 1.0 * risk_multiplier
        elif signal['signal'] == 'BUY_PUT':
            entry_zone = [current_price - atr * 0.3, current_price + atr * 0.3]
            exit_target = current_price - atr * 2.0 * risk_multiplier
            stop_loss = current_price + atr * 1.0 * risk_multiplier
        else:  # HOLD
            entry_zone = [current_price, current_price]
            exit_target = current_price
            stop_loss = current_price
        
        # Calculate risk/reward
        if signal['signal'] != 'HOLD':
            risk = abs(current_price - stop_loss)
            reward = abs(exit_target - current_price)
            risk_reward = reward / risk if risk > 0 else 0
        else:
            risk_reward = 0
        
        return {
            'current_price': round(current_price, 2),
            'entry_zone': [round(entry_zone[0], 2), round(entry_zone[1], 2)],
            'exit_target': round(exit_target, 2),
            'stop_loss': round(stop_loss, 2),
            'risk_reward_ratio': round(risk_reward, 2),
            'timeframe': '1-5 days'
        }
    
    def generate_fallback_signal(self, symbol: str) -> Dict:
        """Generate a fallback signal when data is unavailable"""
        return {
            'symbol': symbol,
            'timestamp': pd.Timestamp.now().isoformat(),
            'signal': 'HOLD',
            'confidence': 0.0,
            'strength': 'weak',
            'risk_level': 'unknown',
            'consensus': 'no_data',
            'current_price': 0,
            'entry_zone': [0, 0],
            'exit_target': 0,
            'stop_loss': 0,
            'risk_reward_ratio': 0,
            'timeframe': 'N/A',
            'reasoning': 'Unable to fetch market data',
            'error': True
        }

def demonstrate_integrated_system():
    """Demonstrate the integrated signal generation"""
    print("="*70)
    print("🎯 GoldenSignalsAI - Integrated ML Signal Generation")
    print("="*70)
    
    generator = IntegratedSignalGenerator()
    symbols = ['AAPL', 'TSLA', 'SPY']
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Generating signal for {symbol}...")
        print("="*50)
        
        signal = generator.generate_integrated_signal(symbol)
        
        # Display signal
        emoji = "🟢" if signal['signal'] == "BUY_CALL" else "🔴" if signal['signal'] == "BUY_PUT" else "⚪"
        
        print(f"\n{emoji} Signal: {signal['signal']}")
        print(f"🎯 Confidence: {signal['confidence']:.1%}")
        print(f"💪 Strength: {signal['strength']}")
        print(f"⚠️  Risk Level: {signal['risk_level']}")
        print(f"🤝 Consensus: {signal['consensus']}")
        
        if signal['signal'] != 'HOLD' and not signal.get('error'):
            print(f"\n📊 Trading Levels:")
            print(f"   Current: ${signal['current_price']}")
            print(f"   Entry: ${signal['entry_zone'][0]} - ${signal['entry_zone'][1]}")
            print(f"   Target: ${signal['exit_target']}")
            print(f"   Stop: ${signal['stop_loss']}")
            print(f"   R:R Ratio: {signal['risk_reward_ratio']}:1")
        
        print(f"\n💡 {signal['reasoning']}")
    
    print("\n" + "="*70)
    print("✅ Integrated signal generation complete!")
    print("="*70)

if __name__ == "__main__":
    demonstrate_integrated_system() 